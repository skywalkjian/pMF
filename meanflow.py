"""
Awesome 1-step MNIST Generation with Mean Flows
1-step Mean Flows training on MNIST dataset.
"""

# Install required library if not present
try:
    import einops
except ImportError:
    !pip install einops
    import einops

import os
import math
import random
import torch
from torch import nn
from torch import einsum
import torch.nn.functional as F
import numpy as np
from torchvision.datasets import MNIST
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime
from inspect import isfunction
from functools import partial
from einops import rearrange
import json

plt.rcParams['figure.figsize'] = (5, 5)
plt.rcParams['image.cmap'] = 'gray'

# Updated paths for Colab environment
LOGS_DIR = './logs'
os.makedirs(LOGS_DIR, exist_ok=True)
DATA_DIR = './data'
os.makedirs(DATA_DIR, exist_ok=True)


# ============================================================================
# Utility functions
# ============================================================================

def grid(array, ncols=8):
    array = np.pad(array, [(0, 0), (1, 1), (1, 1), (0, 0)], 'constant')
    nindex, height, width, intensity = array.shape
    ncols = min(nindex, ncols)
    nrows = (nindex + ncols - 1) // ncols
    r = nrows * ncols - nindex
    arr = np.concatenate([array] + [np.zeros([1, height, width, intensity])] * r)
    result = (
        arr.reshape(nrows, ncols, height, width, intensity)
        .swapaxes(1, 2)
        .reshape(height * nrows, width * ncols, intensity)
    )
    return np.pad(result, [(1, 1), (1, 1), (0, 0)], 'constant')


class NextDataLoader(torch.utils.data.DataLoader):
    def __next__(self):
        try:
            return next(self.iterator)
        except Exception:
            self.iterator = self.__iter__()
            return next(self.iterator)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# ============================================================================
# Utility functions for neural networks
# ============================================================================

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / max(half_dim - 1, 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x


class ConvNextBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )
        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)
        if exists(self.mlp) and exists(time_emb):
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")
        h = self.net(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale
        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            nn.GroupNorm(1, dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# ============================================================================
# Neural Network (Unet)
# ============================================================================

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        with_time_emb=True,
        convnext_mult=2,
    ):
        super().__init__()
        self.channels = channels
        init_dim = default(init_dim, dim // 3 * 2)
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ConvNextBlock, mult=convnext_mult)

        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
            self.time_mlp_h = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None
            self.time_mlp_h = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Downsample(dim_out) if not is_last else nn.Identity(),
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Upsample(dim_in) if not is_last else nn.Identity(),
            ]))

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(
            block_klass(dim, dim),
            nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time, h=None):
        x = self.init_conv(x)
        t = self.time_mlp(time) if exists(self.time_mlp) else None
        if h is not None and exists(self.time_mlp_h):
            t = t + self.time_mlp_h(h)

        h_list = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h_list.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h_list.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)


# ============================================================================
# Core MeanFlow Corruption Logic
# ============================================================================

def _build_meanflow_corruption(x0, eps, tau):
    """
    Standard MeanFlow interpolation path:
        x_tau = (1 - tau) * x0 + tau * eps
    tau shape: [B,1,1,1]
    """
    return (1.0 - tau) * x0 + tau * eps


# ============================================================================
# MeanFlow Loss Function
# ============================================================================

class MeanFlowLoss:
    def __init__(
        self,
        P_mean=-0.4,
        P_std=1.0,
        noise_dist='logit_normal',
        data_proportion=1.0,
        norm_p=1.0,
        norm_eps=1.0,
        t_min=0.02,
        use_fixed_t_r=False,
    ):
        self.P_mean = P_mean
        self.P_std = P_std
        self.data_proportion = data_proportion
        self.norm_p = norm_p
        self.norm_eps = norm_eps
        self.noise_dist = noise_dist
        self.t_min = t_min
        self.use_fixed_t_r = use_fixed_t_r

    def _logit_normal_dist(self, shape, device):
        rnd_normal = torch.randn(shape, device=device)
        return torch.sigmoid(rnd_normal * self.P_std + self.P_mean)

    def _uniform_dist(self, shape, device):
        return torch.rand(shape, device=device)

    def noise_distribution(self, shape, device):
        if self.noise_dist == 'logit_normal':
            return self._logit_normal_dist(shape, device)
        elif self.noise_dist == 'uniform':
            return self._uniform_dist(shape, device)
        else:
            raise ValueError(f"Unknown noise distribution: {self.noise_dist}")

    def __call__(self, net, images):
        x0 = images
        device = x0.device
        B = x0.shape[0]
        shape = (B, 1, 1, 1)

        # sample tau in [t_min, 1]
        tau = self.noise_distribution(shape, device)
        tau = self.t_min + (1.0 - self.t_min) * tau

        # keep r = 0 for simple one-step learning
        r = torch.zeros(shape, device=device)

        eps = torch.randn_like(x0)

        # path x_tau and its derivative v_g = d/dtau G_tau(...)
        def corruption_of_tau(tau_in):
            return _build_meanflow_corruption(x0, eps, tau_in)

        x_t, v_g = torch.func.jvp(
            corruption_of_tau,
            (tau,),
            (torch.ones_like(tau),)
        )

        def u_wrapper(x, tau_val, r_val):
            t_flat = tau_val.view(B)
            h_flat = (tau_val - r_val).view(B)
            return net(x, t_flat, h=h_flat)

        primals = (x_t, tau, r)
        tangents = (v_g, torch.ones_like(tau), torch.zeros_like(r))
        u, du_dtau = torch.func.jvp(u_wrapper, primals, tangents)

        h = torch.clamp(tau - r, min=0.0, max=1.0)
        u_tgt = (v_g - h * du_dtau).detach()

        err2 = (u - u_tgt).pow(2).mean(dim=[1, 2, 3])

        with torch.no_grad():
            adaptive_weight = 1.0 / (err2 + self.norm_eps).pow(self.norm_p)

        loss = (err2 * adaptive_weight).mean()
        return loss


# ============================================================================
# MeanFlow generation
# ============================================================================

def generate(mf, noise_x):
    """
    One-step MeanFlow generation:
        x0 = x1 - u(x1, tau=1, h=1)
    """
    B = noise_x.shape[0]
    device = noise_x.device
    dtype = noise_x.dtype

    tau_val = torch.ones(B, 1, 1, 1, device=device, dtype=dtype)
    h_val = torch.ones_like(tau_val)

    x1 = noise_x
    x0 = x1 - mf(x1, tau_val.view(B), h_val.view(B))
    return x0


# ============================================================================
# Training loop
# ============================================================================

def train(
    mf,
    max_iter,
    batch_size,
    mf_opt_args,
    num_workers,
    val_interval,
    checkpoint=None,
    t_min=0.02,
    p_mean=-0.4,
    p_std=1.0,
    norm_p=1.0,
    norm_eps=1.0,
    noise_dist='logit_normal',
    args_dict=None,
    seed=42,
):
    # ===== Seeds should be set in __main__ before model initialization =====
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    now = datetime.now()
    timestamp = now.strftime('%Y%m%d-%H%M%S')
    run_dir = f'./logs/mf-mnist-{timestamp}'
    logs_loss_dir = os.path.join(run_dir, 'logs-loss-curve')
    logs_sample_dir = os.path.join(run_dir, 'logs-sample')

    os.makedirs(logs_loss_dir, exist_ok=True)
    os.makedirs(logs_sample_dir, exist_ok=True)

    # Save configuration parameters to JSON file
    if args_dict is not None:
        config_file = os.path.join(run_dir, 'config.json')
        config_with_meta = dict(args_dict)
        config_with_meta['_reproducibility'] = {
            'seed': seed,
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set'),
            'torch_version': torch.__version__,
            'numpy_version': np.__version__,
        }
        with open(config_file, 'w') as f:
            json.dump(config_with_meta, f, indent=4)
        print(f"Configuration saved to: {config_file}")

    mean = 0.5
    std = 0.5
    dataset = MNIST(
        './data',
        transform=T.Compose([
            T.ToTensor(),
            T.Pad(2),
            T.Normalize((mean,), (std,))
        ]),
        download=True
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=True,
        generator=train_generator,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )

    eval_generator = torch.Generator()
    eval_generator.manual_seed(seed)

    eval_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        pin_memory=True,
        generator=eval_generator,
        persistent_workers=(num_workers > 0),
    )

    eval_noise = torch.randn(64, 1, 32, 32, generator=train_generator).to(device)

    mf_optimizer = torch.optim.Adam(mf.parameters(), **mf_opt_args)

    mf_loss = MeanFlowLoss(
        P_mean=p_mean,
        P_std=p_std,
        noise_dist=noise_dist,
        data_proportion=1.0,
        norm_p=norm_p,
        norm_eps=norm_eps,
        t_min=t_min,
        use_fixed_t_r=False
    )

    loss_history = []
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'avg_loss_window': 0.0,
        'bes_val': float('inf'),
    }
    avg_val_loss = float('inf')

    mf.train()
    pbar = tqdm(range(max_iter), desc="Training MF-MNIST")

    train_iter = iter(dataloader)

    for i in pbar:
        try:
            x0 = next(train_iter)[0].to(device, non_blocking=True)
        except StopIteration:
            train_iter = iter(dataloader)
            x0 = next(train_iter)[0].to(device, non_blocking=True)

        loss = mf_loss(mf, x0)
        loss_item = loss.item()
        loss_history.append(loss_item)
        metrics['train_loss'].append(loss_item)

        mf_optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(mf.parameters(), 1.0)
        mf_optimizer.step()

        # Update metrics for display
        window_size = min(100, len(metrics['train_loss']))
        metrics['avg_loss_window'] = np.mean(metrics['train_loss'][-window_size:])

        pbar.set_postfix({
            'loss': f"{loss_item:.2e}",
            'avg_loss': f"{metrics['avg_loss_window']:.2e}",
            'val_loss': f"{avg_val_loss:.2e}" if len(metrics['val_loss']) > 0 else "N/A",
            'bes_val': f"{metrics['bes_val']:.2e}" if metrics['bes_val'] != float('inf') else 'N/A'
        })

        # Validation and sample generation at regular intervals and at the final iteration
        if i % val_interval == 0 or i == max_iter - 1:
            mf.eval()
            with torch.no_grad():
                gen_x0 = generate(mf, eval_noise)

                # Calculate validation loss on a subset of data
                val_loss_sum = 0.0
                val_batches = 0
                eval_iter = iter(eval_dataloader)
                for _ in range(min(5, len(eval_dataloader))):
                    try:
                        val_x0 = next(eval_iter)[0].to(device)
                    except StopIteration:
                        break
                    val_loss = mf_loss(mf, val_x0)
                    val_loss_sum += val_loss.item()
                    val_batches += 1

                avg_val_loss = val_loss_sum / max(val_batches, 1)
                metrics['val_loss'].append(avg_val_loss)

                # Track best validation loss
                if avg_val_loss < metrics['bes_val']:
                    metrics['bes_val'] = avg_val_loss

            mf.train()

            plt.figure(figsize=(5, 5))
            gen_x0 = gen_x0.detach().permute(0, 2, 3, 1).clamp(-1, 1) * std + mean
            plt.title('MeanFlow Sample', fontsize=17)
            plt.imshow(grid(gen_x0.cpu()).squeeze())
            plt.savefig(
                os.path.join(logs_sample_dir, f'sample_iter_{i}.png'),
                dpi=100,
                bbox_inches='tight'
            )
            plt.show()
            plt.close()

            if i != 0:
                plt.figure(figsize=(10, 4))

                # Plot training loss
                plt.subplot(1, 2, 1)
                plt.plot(metrics['train_loss'], label='Training Loss', alpha=0.7)
                plt.title('Training Loss Curve')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.grid(True)
                plt.legend()

                # Plot validation loss
                plt.subplot(1, 2, 2)
                val_iters = [k for k, _ in enumerate(metrics['val_loss'])]
                val_iter_steps = [k * val_interval for k in val_iters]
                plt.plot(
                    val_iter_steps,
                    metrics['val_loss'],
                    marker='o',
                    label='Validation Loss',
                    color='orange'
                )
                plt.title('Validation Loss Curve')
                plt.xlabel('Iteration')
                plt.ylabel('Loss')
                plt.grid(True)
                plt.legend()

                plt.tight_layout()
                plt.savefig(
                    os.path.join(logs_loss_dir, f'loss_curve_iter_{i}.png'),
                    dpi=100,
                    bbox_inches='tight'
                )
                plt.show()
                plt.close()

            pbar.set_postfix({
                'loss': f"{loss_item:.2e}",
                'avg_loss': f"{metrics['avg_loss_window']:.2e}",
                'val_loss': f"{avg_val_loss:.2e}",
                'bes_val': f"{metrics['bes_val']:.2e}"
            })

    # Save model and metrics
    torch.save({
        'mf': mf.state_dict(),
        'metrics': metrics,
        'loss_history': loss_history,
    }, os.path.join(run_dir, 'mnist_mf.pt'))

    # Save metrics summary to text file
    with open(os.path.join(run_dir, 'metrics_summary.txt'), 'w') as f:
        f.write("Training Metrics Summary\n")
        f.write("========================\n\n")
        f.write(f"Total iterations: {max_iter}\n")
        f.write(f"Final training loss: {metrics['train_loss'][-1]:.2e}\n")
        f.write(f"Best validation loss: {metrics['bes_val']:.2e}\n")
        f.write(f"Number of validation checkpoints: {len(metrics['val_loss'])}\n")
        if len(metrics['val_loss']) > 0:
            f.write(f"Final validation loss: {metrics['val_loss'][-1]:.2e}\n")
            f.write(f"Min validation loss: {min(metrics['val_loss']):.2e}\n")
            f.write(f"Max validation loss: {max(metrics['val_loss']):.2e}\n")

    # Update configuration file with training results
    if args_dict is not None:
        config_file = os.path.join(run_dir, 'config.json')
        with open(config_file, 'r') as f:
            config = json.load(f)

        config['training_results'] = {
            'train_loss': [round(loss, 3) for loss in metrics['train_loss']],
            'valid_loss': [round(loss, 3) for loss in metrics['val_loss']] if len(metrics['val_loss']) > 0 else [],
            'bes_val': round(metrics['bes_val'], 3) if metrics['bes_val'] != float('inf') else None
        }

        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)

    print(f'\nTraining ends~ Have a good Day!')
    print(f'Results saved to: {run_dir}')
    print(f'Final training loss: {metrics["train_loss"][-1]:.2e}')
    print(f'Best validation loss: {metrics["bes_val"]:.2e}')
    return metrics


# ============================================================================
# Configuration Class (Replacing argparse)
# ============================================================================

class Config:
    def __init__(self):
        # ===== Training Parameters =====
        self.max_iter = 100000
        self.batch_size = 512
        self.val_interval = 5000
        self.num_workers = 2

        # ===== Optimizer Parameters =====
        self.lr = 1e-4
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.eps = 1e-08

        # ===== Model Architecture Parameters =====
        self.model_dim = 32
        self.dim_mults = [1, 2, 4]
        self.channels = 1
        self.convnext_mult = 2

        # ===== MeanFlow Loss Parameters =====
        self.t_min = 0.02
        self.p_mean = -0.4
        self.p_std = 1.0
        self.norm_p = 1.0
        self.norm_eps = 1.0
        self.noise_dist = 'logit_normal'

        # ===== Reproducibility Parameters =====
        self.seed = 1024

        # ===== Device and Other Parameters =====
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.checkpoint = None


# ============================================================================
# Start training!
# ============================================================================

if __name__ == '__main__':
    args = Config()
    device = args.device

    # Convert args to dictionary for saving configuration
    args_dict = vars(args)

    train_params = {
        'max_iter': args.max_iter,
        'batch_size': args.batch_size,
        'mf_opt_args': {
            'lr': args.lr,
            'betas': (args.beta1, args.beta2),
            'eps': args.eps
        },
        'num_workers': args.num_workers,
        'val_interval': args.val_interval
    }

    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Max iterations: {args.max_iter}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Model dim: {args.model_dim}")
    print(f"Dim mults: {args.dim_mults}")
    print(f"Channels: {args.channels}")
    print(f"t_min: {args.t_min}")
    print(f"p_mean: {args.p_mean}")
    print(f"p_std: {args.p_std}")
    print(f"norm_p: {args.norm_p}")
    print(f"norm_eps: {args.norm_eps}")
    print(f"noise_dist: {args.noise_dist}")
    print(f"Seed: {args.seed}")
    print("=" * 60 + "\n")

    # ===== Set all random seeds BEFORE model initialization =====
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Create UNet model AFTER setting seeds
    mf = Unet(
        dim=args.model_dim,
        channels=args.channels,
        dim_mults=tuple(args.dim_mults),
        convnext_mult=args.convnext_mult
    ).to(device)

    train(
        mf,
        **train_params,
        t_min=args.t_min,
        p_mean=args.p_mean,
        p_std=args.p_std,
        norm_p=args.norm_p,
        norm_eps=args.norm_eps,
        noise_dist=args.noise_dist,
        checkpoint=args.checkpoint,
        args_dict=args_dict,
        seed=args.seed
    )