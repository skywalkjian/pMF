"""
One-step MeanFlow training centered on this file.

Commit 1 keeps the original algorithmic scope:
- MNIST
- U-Net backbone
- MeanFlow objective
- Adam optimizer
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import partial
from inspect import isfunction
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import einsum, nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10, MNIST
from torchvision.utils import make_grid, save_image
from tqdm import tqdm


def exists(x: Any) -> bool:
    return x is not None


def default(val: Any, d: Any) -> Any:
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    return value


def channel_last_vector_to_spatial(x: torch.Tensor) -> torch.Tensor:
    return x[:, :, None, None]


def conv_qkv_to_heads(x: torch.Tensor, heads: int) -> torch.Tensor:
    batch, channels, height, width = x.shape
    dim_head = channels // heads
    return x.view(batch, heads, dim_head, height * width)


def heads_to_spatial(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    batch, heads, tokens, dim_head = x.shape
    return x.permute(0, 1, 3, 2).contiguous().view(batch, heads * dim_head, height, width)


def heads_channels_to_spatial(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    batch, heads, channels, tokens = x.shape
    return x.contiguous().view(batch, heads * channels, height, width)


@dataclass
class TrainConfig:
    workdir: str = "./runs/mnist_meanflow"
    data_root: str = "./data"
    resume: str = ""
    dataset: str = "mnist"
    backbone: str = "unet"
    variant: str = "meanflow"
    optimizer: str = "adam"
    seed: int = 1024
    max_steps: int = 100000
    batch_size: int = 512
    eval_every: int = 5000
    sample_every: int = 5000
    save_every: int = 5000
    num_workers: int = 2
    sample_batch_size: int = 16
    lr: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.99
    eps: float = 1e-8
    weight_decay: float = 0.05
    grad_clip: float = 1.0
    model_dim: int = 32
    dim_mults: tuple[int, ...] = (1, 2, 4)
    convnext_mult: int = 2
    patch_size: int = 4
    hidden_size: int = 256
    depth: int = 8
    num_heads: int = 4
    mlp_ratio: float = 4.0
    t_min: float = 0.02
    p_mean: float = -0.4
    p_std: float = 1.0
    norm_p: float = 1.0
    norm_eps: float = 1.0
    noise_dist: str = "logit_normal"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    channels: int
    image_size: int
    num_classes: int
    mean: tuple[float, ...]
    std: tuple[float, ...]


@dataclass
class RunState:
    step: int = 0
    best_val_loss: float = float("inf")
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim: int) -> nn.Module:
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def Downsample(dim: int) -> nn.Module:
    return nn.Conv2d(dim, dim, 4, 2, 1)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        scale = math.log(10000) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -scale)
        embeddings = time[:, None] * freqs[None, :]
        return torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)


class ConvNextBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, *, time_emb_dim: int | None = None, mult: int = 2, norm: bool = True):
        super().__init__()
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim)) if exists(time_emb_dim) else None
        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor | None = None) -> torch.Tensor:
        h = self.ds_conv(x)
        if exists(self.mlp) and exists(time_emb):
            condition = self.mlp(time_emb)
            h = h + channel_last_vector_to_spatial(condition)
        h = self.net(h)
        return h + self.res_conv(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        q, k, v = map(
            lambda t: conv_qkv_to_heads(t, self.heads),
            self.to_qkv(x).chunk(3, dim=1),
        )
        q = q * self.scale
        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = heads_to_spatial(out, height, width)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4, dim_head: int = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, height, width = x.shape
        q, k, v = map(
            lambda t: conv_qkv_to_heads(t, self.heads),
            self.to_qkv(x).chunk(3, dim=1),
        )
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = heads_channels_to_spatial(out, height, width)
        return self.to_out(out)


class PreNorm(nn.Module):
    def __init__(self, dim: int, fn: nn.Module):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(self.norm(x))


class Unet(nn.Module):
    def __init__(
        self,
        dim: int,
        init_dim: int | None = None,
        out_dim: int | None = None,
        dim_mults: tuple[int, ...] = (1, 2, 4, 8),
        channels: int = 1,
        with_time_emb: bool = True,
        convnext_mult: int = 2,
    ):
        super().__init__()
        init_dim = default(init_dim, dim // 3 * 2)
        dims = [init_dim, *map(lambda mult: dim * mult, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block = partial(ConvNextBlock, mult=convnext_mult)

        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

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

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        num_resolutions = len(in_out)

        for index, (dim_in, dim_out) in enumerate(in_out):
            is_last = index >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        block(dim_in, dim_out, time_emb_dim=time_dim),
                        block(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block(mid_dim, mid_dim, time_emb_dim=time_dim)

        for index, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = index >= (num_resolutions - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        block(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(block(dim, dim), nn.Conv2d(dim, out_dim, 1))

    def forward(self, x: torch.Tensor, time: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        x = self.init_conv(x)
        time_emb = self.time_mlp(time) if exists(self.time_mlp) else None
        if exists(h) and exists(self.time_mlp_h):
            time_emb = time_emb + self.time_mlp_h(h)

        residuals = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, time_emb)
            x = block2(x, time_emb)
            x = attn(x)
            residuals.append(x)
            x = downsample(x)

        x = self.mid_block1(x, time_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, time_emb)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, residuals.pop()), dim=1)
            x = block1(x, time_emb)
            x = block2(x, time_emb)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(half, 1))
    args = timesteps.float()[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ScalarConditionEmbed(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(values, self.hidden_size))


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, channels: int, hidden_size: int):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError(f"image_size={image_size} must be divisible by patch_size={patch_size}")
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        self.hidden_size = hidden_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class TransformerAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by num_heads={num_heads}")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(hidden_size, hidden_size * 3)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = x.shape
        qkv = self.qkv(x).view(batch, tokens, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn = torch.matmul(q * self.scale, k.transpose(-1, -2))
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, tokens, self.hidden_size)
        return self.proj(out)


class TransformerMlp(nn.Module):
    def __init__(self, hidden_size: int, mlp_ratio: float):
        super().__init__()
        inner_dim = int(hidden_size * mlp_ratio)
        self.net = nn.Sequential(
            nn.Linear(hidden_size, inner_dim),
            nn.GELU(),
            nn.Linear(inner_dim, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = TransformerAttention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = TransformerMlp(hidden_size, mlp_ratio)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MeanFlowTransformer(nn.Module):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        channels: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels
        self.hidden_size = hidden_size
        self.patch_embed = PatchEmbed(image_size, patch_size, channels, hidden_size)
        self.time_embed = ScalarConditionEmbed(hidden_size)
        self.delta_embed = ScalarConditionEmbed(hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, hidden_size))
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden_size, num_heads, mlp_ratio) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Linear(hidden_size, patch_size * patch_size * channels)
        nn.init.normal_(self.pos_embed, std=0.02)

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        batch, tokens, _ = patches.shape
        grid_size = int(math.sqrt(tokens))
        if grid_size * grid_size != tokens:
            raise ValueError(f"Token count {tokens} is not a square grid.")
        patches = patches.view(
            batch,
            grid_size,
            grid_size,
            self.patch_size,
            self.patch_size,
            self.channels,
        )
        patches = patches.permute(0, 5, 1, 3, 2, 4).contiguous()
        return patches.view(batch, self.channels, self.image_size, self.image_size)

    def forward(self, x: torch.Tensor, time: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        tokens = self.patch_embed(x)
        cond = self.time_embed(time)
        if exists(h):
            cond = cond + self.delta_embed(h)
        tokens = tokens + self.pos_embed + cond[:, None, :]
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        patches = self.head(tokens)
        return self.unpatchify(patches)


def build_meanflow_corruption(x0: torch.Tensor, eps: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    return (1.0 - tau) * x0 + tau * eps


def sample_time_from_noise(
    shape: tuple[int, ...],
    device: torch.device,
    noise_dist: str,
    p_mean: float,
    p_std: float,
    t_min: float,
) -> torch.Tensor:
    if noise_dist == "logit_normal":
        raw = torch.sigmoid(torch.randn(shape, device=device) * p_std + p_mean)
    elif noise_dist == "uniform":
        raw = torch.rand(shape, device=device)
    else:
        raise ValueError(f"Unknown noise distribution: {noise_dist}")
    return t_min + (1.0 - t_min) * raw


class MeanFlowLoss:
    def __init__(
        self,
        p_mean: float = -0.4,
        p_std: float = 1.0,
        noise_dist: str = "logit_normal",
        norm_p: float = 1.0,
        norm_eps: float = 1.0,
        t_min: float = 0.02,
    ):
        self.p_mean = p_mean
        self.p_std = p_std
        self.noise_dist = noise_dist
        self.norm_p = norm_p
        self.norm_eps = norm_eps
        self.t_min = t_min

    def noise_distribution(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        if self.noise_dist == "logit_normal":
            rnd_normal = torch.randn(shape, device=device)
            return torch.sigmoid(rnd_normal * self.p_std + self.p_mean)
        if self.noise_dist == "uniform":
            return torch.rand(shape, device=device)
        raise ValueError(f"Unknown noise distribution: {self.noise_dist}")

    def __call__(self, net: nn.Module, images: torch.Tensor) -> torch.Tensor:
        x0 = images
        batch = x0.shape[0]
        device = x0.device
        shape = (batch, 1, 1, 1)

        tau = self.noise_distribution(shape, device)
        tau = self.t_min + (1.0 - self.t_min) * tau
        r = torch.zeros(shape, device=device)
        eps = torch.randn_like(x0)

        def corruption_of_tau(tau_in: torch.Tensor) -> torch.Tensor:
            return build_meanflow_corruption(x0, eps, tau_in)

        x_t, v_g = torch.func.jvp(corruption_of_tau, (tau,), (torch.ones_like(tau),))

        def u_wrapper(x: torch.Tensor, tau_val: torch.Tensor, r_val: torch.Tensor) -> torch.Tensor:
            return net(x, tau_val.view(batch), h=(tau_val - r_val).view(batch))

        u, du_dt = torch.func.jvp(
            u_wrapper,
            (x_t, tau, r),
            (v_g, torch.ones_like(tau), torch.zeros_like(r)),
        )

        h = torch.clamp(tau - r, min=0.0, max=1.0)
        u_target = (v_g - h * du_dt).detach()
        err2 = (u - u_target).pow(2).mean(dim=[1, 2, 3])

        with torch.no_grad():
            adaptive_weight = 1.0 / (err2 + self.norm_eps).pow(self.norm_p)

        return (err2 * adaptive_weight).mean()


class PixelMeanFlowLoss:
    def __init__(
        self,
        p_mean: float = -0.4,
        p_std: float = 1.0,
        noise_dist: str = "logit_normal",
        t_min: float = 0.02,
    ):
        self.p_mean = p_mean
        self.p_std = p_std
        self.noise_dist = noise_dist
        self.t_min = t_min

    def sample_two_time(self, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        shape = (batch, 1, 1, 1)
        t = sample_time_from_noise(shape, device, self.noise_dist, self.p_mean, self.p_std, self.t_min)
        r = torch.rand(shape, device=device) * t
        return r, t

    def __call__(self, net: nn.Module, images: torch.Tensor) -> torch.Tensor:
        x0 = images
        batch = x0.shape[0]
        device = x0.device
        r, t = self.sample_two_time(batch, device)
        eps = torch.randn_like(x0)
        z_t = build_meanflow_corruption(x0, eps, t)
        v_target = eps - x0

        def u_fn(z: torch.Tensor, r_val: torch.Tensor, t_val: torch.Tensor) -> torch.Tensor:
            t_flat = t_val.view(batch)
            h_flat = (t_val - r_val).view(batch)
            x_pred = net(z, t_flat, h=h_flat)
            return (z - x_pred) / t_val.clamp(min=self.t_min)

        v_direction = u_fn(z_t, t, t).detach()
        u, du_dt = torch.func.jvp(
            u_fn,
            (z_t, r, t),
            (v_direction, torch.zeros_like(r), torch.ones_like(t)),
        )
        v_model = u + (t - r) * du_dt.detach()
        return F.mse_loss(v_model, v_target)


@torch.no_grad()
def generate_meanflow(model: nn.Module, noise_x: torch.Tensor) -> torch.Tensor:
    batch = noise_x.shape[0]
    tau = torch.ones(batch, device=noise_x.device, dtype=noise_x.dtype)
    h = torch.ones_like(tau)
    return noise_x - model(noise_x, tau, h)


@torch.no_grad()
def generate_pmf(model: nn.Module, noise_x: torch.Tensor) -> torch.Tensor:
    batch = noise_x.shape[0]
    tau = torch.ones(batch, device=noise_x.device, dtype=noise_x.dtype)
    h = torch.ones_like(tau)
    return model(noise_x, tau, h)


@torch.no_grad()
def generate_samples(model: nn.Module, noise_x: torch.Tensor, variant: str) -> torch.Tensor:
    if variant == "meanflow":
        return generate_meanflow(model, noise_x)
    if variant == "pmf":
        return generate_pmf(model, noise_x)
    raise ValueError(f"Unsupported variant: {variant}")


def get_dataset_spec(dataset_name: str) -> DatasetSpec:
    if dataset_name == "mnist":
        return DatasetSpec(name="mnist", channels=1, image_size=32, num_classes=10, mean=(0.5,), std=(0.5,))
    if dataset_name == "cifar10":
        return DatasetSpec(
            name="cifar10",
            channels=3,
            image_size=32,
            num_classes=10,
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        )
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_datasets(config: TrainConfig):
    spec = get_dataset_spec(config.dataset)
    if config.dataset == "mnist":
        train_transform = T.Compose([T.ToTensor(), T.Pad(2), T.Normalize(spec.mean, spec.std)])
        eval_transform = train_transform
        train_dataset = MNIST(config.data_root, train=True, transform=train_transform, download=True)
        eval_dataset = MNIST(config.data_root, train=False, transform=eval_transform, download=True)
        return train_dataset, eval_dataset, spec
    if config.dataset == "cifar10":
        train_transform = T.Compose([T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(spec.mean, spec.std)])
        eval_transform = T.Compose([T.ToTensor(), T.Normalize(spec.mean, spec.std)])
        train_dataset = CIFAR10(config.data_root, train=True, transform=train_transform, download=True)
        eval_dataset = CIFAR10(config.data_root, train=False, transform=eval_transform, download=True)
        return train_dataset, eval_dataset, spec
    raise ValueError(f"Unsupported dataset: {config.dataset}")


def build_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int, drop_last: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
        generator=generator,
    )


def build_model(config: TrainConfig, spec: DatasetSpec) -> nn.Module:
    if config.backbone == "unet":
        return Unet(
            dim=config.model_dim,
            channels=spec.channels,
            dim_mults=tuple(config.dim_mults),
            convnext_mult=config.convnext_mult,
        )
    if config.backbone == "transformer":
        return MeanFlowTransformer(
            image_size=spec.image_size,
            patch_size=config.patch_size,
            channels=spec.channels,
            hidden_size=config.hidden_size,
            depth=config.depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
        )
    raise ValueError(f"Unsupported backbone: {config.backbone}")


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), eps=config.eps)
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def build_loss(config: TrainConfig):
    if config.variant == "meanflow":
        return MeanFlowLoss(
            p_mean=config.p_mean,
            p_std=config.p_std,
            noise_dist=config.noise_dist,
            norm_p=config.norm_p,
            norm_eps=config.norm_eps,
            t_min=config.t_min,
        )
    if config.variant == "pmf":
        return PixelMeanFlowLoss(
            p_mean=config.p_mean,
            p_std=config.p_std,
            noise_dist=config.noise_dist,
            t_min=config.t_min,
        )
    raise ValueError(f"Unsupported variant: {config.variant}")


def get_run_dir(config: TrainConfig) -> Path:
    if config.resume:
        return Path(config.resume).resolve().parent.parent
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path(config.workdir).resolve() / timestamp


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_sample_grid(samples: torch.Tensor, spec: DatasetSpec, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mean = torch.tensor(spec.mean, device=samples.device).view(1, -1, 1, 1)
    std = torch.tensor(spec.std, device=samples.device).view(1, -1, 1, 1)
    images = (samples.clamp(-1, 1) * std + mean).clamp(0, 1)
    grid = make_grid(images, nrow=max(1, int(math.sqrt(images.shape[0]))))
    save_image(grid, path)


def evaluate_meanflow(model: nn.Module, loss_fn: MeanFlowLoss, loader: DataLoader, device: torch.device, num_batches: int = 5) -> float:
    model.eval()
    losses = []
    iterator = iter(loader)
    with torch.no_grad():
        for _ in range(min(num_batches, len(loader))):
            images, _ = next(iterator)
            losses.append(loss_fn(model, images.to(device)).item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def make_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    state: RunState,
) -> dict[str, Any]:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": {key: to_serializable(value) for key, value in asdict(config).items()},
        "state": {
            "step": state.step,
            "best_val_loss": state.best_val_loss,
            "train_loss": state.train_loss,
            "val_loss": state.val_loss,
        },
        "rng_state": {
            "torch": torch.random.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def restore_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> RunState:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    torch.random.set_rng_state(checkpoint["rng_state"]["torch"])
    np.random.set_state(checkpoint["rng_state"]["numpy"])
    random.setstate(checkpoint["rng_state"]["python"])
    if torch.cuda.is_available() and checkpoint["rng_state"]["cuda"] is not None:
        torch.cuda.set_rng_state_all(checkpoint["rng_state"]["cuda"])
    raw_state = checkpoint["state"]
    return RunState(
        step=raw_state["step"],
        best_val_loss=raw_state["best_val_loss"],
        train_loss=list(raw_state["train_loss"]),
        val_loss=list(raw_state["val_loss"]),
    )


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train one-step MeanFlow baselines from meanflow.py.")
    parser.add_argument("--workdir", type=str, default="./runs/mnist_meanflow")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10"], default="mnist")
    parser.add_argument("--backbone", type=str, choices=["unet", "transformer"], default="unet")
    parser.add_argument("--variant", type=str, choices=["meanflow", "pmf"], default="meanflow")
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw"], default="adam")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--sample-every", type=int, default=5000)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--sample-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--model-dim", type=int, default=32)
    parser.add_argument("--dim-mults", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--convnext-mult", type=int, default=2)
    parser.add_argument("--t-min", type=float, default=0.02)
    parser.add_argument("--p-mean", type=float, default=-0.4)
    parser.add_argument("--p-std", type=float, default=1.0)
    parser.add_argument("--norm-p", type=float, default=1.0)
    parser.add_argument("--norm-eps", type=float, default=1.0)
    parser.add_argument("--noise-dist", type=str, choices=["logit_normal", "uniform"], default="logit_normal")
    args = parser.parse_args()
    return TrainConfig(
        workdir=args.workdir,
        data_root=args.data_root,
        resume=args.resume,
        dataset=args.dataset,
        backbone=args.backbone,
        variant=args.variant,
        optimizer=args.optimizer,
        seed=args.seed,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        sample_every=args.sample_every,
        save_every=args.save_every,
        num_workers=args.num_workers,
        sample_batch_size=args.sample_batch_size,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        eps=args.eps,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        model_dim=args.model_dim,
        dim_mults=tuple(args.dim_mults),
        convnext_mult=args.convnext_mult,
        t_min=args.t_min,
        p_mean=args.p_mean,
        p_std=args.p_std,
        norm_p=args.norm_p,
        norm_eps=args.norm_eps,
        noise_dist=args.noise_dist,
    )


def train(config: TrainConfig) -> Path:
    set_seed(config.seed)
    device = torch.device(config.device)
    run_dir = get_run_dir(config)
    sample_dir = run_dir / "samples"
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, eval_dataset, spec = build_datasets(config)
    train_loader = build_dataloader(train_dataset, config.batch_size, True, config.num_workers, config.seed, True)
    eval_loader = build_dataloader(eval_dataset, config.batch_size, False, config.num_workers, config.seed, False)
    train_iter = cycle(train_loader)

    model = build_model(config, spec).to(device)
    optimizer = build_optimizer(model, config)
    loss_fn = build_loss(config)
    state = RunState()

    if config.resume:
        state = restore_checkpoint(config.resume, model, optimizer, device)

    save_json(
        run_dir / "config.json",
        {
            "config": {key: to_serializable(value) for key, value in asdict(config).items()},
            "dataset": asdict(spec),
        },
    )

    sample_generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    sample_generator.manual_seed(config.seed)
    fixed_noise = torch.randn(
        config.sample_batch_size,
        spec.channels,
        spec.image_size,
        spec.image_size,
        generator=sample_generator,
        device=device,
    )

    progress = tqdm(
        range(state.step, config.max_steps),
        desc=f"Training {config.variant}-{config.dataset}-{config.backbone}",
    )

    for step in progress:
        images, _ = next(train_iter)
        images = images.to(device, non_blocking=torch.cuda.is_available())

        loss = loss_fn(model, images)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        state.step = step + 1
        state.train_loss.append(float(loss.item()))
        avg_loss = float(np.mean(state.train_loss[-min(100, len(state.train_loss)) :]))
        progress.set_postfix(loss=f"{loss.item():.2e}", avg=f"{avg_loss:.2e}", best=f"{state.best_val_loss:.2e}" if np.isfinite(state.best_val_loss) else "N/A")

        should_eval = state.step % config.eval_every == 0 or state.step == config.max_steps
        should_sample = state.step % config.sample_every == 0 or state.step == config.max_steps
        should_save = state.step % config.save_every == 0 or state.step == config.max_steps

        if should_eval:
            val_loss = evaluate_meanflow(model, loss_fn, eval_loader, device)
            state.val_loss.append(val_loss)
            state.best_val_loss = min(state.best_val_loss, val_loss)
            progress.set_postfix(loss=f"{loss.item():.2e}", avg=f"{avg_loss:.2e}", val=f"{val_loss:.2e}", best=f"{state.best_val_loss:.2e}")

        if should_sample:
            with torch.no_grad():
                samples = generate_samples(model, fixed_noise, config.variant)
            save_sample_grid(samples, spec, sample_dir / f"step_{state.step:07d}.png")

        if should_save:
            checkpoint = make_checkpoint(model, optimizer, config, state)
            torch.save(checkpoint, checkpoint_dir / "last.pt")
            torch.save(checkpoint, checkpoint_dir / f"step_{state.step:07d}.pt")
            if state.val_loss and state.val_loss[-1] <= state.best_val_loss:
                torch.save(checkpoint, checkpoint_dir / "best.pt")
            save_json(
                run_dir / "metrics.json",
                {
                    "step": state.step,
                    "best_val_loss": state.best_val_loss,
                    "train_loss": state.train_loss,
                    "val_loss": state.val_loss,
                },
            )

    return run_dir


def main() -> None:
    config = parse_args()
    run_dir = train(config)
    print(f"Run directory: {run_dir}")


if __name__ == "__main__":
    main()
