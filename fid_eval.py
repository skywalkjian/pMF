"""
Multi-GPU online FID evaluation for pMF.

Computes Inception features on-the-fly during generation — no images are saved
to disk.  Supports ``torchrun`` for multi-GPU parallelism.

Usage (single GPU):
    python fid_eval.py --run-dir <run_dir> --checkpoint best

Usage (multi-GPU):
    torchrun --nproc_per_node=2 fid_eval.py --run-dir <run_dir> --checkpoint best
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F
from tqdm import tqdm

import utils.torch_dist_util as dist
from meanflow import (
    TrainConfig,
    build_model,
    build_optimizer,
    generate_samples,
    get_dataset_spec,
    restore_checkpoint,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-GPU online FID evaluation for pMF.")
    parser.add_argument("--run-dir", type=str, required=True, help="Experiment run directory containing config.json.")
    parser.add_argument("--ckpt-path", type=str, default="", help="Optional explicit checkpoint path.")
    parser.add_argument("--checkpoint", type=str, choices=["best", "last"], default="best")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--ref", type=str, default="cifar10-train",
                        help="FID reference: 'cifar10-train' (auto-computed) or path to .npz with mu/sigma.")
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--gen-bsz", type=int, default=64, help="Per-GPU generation batch size.")
    parser.add_argument("--fid-batch-size", type=int, default=128, help="Inception feature extraction batch size.")
    parser.add_argument("--sample-seed", type=int, default=1024)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--sample-omega", type=float, default=None)
    parser.add_argument("--sample-t-min", type=float, default=None)
    parser.add_argument("--sample-t-max", type=float, default=None)
    parser.add_argument("--datasets-download", action="store_true")
    parser.add_argument("--isc", action="store_true", help="Also compute Inception Score.")
    return parser.parse_args()


def load_run_config(run_dir: Path) -> TrainConfig:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    raw_config = payload.get("config", payload)

    default_config = TrainConfig()
    valid_fields = {field.name for field in fields(TrainConfig)}
    merged: dict[str, Any] = {name: getattr(default_config, name) for name in valid_fields}

    for key, value in raw_config.items():
        if key not in valid_fields:
            continue
        if key == "dim_mults":
            merged[key] = tuple(value)
        else:
            merged[key] = value

    return TrainConfig(**merged)


def resolve_checkpoint(run_dir: Path, ckpt_path: str, checkpoint_name: str) -> Path:
    if ckpt_path:
        path = Path(ckpt_path).expanduser().resolve()
    else:
        path = (run_dir / "checkpoints" / f"{checkpoint_name}.pt").resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


# ---------------------------------------------------------------------------
#  Inception feature extraction helpers
# ---------------------------------------------------------------------------

_INCEPTION_LAYERS = ["2048", "logits_unbiased"]


def build_inception(device: torch.device):
    from torch_fidelity.utils import create_feature_extractor
    fe = create_feature_extractor("inception-v3-compat", _INCEPTION_LAYERS, cuda=device.type == "cuda")
    fe.eval()
    return fe


def images_to_uint8(samples: torch.Tensor, mean: tuple[float, ...], std: tuple[float, ...]) -> torch.Tensor:
    """De-normalise model output → [0, 255] uint8 NCHW."""
    mean_t = torch.tensor(mean, device=samples.device, dtype=samples.dtype).view(1, -1, 1, 1)
    std_t = torch.tensor(std, device=samples.device, dtype=samples.dtype).view(1, -1, 1, 1)
    images = (samples.clamp(-1, 1) * std_t + mean_t).clamp(0, 1)
    return (images * 255).to(torch.uint8)


@torch.no_grad()
def extract_features(inception, images_uint8: torch.Tensor) -> dict[str, torch.Tensor]:
    """Run Inception on a uint8 NCHW batch, return feature dict keyed by layer name."""
    out = inception(images_uint8)
    if isinstance(out, dict):
        return {k: v.cpu() for k, v in out.items()}
    # torch-fidelity returns a tuple ordered by requested feature layers
    return {name: t.cpu() for name, t in zip(_INCEPTION_LAYERS, out)}


# ---------------------------------------------------------------------------
#  FID / IS math
# ---------------------------------------------------------------------------

def compute_fid(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))


def compute_is(logits: np.ndarray, splits: int = 10) -> tuple[float, float]:
    probs = scipy.special.softmax(logits, axis=1)
    scores = []
    n = len(probs)
    for i in range(splits):
        part = probs[i * n // splits : (i + 1) * n // splits]
        kl = part * (np.log(part + 1e-16) - np.log(np.mean(part, axis=0, keepdims=True) + 1e-16))
        scores.append(np.exp(np.mean(np.sum(kl, axis=1))))
    return float(np.mean(scores)), float(np.std(scores))


# ---------------------------------------------------------------------------
#  Reference statistics
# ---------------------------------------------------------------------------

def get_reference_stats(ref: str, data_root: str, fid_batch_size: int, download: bool) -> dict[str, np.ndarray]:
    """Return {mu, sigma} for the reference dataset.

    Supports:
      - A path to a .npz file containing 'mu' and 'sigma' arrays.
      - 'cifar10-train': computes stats from CIFAR-10 train set via torch-fidelity.
    """
    if ref.endswith(".npz"):
        data = np.load(ref)
        return {"mu": data["mu"], "sigma": data["sigma"]}

    from torch_fidelity.utils import create_feature_extractor, extract_featuresdict_from_input_id_cached
    from torch_fidelity.metric_fid import fid_featuresdict_to_statistics_cached

    fe = create_feature_extractor("inception-v3-compat", ["2048"], cuda=torch.cuda.is_available())
    featuresdict = extract_featuresdict_from_input_id_cached(
        2, fe,
        input2=ref,
        input2_cache_name=ref,
        feature_extractor="inception-v3-compat",
        feature_layer_fid="2048",
        datasets_root=data_root,
        datasets_download=download,
        cache=True,
        save_cpu_ram=False,
        verbose=True,
        cuda=torch.cuda.is_available(),
        batch_size=fid_batch_size,
        rng_seed=2020,
        samples_shuffle=True,
    )
    stats = fid_featuresdict_to_statistics_cached(
        featuresdict, ref, fe, "2048",
        feature_extractor="inception-v3-compat",
        feature_layer_fid="2048",
        datasets_root=data_root,
        datasets_download=download,
        cache=True,
        save_cpu_ram=False,
        verbose=True,
        cuda=torch.cuda.is_available(),
        batch_size=fid_batch_size,
        rng_seed=2020,
        samples_shuffle=True,
    )
    return {"mu": stats["mu"], "sigma": stats["sigma"]}


# ---------------------------------------------------------------------------
#  Main evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_online(
    model: torch.nn.Module,
    config: TrainConfig,
    device: torch.device,
    num_samples: int,
    gen_bsz: int,
    sample_seed: int,
    ref_stats: dict[str, np.ndarray],
    compute_isc: bool = False,
) -> dict[str, Any]:
    """Generate samples across GPUs, extract Inception features on-the-fly, compute FID."""
    spec = get_dataset_spec(config.dataset)
    model.eval()

    world_size = dist.process_count()
    rank = dist.process_index()

    samples_per_rank = math.ceil(num_samples / world_size)
    start_idx = rank * samples_per_rank
    end_idx = min(start_idx + samples_per_rank, num_samples)
    local_n = end_idx - start_idx

    inception = build_inception(device)

    all_feats_2048: list[torch.Tensor] = []
    all_logits: list[torch.Tensor] = []

    generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    generator.manual_seed(sample_seed + rank)

    total_batches = math.ceil(local_n / gen_bsz)
    progress = tqdm(total=local_n, desc=f"[rank {rank}] Generating", disable=rank != 0)

    generated = 0
    for batch_idx in range(total_batches):
        current_bsz = min(gen_bsz, local_n - generated)
        if current_bsz <= 0:
            break

        noise = torch.randn(
            current_bsz, spec.channels, spec.image_size, spec.image_size,
            generator=generator, device=device,
        )
        global_start = start_idx + generated
        labels = torch.arange(global_start, global_start + current_bsz, device=device, dtype=torch.long) % spec.num_classes

        samples = generate_samples(model, noise, labels, config)

        images_u8 = images_to_uint8(samples, spec.mean, spec.std)

        feats = extract_features(inception, images_u8)
        all_feats_2048.append(feats["2048"])
        if compute_isc and "logits_unbiased" in feats:
            all_logits.append(feats["logits_unbiased"])

        generated += current_bsz
        progress.update(current_bsz)

    progress.close()

    local_feats = torch.cat(all_feats_2048, dim=0)  # (local_n, 2048)

    local_feats_gpu = local_feats.to(device)
    padded_size = samples_per_rank
    if local_feats_gpu.shape[0] < padded_size:
        pad = torch.zeros(padded_size - local_feats_gpu.shape[0], local_feats_gpu.shape[1],
                          device=device, dtype=local_feats_gpu.dtype)
        local_feats_gpu = torch.cat([local_feats_gpu, pad], dim=0)

    gathered = dist.all_gather(local_feats_gpu)  # (world_size, padded_size, 2048)
    all_features_np = gathered.cpu().numpy().reshape(-1, 2048)[:num_samples]

    local_logits_np = None
    if compute_isc and all_logits:
        local_logits = torch.cat(all_logits, dim=0).to(device)
        if local_logits.shape[0] < padded_size:
            pad = torch.zeros(padded_size - local_logits.shape[0], local_logits.shape[1],
                              device=device, dtype=local_logits.dtype)
            local_logits = torch.cat([local_logits, pad], dim=0)
        gathered_logits = dist.all_gather(local_logits)
        local_logits_np = gathered_logits.cpu().numpy().reshape(-1, gathered_logits.shape[-1])[:num_samples]

    metrics: dict[str, Any] = {}
    if rank == 0:
        mu1 = np.mean(all_features_np, axis=0)
        sigma1 = np.cov(all_features_np, rowvar=False)
        mu2, sigma2 = ref_stats["mu"], ref_stats["sigma"]
        fid_value = compute_fid(mu1, sigma1, mu2, sigma2)
        metrics["frechet_inception_distance"] = fid_value
        dist.print0(f"FID: {fid_value:.4f}")

        if compute_isc and local_logits_np is not None:
            is_mean, is_std = compute_is(local_logits_np)
            metrics["inception_score_mean"] = is_mean
            metrics["inception_score_std"] = is_std
            dist.print0(f"IS: {is_mean:.4f} ± {is_std:.4f}")

    dist.barrier()
    return metrics


def main() -> None:
    args = parse_args()

    dist.initialize()
    rank = dist.process_index()

    run_dir = Path(args.run_dir).expanduser().resolve()
    config = load_run_config(run_dir)
    config.data_root = args.data_root

    if args.sample_steps is not None:
        config.sample_steps = args.sample_steps
    if args.sample_omega is not None:
        config.sample_omega = args.sample_omega
    if args.sample_t_min is not None:
        config.sample_t_min = args.sample_t_min
    if args.sample_t_max is not None:
        config.sample_t_max = args.sample_t_max

    if args.num_samples % 10 != 0:
        raise ValueError("--num-samples should be divisible by 10 for balanced class labels.")

    device = dist.local_device()
    ckpt_path = resolve_checkpoint(run_dir, args.ckpt_path, args.checkpoint)
    set_seed(args.sample_seed)

    spec = get_dataset_spec(config.dataset)
    model = build_model(config, spec).to(device)
    optimizer = build_optimizer(model, config)
    state = restore_checkpoint(str(ckpt_path), model, optimizer, device)

    dist.print0(f"Run dir: {run_dir}")
    dist.print0(f"Checkpoint: {ckpt_path} (step {state.step})")
    dist.print0(f"World size: {dist.process_count()}")
    dist.print0(f"Sampling config: steps={config.sample_steps}, omega={config.sample_omega}, "
                f"t_min={config.sample_t_min}, t_max={config.sample_t_max}")
    dist.print0(f"Generating {args.num_samples} samples (batch size {args.gen_bsz}/GPU)...")
    dist.print0("Mode: online FID (no images saved to disk)")

    ref_stats = None
    if rank == 0:
        dist.print0("Loading reference statistics...")
        ref_stats = get_reference_stats(args.ref, args.data_root, args.fid_batch_size, args.datasets_download)
        dist.print0("Reference statistics loaded.")

    dist.barrier()

    if rank != 0:
        ref_stats = {"mu": np.zeros(2048, dtype=np.float64), "sigma": np.zeros((2048, 2048), dtype=np.float64)}

    metrics = evaluate_online(
        model=model,
        config=config,
        device=device,
        num_samples=args.num_samples,
        gen_bsz=args.gen_bsz,
        sample_seed=args.sample_seed,
        ref_stats=ref_stats,
        compute_isc=args.isc,
    )

    if rank == 0:
        fid_dir = run_dir / "fid"
        fid_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = fid_dir / f"metrics_{ckpt_path.stem}_{args.num_samples}.json"

        payload = {
            "run_dir": str(run_dir),
            "checkpoint": str(ckpt_path),
            "step": state.step,
            "dataset": config.dataset,
            "world_size": dist.process_count(),
            "num_samples": args.num_samples,
            "gen_bsz": args.gen_bsz,
            "sample_seed": args.sample_seed,
            "reference": args.ref,
            "sample_steps": config.sample_steps,
            "sample_omega": config.sample_omega,
            "sample_t_min": config.sample_t_min,
            "sample_t_max": config.sample_t_max,
            "metrics": metrics,
        }
        metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        dist.print0(f"Saved report to: {metrics_path}")

    dist.barrier()


if __name__ == "__main__":
    main()
