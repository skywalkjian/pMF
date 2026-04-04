from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
from torchvision.utils import save_image
from tqdm import tqdm

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
    parser = argparse.ArgumentParser(description="Compute FID for meanflow.py experiments.")
    parser.add_argument("--run-dir", type=str, required=True, help="Experiment run directory that contains config.json.")
    parser.add_argument("--ckpt-path", type=str, default="", help="Optional explicit checkpoint path.")
    parser.add_argument("--checkpoint", type=str, choices=["best", "last"], default="best", help="Checkpoint name under run_dir/checkpoints when --ckpt-path is not given.")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset root used by torch-fidelity built-in CIFAR10 reference.")
    parser.add_argument("--ref", type=str, default="cifar10-train", help="FID reference dataset or stats path.")
    parser.add_argument("--num-samples", type=int, default=50000, help="Number of generated samples for FID.")
    parser.add_argument("--gen-bsz", type=int, default=64, help="Batch size used during sample generation.")
    parser.add_argument("--fid-batch-size", type=int, default=128, help="Batch size used by torch-fidelity feature extraction.")
    parser.add_argument("--sample-seed", type=int, default=1024, help="Seed used for sampling noise.")
    parser.add_argument("--sample-steps", type=int, default=None, help="Override sampling steps from run config.")
    parser.add_argument("--sample-omega", type=float, default=None, help="Override sampling omega from run config.")
    parser.add_argument("--sample-t-min", type=float, default=None, help="Override sampling interval min from run config.")
    parser.add_argument("--sample-t-max", type=float, default=None, help="Override sampling interval max from run config.")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto", help="Device for generation and FID feature extraction.")
    parser.add_argument("--datasets-download", action="store_true", help="Allow torch-fidelity to download CIFAR10 if missing.")
    parser.add_argument("--keep-samples", action="store_true", help="Keep generated PNGs after computing FID.")
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


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for FID evaluation, but no CUDA device is available.")
    return torch.device(device_arg)


def resolve_checkpoint(run_dir: Path, ckpt_path: str, checkpoint_name: str) -> Path:
    if ckpt_path:
        path = Path(ckpt_path).expanduser().resolve()
    else:
        path = (run_dir / "checkpoints" / f"{checkpoint_name}.pt").resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def denormalize_images(samples: torch.Tensor, mean: tuple[float, ...], std: tuple[float, ...]) -> torch.Tensor:
    mean_tensor = torch.tensor(mean, device=samples.device, dtype=samples.dtype).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, device=samples.device, dtype=samples.dtype).view(1, -1, 1, 1)
    return (samples.clamp(-1, 1) * std_tensor + mean_tensor).clamp(0, 1)


@torch.no_grad()
def generate_fid_samples(
    model: torch.nn.Module,
    config: TrainConfig,
    out_dir: Path,
    device: torch.device,
    num_samples: int,
    gen_bsz: int,
    sample_seed: int,
) -> None:
    spec = get_dataset_spec(config.dataset)
    model.eval()
    out_dir.mkdir(parents=True, exist_ok=True)

    generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    generator.manual_seed(sample_seed)

    total_batches = math.ceil(num_samples / gen_bsz)
    progress = tqdm(total=num_samples, desc="Generating FID samples")

    for batch_idx in range(total_batches):
        start = batch_idx * gen_bsz
        current_bsz = min(gen_bsz, num_samples - start)
        if current_bsz <= 0:
            break

        noise = torch.randn(
            current_bsz,
            spec.channels,
            spec.image_size,
            spec.image_size,
            generator=generator,
            device=device,
        )
        labels = None
        if config.variant == "pmf":
            labels = (torch.arange(start, start + current_bsz, device=device, dtype=torch.long) % spec.num_classes)

        samples = generate_samples(model, noise, config.variant, labels=labels, config=config)
        images = denormalize_images(samples, spec.mean, spec.std).cpu()

        for local_idx in range(current_bsz):
            sample_id = start + local_idx
            save_image(images[local_idx], out_dir / f"{sample_id:06d}.png")

        progress.update(current_bsz)

    progress.close()


def main() -> None:
    args = parse_args()
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

    if config.dataset != "cifar10":
        raise ValueError(f"FID evaluation script currently targets CIFAR10 runs, got dataset={config.dataset!r}.")
    if config.variant == "pmf" and args.num_samples % 10 != 0:
        raise ValueError("For CIFAR10 pMF runs, --num-samples should be divisible by 10 so class labels stay balanced.")

    device = resolve_device(args.device)
    ckpt_path = resolve_checkpoint(run_dir, args.ckpt_path, args.checkpoint)
    set_seed(args.sample_seed)

    spec = get_dataset_spec(config.dataset)
    model = build_model(config, spec).to(device)
    optimizer = build_optimizer(model, config)

    try:
        state = restore_checkpoint(str(ckpt_path), model, optimizer, device)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load checkpoint {ckpt_path}. "
            "The checkpoint does not match the current pMF implementation. "
            "For older CIFAR10 runs this usually means the file is corrupted or predates the current JAX-aligned rewrite, "
            "so the experiment needs to be rerun before FID can be computed."
        ) from exc

    fid_dir = run_dir / "fid"
    fid_dir.mkdir(parents=True, exist_ok=True)
    sample_dir = fid_dir / f"samples_{ckpt_path.stem}_{args.num_samples}"
    metrics_path = fid_dir / f"metrics_{ckpt_path.stem}_{args.num_samples}.json"

    if sample_dir.exists():
        shutil.rmtree(sample_dir)

    print(f"Run dir: {run_dir}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Generation device: {device}")
    print(
        "Sampling config:",
        json.dumps(
            {
                "sample_steps": config.sample_steps,
                "sample_omega": config.sample_omega,
                "sample_t_min": config.sample_t_min,
                "sample_t_max": config.sample_t_max,
                "num_samples": args.num_samples,
                "gen_bsz": args.gen_bsz,
            },
            ensure_ascii=False,
        ),
    )

    try:
        from utils.fidelity_wrapper import calculate_metrics
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "FID evaluation requires torch-fidelity. Install project dependencies with `pip install -r requirements.txt`."
        ) from exc

    generate_fid_samples(
        model=model,
        config=config,
        out_dir=sample_dir,
        device=device,
        num_samples=args.num_samples,
        gen_bsz=args.gen_bsz,
        sample_seed=args.sample_seed,
    )

    metrics = calculate_metrics(
        input1=str(sample_dir),
        input2=args.ref,
        cuda=device.type == "cuda",
        batch_size=args.fid_batch_size,
        fid=True,
        isc=args.isc,
        kid=False,
        prc=False,
        verbose=True,
        datasets_root=args.data_root,
        datasets_download=args.datasets_download,
    )

    payload = {
        "run_dir": str(run_dir),
        "checkpoint": str(ckpt_path),
        "step": state.step,
        "dataset": config.dataset,
        "variant": config.variant,
        "optimizer": config.optimizer,
        "attn_impl": config.attn_impl,
        "num_samples": args.num_samples,
        "gen_bsz": args.gen_bsz,
        "fid_batch_size": args.fid_batch_size,
        "sample_seed": args.sample_seed,
        "reference": args.ref,
        "metrics": metrics,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if not args.keep_samples:
        shutil.rmtree(sample_dir)

    fid_value = metrics.get("frechet_inception_distance")
    print(f"FID: {fid_value}")
    print(f"Saved metrics to: {metrics_path}")
    if args.keep_samples:
        print(f"Generated samples kept at: {sample_dir}")


if __name__ == "__main__":
    main()
