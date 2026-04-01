#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RUNS = [
    (
        "pMF + AdamW + naive",
        "runs/cifar10_pmf_adamw_transformer_naive/20260401-141528",
        "#B55D24",
    ),
    (
        "pMF + Muon + naive",
        "runs/cifar10_pmf_muon_transformer_naive/20260401-150007",
        "#2E8B57",
    ),
    (
        "pMF + Muon + residual",
        "runs/cifar10_pmf_muon_transformer_residual/20260401-155822",
        "#2E5AAC",
    ),
]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ema(values: np.ndarray, alpha: float = 0.03) -> np.ndarray:
    smoothed = np.empty_like(values, dtype=float)
    smoothed[0] = float(values[0])
    for idx in range(1, len(values)):
        smoothed[idx] = alpha * float(values[idx]) + (1.0 - alpha) * smoothed[idx - 1]
    return smoothed


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "assets" / "showcase" / "cifar_metrics_comparison.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)

    for label, run_relpath, color in RUNS:
        run_dir = root / run_relpath
        metrics = load_json(run_dir / "metrics.json")
        config = load_json(run_dir / "config.json")["config"]

        train_loss = np.asarray(metrics["train_loss"], dtype=float)
        train_steps = np.arange(1, len(train_loss) + 1)
        train_smooth = ema(train_loss, alpha=0.03)

        axes[0].plot(train_steps, train_smooth, color=color, linewidth=2.2, label=label)

        val_loss = np.asarray(metrics["val_loss"], dtype=float)
        eval_every = int(config["eval_every"])
        val_steps = np.arange(1, len(val_loss) + 1) * eval_every
        axes[1].plot(
            val_steps,
            val_loss,
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=4.5,
            label=label,
        )

        best_idx = int(np.argmin(val_loss))
        axes[1].scatter(
            [val_steps[best_idx]],
            [val_loss[best_idx]],
            color=color,
            edgecolors="black",
            linewidths=0.6,
            s=42,
            zorder=4,
        )

    axes[0].set_title("Smoothed Train Loss")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[0].set_xlim(0, 5000)
    axes[0].grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Loss")
    axes[1].set_yscale("log")
    axes[1].set_xlim(0, 5000)
    axes[1].grid(True, linestyle="--", linewidth=0.6, alpha=0.35)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("CIFAR10 pMF Metrics Comparison", fontsize=16, y=1.08)
    fig.text(
        0.5,
        -0.02,
        "Train subplot uses EMA smoothing (alpha=0.03). Validation subplot shows raw eval points every 250 steps.",
        ha="center",
        fontsize=10,
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
