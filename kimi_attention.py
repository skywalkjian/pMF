"""
Full Attention Residuals (arXiv:2603.15031).

Replaces additive residual connections in Transformers with depth-wise
attention over all preceding layer outputs.  A learned pseudo-query vector
attends over the depth axis via RMS-normalized keys, producing a softmax-
weighted combination that feeds the next sub-layer.

Zero-initialised query => initial behaviour is uniform averaging over
all history entries, degenerating gracefully to a mean-pooling residual.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def rms_norm_last_dim(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Root-mean-square normalisation along the last dimension."""
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class FullAttentionResidual(nn.Module):
    """
    Full Attention Residuals from arXiv:2603.15031.

    Each layer uses a learned pseudo-query to attend over the embedding and all
    preceding layer outputs along the depth axis.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, history: list[torch.Tensor]) -> torch.Tensor:
        if not history:
            raise ValueError("FullAttentionResidual expects a non-empty history.")
        keys = torch.stack(history, dim=2)  # (B, T, N_layers, D)
        scores = torch.einsum(
            "d,btnd->btn",
            self.query,
            rms_norm_last_dim(keys, eps=self.eps),
        )
        weights = scores.softmax(dim=-1, dtype=torch.float32).to(keys.dtype)
        return torch.einsum("btn,btnd->btd", weights, keys)
