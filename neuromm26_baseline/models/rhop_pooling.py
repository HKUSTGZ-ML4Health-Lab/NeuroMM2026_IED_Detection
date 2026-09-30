"""Riemannian high-order pooling blocks for EEG token features."""

from __future__ import annotations

import torch
from torch import nn


class RiemannianHighOrderPooling(nn.Module):
    """Low-dimensional SPD log-covariance pooling over token sequences.

    This keeps the expensive eigendecomposition in a small projected space and
    returns the upper-triangular log-covariance descriptor.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        token_dim: int = 32,
        dropout: float = 0.0,
        eps: float = 1e-4,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.token_dim = int(token_dim)
        self.eps = float(eps)
        if self.input_dim <= 0 or self.token_dim <= 0:
            raise ValueError("RiemannianHighOrderPooling dimensions must be positive")
        self.output_dim = self.token_dim * (self.token_dim + 1) // 2
        self.project = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.token_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.output_norm = nn.LayerNorm(self.output_dim)
        tri = torch.triu_indices(self.token_dim, self.token_dim)
        self.register_buffer("tri_rows", tri[0], persistent=False)
        self.register_buffer("tri_cols", tri[1], persistent=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected token tensor [B,N,D], got {tuple(tokens.shape)}")
        if int(tokens.shape[1]) <= 0:
            raise ValueError("RiemannianHighOrderPooling requires at least one token")

        dtype = tokens.dtype
        z = self.project(tokens).float()
        z = z - z.mean(dim=1, keepdim=True)
        denom = max(int(z.shape[1]) - 1, 1)
        cov = z.transpose(1, 2).matmul(z) / float(denom)
        eye = torch.eye(self.token_dim, device=z.device, dtype=z.dtype).view(1, self.token_dim, self.token_dim)
        cov = cov + eye * self.eps

        eigvals, eigvecs = torch.linalg.eigh(cov)
        eigvals = eigvals.clamp_min(self.eps).log()
        log_cov = eigvecs.matmul(torch.diag_embed(eigvals)).matmul(eigvecs.transpose(1, 2))
        descriptor = log_cov[:, self.tri_rows, self.tri_cols]
        return self.output_norm(descriptor).to(dtype=dtype)
