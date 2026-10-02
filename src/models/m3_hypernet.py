"""
M3: Hypernetwork + closed-form Tikhonov solver.

Unlike M1/M2 (which unroll gradient descent), M3 solves the regularized
normal equations in closed form:
    a_hat = (A^TA + alpha * diag(Gamma_2K))^{-1} A^T y

Gamma is produced by the same CondNet architecture as M1, so the only
architectural difference is closed-form solve vs unrolled iteration.
"""

import torch
import torch.nn as nn

from src.models.solver import CondNet


class ClosedFormTikhonov(nn.Module):
    """M3: Hypernetwork + closed-form Tikhonov solver."""

    def __init__(self, K=50, feat_dim=10):
        super().__init__()
        self.K = K
        self.condnet = CondNet(feat_dim)
        self.log_alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, ATA, ATy, mode_features):
        """
        ATA: (B, 2K, 2K) — pre-normalized
        ATy: (B, 2K)
        mode_features: (B, K, feat_dim)

        Returns:
            a_hat: (B, 2K) estimated modal coefficients
            gamma_k: (B, K) learned regularization spectrum
        """
        alpha = torch.exp(self.log_alpha)
        gamma_k = self.condnet(mode_features)  # (B, K)
        gamma_2k = gamma_k.repeat_interleave(2, dim=1)  # (B, 2K)
        reg = ATA + torch.diag_embed(alpha * gamma_2k)
        a_hat = torch.linalg.solve(reg, ATy.unsqueeze(-1)).squeeze(-1)
        return a_hat, gamma_k

    def extract_amplitudes(self, a_hat):
        """Extract K amplitude components from 2K solution (cos = even indices)."""
        return a_hat[:, 0::2]
