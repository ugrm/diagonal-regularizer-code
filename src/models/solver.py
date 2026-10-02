# Model architecture:
#   M1 = UnrolledRidgeSolver(conditioning=True)   — CondNet-conditioned Gamma
#   M2 = UnrolledRidgeSolver(conditioning=False)  — learned-but-fixed Gamma
"""
Phi-Conditioned Unrolled Ridge Solver for Modal Estimation
==========================================================

Unrolls L iterations of ridge regression gradient descent:
    a^(l) = a^(l-1) - eta^(l) * (A^TA a^(l-1) + alpha^(l) Gamma^(l) * a^(l-1) - A^T y)

Gamma^(l) is produced by a CondNet MLP conditioned on per-mode features
derived from the observation matrix and eigenvalues.

The system operates in 2K dimensions (interleaved cos/sin components per mode).
CondNet outputs K values, each applied to both components of a mode,
matching the analytical Gamma=lambda^p structure from TL v2.
"""

import torch
import torch.nn as nn


class CondNet(nn.Module):
    """Per-mode MLP: maps per-mode features to Gamma_k > 0."""

    def __init__(self, feat_dim=10):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus(),  # ensure Gamma > 0
        )

    def forward(self, mode_features):
        """
        mode_features: (B, K, feat_dim)
        Returns: (B, K) — one Gamma value per physical mode
        """
        B, K, F = mode_features.shape
        out = self.mlp(mode_features.reshape(B * K, F))  # (B*K, 1)
        return out.reshape(B, K)


class FixedGamma(nn.Module):
    """Ablation: learned-but-fixed Gamma (NOT room-conditioned)."""

    def __init__(self, K=50):
        super().__init__()
        self.log_gamma = nn.Parameter(torch.zeros(K))

    def forward(self, mode_features):
        B = mode_features.shape[0]
        return torch.nn.functional.softplus(self.log_gamma).unsqueeze(0).expand(B, -1)


class UnrolledRidgeSolver(nn.Module):
    """
    Unrolled ridge solver with learned {eta, alpha, Gamma} per layer.

    Args:
        L: number of unrolled layers
        K: number of physical modes (system is 2K-dimensional)
        feat_dim: per-mode feature dimension for CondNet
        share_condnet: if True, all layers share one CondNet
        conditioning: if False, use FixedGamma (no Phi-conditioning)
    """

    def __init__(self, L=10, K=50, feat_dim=10, share_condnet=True,
                 conditioning=True):
        super().__init__()
        self.L = L
        self.K = K
        self.share_condnet = share_condnet
        self.conditioning = conditioning

        if conditioning:
            if share_condnet:
                self.condnet = CondNet(feat_dim)
            else:
                self.condnets = nn.ModuleList(
                    [CondNet(feat_dim) for _ in range(L)])
        else:
            if share_condnet:
                self.fixed_gamma = FixedGamma(K)
            else:
                self.fixed_gammas = nn.ModuleList(
                    [FixedGamma(K) for _ in range(L)])

        self.log_eta = nn.Parameter(torch.zeros(L))
        self.log_alpha = nn.Parameter(torch.zeros(L))

    def _get_gamma_module(self, l):
        if self.conditioning:
            if self.share_condnet:
                return self.condnet
            return self.condnets[l]
        else:
            if self.share_condnet:
                return self.fixed_gamma
            return self.fixed_gammas[l]

    def forward(self, ATA, ATy, mode_features, a_init=None, return_gamma=False):
        """
        ATA: (B, 2K, 2K) — pre-normalized by max diagonal
        ATy: (B, 2K)     — same normalization
        mode_features: (B, K, feat_dim)
        a_init: (B, 2K) or None (zero init)

        Returns:
            a_hat: (B, 2K) estimated modal coefficients
            gammas: list of (B, K) tensors if return_gamma else None
        """
        B = ATA.shape[0]
        device = ATA.device

        if a_init is not None:
            a_hat = a_init
        else:
            a_hat = torch.zeros(B, 2 * self.K, device=device, dtype=ATA.dtype)

        gammas = [] if return_gamma else None

        for l in range(self.L):
            eta = torch.exp(self.log_eta[l])
            alpha = torch.exp(self.log_alpha[l])

            Gamma_K = self._get_gamma_module(l)(mode_features)  # (B, K)

            if return_gamma:
                gammas.append(Gamma_K.detach())

            # Expand K -> 2K: same Gamma for cos and sin components
            Gamma_2K = Gamma_K.repeat_interleave(2, dim=1)  # (B, 2K)

            # grad = ATA @ a_hat + alpha * Gamma * a_hat - ATy
            grad = (torch.bmm(ATA, a_hat.unsqueeze(-1)).squeeze(-1)
                    + alpha * Gamma_2K * a_hat
                    - ATy)

            a_hat = a_hat - eta * grad

        return a_hat, gammas

    def extract_amplitudes(self, a_hat):
        """Extract K amplitude components from 2K solution (cos = even indices)."""
        return a_hat[:, 0::2]
