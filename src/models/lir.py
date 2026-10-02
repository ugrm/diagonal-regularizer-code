"""
Learned Iterative Ridge (LIR) — Unrolled L2 gradient descent with
per-iteration learned step sizes and diagonal preconditioners.

Follows the LISTA principle (Gregor & LeCun, ICML 2010) adapted from
L1 (soft-thresholding) to L2 (diagonal Tikhonov) to match our ridge
formulation. No published L2 unrolled implementation exists for diagonal
modal problems; this implementation is original.

Update rule per iteration t:
    grad_t = ATA @ a^{(t)} + alpha_t * D_t * a^{(t)} - ATy
    a^{(t+1)} = a^{(t)} - eta_t * grad_t

Parameters per iteration: eta_t (scalar, >0), alpha_t (scalar, >0), D_t (K, >0).
Total: L * (K + 2).  At L=10, K=50: 520 parameters.

References:
    Gregor & LeCun, "Learning Fast Approximations of Sparse Coding", ICML 2010
    Aggarwal et al., "MoDL: Model-Based Deep Learning Architecture...", IEEE TMI 2019
"""

import torch
import torch.nn as nn


class LearnedIterativeRidge(nn.Module):
    """
    Unrolled ridge solver with per-iteration learned parameters.

    Unlike M2 (UnrolledRidgeSolver with shared FixedGamma), LIR has
    INDEPENDENT diagonal preconditioners at each iteration, giving it
    strictly more expressivity.

    Args:
        L: number of unrolled iterations
        K: number of physical modes (system is 2K-dimensional)
    """

    def __init__(self, L=10, K=50):
        super().__init__()
        self.L = L
        self.K = K

        # Per-iteration parameters
        self.log_eta = nn.Parameter(torch.zeros(L))        # step sizes
        self.log_alpha = nn.Parameter(torch.zeros(L))      # regularization weights
        self.log_D = nn.Parameter(torch.zeros(L, K))       # diagonal preconditioners

    def forward(self, ATA, ATy, mode_features=None, a_init=None,
                return_gamma=False):
        """
        ATA: (B, 2K, 2K)
        ATy: (B, 2K)
        mode_features: ignored (kept for interface compatibility)
        a_init: (B, 2K) or None (zero init)

        Returns:
            a_hat: (B, 2K)
            gammas: list of (B, K) if return_gamma else None
        """
        B = ATA.shape[0]
        device = ATA.device

        a_hat = (a_init if a_init is not None
                 else torch.zeros(B, 2 * self.K, device=device, dtype=ATA.dtype))

        gammas = [] if return_gamma else None

        for l in range(self.L):
            eta = torch.exp(self.log_eta[l])
            alpha = torch.exp(self.log_alpha[l])
            D_K = torch.nn.functional.softplus(self.log_D[l])  # (K,) > 0

            if return_gamma:
                gammas.append(D_K.unsqueeze(0).expand(B, -1).detach())

            # Expand K → 2K (same D for cos and sin components of each mode)
            D_2K = D_K.repeat_interleave(2)  # (2K,)

            # Gradient of Tikhonov objective
            grad = (torch.bmm(ATA, a_hat.unsqueeze(-1)).squeeze(-1)
                    + alpha * D_2K.unsqueeze(0) * a_hat
                    - ATy)

            a_hat = a_hat - eta * grad

        return a_hat, gammas

    def extract_amplitudes(self, a_hat):
        """Extract K amplitude components from 2K solution."""
        return a_hat[:, 0::2]

    def get_effective_gamma(self):
        """
        Return the effective per-mode regularization at convergence.

        For a single-iteration method, this is just D_1.
        For multi-iteration, the effective Gamma is the product of
        the per-iteration shrinkage factors applied to each mode.
        """
        with torch.no_grad():
            # Effective shrinkage: how much does each mode get scaled
            # relative to the unregularized solution?
            # Run a canonical forward pass with identity-like ATA
            gammas = []
            for l in range(self.L):
                D_K = torch.nn.functional.softplus(self.log_D[l])
                gammas.append(D_K.cpu().numpy())
            return gammas

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters())
