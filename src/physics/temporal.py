"""
Temporal matrix construction for ridge regression baselines.
"""

import numpy as np


def build_heat_temporal_matrix(Phi, eigenvalues, dt, T_raw):
    """
    Build A_heat in R^{MT x K}.

    A[m*T + j, n] = Phi[m,n] * exp(-lambda_n * tau_j)
    where tau_j = (j - (T-1)) * dt  (all <= 0, causal window ending at t0)
    """
    M, K = Phi.shape
    tau = (np.arange(T_raw) - (T_raw - 1)) * dt
    basis = np.exp(-eigenvalues[None, :] * tau[:, None])  # (T, K)
    A = Phi[:, None, :] * basis[None, :, :]  # (M, T, K)
    return A.reshape(M * T_raw, K)


def build_wave_temporal_matrix(Phi, eigenvalues, dt, T_raw, gamma, c):
    """
    Build A_wave in R^{MT x 2K} for damped wave (acoustic or membrane).

    State vector x = [a_1, adot_1/omega_1, a_2, adot_2/omega_2, ...] (2K,)
    """
    M, K = Phi.shape
    omega = c * np.sqrt(eigenvalues)
    omega_d = np.sqrt(np.maximum(omega**2 - (gamma / 2)**2, 0.0))

    tau = (np.arange(T_raw) - (T_raw - 1)) * dt  # all <= 0

    A = np.zeros((M * T_raw, 2 * K))

    for n in range(K):
        if omega_d[n] < 1e-10:
            # Overdamped mode
            envelope = np.exp(-gamma * tau / 2)
            for m in range(M):
                row_start = m * T_raw
                A[row_start:row_start + T_raw, 2 * n] = Phi[m, n] * envelope
                A[row_start:row_start + T_raw, 2 * n + 1] = (
                    Phi[m, n] * tau * envelope
                )
        else:
            envelope = np.exp(-gamma * tau / 2)
            cos_part = envelope * np.cos(omega_d[n] * tau)
            sin_part = envelope * np.sin(omega_d[n] * tau)
            for m in range(M):
                row_start = m * T_raw
                A[row_start:row_start + T_raw, 2 * n] = Phi[m, n] * cos_part
                A[row_start:row_start + T_raw, 2 * n + 1] = Phi[m, n] * sin_part

    return A


def generate_heat_signals(room, seed, noise_frac=0.01):
    """
    Generate synthetic heat equation mic signals.

    Heat impulse: a_n(t) = a_n(0) * exp(-lambda_n * t)
    IC: a_n(0) ~ N(0, 1/(1+lambda_n))

    Returns:
        Y: (M, T_total_steps) mic signals with noise
        a_snaps: (K, n_snaps) modal amplitudes at snapshot times
        noise_level: scalar
    """
    rng = np.random.default_rng(seed)
    ev = room["eigenvalues"]
    Phi = room["Phi"]
    dt = room["dt_sim"]
    n_snaps = room["n_snaps"]
    sps = room["steps_per_snap"]
    K = room["K"]

    T_total_steps = n_snaps * sps
    a0 = rng.standard_normal(K) / np.sqrt(1.0 + ev)
    times = np.arange(T_total_steps) * dt
    a_full = a0[:, None] * np.exp(-ev[:, None] * times[None, :])

    Y_clean = Phi @ a_full
    noise_level = noise_frac * np.std(Y_clean)
    Y = Y_clean + rng.standard_normal(Y_clean.shape) * noise_level

    snap_steps = np.arange(n_snaps) * sps
    a_snaps = a_full[:, snap_steps]

    return Y, a_snaps, noise_level
