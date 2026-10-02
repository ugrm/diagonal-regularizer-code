"""
2D FDTD for the damped wave equation with Robin BC, PML, spectral LP.
"""
import numpy as np
from scipy.fft import fft2, ifft2, fftfreq


def run_fdtd_2d(
    nx, ny, n_steps, dx, dt, c, gamma,
    mask,            # (nx,ny) bool
    kappa_grid,      # (nx,ny) Robin κ
    source_pos_idx,  # (i, j)
    source_signal,   # (n_steps,)
    mic_indices,     # (n_mics, 2)
    pml_cells=10, pml_strength=100.0,
    fcut=None, lp_every=0,  # DISABLED - FFT filter causes instability with Robin BC
    save_every=None,
):
    p_prev = np.zeros((nx, ny))
    p_curr = np.zeros((nx, ny))
    p_next = np.zeros((nx, ny))

    # PML sponge
    sigma = np.zeros((nx, ny))
    for i in range(pml_cells):
        frac = ((pml_cells - i) / pml_cells) ** 2
        val = pml_strength * frac
        sigma[i, :] += val;  sigma[nx-1-i, :] += val
        sigma[:, i] += val;  sigma[:, ny-1-i] += val

    total_damp = gamma + sigma
    cfl2 = (c * dt / dx) ** 2

    # Spectral low-pass mask
    lp_mask = None
    if fcut is not None:
        kx = fftfreq(nx, d=dx);  ky = fftfreq(ny, d=dx)
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        K = np.sqrt(KX**2 + KY**2)
        k_cut = fcut / c
        width = k_cut * 0.15
        lp_mask = 1.0 / (1.0 + np.exp((K - k_cut) / (width + 1e-30)))

    n_mics = len(mic_indices)
    y_mics = np.zeros((n_mics, n_steps))

    # Energy computed every N steps (not every step - too expensive)
    energy_every = max(1, n_steps // 100)  # ~100 energy samples
    energy = np.zeros(n_steps // energy_every + 1)

    if save_every is None:
        save_every = max(1, n_steps // 64)
    save_times = list(range(0, n_steps, save_every))
    u_snapshots = np.zeros((len(save_times), nx, ny), dtype=np.float32)
    snap_idx = 0

    si, sj = source_pos_idx

    # Precompute Robin boundary mask (doesn't change during simulation)
    neighbor_outside = (
        ~np.roll(mask, 1, axis=0) |
        ~np.roll(mask, -1, axis=0) |
        ~np.roll(mask, 1, axis=1) |
        ~np.roll(mask, -1, axis=1)
    )
    robin_boundary = mask & neighbor_outside & (kappa_grid > 0)
    robin_factor = 1.0 - kappa_grid * dx  # precompute scaling

    # Precompute valid mic indices
    mic_ix = mic_indices[:, 0]
    mic_iy = mic_indices[:, 1]
    valid_mics = (mic_ix >= 0) & (mic_ix < nx) & (mic_iy >= 0) & (mic_iy < ny)

    for n in range(n_steps):
        # 5-point Laplacian
        lap = np.zeros_like(p_curr)
        lap[1:-1, 1:-1] = (
            p_curr[2:, 1:-1] + p_curr[:-2, 1:-1]
            + p_curr[1:-1, 2:] + p_curr[1:-1, :-2]
            - 4.0 * p_curr[1:-1, 1:-1])

        # Leapfrog with damping
        denom = 1.0 + 0.5 * total_damp * dt
        p_next[1:-1, 1:-1] = (
            (2.0 * p_curr[1:-1, 1:-1]
             - (1.0 - 0.5 * total_damp[1:-1, 1:-1] * dt) * p_prev[1:-1, 1:-1]
             + cfl2 * lap[1:-1, 1:-1])
            / denom[1:-1, 1:-1])

        # Source injection
        if 0 <= si < nx and 0 <= sj < ny:
            p_next[si, sj] += dt**2 * source_signal[min(n, len(source_signal)-1)]

        # Robin BC at boundary (using precomputed mask)
        p_next[robin_boundary] *= robin_factor[robin_boundary]

        # Zero outside room
        p_next[~mask] = 0.0

        # Spectral low-pass
        if lp_mask is not None and n > 0 and n % lp_every == 0:
            p_next = np.real(ifft2(fft2(p_next) * lp_mask))

        # Record mics (vectorized)
        y_mics[valid_mics, n] = p_next[mic_ix[valid_mics], mic_iy[valid_mics]]

        # Energy proxy (computed every N steps for efficiency)
        if n % energy_every == 0:
            vel2 = ((p_next - p_curr) / dt) ** 2
            gx = np.zeros_like(p_curr); gy = np.zeros_like(p_curr)
            gx[1:-1, :] = (p_next[2:, :] - p_next[:-2, :]) / (2*dx)
            gy[:, 1:-1] = (p_next[:, 2:] - p_next[:, :-2]) / (2*dx)
            energy[n // energy_every] = np.sum(mask * (vel2 + c**2 * (gx**2 + gy**2)))

        # Snapshot
        if snap_idx < len(save_times) and n == save_times[snap_idx]:
            u_snapshots[snap_idx] = p_next.astype(np.float32)
            snap_idx += 1

        p_prev[:] = p_curr
        p_curr[:] = p_next

    return dict(u_snapshots=u_snapshots, y_mics=y_mics,
                energy=energy, energy_every=energy_every, save_times=save_times)
