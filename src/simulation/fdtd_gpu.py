"""
GPU-accelerated 2D FDTD solver using Numba CUDA.

This provides 10-50x speedup over the CPU version for large grids.
"""
import numpy as np
from numba import cuda, float32, float64
import math

# CUDA kernel for FDTD time step
@cuda.jit
def fdtd_step_kernel(
    p_prev, p_curr, p_next,
    mask, robin_inv_denom,
    wall_left, wall_right, wall_down, wall_up,
    coeff_curr, coeff_prev, cfl2_denom,
    nx, ny
):
    """Single FDTD time step on GPU."""
    i, j = cuda.grid(2)

    # Bounds check (interior only: 1 to nx-2, 1 to ny-2)
    if i < 1 or i >= nx - 1 or j < 1 or j >= ny - 1:
        return

    if not mask[i, j]:
        p_next[i, j] = 0.0
        return

    # Get Robin factor for this cell
    rid = robin_inv_denom[i, j]

    # Compute effective neighbor values with Robin BC
    # If neighbor is wall, use ghost value: p_center * rid
    p_center = p_curr[i, j]

    # Left neighbor
    if wall_left[i - 1, j - 1]:  # wall arrays are offset by 1
        p_left = p_center * rid
    else:
        p_left = p_curr[i - 1, j]

    # Right neighbor
    if wall_right[i - 1, j - 1]:
        p_right = p_center * rid
    else:
        p_right = p_curr[i + 1, j]

    # Down neighbor
    if wall_down[i - 1, j - 1]:
        p_down = p_center * rid
    else:
        p_down = p_curr[i, j - 1]

    # Up neighbor
    if wall_up[i - 1, j - 1]:
        p_up = p_center * rid
    else:
        p_up = p_curr[i, j + 1]

    # 5-point Laplacian
    lap = p_left + p_right + p_down + p_up - 4.0 * p_center

    # Leapfrog update
    p_next[i, j] = (
        coeff_curr[i, j] * p_center
        - coeff_prev[i, j] * p_prev[i, j]
        + cfl2_denom[i, j] * lap
    )


@cuda.jit
def inject_source_kernel(p, si, sj, value):
    """Inject source at a single point."""
    if cuda.grid(1) == 0:
        p[si, sj] += value


@cuda.jit
def record_mics_kernel(p, mic_ix, mic_iy, y_mics, n, n_mics):
    """Record pressure at microphone locations."""
    mi = cuda.grid(1)
    if mi < n_mics:
        ix = mic_ix[mi]
        iy = mic_iy[mi]
        y_mics[mi, n] = p[ix, iy]


@cuda.jit
def apply_mask_kernel(p, mask, nx, ny):
    """Zero pressure outside room."""
    i, j = cuda.grid(2)
    if i < nx and j < ny:
        if not mask[i, j]:
            p[i, j] = 0.0


@cuda.jit
def compute_energy_kernel(p_curr, p_next, mask, energy_out, dt, dx, c, nx, ny):
    """Compute total energy (reduced sum)."""
    i, j = cuda.grid(2)

    # Shared memory for block reduction
    shared = cuda.shared.array(256, dtype=float64)
    tid = cuda.threadIdx.x + cuda.threadIdx.y * cuda.blockDim.x

    local_energy = 0.0
    if i > 0 and i < nx - 1 and j > 0 and j < ny - 1 and mask[i, j]:
        # Velocity squared
        vel = (p_next[i, j] - p_curr[i, j]) / dt
        vel2 = vel * vel

        # Gradient squared
        gx = (p_next[i + 1, j] - p_next[i - 1, j]) / (2 * dx)
        gy = (p_next[i, j + 1] - p_next[i, j - 1]) / (2 * dx)
        grad2 = gx * gx + gy * gy

        local_energy = vel2 + c * c * grad2

    shared[tid] = local_energy
    cuda.syncthreads()

    # Block reduction
    s = 128
    while s > 0:
        if tid < s and tid + s < 256:
            shared[tid] += shared[tid + s]
        cuda.syncthreads()
        s //= 2

    # Atomic add to global result
    if tid == 0:
        cuda.atomic.add(energy_out, 0, shared[0])


class FDTDSolverGPU:
    """
    GPU-accelerated FDTD solver using Numba CUDA.

    Provides 10-50x speedup over CPU for typical grid sizes.
    """

    def __init__(self, device_id: int = 0):
        """
        Args:
            device_id: CUDA device ID (0 or 1 for dual GPU)
        """
        cuda.select_device(device_id)
        self.device_id = device_id

    def run(
        self,
        nx: int, ny: int, n_steps: int,
        dx: float, dt: float, c: float, gamma: float,
        mask: np.ndarray, kappa_grid: np.ndarray,
        source_pos_idx: tuple, source_signal: np.ndarray,
        mic_indices: np.ndarray,
        pml_cells: int = 20, pml_strength: float = 100.0,
        fcut: float = None, save_every: int = 100,
    ) -> dict:
        """
        Run FDTD simulation on GPU.

        Same interface as CPU solver for drop-in replacement.
        """
        # PML sponge layer
        sigma = np.zeros((nx, ny), dtype=np.float64)
        for i in range(pml_cells):
            frac = ((pml_cells - i) / pml_cells) ** 2
            val = pml_strength * frac
            sigma[i, :] += val
            sigma[nx - 1 - i, :] += val
            sigma[:, i] += val
            sigma[:, ny - 1 - i] += val

        total_damp = gamma + sigma
        cfl2 = (c * dt / dx) ** 2

        # Precompute coefficients
        denom = 1.0 + 0.5 * total_damp * dt
        coeff_curr = (2.0 / denom).astype(np.float64)
        coeff_prev = ((1.0 - 0.5 * total_damp * dt) / denom).astype(np.float64)
        cfl2_denom = (cfl2 / denom).astype(np.float64)

        # Robin BC precomputation
        robin_inv_denom = (1.0 / (1.0 + kappa_grid * dx + 1e-10)).astype(np.float64)

        # Wall masks (for interior region)
        wall_left = (~mask[:-2, 1:-1]).astype(np.bool_)
        wall_right = (~mask[2:, 1:-1]).astype(np.bool_)
        wall_down = (~mask[1:-1, :-2]).astype(np.bool_)
        wall_up = (~mask[1:-1, 2:]).astype(np.bool_)

        # Allocate GPU arrays
        d_p_prev = cuda.to_device(np.zeros((nx, ny), dtype=np.float64))
        d_p_curr = cuda.to_device(np.zeros((nx, ny), dtype=np.float64))
        d_p_next = cuda.to_device(np.zeros((nx, ny), dtype=np.float64))

        d_mask = cuda.to_device(mask.astype(np.bool_))
        d_robin_inv_denom = cuda.to_device(robin_inv_denom)
        d_wall_left = cuda.to_device(wall_left)
        d_wall_right = cuda.to_device(wall_right)
        d_wall_down = cuda.to_device(wall_down)
        d_wall_up = cuda.to_device(wall_up)

        d_coeff_curr = cuda.to_device(coeff_curr)
        d_coeff_prev = cuda.to_device(coeff_prev)
        d_cfl2_denom = cuda.to_device(cfl2_denom)

        # Mic indices
        n_mics = len(mic_indices)
        mic_ix = mic_indices[:, 0].astype(np.int32)
        mic_iy = mic_indices[:, 1].astype(np.int32)
        d_mic_ix = cuda.to_device(mic_ix)
        d_mic_iy = cuda.to_device(mic_iy)

        # Output arrays (on CPU, copy periodically)
        y_mics = np.zeros((n_mics, n_steps), dtype=np.float64)
        d_y_mics = cuda.to_device(y_mics)

        energy_every = max(1, n_steps // 100)
        n_energy = n_steps // energy_every + 1
        energy = np.zeros(n_energy, dtype=np.float64)

        save_times = list(range(0, n_steps, save_every))
        u_snapshots = np.zeros((len(save_times), nx, ny), dtype=np.float32)
        snap_idx = 0

        si, sj = source_pos_idx

        # CUDA grid configuration
        threads_per_block = (16, 16)
        blocks_per_grid_x = math.ceil(nx / threads_per_block[0])
        blocks_per_grid_y = math.ceil(ny / threads_per_block[1])
        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)

        mic_threads = min(256, n_mics)
        mic_blocks = math.ceil(n_mics / mic_threads)

        # Energy computation buffer
        d_energy_out = cuda.to_device(np.zeros(1, dtype=np.float64))

        # Main time loop
        for n in range(n_steps):
            # FDTD step
            fdtd_step_kernel[blocks_per_grid, threads_per_block](
                d_p_prev, d_p_curr, d_p_next,
                d_mask, d_robin_inv_denom,
                d_wall_left, d_wall_right, d_wall_down, d_wall_up,
                d_coeff_curr, d_coeff_prev, d_cfl2_denom,
                nx, ny
            )

            # Source injection (no dt² scaling - probe amplitude maps directly to pressure)
            if 0 <= si < nx and 0 <= sj < ny:
                src_val = source_signal[min(n, len(source_signal) - 1)]
                inject_source_kernel[1, 1](d_p_next, si, sj, src_val)

            # Zero outside room
            apply_mask_kernel[blocks_per_grid, threads_per_block](
                d_p_next, d_mask, nx, ny
            )

            # Record mics
            record_mics_kernel[mic_blocks, mic_threads](
                d_p_next, d_mic_ix, d_mic_iy, d_y_mics, n, n_mics
            )

            # Energy (every N steps)
            if n % energy_every == 0:
                d_energy_out.copy_to_device(np.zeros(1, dtype=np.float64))
                compute_energy_kernel[blocks_per_grid, threads_per_block](
                    d_p_curr, d_p_next, d_mask, d_energy_out,
                    dt, dx, c, nx, ny
                )
                energy[n // energy_every] = d_energy_out.copy_to_host()[0]

            # Snapshot
            if snap_idx < len(save_times) and n == save_times[snap_idx]:
                u_snapshots[snap_idx] = d_p_next.copy_to_host().astype(np.float32)
                snap_idx += 1

            # Rotate buffers (swap pointers)
            d_p_prev, d_p_curr, d_p_next = d_p_curr, d_p_next, d_p_prev

        # Copy results back
        y_mics = d_y_mics.copy_to_host()

        # Temporal bandlimiting on mic recordings
        if fcut is not None:
            y_mics = self._bandlimit_temporal(y_mics, dt, fcut)

        save_times_sec = [t * dt for t in save_times]

        return dict(
            u_snapshots=u_snapshots,
            y_mics=y_mics,
            energy=energy,
            energy_every=energy_every,
            save_times=save_times_sec,
        )

    def _bandlimit_temporal(self, y_mics: np.ndarray, dt: float, fcut: float) -> np.ndarray:
        """Apply temporal low-pass filter to mic recordings."""
        n_mics, n_steps = y_mics.shape
        y_filtered = np.zeros_like(y_mics)

        for mi in range(n_mics):
            signal = y_mics[mi, :]
            spectrum = np.fft.rfft(signal)
            freqs = np.fft.rfftfreq(n_steps, d=dt)

            rolloff_width = fcut * 0.1
            f_start = fcut - rolloff_width
            mask = np.ones_like(freqs)

            transition = (freqs >= f_start) & (freqs <= fcut)
            mask[transition] = 0.5 * (1.0 + np.cos(np.pi * (freqs[transition] - f_start) / rolloff_width))
            mask[freqs > fcut] = 0.0

            spectrum_filtered = spectrum * mask
            y_filtered[mi, :] = np.fft.irfft(spectrum_filtered, n=n_steps)

        return y_filtered


def test_gpu_solver():
    """Quick test to verify GPU solver works."""
    print("Testing GPU FDTD solver...")

    nx, ny = 200, 200
    n_steps = 1000
    dx, dt = 0.02, 2.5e-5
    c, gamma = 343.0, 5.0

    mask = np.zeros((nx, ny), dtype=bool)
    mask[20:180, 20:180] = True

    kappa_grid = np.zeros((nx, ny))
    kappa_grid[mask] = 2.0

    source_signal = np.zeros(n_steps)
    source_signal[0] = 1.0

    mic_indices = np.array([[100, 100], [50, 50]])

    import time

    # GPU test
    solver = FDTDSolverGPU(device_id=0)
    start = time.time()
    result = solver.run(
        nx=nx, ny=ny, n_steps=n_steps,
        dx=dx, dt=dt, c=c, gamma=gamma,
        mask=mask, kappa_grid=kappa_grid,
        source_pos_idx=(100, 100),
        source_signal=source_signal,
        mic_indices=mic_indices,
        pml_cells=20, pml_strength=100.0,
        fcut=2000.0, save_every=100,
    )
    gpu_time = time.time() - start

    print(f"GPU time: {gpu_time:.2f}s for {n_steps} steps")
    print(f"Rate: {n_steps/gpu_time:.0f} steps/sec")
    print(f"Max energy: {result['energy'].max():.2e}")
    print(f"Stable: {result['energy'].max() < 1e10}")

    return result


if __name__ == "__main__":
    test_gpu_solver()
