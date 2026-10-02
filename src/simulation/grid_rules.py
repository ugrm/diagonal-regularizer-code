"""
Discretisation rules enforcing band-limited correctness.
  1. λ_min = c/fcut ≥ k·Δx  →  Δx ≤ c/(k·fcut)
  2. c·Δt/Δx ≤ α            →  Δt ≤ α·Δx/c
"""
import numpy as np


def choose_dx_dt(c, fcut, k=10, alpha=0.5):
    lam_min = c / fcut
    dx = lam_min / k
    dt = alpha * dx / c
    cfl_actual = c * dt / dx
    assert cfl_actual <= alpha + 1e-12, f"CFL violated: {cfl_actual:.4f}"
    return dict(dx=dx, dt=dt, fcut=fcut, k=k, alpha=alpha,
                c=c, lambda_min=lam_min, cfl_actual=cfl_actual)


def grid_size_for_domain(domain_extent, dx):
    return int(np.ceil(domain_extent / dx)) + 1

def next_fft_friendly(n, max_prime=5):
    """Round up n to the next integer whose prime factors are all <= max_prime."""
    while True:
        m = n
        for p in [2, 3, 5]:
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1

def time_steps(T, dt):
    return int(np.ceil(T / dt))