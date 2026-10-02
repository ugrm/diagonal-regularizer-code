#!/usr/bin/env python
"""
Step 15: Rectangular room control experiment — Berry's conjecture validation.

Generates P(p) landscape for rectangular (integrable) rooms using ANALYTICAL
Dirichlet eigenpairs. Compares against generic convex (chaotic) rooms from
the main P-sweep.

Output: data/experiments/rectangular_rooms_psweep.npz

Usage:
    python scripts/rectangular_control.py
    python scripts/rectangular_control.py --seed 42
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analysis.landscape import sweep_p_for_room
from src.geometry.polygon import place_mics

# Fixed parameters — MUST match generic convex comparison
K_TRUNC = 50
M_USE = 8
T_RAW = 100
GAMMA = 5.1          # damping (s^-1), matches dataset default
C_SOUND = 343.0      # speed of sound (m/s)
DT_SIM = 50e-6       # simulation timestep (s), matches dataset
DT_SNAP = 5e-3       # snapshot interval (s), matches dataset
N_SNAPS = 100        # number of snapshots, matches dataset
MIC_WALL_MARGIN = 0.3   # meters
SOURCE_WALL_MARGIN = 0.5 # meters
MIC_SOURCE_MARGIN = 0.5  # meters

# Rooms: 4 rectangles (3x3 dropped — area too small, K_total insufficient)
ROOMS = [
    ("3x6_rect", 3.0, 6.0),
    ("2x8_long", 2.0, 8.0),
    ("4x4_square", 4.0, 4.0),
    ("3x5_rect", 3.0, 5.0),
]

# P-sweep grid — matches configs/default.yaml
P_GRID = np.arange(0.0, 6.01, 0.1)
LAMBDA_GRID = np.array([1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0,
                         100.0, 1e3, 1e4, 1e6, 1e8, 1e10])


def rect_eigenpairs(Lx, Ly, K_total):
    """
    Analytical Dirichlet eigenpairs for rectangle [0,Lx] x [0,Ly].

    Returns eigenvalues (K_total,) and a function to evaluate eigenfunctions.
    Ordered by increasing eigenvalue.
    """
    m_max = int(np.ceil(np.sqrt(K_total * Lx / Ly))) + 10
    n_max = int(np.ceil(np.sqrt(K_total * Ly / Lx))) + 10

    candidates = []
    for m in range(1, m_max + 1):
        for n in range(1, n_max + 1):
            lam = np.pi**2 * (m**2 / Lx**2 + n**2 / Ly**2)
            candidates.append((lam, m, n))

    candidates.sort(key=lambda x: x[0])
    candidates = candidates[:K_total]

    eigenvalues = np.array([c[0] for c in candidates])
    mn_pairs = [(c[1], c[2]) for c in candidates]

    norm = 2.0 / np.sqrt(Lx * Ly)

    def eval_eigenfunctions(points):
        """Evaluate all K_total eigenfunctions at (N, 2) points."""
        x, y = points[:, 0], points[:, 1]
        Phi = np.zeros((len(points), K_total))
        for k, (m, n) in enumerate(mn_pairs):
            Phi[:, k] = norm * np.sin(m * np.pi * x / Lx) * np.sin(n * np.pi * y / Ly)
        return Phi

    return eigenvalues, eval_eigenfunctions, mn_pairs


def simulate_room(Lx, Ly, rng, K_total=200):
    """
    Simulate a rectangular room with analytical eigenpairs.

    Returns a room dict compatible with sweep_p_for_room.
    """
    eigenvalues_all, eval_phi, mn_pairs = rect_eigenpairs(Lx, Ly, K_total)

    vertices = np.array([[0, 0], [Lx, 0], [Lx, Ly], [0, Ly]], dtype=np.float64)
    source_pos = np.array([
        rng.uniform(SOURCE_WALL_MARGIN, Lx - SOURCE_WALL_MARGIN),
        rng.uniform(SOURCE_WALL_MARGIN, Ly - SOURCE_WALL_MARGIN),
    ])

    mic_pos = place_mics(vertices, M_USE, rng, source_pos=source_pos,
                         mic_wall_margin=MIC_WALL_MARGIN,
                         mic_source_margin=MIC_SOURCE_MARGIN)

    Phi_all = eval_phi(mic_pos)

    source_pts = source_pos.reshape(1, 2)
    a0_all = eval_phi(source_pts).flatten()

    omega = C_SOUND * np.sqrt(eigenvalues_all)
    omega_d = np.sqrt(np.maximum(omega**2 - (GAMMA / 2)**2, 0.0))

    sps = int(round(DT_SNAP / DT_SIM))
    T_sig = N_SNAPS * sps

    a_snaps = np.zeros((K_total, N_SNAPS))
    for si in range(N_SNAPS):
        t = si * DT_SNAP
        decay = np.exp(-GAMMA * t / 2)
        cos_part = np.cos(omega_d * t)
        sin_coeff = np.where(omega_d > 1e-10, GAMMA / (2 * omega_d), 0.0)
        sin_part = sin_coeff * np.sin(omega_d * t)
        a_snaps[:, si] = a0_all * decay * (cos_part + sin_part)

    y_click = np.zeros((M_USE, T_sig))
    for step_i in range(T_sig):
        t = step_i * DT_SIM
        decay = np.exp(-GAMMA * t / 2)
        cos_part = np.cos(omega_d * t)
        sin_coeff = np.where(omega_d > 1e-10, GAMMA / (2 * omega_d), 0.0)
        sin_part = sin_coeff * np.sin(omega_d * t)
        a_t = a0_all * decay * (cos_part + sin_part)
        y_click[:, step_i] = Phi_all @ a_t

    K_use = min(K_TRUNC, K_total)

    return {
        "K": K_use,
        "K_total": K_total,
        "eigenvalues": eigenvalues_all[:K_use],
        "Phi": Phi_all[:, :K_use],
        "a": a_snaps[:K_use],
        "y_click": y_click,
        "gamma_room": GAMMA,
        "c": C_SOUND,
        "dt_sim": DT_SIM,
        "dt_snap": DT_SNAP,
        "steps_per_snap": sps,
        "n_snaps": N_SNAPS,
        "source_pos": source_pos,
        "mic_pos": mic_pos,
        "eigenvalues_all": eigenvalues_all,
        "mn_pairs": mn_pairs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-total", type=int, default=200)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = args.out_dir or os.path.join("data", "experiments")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Rectangular control: {len(ROOMS)} rooms, K_trunc={K_TRUNC}, "
          f"K_total={args.k_total}, M={M_USE}, T={T_RAW}")

    room_names = []
    room_areas = []
    P_curves = []

    for name, Lx, Ly in ROOMS:
        print(f"\n--- {name} ({Lx}x{Ly}, area={Lx*Ly}m²) ---")
        t0 = time.time()

        room = simulate_room(Lx, Ly, rng, K_total=args.k_total)

        P_oracle = sweep_p_for_room(
            room, np.array([T_RAW]), P_GRID, LAMBDA_GRID,
            M_values=[M_USE], mic_subsets_4=None,
            is_wave=True, gamma=GAMMA, c=C_SOUND,
        )
        curve = P_oracle[0, 0, :]

        p_star = P_GRID[np.nanargmin(curve)]
        P_star = np.nanmin(curve)
        P_max = np.nanmax(curve)
        elapsed = time.time() - t0
        print(f"  p*={p_star:.1f}, P*={P_star:.4f}, ratio={P_max/P_star:.2f}x ({elapsed:.1f}s)")

        room_names.append(name)
        room_areas.append(Lx * Ly)
        P_curves.append(curve)

    out_path = os.path.join(out_dir, "rectangular_rooms_psweep.npz")
    np.savez(
        out_path,
        p_grid=P_GRID,
        P_curves=np.array(P_curves),
        room_names=np.array(room_names),
        room_areas=np.array(room_areas),
        T_raw=T_RAW,
        K_trunc=K_TRUNC,
        K_total=args.k_total,
        M=M_USE,
        bc_type="dirichlet",
        gamma=GAMMA,
        c=C_SOUND,
        dt_sim=DT_SIM,
        seed=args.seed,
        lambda_grid=LAMBDA_GRID,
    )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
