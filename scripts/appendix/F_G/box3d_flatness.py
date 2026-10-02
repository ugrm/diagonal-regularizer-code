#!/usr/bin/env python
"""Appendix G.5 3D box stress test (T28 3D column; T42 "3D boxes" and "3D worst cells").

Analytic 3D boxes (sine products for Dirichlet, cosine products for Neumann walls with
the DC mode dropped), diffuse-field prior (c_n, beta_n) ~ N(0, lambda_n^-|s|) i.i.d.
over ALL modes below 500 Hz; truncation noise = modes 51..K_total. K = 50 retained,
M = 8 sensors uniform with a 0.3 m wall margin, T in {1, 100, 1000}, the 61-p x 12-alpha
oracle protocol, 20 realizations per cell. Stored per cell: the (20, 3, 61) oracle
curves and meta = [volume, K_total, H, ||E||_op].

The per-cell RNG (sensor positions, then the 20 draws) is seeded with
zlib.crc32((bc, name)) (_common.stable_seed; --seed S appends S to the key and redraws
the sensor layout). The printed values are medians over seeds 0-9 (run_multi_seed.sh);
volume, K_total and H do not depend on the seed. The 8 cells run in a process pool.

Output: data/experiments/appendix/box3d_flatness.npz
Usage:  python scripts/appendix/F_G/box3d_flatness.py [--workers 8] [--seed S --out NAME]
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import time
from multiprocessing import Pool

import numpy as np

from _common import out_path, n_workers, stable_seed, with_seed
from src.analysis.landscape import sweep_p_for_room

S_HAT = 1.13
K_RET = 50
M_USE = 8
GAMMA = 5.1
C_SOUND = 343.0
DT_SIM = 50e-6
DT_SNAP = 5e-3
N_SNAPS = 100
F_CUT = 500.0            # Hz ceiling for K_total
T_VALUES = np.array([1, 100, 1000])
N_REAL = 20              # independent diffuse-field realizations per box
P_GRID = np.round(np.arange(0.0, 6.01, 0.1), 10)
LAMBDA_GRID = np.array([1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0,
                        100.0, 1e3, 1e4, 1e6, 1e8, 1e10])
MARGIN = 0.3
SEED = None             # set by --seed

BOXES = [
    ("small_2.5x3x2.4", 2.5, 3.0, 2.4),
    ("mid_3x4x2.5", 3.0, 4.0, 2.5),
    ("mid_4x5x2.7", 4.0, 5.0, 2.7),
    ("large_6x7x3", 6.0, 7.0, 3.0),
]
BCS = ["dirichlet", "neumann"]


def box_eigenpairs(L, bc):
    """All eigenpairs below the F_CUT ceiling for box L=(Lx,Ly,Lz)."""
    lam_max = (2 * np.pi * F_CUT / C_SOUND) ** 2
    lo = 1 if bc == "dirichlet" else 0
    idx_max = [int(np.floor(np.sqrt(lam_max) * Li / np.pi)) + 1 for Li in L]
    cand = []
    for m in range(lo, idx_max[0] + 1):
        for n in range(lo, idx_max[1] + 1):
            for l in range(lo, idx_max[2] + 1):
                if bc == "neumann" and (m + n + l) == 0:
                    continue  # drop DC mode
                lam = np.pi ** 2 * (m ** 2 / L[0] ** 2 + n ** 2 / L[1] ** 2
                                    + l ** 2 / L[2] ** 2)
                if 0 < lam <= lam_max:
                    cand.append((lam, m, n, l))
    cand.sort(key=lambda x: x[0])
    ev = np.array([c[0] for c in cand])
    mnl = [(c[1], c[2], c[3]) for c in cand]
    V = L[0] * L[1] * L[2]

    def eval_phi(pts):
        Phi = np.empty((len(pts), len(mnl)))
        for k, (m, n, l) in enumerate(mnl):
            if bc == "dirichlet":
                f = (np.sin(m * np.pi * pts[:, 0] / L[0])
                     * np.sin(n * np.pi * pts[:, 1] / L[1])
                     * np.sin(l * np.pi * pts[:, 2] / L[2]))
                norm = np.sqrt(8.0 / V)
            else:
                f = (np.cos(m * np.pi * pts[:, 0] / L[0])
                     * np.cos(n * np.pi * pts[:, 1] / L[1])
                     * np.cos(l * np.pi * pts[:, 2] / L[2]))
                eps = sum(1 for i in (m, n, l) if i == 0)
                norm = np.sqrt(8.0 / V / (2 ** eps))
            Phi[:, k] = norm * f
        return Phi

    return ev, eval_phi


def herfindahl(ev):
    if len(ev) <= K_RET:
        return np.nan
    w = ev[K_RET:] ** (-S_HAT)
    return float(np.sum(w ** 2) / np.sum(w) ** 2)


def e_op(Phi_all, ev):
    if len(ev) <= K_RET:
        return np.nan
    sig = ev[K_RET:] ** (-S_HAT)
    Pt = Phi_all[:, K_RET:]
    R = Pt @ np.diag(sig) @ Pt.T
    s2 = np.trace(R) / M_USE
    E = R / s2 - np.eye(M_USE)
    return float(np.max(np.abs(np.linalg.eigvalsh(E))))


def simulate_realization(ev_all, Phi_all, rng):
    """Diffuse-field draw over all modes -> room dict for sweep_p_for_room."""
    K_total = len(ev_all)
    sd = ev_all ** (-S_HAT / 2)
    c0 = rng.standard_normal(K_total) * sd
    b0 = rng.standard_normal(K_total) * sd

    omega = C_SOUND * np.sqrt(ev_all)
    omega_d = np.sqrt(np.maximum(omega ** 2 - (GAMMA / 2) ** 2, 0.0))
    sps = int(round(DT_SNAP / DT_SIM))
    T_sig = N_SNAPS * sps

    t_sim = np.arange(T_sig) * DT_SIM
    decay = np.exp(-GAMMA * t_sim / 2)
    a_t = (c0[:, None] * np.cos(omega_d[:, None] * t_sim[None, :])
           + b0[:, None] * np.sin(omega_d[:, None] * t_sim[None, :])) * decay[None, :]
    y_click = Phi_all @ a_t

    snap_steps = np.arange(N_SNAPS) * sps
    a_snaps = a_t[:K_RET, snap_steps]

    return {
        "K": min(K_RET, K_total),
        "eigenvalues": ev_all[:K_RET],
        "Phi": Phi_all[:, :K_RET],
        "a": a_snaps,
        "y_click": y_click,
        "gamma_room": GAMMA,
        "c": C_SOUND,
        "dt_sim": DT_SIM,
        "dt_snap": DT_SNAP,
        "steps_per_snap": sps,
        "n_snaps": N_SNAPS,
    }


def delta_at_s(curve):
    j = int(np.nanargmin(curve))
    P_at = np.interp(S_HAT, P_GRID, curve)
    return P_GRID[j], curve[j], (P_at - curve[j]) / curve[j]


def run_cell(cell):
    bc, (name, Lx, Ly, Lz) = cell
    ev_all, eval_phi = box_eigenpairs((Lx, Ly, Lz), bc)
    rng = np.random.default_rng(stable_seed(with_seed((bc, name), SEED)))
    mics = np.column_stack([
        rng.uniform(MARGIN, Lx - MARGIN, M_USE),
        rng.uniform(MARGIN, Ly - MARGIN, M_USE),
        rng.uniform(MARGIN, Lz - MARGIN, M_USE)])
    Phi_all = eval_phi(mics)
    H = herfindahl(ev_all)
    Eop = e_op(Phi_all, ev_all)
    curves = np.full((N_REAL, len(T_VALUES), len(P_GRID)), np.nan)
    for rlz in range(N_REAL):
        room = simulate_realization(ev_all, Phi_all, rng)
        P_or = sweep_p_for_room(room, T_VALUES, P_GRID, LAMBDA_GRID,
                                M_values=[M_USE], mic_subsets_4=None,
                                is_wave=True, gamma=GAMMA, c=C_SOUND)
        curves[rlz] = P_or[:, 0, :]
    return curves, np.array([Lx * Ly * Lz, len(ev_all), H, Eop])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=None,
                    help="extra seed appended to every RNG key (default: the shipped run)")
    ap.add_argument("--out", default=None, help="output name under data/experiments/appendix/")
    args = ap.parse_args()
    global SEED
    SEED = args.seed
    t0 = time.time()
    cells = [(bc, box) for bc in BCS for box in BOXES]
    with Pool(n_workers(args.workers)) as pool:
        res = pool.map(run_cell, cells, chunksize=1)

    rep = [f"# 3D boxes (|s|={S_HAT}, K=50, M=8, fcut={F_CUT:.0f} Hz, {N_REAL} "
           "realizations; per-cell median curve)\n",
           "| cell | V | K_total | H | ||E||_op | T | p* | d(|s|) | flat[0,3] |",
           "|---|---|---|---|---|---|---|---|---|"]
    store = {}
    i3 = np.searchsorted(P_GRID, 3.0) + 1
    for (bc, (name, *_)), (curves, meta) in zip(cells, res):
        store[f"{bc}_{name}_curves"] = curves
        store[f"{bc}_{name}_meta"] = meta
        med = np.nanmedian(curves, axis=0)
        for ti, T in enumerate(T_VALUES):
            ps, Ps, ds = delta_at_s(med[ti])
            flat = np.nanmax(med[ti, :i3]) / np.nanmin(med[ti, :i3])
            pre = (f"| {bc[:3]} {name} | {meta[0]:.0f} | {int(meta[1])} | {meta[2]:.4f} "
                   f"| {meta[3]:.2f} " if ti == 0 else "| | | | | ")
            rep.append(pre + f"| {T} | {ps:.1f} | {100*ds:.2f}% | {flat:.3f} |")
    out = out_path(args.out or "box3d_flatness.npz")
    np.savez(out, p_grid=P_GRID, T_values=T_VALUES, s_hat=S_HAT, **store)
    print("\n".join(rep))
    print(f"wrote {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
