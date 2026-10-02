#!/usr/bin/env python
"""Appendix G.4 matched-K_total flattening (T27 rectangle flatness column; T42
"Flatness ratio with no discarded modes", rectangle value 2.371).

One protocol for both room classes: diffuse-field prior (c, b ~ N(0, lambda^-|s|)) over
exactly K_total modes, K = 50 retained, M = 8, T = 100, oracle-alpha p sweep, so the
only difference is the eigenstructure: 6 generic convex rooms (legacy FEM eigenpairs and
mics) vs the 4 analytic Dirichlet rectangles (sensors uniform with a 0.3 m margin).
Per (room, K_total in {50, 75, 100, 150, 200, 300}) the median curve over 8
realizations is stored; flatness = max/min P over p in [0, 3].

The per-panel RNG is seeded with zlib.crc32(label) (_common.stable_seed). The printed
values came from Python's randomized hash(label), which cannot be regenerated; the seed
sets the rectangle sensor positions and all draws. The 10 panels run in a process pool.

Output: data/experiments/appendix/matched_ktotal.npz
Usage:  python scripts/appendix/F_G/matched_ktotal_rates.py [--workers 10] [--seed S --out NAME]
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import time
from multiprocessing import Pool

import numpy as np

from _common import out_path, n_workers, stable_seed, with_seed
from src.utils.paths import MODAL, check_paper_dataset
from src.analysis.landscape import sweep_p_for_room

S_HAT = 1.13
K_RET = 50
M_USE = 8
GAMMA = 5.1
C_SOUND = 343.0
DT_SIM = 50e-6
DT_SNAP = 5e-3
N_SNAPS = 100
T_RAW = 100
N_REAL = 8
K_TOTALS = [50, 75, 100, 150, 200, 300]
P_GRID = np.round(np.arange(0.0, 6.01, 0.1), 10)
LAMBDA_GRID = np.array([1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0,
                        100.0, 1e3, 1e4, 1e6, 1e8, 1e10])
RECTS = [("3x6_rect", 3.0, 6.0), ("2x8_long", 2.0, 8.0),
         ("4x4_square", 4.0, 4.0), ("3x5_rect", 3.0, 5.0)]
N_GENERIC = 6
SEED = None             # set by --seed


def rect_modes(Lx, Ly, K_total, rng):
    m_max = int(np.ceil(np.sqrt(K_total * Lx / Ly))) + 12
    n_max = int(np.ceil(np.sqrt(K_total * Ly / Lx))) + 12
    cand = [(np.pi ** 2 * (m ** 2 / Lx ** 2 + n ** 2 / Ly ** 2), m, n)
            for m in range(1, m_max + 1) for n in range(1, n_max + 1)]
    cand.sort(key=lambda x: x[0])
    cand = cand[:K_total]
    ev = np.array([c[0] for c in cand])
    mics = np.column_stack([rng.uniform(0.3, Lx - 0.3, M_USE),
                            rng.uniform(0.3, Ly - 0.3, M_USE)])
    norm = 2.0 / np.sqrt(Lx * Ly)
    Phi = np.stack([norm * np.sin(m * np.pi * mics[:, 0] / Lx)
                    * np.sin(n * np.pi * mics[:, 1] / Ly)
                    for _, m, n in cand], axis=1)
    return ev, Phi


def generic_rooms():
    """Pick N_GENERIC legacy rooms spread over the K_total ranking (K_total >= 320)."""
    cands = []
    for i in range(800, 1000):
        sid = f"scene_{i:05d}"
        p = MODAL / sid / "eigenpairs.npz"
        if not p.exists():
            continue
        ev = np.load(p, allow_pickle=True)["eigenvalues"]
        if len(ev) >= max(K_TOTALS) + 20:
            cands.append((len(ev), sid))
    cands.sort()
    step = max(1, len(cands) // N_GENERIC)
    return [sid for _, sid in cands[::step][:N_GENERIC]]


def simulate(ev, Phi, rng):
    """Diffuse-field realization -> room dict for sweep_p_for_room."""
    K_total = len(ev)
    sd = ev ** (-S_HAT / 2)
    c0 = rng.standard_normal(K_total) * sd
    b0 = rng.standard_normal(K_total) * sd
    omega = C_SOUND * np.sqrt(ev)
    omega_d = np.sqrt(np.maximum(omega ** 2 - (GAMMA / 2) ** 2, 0.0))
    sps = int(round(DT_SNAP / DT_SIM))
    t = np.arange(N_SNAPS * sps) * DT_SIM
    decay = np.exp(-GAMMA * t / 2)
    a_t = (c0[:, None] * np.cos(omega_d[:, None] * t[None, :])
           + b0[:, None] * np.sin(omega_d[:, None] * t[None, :])) * decay[None, :]
    K_use = min(K_RET, K_total)
    return {
        "K": K_use, "eigenvalues": ev[:K_use], "Phi": Phi[:, :K_use],
        "a": a_t[:K_use, np.arange(N_SNAPS) * sps],
        "y_click": Phi @ a_t,
        "gamma_room": GAMMA, "c": C_SOUND, "dt_sim": DT_SIM,
        "dt_snap": DT_SNAP, "steps_per_snap": sps, "n_snaps": N_SNAPS,
    }


def room_stats(ev_all, Phi_all, K_total, rng):
    ev, Phi = ev_all[:K_total], Phi_all[:, :K_total]
    curves = []
    for _ in range(N_REAL):
        room = simulate(ev, Phi, rng)
        P = sweep_p_for_room(room, np.array([T_RAW]), P_GRID, LAMBDA_GRID,
                             M_values=[M_USE], mic_subsets_4=None,
                             is_wave=True, gamma=GAMMA, c=C_SOUND)
        curves.append(P[0, 0, :])
    med = np.nanmedian(np.array(curves), axis=0)
    i3 = np.searchsorted(P_GRID, 3.0) + 1
    ratio = np.nanmax(med[:i3]) / np.nanmin(med[:i3])
    j = int(np.nanargmin(med))
    ds = (np.interp(S_HAT, P_GRID, med) - med[j]) / med[j]
    return ratio, ds, med


def run_panel(panel):
    label, kind, spec = panel
    rng = np.random.default_rng(stable_seed(with_seed(label, SEED)))
    if kind == "generic":
        d = MODAL / spec
        ev_all = np.load(d / "eigenpairs.npz", allow_pickle=True)["eigenvalues"]
        Phi_all = np.load(d / "measurement_matrix.npy")[:M_USE]
    else:
        n, Lx, Ly = spec
        ev_all, Phi_all = rect_modes(Lx, Ly, max(K_TOTALS) + 20, rng)
    ratios, deltas, meds = [], [], {}
    for Kt in K_TOTALS:
        ratio, ds, med = room_stats(ev_all, Phi_all, Kt, rng)
        ratios.append(ratio)
        deltas.append(ds)
        meds[f"{label}_K{Kt}_med"] = med
    return label, ratios, deltas, meds


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--seed", type=int, default=None,
                    help="extra seed appended to every RNG key (default: the shipped run)")
    ap.add_argument("--out", default=None, help="output name under data/experiments/appendix/")
    args = ap.parse_args()
    global SEED
    SEED = args.seed
    check_paper_dataset()
    t0 = time.time()
    gens = generic_rooms()
    panels = ([(f"generic:{sid}", "generic", sid) for sid in gens]
              + [(f"rect:{n}", "rect", (n, Lx, Ly)) for n, Lx, Ly in RECTS])
    with Pool(n_workers(args.workers)) as pool:
        results = pool.map(run_panel, panels, chunksize=1)
    store, rows = {}, {}
    for label, ratios, deltas, meds in results:
        rows[label] = (ratios, deltas)
        store.update(meds)

    rep = [f"# Matched-K_total flattening (diffuse prior |s|={S_HAT}, T={T_RAW}, "
           f"{N_REAL} realizations, median curves)\n",
           "| room | " + " | ".join(f"K={k}" for k in K_TOTALS) + " |",
           "|---|" + "---|" * len(K_TOTALS)]
    for label, (ratios, _) in rows.items():
        rep.append(f"| {label} | " + " | ".join(f"{r:.3f}" for r in ratios) + " |")
    gen_med = np.median([rows[k][0] for k in rows if k.startswith("generic")], axis=0)
    rect_med = np.median([rows[k][0] for k in rows if k.startswith("rect")], axis=0)
    rep.append("| **generic median** | " + " | ".join(f"{r:.3f}" for r in gen_med) + " |")
    rep.append("| **rect median** | " + " | ".join(f"{r:.3f}" for r in rect_med) + " |")

    out = out_path(args.out or "matched_ktotal.npz")
    np.savez(out, p_grid=P_GRID, K_totals=np.array(K_TOTALS), **store)
    print("\n".join(rep))
    print(f"wrote {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
