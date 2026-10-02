#!/usr/bin/env python
"""Appendix G.4 cohort flattening (T27 generic flatness column; T42 "Matched-count
cohort" 101, "Flatness ratio with no discarded modes" generic value 2.341, and the
K_total = 300 / IQR comparisons).

Protocol of matched_ktotal_rates.py on a fixed cohort: every in-scope room with
K_total >= 300 (101 rooms), diffuse-field prior over the first K_total modes, K = 50
retained, M = 8 (legacy mics), T = 100, 8 realizations per (room, K_total), oracle sweep
over the 61-p x 12-alpha grid, median curve over realizations, flatness = max/min P over
p in [0, 3].

Notes:
  * cohort selection uses K_total >= 300 in the legacy modal data; requiring it also in
    the full eigensolve (data/rooms) selects the identical 101 rooms, so data/rooms is
    not needed;
  * RNG per (room, K_total) seeded with zlib.crc32 (_common.stable_seed; --seed S appends
    S to the key). The printed values are medians over seeds 0-9 (run_multi_seed.sh);
  * pool size from --workers.

Output: data/experiments/appendix/results_cohort_flatness.npz
Usage:  python scripts/appendix/F_G/cohort_flatness.py [--workers 16] [--seed S --out NAME]
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

S_HAT, K_RET, M_USE = 1.13, 50, 8
GAMMA, C_SOUND = 5.1, 343.0
DT_SIM, DT_SNAP, N_SNAPS = 50e-6, 5e-3, 100
T_RAW, N_REAL = 100, 8
K_TOTALS = [50, 75, 100, 150, 200, 300]
P_GRID = np.round(np.arange(0.0, 6.01, 0.1), 10)
LAMBDA_GRID = np.array([1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0,
                        100.0, 1e3, 1e4, 1e6, 1e8, 1e10])
M03 = (P_GRID >= 0) & (P_GRID <= 3.0000001)
SEED = None             # set by --seed


def select_cohort():
    """scene_00800..00999 rooms whose legacy K_total is >= 300."""
    cohort = []
    for i in range(800, 1000):
        p = MODAL / f"scene_{i:05d}" / "eigenpairs.npz"
        if p.exists() and len(np.load(p, allow_pickle=True)["eigenvalues"]) >= 300:
            cohort.append(f"scene_{i:05d}")
    return cohort


def simulate(ev, Phi, rng):
    K_total = len(ev)
    sd = ev ** (-S_HAT / 2)
    c0 = rng.standard_normal(K_total) * sd
    b0 = rng.standard_normal(K_total) * sd
    om = C_SOUND * np.sqrt(ev)
    omd = np.sqrt(np.maximum(om ** 2 - (GAMMA / 2) ** 2, 0.0))
    sps = int(round(DT_SNAP / DT_SIM))
    T_sig = N_SNAPS * sps
    t = np.arange(T_sig) * DT_SIM
    dec = np.exp(-GAMMA * t / 2)
    a_t = (c0[:, None] * np.cos(omd[:, None] * t[None, :])
           + b0[:, None] * np.sin(omd[:, None] * t[None, :])) * dec[None, :]
    return {"K": min(K_RET, K_total), "eigenvalues": ev[:K_RET],
            "Phi": Phi[:, :K_RET], "a": a_t[:K_RET, np.arange(N_SNAPS) * sps],
            "y_click": Phi @ a_t, "gamma_room": GAMMA, "c": C_SOUND,
            "dt_sim": DT_SIM, "dt_snap": DT_SNAP, "steps_per_snap": sps,
            "n_snaps": N_SNAPS}


def worker(sid):
    d = MODAL / sid
    ev_all = np.asarray(np.load(d / "eigenpairs.npz", allow_pickle=True)["eigenvalues"],
                        float)
    Phi_all = np.load(d / "measurement_matrix.npy")[:M_USE]
    out = {}
    for K in K_TOTALS:
        if len(ev_all) < K:
            continue
        rng = np.random.default_rng(stable_seed(with_seed((sid, K), SEED)))
        curves = []
        for _ in range(N_REAL):
            room = simulate(ev_all[:K], Phi_all[:, :K].astype(float), rng)
            P = sweep_p_for_room(room, np.array([T_RAW]), P_GRID, LAMBDA_GRID,
                                 M_values=[M_USE], mic_subsets_4=None,
                                 is_wave=True, gamma=GAMMA, c=C_SOUND)
            curves.append(P[0, 0, :])
        med = np.nanmedian(np.array(curves), axis=0)[M03]
        out[K] = float(np.nanmax(med) / np.nanmin(med))
    return sid, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=None,
                    help="extra seed appended to every RNG key (default: the shipped run)")
    ap.add_argument("--out", default=None, help="output name under data/experiments/appendix/")
    args = ap.parse_args()
    global SEED
    SEED = args.seed
    check_paper_dataset()
    t0 = time.time()
    cohort = select_cohort()
    print(f"cohort: {len(cohort)} rooms (K_total >= 300)", flush=True)
    with Pool(n_workers(args.workers)) as pool:
        res = pool.map(worker, cohort, chunksize=1)
    print(f"sweep done ({time.time()-t0:.0f}s)", flush=True)

    store = {}
    for K in K_TOTALS:
        fl = np.array([o[K] for _, o in res if K in o])
        store[f"flat_K{K}"] = fl
        print(f"  K={K:3d}: cohort flatness median {np.median(fl):.3f} "
              f"[{np.percentile(fl,25):.3f}, {np.percentile(fl,75):.3f}] n={len(fl)}",
              flush=True)
    out = out_path(args.out or "results_cohort_flatness.npz")
    np.savez(out, cohort=np.array(cohort), K_totals=np.array(K_TOTALS), **store)
    print(f"wrote {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
