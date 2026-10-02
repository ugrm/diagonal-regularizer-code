#!/usr/bin/env python
"""Appendix G.4: generic D/(2H) on the 101-room matched-count cohort (T27 generic
D/(2H) column 1.92 / 1.31 / 1.82 / 1.89; T42 "generic IQR upper bound" 2.38).

Applies d_vs_ktotal_sweep.generic_room to every room of the cohort of cohort_flatness.py
(K_total >= 300) and stores the per-room [H, D, S, C, E_x[V]] at each K_total > 50.
Needs the full eigenfunctions (DR_ROOMS, output of scripts/01_generate_rooms.py).

Output: data/experiments/appendix/cohort_d2h.npz
        keys: cohort (101,), K_totals, arena_dev (101,), K{K} (101, 5) for K in
        75/100/150/200/300 with columns [H, D, S, C, E_x[V]]
Usage:  DR_ROOMS=/path/to/rooms python scripts/appendix/F_G/cohort_d2h.py [--workers 8]
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import time
from multiprocessing import Pool

import numpy as np

from _common import out_path, n_workers
from cohort_flatness import select_cohort
from d_vs_ktotal_sweep import generic_room, require_rooms, K_RET, K_TOTALS
from src.utils.paths import check_paper_dataset

KS = [K for K in K_TOTALS if K > K_RET]


def _task(sid):
    rows, _, dev = generic_room(sid)
    return np.array([rows[K] for K in KS]), dev


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    check_paper_dataset()
    require_rooms()
    t0 = time.time()
    cohort = select_cohort()
    with Pool(n_workers(args.workers)) as pool:
        res = pool.map(_task, cohort, chunksize=1)
    arr = np.array([r[0] for r in res])            # (n_rooms, len(KS), 5)
    dev = np.array([r[1] for r in res])
    print(f"{len(cohort)} rooms, max eigenvalue deviation vs legacy {dev.max():.1e}")
    print("| K_total | H median | D/(2H) median [IQR] |")
    print("|---|---|---|")
    for j, K in enumerate(KS):
        H, D = arr[:, j, 0], arr[:, j, 1]
        q = D / (2 * H)
        print(f"| {K} | {np.median(H):.4f} | {np.median(q):.2f} "
              f"[{np.percentile(q, 25):.2f}, {np.percentile(q, 75):.2f}] |")
    out = out_path("cohort_d2h.npz")
    np.savez(out, cohort=np.array(cohort), K_totals=np.array(KS), arena_dev=dev,
             **{f"K{K}": arr[:, j, :] for j, K in enumerate(KS)})
    print(f"wrote {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
