#!/usr/bin/env python
"""Appendix F single-draw probe (T41 row "Single-draw delta at T = 100 under the
Gaussian prior, range across seeds", 0.7-12.6%).

For each of the 20 control rooms (scene_00800..00819), draw ONE (c, beta) realization
per prior for the K = min(50, K_total) generated modes (no truncation noise), evolve
damped sinusoids over the room's snapshot grid, and run the window-based oracle-alpha
p sweep at T in {1, 100}. Repeat with 5 probe seeds and keep the median-over-rooms
curve per seed: the spread of delta(|s|) across seeds is the single-realization
scatter quoted in T41.

Notes:
  * the room list is fixed (scene_00800..00819);
  * no printed value uses this output; multi_seed_summary.py reports the single-draw
    spread over 10 seeds for reference;
  * draws are seeded with zlib.crc32 (_common.stable_seed, with_seed).

Output: data/experiments/appendix/single_draw_seed_spread.npz
Usage:  python scripts/appendix/F_G/single_draw_seed_spread.py [--workers 16] [--n-seeds N --out NAME]
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import time
from multiprocessing import Pool

import numpy as np

from _common import out_path, n_workers, stable_seed
from misspec_priors_m3 import load_room, S_HAT, P_GRID, LAMBDA_GRID, ROOM_IDS
from src.utils.paths import MODAL, check_paper_dataset
from src.physics.temporal import build_wave_temporal_matrix
from src.estimation.ridge import ridge_sweep_svd, build_gamma_diag

T_EVAL = [1, 100]
SKIP_FIRST, SKIP_LAST = 5, 2
PROBE_SEEDS = [0, 1, 2, 3, 4]
PRIORS = ["gaussian", "heavy_tail", "correlated"]


def room_extras(sid):
    tr = np.load(MODAL / sid / "modal_trajectories.npz", allow_pickle=True)
    a = tr["a"]
    n_snaps = a.shape[1]
    dt_snap = float(tr["dt_snap"])
    dt_sim = float(tr["dt_sim"])
    return n_snaps, int(round(dt_snap / dt_sim))


def single_draw_curve(r, T_raw, prior, rng, n_snaps, sps):
    K = r["K_eff"]
    ev = r["ev"]
    sd = ev ** (-S_HAT / 2)
    if prior == "gaussian":
        c0 = rng.standard_normal(K) * sd
        b0 = rng.standard_normal(K) * sd
    elif prior == "heavy_tail":
        c0 = rng.standard_t(3, K) * sd / np.sqrt(3.0)
        b0 = rng.standard_t(3, K) * sd / np.sqrt(3.0)
    else:
        v = ev ** (-S_HAT)
        Sig = np.diag(v)
        s_ = np.sqrt(v)
        for j in range(K - 1):
            Sig[j, j + 1] = Sig[j + 1, j] = 0.3 * s_[j] * s_[j + 1]
        L = np.linalg.cholesky(Sig + 1e-14 * np.eye(K))
        c0 = L @ rng.standard_normal(K)
        b0 = L @ rng.standard_normal(K)

    omega = r["c"] * np.sqrt(ev)
    omega_d = np.sqrt(np.maximum(omega ** 2 - (r["gamma"] / 2) ** 2, 0.0))
    T_sig = n_snaps * sps
    t = np.arange(T_sig) * r["dt"]
    decay = np.exp(-r["gamma"] * t / 2)
    a_t = (c0[:, None] * np.cos(omega_d[:, None] * t[None, :])
           + b0[:, None] * np.sin(omega_d[:, None] * t[None, :])) * decay[None, :]
    y = r["Phi"] @ a_t                                   # (M, T_sig)

    valid = [si for si in range(SKIP_FIRST, n_snaps - SKIP_LAST)
             if si * sps + 1 - T_raw >= 0 and si * sps + 1 <= T_sig]
    targets = a_t[:, [si * sps for si in valid]].T       # (N, K)
    A = build_wave_temporal_matrix(r["Phi"], ev, r["dt"], T_raw, r["gamma"], r["c"])
    Y = np.stack([y[:, si * sps + 1 - T_raw: si * sps + 1].reshape(-1)
                  for si in valid])

    curve = np.full(len(P_GRID), np.nan)
    for pi, p in enumerate(P_GRID):
        g = build_gamma_diag(ev, p, True)
        P_vals, _ = ridge_sweep_svd(A, Y, targets, g, LAMBDA_GRID, True)
        curve[pi] = np.nanmin(P_vals)
    return curve


_ROOMS, _EXTRAS = [], {}


def _task(args):
    T_raw, prior, seed = args
    curves = []
    for r in _ROOMS:
        n_snaps, sps = _EXTRAS[r["sid"]]
        rng = np.random.default_rng(stable_seed((prior, r["sid"], seed)))
        curves.append(single_draw_curve(r, T_raw, prior, rng, n_snaps, sps))
    return np.nanmedian(np.array(curves), axis=0)


def delta_at_s(mc):
    j = int(np.nanargmin(mc))
    return P_GRID[j], (np.interp(S_HAT, P_GRID, mc) - mc[j]) / mc[j]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n-seeds", type=int, default=len(PROBE_SEEDS),
                    help="probe seeds 0..N-1 (default 5, the shipped run)")
    ap.add_argument("--out", default=None, help="output name under data/experiments/appendix/")
    args = ap.parse_args()
    PROBE_SEEDS[:] = list(range(args.n_seeds))
    check_paper_dataset()
    t0 = time.time()
    _ROOMS[:] = [load_room(sid) for sid in ROOM_IDS]
    _EXTRAS.update({sid: room_extras(sid) for sid in ROOM_IDS})
    keys = [(T, pr, s) for T in T_EVAL for pr in PRIORS for s in PROBE_SEEDS]
    with Pool(n_workers(args.workers)) as pool:
        res = dict(zip(keys, pool.map(_task, keys, chunksize=1)))

    rep = ["# Single-draw probe: delta(|s|) of the median-over-rooms curve per seed\n"]
    store = {}
    for T_raw in T_EVAL:
        for prior in PRIORS:
            med_curves = np.array([res[(T_raw, prior, s)] for s in PROBE_SEEDS])
            store[f"{prior}_T{T_raw}_probe"] = med_curves      # (5 seeds, 61)
            st = [delta_at_s(mc) for mc in med_curves]
            ds = [100 * d for _, d in st]
            rep.append(f"- T={T_raw} {prior}: probe p* " +
                       ", ".join(f"{p:.1f}" for p, _ in st) +
                       f"; d(|s|) per seed " + ", ".join(f"{d:.2f}%" for d in ds) +
                       f" -> range [{min(ds):.1f}%, {max(ds):.1f}%]")
    out = out_path(args.out or "single_draw_seed_spread.npz")
    np.savez(out, p_grid=P_GRID, **store)
    print("\n".join(rep))
    print(f"wrote {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
