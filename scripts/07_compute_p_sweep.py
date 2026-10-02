#!/usr/bin/env python
"""
Step 5: P(p) landscape sweep.

For each validation room, T, M, p: compute oracle-alpha P_modal.
Produces the flat landscape result and p*(T) trajectory.

Output: data/experiments/p_sweep/p_sweep_K{k}_M{m}.npz

Usage:
    python scripts/07_compute_p_sweep.py --config configs/default.yaml
    python scripts/07_compute_p_sweep.py --config configs/default.yaml --rooms 10
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config
from src.utils.paths import check_paper_dataset
from src.utils.io import load_room_auto, get_val_rooms, get_split_params

# Data paths — resolved from config in main()
MODAL_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "modal")
LEGACY_MODAL = None
LEGACY_FDTD = None
MIC_SUBSETS_PATH = None


_CTX = {}


def _init_worker(ctx):
    _CTX.update(ctx)


def _sweep_one(job):
    """P_oracle(T, M in {4, 8}, p) for one validation room."""
    from src.analysis.landscape import sweep_p_for_room
    i, sid, subsets_4 = job
    cache = os.path.join(_CTX["cache_dir"], f"{sid}.npy")
    if os.path.exists(cache):
        return i, np.load(cache)
    room = load_room_auto(sid, modal_root=_CTX["modal_root"], legacy_modal_root=_CTX["legacy_modal"],
                          legacy_fdtd_root=_CTX["legacy_fdtd"], k_trunc=50)
    if room["y_click"] is None:
        raise FileNotFoundError(f"{sid}: no microphone signals (y_mics_eta1.npy); "
                                "run `make download-data` or `make fdtd-signals`")
    P = sweep_p_for_room(room, _CTX["T_values"], _CTX["p_grid"], _CTX["lambda_grid"],
                         M_values=[4, 8], mic_subsets_4=subsets_4, is_wave=True,
                         gamma=room["gamma_room"], c=room["c"])
    np.save(cache + ".tmp.npy", P); os.replace(cache + ".tmp.npy", cache)   # resumable per-room results
    return i, P


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--rooms", type=int, default=None)
    parser.add_argument("--pde", default="acoustic", choices=["acoustic", "heat", "both"])
    parser.add_argument("--modal-root", default=None, help="Override the modal dataset root")
    parser.add_argument("--exp-dir", default=None, help="Override experiments dir")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel worker processes (set OMP_NUM_THREADS=1 when > 1)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    check_paper_dataset(cfg["paths"]["modal_root"])

    global LEGACY_MODAL, LEGACY_FDTD, MIC_SUBSETS_PATH
    LEGACY_MODAL = cfg["paths"].get("modal_root")
    LEGACY_FDTD = cfg["paths"].get("fdtd_dir")
    MIC_SUBSETS_PATH = cfg["paths"].get("mic_subsets", "data/mic_subsets.npz")

    modal_root = args.modal_root or cfg["paths"].get("modal_root", MODAL_ROOT)
    val_start, val_end = get_split_params(cfg)
    val_rooms = get_val_rooms(modal_root, cfg["blacklist"]["scene_ids"],
                              val_start=val_start, val_end=val_end)
    if args.rooms:
        val_rooms = val_rooms[:args.rooms]

    ev_cfg = cfg["evaluation"]
    p_grid = np.array(ev_cfg["p_grid"])
    T_values = np.array(ev_cfg["t_values"])
    lambda_grid = np.array(ev_cfg["lambda_grid"])
    M_values = ev_cfg["m_values"]

    exp_base = args.exp_dir or cfg["paths"]["experiments_dir"]
    out_dir = os.path.join(exp_base, "p_sweep")
    os.makedirs(out_dir, exist_ok=True)

    # Load canonical mic subsets
    mic_subsets = None
    if os.path.exists(MIC_SUBSETS_PATH):
        d = np.load(MIC_SUBSETS_PATH, allow_pickle=True)
        mic_subsets = {4: d["subsets_M4"]}

    from src.analysis.landscape import sweep_p_for_room

    N = len(val_rooms)
    n_T = len(T_values)
    n_M = len(M_values)
    n_p = len(p_grid)

    final_path = os.path.join(out_dir, "p_sweep_K50_M8.npz")
    if os.path.exists(final_path):
        print(f"P-sweep: {final_path} already exists, skipping.")
        print(f"  Delete to re-run: rm {final_path}")
        return

    P_oracle_all = np.full((N, n_T, n_M, n_p), np.nan)
    checkpoint_path = os.path.join(out_dir, "p_sweep_K50_M8.partial.npz")
    checkpoint_every = 20  # save partial results every 20 rooms

    print(f"P-sweep: {N} rooms x {n_T} T x {n_M} M x {n_p} p")
    t0 = time.time()

    jobs = [(i, sid, None if mic_subsets is None or i >= len(mic_subsets[4]) else mic_subsets[4][i])
            for i, sid in enumerate(val_rooms)]
    ctx = dict(modal_root=modal_root, legacy_modal=LEGACY_MODAL, legacy_fdtd=LEGACY_FDTD,
               T_values=T_values, p_grid=p_grid, lambda_grid=lambda_grid,
               cache_dir=os.path.join(out_dir, "per_room"))
    os.makedirs(ctx["cache_dir"], exist_ok=True)
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers, initializer=_init_worker, initargs=(ctx,)) as pool:
            for n, (i, P_oracle) in enumerate(pool.imap_unordered(_sweep_one, jobs)):
                P_oracle_all[i, :, :2, :] = P_oracle
                if (n + 1) % 10 == 0 or n == 0:
                    print(f"  [{n+1}/{N}] ({time.time() - t0:.0f}s)", flush=True)
    else:
        _init_worker(ctx)
        for n, job in enumerate(jobs):
            i, P_oracle = _sweep_one(job)
            P_oracle_all[i, :, :2, :] = P_oracle
            if (n + 1) % 10 == 0 or n == 0:
                elapsed = time.time() - t0
                print(f"  [{n+1}/{N}] ({elapsed:.0f}s, ETA {elapsed / (n + 1) * (N - n - 1) / 60:.1f}m)", flush=True)
            if (n + 1) % checkpoint_every == 0:
                tmp = checkpoint_path.replace(".npz", ".tmp.npz")
                np.savez(tmp, P_oracle=P_oracle_all, p_values=p_grid, T_values=T_values,
                         M_values=np.array(M_values), lambda_grid=lambda_grid,
                         room_ids=np.array(val_rooms), n_complete=n + 1)
                os.replace(tmp, checkpoint_path)

    # Clean up checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    # Save — key name p_values matches src/visualization/fig_flat_landscape.py
    out_path = os.path.join(out_dir, "p_sweep_K50_M8.npz")
    np.savez(
        out_path,
        P_oracle=P_oracle_all,
        p_values=p_grid,
        T_values=T_values,
        M_values=np.array(M_values),
        lambda_grid=lambda_grid,
        room_ids=np.array(val_rooms),
    )
    print(f"\nSaved: {out_path}")

    # alias read by resolve_data_path
    legacy_link = os.path.join(out_dir, "exp2_p_sweep.npz")
    if os.path.islink(legacy_link) or os.path.exists(legacy_link):
        os.remove(legacy_link)
    os.symlink(os.path.basename(out_path), legacy_link)
    print(f"  Symlink: {legacy_link} -> {os.path.basename(out_path)}")

    # Quick summary: p* at each T
    print("\nMedian P(p) curve — best p per T:")
    for t_idx, T in enumerate(T_values):
        med = np.nanmedian(P_oracle_all[:, t_idx, 1, :], axis=0)  # M=8 is index 1
        if np.all(np.isnan(med)):
            continue
        best_p = p_grid[np.nanargmin(med)]
        P_best = np.nanmin(med)
        print(f"  T={T:5d}: p*={best_p:.1f}, P*={P_best:.4f}")


if __name__ == "__main__":
    main()
