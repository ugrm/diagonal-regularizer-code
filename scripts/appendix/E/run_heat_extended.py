#!/usr/bin/env python
"""Heat p-sweep on the extended grid p in [0, 6] (Appendix E, fig03 panels c-e, T40).

Same protocol and seeds as the shipped results: for each of the 197 validation rooms,
synthetic heat signals (src.physics.temporal), ridge with Gamma_k = lambda_k^p for 61
values of p and 12 values of the ridge weight,
M = 8 (all microphones) and M = 4 (the 20 canonical subsets of data/mic_subsets.npz),
10 window lengths T. P_oracle is the best ridge weight of the subset-averaged P.

Output: data/experiments/appendix/exp2_p_sweep_heat_extended.npz (same keys as the
shipped data/experiments/exp2_p_sweep_heat_extended.npz).

Usage (from the repo root):
    python scripts/appendix/E/run_heat_extended.py               # 197 rooms, 16 workers
    python scripts/appendix/E/run_heat_extended.py --rooms 5     # quick test
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heat_common import (  # noqa: E402
    DATA, EXP, MODAL, build_heat_temporal_matrix, check_paper_dataset, compute_P,
    generate_heat_signals, heat_seed, load_heat_room, val_rooms, valid_snaps, window_matrix,
)
import numpy as np  # noqa: E402

P_VALUES = np.round(np.arange(0, 6.05, 0.1), 1)          # 61 values
T_GRID = np.array([1, 5, 10, 20, 50, 100, 200, 500, 1000, 2100])
M_VALUES = [4, 8]
LAMBDA_GRID = np.array([1e-6, 1e-4, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e6, 1e8, 1e10])
N_P, N_T, N_M, N_LAMBDA = len(P_VALUES), len(T_GRID), len(M_VALUES), len(LAMBDA_GRID)


def evaluate_ridge_one_trial(A_full_8, Y_flat_8, targets, mic_indices, T_raw, gamma_diags):
    row_idx = np.concatenate([np.arange(m * T_raw, (m + 1) * T_raw) for m in mic_indices])
    A = A_full_8[row_idx]
    Y = Y_flat_8[:, row_idx]

    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s2 = s**2
    ATA = (Vt.T * s2[None, :]) @ Vt
    UTY = U.T @ Y.T
    ATY = (Vt.T * s[None, :]) @ UTY

    P_ridge = np.full((N_P, N_LAMBDA), np.nan)
    for p_idx, gamma_diag in enumerate(gamma_diags):
        for l_idx, lam in enumerate(LAMBDA_GRID):
            if P_VALUES[p_idx] == 0.0:
                d = s / (s2 + lam)
                X_r = (Vt.T * d[None, :]) @ UTY
            else:
                try:
                    X_r = np.linalg.solve(ATA + lam * np.diag(gamma_diag), ATY)
                except np.linalg.LinAlgError:
                    continue
            P_ridge[p_idx, l_idx] = compute_P(X_r.T, targets)
    return P_ridge


def evaluate_room(args):
    sid, subsets_4, modal_root = args
    t0 = time.time()
    room = load_heat_room(sid, modal_root)
    ev, dt, sps, Phi = room["eigenvalues"], room["dt_sim"], room["steps_per_snap"], room["Phi"]

    Y_full, a_tgt, _ = generate_heat_signals(room, heat_seed(sid))
    gamma_diags = [np.power(ev, p) for p in P_VALUES]

    P_oracle = np.full((N_T, N_M, N_P), np.nan)
    P_full = np.full((N_T, N_M, N_P, N_LAMBDA), np.nan)
    best_alpha_idx = np.full((N_T, N_M, N_P), -1, dtype=int)

    for t_idx, T_raw in enumerate(int(t) for t in T_GRID):
        valid = valid_snaps(room["n_snaps"], sps, Y_full.shape[1], T_raw)
        if len(valid) < 5:
            continue
        targets = a_tgt[:, valid].T
        A_full = build_heat_temporal_matrix(Phi, ev, dt, T_raw)
        Y_flat_8 = window_matrix(Y_full, valid, sps, T_raw)

        for m_idx, M_val in enumerate(M_VALUES):
            subsets = [tuple(range(8))] if M_val == 8 else [tuple(int(x) for x in r) for r in subsets_4]
            trials = [evaluate_ridge_one_trial(A_full, Y_flat_8, targets, mic, T_raw, gamma_diags)
                      for mic in subsets]
            avg_P = np.nanmean(trials, axis=0)                     # (N_P, N_LAMBDA)
            for p_idx in range(N_P):
                if np.isfinite(avg_P[p_idx]).any():
                    best_l = np.nanargmin(avg_P[p_idx])
                    P_oracle[t_idx, m_idx, p_idx] = avg_P[p_idx, best_l]
                    best_alpha_idx[t_idx, m_idx, p_idx] = best_l
                    P_full[t_idx, m_idx, p_idx, :] = avg_P[p_idx, :]
    return sid, P_oracle, P_full, best_alpha_idx, time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rooms", type=int, default=None, help="first N validation rooms only")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--modal-root", default=str(MODAL))
    ap.add_argument("--out", default=str(EXP / "appendix" / "exp2_p_sweep_heat_extended.npz"))
    args = ap.parse_args()

    check_paper_dataset(args.modal_root)
    sub = np.load(DATA / "mic_subsets.npz", allow_pickle=True)
    sub_rooms = list(sub["val_rooms"])
    rooms = val_rooms()[:args.rooms] if args.rooms else val_rooms()
    jobs = [(sid, sub["subsets_M4"][sub_rooms.index(sid)] if sid in sub_rooms else np.arange(4)[None, :],
             args.modal_root) for sid in rooms]

    print(f"Heat p-sweep: {len(rooms)} rooms, p in [{P_VALUES[0]}, {P_VALUES[-1]}] ({N_P}), "
          f"T={list(T_GRID)}, M={M_VALUES}, {args.workers} workers", flush=True)
    t0 = time.time()
    res = {}
    with Pool(args.workers) as pool:
        for i, (sid, Po, Pf, ba, dt_room) in enumerate(pool.imap_unordered(evaluate_room, jobs)):
            res[sid] = (Po, Pf, ba)
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1}/{len(rooms)}] {sid}: best_p(T=1,M=8)={P_VALUES[np.nanargmin(Po[0, 1])]:.1f} "
                      f"({dt_room:.1f}s, elapsed {(time.time()-t0)/60:.1f} min)", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out,
             P_oracle=np.stack([res[s][0] for s in rooms]),
             P_full=np.stack([res[s][1] for s in rooms]),
             best_alpha_idx=np.stack([res[s][2] for s in rooms]),
             p_values=P_VALUES, T_values=T_GRID, M_values=np.array(M_VALUES),
             lambda_grid=LAMBDA_GRID, room_ids=np.array(rooms), pde="heat")
    print(f"Saved {args.out}  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
