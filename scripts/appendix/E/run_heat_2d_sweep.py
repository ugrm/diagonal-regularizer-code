#!/usr/bin/env python
"""Two-parameter sweep Gamma_k = lambda_k^p exp(c lambda_k) for heat and acoustics (fig13, T21).

197 validation rooms, p in [0, 4] step 0.2, c in [0, 0.5] step 0.02, 12 ridge weights,
T in {1, 10, 100, 500}, M = 8.
P is the best-ridge-weight modal error of each (p, c) cell.

As in the shipped results, every room is zero-padded to K = 50 modes (pad_room_to_K) before
the heat signals are drawn. For rooms with fewer than 50 modes this changes the random
stream and adds padded target modes with zero variance and a constant nonzero value,
which no estimator can recover, so their heat P can exceed 1 by orders of magnitude.

  heat     : analytic signals (src.physics.temporal), no microphone data needed.
  acoustic : FDTD microphone signals <fdtd-root>/scene_XXXXX/y_mics_eta1.npy, read by
             src.data.dataset.load_room_modal(..., layout="legacy", fdtd_root=...).

Output: data/experiments/appendix/heat_2d_sweep_results.npz (same keys as the shipped
data/supplementary/heat_2d_sweep_results.npz). A part that is not computed is NaN; when
the output file already exists, that part is carried over from it, so the acoustic half
can be added later with --part acoustic.

Usage (from the repo root):
    python scripts/appendix/E/run_heat_2d_sweep.py --part heat
    python scripts/appendix/E/run_heat_2d_sweep.py --part acoustic --fdtd-root /path/to/fdtd
    python scripts/appendix/E/run_heat_2d_sweep.py               # both (needs all mic files)
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heat_common import (  # noqa: E402
    EXP, K_TRUNC, MODAL, build_heat_temporal_matrix, build_wave_temporal_matrix,
    check_paper_dataset, compute_P, generate_heat_signals, heat_seed, load_heat_room,
    load_room_modal, pad_room_to_K, val_rooms, valid_snaps, window_matrix,
)
import numpy as np  # noqa: E402

P_VALUES = np.arange(0, 4.05, 0.2)       # 21 values
C_VALUES = np.arange(0, 0.52, 0.02)      # 26 values
LAMBDA_GRID = np.array([1e-6, 1e-4, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e6, 1e8, 1e10])
T_VALUES = [1, 10, 100, 500]


def build_gamma_2d(eigenvalues, p, c, is_heat=True):
    ev = np.maximum(eigenvalues, 1e-10)
    gamma_k = np.power(ev, p) * np.exp(c * ev)
    if is_heat:
        return gamma_k
    gamma_2k = np.empty(2 * len(ev))
    gamma_2k[0::2] = gamma_k
    gamma_2k[1::2] = gamma_k
    return gamma_2k


def sweep_2d(A, Y_flat, targets, ev, is_heat):
    ATA = A.T @ A
    ATY = A.T @ Y_flat.T
    P = np.full((len(P_VALUES), len(C_VALUES)), np.nan)
    for p_idx, p in enumerate(P_VALUES):
        for c_idx, c in enumerate(C_VALUES):
            gamma_diag = build_gamma_2d(ev, p, c, is_heat)
            best = np.inf
            for lam in LAMBDA_GRID:
                try:
                    X_r = np.linalg.solve(ATA + lam * np.diag(gamma_diag), ATY)
                except np.linalg.LinAlgError:
                    continue
                preds = X_r.T if is_heat else X_r[0::2, :].T     # cos components
                best = min(best, compute_P(preds, targets))
            P[p_idx, c_idx] = best
    return P


def evaluate_room(args):
    sid, part, modal_root, fdtd_root = args
    t0 = time.time()
    n = (len(T_VALUES), len(P_VALUES), len(C_VALUES))
    P_heat, P_wave = np.full(n, np.nan), np.full(n, np.nan)

    if part in ("heat", "both"):
        room = pad_room_to_K(load_heat_room(sid, modal_root), K_TRUNC)
        ev, dt, sps = room["eigenvalues"], room["dt_sim"], room["steps_per_snap"]
        Y, a_tgt, _ = generate_heat_signals(room, heat_seed(sid))
        for t_idx, T_raw in enumerate(T_VALUES):
            valid = valid_snaps(room["n_snaps"], sps, Y.shape[1], T_raw)
            if len(valid) < 5:
                continue
            A = build_heat_temporal_matrix(room["Phi"], ev, dt, T_raw)
            P_heat[t_idx] = sweep_2d(A, window_matrix(Y, valid, sps, T_raw), a_tgt[:, valid].T, ev, True)

    if part in ("acoustic", "both"):
        room = pad_room_to_K(load_room_modal(sid, modal_root, layout="legacy", M_use=8,
                                             fdtd_root=fdtd_root), K_TRUNC)
        ev, dt, sps = room["eigenvalues"], room["dt_sim"], room["steps_per_snap"]
        y = room["y_click"]
        for t_idx, T_raw in enumerate(T_VALUES):
            valid = valid_snaps(room["n_snaps"], sps, y.shape[1], T_raw)
            if len(valid) < 5:
                continue
            A = build_wave_temporal_matrix(room["Phi"], ev, dt, T_raw, room["gamma_room"], room["c"])
            P_wave[t_idx] = sweep_2d(A, window_matrix(y, valid, sps, T_raw), room["a"][:, valid].T, ev, False)

    return sid, P_heat, P_wave, time.time() - t0


def summary(all_heat, all_wave):
    for t_idx, T_raw in enumerate(T_VALUES):
        print(f"T={T_raw}:")
        for name, S in (("Heat", all_heat), ("Acoustic", all_wave)):
            if np.all(np.isnan(S[:, t_idx])):
                print(f"  {name:8s}: not computed")
                continue
            med = np.nanmedian(S[:, t_idx], axis=0)
            i = np.unravel_index(np.nanargmin(med), med.shape)
            p1 = np.nanmin(med[:, 0])
            print(f"  {name:8s}: median-surface min P={med[i]:.4f} at p={P_VALUES[i[0]]:.1f}, "
                  f"c={C_VALUES[i[1]]:.3f}; c=0 best P={p1:.4f}; improvement {(p1-med[i])/p1*100:.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["heat", "acoustic", "both"], default="both")
    ap.add_argument("--rooms", type=int, default=None, help="first N validation rooms only")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--modal-root", default=str(MODAL))
    ap.add_argument("--fdtd-root", default=None,
                    help="directory with scene_XXXXX/y_mics_eta1.npy (default: --modal-root)")
    ap.add_argument("--out", default=str(EXP / "appendix" / "heat_2d_sweep_results.npz"))
    args = ap.parse_args()
    fdtd_root = args.fdtd_root or args.modal_root

    check_paper_dataset(args.modal_root)
    rooms = val_rooms()[:args.rooms] if args.rooms else val_rooms()
    if args.part in ("acoustic", "both"):
        missing = [s for s in rooms if not os.path.exists(os.path.join(fdtd_root, s, "y_mics_eta1.npy"))]
        if missing:
            sys.exit(f"{len(missing)}/{len(rooms)} rooms lack {fdtd_root}/<scene>/y_mics_eta1.npy "
                     f"(first: {missing[0]}). Run with --part heat, or point --fdtd-root at the "
                     f"FDTD microphone signals.")

    print(f"2-D sweep ({args.part}): {len(rooms)} rooms, p {len(P_VALUES)} x c {len(C_VALUES)} x "
          f"{len(LAMBDA_GRID)} ridge weights, T={T_VALUES}, {args.workers} workers", flush=True)
    t0 = time.time()
    res = {}
    jobs = [(s, args.part, args.modal_root, fdtd_root) for s in rooms]
    with Pool(args.workers) as pool:
        for i, (sid, Ph, Pw, dt_room) in enumerate(pool.imap_unordered(evaluate_room, jobs)):
            res[sid] = (Ph, Pw)
            if (i + 1) % 20 == 0 or i == 0:
                print(f"  [{i+1}/{len(rooms)}] {sid} ({dt_room:.0f}s, elapsed {(time.time()-t0)/60:.1f} min)",
                      flush=True)
    all_heat = np.stack([res[s][0] for s in rooms])
    all_wave = np.stack([res[s][1] for s in rooms])

    if args.part != "both" and os.path.exists(args.out):
        old = np.load(args.out)
        if list(old["room_ids"]) == rooms:
            if args.part == "heat":
                all_wave = old["P_wave_2d"]
            else:
                all_heat = old["P_heat_2d"]
            print(f"Kept the {'acoustic' if args.part == 'heat' else 'heat'} part of the existing {args.out}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, P_heat_2d=all_heat, P_wave_2d=all_wave, p_values=P_VALUES, c_values=C_VALUES,
             T_values=np.array(T_VALUES), lambda_grid=LAMBDA_GRID, room_ids=np.array(rooms))
    print(f"Saved {args.out}  ({(time.time()-t0)/60:.1f} min)")
    summary(all_heat, all_wave)


if __name__ == "__main__":
    main()
