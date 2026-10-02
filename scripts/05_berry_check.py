#!/usr/bin/env python
"""
Step 3: Berry conjecture verification.

For each validation room:
  1. Compute C_kn^2 statistics (empirical vs theoretical 1/(M|Omega|^2))
  2. Compute anisotropy matrix E: ||E||_op, ||E||_F
  3. Stratify by geometry (n_segments)

Output: data/experiments/berry/berry_K{k}_M{m}.npz

Usage:
    python scripts/05_berry_check.py --config configs/default.yaml
    python scripts/05_berry_check.py --config configs/default.yaml --rooms 10
"""

import argparse
import os
import sys
import time

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config
from src.utils.paths import check_paper_dataset
from src.utils.io import load_room_auto, get_val_rooms, get_split_params
from src.analysis.berry import compute_Ckn_squared, compute_anisotropy_E

# Data paths — resolved from config in main(), updated via global
MODAL_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "modal")
LEGACY_MODAL = None
LEGACY_FDTD = None


def _room_dir(scene_id):
    """Resolve room directory: new layout first, legacy fallback."""
    room_id = scene_id.replace("scene_", "room_")
    new_dir = os.path.join(MODAL_ROOT, room_id)
    if os.path.isdir(new_dir):
        return new_dir
    return os.path.join(LEGACY_MODAL, scene_id)


def process_room(scene_id, K_use, M_use=8):
    """Berry check for one room."""
    room = load_room_auto(scene_id, modal_root=MODAL_ROOT,
                          legacy_modal_root=LEGACY_MODAL,
                          legacy_fdtd_root=LEGACY_FDTD, k_trunc=K_use)

    # Load Phi, sliced to M_use mics
    rdir = _room_dir(scene_id)
    Phi_all = np.load(os.path.join(rdir, "measurement_matrix.npy")).astype(np.float64)
    Phi = Phi_all[:M_use]
    eig = np.load(os.path.join(rdir, "eigenpairs.npz"))
    eigenvalues_full = eig["eigenvalues"].astype(np.float64)
    traj = np.load(os.path.join(rdir, "modal_trajectories.npz"))
    a_full = traj["a"].astype(np.float64)
    mic_pos_path = os.path.join(rdir, "mic_pos.npy")
    if not os.path.exists(mic_pos_path):
        mic_pos_path = os.path.join(LEGACY_FDTD, scene_id, "mic_pos.npy")
    mic_pos = np.load(mic_pos_path).astype(np.float64)[:M_use]

    mean_Ckn_sq, theoretical = compute_Ckn_squared(Phi, K_use, room["room_area"])
    E_op, E_fro, _ = compute_anisotropy_E(
        Phi, a_full, eigenvalues_full, mic_pos, K_use
    )

    ratio = np.nanmean(mean_Ckn_sq) / theoretical if theoretical > 0 else np.nan

    # Raw |Omega|*|phi_k(x_m)|^2 for Berry Q-Q (pooled across mics and modes)
    Csq_raw = room["room_area"] * Phi[:, :K_use] ** 2  # (M_use, K_use)
    Csq_flat = Csq_raw.ravel()

    # Per-room KS test against chi2(1)
    ks_stat_r, ks_pval_r = stats.kstest(Csq_flat, "chi2", args=(1,))

    return {
        "scene_id": scene_id,
        "K_total": a_full.shape[0],
        "K_use": K_use,
        "n_seg": room["n_segments"],
        "room_area": room["room_area"],
        "mean_Ckn_sq": mean_Ckn_sq,
        "theoretical_Ckn_sq": theoretical,
        "ratio": ratio,
        "E_op": E_op,
        "E_fro": E_fro,
        "Csq_flat": Csq_flat,
        "ks_stat": ks_stat_r,
        "ks_pval": ks_pval_r,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--rooms", type=int, default=None)
    parser.add_argument("--modal-root", default=None, help="Override MODAL_ROOT (e.g. data/modal_v2)")
    parser.add_argument("--exp-dir", default=None, help="Override experiments dir (e.g. data/experiments_repro)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    check_paper_dataset(cfg["paths"]["modal_root"])

    global MODAL_ROOT, LEGACY_MODAL, LEGACY_FDTD
    MODAL_ROOT = args.modal_root or cfg["paths"].get("modal_root", MODAL_ROOT)
    LEGACY_MODAL = cfg["paths"].get("modal_root")
    LEGACY_FDTD = cfg["paths"].get("fdtd_dir")

    val_start, val_end = get_split_params(cfg)
    val_rooms = get_val_rooms(MODAL_ROOT, cfg["blacklist"]["scene_ids"],
                              val_start=val_start, val_end=val_end)
    if args.rooms:
        val_rooms = val_rooms[:args.rooms]

    exp_base = args.exp_dir or cfg["paths"]["experiments_dir"]
    out_dir = os.path.join(exp_base, "berry")
    os.makedirs(out_dir, exist_ok=True)

    for K_use in cfg["evaluation"]["k_values"]:
        print(f"\nBerry check: K={K_use}, {len(val_rooms)} rooms")
        results = []
        t0 = time.time()

        for i, sid in enumerate(val_rooms):
            # Check if room has enough modes
            rdir = _room_dir(sid)
            traj_path = os.path.join(rdir, "modal_trajectories.npz")
            if not os.path.exists(traj_path):
                continue
            traj = np.load(traj_path)
            if traj["a"].shape[0] <= K_use:
                continue

            r = process_room(sid, K_use)
            results.append(r)

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(val_rooms)}] {elapsed:.0f}s")

        # Save
        N = len(results)
        out_path = os.path.join(out_dir, f"berry_K{K_use}_M8.npz")
        np.savez(
            out_path,
            ratio_per_room=np.array([r["ratio"] for r in results]),
            E_op_norm=np.array([r["E_op"] for r in results]),
            E_fro_norm=np.array([r["E_fro"] for r in results]),
            n_segments=np.array([r["n_seg"] for r in results]),
            room_ids=np.array([r["scene_id"] for r in results]),
            K_use=K_use,
        )
        print(f"  Saved: {out_path} ({N} rooms)")

        # Summary
        ratios = [r["ratio"] for r in results if np.isfinite(r["ratio"])]
        E_ops = [r["E_op"] for r in results if np.isfinite(r["E_op"])]
        print(f"  Ratio median={np.median(ratios):.3f} (1.0 = Berry holds)")
        print(f"  ||E||_op median={np.median(E_ops):.4f}")

        # Save berry_qq_data.npz for src/visualization/fig_berry_qq.py
        if not results:
            print(f"  No rooms with K>={K_use}. Skipping Q-Q data.")
            continue
        all_Csq = np.concatenate([r["Csq_flat"] for r in results])
        ks_pval = np.array([r["ks_pval"] for r in results])
        ks_stat = np.array([r["ks_stat"] for r in results])
        K_total = np.array([r["K_total"] for r in results])

        qq_path = os.path.join(out_dir, "berry_qq_data.npz")
        np.savez(
            qq_path,
            all_Csq=all_Csq,
            ks_pval=ks_pval,
            ks_stat=ks_stat,
            K_total=K_total,
        )
        print(f"  Saved Q-Q data: {qq_path} ({len(all_Csq):,} samples)")

        global_ks, global_pval = stats.kstest(all_Csq, "chi2", args=(1,))
        pass_rate = np.mean(ks_pval > cfg["berry"]["alpha"])
        print(f"  Global KS={global_ks:.4f}, pass rate={pass_rate:.1%}")


if __name__ == "__main__":
    main()
