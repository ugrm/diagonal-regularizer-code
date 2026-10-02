#!/usr/bin/env python
"""
Step 4: Noise isotropy check (Approach A/B) and prior spectral decay |s|.

Output: data/experiments/noise_profile/noise_profile_K{k}_M{m}.npz

Usage:
    python scripts/06_fit_noise_profile.py --config configs/default.yaml
    python scripts/06_fit_noise_profile.py --config configs/default.yaml --rooms 10
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config
from src.utils.paths import check_paper_dataset
from src.utils.io import get_val_rooms, get_split_params
from src.analysis.noise_profile import approach_B, approach_A, measure_prior_s
from src.estimation.metrics import fit_power_law

# Data paths — resolved from config in main(), updated via global
MODAL_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "modal")
LEGACY_MODAL = None
LEGACY_FDTD = None


def _room_dir(scene_id):
    """Resolve room directory: new layout first, legacy fallback."""
    room_id = scene_id.replace("scene_", "room_")
    new_dir = os.path.join(MODAL_ROOT, room_id)
    if os.path.isdir(new_dir):
        return new_dir, True
    return os.path.join(LEGACY_MODAL, scene_id), False


def process_room(scene_id, K_use, M_use=8):
    """Compute noise profile for one room."""
    rdir, is_new = _room_dir(scene_id)

    eig = np.load(os.path.join(rdir, "eigenpairs.npz"))
    traj = np.load(os.path.join(rdir, "modal_trajectories.npz"))
    Phi_full = np.load(os.path.join(rdir, "measurement_matrix.npy")).astype(np.float64)

    # Slice Phi to M_use mics (handles M=16 new dataset correctly)
    Phi = Phi_full[:M_use]

    # FDTD mic signals: in new layout they're in the same dir
    y_mics_path = os.path.join(rdir, "y_mics_eta1.npy")
    if not os.path.exists(y_mics_path):
        y_mics_path = os.path.join(LEGACY_FDTD, scene_id, "y_mics_eta1.npy")
    y_mics = np.load(y_mics_path)
    y_click = y_mics[0, :M_use].astype(np.float64)

    a = traj["a"].astype(np.float64)
    ev = eig["eigenvalues"].astype(np.float64)
    dt_snap = float(traj["dt_snap"])
    dt_sim = float(traj["dt_sim"])
    sps = int(round(dt_snap / dt_sim))
    n_snaps = a.shape[1]

    # Truncation noise
    sigma_B = approach_B(Phi, a, K_use)
    sigma_A = approach_A(Phi, a, y_click, K_use, sps, n_snaps)

    # Prior decay
    sigma_sq_all, s_full, r2_s_full = measure_prior_s(a, ev, K_use)

    # Noise slopes
    ev_k = ev[:K_use]
    q_B, _, r2_B, _ = fit_power_law(ev_k, sigma_B)
    q_A, _, r2_A, _ = fit_power_law(ev_k, sigma_A)

    return {
        "scene_id": scene_id,
        "K_total": a.shape[0],
        "sigma_B": sigma_B,
        "sigma_A": sigma_A,
        "sigma_sq_all": sigma_sq_all,
        "eigenvalues": ev_k,
        "q_B": q_B, "r2_B": r2_B,
        "q_A": q_A, "r2_A": r2_A,
        "s_full": s_full, "r2_s_full": r2_s_full,
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
    out_dir = os.path.join(exp_base, "noise_profile")
    os.makedirs(out_dir, exist_ok=True)

    K_use = 50  # Default for backward compat

    # Filter rooms with K_total > K_use
    rooms_to_process = []
    excluded = []
    for sid in val_rooms:
        rdir, _ = _room_dir(sid)
        traj_path = os.path.join(rdir, "modal_trajectories.npz")
        if not os.path.exists(traj_path):
            excluded.append(sid)
            continue
        traj = np.load(traj_path)
        if traj["a"].shape[0] > K_use:
            rooms_to_process.append(sid)
        else:
            excluded.append(sid)

    print(f"Noise profile: K={K_use}, {len(rooms_to_process)} rooms "
          f"({len(excluded)} excluded for K_total <= {K_use})")

    results = []
    t0 = time.time()

    for i, sid in enumerate(rooms_to_process):
        r = process_room(sid, K_use)
        results.append(r)

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(rooms_to_process) - i - 1)
            print(f"  [{i+1}/{len(rooms_to_process)}] q_B={r['q_B']:.3f}, "
                  f"s={r['s_full']:.3f} ({elapsed:.0f}s, ETA {eta/60:.1f}m)")

    # Save — keys match src/visualization/fig_noise_profile.py
    N = len(results)
    out_path = os.path.join(out_dir, f"noise_profile_K{K_use}_M8.npz")

    q_B_full = np.array([r["q_B"] for r in results])
    q_A_full = np.array([r["q_A"] for r in results])
    s_per_room = np.array([r["s_full"] for r in results])

    np.savez(
        out_path,
        # 2D arrays for ribbon plots (N, K_use)
        eigenvalues=np.array([r["eigenvalues"] for r in results]),
        sigma_B=np.array([r["sigma_B"] for r in results]),
        sigma_A=np.array([r["sigma_A"] for r in results]),
        sigma_sq_a_all=np.array([r["sigma_sq_all"] for r in results]),
        # Per-room scalars — key names match the figure module (_full suffix)
        q_B_full=q_B_full,
        q_A_full=q_A_full,
        r2_B_full=np.array([r["r2_B"] for r in results]),
        r2_A_full=np.array([r["r2_A"] for r in results]),
        s_per_room=s_per_room,
        r2_s=np.array([r["r2_s_full"] for r in results]),
        room_ids=np.array([r["scene_id"] for r in results]),
        excluded_rooms=np.array(excluded),
        K_use=K_use,
    )
    print(f"\nSaved: {out_path}")

    # alias read by resolve_data_path
    legacy_link = os.path.join(out_dir, "exp1_noise_profile.npz")
    if os.path.islink(legacy_link) or os.path.exists(legacy_link):
        os.remove(legacy_link)
    os.symlink(os.path.basename(out_path), legacy_link)
    print(f"  Symlink: {legacy_link} -> {os.path.basename(out_path)}")

    # Summary
    valid_qB = q_B_full[np.isfinite(q_B_full)]
    valid_qA = q_A_full[np.isfinite(q_A_full)]
    valid_s = s_per_room[np.isfinite(s_per_room)]
    print(f"\n  q_B median={np.median(valid_qB):.4f} (should be ~0)")
    print(f"  q_A median={np.median(valid_qA):.4f} (should be ~0)")
    print(f"  |s| median={-np.median(valid_s):.4f} (this predicts p*)")


if __name__ == "__main__":
    main()
