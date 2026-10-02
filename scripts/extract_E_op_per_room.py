#!/usr/bin/env python
"""
Extract per-room ||E||_op (operator norm of anisotropy matrix E) for all
legacy validation rooms and save to .npz.

Method B (matches 15_verify_appendix_a.py check_E_op):
  - Phi_trunc = measurement_matrix[:M_USE, K_USE:]
  - sigma_sq  = var(a[K_USE:, :], axis=1)
  - R         = Phi_trunc @ diag(sigma_sq) @ Phi_trunc.T
  - s2        = trace(R) / M_USE
  - E         = R / s2 - I_M
  - ||E||_op  = max |eigenvalue of E|

Usage: python scripts/extract_E_op_per_room.py
Output: data/experiments/anisotropy/E_op_empirical_187.npz
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.paths import EXP, MODAL, check_paper_dataset  # noqa: E402

BLACKLIST = {"scene_00905", "scene_00913", "scene_00921"}
K_USE = 50
M_USE = 8

MODAL_ROOT = str(MODAL)
EXP_DIR = str(EXP)
OUT_DIR = os.path.join(EXP_DIR, "anisotropy")
OUT_PATH = os.path.join(OUT_DIR, "E_op_empirical_187.npz")


def load_legacy_rooms():
    """Return the 187 legacy room IDs from the noise profile."""
    noise_path = os.path.join(EXP_DIR, "noise_profile", "noise_profile_K50_M8.npz")
    noise = np.load(noise_path, allow_pickle=True)
    return list(noise["room_ids"])


def compute_E_op(sid):
    """Compute ||E||_op for a single room. Returns None if room is unusable."""
    room_dir = os.path.join(MODAL_ROOT, sid)
    Phi_path = os.path.join(room_dir, "measurement_matrix.npy")
    traj_path = os.path.join(room_dir, "modal_trajectories.npz")
    eig_path = os.path.join(room_dir, "eigenpairs.npz")

    if not all(os.path.exists(p) for p in [Phi_path, traj_path, eig_path]):
        return None

    Phi = np.load(Phi_path)[:M_USE]
    a = np.load(traj_path)["a"]
    ev = np.load(eig_path, allow_pickle=True)["eigenvalues"]

    if len(ev) <= K_USE:
        return None

    sigma_sq = np.var(a[K_USE:], axis=1)
    Phi_tr = Phi[:, K_USE:]
    R = Phi_tr @ np.diag(sigma_sq) @ Phi_tr.T
    s2 = np.trace(R) / M_USE

    if s2 < 1e-30:
        return None

    E = R / s2 - np.eye(M_USE)
    return float(np.max(np.abs(np.linalg.eigvalsh(E))))


def main():
    check_paper_dataset(MODAL_ROOT)
    legacy_rooms = load_legacy_rooms()
    print(f"Legacy rooms from noise profile: {len(legacy_rooms)}")

    scene_ids = []
    E_op_values = []
    skipped = 0

    for sid in legacy_rooms:
        val = compute_E_op(sid)
        if val is None:
            skipped += 1
            continue
        scene_ids.append(sid)
        E_op_values.append(val)

    scene_ids = np.array(scene_ids, dtype=str)
    E_op_per_room = np.array(E_op_values, dtype=np.float64)

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(
        OUT_PATH,
        scene_ids=scene_ids,
        E_op_per_room=E_op_per_room,
    )

    # Report
    med = float(np.median(E_op_per_room))
    p95 = float(np.percentile(E_op_per_room, 95))
    print(f"\nRooms computed: {len(E_op_per_room)}  (skipped: {skipped})")
    print(f"Median ||E||_op: {med:.4f}")
    print(f"95th percentile: {p95:.4f}")
    print(f"Saved to: {OUT_PATH}")

    # Verify against paper values
    ok = True
    if abs(med - 0.58) > 0.01:
        print(f"\nFAIL: median {med:.4f} != 0.58 (tol 0.01)")
        ok = False
    else:
        print(f"\nPASS: median {med:.4f} ~ 0.58")

    if abs(p95 - 2.42) > 0.05:
        print(f"FAIL: p95 {p95:.4f} != 2.42 (tol 0.05)")
        ok = False
    else:
        print(f"PASS: p95 {p95:.4f} ~ 2.42")

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
