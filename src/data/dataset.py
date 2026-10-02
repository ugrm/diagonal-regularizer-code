"""
Modal data loading for training and evaluation.

Supports two directory layouts:
  - "legacy": {data_root}/scene_XXXXX/, one directory per room (the layout of
              the released dataset in data/modal)
  - "v2":     {data_root}/room_XXXXX/, one directory per room, with the split read
              from ../rooms/manifest.json (written by scripts/01_generate_rooms.py)

The caller specifies layout explicitly (no auto-detection).
"""

import json
import os

import numpy as np


K_TRUNC = 50
T_GRID = np.array([1, 5, 10, 20, 50, 100, 200, 500, 1000, 2100])

# Degenerate rooms excluded from all experiments.
# scene_00905: interior_frac=0.0003 (near-zero usable area)
# scene_00913: interior_frac=0.0122 (< 5% interior)
# scene_00921: interior_frac=0.0084 (< 5% interior)
BLACKLIST = frozenset({"scene_00905", "scene_00913", "scene_00921"})


def load_room_modal(scene_id, data_root, layout="legacy", M_use=8,
                    K_trunc=K_TRUNC, fdtd_root=None, require_mics=True):
    """
    Load eigenvalues, Phi, timing, mic data for one room.

    Args:
        scene_id: e.g. "scene_00800"
        data_root: root directory for modal data
        layout: "legacy" ({data_root}/scene_XXXXX/) or "v2" ({data_root}/room_XXXXX/)
        M_use: number of microphones to use (slices Phi and y_click)
        K_trunc: truncation to first K modes
        fdtd_root: for legacy layout, path to FDTD data (y_mics_eta1.npy)
        require_mics: raise if the microphone signals are missing (else y_click is None)

    Returns:
        dict with eigenvalues, Phi, a, y_click, gamma_room, dt_sim, etc.
    """
    if layout == "v2":
        room_id = scene_id.replace("scene_", "room_")
        rdir = os.path.join(data_root, room_id)
    else:
        rdir = os.path.join(data_root, scene_id)

    with open(os.path.join(rdir, "metadata.json")) as f:
        meta = json.load(f)

    eig = np.load(os.path.join(rdir, "eigenpairs.npz"))
    traj = np.load(os.path.join(rdir, "modal_trajectories.npz"))

    eigenvalues = eig["eigenvalues"].astype(np.float64)
    c = float(eig["c"])

    if "frequencies" in eig:
        frequencies = eig["frequencies"].astype(np.float64)
    else:
        frequencies = (np.sqrt(np.maximum(eigenvalues, 0)) * c / (2 * np.pi)).astype(np.float64)

    a = traj["a"].astype(np.float64)
    gamma_room = float(traj["gamma_room"])
    dt_sim = float(traj["dt_sim"])
    dt_snap = float(traj["dt_snap"])

    K_total = a.shape[0]
    K_use = min(K_total, K_trunc)
    n_snaps = a.shape[1]
    steps_per_snap = int(round(dt_snap / dt_sim))

    # Measurement matrix
    if layout == "v2":
        Phi_all = np.load(os.path.join(rdir, "measurement_matrix.npy"))
        Phi = Phi_all[:M_use, :K_use].astype(np.float64)
    else:
        Phi_all = np.load(os.path.join(rdir, "measurement_matrix.npy"))
        Phi = Phi_all[:M_use, :K_use].astype(np.float64)

    # Mic signals
    if layout == "v2":
        y_mics = np.load(os.path.join(rdir, "y_mics_eta1.npy"))
        y_click = y_mics[0, :M_use].astype(np.float64)
    else:
        # Legacy layout: FDTD mic signals in the room directory, or under fdtd_root if given
        y_path = os.path.join(fdtd_root if fdtd_root is not None else data_root, scene_id, "y_mics_eta1.npy")
        if os.path.exists(y_path):
            y_mics = np.load(y_path)
            y_click = y_mics[0, :M_use].astype(np.float64)
        elif require_mics:
            raise FileNotFoundError(f"{y_path} not found (microphone signals; see `make download-data`)")
        else:
            y_click = None

    return {
        "scene_id": scene_id,
        "eigenvalues": eigenvalues[:K_use],
        "frequencies": frequencies[:K_use],
        "a": a[:K_use],
        "Phi": Phi,
        "y_click": y_click,
        "gamma_room": gamma_room,
        "dt_sim": dt_sim,
        "dt_snap": dt_snap,
        "steps_per_snap": steps_per_snap,
        "c": c,
        "K": K_use,
        "K_total": K_total,
        "n_snaps": n_snaps,
        "n_segments": meta.get("n_segments", -1),
    }


def get_scene_ids(split, layout="legacy", data_root=None,
                  train_end=800, val_start=None, val_end=1000):
    """
    Get scene IDs for train or val split, excluding blacklisted rooms.

    Args:
        split: "train" or "val"
        layout: "legacy" or "v2"
        data_root: required for v2 layout (reads manifest.json)
        train_end: last train room index (default 800)
        val_start: first val room index (default = train_end)
        val_end: one past last val room index (default 1000)
    """
    if val_start is None:
        val_start = train_end
    if split == "train":
        start, end = 0, train_end
    else:
        start, end = val_start, val_end

    if layout == "v2" and data_root is not None:
        manifest_path = os.path.join(
            os.path.dirname(data_root), "rooms", "manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
            ids = [f"scene_{e['room_idx']:05d}" for e in manifest
                   if e["status"] == "OK" and start <= e["room_idx"] < end]
            return [s for s in ids if s not in BLACKLIST]

    ids = [f"scene_{i:05d}" for i in range(start, end)]
    return [s for s in ids if s not in BLACKLIST]
