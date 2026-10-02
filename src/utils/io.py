"""Data loading utilities for the modal pipeline."""

import json
import os

import numpy as np


def load_room_data(scene_id, data_root, k_trunc=50):
    """
    Load eigenvalues, Phi, timing, mic data for one room
    from {data_root}/scenes/{scene_id}/, truncated to the first k_trunc modes.
    """
    scene_dir = os.path.join(data_root, "scenes", scene_id)

    with open(os.path.join(scene_dir, "metadata.json")) as f:
        meta = json.load(f)

    eig = np.load(os.path.join(scene_dir, "eigenpairs.npz"))
    traj = np.load(os.path.join(scene_dir, "modal_trajectories.npz"))

    eigenvalues = eig["eigenvalues"].astype(np.float64)
    a = traj["a"].astype(np.float64)
    gamma_room = float(traj["gamma_room"])
    dt_sim = float(traj["dt_sim"])
    dt_snap = float(traj["dt_snap"])
    c = float(eig["c"])

    K_total = a.shape[0]
    K_use = min(K_total, k_trunc)
    n_snaps = a.shape[1]
    steps_per_snap = int(round(dt_snap / dt_sim))

    # Measurement matrix
    phi_matrix = np.load(
        os.path.join(scene_dir, "phi_matrix.npy")
    ).astype(np.float64)

    # Mic signals
    fdtd_pressure = np.load(
        os.path.join(scene_dir, "fdtd_pressure.npy")
    ).astype(np.float64)

    return {
        "scene_id": scene_id,
        "eigenvalues": eigenvalues[:K_use],
        "a": a[:K_use],
        "Phi": phi_matrix[:, :K_use],
        "y_click": fdtd_pressure,
        "gamma_room": gamma_room,
        "dt_sim": dt_sim,
        "dt_snap": dt_snap,
        "steps_per_snap": steps_per_snap,
        "c": c,
        "K": K_use,
        "K_total": K_total,
        "n_snaps": n_snaps,
        "n_segments": meta.get("n_segments", -1),
        "room_area": meta.get("room_area_m2", 0.0),
    }


def load_room_modal(scene_id, modal_root, k_trunc=50):
    """
    Load room data from the data/modal/ layout (single directory per room).

    Accepts scene_id ("scene_00800") and maps to room_id ("room_00800").
    All data (eigenvalues, Phi, trajectories, FDTD mic signals) in one dir.
    """
    # Map scene_id → room_id (try both naming conventions)
    room_id = scene_id.replace("scene_", "room_")
    room_dir = os.path.join(modal_root, room_id)

    if not os.path.isdir(room_dir):
        # Fall back to scene_* naming if room_* doesn't exist
        room_dir = os.path.join(modal_root, scene_id)

    if not os.path.isdir(room_dir):
        raise FileNotFoundError(f"Room directory not found: {room_dir}")

    with open(os.path.join(room_dir, "metadata.json")) as f:
        meta = json.load(f)

    eig = np.load(os.path.join(room_dir, "eigenpairs.npz"))
    traj = np.load(os.path.join(room_dir, "modal_trajectories.npz"))
    Phi = np.load(os.path.join(room_dir, "measurement_matrix.npy"))

    eigenvalues = eig["eigenvalues"].astype(np.float64)
    frequencies = eig.get("frequencies", np.sqrt(np.maximum(eigenvalues, 0)) * float(eig["c"]) / (2 * np.pi))
    if hasattr(frequencies, 'astype'):
        frequencies = frequencies.astype(np.float64)
    a = traj["a"].astype(np.float64)
    gamma_room = float(traj["gamma_room"])
    dt_sim = float(traj["dt_sim"])
    dt_snap = float(traj["dt_snap"])
    c = float(eig["c"])

    K_total = a.shape[0]
    K_use = min(K_total, k_trunc)
    n_snaps = a.shape[1]
    steps_per_snap = int(round(dt_snap / dt_sim))

    # FDTD mic signals (click probe). None when the room has no signal file, so code that
    # needs the signals fails loudly instead of running on placeholder values.
    y_mics_path = os.path.join(room_dir, "y_mics_eta1.npy")
    if os.path.exists(y_mics_path):
        y_mics = np.load(y_mics_path)
        y_click = y_mics[0].astype(np.float64)
    else:
        y_click = None

    return {
        "scene_id": scene_id,
        "eigenvalues": eigenvalues[:K_use],
        "frequencies": frequencies[:K_use] if len(frequencies) >= K_use else frequencies,
        "a": a[:K_use],
        "Phi": Phi[:, :K_use].astype(np.float64),
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
        "room_area": meta.get("room_area_m2", 0.0),
    }


def load_room_auto(scene_id, modal_root=None, legacy_modal_root=None,
                   legacy_fdtd_root=None, k_trunc=50):
    """
    Load room data, trying new layout first, then legacy fallback.

    Args:
        modal_root: Path to data/modal/ (new layout)
        legacy_modal_root: Path to the legacy modal root (one scene_XXXXX/ per room)
        legacy_fdtd_root: Path to the legacy FDTD root (scene_XXXXX/y_mics_eta1.npy)
    """
    # Try new layout first
    if modal_root:
        try:
            return load_room_modal(scene_id, modal_root, k_trunc)
        except FileNotFoundError:
            pass

    # Fall back to legacy
    if legacy_modal_root and legacy_fdtd_root:
        return load_room_data_legacy(scene_id, legacy_modal_root, legacy_fdtd_root, k_trunc)

    raise FileNotFoundError(
        f"Could not load {scene_id} from modal_root={modal_root} "
        f"or legacy={legacy_modal_root}")


def load_room_data_legacy(scene_id, modal_data_root, fdtd_root, k_trunc=50):
    """
    Load room data from the LEGACY directory layout: one scene_XXXXX/ directory per
    room under modal_data_root, and the microphone signals under fdtd_root/scene_XXXXX/.
    """
    scene_dir = os.path.join(modal_data_root, scene_id)
    fdtd_dir = os.path.join(fdtd_root, scene_id)

    with open(os.path.join(scene_dir, "metadata.json")) as f:
        meta = json.load(f)

    eig = np.load(os.path.join(scene_dir, "eigenpairs.npz"))
    traj = np.load(os.path.join(scene_dir, "modal_trajectories.npz"))
    Phi = np.load(os.path.join(scene_dir, "measurement_matrix.npy"))

    eigenvalues = eig["eigenvalues"].astype(np.float64)
    frequencies = eig["frequencies"].astype(np.float64)
    a = traj["a"].astype(np.float64)
    gamma_room = float(traj["gamma_room"])
    dt_sim = float(traj["dt_sim"])
    dt_snap = float(traj["dt_snap"])
    c = float(eig["c"])

    K_total = a.shape[0]
    K_use = min(K_total, k_trunc)
    n_snaps = a.shape[1]
    steps_per_snap = int(round(dt_snap / dt_sim))

    y_mics = np.load(os.path.join(fdtd_dir, "y_mics_eta1.npy"))
    y_click = y_mics[0].astype(np.float64)

    return {
        "scene_id": scene_id,
        "eigenvalues": eigenvalues[:K_use],
        "frequencies": frequencies[:K_use],
        "a": a[:K_use],
        "Phi": Phi[:, :K_use].astype(np.float64),
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
        "room_area": meta.get("room_area_m2", 0.0),
    }


def get_split_params(cfg):
    """Extract val_start/val_end from config, with defaults for legacy."""
    rooms_cfg = cfg.get("rooms", {})
    return rooms_cfg.get("val_start", 800), rooms_cfg.get("val_end", 1000)


def get_val_rooms(data_root, blacklist_ids=None, n_rooms=None,
                  val_start=800, val_end=1000):
    """
    Get validation room IDs excluding blacklist.

    Args:
        data_root: Path to data/ directory (new layout) or modal_data/ (legacy)
        blacklist_ids: Set of scene IDs to exclude
        n_rooms: Optional limit
        val_start: First val room index (default 800)
        val_end: One past last val room index (default 1000)
    """
    if blacklist_ids is None:
        blacklist_ids = {"scene_00905", "scene_00913", "scene_00921"}

    rooms = []
    for i in range(val_start, val_end):
        sid = f"scene_{i:05d}"
        if sid in blacklist_ids:
            continue
        rooms.append(sid)

    if n_rooms is not None:
        rooms = rooms[:n_rooms]
    return rooms


def get_train_rooms(data_root, n_rooms=None, train_end=800):
    """Get training room IDs (rooms 0 to train_end-1)."""
    rooms = [f"scene_{i:05d}" for i in range(train_end)]
    if n_rooms is not None:
        rooms = rooms[:n_rooms]
    return rooms
