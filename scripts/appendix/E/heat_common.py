"""Shared helpers for the Appendix E (heat diffusion) scripts: constants, seeds, a room
loader that does not need microphone signals, and window selection. The heat model itself
is the repository's analytic generator,
src.physics.temporal.generate_heat_signals: a_k(t) = a_k(0) e^{-lambda_k t},
a_k(0) ~ N(0, 1/(1+lambda_k)), kappa = 1, Y = Phi a + 1% white noise.
"""
import json
import os
import sys
from pathlib import Path

# One BLAS thread per process: the drivers parallelise over rooms instead.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.dataset import K_TRUNC, get_scene_ids, load_room_modal  # noqa: E402
from src.data.features import SKIP_FIRST, SKIP_LAST, pad_room_to_K  # noqa: E402
from src.estimation.metrics import compute_P  # noqa: E402
from src.physics.temporal import (  # noqa: E402
    build_heat_temporal_matrix, build_wave_temporal_matrix, generate_heat_signals,
)
from src.utils.paths import DATA, EXP, MODAL, SUPP, check_paper_dataset  # noqa: E402

SEED = 42                       # cross_pde_baselines.SEED
DIAG_ROOMS = ["scene_00805", "scene_00826", "scene_00840", "scene_00880", "scene_00950"]

__all__ = [
    "K_TRUNC", "SKIP_FIRST", "SKIP_LAST", "SEED", "DIAG_ROOMS", "DATA", "EXP", "SUPP", "MODAL",
    "check_paper_dataset", "compute_P", "generate_heat_signals", "build_heat_temporal_matrix",
    "build_wave_temporal_matrix", "pad_room_to_K", "load_room_modal", "val_rooms",
    "heat_seed", "load_heat_room", "valid_snaps", "window_matrix",
]


def val_rooms():
    """The 197 validation rooms (scene_00800..00999 minus the 3 blacklisted)."""
    return get_scene_ids("val")


def heat_seed(scene_id, offset=0):
    """Heat-signal seed behind the shipped heat results: SEED + 10 * room_index + 1 (+ offset)."""
    return SEED + int(scene_id.split("_")[1]) * 10 + 1 + offset


def load_heat_room(scene_id, modal_root=MODAL, K_trunc=K_TRUNC):
    """cross_pde_baselines.load_room_data without the FDTD microphone signals.

    The heat experiments only use eigenvalues, Phi (all 8 microphones), dt_sim,
    steps_per_snap and n_snaps; 'a' and 'frequencies' are kept so that
    pad_room_to_K works unchanged.
    """
    rdir = os.path.join(modal_root, scene_id)
    with open(os.path.join(rdir, "metadata.json")) as f:
        meta = json.load(f)
    eig = np.load(os.path.join(rdir, "eigenpairs.npz"))
    traj = np.load(os.path.join(rdir, "modal_trajectories.npz"))
    Phi = np.load(os.path.join(rdir, "measurement_matrix.npy"))

    eigenvalues = eig["eigenvalues"].astype(np.float64)
    a = traj["a"].astype(np.float64)
    dt_sim = float(traj["dt_sim"])
    dt_snap = float(traj["dt_snap"])
    K_total = a.shape[0]
    K_use = min(K_total, K_trunc)
    return {
        "scene_id": scene_id,
        "eigenvalues": eigenvalues[:K_use],
        "frequencies": eig["frequencies"].astype(np.float64)[:K_use],
        "a": a[:K_use],
        "Phi": Phi[:, :K_use].astype(np.float64),
        "gamma_room": float(traj["gamma_room"]),
        "dt_sim": dt_sim,
        "dt_snap": dt_snap,
        "steps_per_snap": int(round(dt_snap / dt_sim)),
        "c": float(eig["c"]),
        "K": K_use,
        "K_total": K_total,
        "n_snaps": a.shape[1],
        "n_segments": meta.get("n_segments", -1),
    }


def valid_snaps(n_snaps, sps, T_sig, T_raw):
    """Snapshots whose causal window [si*sps+1-T_raw, si*sps+1) fits in the signal."""
    out = []
    for si in range(SKIP_FIRST, n_snaps - SKIP_LAST):
        end = si * sps + 1
        if end - T_raw >= 0 and end <= T_sig:
            out.append(si)
    return out


def window_matrix(Y, snaps, sps, T_raw):
    """(N_win, M*T_raw) matrix of mic windows ending at each snapshot step."""
    M = Y.shape[0]
    out = np.empty((len(snaps), M * T_raw), dtype=np.float64)
    for i, si in enumerate(snaps):
        end = si * sps + 1
        out[i] = Y[:, end - T_raw:end].reshape(-1)
    return out
