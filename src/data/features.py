"""
Per-mode feature computation and data windowing utilities for training.

compute_mode_features builds a (K, FEAT_DIM) feature matrix from the
observation matrix ATA and eigenvalue spectrum, used as input to CondNet.
"""

import numpy as np


FEAT_DIM = 10
SKIP_FIRST = 5
SKIP_LAST = 2


def compute_mode_features(eigenvalues, c, gamma_room, ATA_norm, T_raw, M, K):
    """
    Per-mode features for CondNet.

    ATA_norm is already divided by max(diag(ATA)).
    Returns: (K, FEAT_DIM) float32
    """
    omega = c * np.sqrt(eigenvalues)

    # Block structure: ATA_norm is (2K, 2K), interleaved [cos0,sin0,cos1,sin1,...]
    diag_full = np.diag(ATA_norm)  # (2K,)
    diag_cos = diag_full[0::2]  # (K,)
    diag_sin = diag_full[1::2]  # (K,)
    diag_avg = (diag_cos + diag_sin) / 2

    # Block Frobenius norms: (K, K)
    ATA_4d = ATA_norm.reshape(K, 2, K, 2)
    block_norms = np.sqrt(np.sum(ATA_4d ** 2, axis=(1, 3)))  # (K, K)

    # Zero out self-coupling for off-diagonal stats
    np.fill_diagonal(block_norms, 0)
    cross_max = np.max(block_norms, axis=1)  # (K,)
    cross_mean = np.sum(block_norms, axis=1) / max(K - 1, 1)

    eps = 1e-10
    trace_avg = np.sum(diag_full) / (2 * K) + eps

    features = np.stack([
        np.log(eigenvalues + eps),            # 0: log eigenvalue
        np.log(omega + eps),                  # 1: log angular frequency
        np.log(diag_avg + eps),               # 2: log self-coupling (norm)
        np.log(cross_max + eps),              # 3: log max cross-coupling
        np.log(cross_mean + eps),             # 4: log mean cross-coupling
        diag_avg / trace_avg,                 # 5: relative self-energy
        cross_max / (diag_avg + eps),         # 6: coupling ratio
        np.full(K, np.log(max(M, 1))),        # 7: log M
        np.full(K, np.log(max(T_raw, 1))),    # 8: log T
        np.full(K, gamma_room),               # 9: damping
    ], axis=1)  # (K, 10)

    return features.astype(np.float32)


def get_valid_windows(room, T_raw):
    """Return list of valid snapshot indices for a given T_raw."""
    sps = room["steps_per_snap"]
    T_sig = room["y_click"].shape[1]
    valid = []
    for si in range(SKIP_FIRST, room["n_snaps"] - SKIP_LAST):
        sim_step = si * sps
        end = sim_step + 1
        start = end - T_raw
        if start >= 0 and end <= T_sig:
            valid.append(si)
    return valid


def pad_room_to_K(room, K_target=50):
    """Pad room data to K_target modes. Zero columns in Phi produce zero
    columns in A, so the model naturally ignores padded modes."""
    K = room["K"]
    room["K_actual"] = K
    if K >= K_target:
        return room

    def pad1d(arr, n):
        out = np.zeros(n, dtype=arr.dtype)
        out[:len(arr)] = arr
        return out

    def pad2d_cols(arr, n_cols):
        r, c = arr.shape
        out = np.zeros((r, n_cols), dtype=arr.dtype)
        out[:, :c] = arr
        return out

    def pad2d_rows(arr, n_rows):
        r, c = arr.shape
        out = np.zeros((n_rows, c), dtype=arr.dtype)
        out[:r, :] = arr
        return out

    room["eigenvalues"] = pad1d(room["eigenvalues"], K_target)
    room["frequencies"] = pad1d(room["frequencies"], K_target)
    room["Phi"] = pad2d_cols(room["Phi"], K_target)
    room["a"] = pad2d_rows(room["a"], K_target)
    room["K"] = K_target
    return room
