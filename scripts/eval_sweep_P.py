#!/usr/bin/env python
"""
Evaluate trained M1/M2/M3 models on validation rooms → P(T) for fig24 heatmap.

Optimized: precomputes temporal matrices per (room, T), reuses across models/seeds.

Output: data/experiments/sweep_P_eval.npz

Usage:
    python scripts/eval_sweep_P.py
    python scripts/eval_sweep_P.py --config configs/default.yaml --n-rooms 20
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config
from src.utils.paths import check_paper_dataset
from src.utils.io import get_val_rooms, load_room_auto, get_split_params
from src.data.features import compute_mode_features, get_valid_windows, pad_room_to_K, FEAT_DIM
from src.physics.temporal import build_wave_temporal_matrix
from src.models.solver import UnrolledRidgeSolver
from src.models.m3_hypernet import ClosedFormTikhonov
from src.estimation.metrics import compute_P
from src.estimation.ridge import ridge_sweep_svd

K = 50
M_USE = 8
T_GRID = [1, 100, 1000]
LAMBDA_GRID = np.array([1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0,
                         100., 1e3, 1e4, 1e6, 1e8, 1e10])


def load_model(model_type, checkpoint_dir):
    if model_type == "m1_condnet":
        model = UnrolledRidgeSolver(L=10, K=K, feat_dim=FEAT_DIM,
                                   share_condnet=True, conditioning=True)
    elif model_type == "m2_fixedgamma":
        model = UnrolledRidgeSolver(L=10, K=K, feat_dim=FEAT_DIM,
                                   share_condnet=True, conditioning=False)
    elif model_type == "m3_hypernet":
        model = ClosedFormTikhonov(K=K, feat_dim=FEAT_DIM)
    else:
        raise ValueError(f"Unknown model: {model_type}")
    state = torch.load(os.path.join(checkpoint_dir, "model_final.pt"),
                       map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def precompute_room_data(room, T_raw):
    """Precompute all per-snapshot data for a (room, T) pair."""
    pad_room_to_K(room)
    valid_snaps = get_valid_windows(room, T_raw)
    if len(valid_snaps) < 5:
        return None

    Phi = room["Phi"]
    ev = room["eigenvalues"]
    dt = room["dt_sim"]
    gamma = room["gamma_room"]
    c = room["c"]
    sps = room["steps_per_snap"]

    A = build_wave_temporal_matrix(Phi, ev, dt, T_raw, gamma, c)
    targets = room["a"][:, valid_snaps].T

    snap_data = []
    for si in valid_snaps:
        sim_step = si * sps
        end = sim_step + 1
        start = end - T_raw
        y = room["y_click"][:, start:end].reshape(-1)

        ATA = A.T @ A
        ATy = A.T @ y
        eigvals = np.linalg.eigvalsh(ATA)
        scale = max(float(eigvals[-1]), 1e-30)

        features = compute_mode_features(ev, c, gamma, ATA / scale, T_raw, M_USE, K)

        snap_data.append({
            "ATA_norm": torch.from_numpy((ATA / scale).astype(np.float32)),
            "ATy_norm": torch.from_numpy((ATy / scale).astype(np.float32)),
            "features": torch.from_numpy(features),
            "a_true": room["a"][:, si].astype(np.float32),
        })

    return {"snap_data": snap_data, "targets": targets, "A": A,
            "valid_snaps": valid_snaps, "room": room}


def eval_model_cached(model, precomp):
    """Evaluate model using precomputed data. Returns P_end2end and gamma_k."""
    snap_data = precomp["snap_data"]
    is_m3 = isinstance(model, ClosedFormTikhonov)

    all_preds = []
    all_targets = []
    all_gamma_k = []

    for sd in snap_data:
        ATA_t = sd["ATA_norm"].unsqueeze(0)
        ATy_t = sd["ATy_norm"].unsqueeze(0)
        feat_t = sd["features"].unsqueeze(0)

        with torch.no_grad():
            if is_m3:
                a_hat, gamma_k = model(ATA_t, ATy_t, feat_t)
                gk = gamma_k.squeeze(0).numpy()
            else:
                a_hat, gammas = model(ATA_t, ATy_t, feat_t, return_gamma=True)
                gk = gammas[-1].squeeze(0).numpy() if gammas else None
            a_pred = model.extract_amplitudes(a_hat).squeeze(0).numpy()

        all_preds.append(a_pred)
        all_targets.append(sd["a_true"])
        if gk is not None:
            all_gamma_k.append(gk)

    P_val = compute_P(np.array(all_preds), np.array(all_targets))
    gamma_k_med = np.median(np.array(all_gamma_k), axis=0) if all_gamma_k else None
    return P_val, gamma_k_med


def eval_oracle_gamma_cached(precomp, gamma_k):
    """Ridge with model's Gamma, oracle alpha. Uses precomputed A and targets."""
    A = precomp["A"]
    targets = precomp["targets"]

    gamma_diag = np.empty(2 * K)
    gamma_diag[0::2] = gamma_k[:K]
    gamma_diag[1::2] = gamma_k[:K]

    room = precomp["room"]
    valid_snaps = precomp["valid_snaps"]
    sps = room["steps_per_snap"]
    T_raw = A.shape[0] // room["Phi"].shape[0]
    N = len(valid_snaps)
    Y = np.empty((N, A.shape[0]))
    for i, si in enumerate(valid_snaps):
        sim_step = si * sps
        end = sim_step + 1
        start = end - T_raw
        Y[i] = room["y_click"][:, start:end].reshape(-1)

    P_values, _ = ridge_sweep_svd(A, Y, targets, gamma_diag, LAMBDA_GRID, is_wave=True)
    return float(np.nanmin(P_values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--n-rooms", type=int, default=None)
    parser.add_argument("--out", default="data/experiments/sweep_P_eval.npz")
    args = parser.parse_args()

    cfg = load_config(args.config)
    check_paper_dataset(cfg["paths"]["modal_root"])
    modal_root = cfg["paths"].get("modal_root",
                                   os.path.join(os.path.dirname(os.path.dirname(
                                       os.path.abspath(__file__))), "data", "modal"))
    blacklist = set(cfg["blacklist"]["scene_ids"])

    val_start, val_end = get_split_params(cfg)
    val_rooms = get_val_rooms(modal_root, blacklist,
                              val_start=val_start, val_end=val_end)
    if args.n_rooms:
        val_rooms = val_rooms[:args.n_rooms]
    N = len(val_rooms)

    # Phase 1: precompute
    print(f"Phase 1: Loading {N} rooms and precomputing temporal matrices...")
    t0 = time.time()

    precomp = {}
    for ti, T in enumerate(T_GRID):
        for sid in val_rooms:
            room = load_room_auto(sid, modal_root=modal_root, k_trunc=K)
            pc = precompute_room_data(room, T)
            if pc is not None:
                precomp[(sid, T)] = pc
        print(f"  T={T}: {sum(1 for s in val_rooms if (s,T) in precomp)}/{N} rooms")
    print(f"  Precompute: {time.time()-t0:.0f}s")

    # Baselines from p_sweep
    P_ridge_s = np.full(len(T_GRID), np.nan)
    P_oracle_baseline = np.full(len(T_GRID), np.nan)
    psweep_path = os.path.join("data", "experiments", "p_sweep", "p_sweep_K50_M8.npz")
    if os.path.exists(psweep_path):
        psweep = np.load(psweep_path)
        psweep_T = psweep["T_values"]
        psweep_P = psweep["P_oracle"]
        psweep_p = psweep["p_values"]
        idx_s = np.argmin(np.abs(psweep_p - 1.13))
        for ti, T in enumerate(T_GRID):
            t_idx = np.where(psweep_T == T)[0]
            if len(t_idx) > 0:
                t_idx = t_idx[0]
                P_ridge_s[ti] = float(np.nanmedian(psweep_P[:, t_idx, 1, idx_s]))
                P_oracle_baseline[ti] = float(np.nanmedian(
                    np.nanmin(psweep_P[:, t_idx, 1, :], axis=1)))

    print(f"\nBaselines (M=8):")
    for ti, T in enumerate(T_GRID):
        print(f"  T={T:5d}: P_ridge(|s|)={P_ridge_s[ti]:.4f}, "
              f"P_oracle={P_oracle_baseline[ti]:.4f}")

    # Phase 2: evaluate models
    models = ["m1_condnet", "m2_fixedgamma", "m3_hypernet"]
    n_values = [50, 100, 200, 400, 800]
    seeds = [42, 43, 44, 45, 46]

    P_end2end = np.full((len(models), len(n_values), len(T_GRID), len(seeds)), np.nan)
    P_oracle_gamma = np.full_like(P_end2end, np.nan)

    print(f"\nPhase 2: Evaluating models...")
    done = 0
    skipped = 0
    t1 = time.time()

    for mi, model_type in enumerate(models):
        for ni, n_train in enumerate(n_values):
            for ti, T in enumerate(T_GRID):
                for si, seed in enumerate(seeds):
                    run_id = f"{model_type}_n{n_train}_T{T}_s{seed}"
                    ckpt_dir = os.path.join("checkpoints", run_id)

                    if not os.path.exists(os.path.join(ckpt_dir, "model_final.pt")):
                        skipped += 1
                        continue

                    model = load_model(model_type, ckpt_dir)

                    room_P = []
                    room_gamma = []
                    for sid in val_rooms:
                        key = (sid, T)
                        if key not in precomp:
                            room_P.append(np.nan)
                            continue
                        P_val, gk = eval_model_cached(model, precomp[key])
                        room_P.append(P_val)
                        if gk is not None:
                            room_gamma.append(gk)

                    P_end2end[mi, ni, ti, si] = float(np.nanmedian(room_P))

                    if room_gamma:
                        gamma_k_med = np.median(np.array(room_gamma), axis=0)
                        oracle_P = []
                        for sid in val_rooms:
                            key = (sid, T)
                            if key not in precomp:
                                oracle_P.append(np.nan)
                                continue
                            op = eval_oracle_gamma_cached(precomp[key], gamma_k_med)
                            oracle_P.append(op)
                        P_oracle_gamma[mi, ni, ti, si] = float(np.nanmedian(oracle_P))

                    done += 1
                    if done % 10 == 0 or done == 1:
                        elapsed = time.time() - t1
                        print(f"  [{done}] {run_id}: "
                              f"P={P_end2end[mi,ni,ti,si]:.4f} ({elapsed:.0f}s)",
                              flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(
        args.out,
        P_end2end=P_end2end,
        P_oracle_gamma=P_oracle_gamma,
        P_ridge_s=P_ridge_s,
        P_oracle_baseline=P_oracle_baseline,
        models=np.array(models),
        n_values=np.array(n_values),
        T_values=np.array(T_GRID),
        seeds=np.array(seeds),
        n_val_rooms=N,
    )
    print(f"\nSaved: {args.out}")
    print(f"  Done: {done}, Skipped: {skipped}, Time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
