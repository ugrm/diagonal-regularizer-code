#!/usr/bin/env python
"""
Train Learned Iterative Ridge (LIR) baseline.

Uses the same training/evaluation protocol as M1/M2/M3:
  - n=800 training rooms, 197 validation rooms
  - T in {1, 100, 1000}, 5 seeds per config
  - Loss = P_modal = MSE(a_hat, a_true) / Var(a_true)
  - Same build_dataset, same evaluation protocol

Usage:
    # Single run
    python scripts/train_lir.py --t_value 1000 --seed 42 --L 10

    # Full sweep (all T, all seeds, multiple L)
    python scripts/train_lir.py --sweep

    # Evaluate only (load checkpoints)
    python scripts/train_lir.py --eval-only
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.lir import LearnedIterativeRidge
from src.utils.paths import check_paper_dataset
from src.data.dataset import load_room_modal, get_scene_ids, K_TRUNC, T_GRID, BLACKLIST
from src.data.features import compute_mode_features, get_valid_windows, pad_room_to_K, FEAT_DIM
from src.physics.temporal import build_wave_temporal_matrix
from src.estimation.ridge import build_gamma_diag, ridge_sweep_svd
from src.estimation.metrics import fit_power_law
from src.utils.config import load_config

K = K_TRUNC  # 50


def _load_room(sid, data_root, layout, M_use, fdtd_root=None):
    return load_room_modal(sid, data_root, layout=layout, M_use=M_use,
                           K_trunc=K, fdtd_root=fdtd_root)


def build_dataset(n_samples, train_rooms, t_value, seed,
                  data_root, layout, M_use, fdtd_root=None):
    """Same dataset builder as 10_train_models.py."""
    rng = np.random.default_rng(seed)
    dataset = []
    attempts = 0

    while len(dataset) < n_samples and attempts < n_samples * 5:
        attempts += 1
        sid = train_rooms[rng.integers(0, len(train_rooms))]
        room = _load_room(sid, data_root, layout, M_use, fdtd_root)
        pad_room_to_K(room)

        T_raw = int(t_value)
        valid_snaps = get_valid_windows(room, T_raw)
        if len(valid_snaps) < 5:
            continue
        si = rng.choice(valid_snaps)

        A = build_wave_temporal_matrix(
            room["Phi"], room["eigenvalues"], room["dt_sim"],
            T_raw, room["gamma_room"], room["c"]
        )

        sps = room["steps_per_snap"]
        sim_step = si * sps
        end = sim_step + 1
        start = end - T_raw
        y = room["y_click"][:, start:end].reshape(-1)

        ATA = A.T @ A
        ATy = A.T @ y
        eigvals = np.linalg.eigvalsh(ATA)
        scale = max(float(eigvals[-1]), 1e-30)
        ATA = ATA / scale
        ATy = ATy / scale

        features = compute_mode_features(
            room["eigenvalues"], room["c"], room["gamma_room"],
            ATA, T_raw, M_use, K
        )

        a_true = room["a"][:, si].astype(np.float32)
        total_var = float(np.sum(np.var(room["a"], axis=1)))

        dataset.append({
            "ATA": torch.from_numpy(ATA.astype(np.float32)),
            "ATy": torch.from_numpy(ATy.astype(np.float32)),
            "features": torch.from_numpy(features),
            "a_true": torch.from_numpy(a_true),
            "total_var": torch.tensor(max(total_var, 1e-10), dtype=torch.float32),
        })

        if len(dataset) % 1000 == 0:
            print(f"    Built {len(dataset)}/{n_samples}", flush=True)

    return dataset


def train_one_run(model, dataset, epochs, lr, checkpoint_dir, seed, device):
    """Train LIR for one configuration."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    batch_size = 32

    trajectory = {"epoch": [], "loss": []}
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        rng = np.random.default_rng(seed + epoch)
        indices = rng.permutation(len(dataset))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            ATA = torch.stack([dataset[j]["ATA"] for j in batch_idx]).to(device)
            ATy = torch.stack([dataset[j]["ATy"] for j in batch_idx]).to(device)
            feats = torch.stack([dataset[j]["features"] for j in batch_idx]).to(device)
            a_true = torch.stack([dataset[j]["a_true"] for j in batch_idx]).to(device)
            total_var = torch.stack([dataset[j]["total_var"] for j in batch_idx]).to(device)

            a_hat_2K, _ = model(ATA, ATy, feats)
            a_hat = model.extract_amplitudes(a_hat_2K)

            mse = ((a_hat - a_true) ** 2).sum(dim=1)
            P = mse / total_var
            loss = P.mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        trajectory["epoch"].append(epoch)
        trajectory["loss"].append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best.pt"))

        if epoch % 50 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{epochs}: loss={avg_loss:.4f}", flush=True)

    return trajectory


def evaluate_lir(model, val_rooms, data_root, layout, M_use, t_value,
                 device, fdtd_root=None):
    """Evaluate LIR on validation rooms. Returns per-room P values."""
    model.eval()
    P_per_room = []

    for sid in val_rooms:
        try:
            room = _load_room(sid, data_root, layout, M_use, fdtd_root)
            pad_room_to_K(room)
        except Exception:
            P_per_room.append(np.nan)
            continue

        T_raw = int(t_value)
        valid_snaps = get_valid_windows(room, T_raw)
        if len(valid_snaps) < 5:
            P_per_room.append(np.nan)
            continue

        A = build_wave_temporal_matrix(
            room["Phi"], room["eigenvalues"], room["dt_sim"],
            T_raw, room["gamma_room"], room["c"]
        )

        sps = room["steps_per_snap"]
        # Collect all predictions and targets, then compute aggregate P
        # (matches compute_P protocol used by the P-sweep)
        all_preds = []
        all_targets = []

        # ATA is the same for all snapshots (temporal basis doesn't change)
        ATA = A.T @ A
        eigvals_ATA = np.linalg.eigvalsh(ATA)
        scale = max(float(eigvals_ATA[-1]), 1e-30)
        ATA_s = ATA / scale

        features = compute_mode_features(
            room["eigenvalues"], room["c"], room["gamma_room"],
            ATA_s, T_raw, M_use, K
        )

        # Batch all valid snapshots for this room in one GPU forward pass
        ATy_batch = []
        valid_si = []
        for si in valid_snaps:
            sim_step = si * sps
            end = sim_step + 1
            start = end - T_raw
            if start < 0 or end > room["y_click"].shape[1]:
                continue
            y = room["y_click"][:, start:end].reshape(-1)
            ATy_batch.append((A.T @ y) / scale)
            valid_si.append(si)

        if len(ATy_batch) < 5:
            P_per_room.append(np.nan)
            continue

        N_batch = len(ATy_batch)
        ATy_np = np.array(ATy_batch, dtype=np.float32)  # (N, 2K)
        ATA_batch = np.tile(ATA_s.astype(np.float32), (N_batch, 1, 1))  # (N, 2K, 2K)
        feats_batch = np.tile(features, (N_batch, 1, 1))

        with torch.no_grad():
            ATA_t = torch.from_numpy(ATA_batch).to(device)
            ATy_t = torch.from_numpy(ATy_np).to(device)
            feats_t = torch.from_numpy(feats_batch).to(device)
            a_hat_2K, _ = model(ATA_t, ATy_t, feats_t)
            all_preds_np = model.extract_amplitudes(a_hat_2K).cpu().numpy()  # (N, K)

        all_preds = list(all_preds_np)
        all_targets = [room["a"][:, si] for si in valid_si]

        if len(all_preds) >= 5:
            preds = np.array(all_preds)    # (N, K)
            targets = np.array(all_targets) # (N, K)
            mse_per_mode = np.mean((preds - targets)**2, axis=0)
            var_per_mode = np.var(targets, axis=0)
            total_var = np.sum(var_per_mode)
            P_room = float(np.sum(mse_per_mode) / max(total_var, 1e-10))
            P_per_room.append(P_room)
        else:
            P_per_room.append(np.nan)

    return np.array(P_per_room)


def main():
    parser = argparse.ArgumentParser(description="Train LIR baseline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--t_value", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--L", type=int, default=10, help="Unrolling depth")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--n_train_rooms", type=int, default=800)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--sweep", action="store_true",
                        help="Run full sweep: L={5,10,20} x T={1,100,1000} x seeds={42-46}")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--output-dir", default="data/experiments/lir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    check_paper_dataset(cfg["paths"]["modal_root"])
    data_root = cfg["paths"]["modal_root"]
    layout = "legacy"
    M_use = 8
    if args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.sweep:
        configs = []
        for L in [1, 5, 10, 20]:
            for T in [1, 100, 1000]:
                for seed in [42, 43, 44, 45, 46]:
                    configs.append((L, T, seed))
        print(f"LIR sweep: {len(configs)} configurations")
    else:
        configs = [(args.L, args.t_value, args.seed)]

    # Get room lists
    blacklist = cfg["blacklist"]["scene_ids"]
    rooms_cfg = cfg.get("rooms", {})
    train_end = rooms_cfg.get("n_train", 800)
    val_start = rooms_cfg.get("val_start", 800)
    val_end = rooms_cfg.get("val_end", 1000)

    train_rooms = get_scene_ids("train", layout=layout, data_root=data_root,
                                train_end=train_end)
    val_rooms = get_scene_ids("val", layout=layout, data_root=data_root,
                              train_end=train_end, val_start=val_start,
                              val_end=val_end)
    # Filter blacklist
    val_rooms = [r for r in val_rooms if r not in blacklist]
    print(f"Train rooms: {len(train_rooms)}, Val rooms: {len(val_rooms)}")

    all_results = []

    for L, T, seed in configs:
        tag = f"lir_L{L}_T{T}_s{seed}"
        ckpt_dir = os.path.join(args.output_dir, "checkpoints", tag)
        os.makedirs(ckpt_dir, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"LIR: L={L}, T={T}, seed={seed}")
        print(f"{'='*60}")

        model = LearnedIterativeRidge(L=L, K=K).to(device)
        print(f"  Parameters: {model.n_params}")

        # L=1 converges fast (52 params); use fewer epochs
        run_epochs = 200 if L == 1 else args.epochs

        if not args.eval_only:
            # Build dataset
            print("  Building dataset...")
            dataset = build_dataset(args.samples, train_rooms, T, seed,
                                    data_root, layout, M_use)
            print(f"  Dataset: {len(dataset)} samples")

            # Train
            t0 = time.time()
            trajectory = train_one_run(model, dataset, run_epochs, args.lr,
                                       ckpt_dir, seed, device)
            elapsed = time.time() - t0
            print(f"  Training done in {elapsed:.0f}s")

            # Save trajectory
            np.savez(os.path.join(ckpt_dir, "trajectory.npz"),
                     epoch=trajectory["epoch"], loss=trajectory["loss"])
        else:
            best_pt = os.path.join(ckpt_dir, "best.pt")
            if not os.path.exists(best_pt):
                print(f"  SKIP: no checkpoint at {best_pt}")
                continue
            model.load_state_dict(torch.load(best_pt, map_location=device))

        # Load best model
        best_pt = os.path.join(ckpt_dir, "best.pt")
        if os.path.exists(best_pt):
            model.load_state_dict(torch.load(best_pt, map_location=device))

        # Evaluate
        print("  Evaluating on validation set...")
        P_vals = evaluate_lir(model, val_rooms, data_root, layout, M_use,
                              T, device)
        median_P = float(np.nanmedian(P_vals))
        print(f"  Median P = {median_P:.4f}")

        # Extract raw per-layer D_k
        gammas = model.get_effective_gamma()

        # Compute effective spectral filter on the median room
        eff_filter = None
        median_room_id = None
        if not np.all(np.isnan(P_vals)):
            median_idx = np.nanargmin(np.abs(P_vals - median_P))
            median_room_id = val_rooms[median_idx]
            try:
                med_room = _load_room(median_room_id, data_root, layout, M_use)
                pad_room_to_K(med_room)
                A_med = build_wave_temporal_matrix(
                    med_room["Phi"], med_room["eigenvalues"], med_room["dt_sim"],
                    T, med_room["gamma_room"], med_room["c"]
                )
                ATA_med = A_med.T @ A_med
                eigvals_med = np.linalg.eigvalsh(ATA_med)
                scale_med = max(float(eigvals_med[-1]), 1e-30)
                ATA_s = (ATA_med / scale_med).astype(np.float32)

                # Measure shrinkage: for each 2K-dim canonical basis vector e_j,
                # run L iterations starting from (ATA)^{-1} ATy = e_j equivalent
                # Instead: run forward pass with ATy = e_j and measure output
                model.eval()
                with torch.no_grad():
                    I_2K = np.eye(2 * K, dtype=np.float32)
                    ATA_batch = np.tile(ATA_s, (2 * K, 1, 1))
                    ATA_t = torch.from_numpy(ATA_batch).to(device)
                    ATy_t = torch.from_numpy(I_2K).to(device)
                    a_out, _ = model(ATA_t, ATy_t)
                    # eff_filter[k] = ||output for e_{2k}|| (amplitude component)
                    out_np = a_out.cpu().numpy()  # (2K, 2K)
                    # Effective filter: diagonal of the linear map ATy -> a_hat
                    eff_filter = np.diag(out_np)[0::2]  # K values for amplitude modes
                print(f"  Effective filter on {median_room_id}: "
                      f"min={eff_filter.min():.4f} max={eff_filter.max():.4f}")
            except Exception as e:
                print(f"  Warning: effective filter failed: {e}")

        result = {
            "L": L, "T": T, "seed": seed,
            "median_P": median_P,
            "P_per_room": P_vals,
            "gammas_raw": gammas,
            "eff_filter": eff_filter,
            "median_room_id": median_room_id,
            "n_params": model.n_params,
            "tag": tag,
        }
        all_results.append(result)

        # Save per-config results immediately (crash resilience)
        per_config_path = os.path.join(ckpt_dir, "eval_results.npz")
        np.savez(per_config_path,
                 L=L, T=T, seed=seed,
                 median_P=median_P,
                 P_per_room=P_vals,
                 gammas_raw=gammas,
                 eff_filter=eff_filter,
                 median_room_id=str(median_room_id) if median_room_id else "",
                 val_rooms=val_rooms)
        print(f"  Saved to {per_config_path}")

    # Save all results (pickle-based, for debugging)
    results_path = os.path.join(args.output_dir, "lir_results.npz")
    np.savez(results_path,
             results=all_results,
             val_rooms=val_rooms,
             allow_pickle=True)
    print(f"\nResults saved to {results_path}")

    # Save clean array-only summary (for figure scripts, no pickle needed)
    L_vals = sorted(set(r["L"] for r in all_results))
    T_vals = sorted(set(r["T"] for r in all_results))
    seed_vals = sorted(set(r["seed"] for r in all_results))
    n_rooms = len(val_rooms)
    rmap = {(r["L"], r["T"], r["seed"]): r for r in all_results}

    P_median = np.full((len(L_vals), len(T_vals), len(seed_vals)), np.nan)
    P_per_room = np.full((len(L_vals), len(T_vals), len(seed_vals), n_rooms), np.nan)
    eff_filter = np.full((len(L_vals), len(T_vals), len(seed_vals), K), np.nan)
    median_room_ids = np.empty((len(L_vals), len(T_vals), len(seed_vals)), dtype="U12")

    for li, L in enumerate(L_vals):
        for ti, T in enumerate(T_vals):
            for si, seed in enumerate(seed_vals):
                r = rmap.get((L, T, seed))
                if r is None:
                    continue
                P_median[li, ti, si] = r["median_P"]
                P_per_room[li, ti, si, :] = r["P_per_room"][:n_rooms]
                if r.get("eff_filter") is not None:
                    eff_filter[li, ti, si, :] = r["eff_filter"]
                median_room_ids[li, ti, si] = str(r["median_room_id"]) if r["median_room_id"] else ""

    summary_path = os.path.join(args.output_dir, "lir_summary.npz")
    np.savez(summary_path,
             P_median=P_median, P_per_room=P_per_room,
             eff_filter=eff_filter, median_room_ids=median_room_ids,
             L_values=np.array(L_vals), T_values=np.array(T_vals),
             seeds=np.array(seed_vals), val_rooms=np.array(val_rooms))
    print(f"Summary saved to {summary_path}")

    # Print summary table
    print(f"\n{'='*60}")
    print(f"LIR Results Summary")
    print(f"{'='*60}")
    print(f"{'L':>3s}  {'T':>5s}  {'seed':>4s}  {'P_median':>8s}")
    for r in all_results:
        print(f"{r['L']:3d}  {r['T']:5d}  {r['seed']:4d}  {r['median_P']:8.4f}")


if __name__ == "__main__":
    main()
