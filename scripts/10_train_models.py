#!/usr/bin/env python
"""
Step 10: Train a single model (M1 CondNet, M2 FixedGamma, or M3 HyperNet).

Each run produces a checkpoint directory with trajectory.npz for figure generation.

Usage:
    python scripts/10_train_models.py --model m3_hypernet --t_value 1 --seed 42
    python scripts/10_train_models.py --model m2_fixedgamma --t_value 1000 --epochs 500
    python scripts/10_train_models.py --model m1_condnet --t_value mixed --epochs 500
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.solver import UnrolledRidgeSolver
from src.utils.paths import check_paper_dataset
from src.models.m3_hypernet import ClosedFormTikhonov
from src.data.dataset import load_room_modal, get_scene_ids, K_TRUNC, T_GRID, BLACKLIST
from src.data.features import compute_mode_features, get_valid_windows, pad_room_to_K, FEAT_DIM
from src.physics.temporal import build_wave_temporal_matrix
from src.estimation.metrics import fit_power_law
from src.utils.config import load_config


K = K_TRUNC


# ─── Data Loading ────────────────────────────────────────────────────────────

def _load_room(sid, data_root, layout, M_use, fdtd_root=None):
    """Load and return room data dict."""
    return load_room_modal(sid, data_root, layout=layout, M_use=M_use,
                           K_trunc=K_TRUNC, fdtd_root=fdtd_root)


def get_train_rooms(data_root, layout, n_rooms=800):
    """Get training room IDs, optionally limited to first n_rooms."""
    rooms = get_scene_ids("train", layout=layout, data_root=data_root)
    # Filter to rooms that actually have data
    filtered = []
    for sid in rooms:
        if layout == "v2":
            room_id = sid.replace("scene_", "room_")
            path = os.path.join(data_root, room_id, "modal_trajectories.npz")
        else:
            path = os.path.join(data_root, sid, "modal_trajectories.npz")
        if os.path.exists(path):
            filtered.append(sid)
    if n_rooms and n_rooms < len(filtered):
        filtered = filtered[:n_rooms]
    return filtered


def build_dataset(n_samples, train_rooms, t_value, seed,
                  data_root, layout, M_use, fdtd_root=None):
    """
    Build training dataset: list of dicts with ATA, ATy, features, a_true, total_var.

    t_value: int (single-T) or "mixed" (random T from T_GRID).
    """
    rng = np.random.default_rng(seed)
    dataset = []
    attempts = 0
    max_attempts = n_samples * 5

    while len(dataset) < n_samples and attempts < max_attempts:
        attempts += 1
        sid = train_rooms[rng.integers(0, len(train_rooms))]
        room = _load_room(sid, data_root, layout, M_use, fdtd_root)
        pad_room_to_K(room)

        if t_value == "mixed":
            T_raw = int(rng.choice(T_GRID))
        else:
            T_raw = int(t_value)

        M = M_use
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
            ATA, T_raw, M, K
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


# ─── Gamma Extraction ────────────────────────────────────────────────────────

def _extract_gamma_per_room(condnet_fn, data_root, layout, M_use,
                            fdtd_root=None, n_rooms=20,
                            train_end=800, val_start=None, val_end=1000):
    """Evaluate a CondNet on val rooms, per-room power law fit -> median p-hat."""
    val_rooms = get_scene_ids("val", layout=layout, data_root=data_root,
                               train_end=train_end, val_start=val_start, val_end=val_end)[:n_rooms]
    per_room_phat = []
    per_room_r2 = []
    for sid in val_rooms:
        room = _load_room(sid, data_root, layout, M_use, fdtd_root)
        pad_room_to_K(room)
        ev = room["eigenvalues"]
        T_raw = 100
        A = build_wave_temporal_matrix(
            room["Phi"], ev, room["dt_sim"],
            T_raw, room["gamma_room"], room["c"]
        )
        ATA = A.T @ A
        eigvals = np.linalg.eigvalsh(ATA)
        scale = max(float(eigvals[-1]), 1e-30)
        ATA_norm = ATA / scale
        features = compute_mode_features(
            ev, room["c"], room["gamma_room"],
            ATA_norm, T_raw, M_use, K
        )
        feat_t = torch.from_numpy(features).unsqueeze(0)
        g = condnet_fn(feat_t).squeeze(0).numpy()
        p, _, r2, _ = fit_power_law(ev, g)
        if np.isfinite(p):
            per_room_phat.append(p)
            per_room_r2.append(r2)
    if per_room_phat:
        return float(np.median(per_room_phat)), float(np.median(per_room_r2))
    return np.nan, np.nan


def extract_gamma(model, eigenvalues_median, data_root, layout, M_use,
                  fdtd_root=None, train_end=800, val_start=None, val_end=1000):
    """Extract learned Gamma spectrum and fit power law."""
    split_kw = dict(train_end=train_end, val_start=val_start, val_end=val_end)
    if isinstance(model, UnrolledRidgeSolver):
        if model.conditioning:
            model.eval()
            with torch.no_grad():
                condnet = model.condnet if model.share_condnet else model.condnets[0]
                return _extract_gamma_per_room(condnet, data_root, layout, M_use, fdtd_root, **split_kw)
        else:
            with torch.no_grad():
                gamma_k = torch.nn.functional.softplus(
                    model.fixed_gamma.log_gamma).numpy()
            p_hat, _, r2, _ = fit_power_law(eigenvalues_median, gamma_k)
            return float(p_hat), float(r2)

    elif isinstance(model, ClosedFormTikhonov):
        model.eval()
        with torch.no_grad():
            return _extract_gamma_per_room(model.condnet, data_root, layout, M_use, fdtd_root, **split_kw)

    return np.nan, np.nan


# ─── Training Loop ───────────────────────────────────────────────────────────

def train_one_run(model, dataset, epochs, lr, checkpoint_dir,
                  eigenvalues_median, data_root, layout, M_use,
                  fdtd_root=None, batch_size=32, seed=42, verbose=True,
                  train_end=800, val_start=None, val_end=1000):
    """Train one model run. Tracks p_hat at each epoch."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    trajectory = {
        "epoch": [], "train_loss": [], "p_hat": [], "p_hat_r2": [],
    }

    best_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        indices = rng.permutation(len(dataset))
        epoch_loss = 0.0
        n_batches = 0

        for batch_start in range(0, len(dataset), batch_size):
            batch_idx = indices[batch_start:batch_start + batch_size]
            if len(batch_idx) < 2:
                continue

            batch = [dataset[i] for i in batch_idx]
            ATA_b = torch.stack([b["ATA"] for b in batch])
            ATy_b = torch.stack([b["ATy"] for b in batch])
            feat_b = torch.stack([b["features"] for b in batch])
            a_true_b = torch.stack([b["a_true"] for b in batch])
            tvar_b = torch.stack([b["total_var"] for b in batch])

            optimizer.zero_grad()
            a_hat, _ = model(ATA_b, ATy_b, feat_b)
            a_pred = model.extract_amplitudes(a_hat)

            mse = torch.mean((a_pred - a_true_b)**2, dim=1)
            loss = torch.mean(mse / tvar_b.clamp(min=1e-10))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Extract p_hat periodically (expensive for M1/M3)
        if (epoch + 1) % 50 == 0 or epoch == 0 or epoch == epochs - 1:
            p_hat, r2 = extract_gamma(model, eigenvalues_median,
                                      data_root, layout, M_use, fdtd_root,
                                      train_end=train_end, val_start=val_start, val_end=val_end)
        else:
            p_hat = trajectory["p_hat"][-1] if trajectory["p_hat"] else np.nan
            r2 = trajectory["p_hat_r2"][-1] if trajectory["p_hat_r2"] else np.nan

        trajectory["epoch"].append(epoch)
        trajectory["train_loss"].append(avg_loss)
        trajectory["p_hat"].append(p_hat)
        trajectory["p_hat_r2"].append(r2)

        if verbose and ((epoch + 1) % 50 == 0 or epoch == 0):
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, "
                  f"p_hat={p_hat:.3f}", flush=True)

        if (epoch + 1) % 100 == 0:
            torch.save(model.state_dict(),
                       os.path.join(checkpoint_dir, f"model_ep{epoch+1:04d}.pt"))

    if best_state:
        model.load_state_dict(best_state)

    p_hat_final, r2_final = extract_gamma(model, eigenvalues_median,
                                          data_root, layout, M_use, fdtd_root,
                                          train_end=train_end, val_start=val_start, val_end=val_end)

    torch.save(model.state_dict(),
               os.path.join(checkpoint_dir, "model_final.pt"))

    traj_arrays = {k: np.array(v) for k, v in trajectory.items()}
    traj_arrays["phat_final"] = p_hat_final
    traj_arrays["r2_final"] = r2_final
    np.savez(os.path.join(checkpoint_dir, "trajectory.npz"), **traj_arrays)

    return trajectory, p_hat_final


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train a single model (M1/M2/M3)")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", required=True,
                        choices=["m1_condnet", "m2_fixedgamma", "m3_hypernet"])
    parser.add_argument("--t_value", default="mixed",
                        help="T value for training (int or 'mixed')")
    parser.add_argument("--n_train_rooms", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--modal_root", default=None,
                        help="Override modal data root (default: from config)")
    parser.add_argument("--fdtd_root", default=None,
                        help="Override FDTD data root for legacy layout")
    parser.add_argument("--layout", default=None, choices=["legacy", "v2"],
                        help="Data layout (default: auto from config)")
    parser.add_argument("--checkpoint_dir", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    check_paper_dataset(cfg["paths"]["modal_root"])

    # Determine data layout and paths: flag > config > default
    data_root = (args.modal_root
                 or cfg["paths"].get("modal_root")
                 or os.path.join(cfg["paths"]["data_root"], "modal"))
    fdtd_root = (args.fdtd_root
                 or cfg["paths"].get("fdtd_root"))
    if args.layout:
        layout = args.layout
    elif "layout" in cfg.get("dataset", {}):
        layout = cfg["dataset"]["layout"]
    else:
        layout = "legacy"
    M_use = args.m

    # Parse t_value
    t_value = args.t_value
    if t_value != "mixed":
        t_value = int(t_value)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_id = f"{args.model}_n{args.n_train_rooms}_T{args.t_value}_s{args.seed}"
    ckpt_dir = args.checkpoint_dir or os.path.join(
        cfg["paths"]["checkpoints_dir"], run_id
    )

    print(f"Training: {run_id}")
    print(f"  Model: {args.model}, T={args.t_value}, n_rooms={args.n_train_rooms}")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}, Seed: {args.seed}")
    print(f"  Samples: {args.samples}, Batch: {args.batch_size}")
    print(f"  Checkpoint: {ckpt_dir}")
    print(f"  Data: {data_root} (layout={layout}), M={M_use}")

    # Create model
    if args.model == "m2_fixedgamma":
        model = UnrolledRidgeSolver(
            L=10, K=K, feat_dim=FEAT_DIM,
            share_condnet=True, conditioning=False)
    elif args.model == "m3_hypernet":
        model = ClosedFormTikhonov(K=K, feat_dim=FEAT_DIM)
    elif args.model == "m1_condnet":
        model = UnrolledRidgeSolver(
            L=10, K=K, feat_dim=FEAT_DIM,
            share_condnet=True, conditioning=True)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params}")

    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(ckpt_dir, "config.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    # Get split parameters from config
    rooms_cfg = cfg.get("rooms", {})
    val_start = rooms_cfg.get("val_start", 800)
    val_end = rooms_cfg.get("val_end", 1000)
    train_end = rooms_cfg.get("n_train", 800)

    # Get reference eigenvalues for power-law fitting
    ref_rooms = get_scene_ids("val", layout=layout, data_root=data_root,
                               train_end=train_end, val_start=val_start, val_end=val_end)[:50]
    ev_all = []
    for sid in ref_rooms:
        room = _load_room(sid, data_root, layout, M_use, fdtd_root)
        pad_room_to_K(room)
        ev_all.append(room["eigenvalues"])
    ev_median = np.median(np.array(ev_all), axis=0)

    # Build dataset
    print("\nBuilding dataset...")
    train_rooms = get_train_rooms(data_root, layout, args.n_train_rooms)
    dataset = build_dataset(args.samples, train_rooms, t_value, args.seed,
                            data_root, layout, M_use, fdtd_root)
    print(f"  Dataset: {len(dataset)} samples from {len(train_rooms)} rooms")

    # Train
    print("\nTraining...")
    t0 = time.time()
    trajectory, phat_final = train_one_run(
        model, dataset, args.epochs, args.lr, ckpt_dir,
        ev_median, data_root, layout, M_use, fdtd_root,
        batch_size=args.batch_size, seed=args.seed,
        train_end=train_end, val_start=val_start, val_end=val_end
    )
    elapsed = time.time() - t0

    print(f"\nDone in {elapsed:.0f}s ({elapsed/args.epochs:.1f}s/epoch)")
    print(f"  Final p_hat = {phat_final:.4f}")
    print(f"  Saved: {ckpt_dir}/trajectory.npz")


if __name__ == "__main__":
    main()
