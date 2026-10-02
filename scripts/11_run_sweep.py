#!/usr/bin/env python
"""
Step 8: Locatello-style sweep — dispatches many training runs.

Sweeps: {model} x {n_train_rooms} x {T} x {seed}
With checkpointing/resume: skips runs whose checkpoint_dir already has model_final.pt.

Usage:
    python scripts/11_run_sweep.py --config configs/sweep_realdata.yaml
    python scripts/11_run_sweep.py --config configs/sweep_capacity.yaml
"""

import argparse
import itertools
import os
import subprocess
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    parser.add_argument("--base_config", default="configs/default.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        sweep_cfg = yaml.safe_load(f)
    sweep = sweep_cfg["sweep"]

    models = sweep["models"]
    n_rooms_list = sweep["n_train_rooms"]
    seeds = sweep["seeds"]
    k = sweep.get("k", 50)
    m = sweep.get("m", 8)
    epochs = sweep.get("epochs", 1000)

    # Determine sweep axes
    if "t_values" in sweep:
        t_values = sweep["t_values"]
        axes = list(itertools.product(models, n_rooms_list, t_values, seeds))
        axis_names = ("model", "n_rooms", "t_value", "seed")
    elif "p_targets" in sweep:
        p_targets = sweep["p_targets"]
        axes = list(itertools.product(models, n_rooms_list, p_targets, seeds))
        axis_names = ("model", "n_rooms", "p_target", "seed")
    else:
        print("ERROR: sweep config must have 't_values' or 'p_targets'")
        return

    total = len(axes)
    print(f"Sweep: {total} runs")
    print(f"  Axes: {axis_names}")
    print(f"  Models: {models}")
    print(f"  Rooms: {n_rooms_list}")
    print(f"  Seeds: {seeds}")
    print()

    n_done = 0
    n_skip = 0
    n_fail = 0

    for i, combo in enumerate(axes):
        model = combo[0]
        n_rooms = combo[1]
        param = combo[2]
        seed = combo[3]

        if "t_values" in sweep:
            run_id = f"{model}_n{n_rooms}_T{param}_s{seed}"
            extra_args = ["--t_value", str(param)]
        else:
            run_id = f"{model}_n{n_rooms}_p{param}_s{seed}"
            extra_args = ["--t_value", "1"]  # Capacity check uses fixed T

        ckpt_dir = os.path.join("checkpoints", run_id)

        # Resume check
        if os.path.exists(os.path.join(ckpt_dir, "model_final.pt")):
            n_skip += 1
            continue

        cmd = [
            sys.executable, "scripts/10_train_models.py",
            "--config", args.base_config,
            "--model", model,
            "--n_train_rooms", str(n_rooms),
            "--seed", str(seed),
            "--k", str(k), "--m", str(m),
            "--epochs", str(epochs),
            "--checkpoint_dir", ckpt_dir,
        ] + extra_args

        print(f"[{i+1}/{total}] {run_id}")

        if args.dry_run:
            print(f"  DRY RUN: {' '.join(cmd)}")
            continue

        result = subprocess.run(cmd)
        if result.returncode == 0:
            n_done += 1
        else:
            n_fail += 1
            print(f"  FAILED: return code {result.returncode}")

    print(f"\nSweep complete: {n_done} done, {n_skip} skipped, {n_fail} failed")


if __name__ == "__main__":
    main()
