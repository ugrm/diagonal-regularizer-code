#!/usr/bin/env python
"""
Step 6: Cost analysis — cost of using |s| instead of oracle p*.

Computes Method A (cost-of-medians) and Method B (median-of-costs)
at all T values.

Output: data/experiments/cost/cost_K{k}_M{m}.npz

Usage:
    python scripts/09_cost_analysis.py --config configs/default.yaml
    python scripts/09_cost_analysis.py --exp-dir data/experiments_repro  # auto-reads |s| from noise profile
    python scripts/09_cost_analysis.py --s_value 1.27 --exp-dir data/experiments_repro
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config
from src.utils.paths import check_paper_dataset
from src.estimation.metrics import method_A_cost, method_B_cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--s_value", type=float, default=None,
                        help="|s| value (auto-read from noise profile if not set)")
    parser.add_argument("--exp-dir", default=None, help="Override experiments dir")
    args = parser.parse_args()

    cfg = load_config(args.config)
    check_paper_dataset(cfg["paths"]["modal_root"])
    exp_base = args.exp_dir or cfg["paths"]["experiments_dir"]
    p_sweep_dir = os.path.join(exp_base, "p_sweep")
    out_dir = os.path.join(exp_base, "cost")
    os.makedirs(out_dir, exist_ok=True)

    # Load p-sweep results
    p_sweep_path = os.path.join(p_sweep_dir, "p_sweep_K50_M8.npz")
    if not os.path.exists(p_sweep_path):
        print(f"ERROR: {p_sweep_path} not found. Run 05_run_p_sweep.py first.")
        return

    # Resolve |s|: explicit flag > auto-read from noise profile > fail
    if args.s_value is not None:
        s_value = args.s_value
        print(f"|s| = {s_value} (from --s_value flag)")
    else:
        noise_path = os.path.join(exp_base, "noise_profile", "noise_profile_K50_M8.npz")
        if not os.path.exists(noise_path):
            print(f"ERROR: No --s_value and no noise profile at {noise_path}")
            print("  Either pass --s_value or run 04_run_noise_profile.py first.")
            return
        noise_data = np.load(noise_path)
        s_per_room = noise_data["s_per_room"]
        s_value = float(-np.nanmedian(s_per_room))
        print(f"|s| = {s_value:.4f} (auto-read from {noise_path})")

    data = np.load(p_sweep_path)
    P_oracle = data["P_oracle"]  # (N, n_T, n_M, n_p)
    p_grid = data["p_values"] if "p_values" in data else data["p_grid"]
    T_values = data["T_values"]

    print(f"Cost analysis: |s|={s_value:.4f}")
    print(f"  {P_oracle.shape[0]} rooms, {len(T_values)} T values\n")

    m_idx = 1  # M=8

    print(f"{'T':>6s}  {'Method A abs':>12s}  {'Method A rel':>12s}  "
          f"{'Method B abs':>12s}  {'Method B rel':>12s}  {'p*':>5s}")

    results = {
        "T": T_values,
        "method_A_abs": [], "method_A_rel": [],
        "method_B_abs": [], "method_B_rel": [],
        "p_star": [],
    }
    # Per-room distributions for fig_cost_distribution
    per_room_delta_P_all = []
    per_room_delta_rel_all = []

    for t_idx, T in enumerate(T_values):
        P_per_room = P_oracle[:, t_idx, m_idx, :]  # (N, n_p)

        # Method A
        P_median_curve = np.nanmedian(P_per_room, axis=0)
        abs_A, rel_A, p_star = method_A_cost(P_median_curve, p_grid, s_value)

        # Method B (returns per-room arrays)
        abs_B, rel_B, delta_P_rooms, delta_rel_rooms = method_B_cost(P_per_room, p_grid, s_value)

        results["method_A_abs"].append(abs_A)
        results["method_A_rel"].append(rel_A)
        results["method_B_abs"].append(abs_B)
        results["method_B_rel"].append(rel_B)
        results["p_star"].append(p_star)
        per_room_delta_P_all.append(delta_P_rooms)
        per_room_delta_rel_all.append(delta_rel_rooms)

        print(f"{T:6d}  {abs_A:12.6f}  {rel_A*100:11.2f}%  "
              f"{abs_B:12.6f}  {rel_B*100:11.2f}%  {p_star:5.1f}")

    # Save (including per-room distributions for figS3)
    out_path = os.path.join(out_dir, "cost_K50_M8.npz")
    np.savez(
        out_path,
        **{k: np.array(v) for k, v in results.items()},
        s_value=s_value,
        p_grid=p_grid,
        per_room_delta_P=np.array(per_room_delta_P_all),    # (n_T, N)
        per_room_delta_rel=np.array(per_room_delta_rel_all),  # (n_T, N)
    )
    print(f"\nSaved: {out_path}")

    # Three-tier summary (Method B)
    rel_B = np.array(results["method_B_rel"])
    T_arr = np.array(T_values, dtype=float)
    tiers = [(50, "<1%"), (100, "<2%"), (2100, "<6%")]
    print("\nThree-tier summary (Method B):")
    for T_thresh, label in tiers:
        mask = T_arr <= T_thresh
        if mask.any():
            max_cost = np.max(rel_B[mask]) * 100
            print(f"  T<={T_thresh}: max relative cost = {max_cost:.2f}% (claim: {label})")


if __name__ == "__main__":
    main()
