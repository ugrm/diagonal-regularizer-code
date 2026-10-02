#!/usr/bin/env python
"""
Build data/experiments/m3_performance.npz (Table T2: M3 against the closed form and the
per-room power-law oracle) from two shipped files:

  data/experiments/sweep_P_eval.npz         P_end2end[m3_hypernet, n=800, T, seed]
  data/experiments/p_sweep/p_sweep_K50_M8.npz  P_oracle[room, T, M=8, p]  (all 197 rooms)

  P_ridge_s  median over rooms of P(p) linearly interpolated at |s| (rounded to 6 decimals;
             the value is stored as s_value)
  P_oracle   median over rooms of min_p P(p)
  P_m3_all   (seed, T) end-to-end P of M3 at n = 800; NaN at T values not trained
  P_m3_mean / P_m3_std   mean and population std (ddof = 0) over the 5 seeds

This builder reproduces T_values, P_ridge_s, P_oracle and s_value of the shipped file
exactly. The shipped P_m3_* entries come from a separate evaluation pass and differ
from sweep_P_eval.npz by < 1e-7.

Usage: python scripts/build/m3_performance.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import EXP  # noqa: E402

OUT_PATH = EXP / "m3_performance.npz"
M_USE = 8
N_TRAIN = 800


def main():
    sw = np.load(EXP / "sweep_P_eval.npz")
    ps = np.load(EXP / "p_sweep" / "p_sweep_K50_M8.npz")
    cost = np.load(EXP / "cost" / "cost_K50_M8.npz")

    s_value = round(float(cost["s_value"]), 6)
    T_values = ps["T_values"]
    p_grid = ps["p_values"]
    P = ps["P_oracle"][:, :, list(ps["M_values"]).index(M_USE), :]  # (rooms, T, p)

    P_s_room = np.array([[np.interp(s_value, p_grid, P[r, t]) for t in range(len(T_values))]
                         for r in range(P.shape[0])])
    P_ridge_s = np.median(P_s_room, axis=0)
    P_oracle = np.median(P.min(axis=2), axis=0)

    m3 = sw["P_end2end"][list(sw["models"]).index("m3_hypernet"),
                         list(sw["n_values"]).index(N_TRAIN)]  # (T_sweep, seed)
    P_m3_all = np.full((m3.shape[1], len(T_values)), np.nan)
    for j, T in enumerate(sw["T_values"]):
        P_m3_all[:, list(T_values).index(T)] = m3[j]
    P_m3_mean = np.full(len(T_values), np.nan)
    P_m3_std = np.full(len(T_values), np.nan)
    trained = ~np.isnan(P_m3_all).any(axis=0)
    P_m3_mean[trained] = P_m3_all[:, trained].mean(axis=0)
    P_m3_std[trained] = P_m3_all[:, trained].std(axis=0)

    np.savez(OUT_PATH, T_values=T_values, P_ridge_s=P_ridge_s, P_oracle=P_oracle,
             P_m3_mean=P_m3_mean, P_m3_std=P_m3_std, P_m3_all=P_m3_all, s_value=s_value)
    print(f"Wrote {OUT_PATH}")
    for t in np.where(trained)[0]:
        print(f"  T={T_values[t]:5d}  P_M3 {P_m3_mean[t]:.4f} +- {P_m3_std[t]:.4f}  "
              f"P(|s|) {P_ridge_s[t]:.4f}  P_oracle {P_oracle[t]:.4f}")


if __name__ == "__main__":
    main()
