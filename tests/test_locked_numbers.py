#!/usr/bin/env python
"""Check the main-text numbers of the paper against the data in data/experiments.

Usage: python tests/test_locked_numbers.py [--exp-dir data/experiments]
Exits non-zero if any check fails. The appendix numbers are checked by
scripts/appendix/*/check_*.py (`make check` runs all of them).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.paths import EXP, MODAL

ROWS = []


def check(name, value, printed, ok):
    ROWS.append((name, value, printed, bool(ok)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default=str(EXP))
    exp = ap.parse_args().exp_dir

    ps = np.load(os.path.join(exp, "p_sweep", "p_sweep_K50_M8.npz"), allow_pickle=True)
    rooms = [str(r) for r in ps["room_ids"]]
    T = ps["T_values"]; p = ps["p_values"]; Mi = list(ps["M_values"]).index(8)
    P = ps["P_oracle"][:, :, Mi, :]
    K = np.array([json.load(open(MODAL / r / "metadata.json"))["K"] for r in rooms]) if MODAL.exists() else None
    noise = np.load(os.path.join(exp, "noise_profile", "noise_profile_K50_M8.npz"), allow_pickle=True)
    inscope_ids = set(str(r) for r in noise["room_ids"])
    inscope = np.array([r in inscope_ids for r in rooms])
    if K is not None:
        check("in-scope rooms = rooms with K_total > 50", int(inscope.sum()), 187, (inscope == (K > 50)).all())

    s = float(np.median(np.abs(noise["s_per_room"])))
    check("population |s| (§5)", round(s, 4), "1.13", round(s, 2) == 1.13)

    # Table 1: largest median relative cost of the closed form vs the per-room oracle
    Pc = np.array([[np.interp(s, p, P[r, t]) for t in range(len(T))] for r in range(len(rooms))])
    Po = np.nanmin(P, axis=2)
    rel = np.median(((Pc - Po) / Po)[inscope], axis=0) * 100
    for lab, sel, val in (("T <= 50", T <= 50, 0.61), ("T <= 100", T <= 100, 1.46), ("all T", T > 0, 5.83)):
        v = rel[sel].max()
        check(f"Table 1: {lab}", round(v, 2), f"{val:.2f}%", round(v, 2) == val)
    check("Table 1: maximum at T = 1000", int(T[np.argmax(rel)]), 1000, T[np.argmax(rel)] == 1000)

    absg = (Pc - Po)[inscope] * 100
    t1000 = list(T).index(1000)
    frac = np.mean(absg[:, t1000] < 1.1) * 100
    check("rooms below 1.1 pp at T = 1000 (§1, §5, T33)", round(frac, 1), "69.0%", round(frac, 1) == 69.0)
    check("worst in-scope absolute cost over all T (T33)", round(absg.max(), 2), "7.61 pp", round(absg.max(), 2) == 7.61)

    # Table 2: M3 against the closed form and the oracle (197 validation rooms)
    m3 = np.load(os.path.join(exp, "m3_performance.npz"))
    printed = {1: (0.7264, 0.7245, 0.0021, 0.7151), 100: (0.6035, 0.6051, 0.0027, 0.5936), 1000: (0.1308, 0.1339, 0.0003, 0.1224)}
    for t, (pr, pm, sd, po) in printed.items():
        i = list(m3["T_values"]).index(t)
        got = (m3["P_ridge_s"][i], m3["P_m3_mean"][i], m3["P_m3_std"][i], m3["P_oracle"][i])
        check(f"Table 2: T = {t}", tuple(round(float(g), 4) for g in got), (pr, pm, sd, po),
              all(round(float(g), 4) == v for g, v in zip(got, (pr, pm, sd, po))))

    # §6: every valid per-seed evaluation of M1-M3 lies at or above the per-room oracle
    ev = np.load(os.path.join(exp, "sweep_P_eval.npz"), allow_pickle=True)
    dP = ev["P_end2end"] - ev["P_oracle_baseline"][None, None, :, None]
    valid = np.isfinite(dP)
    check("§6: per-seed dP >= 0 vs oracle", f"{int((dP[valid] >= 0).sum())}/{int(valid.sum())}", "every valid run",
          (dP[valid] >= 0).all())
    models = [str(m) for m in ev["models"]]
    m2 = np.nanmean(ev["P_end2end"][models.index("M2") if "M2" in models else 1, -1], axis=-1)
    gap = (m2 - ev["P_ridge_s"]) * 100
    check("abstract: no learned model more than 3 pp above the closed form (M2, n = 800)", round(gap.max(), 2), "< 3 pp",
          gap.max() < 3)

    # §6.1: LIR (mean over seeds) never reaches the oracle; at L = 10 it is below the closed form only at T = 1
    lir = np.load(os.path.join(exp, "lir", "lir_summary.npz"))
    P_lir = lir["P_median"].mean(axis=2)                         # (L, T)
    ti = [list(m3["T_values"]).index(t) for t in lir["T_values"]]
    check("§6.1: LIR above the oracle at every L and T", round(float((P_lir - m3["P_oracle"][ti]).min()), 4), "> 0",
          (P_lir > m3["P_oracle"][ti]).all())
    li = list(lir["L_values"]).index(10)
    below = [int(t) for t, p, c in zip(lir["T_values"], P_lir[li], m3["P_ridge_s"][ti]) if p < c]
    check("§6.1: LIR (L = 10) below the closed form only at T = 1", below, "[1]", below == [1])

    w = max(len(r[0]) for r in ROWS)
    for name, v, pr, ok in ROWS:
        print(f"{'OK  ' if ok else 'FAIL'}  {name:<{w}}  data {v}  printed {pr}")
    n_fail = sum(not r[3] for r in ROWS)
    print(f"\n{len(ROWS) - n_fail}/{len(ROWS)} main-text checks pass")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
