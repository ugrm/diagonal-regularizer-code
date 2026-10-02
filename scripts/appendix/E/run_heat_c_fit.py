#!/usr/bin/env python
"""Continuous spectral fit of the heat rate c (T19, T22, fig14, T40 row 7).

Recomputes data/experiments/heat_c_continuous_fit.npz. The recipe
below reproduces every stored value of the shipped file to ~1e-14 (verified by --check):

  * rooms 00805, 00826, 00840, 00880, 00950; snapshots 10, 20, 30, 50, 75, 100, 130, 160;
  * one heat realisation per room, seed SEED + 10 * room_index + 1 (src.physics.temporal);
  * the "empirical amplitude variance" of mode k at snapshot si is the variance of
    a_k over the six snapshots si-3 .. si+2 (Python slice si-3:si+3), which for heat equals
    a_k(t)^2 times a smooth factor in lambda_k (absorbed by the fit's power and intercept);
  * modes whose variance is <= 1e-30 are dropped (late snapshots of the smaller rooms);
  * OLS of log var on [log lambda, lambda, 1]; c_hat = -coef(lambda), its standard error
    from the OLS covariance (n - 3 dof), R^2 of the fit;
  * c_theory = 2 kappa t with kappa = 1 and t = si * steps_per_snap * dt_sim;
  * per-room, pooled ("all5") and "3good" (00805, 00840, 00880) regressions of c_hat on
    c_theory: np.polyfit slope, R^2, 95% CI = slope +- 1.96 * se(slope) (n - 2 dof).

Usage (from the repo root):
    python scripts/appendix/E/run_heat_c_fit.py            # writes data/experiments/appendix/
    python scripts/appendix/E/run_heat_c_fit.py --check    # also compares with the shipped file
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heat_common import DIAG_ROOMS, EXP, MODAL, check_paper_dataset, generate_heat_signals, heat_seed, load_heat_room  # noqa: E402
import numpy as np  # noqa: E402

SNAPS = [10, 20, 30, 50, 75, 100, 130, 160]
HALF_WINDOW = 3            # var over snapshots si-3 .. si+2
VAR_FLOOR = 1e-30
GOOD3 = ["scene_00805", "scene_00840", "scene_00880"]


def spectral_fit(var, ev):
    m = var > VAR_FLOOR
    y = np.log(var[m])
    X = np.column_stack([np.log(ev[m]), ev[m], np.ones(m.sum())])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ b
    sig2 = np.sum(res**2) / (len(y) - X.shape[1])
    se = np.sqrt(sig2 * np.linalg.inv(X.T @ X)[1, 1])
    r2 = 1 - np.sum(res**2) / np.sum((y - y.mean())**2)
    return -b[1], se, r2


def slope_fit(x, y):
    b, a = np.polyfit(x, y, 1)
    res = y - (a + b * x)
    r2 = 1 - np.sum(res**2) / np.sum((y - y.mean())**2)
    se = np.sqrt(np.sum(res**2) / (len(x) - 2) / np.sum((x - x.mean())**2))
    return {"slope": np.float64(b), "ci_lo": np.float64(b - 1.96 * se),
            "ci_hi": np.float64(b + 1.96 * se), "r2": np.float64(r2)}


def compute(modal_root=MODAL):
    rows = []
    for sid in DIAG_ROOMS:
        room = load_heat_room(sid, modal_root)
        ev, sps, dt = room["eigenvalues"], room["steps_per_snap"], room["dt_sim"]
        _, a_snaps, _ = generate_heat_signals(room, heat_seed(sid))
        for si in SNAPS:
            var = np.var(a_snaps[:, si - HALF_WINDOW:si + HALF_WINDOW], axis=1)
            c, se, r2 = spectral_fit(var, ev)
            rows.append((sid, si, 2 * si * sps * dt, c, se, r2))
    rid = np.array([r[0] for r in rows])
    out = {
        "c_theory": np.array([r[2] for r in rows]),
        "c_fit": np.array([r[3] for r in rows]),
        "c_se": np.array([r[4] for r in rows]),
        "spectral_r2": np.array([r[5] for r in rows]),
        "room_ids": rid,
        "snap_indices": np.array([r[1] for r in rows]),
    }
    ct, cf = out["c_theory"], out["c_fit"]
    for name, mask in (("all5", np.ones(len(rid), bool)), ("3good", np.isin(rid, GOOD3))):
        s = slope_fit(ct[mask], cf[mask])
        out[f"slope_{name}"] = s["slope"]
        out[f"slope_{name}_ci_lo"] = s["ci_lo"]
        out[f"slope_{name}_ci_hi"] = s["ci_hi"]
        out[f"slope_{name}_r2"] = s["r2"]
    out["per_room_slopes"] = {sid: slope_fit(ct[rid == sid], cf[rid == sid]) for sid in DIAG_ROOMS}
    return out


def compare(out, ref_path):
    ref = np.load(ref_path, allow_pickle=True)
    print(f"Comparison with {ref_path}: keys equal = {sorted(ref.files) == sorted(out)}")
    worst = 0.0
    for k in ref.files:
        if k == "per_room_slopes":
            a, b = ref[k].item(), out[k]
            diff = max(abs(float(a[r][q]) - float(b[r][q])) for r in a for q in a[r])
        elif ref[k].dtype.kind in "US":
            diff = 0.0 if np.array_equal(ref[k], out[k]) else np.inf
        else:
            diff = float(np.max(np.abs(np.asarray(ref[k], float) - np.asarray(out[k], float))))
        worst = max(worst, diff)
        print(f"  {k:20s} max |diff| = {diff:.2e}")
    return worst


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modal-root", default=str(MODAL))
    ap.add_argument("--out", default=str(EXP / "appendix" / "heat_c_continuous_fit.npz"))
    ap.add_argument("--check", action="store_true", help="compare with data/experiments/heat_c_continuous_fit.npz")
    args = ap.parse_args()

    check_paper_dataset(args.modal_root)
    out = compute(args.modal_root)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, **out)
    print(f"Saved {args.out}")
    print(f"pooled slope {out['slope_all5']:.4f} [{out['slope_all5_ci_lo']:.4f}, {out['slope_all5_ci_hi']:.4f}], "
          f"R2 {out['slope_all5_r2']:.4f}")
    if args.check:
        worst = compare(out, EXP / "heat_c_continuous_fit.npz")
        sys.exit(0 if worst < 1e-10 else 1)


if __name__ == "__main__":
    main()
