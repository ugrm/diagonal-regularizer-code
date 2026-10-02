#!/usr/bin/env python
"""
Step 13: Geometric predictor analysis for Section 5, Layer 3.

Tests whether per-room p*(T) can be predicted from eigenvalue-derived
or geometric features. If R² < 0.1 and no |ρ| > 0.3, Layer 3 is confirmed:
per-room regularization exponent is unpredictable from available features.

Features:
  Tier 1 (eigenvalue-derived — what M3 actually sees):
    spectral_gap:      λ₂ - λ₁
    lambda_K:          last eigenvalue (λ_{K_total})
    weyl_exponent:     slope of log(λ_k) vs log(k) (pure eigenvalue spacing)
    mean_spacing:      mean(diff(λ))
    eigenvalue_density: K_total / λ_{K_total}
    K_total:           total number of modes

  Tier 2 (geometric metadata):
    room_area:         from metadata.json
    n_segments:        boundary complexity
    aspect_ratio:      bounding box width / height
    perimeter:         sum of segment lengths

  Bonus:
    s_room:            |s|_room = slope of log(var(a_k)) vs log(λ_k)

Methods:
  (a) Spearman ρ per feature with 95% bootstrap CIs, Bonferroni-corrected p-values
  (b) Random forest regression (5-fold CV, R² with std)
  (c) Scatter of top-2 features vs p*
  (d) Permutation importance if RF R² > 0.15

Output: data/experiments/geometric_predictors.npz + figures/fig_geometric_predictors.pdf

Usage:
    python scripts/13_geometric_predictors.py --config configs/default.yaml
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config
from src.utils.paths import check_paper_dataset


def load_eigenvalues_and_metadata(modal_root, room_ids, layout="legacy"):
    """Load eigenvalues and metadata for all rooms from individual files."""
    eigenvalues_list = []
    metadata_list = []

    for sid in room_ids:
        if layout == "v2":
            room_id = sid.replace("scene_", "room_")
            rdir = os.path.join(modal_root, room_id)
        else:
            rdir = os.path.join(modal_root, sid)

        # Eigenvalues
        eig = np.load(os.path.join(rdir, "eigenpairs.npz"))
        ev = eig["eigenvalues"].astype(np.float64)
        c = float(eig["c"])

        # Metadata
        meta_path = os.path.join(rdir, "metadata.json")
        with open(meta_path) as f:
            meta = json.load(f)

        # Modal amplitudes for |s|_room computation
        traj_path = os.path.join(rdir, "modal_trajectories.npz")
        traj = np.load(traj_path)
        a = traj["a"].astype(np.float64)

        eigenvalues_list.append(ev)
        metadata_list.append({
            "scene_id": sid,
            "eigenvalues": ev,
            "K_total": len(ev),
            "c": c,
            "room_area": meta.get("room_area_m2", np.nan),
            "n_segments": meta.get("n_segments", np.nan),
            "a": a,
        })

    return metadata_list


def compute_s_room(eigenvalues, a):
    """Compute |s|_room: slope of log(var(a_k)) vs log(λ_k)."""
    K = min(len(eigenvalues), a.shape[0])
    ev = eigenvalues[:K]
    var_a = np.var(a[:K], axis=1)

    # Filter out zero-variance modes and zero eigenvalues
    mask = (var_a > 1e-30) & (ev > 1e-30)
    if mask.sum() < 3:
        return np.nan, 0.0

    log_ev = np.log(ev[mask])
    log_var = np.log(var_a[mask])

    slope, intercept, r_value, p_value, std_err = stats.linregress(log_ev, log_var)
    return -slope, r_value**2  # |s| is negative of slope


def compute_tier1_features(ev):
    """Compute Tier 1 (eigenvalue-derived) features from full eigenvalue vector."""
    K = len(ev)
    features = {}

    # Spectral gap
    features["spectral_gap"] = ev[1] - ev[0] if K >= 2 else np.nan

    # Last eigenvalue
    features["lambda_K"] = ev[-1]

    # Weyl exponent: slope of log(λ_k) vs log(k)
    if K >= 3:
        log_k = np.log(np.arange(1, K + 1))
        log_ev = np.log(np.maximum(ev, 1e-30))
        slope, _, _, _, _ = stats.linregress(log_k, log_ev)
        features["weyl_exponent"] = slope
    else:
        features["weyl_exponent"] = np.nan

    # Mean eigenvalue spacing
    if K >= 2:
        features["mean_spacing"] = np.mean(np.diff(ev))
    else:
        features["mean_spacing"] = np.nan

    # Eigenvalue density at truncation
    features["eigenvalue_density"] = K / ev[-1] if ev[-1] > 0 else np.nan

    # K_total
    features["K_total"] = K

    return features


def compute_tier2_features(meta):
    """Compute Tier 2 (geometric) features from room metadata."""
    features = {}
    features["room_area"] = meta["room_area"]
    features["n_segments"] = meta["n_segments"]

    # Aspect ratio and perimeter would need room vertices
    # For now, use what's available in metadata
    features["aspect_ratio"] = np.nan  # TODO: compute from room_vertices if available
    features["perimeter"] = np.nan     # TODO: compute from room_vertices if available

    return features


def bootstrap_spearman_ci(x, y, n_boot=2000, alpha=0.05, seed=42):
    """Bootstrap 95% CI for Spearman ρ."""
    rng = np.random.default_rng(seed)
    n = len(x)
    rhos = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        rhos[i] = stats.spearmanr(x[idx], y[idx]).statistic
    lo = np.percentile(rhos, 100 * alpha / 2)
    hi = np.percentile(rhos, 100 * (1 - alpha / 2))
    return lo, hi


def main():
    parser = argparse.ArgumentParser(
        description="Geometric predictor analysis for Section 5 Layer 3")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--T_target", type=int, default=1000,
                        help="Primary target T for p* (default: 1000)")
    parser.add_argument("--T_secondary", type=int, default=50,
                        help="Secondary T for comparison (default: 50)")
    parser.add_argument("--layout", default="legacy", choices=["legacy", "v2"])
    parser.add_argument("--modal_root", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    check_paper_dataset(cfg["paths"]["modal_root"])
    exp_dir = cfg["paths"]["experiments_dir"]
    modal_root = args.modal_root or cfg["paths"]["modal_root"]

    # ─── Load p* from p_sweep ────────────────────────────────────────────
    ps_path = os.path.join(exp_dir, "p_sweep", "p_sweep_K50_M8.npz")
    ps = np.load(ps_path, allow_pickle=True)
    P_oracle = ps["P_oracle"]  # (N_rooms, n_T, n_M, n_p)
    p_values = ps["p_values"]
    T_values = ps["T_values"]
    M_values = ps["M_values"]
    room_ids = ps["room_ids"]

    # M=8 index
    M_idx = int(np.where(M_values == 8)[0][0])

    # p* at target T
    T_idx = int(np.where(T_values == args.T_target)[0][0])
    P_slice = P_oracle[:, T_idx, M_idx, :]
    p_star = p_values[np.argmin(P_slice, axis=1)]

    # Secondary T
    T2_idx = int(np.where(T_values == args.T_secondary)[0][0])
    P_slice2 = P_oracle[:, T2_idx, M_idx, :]
    p_star_secondary = p_values[np.argmin(P_slice2, axis=1)]

    N = len(room_ids)
    print(f"Loaded p* for {N} rooms")
    print(f"  T={args.T_target}: p* mean={p_star.mean():.2f}, "
          f"std={p_star.std():.2f}, median={np.median(p_star):.2f}")

    # ─── Exclude degenerate room (scene_00829: P≈1 everywhere) ──────────
    # Find rooms where P is near 1.0 at ALL p values (degenerate)
    P_range = P_slice.max(axis=1) - P_slice.min(axis=1)
    degenerate_mask = P_range < 1e-3
    n_degen = degenerate_mask.sum()
    print(f"  Excluding {n_degen} degenerate room(s) (P range < 0.001)")

    valid_mask = ~degenerate_mask
    room_ids_valid = room_ids[valid_mask]
    p_star_valid = p_star[valid_mask]
    p_star2_valid = p_star_secondary[valid_mask]
    N_valid = len(room_ids_valid)
    print(f"  Analysis on {N_valid} rooms")

    # ─── Load eigenvalues and metadata for all valid rooms ───────────────
    print("\nLoading eigenvalues and metadata...")
    room_data = load_eigenvalues_and_metadata(
        modal_root, room_ids_valid, layout=args.layout)

    # ─── Build feature matrix ────────────────────────────────────────────
    feature_names = []
    feature_matrix = []

    # Tier 1 features
    tier1_names = ["spectral_gap", "lambda_K", "weyl_exponent",
                   "mean_spacing", "eigenvalue_density", "K_total"]
    for rd in room_data:
        t1 = compute_tier1_features(rd["eigenvalues"])
        feature_matrix.append([t1[n] for n in tier1_names])
    feature_names.extend(tier1_names)

    # Tier 2 features
    tier2_names = ["room_area", "n_segments"]
    for i, rd in enumerate(room_data):
        t2 = compute_tier2_features(rd)
        # Append to existing rows
        feature_matrix[i].extend([t2[n] for n in tier2_names])
    feature_names.extend(tier2_names)

    # Bonus: |s|_room
    s_rooms = np.empty(N_valid)
    s_r2 = np.empty(N_valid)
    for i, rd in enumerate(room_data):
        s_rooms[i], s_r2[i] = compute_s_room(rd["eigenvalues"], rd["a"])

    feature_names.append("s_room")
    for i in range(N_valid):
        feature_matrix[i].append(s_rooms[i])

    X = np.array(feature_matrix, dtype=np.float64)
    y = p_star_valid.astype(np.float64)
    y2 = p_star2_valid.astype(np.float64)

    print(f"\nFeature matrix: {X.shape} ({len(feature_names)} features)")
    print(f"Features: {feature_names}")

    # Drop features with NaN > 10% of rooms
    nan_frac = np.mean(np.isnan(X), axis=0)
    valid_features = nan_frac < 0.1
    if not valid_features.all():
        dropped = [feature_names[i] for i in range(len(feature_names))
                   if not valid_features[i]]
        print(f"  Dropping features with >10% NaN: {dropped}")
        X = X[:, valid_features]
        feature_names = [n for n, v in zip(feature_names, valid_features) if v]

    # Drop rows with any remaining NaN
    row_valid = ~np.any(np.isnan(X), axis=1)
    if not row_valid.all():
        print(f"  Dropping {(~row_valid).sum()} rooms with NaN features")
        X = X[row_valid]
        y = y[row_valid]
        y2 = y2[row_valid]
    N_final = len(y)
    print(f"  Final: {N_final} rooms, {len(feature_names)} features")

    # ─── (a) Spearman ρ per feature with bootstrap CIs ──────────────────
    print(f"\n{'='*70}")
    print(f"Spearman correlations with p*(T={args.T_target})")
    print(f"{'='*70}")

    n_tests = len(feature_names)
    bonferroni_alpha = 0.05 / n_tests

    rho_results = []
    for j, fname in enumerate(feature_names):
        rho, pval = stats.spearmanr(X[:, j], y)
        ci_lo, ci_hi = bootstrap_spearman_ci(X[:, j], y)
        sig = "***" if pval < bonferroni_alpha else ("*" if pval < 0.05 else "")
        rho_results.append({
            "feature": fname,
            "rho": rho,
            "pval": pval,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "significant": pval < bonferroni_alpha,
        })
        print(f"  {fname:25s}: ρ={rho:+.3f} [{ci_lo:+.3f}, {ci_hi:+.3f}]  "
              f"p={pval:.4f} {sig}")

    print(f"\n  Bonferroni threshold: α={bonferroni_alpha:.4f} ({n_tests} tests)")

    # ─── (b) Random forest regression (5-fold CV) ───────────────────────
    print(f"\n{'='*70}")
    print("Random Forest Regression (5-fold CV)")
    print(f"{'='*70}")

    rf = RandomForestRegressor(n_estimators=200, max_depth=5,
                               min_samples_leaf=10, random_state=42)
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="r2")
    r2_mean = cv_scores.mean()
    r2_std = cv_scores.std()
    print(f"  R² = {r2_mean:.3f} ± {r2_std:.3f}")
    print(f"  Per-fold: {[f'{s:.3f}' for s in cv_scores]}")

    # Feature importance (fit on full data for importance only)
    rf.fit(X, y)
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print(f"\n  Feature importances:")
    for idx in sorted_idx:
        print(f"    {feature_names[idx]:25s}: {importances[idx]:.3f}")

    # (d) Permutation importance if R² > 0.15
    if r2_mean > 0.15:
        from sklearn.inspection import permutation_importance
        print(f"\n  R² > 0.15 — running permutation importance...")
        perm_result = permutation_importance(rf, X, y, n_repeats=30,
                                             random_state=42, scoring="r2")
        for idx in sorted_idx:
            print(f"    {feature_names[idx]:25s}: "
                  f"{perm_result.importances_mean[idx]:.3f} ± "
                  f"{perm_result.importances_std[idx]:.3f}")

    # ─── Secondary T check ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"Secondary check: p*(T={args.T_secondary})")
    print(f"{'='*70}")

    cv_scores2 = cross_val_score(
        RandomForestRegressor(n_estimators=200, max_depth=5,
                              min_samples_leaf=10, random_state=42),
        X, y2, cv=5, scoring="r2")
    print(f"  RF R² = {cv_scores2.mean():.3f} ± {cv_scores2.std():.3f}")

    # Top-2 Spearman for secondary T
    rhos2 = []
    for j, fname in enumerate(feature_names):
        rho2, _ = stats.spearmanr(X[:, j], y2)
        rhos2.append(abs(rho2))
    top2 = np.argsort(rhos2)[-2:][::-1]
    for idx in top2:
        rho2, pval2 = stats.spearmanr(X[:, idx], y2)
        print(f"  {feature_names[idx]:25s}: ρ={rho2:+.3f} (p={pval2:.4f})")

    # ─── Summary ────────────────────────────────────────────────────────
    max_rho = max(abs(r["rho"]) for r in rho_results)
    best_feat = max(rho_results, key=lambda r: abs(r["rho"]))

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"  Best |ρ|: {max_rho:.3f} ({best_feat['feature']})")
    print(f"  RF R² (T={args.T_target}): {r2_mean:.3f} ± {r2_std:.3f}")
    print(f"  RF R² (T={args.T_secondary}): {cv_scores2.mean():.3f} ± {cv_scores2.std():.3f}")

    if r2_mean < 0.1 and max_rho < 0.3:
        print(f"\n  LAYER 3 CONFIRMED: No feature predicts p* "
              f"(R²={r2_mean:.3f}, max |ρ|={max_rho:.3f})")
    elif r2_mean > 0.2 or max_rho > 0.4:
        print(f"\n  LAYER 3 WEAKENED: Moderate signal detected "
              f"(R²={r2_mean:.3f}, max |ρ|={max_rho:.3f})")
        print(f"  → Consider Option A (capacity argument) or Option C (adjust |s|)")
    else:
        print(f"\n  AMBIGUOUS: Weak signal (R²={r2_mean:.3f}, max |ρ|={max_rho:.3f})")
        print(f"  → Report effect sizes with CIs, frame quantitatively")

    # Paper sentence (always applicable)
    print(f"\n  Paper sentence: \"The best spectral predictor explains "
          f"{100*r2_mean:.1f}% of per-room p* variance (RF 5-fold CV), "
          f"with the strongest individual correlation ρ={best_feat['rho']:+.3f} "
          f"[{best_feat['ci_lo']:+.3f}, {best_feat['ci_hi']:+.3f}] "
          f"for {best_feat['feature']}.\"")

    # ─── Save results ───────────────────────────────────────────────────
    out_path = os.path.join(exp_dir, "geometric_predictors.npz")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path,
             feature_names=np.array(feature_names),
             X=X,
             p_star=y,
             p_star_secondary=y2,
             T_target=args.T_target,
             T_secondary=args.T_secondary,
             rho=np.array([r["rho"] for r in rho_results]),
             rho_pval=np.array([r["pval"] for r in rho_results]),
             rho_ci_lo=np.array([r["ci_lo"] for r in rho_results]),
             rho_ci_hi=np.array([r["ci_hi"] for r in rho_results]),
             rf_r2_mean=r2_mean,
             rf_r2_std=r2_std,
             rf_r2_folds=cv_scores,
             rf_importances=importances,
             s_rooms=s_rooms,
             s_r2=s_r2,
             room_ids=room_ids_valid[row_valid] if not row_valid.all() else room_ids_valid,
             )
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
