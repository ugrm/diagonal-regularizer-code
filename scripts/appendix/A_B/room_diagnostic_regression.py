#!/usr/bin/env python
"""
Per-room diagnostic regression: which diagnostic best predicts the per-room
residual gap to the diagonal oracle, across rooms and observation times T?

Diagnostics tested (per room):
  physics-only  (eigenvalues alone):  H (Herfindahl), dynamic range (lam_K/lam_1)^|s|,
                                      K_total, area
  noise-stats   (need phi at mics):   ||E||_op (Method B), Berry KS D (chi2(1), 400 samples)
  landscape     (need the sweep):     flatness ratio max/min P(p), curvature at p*,
                                      drift |p* - |s||

Target: per-room relative cost delta_rel(T) = (P(|s|) - P(p*)) / P(p*), Method B,
from the released cost_K50_M8.npz (fractions).

Validation anchors (must reproduce before the new analysis is trusted):
  A1  median delta_rel(T=1000) over 187 in-scope rooms  ~ 5.82%   (Table 1)
  A2  Spearman rho(||E||_op, delta(T=1000)), 187 rooms  ~ -0.30   (Fig 4)
  A3  Spearman rho(KS D, delta(T=1000)), 168 rooms      ~ +0.22   (Eq. 39, B.4)
  A4  Spearman rho(KS D, N_trunc), in-scope rooms       ~ +0.29   (B.2)
  A5  median per-room KS D                              ~ 0.042   (B.2)  [197 rooms]
  A6  H median ~ 0.005, ||E||_op median ~ 0.58          (A.2/A.3)

Inputs: modal data from src.utils.paths.MODAL, experiment npz files from data/experiments.

Outputs (Appendix B T36 per-room KS rows, T6 columns, T32 rho(E_op, delta)):
  data/experiments/appendix/room_diagnostic_regression.npz   assembled per-room table
  data/experiments/appendix/room_diagnostic_regression.md    correlation tables + regression results
Usage: python scripts/appendix/A_B/room_diagnostic_regression.py
"""

import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.utils.paths import DATA, MODAL, check_paper_dataset  # noqa: E402

REL = str(DATA)
LEGACY_MODAL = str(MODAL)
OUT = os.path.join(REL, "experiments", "appendix")
OUT_NAME = "room_diagnostic_regression"

K_USE = 50
M_USE = 8
S_VAL = None  # read from cost npz


def shoelace(verts):
    x, y = verts[:, 0], verts[:, 1]
    return float(np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def main():
    global S_VAL
    check_paper_dataset(LEGACY_MODAL)
    os.makedirs(OUT, exist_ok=True)
    cost = np.load(f"{REL}/experiments/cost/cost_K50_M8.npz", allow_pickle=True)
    psw = np.load(f"{REL}/experiments/p_sweep/p_sweep_K50_M8.npz", allow_pickle=True)
    eop = np.load(f"{REL}/experiments/anisotropy/E_op_empirical_187.npz", allow_pickle=True)

    S_VAL = float(cost["s_value"])
    T_values = cost["T"]
    p_grid = psw["p_values"]
    M_values = list(psw["M_values"])
    m_idx = M_values.index(M_USE)
    rooms = list(psw["room_ids"])
    n_rooms = len(rooms)
    delta_rel = cost["per_room_delta_rel"]  # (10, 197), fractions
    P_oracle = psw["P_oracle"][:, :, m_idx, :]  # (197, 10, 61)

    eop_map = dict(zip(list(eop["scene_ids"]), eop["E_op_per_room"]))

    # ---- per-room physics + noise-stat diagnostics from legacy modal data ----
    cols = {k: np.full(n_rooms, np.nan) for k in
            ["K_total", "N_trunc", "lam1", "lamK", "dyn_range", "H", "area",
             "ks_D", "ks_D_norm", "E_op"]}
    for i, sid in enumerate(rooms):
        d = os.path.join(LEGACY_MODAL, sid)
        eig = np.load(os.path.join(d, "eigenpairs.npz"), allow_pickle=True)
        ev = np.asarray(eig["eigenvalues"], dtype=float)
        cols["K_total"][i] = len(ev)
        cols["N_trunc"][i] = len(ev) - K_USE
        if len(ev) >= K_USE:
            cols["lam1"][i] = ev[0]
            cols["lamK"][i] = ev[K_USE - 1]
            cols["dyn_range"][i] = (ev[K_USE - 1] / ev[0]) ** S_VAL
        if len(ev) > K_USE:
            w = ev[K_USE:] ** (-S_VAL)
            cols["H"][i] = np.sum(w ** 2) / np.sum(w) ** 2
        if "room_vertices" in eig:
            area = shoelace(np.asarray(eig["room_vertices"], dtype=float))
        else:
            area = np.nan
        cols["area"][i] = area
        Phi = np.load(os.path.join(d, "measurement_matrix.npy"))[:M_USE]
        if np.isfinite(area):
            vals = (np.sqrt(area) * Phi[:, :K_USE]).ravel()
            Csq = vals ** 2
            cols["ks_D"][i], _ = stats.kstest(Csq, "chi2", args=(1,))
            cols["ks_D_norm"][i], _ = stats.kstest(vals, "norm")
        cols["E_op"][i] = eop_map.get(sid, np.nan)

    # ---- landscape diagnostics per room per T ----
    nT = len(T_values)
    p_star_room = np.full((n_rooms, nT), np.nan)
    P_star_room = np.full((n_rooms, nT), np.nan)
    flatness = np.full((n_rooms, nT), np.nan)     # max/min over p in [0,3]
    curv_norm = np.full((n_rooms, nT), np.nan)    # P''(p*)/P(p*), 2nd-order FD
    drift = np.full((n_rooms, nT), np.nan)        # |p*_room - |s||
    taylor = np.full((n_rooms, nT), np.nan)       # 0.5*curv*(drift)^2 / ... -> rel
    dp = p_grid[1] - p_grid[0]
    for i in range(n_rooms):
        for t in range(nT):
            c = P_oracle[i, t, :]
            if np.all(np.isnan(c)):
                continue
            j = int(np.nanargmin(c))
            p_star_room[i, t] = p_grid[j]
            P_star_room[i, t] = c[j]
            flatness[i, t] = np.nanmax(c) / np.nanmin(c)
            if 0 < j < len(c) - 1:
                curv = (c[j - 1] - 2 * c[j] + c[j + 1]) / dp ** 2
            elif j == 0:
                curv = (c[0] - 2 * c[1] + c[2]) / dp ** 2
            else:
                curv = (c[-3] - 2 * c[-2] + c[-1]) / dp ** 2
            curv_norm[i, t] = curv / c[j]
            drift[i, t] = abs(p_star_room[i, t] - S_VAL)
            taylor[i, t] = 0.5 * curv * drift[i, t] ** 2 / c[j]

    in_scope = cols["K_total"] > K_USE          # 187 rooms
    b4_mask = cols["N_trunc"] >= K_USE          # 168 rooms (B.4 eligibility)

    # ---------------- validation anchors ----------------
    t1000 = list(T_values).index(1000)
    d1000 = delta_rel[t1000]
    rep = []
    rep.append("# Per-room diagnostic regression - report\n")
    rep.append(f"|s| = {S_VAL:.4f}; rooms n={n_rooms}, in-scope={int(in_scope.sum())}, "
               f"B.4-eligible={int(b4_mask.sum())}\n")
    rep.append("## Anchor validation (paper-published values)\n")

    a1 = 100 * np.nanmedian(d1000[in_scope])
    rep.append(f"- A1 median delta(T=1000), in-scope: **{a1:.2f}%** (paper 5.82%)")
    ok = np.isfinite(cols["E_op"]) & np.isfinite(d1000)
    r2, p2 = stats.spearmanr(cols["E_op"][ok], d1000[ok])
    rep.append(f"- A2 rho(E_op, delta1000): **{r2:+.2f}** (p={p2:.1e}, n={ok.sum()}) (paper -0.30)")
    ok3 = b4_mask & np.isfinite(cols["ks_D"]) & np.isfinite(d1000)
    r3, p3 = stats.spearmanr(cols["ks_D"][ok3], d1000[ok3])
    rep.append(f"- A3 rho(KS D, delta1000), B.4 subset: **{r3:+.2f}** (p={p3:.3f}, n={ok3.sum()}) (paper +0.22, p=0.005, n=168)")
    ok4 = in_scope & np.isfinite(cols["ks_D"])
    r4, p4 = stats.spearmanr(cols["ks_D"][ok4], cols["N_trunc"][ok4])
    rep.append(f"- A4 rho(KS D, N_trunc), in-scope: **{r4:+.2f}** (paper +0.29)")
    a5 = np.nanmedian(cols["ks_D"])
    q25, q75 = np.nanpercentile(cols["ks_D"], [25, 75])
    rep.append(f"- A5 per-room KS D (chi2 variant): median **{a5:.3f}** IQR [{q25:.3f},{q75:.3f}] "
               f"max {np.nanmax(cols['ks_D']):.3f} (paper 0.042 [0.031,0.058] max 0.12)")
    a5n = np.nanmedian(cols["ks_D_norm"])
    q25n, q75n = np.nanpercentile(cols["ks_D_norm"], [25, 75])
    okn = b4_mask & np.isfinite(cols["ks_D_norm"]) & np.isfinite(d1000)
    r3n, p3n = stats.spearmanr(cols["ks_D_norm"][okn], d1000[okn])
    rep.append(f"- A5b per-room KS D (signed vs N(0,1)): median **{a5n:.3f}** IQR [{q25n:.3f},{q75n:.3f}] "
               f"max {np.nanmax(cols['ks_D_norm']):.3f}; rho(D_norm, delta1000)={r3n:+.2f} "
               f"(p={p3n:.3f}, n={okn.sum()})")
    rep.append(f"- A6 H median (in-scope): **{np.nanmedian(cols['H'][in_scope]):.4f}** (paper ~0.005); "
               f"E_op median: **{np.nanmedian(cols['E_op']):.2f}** (paper 0.58)\n")

    # ---------------- head-to-head Spearman table ----------------
    diagnostics = [
        ("H (Herfindahl)", cols["H"]),
        ("||E||_op", cols["E_op"]),
        ("Berry KS D", cols["ks_D"]),
        ("dyn range (lamK/lam1)^s", cols["dyn_range"]),
        ("K_total", cols["K_total"]),
        ("area", cols["area"]),
    ]
    T_report = [1, 50, 100, 500, 1000, 2100]
    rep.append("## Head-to-head: Spearman rho(diagnostic, delta_rel(T)), in-scope rooms (n=187)\n")
    hdr = "| diagnostic | " + " | ".join(f"T={t}" for t in T_report) + " |"
    rep.append(hdr)
    rep.append("|" + "---|" * (len(T_report) + 1))
    rho_table = {}
    for name, x in diagnostics:
        row = [name]
        rho_table[name] = {}
        for t in T_report:
            ti = list(T_values).index(t)
            y = delta_rel[ti]
            ok = in_scope & np.isfinite(x) & np.isfinite(y)
            r, p = stats.spearmanr(x[ok], y[ok])
            star = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            row.append(f"{star}{r:+.2f}{star}")
            rho_table[name][t] = (r, p, int(ok.sum()))
        rep.append("| " + " | ".join(row) + " |")
    rep.append("\n(** p<0.01, * p<0.05)\n")

    # landscape-tier diagnostics (same table, separate tier: they use the sweep itself)
    rep.append("## Landscape-tier (uses the P(p) sweep): Spearman rho with delta_rel(T)\n")
    rep.append(hdr)
    rep.append("|" + "---|" * (len(T_report) + 1))
    for name, arr2 in [("flatness maxP/minP", flatness),
                       ("curvature P''(p*)/P(p*)", curv_norm),
                       ("drift |p*-|s||", drift)]:
        row = [name]
        for t in T_report:
            ti = list(T_values).index(t)
            y = delta_rel[ti]
            x = arr2[:, ti]
            ok = in_scope & np.isfinite(x) & np.isfinite(y)
            r, p = stats.spearmanr(x[ok], y[ok])
            star = "**" if p < 0.01 else ("*" if p < 0.05 else "")
            row.append(f"{star}{r:+.2f}{star}")
        rep.append("| " + " | ".join(row) + " |")

    # Taylor reconstruction check: does 0.5*curv*drift^2 reconstruct delta?
    rep.append("\n## Mechanistic decomposition: delta ~ 0.5 P''(p*) (p*-|s|)^2 / P(p*)\n")
    rep.append("| T | Spearman(taylor, delta) | Pearson log-log R2 | n |")
    rep.append("|---|---|---|---|")
    for t in T_report:
        ti = list(T_values).index(t)
        y = delta_rel[ti]
        x = taylor[:, ti]
        ok = in_scope & np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        r, p = stats.spearmanr(x[ok], y[ok])
        lr = stats.pearsonr(np.log(x[ok]), np.log(y[ok]))
        rep.append(f"| {t} | {r:+.2f} | {lr[0]**2:.2f} | {ok.sum()} |")

    # ---------------- partial correlations controlling for area ----------------
    rep.append("\n## Partial Spearman (controlling for room area), delta_rel(T=1000), in-scope\n")
    rep.append("| diagnostic | raw rho | partial rho | area |")
    rep.append("|---|---|---|---|")

    def partial_spearman(x, y, z):
        rx = stats.rankdata(x); ry = stats.rankdata(y); rz = stats.rankdata(z)
        ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
        ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
        return stats.pearsonr(ex, ey)

    for name, x in diagnostics[:-1]:  # skip area itself
        ok = in_scope & np.isfinite(x) & np.isfinite(d1000) & np.isfinite(cols["area"])
        raw, _ = stats.spearmanr(x[ok], d1000[ok])
        pr, pp = partial_spearman(x[ok], d1000[ok], cols["area"][ok])
        rep.append(f"| {name} | {raw:+.2f} | {pr:+.2f} (p={pp:.1e}) | controlled |")

    # ---------------- multivariate: rank OLS + random forest ----------------
    rep.append("\n## Multivariate (T=1000, in-scope, physics+noise diagnostics)\n")
    feat_names = ["H", "E_op", "ks_D", "dyn_range", "K_total", "area"]
    X = np.column_stack([cols[f] for f in feat_names])
    y = d1000.copy()
    ok = in_scope & np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    Xr = np.column_stack([stats.rankdata(X[ok, j]) for j in range(X.shape[1])])
    yr = stats.rankdata(y[ok])
    Xz = (Xr - Xr.mean(0)) / Xr.std(0)
    yz = (yr - yr.mean()) / yr.std()
    beta, res, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(yz)), Xz]), yz, rcond=None)
    pred = np.column_stack([np.ones(len(yz)), Xz]) @ beta
    r2_ols = 1 - np.sum((yz - pred) ** 2) / np.sum(yz ** 2)
    rep.append(f"Rank-OLS on standardized ranks, n={ok.sum()}, R2={r2_ols:.2f}. Betas:")
    for f, b in zip(feat_names, beta[1:]):
        rep.append(f"- {f}: {b:+.2f}")

    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import permutation_importance
        from sklearn.model_selection import cross_val_score
        rf = RandomForestRegressor(n_estimators=500, random_state=0, min_samples_leaf=5)
        rf.fit(X[ok], y[ok])
        r2_rf = rf.score(X[ok], y[ok])
        imp = permutation_importance(rf, X[ok], y[ok], n_repeats=50, random_state=0)
        cv = cross_val_score(RandomForestRegressor(n_estimators=500, random_state=0,
                                                   min_samples_leaf=5),
                             X[ok], y[ok], cv=5, scoring="r2")
        rep.append(f"\nRandom forest (as in D.3): in-sample R2={r2_rf:.2f}, "
                   f"5-fold CV R2={cv.mean():.2f}+-{cv.std():.2f}. Permutation importances:")
        order = np.argsort(-imp.importances_mean)
        for j in order:
            rep.append(f"- {feat_names[j]}: {imp.importances_mean[j]:.3f} "
                       f"+- {imp.importances_std[j]:.3f}")

        rep.append("\n### Out-of-sample predictability of delta_rel at every T (RF, 5-fold CV R2)\n")
        rep.append("| T | CV R2 (physics+noise features) | median delta_rel |")
        rep.append("|---|---|---|")
        for t in T_report:
            ti = list(T_values).index(t)
            yt = delta_rel[ti]
            okt = in_scope & np.all(np.isfinite(X), axis=1) & np.isfinite(yt)
            cvt = cross_val_score(RandomForestRegressor(n_estimators=500, random_state=0,
                                                        min_samples_leaf=5),
                                  X[okt], yt[okt], cv=5, scoring="r2")
            rep.append(f"| {t} | {cvt.mean():+.2f} +- {cvt.std():.2f} | "
                       f"{100*np.nanmedian(yt[in_scope]):.2f}% |")
    except ImportError:
        rep.append("\n(sklearn unavailable - RF skipped)")

    # factor collapse: correlations among diagnostics
    rep.append("\n## Diagnostic inter-correlations (Spearman, in-scope)\n")
    pairs = [("area", "K_total"), ("H", "K_total"), ("E_op", "K_total"),
             ("E_op", "H"), ("ks_D", "K_total"), ("dyn_range", "area")]
    for a, b in pairs:
        okp = in_scope & np.isfinite(cols[a]) & np.isfinite(cols[b])
        r, p = stats.spearmanr(cols[a][okp], cols[b][okp])
        rep.append(f"- rho({a}, {b}) = {r:+.2f}")

    # save assembled table
    np.savez(os.path.join(OUT, OUT_NAME + ".npz"),
             rooms=np.array(rooms), T_values=T_values, s_value=S_VAL,
             delta_rel=delta_rel, in_scope=in_scope, b4_mask=b4_mask,
             p_star_room=p_star_room, P_star_room=P_star_room,
             flatness=flatness, curv_norm=curv_norm, drift=drift, taylor=taylor,
             **{k: v for k, v in cols.items()})

    txt = "\n".join(rep) + "\n"
    with open(os.path.join(OUT, OUT_NAME + ".md"), "w") as f:
        f.write(txt)
    print(txt)


if __name__ == "__main__":
    main()
