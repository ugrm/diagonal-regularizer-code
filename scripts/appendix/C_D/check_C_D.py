#!/usr/bin/env python
"""
Recompute every number printed in Tables T7-T15, T18, T30, T31, T33, T34, T37-T39 and the
numeric claims in the main-text prose and in the prose of Appendices C, D, H and I, from the
shipped results in data/ and the paper's dataset in data/modal.

Prints one line per check: [table, row, printed, recomputed, status]. Status is
  OK           recomputed from shipped data and matches the printed value
  OK(stored)   matches a stored output of a run whose inputs are not shipped (trained
               model weights); the stored output is what the paper printed
  NOT CHECKED  no shipped data can reproduce the value (reason given)
  DIFF(known)  a printed value the data do not support, listed in KNOWN (exit code unaffected)
  DIFF         recomputed value does not match (exit code 1)

Usage (from the repository root, CPU, about a minute):
    python scripts/appendix/C_D/check_C_D.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from src.utils.paths import DATA, EXP, MODAL, SUPP, check_paper_dataset  # noqa: E402

K, M = 50, 8
ROWS = []

# Printed values the data do not support. They print as DIFF(known) and do not set the exit code.
KNOWN = {}


def add(table, label, printed, recomputed, ok, stored=False, key=None):
    if ok:
        status = "OK(stored)" if stored else "OK"
    else:
        status = "DIFF(known)" if key in KNOWN else "DIFF"
    ROWS.append((table, label, str(printed), str(recomputed), status))


def not_checked(table, label, printed, reason):
    ROWS.append((table, label, str(printed), reason, "NOT CHECKED"))


def agree(printed, value, decimals=None):
    """True if `value` rounds to `printed` (half-unit tolerance at the printed precision)."""
    printed = str(printed)
    if decimals is None:
        decimals = len(printed.split(".")[1]) if "." in printed else 0
    return abs(float(value) - float(printed)) <= 0.5 * 10.0 ** (-decimals) + 1e-9


def f(x, d):
    return f"{x:.{d}f}"


# ----------------------------------------------------------------------------- shared data
def load_common():
    ps = np.load(EXP / "p_sweep" / "p_sweep_K50_M8.npz", allow_pickle=True)
    cost = np.load(EXP / "cost" / "cost_K50_M8.npz", allow_pickle=True)
    ids = [str(r) for r in ps["room_ids"]]
    meta = {r: json.load(open(MODAL / r / "metadata.json")) for r in ids}
    Kt = np.array([meta[r]["K"] for r in ids])
    s = float(cost["s_value"])
    P = ps["P_oracle"][:, :, list(ps["M_values"]).index(M), :]          # rooms x T x p
    pg, T = ps["p_values"], [int(t) for t in ps["T_values"]]
    Ps = np.array([[np.interp(s, pg, P[r, t]) for t in range(len(T))] for r in range(len(ids))])
    Pst = P.min(-1)
    return dict(ps=ps, cost=cost, ids=ids, meta=meta, Kt=Kt, ins=Kt > K, s=s, P=P, pg=pg, T=T,
                Ps=Ps, Pst=Pst, pstar=pg[P.argmin(-1)], gap=(Ps - Pst) * 100)


# ----------------------------------------------------------------------------- T7
def check_T7(c):
    noise = np.load(EXP / "noise_profile" / "noise_profile_K50_M8.npz", allow_pickle=True)
    sr = np.load(EXP / "sroom_vs_pop" / "sroom_per_room.npz", allow_pickle=True)
    s = -noise["s_per_room"]
    add("T7", "median |s_hat| (187 rooms)", "1.13", f(np.median(s), 4), agree("1.13", np.median(s)))
    add("T7", "mean |s_hat|", "1.12", f(s.mean(), 4), agree("1.12", s.mean()))
    add("T7", "std across rooms of |s_hat|", "0.25", f(s.std(), 4), agree("0.25", s.std()))
    se_room = np.median((sr["s_room_ci_hi"] - sr["s_room_ci_lo"]) / (2 * 1.96))
    add("T7", "typical per-room bootstrap SE", "0.27", f(se_room, 4), agree("0.27", se_room))
    dec = s.var() - se_room ** 2
    add("T7", "inter-room std after deconvolving the per-room SE", "~0",
        f"var {s.var():.4f} - SE^2 {se_room ** 2:.4f} = {dec:.4f} (< 0)", dec <= 0)
    rng = np.random.default_rng(42)
    meds = np.array([np.median(rng.choice(s, len(s))) for _ in range(10000)])
    lo, hi = np.percentile(meds, [2.5, 97.5])
    add("T7", "bootstrap SE of the median", "0.03", f(meds.std(), 4), agree("0.03", meds.std()))
    add("T7", "bootstrap 95% CI on the median", "[1.08, 1.18]", f"[{lo:.3f}, {hi:.3f}]",
        agree("1.08", lo) and agree("1.18", hi))
    srs = np.abs(sr["s_room"])
    ins = np.array([str(r) in set(map(str, noise["room_ids"])) for r in sr["room_ids"]])
    add("T7", "median of per-room fits, 197 rooms (187)", "1.1243 (1.1266)",
        f"{np.median(srs):.4f} ({np.median(srs[ins]):.4f})",
        agree("1.1243", np.median(srs)) and agree("1.1266", np.median(srs[ins])))


# ----------------------------------------------------------------------------- T8
T8 = {1: ("0.5", "0.2", "1.9", "4.8", "15.0"), 5: ("0.5", "0.1", "1.6", "5.6", "15.8"),
      10: ("0.5", "0.1", "1.5", "4.9", "16.2"), 20: ("0.5", "0.1", "1.4", "3.9", "20.2"),
      50: ("0.6", "0.2", "1.5", "4.5", "20.5"), 100: ("1.5", "0.6", "2.8", "6.2", "27.5"),
      200: ("2.2", "0.7", "4.1", "7.9", "14.8"), 500: ("5.1", "2.7", "7.0", "12.3", "26.8"),
      1000: ("5.6", "2.8", "10.3", "16.9", "34.8"), 2100: ("3.3", "1.4", "5.7", "10.5", "44.8")}


def check_T8(c):
    dr = c["cost"]["per_room_delta_rel"] * 100
    Tc = [int(t) for t in c["cost"]["T"]]
    rec = (c["Ps"] - c["Pst"]) / c["Pst"] * 100
    add("T8", "cost npz equals recomputation from the p-sweep", "-",
        f"max |diff| {np.abs(rec.T - dr).max():.1e} pp", np.abs(rec.T - dr).max() < 1e-9)
    for T, pr in T8.items():
        x = dr[Tc.index(T)]
        got = (np.median(x), np.percentile(x, 25), np.percentile(x, 75), np.percentile(x, 95), x.max())
        add("T8", f"T={T}: median / IQR / 95th / worst (%)", " / ".join(pr),
            " / ".join(f(g, 2) for g in got), all(agree(p, g) for p, g in zip(pr, got)))
    v = np.median(dr[Tc.index(1000), c["ins"]])
    add("T8", "dagger: T=1000 over the 187 in-scope rooms", "5.83%", f(v, 3), agree("5.83", v))


# ----------------------------------------------------------------------------- T9
T9 = {1: ("0.40", "0.12", "1.42", "0.715"), 5: ("0.35", "0.06", "1.07", "0.715"),
      10: ("0.34", "0.10", "1.03", "0.703"), 20: ("0.36", "0.08", "0.98", "0.679"),
      50: ("0.34", "0.13", "0.88", "0.638"), 100: ("0.89", "0.34", "1.53", "0.594"),
      200: ("0.91", "0.33", "1.87", "0.461"), 500: ("1.04", "0.56", "1.65", "0.220"),
      1000: ("0.68", "0.35", "1.34", "0.122"), 2100: ("0.38", "0.16", "1.21", "0.152")}


def check_T9(c):
    g, Pst, T = c["gap"], c["Pst"], c["T"]
    for Tv, pr in T9.items():
        t = T.index(Tv)
        got = (np.median(g[:, t]), np.percentile(g[:, t], 25), np.percentile(g[:, t], 75), np.median(Pst[:, t]))
        add("T9", f"T={Tv}: median gap / IQR (pp) / median P(p*)", " / ".join(pr),
            " / ".join(f(x, 4) for x in got), all(agree(p, x) for p, x in zip(pr, got)))
    mg, mf = np.median(g, 0), np.median(Pst, 0)
    add("T9", "bold: gap peaks / floor lowest", "T=500 / T=1000",
        f"T={T[int(np.argmax(mg))]} / T={T[int(np.argmin(mf))]}",
        T[int(np.argmax(mg))] == 500 and T[int(np.argmin(mf))] == 1000)


# ----------------------------------------------------------------------------- T10, T11
T10 = {"spectral_gap": ("-0.15", "-0.30", "0.00", "0.03"), "weyl_exponent": ("+0.07", "-0.09", "0.23", "0.33"),
       "s_room": ("-0.04", "-0.18", "0.11", "0.58"), "lambda_K": ("+0.11", "-0.04", "0.25", "0.13")}
T11 = {"room_area": ("+0.21", "0.05", "0.37", "0.003"), "K_total": ("+0.21", "0.05", "0.37", "0.003"),
       "mean_spacing": ("-0.21", "-0.37", "-0.05", "0.003"), "eigenvalue_density": ("+0.21", "0.05", "0.36", "0.003"),
       "n_segments": ("+0.14", "0.00", "0.29", "0.05")}


def check_T10_T11(c):
    g = np.load(SUPP / "geometric_predictors.npz", allow_pickle=True)
    names = [str(n) for n in g["feature_names"]]

    def find(key):
        hits = [i for i, n in enumerate(names) if n.lower().replace(" ", "_").startswith(key.lower())]
        return hits[0] if hits else None

    for tab, rows in (("T10", T10), ("T11", T11)):
        for key, pr in rows.items():
            i = find(key)
            if i is None:
                add(tab, f"{key}: rho / CI / p", " / ".join(pr), f"feature not found in {names}", False)
                continue
            got = (g["rho"][i], g["rho_ci_lo"][i], g["rho_ci_hi"][i], g["rho_pval"][i])
            add(tab, f"{names[i]}: rho / 95% CI / p", " / ".join(pr), " / ".join(f(x, 4) for x in got),
                all(agree(p, x) for p, x in zip(pr, got)))
    add("T11", "RF cross-validated R^2 for p*(T=1000)", "0.14", f(float(g["rf_r2_mean"]), 4),
        agree("0.14", g["rf_r2_mean"]))
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    rf = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_leaf=10, random_state=42)
    cv50 = cross_val_score(rf, g["X"], g["p_star_secondary"], cv=5, scoring="r2").mean()
    add("T11/D.3", "RF at T=50 does worse than the mean", "R^2 < 0", f(cv50, 3), cv50 < 0)
    add("D.3", "regression rooms (one excluded)", "196", str(len(g["room_ids"])), len(g["room_ids"]) == 196)


# ----------------------------------------------------------------------------- T12
def check_T12(c):
    T, P, pg, Pst, ins, Kt = c["T"], c["P"], c["pg"], c["Pst"], c["ins"], c["Kt"]
    t1000 = T.index(1000)
    b1, b0 = np.polyfit(Kt[ins], c["pstar"][ins, t1000], 1)
    padj = np.clip(b0 + b1 * Kt, pg[0], pg[-1])
    rel = lambda pv, t: np.array([(np.interp(pv[r], pg, P[r, t]) - Pst[r, t]) / Pst[r, t]
                                  for r in range(len(pv))]) * 100
    s_vec = np.full(len(Kt), c["s"])
    pr = {1: ("0.57", "2.9"), 50: ("0.61", "4.1"), 100: ("1.46", "5.0"), 1000: ("5.83", "4.8")}
    out = {}
    for Tv, (a, b) in pr.items():
        t = T.index(Tv)
        da, db = np.median(rel(s_vec, t)[ins]), np.median(rel(padj, t)[ins])
        out[Tv] = (da, db)
        add("T12", f"T={Tv}: delta(|s|) / delta(p_adj), 187 rooms (%)", f"{a} / {b}", f"{da:.3f} / {db:.3f}",
            agree(a, da) and agree(b, db))
    add("D.3", "p_adj trims the T=1000 cost by about a point, raises T<=100 by several points",
        "T=1000 gain ~1 pp; T<=100 rise 2-4 pp",
        f"T=1000 {out[1000][0] - out[1000][1]:+.2f} pp; T<=100 "
        + ", ".join(f"{out[t][1] - out[t][0]:+.2f}" for t in (1, 50, 100)),
        0.5 < out[1000][0] - out[1000][1] < 1.5 and all(2 <= out[t][1] - out[t][0] <= 4 for t in (1, 50, 100)))


# ----------------------------------------------------------------------------- T14
T14 = {1: ("+0.05", "0.45", "<1%", "0.55", "0.71"), 50: ("+0.40", "7e-9", "16%", "0.60", "0.88"),
       100: ("+0.40", "7e-9", "16%", "1.46", "1.37"), 500: ("+0.14", "0.043", "2%", "5.13", "4.99"),
       1000: ("+0.16", "0.026", "3%", "5.62", "5.84"), 2100: ("+0.18", "0.010", "3%", "3.33", "3.59")}


def check_T14(c):
    d = np.load(EXP / "sroom_vs_pop" / "sroom_per_room.npz", allow_pickle=True)
    s, po, Ts = d["s_room"], d["p_oracle"], [int(t) for t in d["T_values"]]
    keep = np.array([str(r) not in set(np.array(c["ids"])[~c["ins"]]) for r in d["room_ids"]])
    dp, ds = d["delta_pop"] * 100, d["delta_sroom"] * 100
    max_drho = max_dd = 0.0
    for Tv, (rho_p, p_p, r2_p, dpop_p, dsr_p) in T14.items():
        i = Ts.index(Tv)
        r, p = stats.spearmanr(s, po[:, i])
        r2 = 100 * r * r
        ok_p = (abs(p - float(p_p)) / float(p_p) < 0.15) if "e" in p_p else agree(p_p, p)
        ok_r2 = None
        ok_r2 = r2 < 1 if r2_p == "<1%" else round(r2) == int(r2_p.rstrip("%"))
        a, b = np.median(dp[:, i]), np.median(ds[:, i])
        add("T14", f"T={Tv}: rho / p / R^2 / delta(|s|) / delta(s_room)",
            " / ".join((rho_p, p_p, r2_p, dpop_p + "%", dsr_p + "%")),
            f"{r:+.3f} / {p:.2g} / {r2:.1f}% / {a:.3f}% / {b:.3f}%",
            agree(rho_p, r) and ok_p and ok_r2 and agree(dpop_p, a) and agree(dsr_p, b), key=f"T14 T={Tv}")
        r_in, _ = stats.spearmanr(s[keep], po[keep, i])
        max_drho = max(max_drho, abs(r - r_in))
        max_dd = max(max_dd, abs(a - np.median(dp[keep, i])), abs(b - np.median(ds[keep, i])))
    i = Ts.index(1000)
    wins = int(np.sum(ds[:, i] < dp[:, i]))
    add("T14", "rooms that benefit from per-room tuning at T=1000", "88 of 197", f"{wins} of {len(s)}",
        wins == 88 and len(s) == 197)
    r_in, p_in = stats.spearmanr(s[keep], po[keep, i])
    add("T14", "excluding boundary rooms: max change in rho / in median delta; rho, p at T=1000",
        "<= 0.03 / <= 0.22 pp; 0.14, 0.065", f"{max_drho:.3f} / {max_dd:.3f} pp; {r_in:.3f}, {p_in:.3f}",
        max_drho <= 0.035 and max_dd <= 0.225 and agree("0.14", r_in) and agree("0.065", p_in))
    add("D.5", "fewer than half of the rooms benefit at T=1000", "< half", f"{wins}/197", wins < 197 / 2)


# ----------------------------------------------------------------------------- T15
def check_T15(c):
    lir = np.load(EXP / "lir" / "lir_summary.npz")
    m3 = np.load(EXP / "m3_performance.npz")
    Pm = lir["P_median"]                                         # L x T x seed
    Ls, Ts = [int(x) for x in lir["L_values"]], [int(x) for x in lir["T_values"]]
    printed = {1: ("0.908", "0.001", "0.783", "0.000", "0.368", "0.002"),
               5: ("0.726", "0.002", "0.624", "0.002", "0.147", "0.000"),
               10: ("0.720", "0.001", "0.609", "0.003", "0.141", "0.001"),
               20: ("0.727", "0.004", "0.607", "0.002", "0.140", "0.001")}
    for L, pr in printed.items():
        li = Ls.index(L)
        got = [v for t in range(3) for v in (Pm[li, t].mean(), Pm[li, t].std())]
        add("T15", f"L={L}: mean +- sd over seeds, T=1/100/1000", " / ".join(pr), " / ".join(f(g, 4) for g in got),
            all(agree(p, g) for p, g in zip(pr, got)))
    ti = [list(m3["T_values"]).index(t) for t in Ts]
    for lab, key, pr in (("closed form", "P_ridge_s", ("0.726", "0.603", "0.131")),
                         ("oracle", "P_oracle", ("0.715", "0.594", "0.122"))):
        got = m3[key][ti]
        add("T15", f"{lab}, T=1/100/1000", " / ".join(pr), " / ".join(f(g, 4) for g in got),
            all(agree(p, g) for p, g in zip(pr, got)))
    loss = {}
    for L in (10, 20):
        v = [np.load(p)["loss"][-1] for p in sorted((EXP / "lir" / "checkpoints").glob(f"lir_L{L}_T1_s*/trajectory.npz"))]
        loss[L] = (np.mean(v), len(v))
    add("T15", "training loss at T=1, mean over seeds: L=10 / L=20", "0.709 / 0.702",
        f"{loss[10][0]:.4f} / {loss[20][0]:.4f} (n={loss[10][1]}, {loss[20][1]})",
        agree("0.709", loss[10][0]) and agree("0.702", loss[20][0]) and loss[10][1] == loss[20][1] == 5)
    li10, li20 = Ls.index(10), Ls.index(20)
    add("D.8", "at T=1, L=20 has lower training loss but higher validation P than L=10", "yes",
        f"loss {loss[20][0]:.4f} < {loss[10][0]:.4f}; P {Pm[li20, 0].mean():.4f} > {Pm[li10, 0].mean():.4f}",
        loss[20][0] < loss[10][0] and Pm[li20, 0].mean() > Pm[li10, 0].mean())
    same = all(abs(Pm[li20, t].mean() - Pm[li10, t].mean()) <= Pm[li10, t].std() + Pm[li20, t].std() for t in (1, 2))
    add("D.8", "at T=100 and T=1000 L=10 and L=20 agree within the seed spread", "yes",
        ", ".join(f"T={Ts[t]}: {Pm[li20, t].mean() - Pm[li10, t].mean():+.4f}" for t in (1, 2)), same)
    seeds = [int(x) for x in lir["seeds"]]
    add("D.8", "5 seeds 42-46", "42..46", str(seeds), seeds == [42, 43, 44, 45, 46])


# ----------------------------------------------------------------------------- T18, T39
def check_T18_T39(c):
    from sklearn.ensemble import RandomForestRegressor  # noqa: F401  (import check)
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import cross_val_score
    d = np.load(EXP / "appendix" / "room_diagnostic_regression.npz", allow_pickle=True)
    rooms = [str(r) for r in d["rooms"]]
    assert rooms == c["ids"], "room order differs from the p-sweep"
    ins = d["in_scope"].astype(bool)
    T, P, pg = c["T"], c["P"], c["pg"]
    dr, da = c["cost"]["per_room_delta_rel"], c["cost"]["per_room_delta_P"]
    js = int(np.searchsorted(pg, 1.1 - 1e-9))
    m3 = pg <= 3 + 1e-9
    pr18 = {1: (("0.00", None), ("+0.07", "0.32")), 100: (("-0.17", "0.02"), ("-0.04", "0.55")),
            1000: (("-0.04", "0.61"), ("-0.64", "<1e-4"))}
    for Tv, ((cr, cp), (br, bp)) in pr18.items():
        t = T.index(Tv)
        cur = P[:, t, :]
        curv = (cur[:, js - 1] - 2 * cur[:, js] + cur[:, js + 1]) / 0.01
        basin = np.sum(cur[:, m3] <= 1.05 * np.nanmin(cur, 1)[:, None], 1) * 0.1
        r1, p1 = stats.spearmanr(curv[ins], dr[t, ins])
        r2, p2 = stats.spearmanr(basin[ins], dr[t, ins])
        ok1 = agree(cr, r1) and (cp is None or agree(cp, p1))
        ok2 = agree(br, r2) and (p2 < 1e-4 if bp == "<1e-4" else agree(bp, p2))
        add("T18", f"T={Tv}: curvature at p=|s| rho (p)", f"{cr} ({cp})" if cp else "~0", f"{r1:+.3f} ({p1:.3f})", ok1)
        add("T18", f"T={Tv}: basin width on [0,3] rho (p)", f"{br} ({bp})", f"{r2:+.3f} ({p2:.2g})", ok2)
    names = ["H", "K_total", "area", "E_op", "ks_D", "dyn_range"]
    X = np.column_stack([d[n] for n in names])[ins]
    Xs = (X - X.mean(0)) / X.std(0)
    t = T.index(1000)
    for lab, y, pr in (("relative", dr[t, ins], "0.18"), ("absolute", da[t, ins], "0.25")):
        r2 = cross_val_score(Ridge(alpha=1.0), Xs, y, cv=5, scoring="r2").mean()
        add("T18", f"T=1000: six descriptors, ridge 5-fold CV R^2 ({lab} gap)", pr, f(r2, 3), agree(pr, r2))
    # T39 and the T18 'no out-of-sample prediction' rows: RF CV R^2 at T <= 100 (as in the regression report)
    from sklearn.ensemble import RandomForestRegressor
    feat = ["H", "E_op", "ks_D", "dyn_range", "K_total", "area"]
    Xf = np.column_stack([d[n] for n in feat])
    cvs = {}
    for Tv in (1, 50, 100):
        ti = T.index(Tv)
        y = d["delta_rel"][ti]
        ok = ins & np.all(np.isfinite(Xf), 1) & np.isfinite(y)
        cvs[Tv] = cross_val_score(RandomForestRegressor(n_estimators=500, random_state=0, min_samples_leaf=5),
                                  Xf[ok], y[ok], cv=5, scoring="r2").mean()
    add("T18/T39", "RF CV R^2 of the gap from all descriptors, T=1/50/100", "<= 0",
        " / ".join(f"{cvs[t]:+.3f}" for t in (1, 50, 100)), all(v <= 0 for v in cvs.values()))

    def partial(x, y, z):
        rx, ry, rz = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)
        ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
        ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
        return stats.pearsonr(ex, ey)

    d1000 = d["delta_rel"][t]
    rows = (("H", "-0.42", "-0.08"), ("K_total", "+0.42", None), ("E_op", "-0.30", "+0.10"),
            ("ks_D", "+0.16", "+0.05"), ("dyn_range", "-0.04", "-0.03"))
    pmin = 1.0
    for n, raw_p, par_p in rows:
        x = d[n]
        ok = ins & np.isfinite(x) & np.isfinite(d1000)
        raw = stats.spearmanr(x[ok], d1000[ok])[0]
        par, pp = partial(x[ok], d1000[ok], d["area"][ok])
        pmin = min(pmin, pp)
        add("T39", f"{n}: raw rho / partial rho given area", f"{raw_p} / {par_p or '--'}", f"{raw:+.3f} / {par:+.3f}",
            agree(raw_p, raw) and (par_p is None or agree(par_p, par)))
    add("T39", "all partial correlations have p >= 0.16", ">= 0.16", f"min p {pmin:.3f}", pmin >= 0.155)
    a = stats.spearmanr(d["area"][ins], d["K_total"][ins])[0]
    h = stats.spearmanr(d["H"][ins], d["K_total"][ins])[0]
    add("T39", "rho(area, K_total) / rho(H, K_total)", "+1.00 / -1.00", f"{a:+.3f} / {h:+.3f}",
        agree("1.00", a) and agree("-1.00", h))
    basin = np.sum(P[:, t, m3] <= 1.05 * np.nanmin(P[:, t, :], 1)[:, None], 1) * 0.1
    rb, pb = stats.spearmanr(basin[ins], dr[t, ins])
    add("T39", "basin width on [0,3] (p)", "-0.64 (< 1e-4)", f"{rb:+.3f} ({pb:.1e})", agree("-0.64", rb) and pb < 1e-4)
    r2s = []
    for Tv in (1, 50, 100):
        ti = T.index(Tv)
        x, y = d["taylor"][:, ti], d["delta_rel"][ti]
        ok = ins & np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        r2s.append(stats.pearsonr(np.log(x[ok]), np.log(y[ok]))[0] ** 2)
    add("T39", "drift-times-curvature log-log R^2 at T=1/50/100", "0.87 / 0.78 / 0.75",
        " / ".join(f(v, 3) for v in r2s), all(agree(p, v) for p, v in zip(("0.87", "0.78", "0.75"), r2s)))
    ks_elig = stats.spearmanr(d["ks_D"][d["b4_mask"].astype(bool)], d1000[d["b4_mask"].astype(bool)])[0]
    add("T39", "D_KS on the eligible subset", "+0.19", f"{ks_elig:+.3f}", agree("0.19", ks_elig))


# ----------------------------------------------------------------------------- T30
def check_T30(c):
    sys.path.insert(0, str(REPO / "scripts" / "appendix" / "I"))
    import aperture_rooms as ap
    pr = {"Large": ("60.9", "7017", "2.07", "6.1", "37.9"), "Compact": ("6.9", "879", "0.988", "12.7", "78.0"),
          "Closet": ("1.0", "141", "0.535", "23.6", "134.9")}
    for name, (Lx, Ly, Lz) in ap.ROOMS.items():
        lam = ap.neumann_eigenvalues(Lx, Ly, Lz)
        lmin = 2 * np.pi / np.sqrt(lam[ap.K - 1])
        ratio = ap.D_AP / lmin
        got = (Lx * Ly * Lz, len(lam), lmin, 100 * ratio, 100 * 2 * np.sin(np.pi * ratio))
        add("T30", f"{name}: V / K_total / l_min / D_ap/l_min / max variation", " / ".join(pr[name]),
            " / ".join(f(g, 4) for g in got), all(agree(p, g) for p, g in zip(pr[name], got)))
    k = len(ap.neumann_eigenvalues(*ap.ROOMS["Closet"]))
    add("I.2", "closet has fewer than three times the K=50 retained modes", "< 150", str(k), k < 3 * 50)
    add("I.2", "aperture D_ap", "12.6 cm", f"{100 * ap.D_AP:.1f} cm", agree("12.6", 100 * ap.D_AP))


# ----------------------------------------------------------------------------- T31
T31 = {1: ("0.65 1.03 0.42 0.70 0.82", "0.76 0.79 0.63 0.65 0.91"),
       100: ("0.94 1.01 0.96 0.99 0.99", "0.80 0.82 0.83 0.83 0.78"),
       1000: ("-0.22 0.24 0.50 0.60 1.09", "0.20 0.08 0.21 0.23 0.73")}


def check_T31(c):
    for T, (ph, r2) in T31.items():
        got_p, got_r = [], []
        for seed in range(42, 47):
            tr = np.load(REPO / "checkpoints" / f"m3_hypernet_n800_T{T}_s{seed}" / "trajectory.npz")
            got_p.append(float(tr["phat_final"]))
            got_r.append(float(tr["r2_final"]))
        add("T31", f"T={T}: p_hat, seeds 42-46", ph, " ".join(f(v, 3) for v in got_p),
            all(agree(p, v) for p, v in zip(ph.split(), got_p)), stored=True)
        add("T31", f"T={T}: R^2, seeds 42-46", r2, " ".join(f(v, 3) for v in got_r),
            all(agree(p, v) for p, v in zip(r2.split(), got_r)), stored=True)


# ----------------------------------------------------------------------------- T33
def check_T33(c):
    T, ins = c["T"], c["ins"]
    pr = {1: ("1.2", "0.85"), 50: ("1.2", "0.26"), 100: ("1.3", "0.30"), 500: ("1.6", "0.42"),
          1000: ("2.0", "0.99"), 2100: ("0.6", "1.67")}
    for Tv, (m, sd) in pr.items():
        x = c["pstar"][ins, T.index(Tv)]
        add("T33", f"T={Tv}: p* median / std (in-scope)", f"{m} / {sd}", f"{np.median(x):.2f} / {x.std():.4f}",
            agree(m, np.median(x)) and agree(sd, x.std()))
    g = c["gap"][:, T.index(1000)]
    frac = 100 * np.mean(g[ins] < 1.1)
    add("T33", "rooms below 1.1 pp at T=1000 (in-scope)", "69.0%", f(frac, 2), agree("69.0", frac))
    w = c["gap"][ins].max()
    add("T33", "worst in-scope absolute cost over all T", "7.61 pp", f(w, 3), agree("7.61", w))


# ----------------------------------------------------------------------------- T34
def check_T34(c):
    m3 = np.load(DATA / "results" / "method3_results.npz", allow_pickle=True)
    m2 = np.load(DATA / "results" / "method2_results.npz", allow_pickle=True)
    ti = list(m3["T_extract"]).index(1000)
    keep = np.isin(np.array([str(r) for r in m3["room_ids"]]), np.array(c["ids"])[c["ins"]])
    x = m3["p_hat"][keep, ti]
    q = (np.median(x), np.percentile(x, 25), np.percentile(x, 75))
    add("T34", "M3 per-room p_hat median [IQR], 187 rooms", "1.13 [0.66, 1.39]",
        f"{q[0]:.3f} [{q[1]:.3f}, {q[2]:.3f}] (n={keep.sum()})",
        keep.sum() == 187 and all(agree(p, v) for p, v in zip(("1.13", "0.66", "1.39"), q)), stored=True)
    add("T34", "M2 shared exponent p_hat", "0.62", f(float(m2["p_hat"]), 4), agree("0.62", m2["p_hat"]), stored=True)
    add("T34", "R^2 of a power law fitted to the M2 spectrum", "0.94", f(float(m2["p_hat_r2"]), 4),
        agree("0.94", m2["p_hat_r2"]), stored=True)
    t = c["T"].index(1000)
    med = np.median(c["P"][:, t, :], 0)
    i06, i11 = int(np.argmin(abs(c["pg"] - 0.6))), int(np.argmin(abs(c["pg"] - 1.1)))
    dP = med[i06] - med[i11]
    add("T34", "difference in median P between the two exponents (grid p=0.6 vs 1.1, T=1000)", "0.009",
        f(dP, 4), agree("0.009", dP))
    cap = np.load(SUPP / "capacity_1000ep_results.npz")
    v = float(cap["m2_phat_final_p2_5"])
    add("T34", "largest planted exponent M2 recovers", "0.86", f(v, 4), agree("0.86", v), stored=True)
    ev = np.load(EXP / "sweep_P_eval.npz", allow_pickle=True)
    dPs = ev["P_end2end"] - ev["P_oracle_baseline"][None, None, :, None]
    add("T34", "smallest per-seed dP against the per-room oracle", "+0.002", f"{np.nanmin(dPs):+.5f}",
        agree("0.002", np.nanmin(dPs)), stored=True)
    nan = np.isnan(ev["P_end2end"])
    models = [str(m) for m in ev["models"]]
    nvals = [int(n) for n in ev["n_values"]]
    Tl = [int(x) for x in ev["T_values"]]
    m3i = [i for i, m in enumerate(models) if m.lower().startswith("m3")][0]
    only_m3 = nan.sum() == nan[m3i].sum()
    small_n = all(nvals[j] <= 400 for j in np.where(nan[m3i].any(axis=(1, 2)))[0])
    n1000 = int(nan[:, :, Tl.index(1000)].sum())
    add("T34", "training configurations / failures (all M3, n <= 400) / failures at T=1000", "225 / 13 / 10",
        f"{nan.size} / {int(nan.sum())} (only M3: {only_m3}, n<=400: {small_n}) / {n1000}",
        nan.size == 225 and nan.sum() == 13 and only_m3 and small_n and n1000 == 10, stored=True)


# ----------------------------------------------------------------------------- T37
def check_T37(c):
    T, P, pg, ids, ins = c["T"], c["P"], c["pg"], c["ids"], c["ins"]
    noise = np.load(EXP / "noise_profile" / "noise_profile_K50_M8.npz", allow_pickle=True)
    va = np.load(EXP / "size_ablation" / "val_ablation.npz")
    s_full, s50 = c["s"], float(va["median_s"][list(va["n_val_rooms"]).index(50)])
    dev50 = 100 * abs(s50 - s_full) / s_full
    s_all = -noise["s_per_room"]
    rng = np.random.default_rng(0)
    meds = np.array([np.median(rng.choice(s_all, 50, replace=False)) for _ in range(1000)])
    dev = 100 * np.abs(meds - s_full) / s_full
    add("T37", "|s| from 50 rooms: first 50 / worst of 1000 random subsets (95th pct)", "6% / 9.8% (6.6%)",
        f"{dev50:.2f}% / {dev.max():.2f}% ({np.percentile(dev, 95):.2f}%)",
        agree("6", dev50) and abs(dev.max() - 9.8) < 0.5 and abs(np.percentile(dev, 95) - 6.6) < 0.3)
    A = np.array([[np.interp(s50, pg, P[r, t]) for t in range(len(T))] for r in range(len(ids))])
    rise = 100 * np.median(A - c["Ps"], 0)
    add("T37", "estimate from the first 50 rooms (|s| = 1.06): largest median per-room rise in P over T", "0.38 pp",
        f"{rise.max():.4f} pp (median over rooms of the per-room rise; |s| = {s50:.4f})",
        agree("0.38", rise.max()) and agree("1.06", s50), key="T37 subset rise")
    add("C.1/H", "subset estimate shifts the exponent by up to about a tenth / costs less than half a point",
        "<= ~0.1 / < 0.5 pp", f"{dev.max() / 100 * s_full:.3f} / {rise.max():.3f} pp",
        dev.max() / 100 * s_full <= 0.12 and rise.max() < 0.5)
    i = ids.index("scene_00806")
    t = T.index(100)
    got = (c["Ps"][i, t], c["Pst"][i, t], c["gap"][i, t], 100 * c["Pst"][i, t] / c["Ps"][i, t])
    r_, t_ = np.unravel_index(np.argmax(c["gap"]), c["gap"].shape)
    add("T37", "worst absolute cost of all rooms (00806, K_total=19, T=100): P(|s|) / P(p*) / dP / floor",
        "0.491 / 0.385 / 10.6 pp / 78.4%", " / ".join(f(g, 4) for g in got) + f" [argmax: {ids[r_]}, T={T[t_]}]",
        all(agree(p, g) for p, g in zip(("0.491", "0.385", "10.6", "78.4"), got))
        and ids[r_] == "scene_00806" and T[t_] == 100 and c["Kt"][i] == 19)
    gi = c["gap"].copy()
    gi[~ins] = -np.inf
    r_, t_ = np.unravel_index(np.argmax(gi), gi.shape)
    add("T37", "worst in-scope absolute cost (00931, K_total=89, T=5)", "7.61 pp",
        f"{gi[r_, t_]:.3f} pp ({ids[r_]}, K_total={c['Kt'][r_]}, T={T[t_]})",
        agree("7.61", gi[r_, t_]) and ids[r_] == "scene_00931" and c["Kt"][r_] == 89 and T[t_] == 5)
    t = T.index(1000)
    g = np.where(ins, c["gap"][:, t], -np.inf)
    j = int(np.argmax(g))
    fl = 100 * c["Pst"][j, t] / c["Ps"][j, t]
    add("T37", "worst in-scope gap at T=1000 (00890, K_total=62): dP / floor", "4.84 pp / 95.0%",
        f"{g[j]:.3f} / {fl:.2f}% ({ids[j]}, K_total={c['Kt'][j]})",
        agree("4.84", g[j]) and agree("95.0", fl) and ids[j] == "scene_00890" and c["Kt"][j] == 62)
    a, b = 100 * np.mean(c["gap"][:, t] < 1.1), 100 * np.mean(c["gap"][ins, t] < 1.1)
    add("T37", "rooms below 1.1 pp at T=1000: all / in-scope", "67.0% / 69.0%", f"{a:.2f}% / {b:.2f}%",
        agree("67.0", a) and agree("69.0", b))
    mg, mf = np.median(c["gap"], 0), np.median(c["Pst"], 0)
    add("T37", "adaptation gap peaks / oracle floor lowest", "T=500 / T=1000",
        f"T={T[int(np.argmax(mg))]} / T={T[int(np.argmin(mf))]}",
        T[int(np.argmax(mg))] == 500 and T[int(np.argmin(mf))] == 1000)
    e = [np.exp(-2 * c["meta"][r]["gamma_room"] * 2100 * c["meta"][r]["dt_sim_s"]) for r in ids]
    add("T37", "median signal energy remaining at T=2100", "60%", f"{100 * np.median(e):.1f}%",
        agree("60", 100 * np.median(e)))
    m3 = pg <= 3 + 1e-9
    medP = np.median(P[ins][:, :, m3], axis=0)
    r1, r2 = np.ptp(medP[T.index(1)]), np.ptp(medP[T.index(2100)])
    add("T37", "range of the median P over p in [0,3]: T=1 / T=2100 (ratio)", "0.18 / 0.013 (14x)",
        f"{r1:.4f} / {r2:.4f} ({r1 / r2:.1f}x)", agree("0.18", r1) and agree("0.013", r2) and agree("14", r1 / r2))
    widths = []
    for Tv in T:
        ti = T.index(Tv)
        cur = P[:, ti, :]
        widths.append(np.median(np.sum(cur[:, m3] <= 1.05 * cur[:, m3].min(1)[:, None], 1) * 0.1))
    add("T37", "basin width (within 5% of the minimum) vs grid spacing", "~1-2 vs 0.1",
        f"median {min(widths):.1f}-{max(widths):.1f} vs {pg[1] - pg[0]:.1f}",
        0.8 <= min(widths) and max(widths) <= 2.2 and agree("0.1", pg[1] - pg[0]))
    gal = {"scene_00897": 3, "scene_00930": 4, "scene_00803": 5, "scene_00802": 6, "scene_00850": 8, "scene_00900": 10}
    t1 = T.index(1)
    pst = {r: c["pstar"][ids.index(r), t1] for r in gal}
    inside = all(c["Ps"][ids.index(r), t1] <= 1.05 * c["Pst"][ids.index(r), t1] for r in gal)
    add("T37/C.6", "gallery p*: decagon (00900) to quadrilateral (00930); |s| in every basin; > 3x spread",
        "about 0.4 to 1.4; yes; > 3x",
        f"{pst['scene_00900']:.1f} to {pst['scene_00930']:.1f}; {inside}; "
        f"{max(pst.values()) / min(pst.values()):.1f}x (all six: {sorted(round(float(v), 1) for v in pst.values())})",
        agree("0.4", pst["scene_00900"]) and agree("1.4", pst["scene_00930"]) and inside
        and max(pst.values()) / min(pst.values()) > 3)


# ----------------------------------------------------------------------------- T38
def check_T38(c):
    cap = np.load(SUPP / "capacity_1000ep_results.npz")
    c3 = np.load(SUPP / "capacity_results.npz")
    rows = (("M1", [cap[f"m1_phat_final_p{t}"] for t in ("0_5", "1_5", "2_5")], ("0.18", "0.72", "0.47")),
            ("M2", [cap[f"m2_phat_final_p{t}"] for t in ("0_5", "1_5", "2_5")], ("0.59", "0.81", "0.86")),
            ("M3", [c3[f"phat_m3_p{t}"] for t in ("0_5", "1_5", "2_5")], ("0.38", "1.33", "2.38")))
    for name, got, pr in rows:
        got = [float(g) for g in got]
        add("T38", f"recovered by {name}, targets 0.5 / 1.5 / 2.5", " / ".join(pr), " / ".join(f(g, 4) for g in got),
            all(agree(p, g) for p, g in zip(pr, got)), stored=True)
    m3 = [float(c3[f"phat_m3_p{t}"]) for t in ("0_5", "1_5", "2_5")]
    tg = [0.5, 1.5, 2.5]
    add("D.7/T38", "M3 monotone and slightly short of every target", "monotone; short of each",
        ", ".join(f"{g:.3f} vs {t}" for g, t in zip(m3, tg)),
        m3[0] < m3[1] < m3[2] and all(0 < t - g < 0.2 for g, t in zip(m3, tg)), stored=True)


# ----------------------------------------------------------------------------- prose
def check_prose(c):
    Kt, ins, ids = c["Kt"], c["ins"], c["ids"]
    add("Sec.3/5/C", "validation rooms / in-scope / boundary (K_total <= K)", "197 / 187 / 10",
        f"{len(ids)} / {ins.sum()} / {(~ins).sum()}", len(ids) == 197 and ins.sum() == 187 and (~ins).sum() == 10)
    add("Sec.1/C.1", "a typical room has over 300 modes; fit extends to modes 51-313 (median K_total)",
        "> 300; 313", f"median K_total {np.median(Kt):.0f}", np.median(Kt) == 313)
    add("Sec.1", "one snapshot: M equations for 2K unknowns", "8 / 100", f"{M} / {2 * K}", True)
    add("Sec.5", "61-point p grid on [0, 6]", "61, [0, 6]",
        f"{len(c['pg'])}, [{c['pg'][0]:.0f}, {c['pg'][-1]:.0f}]", len(c["pg"]) == 61 and c["pg"][0] == 0 and c["pg"][-1] == 6)
    add("Sec.5", "windows T", "1, 5, 10, 20, 50, 100, 200, 500, 1000, 2100", str(c["T"]),
        c["T"] == [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2100])
    med1000 = np.median(c["gap"][ins, c["T"].index(1000)])
    w = c["gap"][ins].max()
    add("Sec.5", "worst in-scope absolute cost is several times the 1.1 pp threshold", "several times",
        f"{w:.2f} / 1.1 = {w / 1.1:.1f}x", 3 <= w / 1.1 <= 9)
    dr = c["cost"]["per_room_delta_rel"]
    Tc = [int(t) for t in c["cost"]["T"]]
    r = np.median(dr[Tc.index(1000)]) / np.median(dr[Tc.index(1)])
    add("C.3", "median delta rises by an order of magnitude from T=1 to T=1000, falls at T=2100", "~10x; falls",
        f"{r:.1f}x; {100 * np.median(dr[Tc.index(2100)]):.2f}% < {100 * np.median(dr[Tc.index(1000)]):.2f}%",
        8 <= r <= 15 and np.median(dr[Tc.index(2100)]) < np.median(dr[Tc.index(1000)]))
    mg, mf = np.median(c["gap"], 0), np.median(c["Pst"], 0)
    T = c["T"]
    tri = mg[T.index(500)] / np.max(mg[:T.index(50) + 1])
    tri_lo = mg[T.index(500)] / np.min(mg[:T.index(50) + 1])
    add("C.4", "gap flat for T <= 50, roughly triples by T=500, back to short-window level at T=2100",
        "~3x; back", f"{tri:.1f}-{tri_lo:.1f}x; T=2100 {mg[T.index(2100)]:.2f} vs T<=50 {mg[:T.index(50) + 1].min():.2f}-{mg[:T.index(50) + 1].max():.2f}",
        2.4 <= tri and tri_lo <= 3.3 and mg[T.index(2100)] <= 1.15 * mg[:T.index(50) + 1].max())
    six = mf[T.index(1)] / mf[T.index(1000)]
    add("C.4", "oracle floor falls by a factor of six from T=1 to T=1000 and rebounds at T=2100", "6x; rebounds",
        f"{six:.2f}x; {mf[T.index(2100)]:.3f} > {mf[T.index(1000)]:.3f}",
        agree("6", six) and mf[T.index(2100)] > mf[T.index(1000)])
    import yaml
    bl = sorted(int(x) for x in yaml.safe_load(open(REPO / "configs" / "default.yaml"))["blacklist"]["scenes"])
    add("C.7", "rooms excluded for degenerate geometry", "00905, 00913, 00921", str(bl), bl == [905, 913, 921])
    nseg = [m.get("n_segments") for m in c["meta"].values()]
    add("C.7", "polygons with 3-10 vertices (validation rooms)", "3-10", f"{min(nseg)}-{max(nseg)}",
        min(nseg) == 3 and max(nseg) == 10)
    # parameter counts of the learned models (Sec. 6, App. D.1)
    import torch  # noqa: F401
    from src.models.solver import CondNet
    cn = sum(p.numel() for p in CondNet().parameters())
    add("Sec.6/D.1", "parameters: M1 = CondNet + 10 steps + 10 strengths / M2 = 50 + 10 + 10 / M3 = CondNet + alpha",
        "4,949 / 70 / 4,930", f"{cn + 20} / {K + 20} / {cn + 1}", cn + 20 == 4949 and K + 20 == 70 and cn + 1 == 4930)


def check_T13():
    not_checked("T13", "M3 vs closed form on the K=100 / M=16 / |s|=1.29 dataset (App. D.4)",
                "all cells", "needs the trained v2 M3 models and the v2 K=100 dataset, which were not kept")
    not_checked("D.4", "167 in-scope + 30 boundary rooms of the v2 dataset", "167 / 30", "same v2 dataset, not shipped")


def main():
    check_paper_dataset()
    c = load_common()
    check_T7(c)
    check_T8(c)
    check_T9(c)
    check_T10_T11(c)
    check_T12(c)
    check_T13()
    check_T14(c)
    check_T15(c)
    check_T18_T39(c)
    check_T30(c)
    check_T31(c)
    check_T33(c)
    check_T34(c)
    check_T37(c)
    check_T38(c)
    check_prose(c)

    w = [max(len(r[i]) for r in ROWS) for i in (0, 1)]
    print(f"{'table':{w[0]}}  {'row':{w[1]}}  printed | recomputed | status")
    for t, lab, p, r, st in ROWS:
        print(f"{t:{w[0]}}  {lab:{w[1]}}  {p} | {r} | {st}")
    n = {k: sum(r[4] == k for r in ROWS) for k in ("OK", "OK(stored)", "NOT CHECKED", "DIFF(known)", "DIFF")}
    print(f"\n{len(ROWS)} checks: {n['OK']} OK, {n['OK(stored)']} OK(stored), {n['NOT CHECKED']} NOT CHECKED, "
          f"{n['DIFF(known)']} DIFF(known), {n['DIFF']} DIFF")
    sys.exit(1 if n["DIFF"] else 0)


if __name__ == "__main__":
    main()
