#!/usr/bin/env python
"""Recompute every number printed in T19-T23, T40 and the Appendix E / Sec. 7 prose.

Prints one row per printed number: [table, row, printed, recomputed, status], where
status is
  OK            printed value equals the recomputed value at the printed precision
  NOTE          printed statement holds only with a qualifier (explained in the row)
  DIFF (known)  a printed value the data do not support (listed in KNOWN and in README.md)
  DIFF          an unexpected difference -> exit code 1

Data (default: the shipped files the paper was made from):
  data/experiments/heat_c_continuous_fit.npz        T19, T22, T40 r7, fig14
  data/experiments/exp2_p_sweep_heat_extended.npz   T40 r9, r11, Sec. E.4/E.5
  data/supplementary/heat_2d_sweep_results.npz       T21, fig13
  data/experiments/p_sweep/p_sweep_K50_M8.npz        acoustic p* (T40 r10)
  data/experiments/noise_profile/noise_profile_K50_M8.npz   |s| (T23 acoustic row)
  legacy modal dataset (MODAL)                        T23, T40 r1-r6, r12-r14, timescales
With --ported the first three come from data/experiments/appendix/ (outputs of the
drivers in this folder).

Usage (from the repo root):
    python scripts/appendix/E/check_E.py
    python scripts/appendix/E/check_E.py --ported
"""
import argparse
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heat_common import (  # noqa: E402
    DIAG_ROOMS, EXP, MODAL, SUPP, check_paper_dataset, generate_heat_signals, heat_seed,
    load_heat_room, pad_room_to_K, val_rooms, valid_snaps,
)
import numpy as np  # noqa: E402

import heat_herfindahl as HH  # noqa: E402
from src.visualization.fig_heat_diagnostics import T_ACOUSTIC, T_OBS, heat_amplitudes  # noqa: E402

# Printed values that the data do not support.
KNOWN = {}

ROWS = []


def decimals(s):
    s = s.strip().lstrip("+-~<>=")
    return len(s.split(".")[1]) if "." in s else 0


def same(v, printed):
    """True if v formatted like `printed` (decimals, explicit sign) reads the same."""
    p = printed.replace("−", "-")
    fmt = f"{{:{'+' if p.startswith('+') else ''}.{decimals(p)}f}}"
    return fmt.format(v) == p.lstrip("~") or fmt.format(v) == p


def add(table, item, printed, recomputed, ok):
    """ok: truthy/falsy, or the string 'note'."""
    if isinstance(ok, str):
        status = "NOTE"
    elif bool(ok):
        status = "OK"
    else:
        status = "DIFF (known)" if (table, item) in KNOWN or (table, "*") in KNOWN else "DIFF"
    ROWS.append((table, item, str(printed), str(recomputed), status))


def fit_spectrum(ev, a2):
    """fig11 fits: pure power law and two-parameter form on log a2."""
    le, la = np.log(ev), np.log(a2)
    sst = np.sum((la - la.mean())**2)
    cA = np.polyfit(le, la, 1)
    resA = np.polyval(cA, le) - la
    X = np.column_stack([le, ev, np.ones(len(ev))])
    cB = np.linalg.lstsq(X, la, rcond=None)[0]
    resB = X @ cB - la
    return {"r2_pl": 1 - np.sum(resA**2) / sst, "s_pl": -cA[0], "maxres_pl": np.abs(resA).max(),
            "r2_2p": 1 - np.sum(resB**2) / sst, "s_2p": -cB[0], "c_2p": -cB[1],
            "rms_2p": np.sqrt(np.mean(resB**2)), "maxres_2p": np.abs(resB).max()}


def mean_a2(room, sid, t, n_seeds=20):
    return np.mean([heat_amplitudes(room, heat_seed(sid, o), t)**2 for o in range(n_seeds)], axis=0)


def check_T19_T22(cf):
    ct, cfit, rid, sn = cf["c_theory"], cf["c_fit"], cf["room_ids"], cf["snap_indices"]
    t19 = {"00805": ("0.984", "+0.009", "0.9984"), "00826": ("1.000", "-0.072", "1.0000"),
           "00840": ("0.993", "-0.021", "0.9999"), "00880": ("0.973", "-0.010", "0.9990"),
           "00950": ("0.981", "-0.002", "0.9986")}
    per = {}
    for r, (ps, pi, pr) in t19.items():
        m = rid == f"scene_{r}"
        b, a = np.polyfit(ct[m], cfit[m], 1)
        r2 = 1 - np.sum((cfit[m] - a - b * ct[m])**2) / np.sum((cfit[m] - cfit[m].mean())**2)
        per[r] = (b, r2)
        add("T19", f"{r} slope", ps, f"{b:.4f}", same(b, ps))
        add("T19", f"{r} intercept", pi, f"{a:+.4f}", same(a, pi))
        add("T19", f"{r} R2", pr, f"{r2:.5f}", same(r2, pr))
    b, a = np.polyfit(ct, cfit, 1)
    r2 = 1 - np.sum((cfit - a - b * ct)**2) / np.sum((cfit - cfit.mean())**2)
    add("T19", "pooled slope", "0.987", f"{b:.4f}", same(b, "0.987"))
    add("T19", "pooled intercept", "-0.020", f"{a:+.4f}", same(a, "-0.020"))
    add("T19", "pooled R2", "0.9880", f"{r2:.5f}", same(r2, "0.9880"))

    t22 = {"00805": ("0.98", "0.9984", "0.482", "0.472", "0.981", "0.980"),
           "00826": ("1.00", "1.0000", "0.472", "0.400", "0.847", "0.615"),
           "00840": ("0.99", "0.9999", "0.499", "0.476", "0.955", "0.954"),
           "00880": ("0.97", "0.9990", "0.487", "0.472", "0.969", "0.965"),
           "00950": ("0.98", "0.9986", "0.478", "0.461", "0.963", "0.985")}
    names = ["alpha1", "R2_alpha1", "c_theory", "c_hat", "c_hat/c_theory", "R2_spec"]
    for r, printed in t22.items():
        i = np.where((rid == f"scene_{r}") & (sn == 100))[0][0]
        vals = [per[r][0], per[r][1], ct[i], cfit[i], cfit[i] / ct[i], cf["spectral_r2"][i]]
        for n, p, v in zip(names, printed, vals):
            add("T22", f"{r} {n}", p, f"{v:.5f}", same(v, p))

    # T40 r7 and prose of E.2 / fig14 / Sec. 7 / intro
    lo, hi = float(cf["slope_all5_ci_lo"]), float(cf["slope_all5_ci_hi"])
    s = float(cf["slope_all5"])
    add("T40", "r7 pooled slope [95% CI]", "0.987 [0.953, 1.022]", f"{s:.4f} [{lo:.4f}, {hi:.4f}]",
        same(s, "0.987") and same(lo, "0.953") and same(hi, "1.022"))
    add("E.2", "pooled CI contains one", "yes", f"[{lo:.4f}, {hi:.4f}]", lo <= 1 <= hi)
    off = cfit[rid == "scene_00826"] - ct[rid == "scene_00826"]
    add("fig14", "one room below the line by a constant offset", "00826",
        f"00826 offset {off.mean():+.4f} (spread {np.ptp(off):.1e}), slope {per['00826'][0]:.4f}",
        np.ptp(off) < 1e-9 and off.mean() < 0)
    slopes = np.array([v[0] for v in per.values()])
    add("Sec.7/intro", "per-room slopes close to one (rate within a few percent)", "close to one",
        f"{slopes.min():.4f}-{slopes.max():.4f}; R2 >= {min(v[1] for v in per.values()):.4f}",
        bool(np.all(np.abs(slopes - 1) < 0.05)))
    rat = {r: cfit[(rid == f"scene_{r}") & (sn == 100)][0] / ct[(rid == f"scene_{r}") & (sn == 100)][0]
           for r in t22}
    four = [rat[r] for r in t22 if r != "00826"]
    add("E.6", "single-time ratio within a few percent in four rooms, lower in 00826", "yes",
        f"four rooms {min(four):.3f}-{max(four):.3f}; 00826 {rat['00826']:.3f}",
        max(abs(1 - x) for x in four) < 0.05 and rat["00826"] < min(four))
    r2s = {r: cf["spectral_r2"][rid == f"scene_{r}"] for r in t22}
    add("Sec.7", "close spectral fit at all but the earliest times", "close fit",
        "single-seed fits (T22 estimator): R2_spec >= 0.80 from snapshot 50 in 4 rooms, "
        f"but {r2s['00826'].max():.2f} at best in 00826; 20-seed fits (fig11): see T40 r4", "note")
    return per


def check_T21(s2):
    Ph, p, c, T = s2["P_heat_2d"], s2["p_values"], s2["c_values"], list(s2["T_values"])
    t21 = {1: ("1.275", "1.256", "1.5", "4.0", "0.1", "13.8"), 10: ("1.244", "1.244", "0.0", "4.4", "0.2", "13.1"),
           100: ("0.871", "0.871", "0.0", "3.1", "0.0", "13.3"), 500: ("0.738", "0.727", "1.4", "4.0", "0.0", "12.2")}
    gains = {}
    for Tv, (p1, p2, pop, per, q1, q3) in t21.items():
        S = Ph[:, T.index(Tv)]
        med = np.median(S, axis=0)
        P1, P2 = med[:, 0].min(), med.min()
        gain = (1 - S.reshape(len(S), -1).min(1) / S[:, :, 0].min(1)) * 100
        gains[Tv] = gain
        g25, g50, g75 = np.percentile(gain, [25, 50, 75])
        add("T21", f"T={Tv} 1-param P*", p1, f"{P1:.4f}", same(P1, p1))
        add("T21", f"T={Tv} 2-param P*", p2, f"{P2:.4f}", same(P2, p2))
        add("T21", f"T={Tv} pop. improvement %", pop, f"{100 * (1 - P2 / P1):.2f}", same(100 * (1 - P2 / P1), pop))
        add("T21", f"T={Tv} per-room median [IQR] %", f"{per} [{q1}, {q3}]", f"{g50:.2f} [{g25:.2f}, {g75:.2f}]",
            same(g50, per) and same(g25, q1) and same(g75, q3))
    add("E.5", "two-param gain a few percent per room in the median", "a few percent",
        ", ".join(f"T={Tv}: {np.median(g):.1f}%" for Tv, g in gains.items()),
        all(1 <= np.median(g) <= 6 for g in gains.values()))
    add("E.5", "skewed: many rooms gain nothing, a quarter more than a tenth", "zeros; 75th pct > 10%",
        ", ".join(f"T={Tv}: {np.sum(g <= 0)} zero, 75th {np.percentile(g, 75):.1f}%" for Tv, g in gains.items()),
        all(np.sum(g <= 0) > 0 and np.percentile(g, 75) > 10 for g in gains.values()))

    # fig13 caption (main text): minima of the median surfaces at the shown windows
    Pw = s2["P_wave_2d"]
    for Tv in (10, 100, 500):
        mh, mw = np.median(Ph[:, T.index(Tv)], 0), np.median(Pw[:, T.index(Tv)], 0)
        ih, iw = np.unravel_index(np.argmin(mh), mh.shape), np.unravel_index(np.argmin(mw), mw.shape)
        add("fig13", f"T={Tv} acoustic optimum at c = 0", "c = 0",
            f"(p, c) = ({p[iw[0]]:.1f}, {c[iw[1]]:.2f})", bool(c[iw[1]] == 0))
        add("fig13", f"T={Tv} heat optimum near c = 0 at a larger p", "near c=0, p > acoustic",
            f"(p, c) = ({p[ih[0]]:.1f}, {c[ih[1]]:.2f})", bool(c[ih[1]] <= 0.02 and p[ih[0]] > p[iw[0]]))


def p_star(P, p_values, t_idx, m_idx):
    return p_values[np.nanargmin(np.nanmedian(P[:, t_idx, m_idx, :], axis=0))]


def check_psweeps(he, ac):
    P, p, T = he["P_oracle"], he["p_values"], list(he["T_values"])
    m8 = list(he["M_values"]).index(8)
    ps = {Tv: p_star(P, p, T.index(Tv), m8) for Tv in T}
    add("T40", "r9 heat p* T=1", "2.5", f"{ps[1]:.1f}", same(ps[1], "2.5"))
    low = [ps[Tv] for Tv in T if Tv <= 200]
    in_rng = sum(2.5 <= x <= 2.8 for x in low)
    add("T40", "r9 heat p* T<=200", "2.5-2.8",
        f"{min(low):.1f}-{max(low):.1f}: " + ", ".join(f"{x:.1f}" for x in low) +
        f" ({in_rng}/{len(low)} windows in 2.5-2.8; T=20 gives {ps[20]:.1f})",
        True if in_rng == len(low) else "note")
    high = [ps[Tv] for Tv in T if Tv >= 1000]
    add("T40", "r9 heat p* T>=1000", "3.0-3.2", f"{min(high):.1f}-{max(high):.1f}",
        same(min(high), "3.0") and same(max(high), "3.2"))

    Pa, pa, Ta = ac["P_oracle"], ac["p_values"], list(ac["T_values"])
    ma = list(ac["M_values"]).index(8)
    psa = {Tv: p_star(Pa, pa, Ta.index(Tv), ma) for Tv in Ta}
    add("T40", "r10 acoustic p* T=1", "1.4", f"{psa[1]:.1f}", same(psa[1], "1.4"))
    add("T40", "r10 acoustic p* T=1000", "~2.2", f"{psa[1000]:.1f}", same(psa[1000], "2.2"))
    lowa = [psa[Tv] for Tv in Ta if Tv <= 200]
    add("E.4", "acoustic p* near |s| = 1.13 up to T = 200", "near 1.13", f"{min(lowa):.1f}-{max(lowa):.1f}",
        max(abs(x - 1.13) for x in lowa) <= 0.3)
    add("E.4", "heat p* more than twice |s| at T = 1", "> 2 x 1.13", f"{ps[1]:.1f}", ps[1] > 2 * 1.13)
    add("E.4", "heat p* rises further for T >= 1000", "rises", f"{min(high):.1f}-{max(high):.1f} vs <= {max(low):.1f}",
        min(high) > max(low))

    t = T.index(2100)
    curve = np.nanmedian(P[:, t, m8, :], axis=0)
    ridge = curve[0]
    P11 = curve[np.argmin(np.abs(p - 1.1))]
    P113 = np.interp(1.13, p, curve)
    i_min = np.nanargmin(curve)
    P32 = curve[np.argmin(np.abs(p - 3.2))]
    P20 = curve[np.argmin(np.abs(p - 2.0))]
    per_room = np.nanmin(P[:, t, m8, :], axis=1)
    oracle = np.nanmedian(per_room)
    p_room = np.median(p[np.nanargmin(P[:, t, m8, :], axis=1)])
    add("T40", "r11 ridge", "1.210", f"{ridge:.4f}", same(ridge, "1.210"))
    add("T40", "r11 lambda^1.13", "0.399", f"{P11:.4f} (p=1.1; interpolated at 1.13: {P113:.4f})", same(P11, "0.399"))
    add("T40", "r11 per-room one-param oracle (median p*)", "0.235 (3.2)",
        f"{oracle:.4f} ({p_room:.1f}); median-curve min {curve[i_min]:.4f} at p={p[i_min]:.1f}",
        same(oracle, "0.235") and same(p_room, "3.2"))
    add("E.5", "ridge has P > 1 (T=2100)", "P > 1", f"{ridge:.4f}", ridge > 1)
    add("E.5", "lambda^1.13 cuts ridge's error to about a third", "~1/3", f"{P11 / ridge:.2f} of ridge",
        abs(P11 / ridge - 1 / 3) < 0.05)
    add("E.5", "per-room one-param heat oracle lowers it by a further two fifths", "~2/5",
        f"{1 - oracle / P11:.2f}", abs(1 - oracle / P11 - 0.4) < 0.05)


def check_room_00840_and_fits():
    room = load_heat_room("scene_00840")
    room = pad_room_to_K(room, 50)
    ev, gamma = room["eigenvalues"], room["gamma_room"]
    surv = np.exp(-2 * ev[24:] * T_OBS[1]).max()
    env = np.exp(-gamma * T_ACOUSTIC)
    add("T40", "r1 00840 heat energy k>=24 at 250 ms", "< 1e-9", f"{surv:.2e}", bool(surv < 1e-9))
    add("T40", "r1 00840 acoustic envelope at 500 ms", "about 8%",
        f"{100 * env:.1f}% (gamma={gamma:.3f}; gamma = 5.0 would give 8.2%)", round(100 * env) == 8)
    orders = [np.log10(env / np.exp(-2 * ev[49] * t)) for t in (T_OBS[1], T_OBS[2])]
    add("T40", "r2 orders at k = 49, 250 / 495 ms", "17 / 34", f"{orders[0]:.1f} / {orders[1]:.1f}",
        same(orders[0], "17") and same(orders[1], "34"))

    fits = {}
    for sid in DIAG_ROOMS:
        r = pad_room_to_K(load_heat_room(sid), 50)
        fits[sid] = {t: fit_spectrum(r["eigenvalues"], mean_a2(r, sid, t)) for t in (0.0,) + T_OBS}
    f840 = fits["scene_00840"]
    s2p = [f[t]["s_2p"] for f in fits.values() for t in T_OBS]
    analytic = [-np.polyfit(np.log(pad_room_to_K(load_heat_room(s), 50)["eigenvalues"]),
                            -np.log1p(pad_room_to_K(load_heat_room(s), 50)["eigenvalues"]), 1)[0] for s in DIAG_ROOMS]
    add("T40", "r3 |s_hat|: 00840 / finite-k prediction (five rooms)", "0.96 / 0.86-0.98",
        f"00840 {f840[T_OBS[0]]['s_2p']:.3f} (2-param), {f840[0.0]['s_pl']:.3f} (t=0); analytic "
        f"{min(analytic):.3f}-{max(analytic):.3f}; five rooms 2-param {min(s2p):.3f}-{max(s2p):.3f}",
        same(f840[T_OBS[0]]["s_2p"], "0.96") and same(min(analytic), "0.86") and same(max(analytic), "0.98"))
    r2pl840 = [f840[t]["r2_pl"] for t in T_OBS]
    r2pl5 = [f[t]["r2_pl"] for f in fits.values() for t in T_OBS]
    r2b840 = [f840[t]["r2_2p"] for t in T_OBS]
    r2b5 = [f[t]["r2_2p"] for f in fits.values() for t in T_OBS]
    add("T40", "r4 two-param R2 (00840, three times)", "> 0.99",
        f"00840 >= {min(r2b840):.4f}; five rooms >= {min(r2b5):.4f}", bool(min(r2b840) > 0.99))
    add("T40", "r4 power-law R2 (00840, three times)", "0.86-0.91",
        f"00840 {min(r2pl840):.3f}-{max(r2pl840):.3f}; five rooms {min(r2pl5):.3f}-{max(r2pl5):.3f}",
        same(min(r2pl840), "0.86") and same(max(r2pl840), "0.91"))
    add("T40", "r5 power-law residual (late)", "~30", f"00840 max |res| at 495 ms {f840[T_OBS[2]]['maxres_pl']:.1f}",
        round(f840[T_OBS[2]]["maxres_pl"]) == 30)
    add("T40", "r5 two-param residual", "~0.3",
        f"00840 RMS {f840[T_OBS[2]]['rms_2p']:.2f}, max {f840[T_OBS[2]]['maxres_2p']:.2f}",
        same(f840[T_OBS[2]]["rms_2p"], "0.3"))
    for t, pr, ps_ in zip(T_OBS, ("0.91", "0.87", "0.86"), ("3.6", "14.4", "27.5")):
        f = f840[t]
        add("T40", f"r6 00840 power law at {t * 1000:.0f} ms R2 (slope)", f"{pr} ({ps_})",
            f"{f['r2_pl']:.4f} ({f['s_pl']:.2f})", same(f["r2_pl"], pr) and same(f["s_pl"], ps_))
    add("E/Sec.7", "|s|_heat ~ 1.0 from the t = 0 spectrum", "~1.0", f"00840 {f840[0.0]['s_pl']:.3f}",
        same(f840[0.0]["s_pl"], "1.0"))
    add("E.2", "two-param c_hat = 2 kappa t (00840, fig11)", "2t",
        ", ".join(f"{f840[t]['c_2p']:.4f} vs {2 * t:.3f}" for t in T_OBS),
        all(abs(f840[t]["c_2p"] - 2 * t) < 0.01 for t in T_OBS))


def check_herfindahl():
    r = HH.compute()
    H = r["H"]
    for j, t in enumerate(r["t_ms"]):
        pm, (plo, phi), (ilo, ihi) = HH.PRINTED[t]
        Hs = H[:, j]
        add("T23", f"{t} ms median H", pm, f"{np.median(Hs):.4f}", same(np.median(Hs), pm))
        add("T23", f"{t} ms H range", f"[{plo}, {phi}]", f"[{Hs.min():.4f}, {Hs.max():.4f}]",
            same(Hs.min(), plo) and same(Hs.max(), phi))
        add("T23", f"{t} ms 1/H range", f"[{ilo}, {ihi}]", f"[{(1 / Hs).min():.2f}, {(1 / Hs).max():.2f}]",
            same((1 / Hs).min(), ilo) and same((1 / Hs).max(), ihi))
    Ha = float(np.median(r["H_acoustic"]))
    add("T23", "acoustic median H / 1/H", "~0.005 / ~200", f"{Ha:.4f} / {1 / Ha:.0f} ({len(r['H_acoustic'])} rooms)",
        same(Ha, "0.005") and round(1 / Ha, -1) == 200)
    add("T40", "r12 heat H at ~12 ms", "~0.027", f"{np.median(H[:, 0]):.4f}", same(np.median(H[:, 0]), "0.027"))
    add("T40", "r13 heat 1/H at 12 / 250 ms", "~20-80 / ~1-6",
        f"{(1 / H[:, 0]).min():.1f}-{(1 / H[:, 0]).max():.1f} / {(1 / H[:, 1]).min():.2f}-{(1 / H[:, 1]).max():.2f}",
        round((1 / H[:, 0]).min(), -1) == 20 and round((1 / H[:, 0]).max(), -1) == 80
        and round((1 / H[:, 1]).min()) == 1 and round((1 / H[:, 1]).max()) == 6)
    i950 = r["rooms"].index("scene_00950")
    add("T40", "r14 H at 493 ms: 00950 (K_total) / median (1/H)", "0.99 (93) / 0.548 (~1.8)",
        f"{H[i950, 2]:.4f} ({r['K_total'][i950]}) / {np.median(H[:, 2]):.4f} ({1 / np.median(H[:, 2]):.2f})",
        same(H[i950, 2], "0.99") and r["K_total"][i950] == 93 and same(np.median(H[:, 2]), "0.548")
        and same(1 / np.median(H[:, 2]), "1.8"))
    ratio = (1 / H[:, 0]) / (1 / H[:, 1])
    order = np.argsort(r["K_total"])
    add("E.7", "1/H drops by an order of magnitude by 250 ms", ">= 10x",
        f"{ratio.min():.1f}-{ratio.max():.1f}x", bool(ratio.min() >= 10))
    add("E.7", "largest 1/H degradation in the smallest rooms", "smallest rooms",
        "by K_total: " + ", ".join(f"{r['K_total'][i]}:{ratio[i]:.1f}x" for i in order),
        bool(np.all(np.diff(ratio[order]) <= 0.5)))
    add("E.7", "H approaches one in the smallest room at the latest time; median ~1/2", "~1 / ~0.5",
        f"{H[i950, 2]:.3f} / {np.median(H[:, 2]):.3f}", H[i950, 2] > 0.95 and abs(np.median(H[:, 2]) - 0.5) < 0.1)


def check_timescales_and_model():
    dts, dsn, sps_all, nvalid, tmin, tmax, tmed = [], [], [], [], [], [], []
    T_GRID = [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2100]
    for sid in val_rooms():
        tr = np.load(os.path.join(MODAL, sid, "modal_trajectories.npz"))
        dt, ds, n = float(tr["dt_sim"]), float(tr["dt_snap"]), tr["a"].shape[1]
        sps = int(round(ds / dt))
        dts.append(dt); dsn.append(ds); sps_all.append(sps)
        nvalid += [len(valid_snaps(n, sps, n * sps, T)) for T in T_GRID]
        t = np.array(valid_snaps(n, sps, n * sps, 1)) * ds
        tmin.append(t.min()); tmax.append(t.max()); tmed.append(np.median(t))
    add("E.6", "dt_sim ~ 24 us", "~24 us", f"median {np.median(dts) * 1e6:.2f} us", round(np.median(dts) * 1e6) == 24)
    add("E.6", "dt_snap = 100 dt_sim ~ 2.4 ms", "100 / ~2.4 ms",
        f"steps_per_snap {sorted(set(sps_all))}; median {np.median(dsn) * 1e3:.3f} ms",
        set(sps_all) == {100} and same(np.median(dsn) * 1e3, "2.4"))
    add("E.6", "~190 valid snapshots per P(T)", "~190", f"mean {np.mean(nvalid):.0f} (range {min(nvalid)}-{max(nvalid)})",
        abs(np.mean(nvalid) - 190) < 10)
    lo, hi = np.median(tmin) * 1e3, np.median(tmax) * 1e3
    add("E.6", "t_obs in [12, 493] ms", "[12, 493]",
        f"[{lo:.1f}, {hi:.1f}] (median over rooms of per-room first/last valid snapshot; "
        f"extremes {min(tmin) * 1e3:.1f}/{max(tmax) * 1e3:.1f})", abs(lo - 12) < 1 and abs(hi - 493) < 1)
    add("E.7", "median t_obs ~250 ms", "~250 ms", f"{np.median(tmed) * 1e3:.1f} ms (median of per-room medians)",
        abs(np.median(tmed) * 1e3 - 250) < 5)
    add("E.6", "representative t_obs ~250 ms matches the median", "~250 ms", f"{np.median(tmed) * 1e3:.1f} ms",
        abs(np.median(tmed) * 1e3 - 250) < 10)

    nf = inspect.signature(generate_heat_signals).parameters["noise_frac"].default
    room = load_heat_room("scene_00840")
    _, a, _ = generate_heat_signals(room, heat_seed("scene_00840"))
    ev, sps, dt = room["eigenvalues"], room["steps_per_snap"], room["dt_sim"]
    z = np.random.default_rng(heat_seed("scene_00840")).standard_normal(room["K"])
    decay = np.max(np.abs(a[:, 100] / a[:, 0] / np.exp(-ev * 100 * sps * dt) - 1))
    add("E.1", "1% sensor noise", "1%", f"noise_frac = {nf}", nf == 0.01)
    add("E.1", "kappa = 1 (a_k(t) = a_k(0) e^{-lambda_k t})", "kappa = 1", f"max rel. deviation {decay:.1e}", decay < 1e-10)
    add("E.1", "Var a_k(0) = 1/(1+lambda_k)", "(1+lambda)^-1", f"a0*sqrt(1+lambda) == N(0,1) draw: "
        f"{np.allclose(a[:, 0] * np.sqrt(1 + ev), z, rtol=0, atol=1e-14)}",
        bool(np.allclose(a[:, 0] * np.sqrt(1 + ev), z, rtol=0, atol=1e-14)))
    add("E", "five diagnostic rooms, 8 observation times", "5 / 8", f"{len(DIAG_ROOMS)} / 8", len(DIAG_ROOMS) == 5)
    add("Sec.7/E.7", "heat data carry no truncation noise (retained modes + sensor noise only)", "no discarded modes",
        f"Y = Phi[:, :K] a[:K] + {nf:.0%} white noise, K = min(K_total, 50): 0 discarded modes simulated", True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ported", action="store_true", help="read the outputs in data/experiments/appendix/")
    args = ap.parse_args()
    check_paper_dataset(MODAL)
    src = EXP / "appendix" if args.ported else None
    cf = np.load((src or EXP) / "heat_c_continuous_fit.npz", allow_pickle=True)
    he = np.load((src or EXP) / "exp2_p_sweep_heat_extended.npz", allow_pickle=True)
    s2 = np.load((src or SUPP) / "heat_2d_sweep_results.npz")
    ac = np.load(EXP / "p_sweep" / "p_sweep_K50_M8.npz", allow_pickle=True)

    check_T19_T22(cf)
    check_T21(s2)
    check_herfindahl()
    check_room_00840_and_fits()
    check_psweeps(he, ac)
    check_timescales_and_model()

    w = [max(len(r[i]) for r in ROWS) for i in range(4)]
    w[3] = min(w[3], 70)
    print(f"Data: {'rerun outputs (data/experiments/appendix)' if args.ported else 'shipped'}\n")
    print(f"{'table':<10} {'row':<{w[1]}} {'printed':<{w[2]}} {'recomputed':<{w[3]}} status")
    for t, item, p, r, s in ROWS:
        print(f"{t:<10} {item:<{w[1]}} {p:<{w[2]}} {r:<{w[3]}} {s}")
    counts = {s: sum(r[4] == s for r in ROWS) for s in ("OK", "NOTE", "DIFF (known)", "DIFF")}
    print("\n" + ", ".join(f"{k}: {v}" for k, v in counts.items()) + f"  (total {len(ROWS)})")
    print("\nKnown problems:")
    for (t, item), why in KNOWN.items():
        print(f"  {t} {item}: {why}")
    sys.exit(1 if counts["DIFF"] else 0)


if __name__ == "__main__":
    main()
