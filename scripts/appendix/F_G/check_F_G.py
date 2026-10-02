#!/usr/bin/env python
"""Recompute every number printed in Tables T24, T26-T28, T41, T42 and in the prose of
Appendices F and G from the data, and compare with the printed values.

Each row prints: table | row | printed | recomputed | status | note
  OK            recomputed value equals the printed one at the printed precision
  DIFF          it does not (marked ! and exit status 1 unless the note names a known
                problem)
  UNVERIFIABLE  the data or the trained weights needed are not included
  MISSING       an input file is absent (run the producing script first)

Values that depend on random draws (T27 flatness, T28 3D column, T41 KS gap, T42
flatness and 3D rows) are printed as median [min, max] over 10 fixed seeds and are
checked against the per-seed outputs of run_multi_seed.sh in <data-dir>/seeds/.

Inputs: --data-dir (default data/experiments/appendix) holds the npz files written by
the scripts of this folder; hd_decomposition.npz (written by
scripts/appendix/A_B/hd_decomposition.py) is optional. Released data (cost, p sweep,
anisotropy, rectangular_*.npz) and the modal dataset come from src.utils.paths.

Usage: python scripts/appendix/F_G/check_F_G.py [--data-dir DIR]
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import re
import sys
from pathlib import Path

import numpy as np

from _common import REPO, OUT
from src.utils.paths import MODAL, EXP, check_paper_dataset

sys.path.insert(0, str(REPO / "scripts"))

ROWS = []
MISSING_FILES = set()
P_GRID = np.round(np.arange(0.0, 6.01, 0.1), 10)
M03 = (P_GRID >= 0) & (P_GRID <= 3.0000001)
S_HAT = 1.13

M3_LOST = ("needs the trained M3 n800 weights, which are not included; rerun the producing "
           "script with --m3-checkpoint after retraining")


def norm(s):
    """Compare at printed precision: drop spaces, unify minus signs, -0.0 -> 0.0."""
    s = str(s).replace("−", "-").replace(" ", "")
    return re.sub(r"-(0\.0+)(?![0-9]*[1-9])", r"\1", s)


def add(table, row, printed, recomputed, ok=None, known=None):
    """recomputed: string at the printed precision, or None when it cannot be computed."""
    if recomputed is None:
        status = "MISSING" if (known or "").startswith("missing") else "UNVERIFIABLE"
        recomputed = "-"
    else:
        if ok is None:
            ok = norm(printed) == norm(recomputed)
        status = "OK" if ok else "DIFF"
    note = known or ""
    unexpected = status == "DIFF" and not known
    ROWS.append((table, row, str(printed), str(recomputed), status, note, unexpected))


class Data:
    def __init__(self, data_dir):
        self.dir = Path(data_dir)
        self.cache = {}

    def __call__(self, name):
        if name not in self.cache:
            p = self.dir / name
            self.cache[name] = np.load(p, allow_pickle=True) if p.exists() else None
            if self.cache[name] is None:
                MISSING_FILES.add(name)
        return self.cache[name]

    def first(self, *names):
        """The first of several alternative file names that exists."""
        for n in names:
            if (self.dir / n).exists():
                return self(n)
        MISSING_FILES.add(" or ".join(names))
        return None


def mr(v, fmt="{:.2f}", fmt_lo=None, fmt_hi=None):
    """median [min, max] of a per-seed array, at the printed precision."""
    v = np.asarray(v, float)
    return (fmt.format(np.median(v)) + " [" + (fmt_lo or fmt).format(v.min()) + ", "
            + (fmt_hi or fmt).format(v.max()) + "]")


_SEEDS = {}


def seed_summary(D):
    """Per-seed values over the 10 fixed seeds (multi_seed_summary.compute), or None."""
    if "S" not in _SEEDS:
        sd = D.dir / "seeds"
        if not (sd / "box3d_flatness_seed9.npz").exists():
            MISSING_FILES.add("seeds/*_seed{0..9}.npz (run_multi_seed.sh)")
            _SEEDS["S"] = None
        else:
            from multi_seed_summary import compute
            _SEEDS["S"] = compute(sd)
    return _SEEDS["S"]


def f1(x): return f"{x:.1f}"
def f2(x): return f"{x:.2f}"
def f3(x): return f"{x:.3f}"
def f4(x): return f"{x:.4f}"


def stats_curve(curve):
    """p*, P*, delta(|s|), flatness of a 62-point curve (61 grid points + P(|s|))."""
    g = curve[:len(P_GRID)]
    j = int(np.nanargmin(g))
    return (P_GRID[j], g[j], (curve[len(P_GRID)] - g[j]) / g[j],
            float(np.nanmax(g[M03]) / np.nanmin(g[M03])))


def delta_interp(curve, pg=P_GRID, s=S_HAT):
    j = int(np.nanargmin(curve))
    return pg[j], curve[j], (np.interp(s, pg, curve) - curve[j]) / curve[j]


def flat03(curve, pg=P_GRID):
    m = (pg >= 0) & (pg <= 3.0000001)
    return float(np.nanmax(curve[m]) / np.nanmin(curve[m]))


# ======================================================================== Appendix F
PRIOR_LABEL = {"gaussian": "Gaussian", "heavy_tail": "Heavy-tail (t3)",
               "correlated": "Correlated"}
T24_POP = {("gaussian", 1): ("1.1", "0.659", "0.1%", "1.322"),
           ("gaussian", 100): ("1.1", "0.566", "0.0%", "1.274"),
           ("heavy_tail", 1): ("1.1", "0.659", "0.1%", "1.322"),
           ("heavy_tail", 100): ("1.1", "0.566", "0.0%", "1.274"),
           ("correlated", 1): ("1.1", "0.656", "0.1%", "1.329"),
           ("correlated", 100): ("1.1", "0.567", "0.0%", "1.285")}
T24_EMP = {("gaussian", 1): ("1.0 [0.8, 1.4]", "0.702", "0.6% [0.0, 2.6]", "1.253"),
           ("gaussian", 100): ("1.1 [1.0, 1.4]", "0.603", "0.3% [-0.1, 1.6]", "1.283"),
           ("heavy_tail", 1): ("0.9 [0.5, 1.6]", "0.759", "0.7% [0.0, 4.0]", "1.182"),
           ("heavy_tail", 100): ("1.1 [0.6, 1.4]", "0.658", "0.7% [0.0, 3.5]", "1.219"),
           ("correlated", 1): ("1.0 [0.9, 1.4]", "0.705", "0.4% [0.0, 1.5]", "1.252"),
           ("correlated", 100): ("1.2 [1.0, 1.4]", "0.601", "0.3% [0.0, 1.2]", "1.273")}
T26 = {("gaussian", 1): ("0.46%", "1.20 ± 0.13%"), ("gaussian", 100): ("1.41%", "2.74 ± 0.13%"),
       ("correlated", 1): ("0.41%", "1.11 ± 0.15%"),
       ("correlated", 100): ("1.35%", "2.78 ± 0.20%")}


def check_T24(D):
    z = D("appendixF_inscope.npz")
    for (pr, T), printed in T24_POP.items():
        lab = f"population {PRIOR_LABEL[pr]} T={T}"
        if z is None:
            add("T24", lab, " / ".join(printed), None, known="missing appendixF_inscope.npz")
            continue
        ps, Ps, d, fl = stats_curve(z[f"pop_T{T}_{pr}"])
        for name, pv, rv in zip(["p*", "P*", "delta", "flatness"], printed,
                                [f1(ps), f3(Ps), f"{100*d:.1f}%", f3(fl)]):
            add("T24", f"{lab}: {name}", pv, rv)
    for (pr, T), printed in T24_EMP.items():
        lab = f"empirical {PRIOR_LABEL[pr]} T={T}"
        if z is None:
            add("T24", lab, " / ".join(printed), None, known="missing appendixF_inscope.npz")
            continue
        st = np.array([stats_curve(c) for c in z[f"emp_T{T}_{pr}"]])
        rec = [f"{np.median(st[:,0]):.1f} [{st[:,0].min():.1f}, {st[:,0].max():.1f}]",
               f3(np.median(st[:, 1])),
               f"{100*np.median(st[:,2]):.1f}% [{100*st[:,2].min():.1f}, {100*st[:,2].max():.1f}]",
               f3(np.median(st[:, 3]))]
        for name, pv, rv in zip(["p* median [range]", "P*", "delta median [range]",
                                 "flatness"], printed, rec):
            add("T24", f"{lab}: {name}", pv, rv)
    if z is not None:
        add("T24", "empirical: number of seeds", "24", str(int(z["n_seed"])))
        gt = max(np.nanmax(np.abs(z[f"pop_T{T}_gaussian"] - z[f"pop_T{T}_heavy_tail"]))
                 for T in [1, 100])
        add("F prose", "t3 and Gaussian population curves coincide (max |diff|)",
            "0", f"{gt:.1e}", ok=gt < 1e-12)


def check_F_prose_and_T41(D):
    # control set: 20 rooms, 16 with discarded modes
    n_all = n_in = 0
    for i in range(800, 820):
        p = MODAL / f"scene_{i:05d}" / "eigenpairs.npz"
        if p.exists():
            n_all += 1
            n_in += len(np.load(p, allow_pickle=True)["eigenvalues"]) > 50
    add("T41", "control-set rooms / rooms with discarded modes", "20 / 16",
        f"{n_all} / {n_in}")
    z = D("appendixF_inscope.npz")
    if z is not None:
        add("F prose", "rooms in the primary test with K_total > K", "16", str(len(z["rooms"])))
        add("F prose", "61-point p grid", "61", str(len(z["p_grid"])))
        pop = [stats_curve(z[f"pop_T{T}_{pr}"]) for T in [1, 100] for pr in PRIOR_LABEL]
        dev = max(abs(s[0] - S_HAT) for s in pop)
        add("F prose", "population p* equals |s| at grid resolution (max |p*-1.13|)",
            "< 0.05", f2(dev), ok=dev < 0.05 + 1e-9)
        emp_d = max(np.median([stats_curve(c)[2] for c in z[f"emp_T{T}_{pr}"]])
                    for T in [1, 100] for pr in PRIOR_LABEL)
        add("F prose", "empirical median delta below one percent (largest median)",
            "< 1%", f"{100*emp_d:.1f}%", ok=emp_d < 0.01)
        pg, pt = [np.median([stats_curve(c)[0] for c in z[f"emp_T100_{pr}"]])
                  for pr in ["gaussian", "heavy_tail"]]
        add("F prose", "Gaussian and t3 give the same median p* at T=100", "equal",
            f"{pg:.1f} / {pt:.1f}", ok=f1(pg) == f1(pt))
        for T in [1, 100]:
            key = f"fixed_T{T}"
            if key not in z.files:
                add("T41", f"pooled ratio under t3 vs Gaussian, T={T}", "about 6% higher",
                    None, known="appendixF_inscope.npz has no fixed_T{T} key; rerun "
                    "appendixF_inscope.py")
                continue
            d = z[key]
            rel = 100 * np.median((d[:, 1] - d[:, 0]) / d[:, 0])
            add("T41", f"pooled ratio under t3 vs Gaussian, T={T} (|rel-6| <= 1)",
                "about 6% higher", f"{rel:+.2f}%", ok=abs(rel - 6) <= 1)
    add("F prose", "nu = 5 gives kurtosis 9 (3 + 6/(nu-4))", "9", f"{3 + 6/(5-4):.0f}")

    # KS gap (prior_tails_qq over 10 seeds)
    S = seed_summary(D)
    if S is None:
        add("T41", "largest KS gap of the noise marginals (10 seeds)", "0.016 (at most 0.017)",
            None, known="missing seeds/prior_tails_qq_seed*.npz")
    else:
        ks = S[0]["ks_one"]
        add("T41", "largest KS gap vs N(0,1) of the t3 / correlated noise marginals: median "
            "(max) over 10 seeds", "0.016 (at most 0.017)",
            f"{np.median(ks):.3f} (at most {ks.max():.3f})")
        z = np.load(D.dir / "seeds" / "prior_tails_qq_seed0.npz")
        add("T41", "KS gap: in-scope rooms it is computed on (400 samples each)", "187",
            str(len(z["z_gaussian"]) // 400), ok=len(z["z_gaussian"]) == 187 * 400)

    # T26 + largest single-room movement + prose on M3
    mp = D("misspec_population_187.npz")
    if mp is None:
        add("T26", "all rows", "", None, known="missing misspec_population_187.npz")
    else:
        add("F prose", "in-scope rooms of the learned-model comparison", "187",
            str(len(mp["room_ids"])))
        has_m3 = "gaussian_T1_m3or" in mp.files
        for (pr, T), (cf, m3) in T26.items():
            lab = f"{'Gaussian' if pr == 'gaussian' else 'Correlated (rho=0.3)'} T={T}"
            add("T26", f"{lab}: closed form delta", cf,
                f"{100*np.median(mp[f'{pr}_T{T}_dsl']):.2f}%")
            if has_m3:
                med = [np.median(r) for r in mp[f"{pr}_T{T}_m3or"]]
                add("T26", f"{lab}: M3 delta (mean ± sd over seeds)", m3,
                    f"{100*np.mean(med):.2f} ± {100*np.std(med):.2f}%")
            else:
                add("T26", f"{lab}: M3 delta (mean ± sd over seeds)", m3, None, known=M3_LOST)
        mv = max(abs(100 * (np.median(mp[f"correlated_T{T}_dsl"])
                            - np.median(mp[f"gaussian_T{T}_dsl"]))) for T in [1, 100])
        add("F prose", "correlation moves the closed form's median cost negligibly "
            "(max |shift|, criterion < 0.1 pp)", "negligible", f"{mv:.2f} pp", ok=mv < 0.1)
        cf_room = max(np.nanmax(np.abs(100 * (mp[f"correlated_T{T}_dsl"]
                                               - mp[f"gaussian_T{T}_dsl"]))) for T in [1, 100])
        if has_m3:
            best = None
            for T in [1, 100]:
                dlt = 100 * (mp[f"correlated_T{T}_m3or"].mean(0)
                             - mp[f"gaussian_T{T}_m3or"].mean(0))
                i = int(np.nanargmax(np.abs(dlt)))
                if best is None or abs(dlt[i]) > abs(best[0]):
                    best = (dlt[i], T, str(mp["room_ids"][i]))
            add("T41", "largest single-room movement under the correlated prior "
                "(M3, T=1, favors M3)", "13.3 pp",
                f"{abs(best[0]):.1f} pp (T={best[1]}, {best[2]})",
                ok=(f"{abs(best[0]):.1f}" == "13.3" and best[1] == 1 and best[0] < 0))
            m3mv = max(abs(100 * (np.mean([np.median(r) for r in mp[f"correlated_T{T}_m3or"]])
                                  - np.mean([np.median(r) for r in mp[f"gaussian_T{T}_m3or"]])))
                       for T in [1, 100])
            add("F prose", "correlation moves M3's median cost negligibly too "
                "(max |shift|, criterion < 0.1 pp)", "negligible", f"{m3mv:.2f} pp", ok=m3mv < 0.1)
            add("F prose", "largest single-room movement is in M3's evaluation "
                "(M3 vs closed form)", "M3 > CF", f"{abs(best[0]):.1f} vs {cf_room:.1f} pp",
                ok=abs(best[0]) > cf_room)
            below = all(np.median(mp[f"{pr}_T{T}_dsl"]) <
                        np.mean([np.median(r) for r in mp[f"{pr}_T{T}_m3or"]])
                        for pr in ["gaussian", "correlated"] for T in [1, 100])
            add("F prose", "closed form's median cost stays below M3's at both T",
                "yes", "yes" if below else "no")
        else:
            add("T41", "largest single-room movement under the correlated prior "
                "(M3, T=1, favors M3)", "13.3 pp", None, known=M3_LOST)
            add("F prose", "largest single-room movement is in M3's evaluation",
                "M3 > CF", None, known=M3_LOST)
            add("F prose", "closed form's median cost stays below M3's at both T", "yes",
                None, known=M3_LOST)
    mq = D("misspec_priors.npz")
    if mq is not None and "m3_gain_pp_T100_gaussian_emp" in mq.files:
        g = mq["m3_gain_pp_T100_gaussian_emp"]
        add("T41", "no-truncation control: median M3 gain over the best power law, T=100",
            "2.8 pp", f"{np.median(g):.1f} pp (seed range {g.min():.1f}-{g.max():.1f})",
            ok=f"{np.median(g):.1f}" == "2.8")
        if "m3_gain_pp_T100_gaussian_pop" in mq.files:
            gp = mq["m3_gain_pp_T100_gaussian_pop"]
            add("T41", "  same under exact population risk", "2.8 pp", f"{np.median(gp):.1f} pp")
    else:
        add("T41", "no-truncation control: median M3 gain over the best power law, T=100",
            "2.8 pp", None, known="no stored value in the release; " + M3_LOST)



# ======================================================================== Appendix G
RECTS = [("3x6_rect", 3.0, 6.0), ("2x8_long", 2.0, 8.0), ("4x4_square", 4.0, 4.0),
         ("3x5_rect", 3.0, 5.0)]
K_T27 = [75, 100, 200, 300]
T27 = {75: ("0.0407", "1.92 / 1.63", "1.33 [1.32, 1.35] / 1.37 [1.34, 1.43]"),
       100: ("0.0210", "1.31 / 2.59", "1.30 [1.29, 1.32] / 1.30 [1.27, 1.39]"),
       200: ("0.0081", "1.82 / 4.85", "1.29 [1.27, 1.30] / 1.31 [1.24, 1.34]"),
       300: ("0.0055", "1.89 / 6.09", "1.28 [1.27, 1.29] / 1.29 [1.21, 1.34]")}


def rect_flat_medians(mk):
    """median over the 4 rectangles of max/min P over [0,3], per K_total."""
    pg = mk["p_grid"]
    out = {}
    for K in [int(k) for k in mk["K_totals"]]:
        v = [flat03(mk[f"rect:{n}_K{K}_med"], pg) for n, *_ in RECTS]
        out[K] = float(np.median(v))
    return out


def rect_checks():
    """G.1-G.3 rows from the released rectangular_*.npz and analytic recomputes."""
    from rectangular_control import rect_eigenpairs, simulate_room, GAMMA, C_SOUND
    from src.physics.temporal import build_wave_temporal_matrix
    from src.estimation.ridge import ridge_sweep_svd, build_gamma_diag

    nn = np.load(EXP / "rectangular_nnsd.npz")
    D_, p_ = nn["convex_ks_goe_pooled"]
    add("T42", "NNSD generic convex rooms: D_KS vs GOE (p)", "0.010 (0.17)",
        f"{D_:.3f} ({p_:.2f})")
    add("T42", "NNSD generic convex rooms: number of contributing rooms", "48",
        str(len(nn["convex_ks_goe_per_room"])))
    ns = len(nn["convex_pooled_spacings"])
    add("T42", "NNSD generic convex rooms: spacings (within 10%)", "about 13,000",
        f"{ns:,}", ok=abs(ns / 13000 - 1) < 0.10)
    names = [n for n, *_ in RECTS]
    add("T42", "NNSD rectangles: D_KS vs GOE, 3x6 / 2x8 / 4x4 / 3x5",
        "0.40 / 0.34 / 0.58 / 0.20",
        " / ".join(f2(nn[f"rect_{n}_ks_goe"][0]) for n in names))
    pmax = max(nn[f"rect_{n}_ks_goe"][1] for n in names)
    add("T42", "NNSD rectangles: all p < 1e-5 (largest p)", "< 1e-5", f"{pmax:.1e}",
        ok=pmax < 1e-5)
    nr = sum(len(nn[f"rect_{n}_spacings"]) for n in names)
    add("T42", "NNSD rectangles: spacings (within 10%)", "about 2,000", f"{nr:,}",
        ok=abs(nr / 2000 - 1) < 0.10)
    worst = max(names, key=lambda n: nn[f"rect_{n}_ks_goe"][0])
    add("G prose", "the 4x4 square deviates most from GOE", "4x4_square", worst)
    gap = {n: nn[f"rect_{n}_ks_goe"][0] - nn[f"rect_{n}_ks_poisson"][0] for n in names}
    closer = [n for n in names if gap[n] > 1e-9]
    ties = [n for n in names if abs(gap[n]) <= 1e-9]
    add("G prose", "two rectangles lie closer to Poisson; 3x6 and 4x4 are as far from "
        "Poisson as from GOE", "closer: 2x8, 3x5; ties: 3x6, 4x4",
        f"closer: {', '.join(n.split('_')[0] for n in closer)}; ties: "
        f"{', '.join(n.split('_')[0] for n in ties) or 'none'}")

    # K_total of the rectangles: Weyl estimate vs exact counts
    lam_1k = (2 * np.pi * 1000.0 / 343.0) ** 2
    exact, weyl = [], []
    for n, Lx, Ly in RECTS:
        m = np.arange(1, 200)
        lam = np.pi ** 2 * (m[:, None] ** 2 / Lx ** 2 + m[None, :] ** 2 / Ly ** 2)
        exact.append(int(np.sum(lam <= lam_1k)))
        weyl.append(int(np.ceil(Lx * Ly * 200 / (4 * np.pi))))
    add("G prose", "rectangle K_total 3x6 / 2x8 / 4x4 / 3x5 (Weyl estimate "
        "ceil(A*200/(4*pi)))", "287 / 255 / 255 / 239", " / ".join(map(str, weyl)))
    add("G prose", "  (exact Dirichlet counts below 1000 Hz, for reference; not printed)",
        "-", " / ".join(map(str, exact)), ok=True)
    add("T42", "3x6 at K_total=50: discarded modes at its full K_total (Weyl estimate)",
        "237", str(weyl[0] - 50))
    areas = [Lx * Ly for _, Lx, Ly in RECTS]
    add("G prose", "rectangle areas", "15–18 m2", f"{min(areas):.0f}–{max(areas):.0f} m2")
    lmin = min(rect_eigenpairs(Lx, Ly, 50)[0].min() for _, Lx, Ly in RECTS)
    add("G prose", "every lambda_k > 1 (smallest rectangle eigenvalue)", "> 1", f2(lmin),
        ok=lmin > 1)

    # flatness of the 3x6 room vs K_total, generic reference
    sw = np.load(EXP / "rectangular_ktotal_sweep.npz")
    Ks = [int(k) for k in sw["K_total_values"]]
    r = dict(zip(Ks, sw["3x6_rect_ratios"]))
    psw = np.load(EXP / "p_sweep" / "p_sweep_K50_M8.npz", allow_pickle=True)
    P = psw["P_oracle"][:, list(psw["T_values"]).index(100), list(psw["M_values"]).index(8), :]
    Pc = P[:, psw["p_values"] <= 3.0]
    ref = float(np.median(Pc.max(1) / Pc.min(1)))
    add("T42", "3x6 flatness at K_total = 50/75/100/150/200/287 (generic reference)",
        "3.41 / 1.29 / 1.26 / 1.24 / 1.23 / 1.22 (1.28)",
        " / ".join(f2(r[K]) for K in [50, 75, 100, 150, 200, 287]) + f" ({ref:.2f})")
    p50 = sw["3x6_rect_p_stars"][Ks.index(50)]
    add("G prose", "3x6 at K_total = 50: p* = 0", "0", f1(p50).rstrip("0").rstrip(".") or "0",
        ok=p50 == 0)
    add("G prose", "by K_total = 100 the 3x6 ratio reaches the generic reference",
        "<= ref", f"{r[100]:.3f} vs {ref:.3f}", ok=r[100] <= ref)
    seq = [r[K] for K in sorted(K for K in Ks if 100 <= K <= 287)]
    add("G prose", "3x6 ratio keeps falling up to the full mode count (K_total 100..287)",
        "decreasing", "decreasing" if all(np.diff(seq) < 0) else "not monotone")
    within = min(K for K in Ks if r[K] <= 1.05 * ref) - 50
    add("T42", "discarded modes bringing the 3x6 ratio within 5% of the reference",
        "about 25", str(within), ok=abs(within - 25) <= 5)

    # condition number of Phi~ and the best alpha at K_total = 50 (T = 100)
    room = simulate_room(3.0, 6.0, np.random.default_rng(42), K_total=50)
    T = 100
    A = build_wave_temporal_matrix(room["Phi"], room["eigenvalues"], room["dt_sim"], T,
                                   GAMMA, C_SOUND)
    cond = np.linalg.cond(A)
    add("T42", "3x6 at K_total=50: condition number of Phi~ (order of magnitude)", "~10^8",
        f"{cond:.2e}", ok=int(np.floor(np.log10(cond))) == 8)
    add("G prose", "Phi~ has full column rank, MT = 800 > 2K = 100",
        "rank 100 of 100, 800 rows",
        f"rank {np.linalg.matrix_rank(A)} of {A.shape[1]}, {A.shape[0]} rows")
    sps, ns, y = room["steps_per_snap"], room["n_snaps"], room["y_click"]
    valid = [si for si in range(5, ns - 2) if si * sps + 1 - T >= 0
             and si * sps + 1 <= y.shape[1]]
    Y = np.stack([y[:, si * sps + 1 - T: si * sps + 1].reshape(-1) for si in valid])
    tg = room["a"][:, valid].T
    lam_grid = np.array([1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4, 1e6, 1e8,
                         1e10])
    best = [ridge_sweep_svd(A, Y, tg, build_gamma_diag(room["eigenvalues"], p, True),
                            lam_grid, True)[1] for p in P_GRID[P_GRID <= 2.0 + 1e-9]]
    add("G prose", "3x6 at K_total=50: best alpha is the grid minimum at every p <= 2",
        "1e-06 at all p", f"{sum(b == 0 for b in best)} of {len(best)} p values",
        ok=all(b == 0 for b in best))


def check_G(D):
    rect_checks()

    co, cd, dk = (D("results_cohort_flatness.npz"), D("cohort_d2h.npz"),
                  D("results_d_vs_ktotal.npz"))
    SS = seed_summary(D)
    S = SS[0] if SS is not None else None
    for K in K_T27:
        H_p, d2h_p, fl_p = T27[K]
        # H and D/(2H), generic side from the 101-room cohort
        if cd is not None:
            a = cd[f"K{K}"]
            q = a[:, 1] / (2 * a[:, 0])
            add("T27", f"K_total={K}: H (cohort median)", H_p, f4(np.median(a[:, 0])))
            gen_d2h = f2(np.median(q))
        else:
            add("T27", f"K_total={K}: H (cohort median)", H_p, None,
                known="missing cohort_d2h.npz")
            gen_d2h = None
        if dk is not None:
            rows = [dk[f"rect_{n}_K{K}"] for n, *_ in RECTS]
            rect_d2h = f2(np.median([r[1] / (2 * r[0]) for r in rows]))
            rH = np.median([r[0] for r in rows])
            add("T27", f"K_total={K}: H of the rectangles (caption: families share H; "
                "within 10%)", H_p, f4(rH), ok=abs(rH / float(H_p) - 1) < 0.10)
            g6 = np.median([dk[f"generic_{s}_K{K}"][0] for s in dk["generic_rooms"]])
            add("T27", f"K_total={K}: H, 6-room matched-K_total set", H_p, f4(g6))
        else:
            rect_d2h = None
        add("T27", f"K_total={K}: D/(2H) generic / rectangle", d2h_p,
            f"{gen_d2h} / {rect_d2h}" if gen_d2h and rect_d2h else None,
            known=None if gen_d2h and rect_d2h else "missing cohort_d2h.npz or "
            "results_d_vs_ktotal.npz")
        if S is not None:
            add("T27", f"K_total={K}: flatness generic / rectangle, median [range] over 10 "
                "seeds", fl_p, f"{mr(S[f'flat_gen_K{K}'])} / {mr(S[f'flat_rect_K{K}'])}")
        else:
            add("T27", f"K_total={K}: flatness generic / rectangle", fl_p, None,
                known="missing seeds/ (run_multi_seed.sh)")

    # T42 cohort rows
    if co is not None:
        add("T42", "matched-count cohort size", "101", str(len(co["cohort"])))
        if cd is not None:
            add("G prose", "D/(2H) cohort equals the flatness cohort", "same rooms",
                "same rooms" if list(cd["cohort"]) == list(co["cohort"]) else "different")
    if S is not None:
        add("T42", "flatness with no discarded modes (K_total=50), median [range] over 10 "
            "seeds: generic / rectangles", "2.34 [2.28, 2.35] / 2.24 [2.09, 2.64]",
            f"{mr(S['flat_gen_K50'])} / {mr(S['flat_rect_K50'])}")
        ratio = S["flat_gen_K50"] / S["flat_rect_K50"]
        add("G prose", "with no discarded modes the two families are equally curved within "
            "the scatter over seeds (generic/rectangle ratio straddles 1)", "straddles 1",
            f"{ratio.min():.2f}-{ratio.max():.2f}", ok=ratio.min() < 1 < ratio.max())
        gap = {K: abs(np.median(S[f"flat_gen_K{K}"]) - np.median(S[f"flat_rect_K{K}"]))
               for K in [75, 100]}
        add("G prose", "a few dozen discarded modes flatten both to nearly the same level "
            "(|median gap| at K_total = 75 / 100, criterion < 0.05)", "< 0.05",
            f"{gap[75]:.3f} / {gap[100]:.3f}", ok=max(gap.values()) < 0.05)
        words = {6: "six", 5: "five", 4: "four", 3: "three", 2: "two", 1: "one", 0: "zero"}
        ins = sorted({int(x) for x in S["inside_iqr"]})
        add("T42", "flatness medians at K_total=300: difference, median [range] over 10 "
            "seeds / rectangle median inside the generic IQR", "0.04 [0.004, 0.07] / four "
            "or five of six checkpoints",
            f"{mr(S['diff_K300'], fmt_lo='{:.3f}')} / "
            + " or ".join(words[i] for i in ins) + " of six checkpoints")
    if cd is not None and dk is not None:
        a = cd["K300"]
        q = a[:, 1] / (2 * a[:, 0])
        rmin = min(dk[f"rect_{n}_K300"][1] / (2 * dk[f"rect_{n}_K300"][0]) for n, *_ in RECTS)
        add("T42", "D/(2H) at K_total=300: generic IQR upper bound / rectangle minimum",
            "2.38 / 5.80", f"{np.percentile(q, 75):.2f} / {rmin:.2f}")
        add("G prose", "at K_total=300 the D/(2H) ranges do not overlap "
            "(generic max < rectangle min)", "no overlap",
            f"generic max {q.max():.2f} vs {rmin:.2f}", ok=q.max() < rmin)
        dips = []
        for j in range(len(a)):
            v = {K: cd[f"K{K}"][j, 1] / (2 * cd[f"K{K}"][j, 0]) for K in [75, 100, 150]}
            dips.append(v[100] < v[75] and v[100] < v[150])
        rdip = [dk[f"rect_{n}_K100"][1] / (2 * dk[f"rect_{n}_K100"][0])
                < min(dk[f"rect_{n}_K{K}"][1] / (2 * dk[f"rect_{n}_K{K}"][0]) for K in [75, 150])
                for n, *_ in RECTS]
        add("G prose", "generic D/(2H) dip at K_total=100 reproducible across the cohort "
            "(majority of rooms) and absent in rectangles", "dip / no dip",
            f"{np.mean(dips):.0%} of rooms / {sum(rdip)} of 4 rectangles",
            ok=np.mean(dips) > 0.5 and not any(rdip))
    if dk is not None:
        s2h = [dk[f"rect_{n}_K{K}"][2] / (2 * dk[f"rect_{n}_K{K}"][0])
               for n, *_ in RECTS for K in [75, 100, 150, 200, 300]]
        add("G prose", "rectangles: S/(2H) = (9/4 - 1)/2 in every cell", "0.625",
            f"{min(s2h):.3f}–{max(s2h):.3f}", ok=max(abs(np.array(s2h) - 0.625)) < 1e-9)
        box = {f"{bc}_{n}": [dk[f"{bc}_{n}_K{K}"][1] / (2 * dk[f"{bc}_{n}_K{K}"][0])
                             for K in [75, 100, 150, 200, 300]]
               for bc in ["boxD", "boxN"] for n in ["box_small", "box_mid1", "box_mid2",
                                                    "box_large"]}
        grow = sum(all(np.diff(v) > 0) for v in box.values())
        add("G prose", "3D boxes: D/(2H) grows with mode count (cells increasing over "
            "K_total = 75..300)", "growing", f"{grow} of {len(box)} cells",
            ok=grow == len(box))
        if cd is not None:
            gen300 = np.median(cd["K300"][:, 1] / (2 * cd["K300"][:, 0]))
            bmin = min(v[-1] for v in box.values())
            add("G prose", "3D boxes: D/(2H) an order of magnitude above generic rooms "
                "(smallest box / generic median at K_total=300, criterion >= 10)",
                ">= 10x", f"{bmin/gen300:.1f}x", ok=bmin / gen300 >= 10)

    hd = D.first("hd_decomposition.npz", "results_HD.npz")
    if hd is None:
        add("T42", "D/(2H) over the full band: matched cohort / all in-scope rooms",
            "2.24 / 2.16", None, known="missing hd_decomposition.npz (written by "
            "scripts/appendix/A_B/hd_decomposition.py)")
    else:
        q = hd["D"] / (2 * hd["H"])
        ids = [int(r) for r in hd["room_ids"]]
        coh = ([int(str(s).split("_")[1]) for s in co["cohort"]] if co is not None else [])
        qc = np.median([q[ids.index(r)] for r in coh if r in ids]) if coh else np.nan
        leg = [k for k, r in enumerate(ids)
               if (MODAL / f"scene_{r:05d}" / "eigenpairs.npz").exists()
               and len(np.load(MODAL / f"scene_{r:05d}" / "eigenpairs.npz")["eigenvalues"]) > 50]
        add("T42", f"D/(2H) over the full band: matched cohort / all {len(leg)} in-scope rooms",
            "2.24 / 2.16", f"{qc:.2f} / {np.median(q[leg]):.2f}")

    # T28 and T42 3D/2D rows
    from t28_t42_summary import summary_2d, summary_3d
    d2 = summary_2d()
    bx_path = D.dir / "box3d_flatness.npz"
    d3 = summary_3d(bx_path) if bx_path.exists() else None
    if d3 is None:
        MISSING_FILES.add("box3d_flatness.npz")
    add("T28", "cases (2D)", "187 in-scope rooms", f"{d2['n_in_scope']} in-scope rooms")
    add("T28", "median K_total (2D)", "337", f"{d2['med_K_total']:.0f}")
    add("T28", "median H (2D)", "0.0050", f4(d2["med_H"]))
    add("T28", "median landscape ratio T=1/100/1000 (2D)", "1.265 / 1.280 / 1.452",
        " / ".join(f3(d2[f"med_flat_T{T}"]) for T in [1, 100, 1000]))
    add("T28", "cost T=100 (2D)", "0.87 pp (1.46%)",
        f"{d2['med_cost_pp_T100']:.2f} pp ({d2['med_cost_rel_T100']:.2f}%)")
    add("T28", "cost T=1000 (2D)", "0.62 pp (5.83%)",
        f"{d2['med_cost_pp_T1000']:.2f} pp ({d2['med_cost_rel_T1000']:.2f}%)")
    add("T42", "2D boundary-inclusive maxima, T<=100: relative (absolute) of the worst room",
        "27.5% (10.61 pp), room 00806",
        f"{d2['max_rel_Tle100']:.1f}% ({d2['abs_at_max_rel_Tle100']:.2f} pp), room "
        f"{d2['max_rel_Tle100_room'].split('_')[-1]}")
    add("T42", "2D maxima, T<=100: the worst-relative room also has the largest absolute gap",
        "10.61 pp", f"{d2['max_abs_Tle100']:.2f} pp")
    add("T42", "2D boundary-inclusive maxima, T=1000: largest relative / largest absolute "
        "(different rooms)", "34.8% / 5.33 pp (different rooms)",
        f"{d2['max_rel_T1000']:.1f}% / {d2['max_abs_T1000']:.2f} pp ("
        + ("different rooms" if d2["max_rel_T1000_room"] != d2["max_abs_T1000_room"]
           else "same room") + ")")
    add("G prose", "Weyl gives practical rooms hundreds of discarded modes "
        "(median K_total - K, in-scope)", ">= 100", f"{d2['med_K_total'] - 50:.0f}",
        ok=d2["med_K_total"] - 50 >= 100)
    if d3 is not None:
        add("T28", "median K_total (3D)", "530", f"{d3['med_K_total']:.0f}")
        add("T28", "median H (3D)", "0.0027", f4(d3["med_H"]))
    if SS is None:
        add("T28", "3D column (10 sensor layouts)", "", None,
            known="missing seeds/ (run_multi_seed.sh)")
        return
    S, d2s, d3s = SS
    n_lay = len(d3s)
    add("T28", "cases (3D)", "8 cells x 10 layouts", f"{d3s[0]['n_cells']} cells x {n_lay} layouts")
    add("T28", "median landscape ratio T=1/100/1000 (3D), median over layouts",
        "1.089 / 1.138 / 1.529", " / ".join(f3(np.median(S[f"d3_med_flat_T{T}"]))
                                            for T in [1, 100, 1000]))
    for T, pr in [(100, "1.03 pp (1.45%)"), (1000, "1.57 pp (13.58%)")]:
        add("T28", f"cost T={T} (3D), median over layouts", pr,
            f"{np.median(S[f'd3_med_cost_pp_T{T}']):.2f} pp "
            f"({np.median(S[f'd3_med_cost_rel_T{T}']):.2f}%)")
    add("T28", "3D cost range over layouts, T=100 / 1000", "0.69–1.43 / 1.39–2.10 pp",
        " / ".join(f"{S[f'd3_med_cost_pp_T{T}'].min():.2f}–{S[f'd3_med_cost_pp_T{T}'].max():.2f}"
                   for T in [100, 1000]) + " pp")
    k3 = d3s[0]
    same_geo = all(d["K_total_min"] == k3["K_total_min"] and d["n_real"] == k3["n_real"]
                   for d in d3s)
    add("T42", "3D boxes: volumes / K_total below 500 Hz / cells / sensor layouts / "
        "realizations per layout", "18–126 m3 / 169–1,921 / 8 / 10 / 20",
        f"{k3['V_min']:.0f}–{k3['V_max']:.0f} m3 / {k3['K_total_min']}–"
        f"{k3['K_total_max']:,} / {k3['n_cells']} / {n_lay} / {k3['n_real']}",
        ok=same_geo and f"{k3['V_min']:.0f}–{k3['V_max']:.0f} m3 / {k3['K_total_min']}–"
        f"{k3['K_total_max']:,} / {k3['n_cells']} / {n_lay} / {k3['n_real']}"
        == "18–126 m3 / 169–1,921 / 8 / 10 / 20")
    add("T42", "3D worst cells over 10 layouts, relative median [range] (absolute, median): "
        "T<=100 / T=1000", "3.4% [2.7, 3.8] (2.8 pp) / 20.4% [15.6, 34.2] (1.7 pp)",
        " / ".join(f"{mr(S[f'd3_max_rel_{t}'], '{:.1f}').replace(' [', '% [', 1)} "
                   f"({np.median(S[f'd3_abs_at_max_rel_{t}']):.1f} pp)"
                   for t in ["Tle100", "T1000"]))
    add("G prose", "all eight 3D cells lie within the observed 2D room-to-room range for every "
        "sensor layout (T = 1, 100, 1000)", "10 of 10 layouts",
        f"{int(S['in_2d_range'].sum())} of {n_lay} layouts")
    r_lo, r_hi = S["ratio_2d_3d_Tle100"], S["ratio_2d_3d_T1000"]
    add("G prose", "worst 3D cell costs several times less than the worst 2D room at "
        "T<=100 (2D/3D ratio over layouts, criterion >= 3 in every layout)", ">= 3x",
        f"{mr(r_lo, '{:.1f}')}x", ok=r_lo.min() >= 3)
    add("G prose", "... and less, by up to a factor of two, at T=1000 (2D/3D ratio over "
        "layouts, criterion 1 < ratio <= 2.25)", "1-2x", f"{mr(r_hi, '{:.2f}')}x",
        ok=r_hi.min() > 1 and r_hi.max() <= 2.25)
    fl = S["flatter_than_2d"].all(axis=0)
    nf = S["flatter_than_2d"].any(axis=0)
    add("G prose", "3D flatter than 2D at T<=100 but not at T=1000 (every layout)",
        "yes / yes / no", " / ".join("yes" if f else ("no" if not n else "mixed")
                                     for f, n in zip(fl, nf)))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-dir", default=str(OUT))
    args = ap.parse_args()
    check_paper_dataset()
    D = Data(args.data_dir)
    check_T24(D)
    check_F_prose_and_T41(D)
    check_G(D)

    order = ["T24", "T26", "T41", "F prose", "T27", "T28", "T42", "G prose"]
    rows = sorted(ROWS, key=lambda r: order.index(r[0]))
    w = [7, 80, 34, 34, 12]
    print(f"data: {args.data_dir}\n")
    print(" | ".join(h.ljust(n) for h, n in zip(["table", "row", "printed", "recomputed",
                                                  "status"], w)) + " | note")
    print("-" * 200)
    for t, r, p, c, s, note, unexp in rows:
        print(" | ".join(x.ljust(n) for x, n in zip([t, r, p, c, s + ("!" if unexp else "")],
                                                    w)) + (" | " + note if note else ""))
    from collections import Counter
    cnt = Counter(r[4] for r in rows)
    n_unexp = sum(r[6] for r in rows)
    print(f"\n{dict(cnt)}; unexpected DIFF (marked !): {n_unexp}")
    if MISSING_FILES:
        print("missing inputs: " + ", ".join(sorted(MISSING_FILES)))
    sys.exit(1 if n_unexp else 0)


if __name__ == "__main__":
    main()
