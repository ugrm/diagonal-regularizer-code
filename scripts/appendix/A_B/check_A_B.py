#!/usr/bin/env python
"""
Recompute every number printed in Tables T32, T35, T4, T36, T5, T6 and the numeric claims
in the prose of Appendices A and B of the paper, from the shipped data and the
outputs of the scripts in this folder.

Prints one line per check: [table, row, printed, recomputed, status]. Status is OK, DIFF
(unexpected; exit code 1) or DIFF(known) (a printed value or description that the data do not
support, listed in README.md in this folder; exit code unaffected).

Prerequisites (run in this order, from the repository root):
    python scripts/appendix/A_B/run_berry_qq.py
    python scripts/appendix/A_B/room_diagnostic_regression.py
    DR_ROOMS=... python scripts/appendix/A_B/hd_decomposition.py
    DR_ROOMS=... python scripts/appendix/A_B/verify_tracefree_identity.py
    python scripts/appendix/A_B/aniso_sensitivity_sweep.py
Usage:
    python scripts/appendix/A_B/check_A_B.py
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy import special, stats

sys.dont_write_bytecode = True     # the helper scripts imported below stay cache-free
REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from src.utils.paths import EXP, MODAL, SUPP, check_paper_dataset  # noqa: E402

APP = EXP / "appendix"
K, M = 50, 8
T_IDX = None          # filled in main(): T value -> column of the cost arrays

# Printed values that the data do not support (see README.md). The check still prints
# printed and recomputed values; these keys only stop them from failing the exit code.
KNOWN = {
}

ROWS = []


def add(table, label, printed, recomputed, ok, key=None):
    if ok:
        status = "OK"
    elif key is not None and key in KNOWN:
        status = "DIFF(known)"
    else:
        status = "DIFF"
    ROWS.append((table, label, printed, recomputed, status))


def agree(printed, value, decimals):
    """True if `value` rounds to `printed` at `decimals` places (half-unit tolerance)."""
    return abs(float(value) - float(printed)) <= 0.5 * 10.0 ** (-decimals) + 1e-9


def load(name):
    path = APP / f"{name}.npz"
    if not path.exists():
        sys.exit(f"{path} missing: run scripts/appendix/A_B/{name}.py first")
    return np.load(path, allow_pickle=True)


def import_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def dist_to_boundary(p, v):
    d = []
    for i in range(len(v)):
        a, b = v[i], v[(i + 1) % len(v)]
        ab = b - a
        t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
        d.append(np.linalg.norm(p - (a + t * ab)))
    return min(d)


# ----------------------------------------------------------------------------- data
def load_rooms(ids, s):
    """Per-room quantities from the paper's modal dataset, in p_sweep room order."""
    R = {k: [] for k in ("K_total", "nseg", "area", "H", "Phi", "mic", "verts", "var_a")}
    for sid in ids:
        d = MODAL / sid
        md = json.load(open(d / "metadata.json"))
        eig = np.load(d / "eigenpairs.npz")
        ev = eig["eigenvalues"].astype(float)
        R["K_total"].append(len(ev))
        R["nseg"].append(md["n_segments"])
        R["area"].append(md["room_area_m2"])
        if len(ev) > K:
            w = ev[K:] ** (-s)
            R["H"].append((w ** 2).sum() / w.sum() ** 2)
            R["var_a"].append(np.var(np.load(d / "modal_trajectories.npz")["a"].astype(float),
                                     axis=1)[K:])
        else:
            R["H"].append(np.nan)
            R["var_a"].append(None)
        R["Phi"].append(np.load(d / "measurement_matrix.npy").astype(float)[:M])
        R["mic"].append(np.load(d / "mic_pos.npy")[:M])
        R["verts"].append(eig["room_vertices"])
    for k in ("K_total", "nseg", "area", "H"):
        R[k] = np.array(R[k])
    R["ev"] = [np.load(MODAL / sid / "eigenpairs.npz")["eigenvalues"].astype(float) for sid in ids]
    return R


# ----------------------------------------------------------------------------- T32
def check_T32(ids, R, s, keep, eop, cost, hd, tf, noise, ref_mc):
    nt = R["K_total"] - K
    add("T32", "discarded modes, median 197 rooms", "263", f"{np.median(nt):.0f}",
        np.median(nt) == 263)
    add("T32", "discarded modes, median 187 in-scope", "287", f"{np.median(nt[keep]):.0f}",
        np.median(nt[keep]) == 287)
    Hmed = np.median(R["H"][keep])
    add("T32", "H median (187 in-scope)", "0.005", f"{Hmed:.5f}", agree(0.005, Hmed, 3))

    # D/(2H) on the 187 in-scope rooms (hd covers the 189 rooms with K_total > 50 in ROOMS)
    hd_ids = [f"scene_{int(r):05d}" for r in hd["room_ids"]]
    d2h_all = hd["D"] / (2 * hd["H"])
    in187 = np.array([r in set(eop["scene_ids"]) for r in hd_ids])
    d2h = d2h_all[in187]
    q25, q50, q75 = np.percentile(d2h, [25, 50, 75])
    add("T32", "D/(2H) median [IQR] (187 rooms)", "2.16 [1.70, 2.72]",
        f"{q50:.4f} [{q25:.3f}, {q75:.3f}] (n={len(d2h)}; 189-room run: "
        f"{np.median(d2h_all):.3f} [{np.percentile(d2h_all, 25):.3f}, {np.percentile(d2h_all, 75):.3f}])",
        agree(2.16, q50, 2) and agree(1.70, q25, 2) and agree(2.72, q75, 2), "T32 D/(2H)")
    excess = 100 * ((M * (M - 1) + 2 * M * q50) / (M * (M + 1)) - 1)
    add("T32", "excess of E||E||_F^2 over Berry D=2H", "26%",
        f"{excess:.1f}% = (56 + 16*{q50:.3f})/72 - 1", agree(26, excess, 0))
    jen = np.sqrt(Hmed * (M * (M - 1) + 2 * M * q50))
    jen_b = np.sqrt(M * (M + 1) * Hmed)
    add("T32", "Jensen bound (under D=2H)", "0.67 (0.60)",
        f"{jen:.4f} ({jen_b:.4f}) at median H={Hmed:.5f}",
        agree(0.67, jen, 2) and agree(0.60, jen_b, 2))
    E = eop["E_op_per_room"]
    add("T32", "empirical ||E||_op median / mean / 95th", "0.58 / 0.88 / 2.42",
        f"{np.median(E):.4f} / {E.mean():.4f} / {np.percentile(E, 95):.4f} (n={len(E)})",
        agree(0.58, np.median(E), 2) and agree(0.88, E.mean(), 2)
        and agree(2.42, np.percentile(E, 95), 2))
    d1000 = cost["per_room_delta_rel"][T_IDX[1000]]
    dmap = dict(zip(ids, d1000))
    rho, p = stats.spearmanr(E, [dmap[r] for r in eop["scene_ids"]])
    add("T32", "Spearman rho(||E||_op, delta(T=1000))", "-0.30 (p < 1e-4)",
        f"{rho:+.4f} (p = {p:.1e}, n={len(E)})", agree(-0.30, rho, 2) and p < 1e-4)

    qq = load("run_berry_qq")
    ks = stats.kstest(qq["all_Csq"], "chi2", args=(1,))
    add("T32", "pooled KS vs chi2(1), N = 77,968", "0.037",
        f"{ks.statistic:.5f} (N={len(qq['all_Csq'])})",
        agree(0.037, ks.statistic, 3) and len(qq["all_Csq"]) == 77968)

    r = np.asarray(ref_mc)
    real_off = np.median(hd["mc_off_ratio"])
    add("T32", "MC off-diagonal identity E[E_ij^2]/H (50,000 i.i.d. Gaussian random-wave draws, 6 rooms)",
        "1.00 +- 0.01",
        f"{r.mean():.4f} +- {r.std():.4f} (range {r.min():.3f}-{r.max():.3f}) from "
        f"15_verify_appendix_a.check_Eij_MC = 50,000 i.i.d. Gaussian Phi ~ N(0, 1/M) draws; "
        f"the sensor-draw check on the real eigenfunctions gives {real_off:.3f} "
        f"({len(hd['mc_rooms'])} rooms x {int(hd['n_draws'])} draws)",
        agree(1.00, r.mean(), 2) and r.std() <= 0.01, "T32 MC identity")

    S2H_all = hd["S"] / (2 * hd["H"])
    add("T32", "single-mode part S/(2H)", "0.877",
        f"{np.median(S2H_all):.4f} (189 rooms; 187-room subset {np.median(S2H_all[in187]):.4f})",
        agree(0.877, np.median(S2H_all), 3))
    rect = hd["rect_D"] / (2 * hd["rect_H"])
    box = np.concatenate([hd["box_d_D"] / (2 * hd["box_d_H"]),
                          hd["box_n_D"] / (2 * hd["box_n_H"])])
    boxd = hd["box_d_D"] / (2 * hd["box_d_H"])
    add("T32", "D/(2H) separable: rectangles / 3D boxes", "6.6-8.3 / 15-76",
        f"{rect.min():.3f}-{rect.max():.3f} / {box.min():.2f}-{box.max():.2f} "
        f"(Dirichlet {boxd.min():.2f}-{boxd.max():.2f})",
        agree(6.6, rect.min(), 1) and agree(8.3, rect.max(), 1)
        and agree(15, box.min(), 0) and agree(76, box.max(), 0))
    add("T32", "sensor-draw MC (32 rooms, 500 draws): off-diag / diag ratio", "0.97 / 0.99",
        f"{real_off:.4f} / {np.median(hd['mc_diag_ratio']):.4f} "
        f"({len(hd['mc_rooms'])} rooms, {int(hd['n_draws'])} draws)",
        agree(0.97, real_off, 2) and agree(0.99, np.median(hd["mc_diag_ratio"]), 2)
        and len(hd["mc_rooms"]) == 32 and int(hd["n_draws"]) == 500)
    op2 = np.median(hd["mc_OP2"] / hd["mc_pred"])
    add("T32", "||E||_op^2 / E||E||_F^2 at the median", "0.51", f"{op2:.4f}", agree(0.51, op2, 2))

    # trace factor 1/(1 + dbar), dbar = tr(E_pop)/M, at the actual 8 microphones of each room
    dbar = {"pop": [], "real": []}
    for i in np.where(keep)[0]:
        Pt = R["Phi"][i][:, K:]
        for tag, w in (("pop", R["ev"][i][K:] ** (-s)), ("real", R["var_a"][i])):
            n = min(len(w), Pt.shape[1])
            w = w[:n] / w[:n].sum()
            dbar[tag].append(R["area"][i] * np.sum(w * Pt[:, :n] ** 2) / M - 1)
    f_pop = 1 / (1 + np.array(dbar["pop"]))
    f_real = 1 / (1 + np.array(dbar["real"]))
    src = np.median(tf["sd_db_real"])
    add("T32", "std of the trace factor between E_tf and E-bar over sensor draws (median over 30 rooms)", "0.08",
        f"{src:.4f} = median over {len(tf['mc_rooms'])} rooms of the within-room SD "
        f"of dbar over 800 sensor draws (realized weights; population weights "
        f"{np.median(tf['sd_db_pop']):.4f}); across-room SD of 1/(1+dbar) at the actual mics: "
        f"{f_real.std(ddof=1):.3f} realized / {f_pop.std(ddof=1):.3f} population weights",
        agree(0.08, src, 2) and len(tf["mc_rooms"]) == 30, "T32 trace-factor std")
    slack = jen / np.median(E)
    add("T32", "Frobenius estimate vs empirical median ||E||_op", "within 16%",
        f"{jen:.4f} / {np.median(E):.4f} = {slack:.3f} (tracefree run: "
        f"{np.median(tf['corrected']) / np.median(tf['op']):.3f})",
        agree(16, 100 * (slack - 1), 0))
    emap = dict(zip(eop["scene_ids"], E))
    both = [(d2h_all[j], emap[r]) for j, r in enumerate(hd_ids) if r in emap]
    rho = stats.spearmanr(*np.array(both).T).correlation
    add("T32", "Spearman rho(D/(2H), ||E||_op)", "-0.16", f"{rho:+.4f} (n={len(both)})",
        agree(-0.16, rho, 2))
    worst = [hd_ids.index(f"scene_{r:05d}") for r in (963, 924, 860, 835, 900)]
    w2h = d2h_all[worst]
    add("T32", "D/(2H) of the worst-Berry rooms", "1.4-2.5",
        f"{w2h.min():.3f}-{w2h.max():.3f}", agree(1.4, w2h.min(), 1) and agree(2.5, w2h.max(), 1))
    qB, qA = np.median(noise["q_B_full"]), np.median(noise["q_A_full"])
    add("T32", "median noise slope q, FEM-only / total residual", "-0.09 / +0.12",
        f"{qB:+.4f} / {qA:+.4f}", agree(-0.09, qB, 2) and agree(0.12, qA, 2))
    return dict(Hmed=Hmed, d2h=d2h, q50=q50, E=E, jen=jen, d2h_all=d2h_all, hd_ids=hd_ids)


# ----------------------------------------------------------------------------- T35
def check_T35(ids, R, keep, s):
    h = R["H"][keep]
    kid = np.array(ids)[keep]
    nt = (R["K_total"] - K)[keep]
    nseg = R["nseg"][keep]
    i0 = int(np.argmin(h))
    dec = nseg == 10
    add("T35", "H smallest (large decagons, ~1000 discarded)", "0.002",
        f"{h[i0]:.5f} ({kid[i0]}, {nseg[i0]}-gon, {nt[i0]} discarded); decagons "
        f"{h[dec].min():.4f}-{h[dec].max():.4f}, up to {nt[dec].max()} discarded",
        agree(0.002, h[i0], 3) and agree(0.002, h[dec].min(), 3))
    i1 = int(np.argmax(h))
    tri = nseg == 3
    add("T35", "H largest among triangles (~30 discarded)", "0.03",
        f"{h[i1]:.4f} ({kid[i1]}, {nseg[i1]}-gon, {nt[i1]} discarded); triangle max "
        f"{h[tri].max():.4f} ({nt[tri][np.argmax(h[tri])]} discarded); 95th pct "
        f"{np.percentile(h, 95):.4f}",
        agree(0.03, h[tri].max(), 2) and 25 <= nt[tri][np.argmax(h[tri])] <= 35, "T35 H largest")
    add("T35", "H median over in-scope rooms", "0.005", f"{np.median(h):.5f}",
        agree(0.005, np.median(h), 3))
    j = ids.index("scene_00850")
    add("T35", "room 00850: K_total / discarded", "483 / 433",
        f"{R['K_total'][j]} / {R['K_total'][j] - K} ({R['nseg'][j]}-gon)",
        R["K_total"][j] == 483 and R["nseg"][j] == 8)
    add("T35", "room 00850: H / 1/H", "0.0038 / 260",
        f"{R['H'][j]:.5f} / {1 / R['H'][j]:.1f}",
        agree(0.0038, R["H"][j], 4) and agree(260, 1 / R["H"][j], 0))
    s_p = 1.13
    lim = special.zeta(2 * s_p, K + 1) / special.zeta(s_p, K + 1) ** 2
    cont = (s_p - 1) ** 2 / ((2 * s_p - 1) * K)
    add("T35", "limit of H as K_total -> inf (|s| = 1.13, K = 50)", "~0.0003",
        f"{lim:.6f} (Weyl lambda_n ~ n: zeta(2s,51)/zeta(s,51)^2; continuum "
        f"(s-1)^2/((2s-1)K) = {cont:.6f})", agree(0.0003, lim, 4))
    # prose (A.3)
    jd, jt = np.where(dec)[0][np.argmin(h[dec])], np.where(tri)[0][np.argmax(h[tri])]
    ratio = h[jt] / h[jd]
    add("A prose", "H an order of magnitude smaller in decagons than in triangles",
        "order of magnitude", f"max triangle / min decagon H = {ratio:.1f}",
        10 ** 0.5 <= ratio < 10 ** 1.5)
    add("A prose", "the large decagons discard ~a thousand modes, the small triangles a few dozen",
        "~1000 / a few dozen",
        f"min-H decagon {kid[jd]} {nt[jd]}, max-H triangle {kid[jt]} {nt[jt]} discarded "
        f"(all decagons {nt[dec].min()}-{nt[dec].max()}, triangles {nt[tri].min()}-{nt[tri].max()})",
        500 <= nt[jd] <= 2000 and nt[jt] <= 100)
    frac = (1 / R["H"][j]) / (R["K_total"][j] - K)
    add("A prose", "octagon 00850: 1/H a large fraction of its discarded modes", "large fraction",
        f"{1 / R['H'][j]:.0f} / {R['K_total'][j] - K} = {frac:.2f}", frac >= 0.5)


# ----------------------------------------------------------------------------- T4
def t4_cells(res, T):
    sub = res[res["T"] == T]
    base_rows = sub[(sub["fam"] == "F1_empirical") & (sub["a"] == 0)]
    base_map = dict(zip(base_rows["rid"], base_rows["d_rel"]))
    gbase = np.median(list(base_map.values()))
    cells = []
    for fam in ("F1_empirical", "F2_balanced", "k2", "F3_spike"):
        for a in np.unique(sub["a"]):
            c = sub[(sub["fam"] == fam) & np.isclose(sub["a"], a)]
            if not len(c):
                continue
            if fam == "F1_empirical":
                rids = np.unique(c["rid"])
                med, ps, smax = np.median(c["d_rel"]), np.median(c["p_star"]), np.nan
                base = np.median([base_map[r] for r in c["rid"]])   # same rooms' isotropic cost
            else:
                rnd = c[c["lbl"] != "aligned"]
                rids = np.unique(rnd["rid"])
                med, ps, base = np.median(rnd["d_rel"]), np.median(rnd["p_star"]), gbase
                smax = np.median([np.max(rnd[rnd["rid"] == r]["d_rel"]) for r in rids])
            cells.append(dict(fam=fam, a=float(a), n=len(rids), incr=100 * (med - base),
                              sampled=100 * (smax - gbase), p_star=ps))
    return 100 * gbase, len(base_map), cells


def t4_summary(res, a_max=np.inf):
    out = {}
    for T in (1, 100, 1000):
        base, nfull, cells = t4_cells(res, T)
        cells = [c for c in cells if c["a"] <= a_max + 1e-9]
        full = [c for c in cells if c["n"] == nfull]
        worst_all = max(cells, key=lambda c: c["incr"])
        out[T] = dict(base=base, nfull=nfull, full_incr=max(c["incr"] for c in full),
                      all_incr=worst_all["incr"], all_cell=worst_all,
                      sampled=np.nanmax([c["sampled"] for c in cells]),
                      p_lo=min(c["p_star"] for c in full), p_hi=max(c["p_star"] for c in full))
    return out


def check_T4(aniso, eop, s):
    res = aniso["results"]
    mod = import_script(Path(__file__).with_name("aniso_sensitivity_sweep.py"), "aniso_sweep")
    summ = t4_summary(res)
    mean_marker = round(float(np.mean(eop["E_op_per_room"])), 2)
    summ_mean = t4_summary(res, a_max=mean_marker)
    # printed T4: (median, sampled) up to the mean marker, then up to the spike level
    printed = {1: (0.01, 0.05, 0.04, 0.26), 100: (0.01, 0.04, 0.02, 0.08), 1000: (0.08, 0.50, 0.21, 1.42)}
    for T in (1, 100, 1000):
        qm, q = summ_mean[T], summ[T]
        for lab, v, pr in (("up to the mean marker: largest median increase", qm["full_incr"], printed[T][0]),
                           ("up to the mean marker: largest sampled-orientation increase", qm["sampled"], printed[T][1]),
                           ("up to the spike level: largest median increase", q["full_incr"], printed[T][2]),
                           ("up to the spike level: largest sampled-orientation increase", q["sampled"], printed[T][3])):
            add("T4", f"T={T}: {lab}", f"{pr:.2f} pp", f"{v:+.3f} pp (isotropic baseline {q['base']:.3f}%, "
                f"n={q['nfull']} rooms)", agree(pr, v, 2), f"T4 T={T}")
    lo = min(summ[T]["p_lo"] for T in summ)
    hi = max(summ[T]["p_hi"] for T in summ)
    add("T4", "optimal exponent over all full-room-set cells", "p* in [1.10, 1.20]",
        f"[{lo:.2f}, {hi:.2f}] (per T: " + ", ".join(
            f"T={T} {summ[T]['p_lo']:.2f}-{summ[T]['p_hi']:.2f}" for T in summ) + ")",
        agree(1.10, lo, 2) and agree(1.20, hi, 2), "T4 p*")
    # prose (A.5)
    add("A prose", "up to the mean marker: median barely moves (<= 0.1 pp), sampled orientation <= half a point",
        "<= 0.1 / <= 0.5 pp", ", ".join(f"T={T} {summ_mean[T]['all_incr']:+.3f} / {summ_mean[T]['sampled']:.2f} pp"
                                    for T in summ_mean),
        all(summ_mean[T]["all_incr"] <= 0.105 and summ_mean[T]["sampled"] <= 0.505 for T in summ_mean))
    add("A prose", "beyond it: same at T <= 100; at T = 1000 median up a few tenths, sampled > 1 pp",
        "T<=100 <= 0.1 / < 1 pp; T=1000 0.1-0.5 / > 1 pp",
        ", ".join(f"T={T} {summ[T]['full_incr']:+.3f} / {summ[T]['sampled']:.2f} pp" for T in summ),
        all(summ[T]["full_incr"] <= 0.105 and summ[T]["sampled"] < 1 for T in (1, 100))
        and 0.1 < summ[1000]["full_incr"] < 0.5 and summ[1000]["sampled"] > 1)
    add("A prose", "reduced cells of the empirical-direction family rise further",
        "reduced > full-set", ", ".join(f"T={T} {summ[T]['all_incr']:+.3f} ({summ[T]['all_cell']['fam']} "
                                        f"a={summ[T]['all_cell']['a']}, n={summ[T]['all_cell']['n']})" for T in summ),
        all(summ[T]["all_incr"] > summ[T]["full_incr"] for T in summ))
    dev = max(max(abs(summ[T]["p_lo"] - s), abs(summ[T]["p_hi"] - s)) for T in summ)
    add("A prose", "p* within a tenth of |s| in every full-room-set cell", "|p* - |s|| <= 0.1",
        f"max |p* - |s|| = {dev:.3f}", dev <= 0.1 + 1e-9)
    n_rooms = len(aniso["rooms"])
    add("A prose", "sweep rooms / (p, alpha) grid", "24 / 61 x 12",
        f"{n_rooms} / {len(mod.P_GRID)} x {len(mod.LAMBDA_GRID)}",
        n_rooms == 24 and len(mod.P_GRID) == 61 and len(mod.LAMBDA_GRID) == 12)
    a_sp = res[res["fam"] == "F3_spike"]["a"].max()
    add("A prose", "spike family: 9:1 eigenvalue ratio at a = 3.5 (largest a)", "9:1 at 3.5",
        f"(1+a)/(1-a/7) = {(1 + a_sp) / (1 - a_sp / 7):.2f} at a = {a_sp}",
        a_sp == 3.5 and agree(9, (1 + a_sp) / (1 - a_sp / 7), 6))
    E = eop["E_op_per_room"]
    marks = (np.median(E), E.mean(), np.percentile(E, 95))
    add("A prose", "markers a = median / mean / 95th pct of ||E||_op", "0.58 / 0.88 / 2.42",
        " / ".join(f"{m:.3f}" for m in marks) + f"; sweep grid {[float(a) for a in mod.A_GRID]}",
        all(np.any(np.isclose(mod.A_GRID, round(m, 2))) for m in marks))
    return summ


# ----------------------------------------------------------------------------- T36, T5, T6
def check_B(ids, R, keep, s, cost, diag):
    qq = load("run_berry_qq")
    sh = np.load(SUPP / "berry_qq_data.npz", allow_pickle=True)
    same = all(np.array_equal(qq[k], sh[k]) for k in sh.files)
    add("T36", "run_berry_qq.npz == shipped supplementary/berry_qq_data.npz", "identical",
        "identical" if same else "differs", same)
    Kt = qq["K_total"]
    N = len(qq["all_Csq"])
    ks = stats.kstest(qq["all_Csq"], "chi2", args=(1,))
    add("T36", "pooled D_KS all retained modes (N = 77,968)", "0.037 (p < 1e-10)",
        f"{ks.statistic:.5f} (p = {ks.pvalue:.1e}, N = {N})",
        agree(0.037, ks.statistic, 3) and ks.pvalue < 1e-10 and N == 77968)
    k10 = np.concatenate([(R["area"][i] * R["Phi"][i][:, 9:K] ** 2).ravel() for i in range(len(ids))])
    d10 = stats.kstest(k10, "chi2", args=(1,)).statistic
    add("T36", "pooled D_KS, modes k >= 10 only", "0.016",
        f"{d10:.4f} (modes 10-50, N = {len(k10)})", agree(0.016, d10, 3), "T36 pooled k>=10")
    nsamp = np.array([8 * min(k, K) for k in Kt])
    crit = stats.kstwo.ppf(0.95, 400)
    add("T36", "samples per room (K_total >= K) / 0.05 critical value", "400 / 0.067",
        f"{sorted(int(n) for n in set(nsamp[Kt >= K]))} / {crit:.5f} (exact kstwo; asymptotic "
        f"1.358/sqrt(400) = {1.358 / 20:.4f})",
        set(nsamp[Kt >= K]) == {400} and agree(0.067, crit, 3))
    pv, D = qq["ks_pval"], qq["ks_stat"]
    peak = (Kt >= 50) & (Kt < 200)
    pr = 100 * np.mean(pv[peak] > 0.05)
    add("T36", "pass rate at 0.05, K_total in [50, 200)", "88.5%", f"{pr:.2f}%", agree(88.5, pr, 1))
    bins = [(50, 200), (200, 500), (500, 10 ** 9)]
    meds = [(np.median(D[(Kt >= lo) & (Kt < hi)]), int(((Kt >= lo) & (Kt < hi)).sum()))
            for lo, hi in bins]
    add("T36", "median per-room D_KS by bin (rooms)", "0.054 (61) / 0.061 (54) / 0.068 (73)",
        " / ".join(f"{m:.4f} ({n})" for m, n in meds),
        all(agree(p, m, 3) and n == pn for (m, n), p, pn in
            zip(meds, (0.054, 0.061, 0.068), (61, 54, 73))))
    low = Kt < K
    add("T36", "rooms with K_total < K: count / samples / median D_KS", "9 / 136-376 / 0.089",
        f"{low.sum()} / {nsamp[low].min()}-{nsamp[low].max()} / {np.median(D[low]):.4f}",
        low.sum() == 9 and nsamp[low].min() == 136 and nsamp[low].max() == 376
        and agree(0.089, np.median(D[low]), 3))
    kD, kS = diag["ks_D"], diag["ks_D_norm"]
    q = np.percentile(kD, [25, 50, 75])
    add("T36", "per-room D_KS chi2: median [IQR] / max / rooms > 0.12",
        "0.061 [0.046, 0.073] / 0.197 / 1",
        f"{q[1]:.4f} [{q[0]:.4f}, {q[2]:.4f}] / {kD.max():.4f} / {(kD > 0.12).sum()}",
        agree(0.061, q[1], 3) and agree(0.046, q[0], 3) and agree(0.073, q[2], 3)
        and agree(0.197, kD.max(), 3) and (kD > 0.12).sum() == 1)
    q = np.percentile(kS, [25, 50, 75])
    add("T36", "per-room D_KS signed: median [IQR] / max", "0.054 [0.042, 0.066] / 0.205",
        f"{q[1]:.4f} [{q[0]:.4f}, {q[2]:.4f}] / {kS.max():.4f}",
        agree(0.054, q[1], 3) and agree(0.042, q[0], 3) and agree(0.066, q[2], 3)
        and agree(0.205, kS.max(), 3))
    rc = stats.spearmanr(kD, kS).correlation
    add("T36", "Spearman between the two conventions", "0.57", f"{rc:.4f}", agree(0.57, rc, 2))
    hi = kD > 0.08
    counts = [int((R["nseg"][hi] == k).sum()) for k in range(3, 11)]
    add("T36", "rooms with D_KS > 0.08 (tri/quad/pent/hex/hept/oct/non/dec)",
        "38 (4 / 6 / 2 / 5 / 3 / 9 / 5 / 4)", f"{hi.sum()} ({' / '.join(map(str, counts))})",
        hi.sum() == 38 and counts == [4, 6, 2, 5, 3, 9, 5, 4])
    ins = Kt > K
    sp = stats.spearmanr(kD[ins], Kt[ins] - K)
    pe = stats.pearsonr(kD[ins], Kt[ins] - K)
    sa = stats.spearmanr(kD, Kt - K)
    add("T36", "rho(D_KS, K_total - K) in-scope (Pearson) / all rooms",
        "+0.29 (+0.30, p < 1e-4) / +0.16",
        f"{sp.correlation:+.4f} ({pe[0]:+.4f}, p = {pe[1]:.1e}) / {sa.correlation:+.4f}",
        agree(0.29, sp.correlation, 2) and agree(0.30, pe[0], 2) and pe[1] < 1e-4
        and agree(0.16, sa.correlation, 2))

    dr = cost["per_room_delta_rel"]
    T = list(cost["T"])
    elig = (Kt - K) >= 50
    top = np.argsort(-np.where(elig, kD, -1))[:5]
    med1000 = np.median(dr[T_IDX[1000], keep])
    mx100 = np.array([dr[:T.index(100) + 1, i].max() for i in top])
    ratios = dr[T_IDX[1000], top] / med1000
    j835 = ids.index("scene_00835")
    add("T36", "worst-five: largest delta at T <= 100 / delta(1000) vs in-scope median",
        "3.4% / four within 1.5x, 00835 at 3.9x",
        f"{100 * mx100.max():.2f}% / ratios " + ", ".join(
            f"{ids[i][6:]} {r:.2f}x" for i, r in zip(top, ratios)),
        agree(3.4, 100 * mx100.max(), 1) and np.sum(ratios[np.array(top) != j835] <= 1.5) == 4
        and agree(3.9, dr[T_IDX[1000], j835] / med1000, 1))

    # boundary-stratified KS (T5 + near-wall row of T36), signed values, in-scope rooms
    near, inter, allS, near_pr, inter_pr = [], [], [], [], []
    for i in np.where(Kt > K)[0]:
        S = np.sqrt(R["area"][i]) * R["Phi"][i][:, :K]
        v = R["verts"][i]
        diam = max(np.linalg.norm(v[a] - v[b]) for a in range(len(v)) for b in range(len(v)))
        d = np.array([dist_to_boundary(p, v) for p in R["mic"][i]]) / diam
        o = np.argsort(d)
        near.append(S[o[:2]].ravel())
        inter.append(S[o[-2:]].ravel())
        allS.append(S.ravel())
        near_pr.append(stats.kstest(S[o[:2]].ravel(), "norm").statistic)
        inter_pr.append(stats.kstest(S[o[-2:]].ravel(), "norm").statistic)
    near, inter, allS = map(np.concatenate, (near, inter, allS))
    dn, di, da = (stats.kstest(x, "norm").statistic for x in (near, inter, allS))
    add("T5", "interior (top quartile): samples / D_KS", "18,700 / 0.0198",
        f"{len(inter)} / {di:.5f}", len(inter) == 18700 and agree(0.0198, di, 4))
    add("T5", "near-boundary (bottom quartile): samples / D_KS", "18,700 / 0.0242",
        f"{len(near)} / {dn:.5f}", len(near) == 18700 and agree(0.0242, dn, 4))
    add("T5", "all sensors: samples / D_KS", "74,800 / 0.0203",
        f"{len(allS)} / {da:.5f}", len(allS) == 74800 and agree(0.0203, da, 4))
    add("T36", "near-wall vs interior: pooled excess / per-room medians", "22% / 0.084 vs 0.085",
        f"{100 * (dn / di - 1):.1f}% / {np.median(near_pr):.4f} vs {np.median(inter_pr):.4f}",
        agree(22, 100 * (dn / di - 1), 0) and agree(0.084, np.median(near_pr), 3)
        and agree(0.085, np.median(inter_pr), 3))
    add("T36", "rooms excluded (< 50 discarded) / eligible", "29 / 168",
        f"{(~elig).sum()} / {elig.sum()}", (~elig).sum() == 29 and elig.sum() == 168)
    Hw = R["H"][top]
    add("T36", "worst-Berry rooms: vertices / K_total / H", "6-10 / > 400 / 0.0025-0.0042",
        f"{R['nseg'][top].min()}-{R['nseg'][top].max()} / {Kt[top].min()}-{Kt[top].max()} / "
        f"{Hw.min():.4f}-{Hw.max():.4f}",
        R["nseg"][top].min() == 6 and R["nseg"][top].max() == 10 and Kt[top].min() > 400
        and agree(0.0025, Hw.min(), 4) and agree(0.0042, Hw.max(), 4), "T36 worst-Berry H")
    mx835 = dr[:T.index(100) + 1, j835].max()
    add("T36", "room 00835: D_KS / delta(1000) / delta at T <= 100", "0.098 / 22.5% / <= 3%",
        f"{kD[j835]:.4f} / {100 * dr[T_IDX[1000], j835]:.2f}% / max {100 * mx835:.2f}%",
        agree(0.098, kD[j835], 3) and agree(22.5, 100 * dr[T_IDX[1000], j835], 1)
        and 100 * mx835 <= 3.0)
    rr = stats.spearmanr(kD[elig], dr[T_IDX[1000], elig])
    add("T36", "rho(D_KS, delta(1000)) over eligible rooms", "0.19 (p = 0.012)",
        f"{rr.correlation:+.4f} (p = {rr.pvalue:.4f}, n = {elig.sum()})",
        agree(0.19, rr.correlation, 2) and agree(0.012, rr.pvalue, 3))

    # T6
    printed = {"00963": (6, 15.6, 422, 0.0042, 0.103, 0.0, 1.8, 5.4),
               "00924": (9, 18.8, 508, 0.0037, 0.101, 0.5, 0.1, 5.8),
               "00860": (8, 22.4, 604, 0.0033, 0.100, 0.0, 1.5, 3.8),
               "00835": (10, 36.8, 987, 0.0025, 0.098, 0.1, 2.8, 22.5),
               "00900": (10, 34.5, 921, 0.0026, 0.097, 3.2, 2.2, 8.5)}
    dec = (0, 1, 0, 4, 3, 1, 1, 1)
    order = [ids[i][6:] for i in top]
    add("T6", "rooms and order (top-5 D_KS among eligible)", " ".join(printed),
        " ".join(order), order == list(printed))
    for rid, pv_ in printed.items():
        i = ids.index(f"scene_{rid}")
        val = (R["nseg"][i], R["area"][i], Kt[i], R["H"][i], kD[i],
               100 * dr[T_IDX[1], i], 100 * dr[T_IDX[100], i], 100 * dr[T_IDX[1000], i])
        ok = all(agree(p, v, d) for p, v, d in zip(pv_, val, dec))
        add("T6", f"room {rid}: verts/area/K_total/H/D_KS/delta(1,100,1000)%",
            " / ".join(f"{p:.{d}f}" for p, d in zip(pv_, dec)),
            " / ".join(f"{v:.{d + 1}f}" if d else f"{v:.0f}" for v, d in zip(val, dec)), ok)

    # prose (B)
    add("B prose", "N = sum over 197 rooms of 8 * min(K_total, 50)", "77,968",
        f"{nsamp.sum()} ({len(ids)} rooms)", nsamp.sum() == 77968 and len(ids) == 197)
    p999 = np.percentile(qq["all_Csq"], 99.9)
    add("B prose", "lighter upper tail than chi2(1)", "lighter tail",
        f"99.9th pct {p999:.2f} vs chi2(1) {stats.chi2.ppf(0.999, 1):.2f}",
        p999 < stats.chi2.ppf(0.999, 1))
    add("B prose", "restricting to k >= 10 lowers the pooled statistic", "lower",
        f"{d10:.4f} < {ks.statistic:.4f}", d10 < ks.statistic)
    mv = [m for m, _ in meds]
    add("B prose", "bin medians rise with mode count; only the highest crosses the critical value",
        "rising, crosses in last bin", " < ".join(f"{m:.4f}" for m in mv) + f"; crit {crit:.4f}",
        mv[0] < mv[1] < mv[2] and mv[1] < crit < mv[2])
    add("B prose", "K_total < K rooms deviate more", "higher median D",
        f"{np.median(D[low]):.4f} > {max(mv):.4f}", np.median(D[low]) > max(mv))
    add("B prose", "per-room chi2 values just below the critical value, one outlier", "median < crit",
        f"median {np.median(kD):.4f} < {crit:.4f}; {(kD > 0.12).sum()} room > 0.12",
        np.median(kD) < crit and (kD > 0.12).sum() == 1)
    add("B prose", "signed convention gives somewhat lower levels", "lower",
        f"{np.median(kS):.4f} < {np.median(kD):.4f}", np.median(kS) < np.median(kD))
    add("B prose", "mode-count correlation significant in-scope, weaker over all rooms",
        "in-scope significant, all-room weaker",
        f"in-scope {sp.correlation:+.3f} (p = {sp.pvalue:.1e}); all {sa.correlation:+.3f}",
        sp.pvalue < 0.01 and abs(sa.correlation) < abs(sp.correlation))
    add("B prose", "near-boundary pooled D_KS above interior", "higher", f"{dn:.4f} > {di:.4f}",
        dn > di)
    nmed = stats.kstwo.ppf(0.5, 100)
    add("B prose", "per-room quartile medians at the sampling-noise level (100 samples)",
        "noise level", f"{np.median(near_pr):.4f} / {np.median(inter_pr):.4f} vs null median "
        f"{nmed:.4f}", abs(np.median(near_pr) - nmed) < 0.01 and abs(np.median(inter_pr) - nmed) < 0.01)
    medK = np.median(Kt)
    add("B prose", "worst-Berry rooms well above the median mode count", "above median",
        f"min {Kt[top].min()} vs median {medK:.0f} (197) / {np.median(Kt[keep]):.0f} (187)",
        Kt[top].min() > np.median(Kt[keep]))
    return top


def check_prose_A(ids, R, keep, cost, eop, t32):
    Kt = R["K_total"]
    add("A prose", "rooms: in-scope / K_total <= 50 / total", "187 / 10 / 197",
        f"{keep.sum()} / {(Kt <= K).sum()} / {len(ids)}",
        keep.sum() == 187 and (Kt <= K).sum() == 10 and len(ids) == 197)
    q50, Hmed = t32["q50"], t32["Hmed"]
    add("A prose", "off-diagonal 56H is the larger term at M = 8", "56H > 8D",
        f"56H = {56 * Hmed:.4f} vs 8D = {8 * 2 * q50 * Hmed:.4f}", 56 > 16 * q50)
    add("A prose", "measured D about twice Berry's value", "~2x", f"D/(2H) = {q50:.3f}",
        1.5 <= q50 <= 2.5)
    E = t32["E"]
    i = int(np.argmax(E))
    sid = eop["scene_ids"][i]
    j = ids.index(sid)
    add("A prose", "extreme room: a single discarded mode reaches M - 1", "||E||_op = 7",
        f"{sid}: ||E||_op = {E[i]:.4f}, K_total - K = {Kt[j] - K}",
        agree(7, E[i], 3) and Kt[j] - K == 1)
    add("A prose", "empirical ||E||_op median below one, long right tail", "median < 1",
        f"median {np.median(E):.3f}, mean/median {E.mean() / np.median(E):.2f}, max {E.max():.2f}",
        np.median(E) < 1 and E.mean() > np.median(E))
    dr, dP = cost["per_room_delta_rel"], cost["per_room_delta_P"]
    meds = {int(T): np.median(dr[k, keep]) for k, T in enumerate(cost["T"])}
    Tpk = max(meds, key=meds.get)
    add("A prose", "in-scope median relative cost peaks at 5.83%", "5.83% (T = 1000)",
        f"{100 * meds[Tpk]:.3f}% at T = {Tpk}", agree(5.83, 100 * meds[Tpk], 2) and Tpk == 1000)
    frac = np.mean(dP[T_IDX[1000], keep] < 0.011)
    add("A prose", "at T = 1000 two in-scope rooms in three lose < 1.1 pp", ">= 2/3",
        f"{100 * frac:.1f}% ({int(round(frac * keep.sum()))}/{keep.sum()})", frac >= 2 / 3)
    all_med = np.median(dr, axis=1)
    add("A prose", "over all rooms the median cost falls again at the longest window",
        "delta(2100) < delta(1000)",
        f"197-room medians {100 * all_med[T_IDX[1000]]:.2f}% (T=1000) -> "
        f"{100 * all_med[T_IDX[2100]]:.2f}% (T=2100)", all_med[T_IDX[2100]] < all_med[T_IDX[1000]])


# ----------------------------------------------------------------------------- main
def main():
    global T_IDX
    check_paper_dataset(MODAL)
    ps = np.load(EXP / "p_sweep" / "p_sweep_K50_M8.npz", allow_pickle=True)
    cost = np.load(EXP / "cost" / "cost_K50_M8.npz", allow_pickle=True)
    eop = np.load(EXP / "anisotropy" / "E_op_empirical_187.npz", allow_pickle=True)
    noise = np.load(EXP / "noise_profile" / "noise_profile_K50_M8.npz", allow_pickle=True)
    diag = load("room_diagnostic_regression")
    hd = load("hd_decomposition")
    tf = load("verify_tracefree_identity")
    aniso = load("aniso_sensitivity_sweep")
    ids = list(ps["room_ids"])
    assert list(diag["rooms"]) == ids
    s = float(cost["s_value"])
    T_IDX = {int(T): k for k, T in enumerate(cost["T"])}

    R = load_rooms(ids, s)
    keep = R["K_total"] > K
    verify = import_script(REPO / "scripts" / "15_verify_appendix_a.py", "verify_appendix_a")
    ref_mc = verify.check_Eij_MC(str(MODAL), -float(np.median(noise["s_per_room"])))

    t32 = check_T32(ids, R, s, keep, eop, cost, hd, tf, noise, ref_mc)
    check_T35(ids, R, keep, s)
    check_T4(aniso, eop, s)
    check_B(ids, R, keep, s, cost, diag)
    check_prose_A(ids, R, keep, cost, eop, t32)

    w = [max(len(r[i]) for r in ROWS) for i in (0, 1)]
    print(f"{'table':{w[0]}}  {'row':{w[1]}}  printed | recomputed | status")
    for t, lab, p, r, st in ROWS:
        print(f"{t:{w[0]}}  {lab:{w[1]}}  {p} | {r} | {st}")
    n = {k: sum(r[4] == k for r in ROWS) for k in ("OK", "DIFF(known)", "DIFF")}
    print(f"\n{len(ROWS)} checks: {n['OK']} OK, {n['DIFF(known)']} DIFF(known), {n['DIFF']} DIFF")
    sys.exit(1 if n["DIFF"] else 0)


if __name__ == "__main__":
    main()
