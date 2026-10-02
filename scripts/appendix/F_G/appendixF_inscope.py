#!/usr/bin/env python
"""Appendix F, primary test (Table T24; T41 rows "20 / 16" and "about 6% higher").

Prior-robustness control IN the truncation regime: the measurements are built from
ALL K_total modes, the estimator models only the retained K = 50, so truncation noise
is present as in Section 5.

Fixed protocol: the 20 rooms scene_00800..00819 (16 have K_total > 50), the 61-point
p grid (0..6 step 0.1; flatness on [0,3]), the 12-point alpha grid searched at every p,
T in {1, 100}, |s| = 1.13, M = 8, and three priors (Gaussian; Student-t3 scaled by
1/sqrt(3) to the same variance; adjacent-mode correlation rho = 0.3), applied over all
K_total modes.

  (i)  population risk, exact from second moments (Gaussian and t3 coincide exactly);
  (ii) empirical: one initial condition per room per seed, windowed as in Section 5,
       24 seeds (default_rng(10000 + 97 s));
  (iii) "task 3": Gamma = lambda^|s|, alpha fixed at the Gaussian population optimum,
       empirical P under Gaussian vs t3 with paired seeds (default_rng(500 + 13 s)).

Independent (T, prior, seed) tasks run in a process pool (the draws do not depend on
the pool size). Task (iii) is saved under keys fixed_T{T}: per-room [mean_G, mean_t3,
std_G, std_t3] over seeds.

Output: data/experiments/appendix/appendixF_inscope.npz
Usage:  python scripts/appendix/F_G/appendixF_inscope.py [--workers 16]
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import time
from multiprocessing import Pool

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from _common import out_path, n_workers
from src.utils.paths import MODAL, check_paper_dataset
from src.physics.temporal import build_wave_temporal_matrix
from src.estimation.ridge import build_gamma_diag
from src.estimation.metrics import compute_P

S_HAT = 1.13
K_RET, M_USE = 50, 8
RHO = 0.3
T_EVAL = [1, 100]
P_GRID = np.round(np.arange(0.0, 6.01, 0.1), 10)
M03 = (P_GRID >= 0) & (P_GRID <= 3.0000001)
LAM = np.array([1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100., 1e3, 1e4, 1e6, 1e8, 1e10])
PRIORS = ["gaussian", "heavy_tail", "correlated"]
ROOMS = [f"scene_{i:05d}" for i in range(800, 820)]
N_SEED = 24
SKIP_F, SKIP_L = 5, 2


def load(sid):
    d = MODAL / sid
    eig = np.load(d / "eigenpairs.npz", allow_pickle=True)
    tr = np.load(d / "modal_trajectories.npz", allow_pickle=True)
    ev = np.asarray(eig["eigenvalues"], float)
    Phi = np.asarray(np.load(d / "measurement_matrix.npy"), float)[:M_USE]
    return dict(sid=sid, ev=ev, Phi=Phi, K=len(ev),
                c=float(eig["c"]) if "c" in eig else 343.0,
                gamma=float(tr["gamma_room"]), dt=float(tr["dt_sim"]),
                n_snaps=int(np.asarray(tr["a"]).shape[1]),
                sps=int(round(float(tr["dt_snap"]) / float(tr["dt_sim"]))))


def sigma_diag(ev):
    return ev ** (-S_HAT)


def corr_offdiag(ev):
    """adjacent-mode covariance sigma_{n,n+1} for the correlated prior."""
    v = sigma_diag(ev)
    sd = np.sqrt(v)
    return RHO * sd[:-1] * sd[1:]


_PREP = {}


def prep(room, T_raw):
    """Cache the (room, T) pieces: ATA, B = A_ret^T A_tr, prior variances."""
    key = (room["sid"], int(T_raw))
    if key in _PREP:
        return _PREP[key]
    ev, Phi = room["ev"], room["Phi"]
    nK = len(ev)
    A_full = build_wave_temporal_matrix(Phi, ev, room["dt"], int(T_raw),
                                        room["gamma"], room["c"])
    nr = 2 * K_RET
    ATA = A_full[:, :nr].T @ A_full[:, :nr]
    B = A_full[:, :nr].T @ A_full[:, nr:]
    v = sigma_diag(ev)
    v2 = np.empty(2 * nK); v2[0::2] = v; v2[1::2] = v
    out = dict(ev=ev, ATA=ATA, B=B, vr=v2[:nr], vt=v2[nr:], nr=nr, nK=nK)
    _PREP[key] = out
    return out


def pop_risk_curve(room, T_raw, prior):
    """Exact second-moment risk on the released grid; truncation noise included."""
    if len(room["ev"]) <= K_RET:
        return None
    P_ = prep(room, T_raw)
    ev, ATA, B, vr, vt, nr = P_["ev"], P_["ATA"], P_["B"], P_["vr"], P_["vt"], P_["nr"]
    nK = P_["nK"]
    C1 = (B * vt[None, :]) @ B.T
    Xrt = None
    if prior == "correlated":
        off = corr_offdiag(ev)
        # within-truncated-band adjacent pairs (c and beta blocks separately)
        idx = np.arange(K_RET, nK - 1)
        if len(idx):
            for blk in (0, 1):
                i0 = 2 * (idx - K_RET) + blk
                i1 = i0 + 2
                Bi, Bj = B[:, i0], B[:, i1]
                C1 += (Bi * off[idx][None, :]) @ Bj.T + (Bj * off[idx][None, :]) @ Bi.T
        # retained<->truncated boundary pair (mode K_RET-1 <-> K_RET)
        ob = off[K_RET - 1]
        Xrt = []
        for blk in (0, 1):
            Xrt.append((2 * (K_RET - 1) + blk, blk, ob))
    den = float(np.sum(vr[0::2]))  # variances only; off-diagonals do not enter tr
    curve = np.full(len(P_GRID) + 1, np.nan)
    for pi, p in enumerate(np.concatenate([P_GRID, [S_HAT]])):
        g = build_gamma_diag(ev[:K_RET], p, True)
        best = np.inf
        for lam in LAM:
            reg = ATA + lam * np.diag(g)
            try:
                f = cho_factor(reg, lower=True)
            except np.linalg.LinAlgError:
                continue
            S = cho_solve(f, ATA)
            D = S - np.eye(nr)
            num = float(np.sum(D[0::2, :] ** 2 * vr[None, :]))       # retained
            Z = cho_solve(f, cho_solve(f, C1).T)
            num += float(Z.diagonal()[0::2].sum())                   # truncation
            if prior == "correlated":
                # retained-band adjacent covariance
                off = corr_offdiag(ev)[:K_RET - 1]
                Dc = D[0::2, :]
                for blk in (0, 1):
                    j0 = 2 * np.arange(K_RET - 1) + blk
                    num += 2.0 * float(np.sum(off * np.sum(Dc[:, j0] * Dc[:, j0 + 2], axis=0)))
                # boundary cross term 2*tr_c(D Sigma_rt B^T reg^-1)
                WB = cho_solve(f, B)
                for (ir, blk, ob) in Xrt:
                    it = blk
                    num += 2.0 * ob * float(np.dot(D[0::2, ir], WB[0::2, it]))
            if np.isfinite(num) and num >= 0:
                best = min(best, num / den)
        curve[pi] = best if np.isfinite(best) else np.nan
    return curve


def draw_ic(ev, prior, rng):
    v = sigma_diag(ev)
    sd = np.sqrt(v)
    n = len(ev)
    if prior == "gaussian":
        return rng.standard_normal(n) * sd, rng.standard_normal(n) * sd
    if prior == "heavy_tail":
        return (rng.standard_t(3, n) * sd / np.sqrt(3.0),
                rng.standard_t(3, n) * sd / np.sqrt(3.0))
    off = corr_offdiag(ev)
    Sig = np.diag(v)
    Sig[np.arange(n - 1), np.arange(1, n)] = off
    Sig[np.arange(1, n), np.arange(n - 1)] = off
    L = np.linalg.cholesky(Sig + 1e-14 * np.eye(n))
    return L @ rng.standard_normal(n), L @ rng.standard_normal(n)


def emp_curve(room, T_raw, prior, rng, gamma_fixed=None):
    """One IC over ALL modes, windows from the full-mode signal, estimator models only
    the retained band."""
    ev, Phi = room["ev"], room["Phi"]
    c0, b0 = draw_ic(ev, prior, rng)
    om = room["c"] * np.sqrt(ev)
    omd = np.sqrt(np.maximum(om ** 2 - (room["gamma"] / 2) ** 2, 0.0))
    sps, ns = room["sps"], room["n_snaps"]
    T_sig = ns * sps
    t = np.arange(T_sig) * room["dt"]
    dec = np.exp(-room["gamma"] * t / 2)
    a_t = (c0[:, None] * np.cos(omd[:, None] * t[None, :])
           + b0[:, None] * np.sin(omd[:, None] * t[None, :])) * dec[None, :]
    y = Phi @ a_t                                        # ALL modes -> truncation
    valid = [si for si in range(SKIP_F, ns - SKIP_L)
             if si * sps + 1 - int(T_raw) >= 0 and si * sps + 1 <= T_sig]
    if len(valid) < 5:
        return None
    targets = a_t[:K_RET, [si * sps for si in valid]].T
    A = build_wave_temporal_matrix(Phi[:, :K_RET], ev[:K_RET], room["dt"],
                                   int(T_raw), room["gamma"], room["c"])
    Y = np.stack([y[:, si * sps + 1 - int(T_raw): si * sps + 1].reshape(-1)
                  for si in valid])
    ATA, ATY = A.T @ A, A.T @ Y.T
    if gamma_fixed is not None:
        g, lam = gamma_fixed
        X = np.linalg.solve(ATA + lam * np.diag(g), ATY)
        return compute_P(X[0::2, :].T, targets)
    curve = np.full(len(P_GRID) + 1, np.nan)
    for pi, p in enumerate(np.concatenate([P_GRID, [S_HAT]])):
        g = build_gamma_diag(ev[:K_RET], p, True)
        best = np.inf
        for lam in LAM:
            try:
                X = np.linalg.solve(ATA + lam * np.diag(g), ATY)
            except np.linalg.LinAlgError:
                continue
            best = min(best, compute_P(X[0::2, :].T, targets))
        curve[pi] = best
    return curve


def stats(curve):
    g = curve[:len(P_GRID)]
    if np.all(np.isnan(g)):
        return (np.nan,) * 4
    j = int(np.nanargmin(g))
    Ps = g[j]
    d = (curve[len(P_GRID)] - Ps) / Ps
    fl = float(np.nanmax(g[M03]) / np.nanmin(g[M03]))
    return P_GRID[j], Ps, d, fl


def _fixed_pop(room, T_raw, gd, lam):
    """Gaussian population risk at fixed (Gamma, alpha) for alpha selection."""
    P_ = prep(room, T_raw)
    ATA, B, vr, vt, nr = P_["ATA"], P_["B"], P_["vr"], P_["vt"], P_["nr"]
    reg = ATA + lam * np.diag(gd)
    try:
        f = cho_factor(reg, lower=True)
    except np.linalg.LinAlgError:
        return np.inf
    D = cho_solve(f, ATA) - np.eye(nr)
    num = float(np.sum(D[0::2, :] ** 2 * vr[None, :]))
    C1 = (B * vt[None, :]) @ B.T
    Z = cho_solve(f, cho_solve(f, C1).T)
    num += float(Z.diagonal()[0::2].sum())
    return num / float(np.sum(vr[0::2]))


# ---------------- pool tasks (rooms are inherited from the parent via fork) ----------
_INS = []


def _task_pop(args):
    T, pr = args
    cs = [pop_risk_curve(r, T, pr) for r in _INS]
    cs = [c for c in cs if c is not None]
    return np.nanmedian(np.array(cs), axis=0)


def _task_emp(args):
    T, pr, s = args
    rng = np.random.default_rng(10_000 + 97 * s)
    cs = [emp_curve(r, T, pr, rng) for r in _INS]
    cs = [c for c in cs if c is not None]
    return np.nanmedian(np.array(cs), axis=0)


def _task_fixed(args):
    T, ri = args
    r = _INS[ri]
    gd = build_gamma_diag(r["ev"][:K_RET], S_HAT, True)
    lam_star = LAM[int(np.argmin([_fixed_pop(r, T, gd, lam) for lam in LAM]))]
    pg_, pt_ = [], []
    for s in range(N_SEED):
        pg_.append(emp_curve(r, T, "gaussian", np.random.default_rng(500 + 13 * s),
                             gamma_fixed=(gd, lam_star)))
        pt_.append(emp_curve(r, T, "heavy_tail", np.random.default_rng(500 + 13 * s),
                             gamma_fixed=(gd, lam_star)))
    pg_, pt_ = np.array(pg_), np.array(pt_)
    return (np.mean(pg_), np.mean(pt_), np.std(pg_), np.std(pt_))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    check_paper_dataset()
    t0 = time.time()
    rooms = [load(s) for s in ROOMS]
    _INS[:] = [r for r in rooms if r["K"] > K_RET]
    ins = _INS
    print(f"rooms: {len(rooms)} total, {len(ins)} in-scope (K_total>50)", flush=True)

    pop_keys = [(T, pr) for T in T_EVAL for pr in PRIORS]
    emp_keys = [(T, pr, s) for T in T_EVAL for pr in PRIORS for s in range(N_SEED)]
    fix_keys = [(T, ri) for T in T_EVAL for ri in range(len(ins))]
    with Pool(n_workers(args.workers)) as pool:
        pop = dict(zip(pop_keys, pool.map(_task_pop, pop_keys)))
        print(f"  population done ({time.time()-t0:.0f}s)", flush=True)
        emp_list = pool.map(_task_emp, emp_keys, chunksize=1)
        print(f"  empirical done ({time.time()-t0:.0f}s)", flush=True)
        fix_list = pool.map(_task_fixed, fix_keys, chunksize=1)
        print(f"  fixed-estimator done ({time.time()-t0:.0f}s)", flush=True)
    emp = {(T, pr): np.array([emp_list[emp_keys.index((T, pr, s))] for s in range(N_SEED)])
           for T in T_EVAL for pr in PRIORS}
    fixed_rows = {T: np.array([fix_list[fix_keys.index((T, ri))] for ri in range(len(ins))])
                  for T in T_EVAL}
    gt = np.nanmax(np.abs(pop[(1, "gaussian")] - pop[(1, "heavy_tail")]))
    gt100 = np.nanmax(np.abs(pop[(100, "gaussian")] - pop[(100, "heavy_tail")]))

    # ---------------- report ----------------
    L = ["# Appendix F in the truncation regime (T24)\n"]
    L.append(f"max |P_gauss(p) - P_t3(p)| (population): {gt:.3e} (T=1), {gt100:.3e} (T=100)\n")
    L.append("## Population risk (median curve over rooms)\n")
    L.append("| prior | T | p* | P* | delta(|s|) | flatness [0,3] |")
    L.append("|---|---|---|---|---|---|")
    for pr in PRIORS:
        for T in T_EVAL:
            ps, Ps, d, fl = stats(pop[(T, pr)])
            L.append(f"| {pr} | {T} | {ps:.1f} | {Ps:.3f} | {100*d:.1f}% | {fl:.3f} |")
    L.append(f"\n## Empirical ({N_SEED} seeds; median [range] over seeds)\n")
    L.append("| prior | T | p* | P* | delta(|s|) | flatness |")
    L.append("|---|---|---|---|---|---|")
    for pr in PRIORS:
        for T in T_EVAL:
            st = np.array([stats(c) for c in emp[(T, pr)]])
            L.append(f"| {pr} | {T} | {np.nanmedian(st[:,0]):.1f} "
                     f"[{np.nanmin(st[:,0]):.1f}, {np.nanmax(st[:,0]):.1f}] "
                     f"| {np.nanmedian(st[:,1]):.3f} "
                     f"| {100*np.nanmedian(st[:,2]):.1f}% "
                     f"[{100*np.nanmin(st[:,2]):.1f}%, {100*np.nanmax(st[:,2]):.1f}%] "
                     f"| {np.nanmedian(st[:,3]):.3f} |")
    L.append("\n## Task 3: fixed estimator (Gamma = lambda^|s|, alpha at the Gaussian "
             "population optimum)\n")
    L.append("| T | P Gaussian | P t3 | (t3-G)/G median [2.5, 97.5 pct] | n rooms |")
    L.append("|---|---|---|---|---|")
    for T in T_EVAL:
        d = fixed_rows[T]
        rel = (d[:, 1] - d[:, 0]) / d[:, 0]
        L.append(f"| {T} | {np.median(d[:,0]):.4f} | {np.median(d[:,1]):.4f} "
                 f"| {100*np.median(rel):+.2f}% [{100*np.percentile(rel,2.5):+.2f}%, "
                 f"{100*np.percentile(rel,97.5):+.2f}%] | {len(d)} |")
    out = out_path("appendixF_inscope.npz")
    np.savez(out, p_grid=P_GRID, s_hat=S_HAT, n_seed=N_SEED,
             rooms=np.array([r["sid"] for r in ins]),
             **{f"pop_T{T}_{pr}": pop[(T, pr)] for T in T_EVAL for pr in PRIORS},
             **{f"emp_T{T}_{pr}": emp[(T, pr)] for T in T_EVAL for pr in PRIORS},
             **{f"fixed_T{T}": fixed_rows[T] for T in T_EVAL})
    print("\n".join(L), flush=True)
    print(f"wrote {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
