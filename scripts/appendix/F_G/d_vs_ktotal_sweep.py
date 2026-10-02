#!/usr/bin/env python
"""Appendix G.4: D, 2H and D/(2H) along the matched-K_total sweep (T27 H column and
rectangle D/(2H) column; T42 "rectangle minimum" 5.80; prose S/(2H) = 0.625).

Sweep axis K_total in {50, 75, 100, 150, 200, 300} as in matched_ktotal_rates.py, K = 50
retained, so the discarded band is modes 51..K_total (empty at K_total = 50). Per
(domain, K_total), with w_n = lambda_n^-|s| / sum_band lambda^-|s| and
psi_n = sqrt(|Omega|) phi_n:
    H = sum w_n^2,  V(x) = sum w_n psi_n(x)^2,  D = Var_x[V],
    S = sum w_n^2 (C4_n - 1),  C = D - S;   Berry <=> D/(2H) = 1.
Stored per cell: [H, D, S, C, E_x[V]].

Domains: (a) the 6 generic rooms of matched_ktotal_rates.py, eigenfunctions from the
full eigensolve (DR_ROOMS, output of scripts/01_generate_rooms.py; its eigenvalue
prefix equals the legacy one, checked per room as arena_dev), integrated with a
degree-4 triangle rule; (b) the 4 rectangles, (c) the 4 3D boxes x 2 BCs, both from
exact analytic moments. generic_room() is reused by cohort_d2h.py for the 101-room
cohort.

Paths: DR_ROOMS for data/rooms, DR_MODAL for the legacy eigenvalues. No randomness.

Output: data/experiments/appendix/results_d_vs_ktotal.npz
Usage:  DR_ROOMS=/path/to/rooms python scripts/appendix/F_G/d_vs_ktotal_sweep.py
"""
import time

import numpy as np

from _common import out_path
from src.utils.paths import MODAL, ROOMS, check_paper_dataset

S_HAT = 1.13                      # matched to matched_ktotal_rates.py
K_RET = 50
K_TOTALS = [50, 75, 100, 150, 200, 300]
GENERIC = ["scene_00837", "scene_00844", "scene_00936",
           "scene_00970", "scene_00974", "scene_00984"]
RECTS = [("3x6_rect", (3.0, 6.0)), ("2x8_long", (2.0, 8.0)),
         ("4x4_square", (4.0, 4.0)), ("3x5_rect", (3.0, 5.0))]
BOXES = [("box_small", (2.5, 3.0, 2.4)), ("box_mid1", (3.0, 4.0, 2.5)),
         ("box_mid2", (4.0, 5.0, 2.7)), ("box_large", (6.0, 7.0, 3.0))]

QP_L = np.array([
    [0.108103018168070, 0.445948490915965, 0.445948490915965],
    [0.445948490915965, 0.108103018168070, 0.445948490915965],
    [0.445948490915965, 0.445948490915965, 0.108103018168070],
    [0.816847572980459, 0.091576213509771, 0.091576213509771],
    [0.091576213509771, 0.816847572980459, 0.091576213509771],
    [0.091576213509771, 0.091576213509771, 0.816847572980459]])
QP_W = np.array([0.223381589678011] * 3 + [0.109951743655322] * 3)


def require_rooms():
    if not (ROOMS / "room_00837.npz").exists():
        raise SystemExit(f"{ROOMS}/room_00837.npz not found: set DR_ROOMS to the output "
                         "folder of scripts/01_generate_rooms.py")


def band_stats_from_parts(lam_band, cum_q, c4_band, aw, area):
    """H, D, S, C for one band given cumulative sum_n lam^-s phi_n^2 at the
    quadrature points (cum_q) and the per-mode C4 of the band."""
    u = lam_band ** (-S_HAT)
    S_band = u.sum()
    w = u / S_band
    H = float(np.sum(w ** 2))
    Vq = area * cum_q / S_band                      # psi^2 = |Omega| phi^2
    pw = aw[:, None] * QP_W[None, :]
    EV = float(np.sum(pw * Vq))
    D = float(np.sum(pw * Vq * Vq)) - EV ** 2
    S = float(np.sum(w ** 2 * (c4_band - 1.0)))
    return H, D, S, D - S, EV


def generic_room(sid, block=64):
    idx = int(sid.split("_")[1])
    d = np.load(ROOMS / f"room_{idx:05d}.npz")
    ev = np.asarray(d["eigenvalues"], np.float64)
    # arena check: legacy prefix must match to the deepest K_total used
    ev_leg = np.asarray(np.load(MODAL / sid / "eigenpairs.npz",
                                allow_pickle=True)["eigenvalues"], np.float64)
    n_chk = min(max(K_TOTALS), len(ev), len(ev_leg))
    arena_dev = float(np.max(np.abs(ev[:n_chk] - ev_leg[:n_chk])))

    V = d["eigenvectors"]
    verts = np.asarray(d["mesh_nodes"], np.float64)
    tris = np.asarray(d["mesh_elements"], np.int64)
    p0, p1, p2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    tri_area = 0.5 * np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
                            - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))
    area = float(tri_area.sum())
    aw = tri_area / area
    t0, t1, t2 = tris[:, 0], tris[:, 1], tris[:, 2]

    n_max = min(max(K_TOTALS), len(ev))
    cum_q = np.zeros((len(tris), 6))
    c4 = np.zeros(n_max - K_RET)
    checkpoints = {}
    for lo in range(K_RET, n_max, block):
        hi = min(lo + block, n_max)
        B = np.asarray(V[:, lo:hi], np.float64)
        u = ev[lo:hi] ** (-S_HAT)
        for q in range(6):
            f = QP_L[q, 0] * B[t0] + QP_L[q, 1] * B[t1] + QP_L[q, 2] * B[t2]
            f2 = f * f
            c4[lo - K_RET:hi - K_RET] += QP_W[q] * (aw @ (f2 * f2))
            cum_q[:, q] += f2 @ u
        for K in K_TOTALS:
            if lo < K <= hi:
                checkpoints[K] = cum_q.copy()
    c4 *= area * area                                # m2_exact = 1/|Omega|

    rows = {}
    for K in K_TOTALS:
        if K <= K_RET or K > n_max:
            continue
        rows[K] = band_stats_from_parts(ev[K_RET:K], checkpoints[K],
                                        c4[:K - K_RET], aw, area)
    return rows, len(ev), arena_dev


def analytic_rows(L, bc):
    """Exact H, D, S, C per K_total for separable sin/cos products."""
    dim = len(L)
    lo = 1 if bc == "dirichlet" else 0
    n_need = max(K_TOTALS) + 40
    m_max = [int(np.ceil((n_need) ** (1.0 / dim) * 3)) + 6 for _ in L]
    grids = [np.arange(lo, m + 1) for m in m_max]
    idx = np.stack(np.meshgrid(*grids, indexing="ij"), -1).reshape(-1, dim)
    if bc == "neumann":
        idx = idx[idx.sum(1) > 0]
    lam = np.sum((idx * np.pi / np.asarray(L)) ** 2, axis=1)
    keep = lam > 0
    idx, lam = idx[keep], lam[keep]
    o = np.argsort(lam, kind="stable")
    idx, lam = idx[o], lam[o]

    def r_pair(a, b):
        r = np.ones(a.shape)
        both = (a >= 1) & (b >= 1)
        r[both & (a == b)] = 1.5
        r[both & (a != b)] = 1.0
        return r

    rows = {}
    for K in K_TOTALS:
        if K <= K_RET:
            continue
        ib, lb = idx[K_RET:K], lam[K_RET:K]
        u = lb ** (-S_HAT)
        w = u / u.sum()
        n = len(lb)
        ratio = np.ones((n, n))
        for dd in range(dim):
            A = np.broadcast_to(ib[:, dd][:, None], (n, n))
            B = np.broadcast_to(ib[:, dd][None, :], (n, n))
            ratio *= r_pair(A, B)
        H = float(np.sum(w ** 2))
        D = float(w @ (ratio - 1.0) @ w)
        C4 = np.diag(ratio)
        S = float(np.sum(w ** 2 * (C4 - 1.0)))
        rows[K] = (H, D, S, D - S, 1.0)
    return rows


def main():
    check_paper_dataset()
    require_rooms()
    t0 = time.time()
    gen, arena = {}, {}
    for sid in GENERIC:
        gen[sid], _, arena[sid] = generic_room(sid)
        print(f"  {sid} done ({time.time()-t0:.0f}s)", flush=True)
    rect = {n: analytic_rows(L, "dirichlet") for n, L in RECTS}
    boxd = {n: analytic_rows(L, "dirichlet") for n, L in BOXES}
    boxn = {n: analytic_rows(L, "neumann") for n, L in BOXES}

    Ks = [K for K in K_TOTALS if K > K_RET]
    med = lambda v: float(np.median(v)) if len(v) else np.nan
    rep = [f"# D/(2H) along the K_total sweep (6 generic rooms, 4 rectangles); max "
           f"legacy-vs-eigensolve eigenvalue deviation {max(arena.values()):.1e}\n",
           "| K_total | generic H | generic D/(2H) | rect H | rect D/(2H) |",
           "|---|---|---|---|---|"]
    for K in Ks:
        g = [gen[s][K] for s in GENERIC if K in gen[s]]
        r = [rect[n][K] for n in rect if K in rect[n]]
        rep.append(f"| {K} | {med([x[0] for x in g]):.4f} "
                   f"| {med([x[1]/(2*x[0]) for x in g]):.2f} "
                   f"| {med([x[0] for x in r]):.4f} "
                   f"| {med([x[1]/(2*x[0]) for x in r]):.2f} |")

    store = {}
    for s in GENERIC:
        for K, v in gen[s].items():
            store[f"generic_{s}_K{K}"] = np.array(v)
    for grp, nm in [(rect, "rect"), (boxd, "boxD"), (boxn, "boxN")]:
        for name, rows in grp.items():
            for K, v in rows.items():
                store[f"{nm}_{name}_K{K}"] = np.array(v)
    out = out_path("results_d_vs_ktotal.npz")
    np.savez(out, K_totals=np.array(K_TOTALS), s_hat=S_HAT, K_ret=K_RET,
             arena_dev=np.array([arena[s] for s in GENERIC]),
             generic_rooms=np.array(GENERIC), **store)
    print("\n".join(rep))
    print(f"wrote {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
