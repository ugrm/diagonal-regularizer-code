"""
Anisotropy sensitivity sweep.

Question: H+D showed D/(2H) = 2.15 median in generic rooms (Berry: 1.0) and
7-76 in separable domains. Does that excess anisotropy MATTER for the cost of
the fixed exponent |s|?

Model (exact population risk, no MC, no selection noise):
    y = A x + eta,  x ~ N(0, Sigma_prior), eta ~ N(0, R_MT),
    A = build_wave_temporal_matrix(Phi[:8] retained band), 2K columns,
    Sigma_prior = interleave(diag(lambda_k^{-|s|})) over the K=50 retained modes,
    R_MT = kron(R, I_T)   [MIC-MAJOR stacking — unit-tested below],
    R(a) = s2 * (I + a * E_hat),  tr(E_hat)=0, ||E_hat||_op = 1,
    s2 = tr(R_trunc)/M,  R_trunc = Phi_tr diag(lambda_n^{-|s|}) Phi_tr^T (n>K).
Estimator: closed-form diagonal Tikhonov W = (A^T A + alpha Gamma_p)^{-1} A^T,
Gamma_p = diag(lambda^p interleaved), oracle over the released 61-p x 12-alpha
grids; |s| evaluated at the exact 62nd p column. No learned models (M1/M2/M3
excluded — their features are not R-portable).

Risk (c-components, matching compute_P convention):
    num(p,alpha; a) = tr(D_c Sigma D_c^T) + s2*(N_I + a*N_dir),   D = WA - I,
    N_C = trace_cc( reg^{-1} C reg^{-1} ),  C_I = sum_m G_mm,
    C_dir = sum_{mm'} Ehat_mm' G_mm',  G_mm' = A_m^T A_m' (mic blocks),
    den = sum_k lambda_k^{-|s|}.
The noise term is LINEAR in a for fixed (p, alpha), so every a-value is free
once N_I and N_dir are computed; curvature of the a->cost curve comes entirely
from re-optimizing (p*, alpha*).

Directions:
  F1 empirical: Ehat = E_room/||E_room||_op with E_room = R_trunc/s2 - I
     (population lambda^{-|s|} tail weights — the same object H and D are built
     from; the stored E_op_empirical uses realized per-mode energies instead,
     reported alongside). PSD limit per room: a <= 1/|lambda_min(Ehat)|.
  F2 balanced k=4: diag(+a x4, -a x4), PSD a<1, eig ratio (1+a)/(1-a).
  F3 spike k=1: diag(a, -a/7 x7), PSD a<7.
  k2: diag(a,a, -a/3 x6), PSD a<3.
  Orientations for F2/F3/k2: 5 Haar-random Q (fixed seed) + 1 forward-aligned Q
  (eigenvectors of the mic-space Gram sum_t A_t A_t^T, descending — noise
  concentrated where the forward map carries signal). Sampled max is reported as
  max-over-sampled-Q, NOT an adversarial bound.

Arena: the paper's modal dataset (src.utils.paths.MODAL, 8 mics). All absolute
costs are on that dataset. Noise is synthetic (replaces truncation noise; temporally
white). Note: real truncation noise is temporally correlated; at T=1 the distinction
vanishes. |s| = 1.1265513951271393.

The printed Table T4 comes from this sweep run on a regenerated version of the modal
dataset (16 mics per room, first 8 used) instead of the paper's dataset; the 24-room
selection rule over the hd_decomposition room list, directions, markers, Haar seed, grids
and risk formula are the same. On the paper's dataset three T4 cells (one at T = 100,
two at T = 1000) differ from the printed values (see check_A_B.py).

Inputs:  data/modal (paper dataset), data/experiments/appendix/hd_decomposition.npz
         (room list for the selection rule; run hd_decomposition.py first)
Output:  data/experiments/appendix/aniso_sensitivity_sweep.{md,npz}
Usage:   python scripts/appendix/A_B/aniso_sensitivity_sweep.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.utils.paths import EXP, MODAL, check_paper_dataset  # noqa: E402
from src.physics.temporal import build_wave_temporal_matrix  # noqa: E402
from src.estimation.ridge import build_gamma_diag, ridge_solve  # noqa: E402
from src.utils.io import load_room_auto  # noqa: E402

OUT = os.path.join(str(EXP), "appendix")
OUT_NAME = "aniso_sensitivity_sweep"
MODAL_ROOT = str(MODAL)
S_HAT = 1.1265513951271393
K_RET = 50
M_USE = 8
T_EVAL = [1, 100, 1000]
A_GRID = np.array([0.0, 0.25, 0.58, 0.88, 1.5, 2.42, 3.5])
EMP_MARKS = {0.58: "median", 0.88: "mean", 2.42: "95th pct"}
N_HAAR = 5
HAAR_SEED = 314159
LAMBDA_GRID = np.array([1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0,
                        100.0, 1e3, 1e4, 1e6, 1e8, 1e10])
P_GRID = np.round(np.arange(0.0, 6.01, 0.1), 10)
WORST_BERRY = [963, 924, 860, 835, 900]
TOP_EOP = [951, 890, 929, 817, 902]


# ------------------------------------------------------------- stacking test
def stacking_unit_test():
    """Mic-major layout: row m*T+t. kron(R, I_T) is the spatially-correlated,
    temporally-white covariance; kron(I_T, R) is WRONG. Deterministic check."""
    rng = np.random.default_rng(0)
    M, T = 4, 5
    X = rng.standard_normal((M, M))
    R = X @ X.T
    K1 = np.kron(R, np.eye(T))
    for _ in range(200):
        m, mp = rng.integers(0, M, 2)
        t, tp = rng.integers(0, T, 2)
        want = R[m, mp] * (1.0 if t == tp else 0.0)
        assert abs(K1[m * T + t, mp * T + tp] - want) < 1e-12
    # verify the actual pipeline layout: rows of build_wave_temporal_matrix are
    # mic-major (row m*T+t belongs to mic m) — mic blocks are proportional to
    # Phi[m, n] within each column.
    Phi = rng.standard_normal((M, 3)) + 2.0
    ev = np.array([10.0, 40.0, 90.0])
    A = build_wave_temporal_matrix(Phi, ev, 1e-4, T, 5.1, 343.0)
    for n in range(3):
        col = A[:, 2 * n].reshape(M, T)          # mic-major reshape
        base = col[0] / Phi[0, n]
        for m in range(1, M):
            assert np.allclose(col[m], Phi[m, n] * base, atol=1e-12)
    # MC corroboration: flatten mic-major noise with spatial cov R
    L = np.linalg.cholesky(R)
    draws = np.einsum("ij,njt->nit", L, rng.standard_normal((4000, M, T)))
    flat = draws.reshape(4000, M * T)
    emp = flat.T @ flat / 4000
    err_good = np.max(np.abs(emp - K1))
    err_bad = np.max(np.abs(emp - np.kron(np.eye(T), R)))
    assert err_good < 0.25 and err_bad > 1.0
    return ("PASS: rows are mic-major (row m*T+t = mic m); R_MT = kron(R, I_T) "
            f"(MC residual {err_good:.3f}); kron(I_T, R) rejected "
            f"(residual {err_bad:.3f}).")


# ------------------------------------------------------------- per-room setup
def room_setup(sid):
    room = load_room_auto(sid, modal_root=MODAL_ROOT, k_trunc=10 ** 9)  # all modes
    ev_all = np.asarray(room["eigenvalues"], float)
    Phi_all = np.asarray(room["Phi"], float)[:M_USE]
    if len(ev_all) <= K_RET:
        return None
    ev = ev_all[:K_RET]
    Phi = Phi_all[:, :K_RET]
    sig_tr = ev_all[K_RET:] ** (-S_HAT)
    R_tr = Phi_all[:, K_RET:] @ (sig_tr[:, None] * Phi_all[:, K_RET:].T)
    s2 = float(np.trace(R_tr)) / M_USE
    E_room = R_tr / s2 - np.eye(M_USE)
    e_op = float(np.max(np.abs(np.linalg.eigvalsh(E_room))))
    Ehat = E_room / e_op
    lam_min = float(np.linalg.eigvalsh(Ehat)[0])
    a_max_f1 = np.inf if lam_min >= -1e-12 else 1.0 / abs(lam_min)
    return dict(sid=sid, ev=ev, Phi=Phi, s2=s2, Ehat=Ehat, e_op_pop=e_op,
                a_max_f1=a_max_f1, K_total=len(ev_all),
                gamma=float(room["gamma_room"]), c=float(room["c"]),
                dt=float(room["dt_sim"]))


def directions_for(A, rng_qs):
    """dict name -> list of (label, Ehat M x M). F2/F3/k2 get 5 Haar + aligned."""
    M = M_USE
    T = A.shape[0] // M
    blocks = A.reshape(M, T, -1)
    Gram = np.einsum("mtk,ntk->mn", blocks, blocks)
    w, U = np.linalg.eigh(Gram)
    U_al = U[:, ::-1]                              # descending: top signal dirs
    fams = {}
    specs = {"F2_balanced": np.array([1.0] * 4 + [-1.0] * 4),
             "F3_spike": np.array([1.0] + [-1.0 / 7] * 7),
             "k2": np.array([1.0, 1.0] + [-1.0 / 3] * 6)}
    for name, dvec in specs.items():
        qs = [("haar%d" % i, q) for i, q in enumerate(rng_qs)]
        qs.append(("aligned", U_al))
        fams[name] = [(lbl, (q * dvec[None, :]) @ q.T) for lbl, q in qs]
    return fams


def sweep_room_T(rs, T_raw, fams_dirs):
    """Returns bias(p,a-grid... ), N_I, N_dir per direction; then curves per a."""
    ev, Phi = rs["ev"], rs["Phi"]
    A = build_wave_temporal_matrix(Phi, ev, rs["dt"], int(T_raw),
                                   rs["gamma"], rs["c"])
    n2k = A.shape[1]
    ATA = A.T @ A
    blocks = A.reshape(M_USE, int(T_raw), n2k)
    Gb = np.empty((M_USE, M_USE, n2k, n2k))
    for m in range(M_USE):
        for mp in range(m, M_USE):
            g = blocks[m].T @ blocks[mp]
            Gb[m, mp] = g
            Gb[mp, m] = g.T
    C_I = np.einsum("mmij->ij", Gb)

    sig2K = np.empty(n2k)
    sig2K[0::2] = ev ** (-S_HAT)
    sig2K[1::2] = ev ** (-S_HAT)
    den = float(np.sum(sig2K[0::2]))

    dir_list = [("F1_empirical", "emp", rs["Ehat"])]
    for fam, lst in fams_dirs.items():
        for lbl, Eh in lst:
            dir_list.append((fam, lbl, Eh))
    C_dirs = [np.einsum("mn,mnij->ij", Eh, Gb) for _, _, Eh in dir_list]

    p_cols = np.concatenate([P_GRID, [S_HAT]])
    nP, nA = len(p_cols), len(LAMBDA_GRID)
    bias = np.full((nP, nA), np.nan)
    N_I = np.full((nP, nA), np.nan)
    N_D = np.full((len(dir_list), nP, nA), np.nan)
    for pi, p in enumerate(p_cols):
        g = build_gamma_diag(ev, p, is_wave=True)
        for ai, lam in enumerate(LAMBDA_GRID):
            reg = ATA + lam * np.diag(g)
            try:
                f = cho_factor(reg, lower=True)
            except np.linalg.LinAlgError:
                continue
            X1 = cho_solve(f, ATA)
            Dm = X1 - np.eye(n2k)
            bias[pi, ai] = float(np.sum(Dm[0::2, :] ** 2 * sig2K[None, :]))

            def n_of(C):
                Z = cho_solve(f, cho_solve(f, C).T)
                return float(Z.diagonal()[0::2].sum())
            N_I[pi, ai] = n_of(C_I)
            for di, C in enumerate(C_dirs):
                N_D[di, pi, ai] = n_of(C)
    return dict(bias=bias, N_I=N_I, N_D=N_D, den=den, dir_list=dir_list,
                A=A, sig2K=sig2K)


def curves_at_a(sw, s2, di, a):
    """P(p, alpha) at anisotropy amplitude a along direction di."""
    return (sw["bias"] + s2 * (sw["N_I"] + a * sw["N_D"][di])) / sw["den"]


def oracle_delta(Pmat):
    """(p*, delta_rel, delta_abs, P*) with alpha-min then p-min on the 61-grid;
    |s| at the exact 62nd column."""
    curve = np.nanmin(Pmat, axis=1)
    grid = curve[:len(P_GRID)]
    if np.all(np.isnan(grid)):
        return np.nan, np.nan, np.nan, np.nan
    j = int(np.nanargmin(grid))
    Ps = curve[len(P_GRID)]
    return P_GRID[j], (Ps - grid[j]) / grid[j], Ps - grid[j], grid[j]


# ------------------------------------------------------------- MC sanity
def mc_sanity(rs, T_raw, sw, a, di, n_draw=3000, seed=5):
    rng = np.random.default_rng(seed)
    _, _, Eh = sw["dir_list"][di]
    R = rs["s2"] * (np.eye(M_USE) + a * Eh)
    L = np.linalg.cholesky(R + 1e-14 * np.eye(M_USE))
    A = sw["A"]
    n2k = A.shape[1]
    sd = np.sqrt(sw["sig2K"])
    x = rng.standard_normal((n_draw, n2k)) * sd[None, :]
    eta = np.einsum("ij,njt->nit", L,
                    rng.standard_normal((n_draw, M_USE, int(T_raw))))
    Y = x @ A.T + eta.reshape(n_draw, -1)
    p_idx, a_idx = 25, 4                                    # p=2.5, alpha=1.0
    g = build_gamma_diag(rs["ev"], P_GRID[p_idx], is_wave=True)
    X = ridge_solve(A, Y, g, LAMBDA_GRID[a_idx])            # release solver
    err = X[0::2, :].T - x[:, 0::2]
    P_mc = float(np.sum(err ** 2) / n_draw / sw["den"])
    P_form = float(curves_at_a(sw, rs["s2"], di, a)[p_idx, a_idx])
    return P_mc, P_form


# ------------------------------------------------------------- main
def main():
    t0 = time.time()
    check_paper_dataset(MODAL_ROOT)
    os.makedirs(OUT, exist_ok=True)
    stack_msg = stacking_unit_test()
    print(stack_msg, flush=True)

    hd = np.load(os.path.join(OUT, "hd_decomposition.npz"))
    hd_ids = list(hd["room_ids"])
    H_map = dict(zip(hd_ids, hd["H"]))
    D_map = dict(zip(hd_ids, hd["D"]))

    # deterministic room list: worst-Berry + top-Eop + H-spread to ~24 in-scope
    spread = [int(hd_ids[j]) for j in
              np.linspace(0, len(hd_ids) - 1, 16).astype(int)]
    picks = list(dict.fromkeys(WORST_BERRY + TOP_EOP + spread))[:24]
    rooms = []
    for rid in picks:
        rs = room_setup(f"scene_{rid:05d}")
        if rs is not None:
            rs["rid"] = rid
            rooms.append(rs)
    print(f"rooms: {len(rooms)} in-scope; F1 PSD limits "
          f"[{min(r['a_max_f1'] for r in rooms):.2f}, "
          f"{max(r['a_max_f1'] for r in rooms):.2f}]", flush=True)

    rng = np.random.default_rng(HAAR_SEED)
    rng_qs = [np.linalg.qr(rng.standard_normal((M_USE, M_USE)))[0]
              for _ in range(N_HAAR)]

    fam_limits = {"F1_empirical": None, "F2_balanced": 1.0, "F3_spike": 7.0,
                  "k2": 3.0}
    results = []          # rows: (rid, T, fam, label, a, p*, d_rel, d_abs, P*)
    mc_rows = []
    for ri, rs in enumerate(rooms):
        fams_dirs = None
        for T_raw in T_EVAL:
            A_probe = build_wave_temporal_matrix(rs["Phi"], rs["ev"], rs["dt"],
                                                 int(T_raw), rs["gamma"], rs["c"])
            fams_dirs = directions_for(A_probe, rng_qs)
            sw = sweep_room_T(rs, T_raw, fams_dirs)
            for di, (fam, lbl, Eh) in enumerate(sw["dir_list"]):
                lim = rs["a_max_f1"] if fam == "F1_empirical" else fam_limits[fam]
                for a in A_GRID:
                    if a > 0 and lim is not None and a >= lim - 1e-9:
                        continue
                    Pmat = curves_at_a(sw, rs["s2"], di, a)
                    ps, dr, da, Pst = oracle_delta(Pmat)
                    results.append((rs["rid"], int(T_raw), fam, lbl, float(a),
                                    ps, dr, da, Pst))
            if ri < 2 and T_raw in (1, 100):
                for a, di in [(0.0, 0), (0.58, 1)]:
                    P_mc, P_form = mc_sanity(rs, T_raw, sw, a, di)
                    mc_rows.append((rs["rid"], int(T_raw), a, P_mc, P_form))
        print(f"  [{ri+1}/{len(rooms)}] room {rs['rid']} "
              f"({time.time()-t0:.0f}s)", flush=True)

    dt = np.dtype([("rid", int), ("T", int), ("fam", "U16"), ("lbl", "U10"),
                   ("a", float), ("p_star", float), ("d_rel", float),
                   ("d_abs", float), ("P_star", float)])
    res = np.array(results, dtype=dt)
    np.savez(os.path.join(OUT, OUT_NAME + ".npz"), results=res,
             rooms=np.array([r["rid"] for r in rooms]),
             a_max_f1=np.array([r["a_max_f1"] for r in rooms]),
             e_op_pop=np.array([r["e_op_pop"] for r in rooms]),
             s2=np.array([r["s2"] for r in rooms]),
             mc_sanity=np.array(mc_rows), s_value=S_HAT,
             stacking=stack_msg)

    write_report(res, rooms, mc_rows, stack_msg, H_map, D_map)
    print(f"DONE ({time.time()-t0:.0f}s)", flush=True)


def med(x):
    x = x[np.isfinite(x)]
    return np.median(x) if len(x) else np.nan


def write_report(res, rooms, mc_rows, stack_msg, H_map, D_map):
    L = ["# Anisotropy sensitivity sweep — does D/(2H) excess matter for cost?\n"]
    L.append(f"Arena: paper's modal dataset (data/modal, 8 mics); estimator = closed-form "
             f"diagonal Tikhonov, released 61-p x 12-alpha grids, |s| = "
             f"{S_HAT:.10f} at the exact p column; exact population risk "
             f"(no MC, no selection noise); prior and truncation weights "
             f"lambda^-|s| (population); noise synthetic, temporally white "
             f"(real truncation noise is temporally correlated; at T=1 the "
             f"distinction vanishes). n = {len(rooms)} in-scope rooms. All "
             f"absolute costs are on the paper's dataset.\n")
    L.append(f"**Stacking check**: {stack_msg}\n")
    L.append("Sanity: tr(R) = M*s2 at every a by construction (E_hat traceless "
             "— asserted); a = 0 vs the release pipeline solver (ridge_solve on "
             "simulated draws, 3000 each):\n")
    L.append("| room | T | a | P (pipeline MC) | P (formula) | rel diff |")
    L.append("|---|---|---|---|---|---|")
    for rid, T, a, pmc, pf in mc_rows:
        L.append(f"| {rid} | {T} | {a} | {pmc:.4f} | {pf:.4f} "
                 f"| {abs(pmc-pf)/pf*100:.1f}% |")
    L.append("")
    f1lim = [r["a_max_f1"] for r in rooms]
    L.append(f"F1 per-room PSD limits: median {np.median(f1lim):.2f}, range "
             f"[{min(f1lim):.2f}, {max(f1lim):.2f}] — cells with a >= limit "
             f"are skipped (reported n per cell). F2 limit 1, k2 limit 3, "
             f"F3 limit 7. Empirical markers: a = 0.58 (median), 0.88 (mean), "
             f"2.42 (95th pct) of the stored ||E||_op distribution.\n")

    def eig_ratio(fam, a):
        return {"F2_balanced": (1 + a) / (1 - a) if a < 1 else np.inf,
                "F3_spike": (1 + a) / (1 - a / 7),
                "k2": (1 + a) / (1 - a / 3) if a < 3 else np.inf,
                "F1_empirical": np.nan}[fam]

    for T in T_EVAL:
        L.append(f"## T = {T}\n")
        for fam in ["F1_empirical", "F2_balanced", "k2", "F3_spike"]:
            sub = res[(res["T"] == T) & (res["fam"] == fam)]
            if not len(sub):
                continue
            L.append(f"### {fam}\n")
            if fam == "F1_empirical":
                L.append("| a | n rooms | median d_rel | IQR | median d_abs (pp) "
                         "| median p* |")
                L.append("|---|---|---|---|---|---|")
                for a in A_GRID:
                    s = sub[np.isclose(sub["a"], a)]
                    if not len(s):
                        continue
                    dr = s["d_rel"]
                    mark = f" ({EMP_MARKS[a]})" if a in EMP_MARKS else ""
                    L.append(f"| {a}{mark} | {len(s)} | {100*med(dr):.2f}% "
                             f"| [{100*np.nanpercentile(dr,25):.2f}, "
                             f"{100*np.nanpercentile(dr,75):.2f}] "
                             f"| {100*med(s['d_abs']):.3f} "
                             f"| {med(s['p_star']):.2f} |")
            else:
                L.append("| a | eig ratio | median cost (rand Q) | "
                         "max-over-sampled-Q (med rooms) | aligned | median p* |")
                L.append("|---|---|---|---|---|---|")
                for a in A_GRID:
                    s = sub[np.isclose(sub["a"], a)]
                    if not len(s):
                        continue
                    rnd = s[s["lbl"] != "aligned"]
                    alg = s[s["lbl"] == "aligned"]
                    per_room_max = [np.nanmax(rnd[rnd["rid"] == r]["d_rel"])
                                    for r in np.unique(rnd["rid"])]
                    L.append(f"| {a} | {eig_ratio(fam, a):.2f} "
                             f"| {100*med(rnd['d_rel']):.2f}% "
                             f"| {100*med(np.array(per_room_max)):.2f}% "
                             f"| {100*med(alg['d_rel']):.2f}% "
                             f"| {med(rnd['p_star']):.2f} |")
            L.append("")

    # C_emp fit: excess cost vs a^2 on F1, a <= 0.88
    L.append("## Local perturbation fit (empirical, NOT a bound, NOT C_room)\n")
    L.append("| T | quantity | C_emp (per unit a^2) | R^2 | fit range |")
    L.append("|---|---|---|---|---|")
    for T in T_EVAL:
        sub = res[(res["T"] == T) & (res["fam"] == "F1_empirical")]
        for qty, col, sc in [("d_rel (rel)", "d_rel", 100), ("d_abs (pp)", "d_abs", 100)]:
            xs, ys = [], []
            for a in [0.0, 0.25, 0.58, 0.88]:
                s = sub[np.isclose(sub["a"], a)]
                if len(s):
                    xs.append(a ** 2)
                    ys.append(sc * med(s[col]))
            xs, ys = np.array(xs), np.array(ys)
            X = np.column_stack([xs, np.ones_like(xs)])
            coef, resid, *_ = np.linalg.lstsq(X, ys, rcond=None)
            ss = np.sum((ys - ys.mean()) ** 2)
            r2 = 1 - (resid[0] / ss if len(resid) and ss > 0 else np.nan)
            L.append(f"| {T} | {qty} | {coef[0]:+.3f} | {r2:.3f} | a in [0, 0.88] |")
    L.append("")

    # money calculation
    L.append("## Money calculation — implied ||E||_op and cost, Berry vs measured D\n")
    L.append("Using E||E||_F^2 = M(M-1)H + M*D at M=8 and the MEASURED "
             "Frobenius-relaxation factor ||E||_op^2/pred = 0.513 (uniform-sensor "
             "MC, hd_decomposition.md) — this makes the implied ||E||_op an EMPIRICAL "
             "estimate, not a bound. Cost read off the F1 sweep median curve "
             "(paper dataset) at each T by interpolation in a.\n")
    doms = [("generic polygons", 0.0050, 2.148),
            ("App-G rectangles", 0.0042, 7.0),
            ("3D boxes (small)", 0.0089, 15.4),
            ("3D boxes (large)", 0.0011, 75.8)]
    L.append("| domain | H | D/(2H) | implied op (Berry) | implied op (measured) "
             + "".join(f"| cost@Berry T={T} | cost@meas T={T} " for T in T_EVAL) + "|")
    L.append("|---|---|---|---|---" + "|---|---" * len(T_EVAL) + "|")
    for name, Hd, ratio in doms:
        fb = np.sqrt(0.513 * (56 + 16 * 1.0) * Hd)
        fm = np.sqrt(0.513 * (56 + 16 * ratio) * Hd)
        row = f"| {name} | {Hd:.4f} | {ratio:.1f} | {fb:.2f} | {fm:.2f} "
        for T in T_EVAL:
            sub = res[(res["T"] == T) & (res["fam"] == "F1_empirical")]
            aa, cc = [], []
            for a in A_GRID:
                s = sub[np.isclose(sub["a"], a)]
                if len(s):
                    aa.append(a)
                    cc.append(100 * med(s["d_rel"]))
            cb = np.interp(fb, aa, cc)
            cm = np.interp(fm, aa, cc)
            row += f"| {cb:.2f}% | {cm:.2f}% "
        L.append(row + "|")
    L.append("")
    L.append("Reading: 'moving from Berry's D = 2H to the measured D changes the "
             "implied anisotropy from X to Y and the cost of the fixed exponent "
             "from Z% to W%' — fill from the table above; the separable-domain "
             "rows use their own H so implied op values are those domains' "
             "own scales.\n")

    with open(os.path.join(OUT, OUT_NAME + ".md"), "w") as f:
        f.write("\n".join(L))
    print(f"Wrote {OUT_NAME}.md", flush=True)


if __name__ == "__main__":
    main()
