"""
H+D anisotropy decomposition.

For the discarded band (n > K = 50) with normalized weights w_n prop lambda_n^{-|s|}:
    V(x)  = sum_n w_n psi_n(x)^2,          E_x[V] = 1
    D     = Var_x[V] = E_x[V^2] - 1
    S     = sum_n w_n^2 (C4_n - 1)         [single-mode]
    C     = D - S                          [cross-mode covariance term]
    H     = sum_n w_n^2
Berry random waves give C4_n = 3, cross-covariances 0  =>  D = 2H exactly.
S/(2H), C/(2H), D/(2H) measure departure from Berry.

Sensor identity (i.i.d. uniform sensors, exact given orthonormality):
    E_ij = sum_n w_n psi_n(x_i) psi_n(x_j) - delta_ij   [population-normalized]
    E[E_ij^2] = H (i != j),  E[E_ii^2] = D,  E||E||_F^2 = M(M-1)H + M D
NOTE two normalizations exist and the identities split between them:
  - population-normalized E (above): identity exact, but realized tr(E) != 0
    (only E[tr E] = 0);
  - self-normalized E_hat = R/(tr(R)/M) - I (the paper's Method-B object):
    tr(E_hat) = 0 to machine precision, identity only approximate (O(sqrt(H))
    self-normalization absorption).
Both are computed; the report quotes both and their Frobenius-mass ratio.

psi normalization: psi_n = sqrt(|Omega|) * phi_n EXACTLY — the stored eigenvectors
are ARPACK mass-orthonormal (phi^T M phi = I to ~1e-9), so under the continuous
uniform measure E_x[psi_n psi_m] = delta_nm holds exactly and needs no per-mode
correction. (A per-mode division by the LUMPED second moment would be wrong:
lumped-area quadrature of phi^2 on these meshes carries a 3-5% per-mode
error — median | |Omega|*m2_lumped - 1 | ~ 3e-2 — which corrupts the exact
orthonormality and shows up quadratically in the sensor identity, E[E_ij^2]
dropping to ~0.81 H. C4 = m4/m2^2 is a ratio and is immune, which is why the
published 2.89 reproduces under either convention.) Consequently E_x[V] = 1 is an
analytic identity; the lumped quadrature reproduces it to ~3e-2 (reported), the
consistent-mass quadrature to ~1e-9 (5-room check). C4_n uses the same lumped
ratio as the published C4 median (2.89), so the 2.89 / 3.03 reconciliation is exact.

Eigenfunctions: $DR_ROOMS/room_*.npz (src.utils.paths.ROOMS; the full eigensolve
written by scripts/01_generate_rooms.py — identical meshes and identical-or-prefix
eigenvalues vs the paper's modal dataset).
Weights use lambda from the same files. |s| = 1.1265513951271393 (canonical
released value; rounding it to 1.13 changes H only in the 4th decimal —
quantified in the report).

Domains: (a) all in-scope generic rooms; (b) the 4 Appendix-G rectangles
(analytic Dirichlet sines, fcut = 1000 Hz, closed-form moments);
(c) the 4 3D boxes x {dirichlet, neumann} of box3d_flatness.py (fcut = 500 Hz,
closed-form moments, DC dropped for neumann).

The per-room diagnostic table is read from room_diagnostic_regression.npz. Besides the
per-room results, the npz stores the analytic-domain results and the per-room MC
off-/diagonal ratios (read by check_A_B.py).

Inputs:  $DR_ROOMS/room_*.npz (rooms 800-999), data/experiments/anisotropy/E_op_empirical_187.npz,
         data/experiments/appendix/room_diagnostic_regression.npz (run that script first)
Output:  data/experiments/appendix/hd_decomposition.{md,npz}
Usage:   DR_ROOMS=/path/to/rooms python scripts/appendix/A_B/hd_decomposition.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.utils.paths import EXP, ROOMS  # noqa: E402

ROOMS_DIR = str(ROOMS)
OUT = os.path.join(str(EXP), "appendix")
OUT_NAME = "hd_decomposition"

S_HAT = 1.1265513951271393
K_RET = 50
M_USE = 8
N_DRAWS = 500          # V has kurtosis ~12: diag-ratio sem ~ sqrt(11/(8*N)) ~ 0.05
MC_SEED = 20260727

WORST_BERRY = [963, 924, 860, 835, 900]
BOXES = [("small_2.5x3x2.4", (2.5, 3.0, 2.4)), ("mid_3x4x2.5", (3.0, 4.0, 2.5)),
         ("mid_4x5x2.7", (4.0, 5.0, 2.7)), ("large_6x7x3", (6.0, 7.0, 3.0))]
RECTS = [("3x6_rect", (3.0, 6.0)), ("2x8_long", (2.0, 8.0)),
         ("4x4_square", (4.0, 4.0)), ("3x5_rect", (3.0, 5.0))]
C_SOUND = 343.0
FCUT_2D, FCUT_3D = 1000.0, 500.0


# ---------------------------------------------------------------- quadrature
def tri_weights(verts, tris):
    """Lumped-area weights, normalized to sum 1 (the convention of the published C4)."""
    p0, p1, p2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    area = 0.5 * np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
                        - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))
    w = np.zeros(len(verts))
    for j in range(3):
        np.add.at(w, tris[:, j], area / 3.0)
    total = w.sum()
    return w / total, area, total


def exact_p1_moments(vals, tris, tri_area, total_area, power):
    """Exact integral of (P1 interpolant)^power per triangle / |Omega|.
    Uses int_T lam0^a lam1^b lam2^c dA = 2A a!b!c!/(a+b+c+2)!  (power 2 or 4)."""
    from math import factorial
    f = vals[tris]                                        # (n_tri, 3)
    n = power
    acc = np.zeros(len(tris))
    for a in range(n + 1):
        for b in range(n + 1 - a):
            c = n - a - b
            coef = (factorial(n) / (factorial(a) * factorial(b) * factorial(c))) * \
                   (2.0 * factorial(a) * factorial(b) * factorial(c)
                    / factorial(n + 2))
            acc += coef * f[:, 0] ** a * f[:, 1] ** b * f[:, 2] ** c
    return float(np.sum(acc * tri_area) / total_area)


# Dunavant degree-4 rule (6 points): exact for quartics on a triangle, hence
# exact for (P1 interpolant)^4 and for V^2 (V is piecewise quadratic).
QP_L = np.array([
    [0.108103018168070, 0.445948490915965, 0.445948490915965],
    [0.445948490915965, 0.108103018168070, 0.445948490915965],
    [0.445948490915965, 0.445948490915965, 0.108103018168070],
    [0.816847572980459, 0.091576213509771, 0.091576213509771],
    [0.091576213509771, 0.816847572980459, 0.091576213509771],
    [0.091576213509771, 0.091576213509771, 0.816847572980459],
])
QP_W = np.array([0.223381589678011] * 3 + [0.109951743655322] * 3)


# ---------------------------------------------------------------- generic rooms
def room_hd(path, block=256):
    d = np.load(path)
    ev = np.asarray(d["eigenvalues"], np.float64)
    if len(ev) <= K_RET:
        return None
    V = d["eigenvectors"]                                  # (nodes, K_total) f32
    verts = np.asarray(d["mesh_nodes"], np.float64)
    tris = np.asarray(d["mesh_elements"], np.int64)
    w_lump, tri_area, area = tri_weights(verts, tris)
    t0, t1, t2 = tris[:, 0], tris[:, 1], tris[:, 2]
    aw = tri_area / area                                   # triangle prob weights

    lam_d = ev[K_RET:]
    wn = lam_d ** (-S_HAT)
    wn = wn / wn.sum()
    H = float(np.sum(wn ** 2))

    Nd = len(lam_d)
    m2_l = np.empty(Nd)                                    # lumped (published path)
    m4_l = np.empty(Nd)
    m4_e = np.zeros(Nd)                                    # exact P1 quartic
    Vq = np.zeros((len(tris), 6))                          # V at quadrature points
    for lo in range(0, Nd, block):
        hi = min(lo + block, Nd)
        B = np.asarray(V[:, K_RET + lo:K_RET + hi], np.float64)
        B2 = B * B
        m2_l[lo:hi] = w_lump @ B2
        m4_l[lo:hi] = w_lump @ (B2 * B2)
        wb = wn[lo:hi]
        for q in range(6):
            f = QP_L[q, 0] * B[t0] + QP_L[q, 1] * B[t1] + QP_L[q, 2] * B[t2]
            f2 = f * f
            m4_e[lo:hi] += QP_W[q] * (aw @ (f2 * f2))
            Vq[:, q] += area * (f2 @ wb)                   # psi^2 = |Omega| phi^2
    C4_l = m4_l / m2_l ** 2
    C4_e = m4_e * area * area                              # m2_exact = 1/|Omega|
    EV = float(np.sum((aw[:, None] * QP_W[None, :]) * Vq))          # exact: = 1
    EV2 = float(np.sum((aw[:, None] * QP_W[None, :]) * (Vq * Vq)))  # exact E[V^2]
    D = EV2 - EV ** 2
    S = float(np.sum(wn ** 2 * (C4_e - 1.0)))
    S_l = float(np.sum(wn ** 2 * (C4_l - 1.0)))
    raw_dev = np.abs(area * m2_l - 1.0)                    # lumped quadrature error
    return dict(H=H, D=D, S=S, C=D - S, K_total=len(ev), Nd=Nd,
                C4_med=float(np.median(C4_l)), C4_95=float(np.percentile(C4_l, 95)),
                C4_max=float(C4_l.max()),
                C4_med_ex=float(np.median(C4_e)), S_lump=S_l, EV=EV,
                raw_dev_med=float(np.median(raw_dev)),
                wsum_C4=float(np.sum(wn ** 2 * C4_e) / np.sum(wn ** 2)),
                area=area, path=path)


def load_mesh_modes(path):
    """Reload one room's mesh + discarded-band data (kept out of the main recs
    to avoid holding ~23 GB of eigenvectors in memory)."""
    d = np.load(path)
    ev = np.asarray(d["eigenvalues"], np.float64)
    verts = np.asarray(d["mesh_nodes"], np.float64)
    tris = np.asarray(d["mesh_elements"], np.int64)
    w_lump, tri_area, area = tri_weights(verts, tris)
    lam_d = ev[K_RET:]
    wn = lam_d ** (-S_HAT)
    wn = wn / wn.sum()
    Vec = d["eigenvectors"]
    B = np.asarray(Vec[:, K_RET:], np.float64)
    m2 = w_lump @ (B * B)
    return verts, tris, tri_area, area, w_lump, wn, m2, B


def massmatrix_ev_check(path):
    """Consistent-mass check on one room: E[V] via phi^T M phi (should be 1 to
    ~1e-9, limited only by ARPACK orthonormality + f32 storage)."""
    verts, tris, tri_area, area, w_lump, wn, m2_l, B = load_mesh_modes(path)
    from src.geometry.mesh import assemble_stiffness_mass
    _, M_mat = assemble_stiffness_mass(verts, tris)
    m2_m = np.array([float(B[:, j] @ (M_mat @ B[:, j])) / area
                     for j in range(len(wn))])
    return float(np.abs(area * m2_m - 1.0).max())


# ---------------------------------------------------------------- MC identity
def sample_uniform(verts, tris, tri_area, n, rng):
    """n uniform points over Omega via area-weighted triangle + sqrt-barycentric;
    returns points' triangle rows for exact P1 interpolation."""
    probs = tri_area / tri_area.sum()
    ti = rng.choice(len(tris), size=n, p=probs)
    r1 = np.sqrt(rng.random(n))
    r2 = rng.random(n)
    l0 = 1.0 - r1
    l1 = r1 * (1.0 - r2)
    l2 = r1 * r2
    return ti, np.column_stack([l0, l1, l2])


def mc_identity(rec, rng):
    verts, tris, tri_area, area, w_lump, wn, m2, B = load_mesh_modes(rec["path"])
    H, D = rec["H"], rec["D"]
    pred = M_USE * (M_USE - 1) * H + M_USE * D
    scale = np.sqrt(area)                                  # psi = sqrt(|Omega|) phi
    offF2, diagF2, F2, OP2, TR = [], [], [], [], []
    F2h, TRh = [], []
    for _ in range(N_DRAWS):
        ti, lam = sample_uniform(verts, tris, tri_area, M_USE, rng)
        nodes = tris[ti]                                   # (M, 3)
        Psi = scale * np.einsum("mj,mjn->mn", lam, B[nodes])
        R = (Psi * wn[None, :]) @ Psi.T                    # sum_n w_n psi psi^T
        E = R - np.eye(M_USE)
        off = E - np.diag(np.diag(E))
        offF2.append(np.sum(off * off))
        diagF2.append(np.sum(np.diag(E) ** 2))
        F2.append(np.sum(E * E))
        OP2.append(np.max(np.abs(np.linalg.eigvalsh(E))) ** 2)
        TR.append(np.trace(E))
        s2 = np.trace(R) / M_USE
        Eh = R / s2 - np.eye(M_USE)
        F2h.append(np.sum(Eh * Eh))
        TRh.append(np.trace(Eh))
    return dict(pred=pred, F2=np.mean(F2), OP2=np.mean(OP2),
                off_ratio=np.mean(offF2) / (M_USE * (M_USE - 1) * H),
                diag_ratio=np.mean(diagF2) / (M_USE * D),
                tr_std=float(np.std(TR)), F2_selfnorm=np.mean(F2h),
                tr_selfnorm_max=float(np.max(np.abs(TRh))))


# ---------------------------------------------------------------- analytic
def analytic_domain(L, bc, fcut):
    """Closed-form H, S, C, D for separable sin/cos products below fcut."""
    lam_max = (2 * np.pi * fcut / C_SOUND) ** 2
    dim = len(L)
    lo = 1 if bc == "dirichlet" else 0
    grids = [np.arange(lo, int(np.floor(np.sqrt(lam_max) * Li / np.pi)) + 2)
             for Li in L]
    idx = np.stack(np.meshgrid(*grids, indexing="ij"), -1).reshape(-1, dim)
    if bc == "neumann":
        idx = idx[idx.sum(1) > 0]
    lam = np.sum((idx * np.pi / np.asarray(L)) ** 2, axis=1)
    keep = (lam > 0) & (lam <= lam_max)
    idx, lam = idx[keep], lam[keep]
    order = np.argsort(lam, kind="stable")
    idx, lam = idx[order], lam[order]
    if len(lam) <= K_RET:
        return None
    idx_d, lam_d = idx[K_RET:], lam[K_RET:]
    wn = lam_d ** (-S_HAT)
    wn = wn / wn.sum()
    H = float(np.sum(wn ** 2))

    # per-dimension normalized pair table r(a,b) = E[f_a^2 f_b^2]/(E[f_a^2]E[f_b^2])
    def r_pair(a, b):
        # a,b integer arrays; sin for dirichlet (a>=1); cos for neumann (a>=0)
        r = np.full(a.shape, 1.0)
        both = (a >= 1) & (b >= 1)
        r[both & (a == b)] = 1.5          # E[f^4]/E[f^2]^2 = (3/8)/(1/4)
        r[both & (a != b)] = 1.0          # (1/4)/(1/4)
        one0 = (a == 0) ^ (b == 0)
        r[one0] = 1.0                      # (1/2)/(1*1/2)
        r[(a == 0) & (b == 0)] = 1.0
        return r

    Nd = len(lam_d)
    ratio = np.ones((Nd, Nd))
    for d_ in range(dim):
        A = np.broadcast_to(idx_d[:, d_][:, None], (Nd, Nd))
        B = np.broadcast_to(idx_d[:, d_][None, :], (Nd, Nd))
        ratio *= r_pair(A, B)
    D = float(wn @ (ratio - 1.0) @ wn)
    C4 = np.diag(ratio)
    S = float(np.sum(wn ** 2 * (C4 - 1.0)))
    return dict(H=H, D=D, S=S, C=D - S, K_total=len(lam), Nd=Nd,
                C4_med=float(np.median(C4)), C4_max=float(C4.max()))


# ---------------------------------------------------------------- main
def fmt(rec):
    H = rec["H"]
    return (f"H={rec['H']:.4g} S/2H={rec['S']/(2*H):.3f} "
            f"C/2H={rec['C']/(2*H):.3f} D/2H={rec['D']/(2*H):.3f}")


def main():
    t0 = time.time()
    if not os.path.isdir(ROOMS_DIR):
        raise FileNotFoundError(f"{ROOMS_DIR} not found: set DR_ROOMS to the output of "
                                "scripts/01_generate_rooms.py")
    os.makedirs(OUT, exist_ok=True)
    diag = np.load(os.path.join(OUT, "room_diagnostic_regression.npz"))
    eop = np.load(os.path.join(str(EXP), "anisotropy", "E_op_empirical_187.npz"))

    recs, ids = [], []
    for i in range(800, 1000):
        p = os.path.join(ROOMS_DIR, f"room_{i:05d}.npz")
        if not os.path.exists(p):
            continue
        r = room_hd(p)
        if r is None:
            continue
        recs.append(r)
        ids.append(i)
        if len(recs) % 25 == 0:
            print(f"  [{len(recs)}] rooms, {time.time()-t0:.0f}s", flush=True)
    ids = np.array(ids)
    print(f"in-scope generic rooms: {len(recs)} ({time.time()-t0:.0f}s)", flush=True)

    get = lambda k: np.array([r[k] for r in recs])
    H_, D_, S_, C_ = get("H"), get("D"), get("S"), get("C")

    # consistent-mass EV check on 3 rooms (small / anchor / large)
    order = np.argsort(get("Nd"))
    three = [int(ids[order[0]]), 802, int(ids[order[-2]])]
    mm_dev = {rid: massmatrix_ev_check(
        os.path.join(ROOMS_DIR, f"room_{rid:05d}.npz")) for rid in three}
    print(f"consistent-mass m2 check: {mm_dev}", flush=True)

    # MC identity: worst-Berry + top-5 stored E_op + spread to >= 30 rooms
    eop_ids = [int(s.split("_")[1]) for s in eop["scene_ids"]]
    top_eop = [eop_ids[k] for k in np.argsort(eop["E_op_per_room"])[-5:]]
    mc_rooms = list(dict.fromkeys(
        WORST_BERRY + top_eop +
        [int(ids[j]) for j in np.linspace(0, len(ids) - 1, 22).astype(int)]))[:32]
    rng = np.random.default_rng(MC_SEED)
    mc = {}
    for rid in mc_rooms:
        j = np.where(ids == rid)[0]
        if not len(j):
            continue
        mc[rid] = mc_identity(recs[int(j[0])], rng)
    print(f"MC identity on {len(mc)} rooms done ({time.time()-t0:.0f}s)", flush=True)

    # analytic domains
    rect = {n: analytic_domain(L, "dirichlet", FCUT_2D) for n, L in RECTS}
    box_d = {n: analytic_domain(L, "dirichlet", FCUT_3D) for n, L in BOXES}
    box_n = {n: analytic_domain(L, "neumann", FCUT_3D) for n, L in BOXES}

    # arena check vs the per-room diagnostic table (legacy-eigensolve H) and paper values
    kH = diag["H"]
    kKt = diag["K_total"]
    kin = diag["in_scope"]
    krooms = [int(s.split("_")[1]) for s in diag["rooms"]]
    common = [(np.where(ids == r)[0][0], j) for j, r in enumerate(krooms)
              if kin[j] and r in set(ids.tolist())]
    Hdiff = np.array([abs(H_[a] - kH[b]) for a, b in common])
    s_diag = float(diag["s_value"]) if "s_value" in diag.files else np.nan

    # H at 1.13 vs canonical on one room
    r0 = recs[0]
    lam_d0 = None  # recompute quickly
    d0 = np.load(os.path.join(ROOMS_DIR, f"room_{ids[0]:05d}.npz"))
    lam_d0 = np.asarray(d0["eigenvalues"], np.float64)[K_RET:]
    w13 = lam_d0 ** (-1.13); w13 /= w13.sum()
    wc = lam_d0 ** (-S_HAT); wc /= wc.sum()
    H13, Hc = float(np.sum(w13 ** 2)), float(np.sum(wc ** 2))

    # -------------------------------------------------------------- report
    L = ["# H+D anisotropy decomposition (discarded band n > 50)\n"]
    L.append(f"|s| = {S_HAT:.10f} (canonical; using 1.13 instead changes H by "
             f"{100*abs(H13-Hc)/Hc:.3f}% on room_{ids[0]:05d}). "
             f"psi = sqrt(|Omega|) phi exactly (ARPACK mass-orthonormality), so "
             f"E_x[V] = 1 and E_x[psi_n psi_m] = delta_nm are analytic "
             f"identities. S, C, D use a degree-4 (6-point) triangle rule that "
             f"is EXACT for the piecewise-quartic integrands; E[V] = 1 is "
             f"reproduced to max |E[V]-1| = {np.max(np.abs(get('EV')-1)):.1e} "
             f"across rooms, which bounds the total numerical error. C4 for the "
             f"2.89-reconciliation uses the published lumped path. Two "
             f"quadrature traps found and documented below: per-mode lumped "
             f"normalization corrupts the sensor identity (E[E_ij^2] -> 0.81 H), "
             f"and lumped integration inflates D by ~1.5x.\n")

    L.append("## Arena check\n")
    L.append(f"- This run: v2 eigensolve (`data/rooms/`), in-scope (K_total>50) "
             f"n = {len(recs)}. Median K_total = {np.median(get('K_total')):.0f}; "
             f"median H = {np.median(H_):.4f}.")
    L.append(f"- diagnostic table (legacy eigensolve) in-scope n = {int(kin.sum())}: median "
             f"K_total = {np.nanmedian(kKt[kin]):.0f}, median H = "
             f"{np.nanmedian(kH[kin]):.4f} (table s_value = {s_diag}).")
    L.append(f"- Common in-scope rooms n = {len(common)}: max |H_v2 - H_legacy| = "
             f"{Hdiff.max():.2e}, median {np.median(Hdiff):.2e} — the eigensolves "
             f"agree; only near-cutoff mask tails differ.\n")

    L.append("## Per-domain decomposition (medians [IQR])\n")
    L.append("| domain | n | H | S/(2H) | C/(2H) | D/(2H) |")
    L.append("|---|---|---|---|---|---|")

    def med_iqr(x):
        return f"{np.median(x):.3f} [{np.percentile(x,25):.3f}, {np.percentile(x,75):.3f}]"

    L.append(f"| generic polygons (v2) | {len(recs)} | "
             f"{np.median(H_):.4f} [{np.percentile(H_,25):.4f}, {np.percentile(H_,75):.4f}] "
             f"| {med_iqr(S_/(2*H_))} | {med_iqr(C_/(2*H_))} | {med_iqr(D_/(2*H_))} |")
    for label, group in [("rectangles (Dirichlet, 1000 Hz)", rect),
                         ("3D boxes (Dirichlet, 500 Hz)", box_d),
                         ("3D boxes (Neumann, 500 Hz)", box_n)]:
        g = [v for v in group.values() if v]
        Hg = np.array([v["H"] for v in g])
        L.append(f"| {label} | {len(g)} | "
                 f"{np.median(Hg):.4f} [{Hg.min():.4f}, {Hg.max():.4f}] "
                 f"| {med_iqr(np.array([v['S']/(2*v['H']) for v in g]))} "
                 f"| {med_iqr(np.array([v['C']/(2*v['H']) for v in g]))} "
                 f"| {med_iqr(np.array([v['D']/(2*v['H']) for v in g]))} |")
    L.append("")
    L.append("Per-domain detail (analytic):\n")
    for label, group in [("rect", rect), ("box-D", box_d), ("box-N", box_n)]:
        for n, v in group.items():
            if v:
                L.append(f"- {label} {n}: K_total={v['K_total']} {fmt(v)} "
                         f"C4_med={v['C4_med']:.3f}")
    L.append("")

    L.append("## Reconciling the published C4 = 2.89\n")
    two_h = 2 * H_
    L.append(f"S/(2H) three ways (generic rooms, median over rooms):")
    L.append(f"- (i) scalar median C4 (2.89-style): median of (C4_med-1)*H/(2H) "
             f"= {np.median((get('C4_med')-1)/2):.3f}")
    L.append(f"- (ii) scalar 95th C4: {np.median((get('C4_95')-1)/2):.3f}")
    L.append(f"- (iii) actual w_n^2-weighted sum_n w_n^2(C4_n-1)/(2H): "
             f"{np.median(S_/two_h):.3f}")
    L.append(f"Median per-room C4 over rooms: {np.median(get('C4_med')):.3f} "
             f"(published 2.89); median w^2-weighted C4: "
             f"{np.median(get('wsum_C4')):.3f}.\n")

    L.append("## Sensor identity check (population-normalized E; "
             f"{N_DRAWS} uniform draws (V kurtosis ~12: per-room diag-ratio sem ~0.05), M = {M_USE})\n")
    L.append("| room | pred = M(M-1)H+MD | mean ||E||_F^2 | total ratio | "
             "offdiag/(M(M-1)H) | diag/(MD) | ||E||_op^2/pred | "
             "selfnorm ||E||_F^2/pred | max |tr selfnorm| |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for rid, m in sorted(mc.items()):
        L.append(f"| {rid} | {m['pred']:.3f} | {m['F2']:.3f} "
                 f"| {m['F2']/m['pred']:.3f} | {m['off_ratio']:.3f} "
                 f"| {m['diag_ratio']:.3f} | {m['OP2']/m['pred']:.3f} "
                 f"| {m['F2_selfnorm']/m['pred']:.3f} "
                 f"| {m['tr_selfnorm_max']:.1e} |")
    rat = np.array([m["F2"] / m["pred"] for m in mc.values()])
    ofr = np.array([m["off_ratio"] for m in mc.values()])
    dgr = np.array([m["diag_ratio"] for m in mc.values()])
    opr = np.array([m["OP2"] / m["pred"] for m in mc.values()])
    sfr = np.array([m["F2_selfnorm"] / m["pred"] for m in mc.values()])
    L.append("")
    L.append(f"Across MC rooms (medians): total ||E||_F^2/pred {np.median(rat):.3f}; "
             f"off-diagonal mass / M(M-1)H = {np.median(ofr):.3f} (exact identity, "
             f"hypothesis-free — MC noise only); diagonal mass / MD = "
             f"{np.median(dgr):.3f} (tests the quadrature-D against sampled "
             f"Var[V]); ||E||_op^2/pred = {np.median(opr):.3f} (looseness of the "
             f"Frobenius relaxation); self-normalized Frobenius mass / pred = "
             f"{np.median(sfr):.3f} (the O(sqrt(H)) absorption). tr(E)=0 holds for "
             f"the SELF-normalized convention only (max |tr| "
             f"{max(m['tr_selfnorm_max'] for m in mc.values()):.1e}); "
             f"population-normalized tr(E) fluctuates with sd ~ sqrt(M D).\n")

    L.append("## Quadrature audit (lumped vs exact P1)\n")
    dC4 = np.abs(get("C4_med") - get("C4_med_ex"))
    dS = np.abs(get("S_lump") - S_) / np.maximum(np.abs(S_), 1e-30)
    L.append(f"- C4 median, lumped vs exact-quartic: median |diff| = "
             f"{np.median(dC4):.4f}, max = {dC4.max():.4f} — the published "
             f"lumped ratio is safe for C4 (both moments share the bias).")
    L.append(f"- S term, lumped vs exact: median rel |diff| = {np.median(dS):.3f}, "
             f"max = {dS.max():.3f}.")
    L.append(f"- **Lumped quadrature is NOT safe for D**: nodal-lumped integration "
             f"of squared fields exceeds the exact P1 integral by a "
             f"gradient-energy term that grows with lambda (measured per-mode "
             f"| |Omega| m2_lumped - 1 | median "
             f"{np.median(get('raw_dev_med')):.2e}), and it inflated Var[V] by "
             f"~1.5x in the first implementation. All S, C, D, C4_ex values in "
             f"this report therefore use the exact degree-4 rule; E[V] = 1 is "
             f"reproduced to max |E[V]-1| = {np.max(np.abs(get('EV')-1)):.1e} "
             f"across all rooms (analytic identity, so this bounds the total "
             f"numerical error).")
    L.append(f"- Consistent-mass m2 spot check (3 rooms, small/anchor/large): "
             f"max | |Omega| m2_M - 1 | = "
             + ", ".join(f"room_{r:05d}: {v:.1e}" for r, v in mm_dev.items())
             + " — ARPACK mass-orthonormality at f32 storage precision.\n")

    L.append("## Targeted rooms\n")
    L.append("| room | why | H | D/(2H) | C/(2H) | stored ||E||_op |")
    L.append("|---|---|---|---|---|---|")
    eop_map = dict(zip(eop_ids, eop["E_op_per_room"]))
    for rid in WORST_BERRY + top_eop:
        j = np.where(ids == rid)[0]
        why = "worst-Berry B.4" if rid in WORST_BERRY else "top-5 stored E_op"
        if not len(j):
            L.append(f"| {rid} | {why} | (not in-scope) | | | |")
            continue
        r = recs[int(j[0])]
        L.append(f"| {rid} | {why} | {r['H']:.4f} | {r['D']/(2*r['H']):.3f} "
                 f"| {r['C']/(2*r['H']):.3f} | {eop_map.get(rid, np.nan):.2f} |")
    L.append("")
    both = [(int(ids[j]), D_[j] / (2 * H_[j])) for j in range(len(ids))]
    common_eop = [(v, eop_map[r]) for r, v in both if r in eop_map]
    if len(common_eop) > 10:
        from scipy.stats import spearmanr
        a, b = np.array(common_eop).T
        rho1 = spearmanr(a, b).correlation
        ksmap = dict(zip(krooms, diag["ks_D"]))
        c = np.array([(v, ksmap[r]) for r, v in both if r in ksmap]).T
        rho2 = spearmanr(c[0], c[1]).correlation
        L.append(f"Spearman(D/(2H), stored ||E||_op) = {rho1:+.2f}; "
                 f"Spearman(D/(2H), KS D) = {rho2:+.2f} (n={len(common_eop)}).\n")
    flag = [(int(ids[j]), D_[j] / (2 * H_[j])) for j in range(len(ids))
            if D_[j] / (2 * H_[j]) > 1.5]
    L.append(f"Rooms with D/(2H) > 1.5: {len(flag)}" +
             (": " + ", ".join(f"room_{r:05d} ({v:.2f})" for r, v in
                               sorted(flag, key=lambda t: -t[1])[:15]) if flag else "") + "\n")

    def analytic_arrays(prefix, group):
        g = [(n, v) for n, v in group.items() if v]
        return {f"{prefix}_names": np.array([n for n, _ in g]),
                **{f"{prefix}_{k}": np.array([v[k] for _, v in g])
                   for k in ("K_total", "H", "S", "C", "D")}}

    np.savez(os.path.join(OUT, OUT_NAME + ".npz"),
             room_ids=ids, H=H_, D=D_, S=S_, C=C_,
             C4_med=get("C4_med"), C4_95=get("C4_95"), C4_max=get("C4_max"),
             K_total=get("K_total"), area=get("area"),
             mc_rooms=np.array(list(mc.keys())),
             mc_pred=np.array([m["pred"] for m in mc.values()]),
             mc_F2=np.array([m["F2"] for m in mc.values()]),
             mc_OP2=np.array([m["OP2"] for m in mc.values()]),
             mc_F2_selfnorm=np.array([m["F2_selfnorm"] for m in mc.values()]),
             mc_off_ratio=np.array([m["off_ratio"] for m in mc.values()]),
             mc_diag_ratio=np.array([m["diag_ratio"] for m in mc.values()]),
             **analytic_arrays("rect", rect), **analytic_arrays("box_d", box_d),
             **analytic_arrays("box_n", box_n),
             s_value=S_HAT, n_draws=N_DRAWS, mc_seed=MC_SEED)
    with open(os.path.join(OUT, OUT_NAME + ".md"), "w") as f:
        f.write("\n".join(L))
    print(f"Wrote {OUT_NAME}.md + npz ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    main()
