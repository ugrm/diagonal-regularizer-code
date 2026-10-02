"""
Verify the trace-free Frobenius identity against code.

Claims:
 (A) E||E^pop||_F^2 = M(M-1)H + M*D
 (B) E||E_tf ||_F^2 = (M-1)(M*H + D)
 (C) E^paper = E_tf/(1+dbar),  dbar = tr(E^pop)/M,  sd(dbar) = sqrt(D/M)
 (D) which object produced 0.58/0.88/2.42

Weight-convention note (checked, not assumed): the paper's H uses POPULATION
weights w_n prop lambda_n^-|s| (Appendix A), while the stored
||E||_op path (extract_E_op_per_room.py) uses REALIZED per-mode variances
var_t(a_n). Both conventions are computed here and reported separately.

Paths: MODAL = paper dataset, ROOMS = full eigensolve, data/experiments. The H/D table is
read from hd_decomposition.npz. Besides the report, the npz stores the per-room sd(dbar)
values (T32 "trace factor" row).

Inputs:  $DR_ROOMS/room_*.npz, data/modal (paper dataset), E_op_empirical_187.npz,
         data/experiments/appendix/hd_decomposition.npz (run hd_decomposition.py first)
Output:  data/experiments/appendix/verify_tracefree_identity.{md,npz}
Usage:   DR_ROOMS=/path/to/rooms python scripts/appendix/A_B/verify_tracefree_identity.py
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.utils.paths import EXP, MODAL, check_paper_dataset  # noqa: E402
from src.utils.paths import ROOMS as _ROOMS  # noqa: E402

LEGACY = str(MODAL)
ROOMS = str(_ROOMS)
OUT = os.path.join(str(EXP), "appendix")
OUT_NAME = "verify_tracefree_identity"

S_HAT = 1.1265513951271393
K_RET, M_USE = 50, 8
N_DRAW = 800
SEED = 90210
QP_L = np.array([
    [0.108103018168070, 0.445948490915965, 0.445948490915965],
    [0.445948490915965, 0.108103018168070, 0.445948490915965],
    [0.445948490915965, 0.445948490915965, 0.108103018168070],
    [0.816847572980459, 0.091576213509771, 0.091576213509771],
    [0.091576213509771, 0.816847572980459, 0.091576213509771],
    [0.091576213509771, 0.091576213509771, 0.816847572980459]])
QP_W = np.array([0.223381589678011] * 3 + [0.109951743655322] * 3)


def load_room(idx):
    d = np.load(os.path.join(ROOMS, f"room_{idx:05d}.npz"))
    ev = np.asarray(d["eigenvalues"], np.float64)
    V = np.asarray(d["eigenvectors"], np.float64)
    verts = np.asarray(d["mesh_nodes"], np.float64)
    tris = np.asarray(d["mesh_elements"], np.int64)
    p0, p1, p2 = verts[tris[:, 0]], verts[tris[:, 1]], verts[tris[:, 2]]
    ta = 0.5 * np.abs((p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
                      - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1]))
    return ev, V, verts, tris, ta, float(ta.sum())


def HD_from_weights(w, V, tris, ta, area):
    """H and exact-quadrature D for weights w over the discarded band."""
    H = float(np.sum(w ** 2))
    t0, t1, t2 = tris[:, 0], tris[:, 1], tris[:, 2]
    aw = ta / area
    Vq = np.zeros((len(tris), 6))
    B = V[:, K_RET:K_RET + len(w)]
    for q in range(6):
        f = QP_L[q, 0] * B[t0] + QP_L[q, 1] * B[t1] + QP_L[q, 2] * B[t2]
        Vq[:, q] = area * ((f * f) @ w)
    pw = aw[:, None] * QP_W[None, :]
    EV = float(np.sum(pw * Vq))
    D = float(np.sum(pw * Vq * Vq)) - EV ** 2
    return H, D, EV


def sample_pts(verts, tris, ta, n, rng):
    p = ta / ta.sum()
    ti = rng.choice(len(tris), size=n, p=p)
    r1 = np.sqrt(rng.random(n)); r2 = rng.random(n)
    lam = np.column_stack([1 - r1, r1 * (1 - r2), r1 * r2])
    return ti, lam


def mc_room(ev, V, verts, tris, ta, area, w, rng, n_draw=N_DRAW):
    """MC over sensor draws: realized ||E^pop||_F^2, ||E_tf||_F^2, dbar, ops."""
    nb = len(w)
    B = V[:, K_RET:K_RET + nb]
    scale = np.sqrt(area)
    F2p, F2t, dbars, op_p, op_t = [], [], [], [], []
    for _ in range(n_draw):
        ti, lam = sample_pts(verts, tris, ta, M_USE, rng)
        Psi = scale * np.einsum("mj,mjn->mn", lam, B[tris[ti]])
        R = (Psi * w[None, :]) @ Psi.T
        Ep = R - np.eye(M_USE)
        db = np.trace(Ep) / M_USE
        Et = Ep - db * np.eye(M_USE)
        F2p.append(np.sum(Ep * Ep)); F2t.append(np.sum(Et * Et))
        dbars.append(db)
        op_p.append(np.max(np.abs(np.linalg.eigvalsh(Ep))) ** 2)
        op_t.append(np.max(np.abs(np.linalg.eigvalsh(Et))) ** 2)
        # (C): the paper's object built the way the code builds it
        s2 = np.trace(R) / M_USE
        E_paper = R / s2 - np.eye(M_USE)
        assert np.max(np.abs(E_paper - Et / (1 + db))) < 1e-10 * max(1.0, np.abs(Et).max())
    return (float(np.mean(F2p)), float(np.mean(F2t)), float(np.std(dbars)),
            float(np.mean(op_p)), float(np.mean(op_t)))


def main():
    t0 = time.time()
    check_paper_dataset(LEGACY)
    if not os.path.isdir(ROOMS):
        raise FileNotFoundError(f"{ROOMS} not found: set DR_ROOMS to the output of "
                                "scripts/01_generate_rooms.py")
    hd = np.load(os.path.join(OUT, "hd_decomposition.npz"))
    hd_ids = [int(r) for r in hd["room_ids"]]
    eop = np.load(os.path.join(str(EXP), "anisotropy", "E_op_empirical_187.npz"))
    eop_ids = [int(s.split("_")[1]) for s in eop["scene_ids"]]
    eop_map = dict(zip(eop_ids, eop["E_op_per_room"]))
    print(f"stored E_op: n={len(eop_ids)} median={np.median(eop['E_op_per_room']):.4f} "
          f"mean={np.mean(eop['E_op_per_room']):.4f} "
          f"95th={np.percentile(eop['E_op_per_room'],95):.4f}", flush=True)

    # ---- (D) reproduce 0.58 from the raw legacy path A.2 used ----
    rep_op = []
    for r in eop_ids:
        d = os.path.join(LEGACY, f"scene_{r:05d}")
        Phi = np.load(os.path.join(d, "measurement_matrix.npy"))[:M_USE]
        a = np.load(os.path.join(d, "modal_trajectories.npz"))["a"]
        sig = np.var(np.asarray(a, float), axis=1)[K_RET:]
        Pt = np.asarray(Phi, float)[:, K_RET:]
        R = Pt @ (sig[:, None] * Pt.T)
        E = R / (np.trace(R) / M_USE) - np.eye(M_USE)
        rep_op.append(np.max(np.abs(np.linalg.eigvalsh(E))))
    rep_op = np.array(rep_op)
    print(f"(D) reproduced from raw legacy: median={np.median(rep_op):.4f} "
          f"mean={np.mean(rep_op):.4f} 95th={np.percentile(rep_op,95):.4f} "
          f"max|diff vs stored|={np.max(np.abs(rep_op-np.array([eop_map[r] for r in eop_ids]))):.2e}",
          flush=True)

    # ---- identity MC on a spread of rooms, BOTH weight conventions ----
    sub = [eop_ids[j] for j in np.linspace(0, len(eop_ids) - 1, 30).astype(int)]
    rows = []
    rng = np.random.default_rng(SEED)
    for r in sub:
        ev, V, verts, tris, ta, area = load_room(r)
        nb = len(ev) - K_RET
        if nb < 5:
            continue
        wp = ev[K_RET:] ** (-S_HAT); wp /= wp.sum()
        a = np.load(os.path.join(LEGACY, f"scene_{r:05d}",
                                 "modal_trajectories.npz"))["a"]
        vr = np.var(np.asarray(a, float), axis=1)[K_RET:]
        nb2 = min(len(vr), nb)
        wr = np.maximum(vr[:nb2], 1e-300); wr = wr / wr.sum()
        out = {}
        for tag, w in [("pop", wp), ("real", wr)]:
            H, D, EV = HD_from_weights(w, V, tris, ta, area)
            F2p, F2t, sd_db, opp, opt = mc_room(ev, V, verts, tris, ta, area,
                                                w, rng)
            predA = M_USE * (M_USE - 1) * H + M_USE * D
            predB = (M_USE - 1) * (M_USE * H + D)
            out[tag] = dict(H=H, D=D, EV=EV, F2p=F2p, F2t=F2t, sd_db=sd_db,
                            predA=predA, predB=predB, ratioA=F2p / predA,
                            ratioB=F2t / predB, opt=opt,
                            relax_tf=opt / predB, sd_pred=np.sqrt(D / M_USE))
        rows.append((r, out))
        if len(rows) % 10 == 0:
            print(f"  MC {len(rows)}/{len(sub)} ({time.time()-t0:.0f}s)", flush=True)

    def med(key, tag):
        return float(np.median([o[tag][key] for _, o in rows]))

    def iqr(key, tag):
        v = [o[tag][key] for _, o in rows]
        return float(np.percentile(v, 25)), float(np.percentile(v, 75))

    # ---- bound comparison on all 187 in-scope rooms of the paper ----
    common = [r for r in eop_ids if r in hd_ids]
    H187 = np.array([hd["H"][hd_ids.index(r)] for r in common])
    D187 = np.array([hd["D"][hd_ids.index(r)] for r in common])
    op187 = np.array([eop_map[r] for r in common])
    berry = np.sqrt((M_USE - 1) * (M_USE * H187 + 2 * H187))
    corr = np.sqrt((M_USE - 1) * (M_USE * H187 + D187))
    old = np.sqrt(M_USE * (M_USE + 1) * H187)          # the paper's printed form

    L = ["# Trace-free Frobenius identity\n"]
    L.append(f"M = {M_USE}, {N_DRAW} sensor draws/room, 30-room MC subset, "
             f"exact degree-4 quadrature for D. Identity (C) is asserted "
             f"numerically inside every draw (max dev < 1e-10).\n")
    L.append("## Analytic derivation (confirmed by the MC below)\n")
    L.append("`||E_tf||_F^2 = ||E^pop||_F^2 - M*dbar^2` and `E[dbar^2] = D/M` for "
             "i.i.d. sensors, so\n")
    L.append("```\nE||E_tf||_F^2 = M(M-1)H + M*D - D = (M-1)(M*H + D)\n```\n")
    L.append("and `R_trunc = (S/|Omega|)(E^pop + I)` gives `E^paper = "
             "(E^pop+I)/(1+dbar) - I = E_tf/(1+dbar)` exactly.\n")

    L.append("## (A) and (B): empirical / predicted, median [IQR] over 30 rooms\n")
    L.append("| weights | (A) ||E^pop||_F^2 ratio | (B) ||E_tf||_F^2 ratio | "
             "sd(dbar) measured / sqrt(D/M) |")
    L.append("|---|---|---|---|")
    for tag, nm in [("pop", "population lambda^-|s|"), ("real", "realized var(a)")]:
        a1, a2 = iqr("ratioA", tag); b1, b2 = iqr("ratioB", tag)
        L.append(f"| {nm} | {med('ratioA',tag):.3f} [{a1:.3f}, {a2:.3f}] "
                 f"| **{med('ratioB',tag):.3f}** [{b1:.3f}, {b2:.3f}] "
                 f"| {med('sd_db',tag):.4f} / {med('sd_pred',tag):.4f} |")
    L.append("")

    L.append("## Bound comparison, all 187 in-scope rooms of the paper\n")
    L.append("| quantity | median | IQR |")
    L.append("|---|---|---|")
    for nm, v in [("empirical ||E^paper||_op (stored)", op187),
                  ("paper's printed form sqrt(M(M+1)H)", old),
                  ("Berry trace-free sqrt((M-1)(M*H+2H))", berry),
                  ("corrected trace-free sqrt((M-1)(M*H+D))", corr)]:
        L.append(f"| {nm} | {np.median(v):.4f} | [{np.percentile(v,25):.4f}, "
                 f"{np.percentile(v,75):.4f}] |")
    L.append("")
    L.append(f"- fraction of rooms with empirical ||E||_op **below** the corrected "
             f"bound: **{np.mean(op187 < corr):.3f}** ({int(np.sum(op187<corr))}/187)")
    L.append(f"- fraction below the Berry trace-free bound: "
             f"{np.mean(op187 < berry):.3f} ({int(np.sum(op187<berry))}/187)")
    L.append(f"- fraction below the paper's printed sqrt(M(M+1)H): "
             f"{np.mean(op187 < old):.3f}")
    L.append(f"- median slack, corrected bound vs empirical: "
             f"{np.median(corr)/np.median(op187):.3f}x "
             f"(paper's printed form: {np.median(old)/np.median(op187):.3f}x)")
    L.append(f"- median per-room ratio empirical/corrected: "
             f"{np.median(op187/corr):.3f}\n")

    L.append("## Trace-free relaxation ratio (analogue of the 0.513)\n")
    for tag, nm in [("pop", "population weights"), ("real", "realized weights")]:
        r1, r2 = iqr("relax_tf", tag)
        L.append(f"- {nm}: median ||E_tf||_op^2 / [(M-1)(M*H+D)] = "
                 f"**{med('relax_tf',tag):.3f}** [{r1:.3f}, {r2:.3f}]")
    L.append("")

    d2h_all = hd["D"] / (2 * hd["H"])
    d2h_187 = D187 / (2 * H187)
    L.append("## n for D/(2H)\n")
    L.append(f"- as reported in hd_decomposition.md: n = {len(d2h_all)} (v2 in-scope, "
             f"K_total > 50), median **{np.median(d2h_all):.3f}**")
    L.append(f"- on exactly the 187 in-scope rooms of the paper: n = {len(d2h_187)}, "
             f"median **{np.median(d2h_187):.3f}** "
             f"[{np.percentile(d2h_187,25):.2f}, {np.percentile(d2h_187,75):.2f}]\n")

    np.savez(os.path.join(OUT, OUT_NAME + ".npz"),
             rooms=np.array(common), H=H187, D=D187, op=op187,
             berry=berry, corrected=corr, printed=old,
             mc_rooms=np.array([r for r, _ in rows]),
             ratioA_pop=np.array([o["pop"]["ratioA"] for _, o in rows]),
             ratioB_pop=np.array([o["pop"]["ratioB"] for _, o in rows]),
             ratioA_real=np.array([o["real"]["ratioA"] for _, o in rows]),
             ratioB_real=np.array([o["real"]["ratioB"] for _, o in rows]),
             relax_tf_pop=np.array([o["pop"]["relax_tf"] for _, o in rows]),
             relax_tf_real=np.array([o["real"]["relax_tf"] for _, o in rows]),
             sd_db_pop=np.array([o["pop"]["sd_db"] for _, o in rows]),
             sd_db_real=np.array([o["real"]["sd_db"] for _, o in rows]),
             sd_pred_pop=np.array([o["pop"]["sd_pred"] for _, o in rows]),
             sd_pred_real=np.array([o["real"]["sd_pred"] for _, o in rows]),
             rep_op=rep_op)
    with open(os.path.join(OUT, OUT_NAME + ".md"), "w") as f:
        f.write("\n".join(L))
    print("\n".join(L[8:]), flush=True)
    print(f"DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
