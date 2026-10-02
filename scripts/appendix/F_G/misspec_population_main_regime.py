#!/usr/bin/env python
"""Appendix F.5 (Table T26; T41 row "largest single-room movement", 13.3 pp).

Exact population risk in the main regime (truncation noise included) on all 187
in-scope rooms (scene_00800..00999, K_total > 50, blacklist excluded), for the Gaussian
and the correlated (rho = 0.3) prior; the variance-matched t3 prior has the same
covariance and therefore exactly the Gaussian risk, so it is not computed.

Model per room: all K_total modes evolve as damped sinusoids; y = A_full x with
x = interleaved (c, beta) over ALL modes and prior covariance Sigma_full. Estimators act
on the retained 2K block: W(p, alpha) = (A_ret^T A_ret + alpha Gamma_p)^-1 A_ret^T, and
risk = tr(D_c Sigma_full D_c^T) / tr(Sigma_ret,cc) with D = W A_full - Sel_ret.

Prior diagonal (convention note): Sigma_full's diagonal is each room's REALIZED per-mode
energy var(modal_trajectories["a"]) (what the main experiments operate on), not
lambda^-|s|; the correlated prior adds the rho band on the same marginals.

Estimators: the power-law family over the 61-point p grid (oracle alpha on the 12-point
grid) -> per-room oracle p*, P* and delta(|s|) (key *_dsl); and, only with
--m3-checkpoint, the five trained M3 seeds with oracle alpha (*_m3or) and trained alpha
(*_m3tr), stored as relative cost against the per-room power-law oracle, shape (5, 187).

The M3 weights come from --m3-checkpoint (e.g. data/experiments/m3_retrain_20261003;
without the argument the *_m3or / *_m3tr keys are not written). Rooms run in a process pool.
At near-singular alpha RoomRisk.risk loses precision to cancellation, so a few curve
entries depend on BLAS threading at the 1e-6 level; no printed value moves.

Output: data/experiments/appendix/misspec_population_187.npz
Usage:  python scripts/appendix/F_G/misspec_population_main_regime.py [--m3-checkpoint DIR]
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import time
from multiprocessing import Pool

import numpy as np

from _common import out_path, n_workers
from misspec_priors_m3 import (load_m3_gammas, require_m3_checkpoints, S_HAT, P_GRID,
                               LAMBDA_GRID)
from src.utils.paths import MODAL, check_paper_dataset
from src.physics.temporal import build_wave_temporal_matrix

K_RET = 50
M_USE = 8
T_EVAL = [1, 100]
SEEDS = [42, 43, 44, 45, 46]
RHO = 0.3
BLACKLIST = {"scene_00905", "scene_00913", "scene_00921"}


def room_ids():
    out = []
    for i in range(800, 1000):
        sid = f"scene_{i:05d}"
        p = MODAL / sid / "eigenpairs.npz"
        if sid in BLACKLIST or not p.exists():
            continue
        if len(np.load(p, allow_pickle=True)["eigenvalues"]) > K_RET:
            out.append(sid)
    return out


def load_room_full(sid):
    d = MODAL / sid
    eig = np.load(d / "eigenpairs.npz", allow_pickle=True)
    traj = np.load(d / "modal_trajectories.npz", allow_pickle=True)
    ev = np.asarray(eig["eigenvalues"], float)
    Phi = np.load(d / "measurement_matrix.npy")[:M_USE]
    var_real = np.var(np.asarray(traj["a"], float), axis=1)  # realized energies
    return dict(sid=sid, ev=ev, Phi=Phi, K_eff=min(K_RET, len(ev)),
                K_total=len(ev), var_real=var_real,
                c=float(eig["c"]) if "c" in eig else 343.0,
                gamma=float(traj["gamma_room"]), dt=float(traj["dt_sim"]))


def sigma_full_2k(room, prior):
    """(2K_total, 2K_total) operating covariance, interleaved (c,beta).

    Diagonal = the room's REALIZED per-mode energies; 'correlated' adds the rho=0.3
    adjacent-mode band on top of the same marginals.
    """
    v = np.maximum(room["var_real"], 1e-30)
    n = len(v)
    S = np.diag(v)
    if prior == "correlated":
        sd = np.sqrt(v)
        for j in range(n - 1):
            S[j, j + 1] = S[j + 1, j] = RHO * sd[j] * sd[j + 1]
    S2 = np.zeros((2 * n, 2 * n))
    S2[0::2, 0::2] = S
    S2[1::2, 1::2] = S
    return S2


class RoomRisk:
    """Exact population risk, reduced to the 2K_eff retained space.

    y = A_ret x_r + A_tr x_t;  W = reg^-1 A_ret^T;  reg = ATA + alpha Gamma.
    error = W y - x_r (c-rows) = D_r x_r + D_t x_t,
      D_r = -alpha reg^-1 Gamma  (exact),  D_t = reg^-1 G_rt,
    risk = tr_c[ D_r S_rr D_r' + D_t S_tt' terms + cross ] with
      C1 = G_rt S_tt G_rt',  C2 = G_rt S_tr  (both 2K_eff x 2K_eff, precomputed).
    """

    def __init__(self, r, T_raw, prior):
        K_eff = r["K_eff"]
        Sig = sigma_full_2k(r, prior)
        nr = 2 * K_eff
        A_full = build_wave_temporal_matrix(r["Phi"], r["ev"], r["dt"], T_raw,
                                            r["gamma"], r["c"])
        A_ret, A_tr = A_full[:, :nr], A_full[:, nr:]
        self.K_eff = K_eff
        self.ATA = A_ret.T @ A_ret
        self.S_rr = Sig[:nr, :nr]
        S_tt = Sig[nr:, nr:]
        S_rt = Sig[:nr, nr:]
        G_rt = A_ret.T @ A_tr                       # (2K_eff, 2N_t)
        self.C1 = G_rt @ S_tt @ G_rt.T if S_tt.size else np.zeros((nr, nr))
        self.C2 = G_rt @ S_rt.T if S_rt.size else np.zeros((nr, nr))
        self.den = float(np.trace(self.S_rr[0::2, 0::2]))
        self.cmask = np.arange(0, nr, 2)

    def risk(self, gamma_diag, alpha):
        reg = self.ATA + alpha * np.diag(gamma_diag)
        try:
            Rinv_G = np.linalg.solve(reg, np.diag(gamma_diag))
            Rinv_C1 = np.linalg.solve(reg, self.C1)
        except np.linalg.LinAlgError:
            return np.nan
        D_r = -alpha * Rinv_G                        # (nr, nr)
        term_rr = D_r @ self.S_rr @ D_r.T
        term_tt = np.linalg.solve(reg, Rinv_C1.T).T  # reg^-1 C1 reg^-1
        cross = D_r @ np.linalg.solve(reg, self.C2).T    # D_r S_rt D_t' = D_r C2^T reg^-1
        # D_t S_tr D_r' = cross.T ; total = term_rr + term_tt + cross + cross.T
        total = term_rr + term_tt + cross + cross.T
        num = float(np.trace(total[np.ix_(self.cmask, self.cmask)]))
        # population risk is a PSD quadratic form; negative/non-finite values
        # signal a near-singular solve (e.g. tiny Gamma at tiny alpha)
        if not np.isfinite(num) or num < -1e-9 * self.den:
            return np.nan
        return max(num, 0.0) / self.den


def sweep_room(r, T_raw, prior, m3_gammas):
    rr = RoomRisk(r, T_raw, prior)
    K_eff = r["K_eff"]
    ev_r = r["ev"][:K_eff]

    P_curve = np.full(len(P_GRID), np.nan)
    for pi, p in enumerate(P_GRID):
        g = np.empty(2 * K_eff)
        g[0::2] = ev_r ** p
        g[1::2] = ev_r ** p
        vals = [rr.risk(g, al) for al in LAMBDA_GRID]
        P_curve[pi] = np.nanmin(vals)

    m3_or, m3_tr = {}, {}
    if m3_gammas:
        for s in SEEDS:
            gam = m3_gammas[s]["gamma"][r["sid"]]
            g = np.empty(2 * K_eff)
            g[0::2] = gam
            g[1::2] = gam
            # deployment-consistent jitter (the model solves with alpha*gamma+1e-8)
            m3_or[s] = np.nanmin([rr.risk(g + 1e-8 / al, al) for al in LAMBDA_GRID])
            sc = m3_gammas[s]["scale"][r["sid"]]
            a_tr = m3_gammas[s]["alpha"]
            m3_tr[s] = rr.risk(g + 1e-8 / a_tr, sc * a_tr)
    return P_curve, m3_or, m3_tr


def delta_at_s(curve):
    j = int(np.nanargmin(curve))
    P_at = np.interp(S_HAT, P_GRID, curve)
    return P_GRID[j], curve[j], (P_at - curve[j]) / curve[j]


# ---------------- pool task (inputs inherited from the parent via fork) --------------
_ROOMS, _M3 = [], {}


def _task(args):
    T_raw, prior, ri = args
    return sweep_room(_ROOMS[ri], T_raw, prior, _M3.get(T_raw))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--m3-checkpoint", default=None,
                    help="directory with m3_hypernet_n800_T{T}_s{seed}/model_final.pt")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    check_paper_dataset()
    t0 = time.time()
    ids = room_ids()
    _ROOMS[:] = [load_room_full(sid) for sid in ids]
    rooms = _ROOMS
    if args.m3_checkpoint:
        require_m3_checkpoints(args.m3_checkpoint)
        m3_rooms = [dict(sid=r["sid"], ev=r["ev"][:r["K_eff"]],
                         Phi=r["Phi"][:, :r["K_eff"]], K_eff=r["K_eff"],
                         c=r["c"], gamma=r["gamma"], dt=r["dt"]) for r in rooms]
        for T_raw in T_EVAL:
            _M3[T_raw] = load_m3_gammas(m3_rooms, T_raw, args.m3_checkpoint)
    else:
        print("no --m3-checkpoint: skipping the M3 columns",
              flush=True)
    print(f"{len(rooms)} rooms", flush=True)

    priors = ["gaussian", "correlated"]
    keys = [(T, pr, ri) for T in T_EVAL for pr in priors for ri in range(len(rooms))]
    with Pool(n_workers(args.workers)) as pool:
        res = dict(zip(keys, pool.map(_task, keys, chunksize=2)))
    print(f"sweeps done ({time.time()-t0:.0f}s)", flush=True)

    rep = ["# Exact population risk, main regime (truncation noise included)\n",
           f"{len(rooms)} rooms, K=50, M=8, |s|={S_HAT}\n",
           "| prior | T | p* | P* | d(|s|) | d(M3 or-a) mean+-std | d(M3 tr-a) mean+-std |",
           "|---|---|---|---|---|---|---|"]
    store = {}
    for T_raw in T_EVAL:
        for prior in priors:
            curves, d_s_list, p_star_list, P_star_list = [], [], [], []
            d_o = {s: [] for s in SEEDS}
            d_t = {s: [] for s in SEEDS}
            for ri in range(len(rooms)):
                curve, m3_or, m3_tr = res[(T_raw, prior, ri)]
                curves.append(curve)
                ps, Ps, ds = delta_at_s(curve)
                p_star_list.append(ps); P_star_list.append(Ps); d_s_list.append(ds)
                for s in m3_or:
                    d_o[s].append((m3_or[s] - Ps) / Ps)
                    d_t[s].append((m3_tr[s] - Ps) / Ps)
            m3s = "| n/a | n/a |"
            if _M3:
                med_o = [np.median(d_o[s]) for s in SEEDS]
                med_t = [np.median(d_t[s]) for s in SEEDS]
                m3s = (f"| {100*np.mean(med_o):.2f}+-{100*np.std(med_o):.2f}% "
                       f"| {100*np.mean(med_t):.2f}+-{100*np.std(med_t):.2f}% |")
                store[f"{prior}_T{T_raw}_m3or"] = np.array([d_o[s] for s in SEEDS])
                store[f"{prior}_T{T_raw}_m3tr"] = np.array([d_t[s] for s in SEEDS])
            rep.append(f"| {prior} | {T_raw} | {np.median(p_star_list):.1f} "
                       f"| {np.median(P_star_list):.3f} "
                       f"| {100*np.median(d_s_list):.2f}% " + m3s)
            store[f"{prior}_T{T_raw}_curves"] = np.array(curves)
            store[f"{prior}_T{T_raw}_dsl"] = np.array(d_s_list)
    rep.append("| heavy_tail | any | = gaussian row (exact) | | | | |")

    rep.append("\n## Landscape shift, correlated vs gaussian (exact)\n")
    for T_raw in T_EVAL:
        g = store[f"gaussian_T{T_raw}_curves"]
        c = store[f"correlated_T{T_raw}_curves"]
        rel = np.nanmax(np.abs(c - g) / np.maximum(g, 1e-12), axis=1)
        rep.append(f"- T={T_raw}: median over rooms of max_p |P_corr-P_gauss|/P_gauss "
                   f"= {100*np.median(rel):.2f}% (max room {100*np.max(rel):.2f}%)")

    out = out_path("misspec_population_187.npz")
    np.savez(out, p_grid=P_GRID, room_ids=np.array(ids), s_hat=S_HAT, **store)
    print("\n".join(rep))
    print(f"wrote {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
