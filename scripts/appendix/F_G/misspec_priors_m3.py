#!/usr/bin/env python
"""Appendix F no-truncation control: closed form vs learned M3 under three priors.

Protocol (the Appendix F control): the 20 rooms scene_00800..00819, K = min(50, K_total)
modes generated (NO truncation noise), M = 8, T in {1, 100}, 61-point p grid, 12-point
alpha grid, matched marginal variance lambda^-|s|, |s| = 1.13.

  (1) power-law family, oracle alpha at every p:
      empirical P(p) from N_DRAWS = 512 i.i.d. amplitude draws per room, and exact
      population P(p) = tr((S-I) Sigma (S-I)^T)_c / tr(Sigma)_c, S = W A;
  (2) M3 (only with --m3-checkpoint): the Gamma emitted by the five trained n = 800
      seeds (prior-independent), with oracle alpha and with the trained alpha;
  (3) finite-sample check: heavy-tail vs Gaussian delta(|s|) at T = 100 for
      N_draws in {128, 512, 4096}.
The T41 row "No-truncation control: median gain of the trained M3 over the best power
law at T = 100" (2.8 pp) is the Gaussian, T = 100, oracle-alpha gap
median_rooms(P(p*_room) - P_M3) per seed; it is stored as m3_gain_pp_T100_gaussian_*.

Notes:
  * the room list is fixed (scene_00800..00819);
  * empirical draws are seeded with zlib.crc32 (_common.stable_seed);
  * M3 weights are read from --m3-checkpoint (a directory holding
    m3_hypernet_n800_T{1,100}_s{42..46}/model_final.pt), e.g. the shipped
    data/experiments/m3_retrain_20261003 (see README.md in this folder).
    Without the argument the M3 part is skipped; with it the M3 values are saved too;
  * independent (T, prior, room) tasks run in a process pool.

Output: data/experiments/appendix/misspec_priors.npz
Usage:  python scripts/appendix/F_G/misspec_priors_m3.py [--m3-checkpoint DIR] [--workers 16]
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import time
from multiprocessing import Pool

import numpy as np

from _common import out_path, n_workers, stable_seed
from src.utils.paths import MODAL, check_paper_dataset
from src.physics.temporal import build_wave_temporal_matrix
from src.estimation.metrics import compute_P

S_HAT = 1.13
K_PAD = 50
M_USE = 8
T_EVAL = [1, 100]
P_GRID = np.round(np.arange(0, 6.01, 0.1), 10)
LAMBDA_GRID = np.array([1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0,
                        100., 1e3, 1e4, 1e6, 1e8, 1e10])
N_DRAWS = 512
SEEDS = [42, 43, 44, 45, 46]
PRIORS = ["gaussian", "heavy_tail", "correlated"]
RHO_CORR = 0.3
ROOM_IDS = [f"scene_{i:05d}" for i in range(800, 820)]
N_CONV = [128, 512, 4096]


def load_room(sid):
    d = MODAL / sid
    eig = np.load(d / "eigenpairs.npz", allow_pickle=True)
    traj = np.load(d / "modal_trajectories.npz", allow_pickle=True)
    ev_full = np.asarray(eig["eigenvalues"], float)
    K_eff = min(K_PAD, len(ev_full))
    Phi_full = np.load(d / "measurement_matrix.npy")[:M_USE]
    return dict(
        sid=sid,
        ev=ev_full[:K_eff],
        Phi=Phi_full[:, :K_eff],
        K_eff=K_eff,
        c=float(eig["c"]) if "c" in eig else 343.0,
        gamma=float(traj["gamma_room"]),
        dt=float(traj["dt_sim"]),
    )


def make_sigma(ev, prior):
    """K_eff x K_eff prior covariance for the c (and beta) component."""
    v = ev ** (-S_HAT)
    if prior == "correlated":
        S = np.diag(v)
        sd = np.sqrt(v)
        for j in range(len(v) - 1):
            S[j, j + 1] = S[j + 1, j] = RHO_CORR * sd[j] * sd[j + 1]
        return S
    return np.diag(v)


def draw_amplitudes(ev, prior, n, rng):
    """(n, 2K_eff) interleaved (c,beta) draws with matched marginal variance."""
    K = len(ev)
    sd = ev ** (-S_HAT / 2)
    if prior == "gaussian":
        c = rng.standard_normal((n, K)) * sd
        b = rng.standard_normal((n, K)) * sd
    elif prior == "heavy_tail":
        c = rng.standard_t(3, (n, K)) * sd / np.sqrt(3.0)
        b = rng.standard_t(3, (n, K)) * sd / np.sqrt(3.0)
    elif prior == "correlated":
        L = np.linalg.cholesky(make_sigma(ev, prior) + 1e-14 * np.eye(K))
        c = rng.standard_normal((n, K)) @ L.T
        b = rng.standard_normal((n, K)) @ L.T
    a = np.empty((n, 2 * K))
    a[:, 0::2] = c
    a[:, 1::2] = b
    return a


def sigma_2k(ev, prior):
    K = len(ev)
    S = make_sigma(ev, prior)
    S2 = np.zeros((2 * K, 2 * K))
    S2[0::2, 0::2] = S
    S2[1::2, 1::2] = S
    return S2


def pop_P(A, gamma_diag, alpha, Sigma2K):
    """Exact population P for the fixed linear estimator W=(A'A+aG)^-1 A'."""
    n = A.shape[1]
    reg = A.T @ A + alpha * np.diag(gamma_diag)
    try:
        S = np.linalg.solve(reg, A.T @ A)  # W A
    except np.linalg.LinAlgError:
        return np.nan
    D = S - np.eye(n)
    # target = c components only (even indices), matching compute_P convention
    Dc = D[0::2, :]
    num = float(np.trace(Dc @ Sigma2K @ Dc.T))
    den = float(np.trace(Sigma2K[0::2, 0::2]))
    if not np.isfinite(num) or num < -1e-9 * den:
        return np.nan
    return max(num, 0.0) / den


def emp_P(A, gamma_diag, alpha, a_draws):
    reg = A.T @ A + alpha * np.diag(gamma_diag)
    try:
        W = np.linalg.solve(reg, A.T)
    except np.linalg.LinAlgError:
        return np.nan
    Y = a_draws @ A.T                     # (N, MT), no noise
    est = Y @ W.T                          # (N, 2K)
    return compute_P(est[:, 0::2], a_draws[:, 0::2])


def gamma_pow(ev, p):
    g = np.empty(2 * len(ev))
    g[0::2] = ev ** p
    g[1::2] = ev ** p
    return g


def eval_curve(A, ev, draws_or_sigma, mode):
    """P(p) with oracle alpha at each p. mode: 'emp' (draws) or 'pop' (Sigma2K)."""
    out = np.full(len(P_GRID), np.nan)
    for pi, p in enumerate(P_GRID):
        g = gamma_pow(ev, p)
        vals = []
        for al in LAMBDA_GRID:
            v = (emp_P(A, g, al, draws_or_sigma) if mode == "emp"
                 else pop_P(A, g, al, draws_or_sigma))
            vals.append(v)
        out[pi] = np.nanmin(vals)
    return out


def eval_fixed_gamma(A, gamma_diag, draws_or_sigma, mode, alphas=LAMBDA_GRID):
    vals = []
    for al in alphas:
        v = (emp_P(A, gamma_diag, al, draws_or_sigma) if mode == "emp"
             else pop_P(A, gamma_diag, al, draws_or_sigma))
        vals.append(v)
    return float(np.nanmin(vals))


def m3_checkpoint_paths(ckpt_root, T_raw):
    return {s: os.path.join(ckpt_root, f"m3_hypernet_n800_T{T_raw}_s{s}", "model_final.pt")
            for s in SEEDS}


def require_m3_checkpoints(ckpt_root):
    """Fail early, with the retraining hint, if any of the ten weight files is missing."""
    missing = [p for T in T_EVAL for p in m3_checkpoint_paths(ckpt_root, T).values()
               if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            f"{len(missing)} M3 weight files missing under {ckpt_root}, e.g. {missing[0]}.\n"
            "Point --m3-checkpoint at data/experiments/m3_retrain_20261003, or retrain with\n"
            "  python scripts/10_train_models.py --model m3_hypernet --t_value T --seed S\n"
            "for T in {1, 100} and S in 42..46 (GPU), or run without --m3-checkpoint.")


def load_m3_gammas(rooms_data, T_raw, ckpt_root):
    """Emit per-room M3 Gamma (K_eff,) for each seed. Prior-independent."""
    import torch
    from src.data.features import compute_mode_features
    from src.models.m3_hypernet import ClosedFormTikhonov

    gammas = {}
    for seed, ck in m3_checkpoint_paths(ckpt_root, T_raw).items():
        state = torch.load(ck, map_location="cpu", weights_only=True)
        model = ClosedFormTikhonov(K=K_PAD, feat_dim=10)
        model.load_state_dict(state)
        model.eval()
        alpha_tr = float(torch.exp(model.log_alpha))
        per_room, per_scale = {}, {}
        for r in rooms_data:
            ev_pad = np.zeros(K_PAD)
            ev_pad[:r["K_eff"]] = r["ev"]
            Phi_pad = np.zeros((M_USE, K_PAD))
            Phi_pad[:, :r["K_eff"]] = r["Phi"]
            A_pad = build_wave_temporal_matrix(Phi_pad, ev_pad, r["dt"], T_raw,
                                               r["gamma"], r["c"])
            ATA = A_pad.T @ A_pad
            # training/eval convention: normalize by the largest eigenvalue
            scale = max(float(np.linalg.eigvalsh(ATA)[-1]), 1e-30)
            feats = compute_mode_features(ev_pad, r["c"], r["gamma"],
                                          ATA / scale, T_raw, M_USE, K_PAD)
            with torch.no_grad():
                g = model.condnet(torch.from_numpy(feats[None])).numpy()[0]
            per_room[r["sid"]] = g[:r["K_eff"]]
            per_scale[r["sid"]] = scale
        gammas[seed] = dict(alpha=alpha_tr, gamma=per_room, scale=per_scale)
    return gammas


def delta_at_s(curve):
    j = int(np.nanargmin(curve))
    P_at = np.interp(S_HAT, P_GRID, curve)
    return P_GRID[j], curve[j], (P_at - curve[j]) / curve[j]


# ---------------- pool tasks (inputs inherited from the parent via fork) -------------
_ROOMS, _M3 = [], {}


def _A(r, T_raw):
    return build_wave_temporal_matrix(r["Phi"], r["ev"], r["dt"], T_raw, r["gamma"], r["c"])


def _task_room(args):
    T_raw, prior, ri = args
    r = _ROOMS[ri]
    sid = r["sid"]
    A = _A(r, T_raw)
    rng = np.random.default_rng(stable_seed((prior, sid, T_raw)))
    draws = draw_amplitudes(r["ev"], prior, N_DRAWS, rng)
    S2 = sigma_2k(r["ev"], prior)
    out = dict(emp=eval_curve(A, r["ev"], draws, "emp"),
               pop=eval_curve(A, r["ev"], S2, "pop"))
    if _M3:
        m3 = _M3[T_raw]
        eo, et, po = [], [], []
        for s in SEEDS:
            g2 = np.empty(2 * r["K_eff"])
            g2[0::2] = m3[s]["gamma"][sid]
            g2[1::2] = m3[s]["gamma"][sid]
            eo.append(eval_fixed_gamma(A, g2 + 1e-8, draws, "emp"))
            # trained alpha acts on the eigval-normalized system:
            # (ATA/sc + a(G+1e-8/a)) <=> raw alpha_eff = sc*a on G + jitter
            sc = m3[s]["scale"][sid]
            a_tr = m3[s]["alpha"]
            et.append(emp_P(A, g2 + 1e-8 / a_tr, sc * a_tr, draws))
            po.append(eval_fixed_gamma(A, g2 + 1e-8, S2, "pop"))
        out.update(m3_emp_oracle=eo, m3_emp_trained=et, m3_pop_oracle=po)
    return out


def _task_conv(args):
    n_d, prior, ri = args
    r = _ROOMS[ri]
    T_raw = 100
    A = _A(r, T_raw)
    if n_d == "pop":
        c = eval_curve(A, r["ev"], sigma_2k(r["ev"], "gaussian"), "pop")
    else:
        rng = np.random.default_rng(stable_seed((prior, r["sid"], T_raw, n_d)))
        draws = draw_amplitudes(r["ev"], prior, n_d, rng)
        c = eval_curve(A, r["ev"], draws, "emp")
    return delta_at_s(c)[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--m3-checkpoint", default=None,
                    help="directory with m3_hypernet_n800_T{T}_s{seed}/model_final.pt")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    check_paper_dataset()
    t0 = time.time()
    _ROOMS[:] = [load_room(sid) for sid in ROOM_IDS]
    if args.m3_checkpoint:
        require_m3_checkpoints(args.m3_checkpoint)
        for T_raw in T_EVAL:
            _M3[T_raw] = load_m3_gammas(_ROOMS, T_raw, args.m3_checkpoint)
    else:
        print("no --m3-checkpoint: skipping the M3 columns (trained weights are not included)",
              flush=True)

    keys = [(T, pr, ri) for T in T_EVAL for pr in PRIORS for ri in range(len(_ROOMS))]
    conv_keys = ([(n, pr, ri) for n in N_CONV for pr in ["heavy_tail", "gaussian"]
                  for ri in range(len(_ROOMS))]
                 + [("pop", "gaussian", ri) for ri in range(len(_ROOMS))])
    with Pool(n_workers(args.workers)) as pool:
        res = dict(zip(keys, pool.map(_task_room, keys, chunksize=1)))
        print(f"landscapes done ({time.time()-t0:.0f}s)", flush=True)
        conv = dict(zip(conv_keys, pool.map(_task_conv, conv_keys, chunksize=1)))
        print(f"finite-sample check done ({time.time()-t0:.0f}s)", flush=True)

    n_r = len(_ROOMS)
    store = {}
    for T in T_EVAL:
        for pr in PRIORS:
            rows = [res[(T, pr, ri)] for ri in range(n_r)]
            store[f"{pr}_T{T}_curves_emp"] = np.array([x["emp"] for x in rows])
            store[f"{pr}_T{T}_curves_pop"] = np.array([x["pop"] for x in rows])
            if _M3:
                for k in ["m3_emp_oracle", "m3_emp_trained", "m3_pop_oracle"]:
                    store[f"{pr}_T{T}_{k}"] = np.array([x[k] for x in rows]).T  # (5, n_r)

    rep = ["# No-truncation control: closed form (and M3) under three priors\n",
           f"{n_r} rooms, K = min(50, K_total), M=8, N_draws={N_DRAWS}, |s|={S_HAT}\n",
           "| prior | T | p* | P* | d(|s|) | flat[0,3] | d(M3, or-alpha) | d(M3, tr-alpha) |",
           "|---|---|---|---|---|---|---|---|"]
    i3 = np.searchsorted(P_GRID, 3.0) + 1
    for pr in PRIORS:
        for T in T_EVAL:
            C = store[f"{pr}_T{T}_curves_emp"]
            st = np.array([delta_at_s(c) for c in C])
            fl = [np.nanmax(c[:i3]) / np.nanmin(c[:i3]) for c in C]
            m3s = "| n/a | n/a |"
            if _M3:
                Ps = st[:, 1]
                do = [np.median((store[f"{pr}_T{T}_m3_emp_oracle"][k] - Ps) / Ps)
                      for k in range(len(SEEDS))]
                dt_ = [np.median((store[f"{pr}_T{T}_m3_emp_trained"][k] - Ps) / Ps)
                       for k in range(len(SEEDS))]
                m3s = (f"| {100*np.mean(do):.1f}+-{100*np.std(do):.1f}% "
                       f"| {100*np.mean(dt_):.1f}+-{100*np.std(dt_):.1f}% |")
            rep.append(f"| {pr} | {T} | {np.median(st[:,0]):.1f} | {np.median(st[:,1]):.3f} "
                       f"| {100*np.median(st[:,2]):.1f}% | {np.median(fl):.3f} " + m3s)
    if _M3:
        for est in ["emp", "pop"]:
            C = store[f"gaussian_T100_curves_{est}"]
            Ps = np.nanmin(C, axis=1)
            M3 = store[f"gaussian_T100_m3_{est}_oracle"]
            gain = np.array([100 * np.median(Ps - M3[k]) for k in range(len(SEEDS))])
            store[f"m3_gain_pp_T100_gaussian_{est}"] = gain
            rep.append(f"\nM3 gain over the best power law, Gaussian T=100 ({est}): "
                       f"median {np.median(gain):.2f} pp, seed range "
                       f"[{gain.min():.2f}, {gain.max():.2f}] pp")
    rep.append("\n## Finite-sample check, T=100: median delta(|s|) over rooms\n")
    rep.append("| N_draws | t3 | Gaussian |")
    rep.append("|---|---|---|")
    for n in N_CONV:
        t3 = np.median([conv[(n, "heavy_tail", ri)] for ri in range(n_r)])
        ga = np.median([conv[(n, "gaussian", ri)] for ri in range(n_r)])
        rep.append(f"| {n} | {100*t3:.1f}% | {100*ga:.1f}% |")
    rep.append(f"| population | {100*np.median([conv[('pop','gaussian',ri)] for ri in range(n_r)]):.1f}% | same |")

    out = out_path("misspec_priors.npz")
    np.savez(out, room_ids=np.array(ROOM_IDS), p_grid=P_GRID, s_hat=S_HAT, **store)
    print("\n".join(rep))
    print(f"wrote {out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
