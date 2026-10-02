#!/usr/bin/env python
"""Appendix F, T41 row "Largest KS gap between the noise marginals of the
finite-variance alternatives and the Gaussian" (0.017).

Pooled normality of the realized truncation noise eta_m = sum_{n>K} a_n phi_n(x_m)
under three amplitude priors (Gaussian / variance-matched t3 / adjacent-mode rho = 0.3),
standardized per (room, sensor) by the exact std. Rooms: the 187 in-scope rooms of the
main p sweep (K_total > 50), 50 realizations x 8 sensors each, so 74,800 samples per
prior. Note the scope: this runs on the 187 main-regime rooms, not on the 20-room
Appendix F control set, and the printed statistic is the one-sample KS distance to
N(0, 1).

Draws are seeded with zlib.crc32 (_common.stable_seed; --seed S appends S to the key).
The printed value is the median (and maximum) over seeds 0-9 (run_multi_seed.sh).

Output: data/experiments/appendix/prior_tails_qq.npz (keys z_gaussian, z_heavy_tail,
        z_correlated)
Usage:  python scripts/appendix/F_G/prior_tails_qq.py [--seed S --out NAME]
"""
import argparse

import numpy as np
from scipy import stats

from _common import out_path, stable_seed, with_seed
from src.utils.paths import MODAL, EXP, check_paper_dataset

K_USE = 50
M_USE = 8
S_HAT = 1.1266
RHO = 0.3
N_REAL = 50


def in_scope_rooms():
    psw = np.load(EXP / "p_sweep" / "p_sweep_K50_M8.npz", allow_pickle=True)
    rooms = []
    for sid in psw["room_ids"]:
        ev = np.load(MODAL / str(sid) / "eigenpairs.npz", allow_pickle=True)["eigenvalues"]
        if len(ev) > K_USE:
            rooms.append(str(sid))
    return rooms


def part_a(rooms, seed=None):
    rep = ["## Truncation-noise normality under three priors "
           f"({len(rooms)} in-scope rooms x {N_REAL} realizations x {M_USE} sensors)\n",
           "| prior | pooled KS D vs N(0,1) | excess kurtosis | skew | N |",
           "|---|---|---|---|---|"]
    samples = {}
    for prior in ["gaussian", "heavy_tail", "correlated"]:
        pool = []
        for sid in rooms:
            d = MODAL / sid
            ev = np.load(d / "eigenpairs.npz", allow_pickle=True)["eigenvalues"]
            Phi = np.load(d / "measurement_matrix.npy")[:M_USE]
            ev_t = ev[K_USE:]
            Phi_t = Phi[:, K_USE:]
            var = ev_t ** (-S_HAT)
            rng = np.random.default_rng(stable_seed(with_seed((prior, sid), seed)))
            if prior == "correlated":
                Sig = np.diag(var)
                sd = np.sqrt(var)
                for j in range(len(var) - 1):
                    Sig[j, j + 1] = Sig[j + 1, j] = RHO * sd[j] * sd[j + 1]
                L = np.linalg.cholesky(Sig + 1e-14 * np.eye(len(var)))
                a = rng.standard_normal((N_REAL, len(var))) @ L.T
                true_var = np.einsum("mi,ij,mj->m", Phi_t, Sig, Phi_t)
            else:
                if prior == "gaussian":
                    a = rng.standard_normal((N_REAL, len(var))) * np.sqrt(var)
                else:
                    a = rng.standard_t(3, (N_REAL, len(var))) * np.sqrt(var / 3.0)
                true_var = Phi_t ** 2 @ var
            eta = a @ Phi_t.T                       # (N_REAL, M)
            z = eta / np.sqrt(true_var)[None, :]
            pool.append(z.ravel())
        z = np.concatenate(pool)
        D, _ = stats.kstest(z, "norm")
        rep.append(f"| {prior} | {D:.4f} | {stats.kurtosis(z):+.3f} "
                   f"| {stats.skew(z):+.3f} | {len(z)} |")
        samples[prior] = z
    for prior in ["heavy_tail", "correlated"]:
        D2 = stats.ks_2samp(samples[prior], samples["gaussian"]).statistic
        rep.append(f"\ntwo-sample KS {prior} vs gaussian sample: {D2:.4f}")
    return rep, samples


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seed", type=int, default=None,
                    help="extra seed appended to every RNG key (default: the shipped run)")
    ap.add_argument("--out", default=None, help="output name under data/experiments/appendix/")
    args = ap.parse_args()
    check_paper_dataset()
    rooms = in_scope_rooms()
    rep, samples = part_a(rooms, args.seed)
    out = out_path(args.out or "prior_tails_qq.npz")
    np.savez(out, **{f"z_{k}": v for k, v in samples.items()})
    print("\n".join(rep))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
