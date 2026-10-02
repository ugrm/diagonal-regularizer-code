#!/usr/bin/env python
"""Herfindahl index of the discarded-mode noise under heat diffusion (T23; T40 rows 12-14).

For the discarded modes n > K = 50 of each diagnostic room, the heat noise weights are
    w_n(t) = lambda_n^{-s} exp(-2 kappa lambda_n t),   s = 1 (design value), kappa = 1,
and H(t) = sum_n w_n^2 / (sum_n w_n)^2 (eq. herfindahl with the decaying weights).
The acoustic reference row uses the time-independent weights lambda_n^{-|s|} with the
population |s| (median of noise_profile_K50_M8.npz, 1.1266) over its 187 in-scope rooms.
Eigenvalues come from the legacy modal dataset (MODAL).

Usage (from the repo root):
    python scripts/appendix/E/heat_herfindahl.py      # prints T23, exits 1 on a mismatch
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heat_common import DIAG_ROOMS, EXP, K_TRUNC, MODAL, check_paper_dataset  # noqa: E402
import numpy as np  # noqa: E402

T_OBS_MS = (12, 250, 493)        # earliest, median and latest observation times of T23

# Printed T23 cells: (median H, H range, 1/H range)
PRINTED = {
    12: ("0.027", ("0.013", "0.052"), ("19", "76")),
    250: ("0.357", ("0.156", "0.815"), ("1.2", "6.4")),
    493: ("0.548", ("0.283", "0.987"), ("1.0", "3.5")),
}
PRINTED_ACOUSTIC = ("0.005", "200")


def herfindahl(w):
    w = np.asarray(w, float)
    return float(np.sum(w**2) / np.sum(w)**2)


def eigenvalues(sid, modal_root=MODAL):
    return np.load(os.path.join(modal_root, sid, "eigenpairs.npz"))["eigenvalues"].astype(np.float64)


def heat_H(ev, t, K=K_TRUNC, s=1.0, kappa=1.0):
    d = ev[K:]
    return herfindahl(d**(-s) * np.exp(-2 * kappa * d * t))


def compute(modal_root=MODAL):
    """H[room, time] for the diagnostic rooms, plus the acoustic population median."""
    H = np.array([[heat_H(eigenvalues(sid, modal_root), t / 1000) for t in T_OBS_MS] for sid in DIAG_ROOMS])
    K_total = np.array([len(eigenvalues(sid, modal_root)) for sid in DIAG_ROOMS])
    prof = np.load(EXP / "noise_profile" / "noise_profile_K50_M8.npz", allow_pickle=True)
    s_abs = abs(float(np.nanmedian(prof["s_per_room"])))
    H_ac = np.array([herfindahl(eigenvalues(str(sid), modal_root)[K_TRUNC:]**(-s_abs)) for sid in prof["room_ids"]])
    return {"rooms": DIAG_ROOMS, "K_total": K_total, "t_ms": T_OBS_MS, "H": H,
            "s_abs": s_abs, "H_acoustic": H_ac, "acoustic_rooms": [str(s) for s in prof["room_ids"]]}


def fmt(v, like):
    """Format v with the number of decimals of the printed string `like`."""
    dec = len(like.split(".")[1]) if "." in like else 0
    return f"{v:.{dec}f}"


def main():
    check_paper_dataset(MODAL)
    r = compute()
    bad = 0
    print(f"{'t_obs':>7} | {'median H':>16} | {'H range':>30} | {'1/H range':>26}")
    for j, t in enumerate(r["t_ms"]):
        Hs = r["H"][:, j]
        pm, (plo, phi), (ilo, ihi) = PRINTED[t]
        got = (fmt(np.median(Hs), pm), (fmt(Hs.min(), plo), fmt(Hs.max(), phi)),
               (fmt((1 / Hs).min(), ilo), fmt((1 / Hs).max(), ihi)))
        ok = got == PRINTED[t]
        bad += not ok
        print(f"{t:>4} ms | {np.median(Hs):.4f} ({pm}) | [{Hs.min():.4f}, {Hs.max():.4f}] ([{plo}, {phi}]) | "
              f"[{(1/Hs).min():.2f}, {(1/Hs).max():.2f}] ([{ilo}, {ihi}]) {'OK' if ok else 'DIFF'}")
    for sid, kt, h in zip(r["rooms"], r["K_total"], r["H"]):
        print(f"   {sid} K_total={kt:4d}  H = " + "  ".join(f"{x:.4f}" for x in h))
    Ha = float(np.median(r["H_acoustic"]))
    ok = fmt(Ha, PRINTED_ACOUSTIC[0]) == PRINTED_ACOUSTIC[0] and f"{1 / Ha:.2g}" == "2e+02"
    bad += not ok
    print(f"acoustic (|s|={r['s_abs']:.4f}, {len(r['H_acoustic'])} rooms): median H {Ha:.4f} (~0.005), "
          f"1/H {1 / Ha:.0f} (~200) {'OK' if ok else 'DIFF'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
