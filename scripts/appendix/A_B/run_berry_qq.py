#!/usr/bin/env python
"""
Berry Q-Q data: test |C_kn|^2 ~ chi^2(1) for all 197 validation rooms
(Appendix B, Table T36, Figure 15).

Under Berry's conjecture, for L2-normalized eigenmodes (integral |phi|^2 = 1):
  phi_k(x) ~ N(0, 1/|Omega|) at a random point x
Therefore:
  |Omega| * |phi_k(x_n)|^2 ~ chi^2(1)   (squared standard normal)

|C_kn|^2 = |Omega| * |phi_k(x_n)|^2 is computed from measurement_matrix.npy
(retained modes k <= min(K_total, 50), all M = 8 microphones) and tested against chi^2(1).

Inputs:
  - modal data from src.utils.paths.MODAL (paper dataset, checked);
  - room list from data/experiments/p_sweep/p_sweep_K50_M8.npz (the 197 validation rooms).
The paper's Figure 15 is drawn by src/visualization/fig_berry_qq.py from this npz
(shipped as data/supplementary/berry_qq_data.npz).

Output: data/experiments/appendix/run_berry_qq.npz (same keys as data/supplementary/berry_qq_data.npz)
Usage:  python scripts/appendix/A_B/run_berry_qq.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from src.utils.paths import EXP, MODAL, check_paper_dataset  # noqa: E402

MODAL_DIR = str(MODAL)
OUT_DIR = os.path.join(str(EXP), "appendix")
OUT_PATH = os.path.join(OUT_DIR, "run_berry_qq.npz")

M = 8
K_TRUNC = 50


def compute_Ckn_for_room(room_id):
    """Compute area-normalized |C_kn|^2 for one room."""
    meas_path = os.path.join(MODAL_DIR, room_id, "measurement_matrix.npy")
    meta_path = os.path.join(MODAL_DIR, room_id, "metadata.json")

    if not os.path.exists(meas_path):
        return None, None, None

    with open(meta_path) as f:
        meta = json.load(f)

    Phi = np.load(meas_path)  # (M, K_total)
    area = meta['room_area_m2']
    n_seg = meta['n_segments']
    K_total = meta['K']
    K = min(Phi.shape[1], K_TRUNC)

    # Berry: phi_k(x) ~ N(0, 1/|Omega|), so |Omega|*|phi_k(x)|^2 ~ chi2(1)
    Csq = area * Phi[:, :K]**2  # (M, K)

    return Csq, n_seg, K_total


def main():
    check_paper_dataset(MODAL)
    psw = np.load(os.path.join(str(EXP), "p_sweep", "p_sweep_K50_M8.npz"), allow_pickle=True)
    VAL_ROOMS = list(psw['room_ids'])
    print(f"Computing |C_kn|^2 for {len(VAL_ROOMS)} validation rooms...")

    all_Csq = []
    per_room = []

    for room_id in VAL_ROOMS:
        Csq, n_seg, K_total = compute_Ckn_for_room(room_id)
        if Csq is None:
            continue

        flat = Csq.flatten()
        all_Csq.extend(flat.tolist())

        # KS test against chi2(1)
        ks_stat, ks_pval = stats.kstest(flat, 'chi2', args=(1,))

        per_room.append({
            'room_id': room_id,
            'n_seg': n_seg,
            'K_total': K_total,
            'mean_Csq': float(Csq.mean()),
            'median_Csq': float(np.median(flat)),
            'ks_stat': ks_stat,
            'ks_pval': ks_pval,
        })

    all_Csq = np.array(all_Csq)
    chi2_med = stats.chi2.ppf(0.5, df=1)
    print(f"Total |C_kn|^2 values: {len(all_Csq)}")
    print(f"Overall mean: {all_Csq.mean():.4f} (chi2(1) mean = 1.0)")
    print(f"Overall median: {np.median(all_Csq):.4f} (chi2(1) median = {chi2_med:.4f})")
    print(f"Overall var: {all_Csq.var():.4f} (chi2(1) var = 2.0)")

    # Global KS
    ks_global, pval_global = stats.kstest(all_Csq, 'chi2', args=(1,))
    print(f"Global KS vs chi2(1): D={ks_global:.4f}, p={pval_global:.2e}")

    # Save data
    os.makedirs(OUT_DIR, exist_ok=True)
    room_ids = [r['room_id'] for r in per_room]
    np.savez(OUT_PATH,
             all_Csq=all_Csq,
             room_ids=np.array(room_ids),
             n_seg=np.array([r['n_seg'] for r in per_room]),
             K_total=np.array([r['K_total'] for r in per_room]),
             mean_Csq=np.array([r['mean_Csq'] for r in per_room]),
             ks_stat=np.array([r['ks_stat'] for r in per_room]),
             ks_pval=np.array([r['ks_pval'] for r in per_room]),
             ref_distribution='chi2_df1',
             M=M, K_trunc=K_TRUNC)
    print(f"Saved: {OUT_PATH}")

    # Summary table
    print(f"\n{'='*60}")
    print("BERRY Q-Q SUMMARY (vs chi2(1)) BY GEOMETRY CLASS")
    print(f"{'='*60}")
    print(f"{'seg':>4s}  {'N':>3s}  {'mean|C|²':>9s}  {'med KS':>7s}  "
          f"{'% reject':>8s}")

    for n_seg in sorted(set(r['n_seg'] for r in per_room)):
        group = [r for r in per_room if r['n_seg'] == n_seg]
        ks_vals = [r['ks_stat'] for r in group]
        means = [r['mean_Csq'] for r in group]
        reject_frac = sum(1 for r in group if r['ks_pval'] < 0.05) / len(group) * 100
        print(f"{n_seg:4d}  {len(group):3d}  {np.median(means):9.3f}  "
              f"{np.median(ks_vals):7.3f}  {reject_frac:7.1f}%")

    overall_ks_vals = [r['ks_stat'] for r in per_room]
    overall_reject = sum(1 for r in per_room if r['ks_pval'] < 0.05) / len(per_room) * 100
    print(f"{'ALL':>4s}  {len(per_room):3d}  "
          f"{np.median([r['mean_Csq'] for r in per_room]):9.3f}  "
          f"{np.median(overall_ks_vals):7.3f}  {overall_reject:7.1f}%")


if __name__ == "__main__":
    main()
