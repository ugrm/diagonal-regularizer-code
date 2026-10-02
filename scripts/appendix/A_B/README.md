# Appendices A and B: anisotropy and Berry statistics

These scripts recompute Tables T32, T35, T4, T36, T5 and T6 and the numbers in the prose of Appendices A (noise anisotropy, the H + D decomposition, the trace-free identity, the anisotropy sweep) and B (Berry's conjecture: χ²(1) Q-Q statistics, per-room KS, diagnostics). `run_berry_qq.py` also writes the data behind Figure 15.

## Run

From the repository root, CPU only (`make appendix-A-B` runs all of it):

| step | command | output in `data/experiments/` | time |
|---|---|---|---|
| 1 | `python scripts/appendix/A_B/run_berry_qq.py` | `appendix/run_berry_qq.npz` | seconds |
| 2 | `python scripts/appendix/A_B/room_diagnostic_regression.py` | `appendix/room_diagnostic_regression.{npz,md}` | < 1 min |
| 3 | `python scripts/appendix/A_B/hd_decomposition.py` | `appendix/hd_decomposition.{npz,md}` | ~6 min (up to ~50 min on a busy machine) |
| 4 | `python scripts/appendix/A_B/verify_tracefree_identity.py` | `appendix/verify_tracefree_identity.{npz,md}` | ~7 min |
| 5 | `python scripts/appendix/A_B/aniso_sensitivity_sweep.py` | `appendix/aniso_sensitivity_sweep.{npz,md}` | 2–6 min |
| 6 | `python scripts/extract_E_op_per_room.py` | `anisotropy/E_op_empirical_187.npz` (same values as the shipped file) | seconds |
| 7 | `python scripts/appendix/A_B/check_A_B.py` | table on stdout | < 1 min |

Step 3 needs step 2, steps 4 and 5 need step 3, and the checker needs all of them. Set `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` (the Makefile does): multi-threaded BLAS on a loaded machine slows the sweeps several-fold.

## Inputs

- `data/modal`, the paper's dataset (every script checks it first).
- `DR_ROOMS` (default `data/rooms`, from `make rooms`): the full eigenfunctions on the mesh, needed by steps 3 and 4.
- Shipped results: `p_sweep/p_sweep_K50_M8.npz`, `cost/cost_K50_M8.npz`, `noise_profile/noise_profile_K50_M8.npz`, `anisotropy/E_op_empirical_187.npz`, `supplementary/berry_qq_data.npz`.

`scripts/build/room_geometries.py` and `scripts/build/m3_performance.py` rebuild two shipped derived files (`data/modal_summary/room_geometries.npz`, read by `fig_delta_vs_Eop`, and `data/experiments/m3_performance.npz`, Table T2). The rebuilt `m3_performance.npz` matches the shipped one exactly except `P_m3_*`, which differ by < 1e-7.

## Check

`check_A_B.py` prints `table | row | printed | recomputed | status` for 94 values: 94 OK, 0 DIFF (exit code 0). An unexpected DIFF exits with 1. Every script here uses fixed seeds, and its numeric outputs reproduce the shipped results in `data/experiments/appendix/` exactly.

## Notes

- `scripts/05_berry_check.py` also writes a `berry_qq_data.npz`, but overwrites it at each K of its loop and keeps only the rooms with enough modes for that K. Use `run_berry_qq.py` for the Figure 15 data.
- All absolute costs in the anisotropy sweep use synthetic, temporally white noise in place of truncation noise. At T = 1 the difference vanishes.
- T4 reports the largest increases up to the mean anisotropy marker (a ≤ 0.88) and up to the spike level (a ≤ 3.5); medians use the cells with the full 24-room set. The empirical-direction (F1) cells at a ≥ 1.5 keep only 2–6 rooms because of positive definiteness and rise further. The full cell-by-cell table is in `aniso_sensitivity_sweep.md`.
