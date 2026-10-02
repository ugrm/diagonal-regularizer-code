# Appendices F and G: prior robustness, rectangles and 3D boxes

These scripts recompute the tables of Appendix F (prior robustness: T24, T26, T41) and Appendix G (rectangles and 3D boxes: T27, T28, T42), and the numbers in the prose of both appendices.

## Run

From the repository root, CPU only (`make appendix-F-G` runs all of it). Every script writes to `data/experiments/appendix/`.

| script | output | feeds | time |
|---|---|---|---|
| `appendixF_inscope.py [--workers 16]` | `appendixF_inscope.npz` | T24; T41 rows 1 and 3 | ~13 min (16 workers) |
| `misspec_priors_m3.py [--m3-checkpoint DIR]` | `misspec_priors.npz` | Appendix F no-truncation control; T41 "2.8 pp" (M3 only) | ~18 min (16 workers) |
| `misspec_population_main_regime.py [--m3-checkpoint DIR]` | `misspec_population_187.npz` | T26; T41 "13.3 pp" (M3 only) | ~1 min without M3 |
| `single_draw_seed_spread.py [--n-seeds N]` | `single_draw_seed_spread.npz` | single-realization spread of the no-truncation control (reported by `multi_seed_summary.py`; no printed value) | < 1 min |
| `prior_tails_qq.py [--seed S]` | `prior_tails_qq.npz` | T41 KS gap (printed over 10 seeds) | seconds |
| `matched_ktotal_rates.py [--seed S]` | `matched_ktotal.npz` | T27 rectangle flatness; T42 (printed over 10 seeds) | ~1.5 min (10 workers) |
| `cohort_flatness.py [--seed S]` | `results_cohort_flatness.npz` | T27 generic flatness; T42 (printed over 10 seeds) | ~8 min (16 workers) |
| `d_vs_ktotal_sweep.py` | `results_d_vs_ktotal.npz` | T27 rectangle D/(2H); T42; G prose | < 1 min |
| `cohort_d2h.py` | `cohort_d2h.npz` | T27 H and generic D/(2H); T42 | ~1.5 min (16 workers) |
| `box3d_flatness.py [--seed S]` | `box3d_flatness.npz` | T28 3D column; T42 3D rows (printed over 10 sensor layouts) | ~5 min (8 workers) |
| `t28_t42_summary.py` | `t28_t42_summary.npz` | T28; T42 2D maxima and 3D rows | seconds |
| `check_F_G.py [--data-dir DIR]` | stdout | every number above and the F/G prose | seconds |
| `run_multi_seed.sh` | `seeds/*_seed{0..9}.npz` | the random-draw scripts above over 10 fixed seeds (`--seed S`), sensor layouts redrawn per seed | ~80 min (34 cores) |
| `multi_seed_summary.py` | `seeds/multi_seed_summary.npz` | median [min, max] over the 10 seeds of every draw-dependent value in T27, T28, T41, T42 and the G.5 prose | seconds |

The times were measured on a busy shared machine. `t28_t42_summary.py` runs after `box3d_flatness.py`, `run_multi_seed.sh` (which ends with `multi_seed_summary.py`) after the single-seed scripts, and `check_F_G.py` runs last. Without `--seed` every script reproduces its shipped single-seed output. The Makefile target also runs `scripts/rectangular_nnsd.py`, which writes the shipped `data/experiments/rectangular_*.npz` in place. `_common.py` holds the shared helpers: the repository root on `sys.path`, the output folder, the worker count and `stable_seed`.

## Inputs

- `data/modal`, the paper's dataset (checked first).
- `DR_ROOMS` (default `data/rooms`, from `make rooms`), the full eigenfunctions. Only `d_vs_ktotal_sweep.py` and `cohort_d2h.py` need it. They use only eigenvalue prefixes that agree with the modal dataset to within 1.6e-12, checked per room.
- Shipped results: `cost/cost_K50_M8.npz`, `p_sweep/p_sweep_K50_M8.npz`, `anisotropy/E_op_empirical_187.npz`, `rectangular_*.npz`.
- `hd_decomposition.npz` from `scripts/appendix/A_B/` (optional; T42's 2.24 / 2.15).

## Check

`check_F_G.py` prints `table | row | printed | recomputed | status | note`. On the shipped outputs it gives 154 OK, 0 DIFF, 0 UNVERIFIABLE, with exit code 0. An unexpected DIFF is marked `!` and exits with 1.

## Notes

**Random draws.** Several scripts draw random amplitudes or sensor positions. Every printed value that depends on them (T27 flatness, the T28 3D column, the T41 KS gap, the T42 flatness and 3D rows, and the G.5 comparisons) is the median [min, max] over 10 fixed seeds (`run_multi_seed.sh`, seeds 0–9; each seed also redraws the rectangle and 3D-box sensor layouts). Seeds are derived from `zlib.crc32` of a string key (`_common.stable_seed`, `with_seed`), so every run reproduces the shipped `seeds/*.npz` exactly. `check_F_G.py` recomputes the per-seed statistics from those files through `multi_seed_summary.compute`. K_total, H and D/(2H) do not depend on the draws.

**M3 weights.** T26's M3 column and the M3 rows of T41 use M3 retrained with the protocol of Section 6 (n = 800, T ∈ {1, 100}, seeds 42–46; `data/experiments/m3_retrain_20261003/`, see its README). `make appendix-F-G` passes these weights to `misspec_priors_m3.py` and `misspec_population_main_regime.py` with `--m3-checkpoint`. Final training losses agree with the shipped training curves to four significant digits; the per-seed learned exponents differ, as expected from the flat landscape. The main-text M3 numbers (Table 2) come from the original runs' stored evaluation.

Also:

- **Rectangle K_total.** 287 / 255 / 255 / 239 (and T42's 237 discarded modes) are the one-term Weyl estimate ceil(A·200/(4π)) used as the rectangles' mode budget. The exact Dirichlet counts below 1000 Hz are 455 / 397 / 402 / 375; the checker prints them for reference.
- **Generic NNSD.** 48 of the 50 rooms tried contribute; rooms with fewer than 20 eigenvalues are skipped.
- **T41 "about 6 %".** The values are +6.57 % (T = 1) and +6.06 % (T = 100). The checker accepts |x − 6| ≤ 1.
- In `misspec_population_main_regime.py` a few curve entries depend on BLAS threading at the 1e-6 level (cancellation at near-singular α). No printed value moves.
