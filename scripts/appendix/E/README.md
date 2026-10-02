# Appendix E: heat diffusion

These scripts recompute Tables T19 and T21–T23 and T40, the numbers in the prose of Appendix E and the heat statements of Section 7, and the data behind the heat panels of Figures 3, 13 and 14. The heat model is analytic (`src.physics.temporal.generate_heat_signals`): a_k(t) = a_k(0) e^{−λ_k t}, a_k(0) ~ N(0, 1/(1+λ_k)), κ = 1, plus 1 % white measurement noise. It needs no microphone signals.

## Run

From the repository root, CPU only (`make appendix-E` runs all of it):

| script | what it computes | output in `data/experiments/appendix/` | time |
|---|---|---|---|
| `run_heat_extended.py [--workers 16]` | heat P(p) sweep, p ∈ [0, 6], 197 rooms, M = 8 and 4, 10 windows (Fig. 3 c–e, T40) | `exp2_p_sweep_heat_extended.npz` | ~12 min with 16 workers (2.6 CPU-h) |
| `run_heat_2d_sweep.py --part both` | two-parameter sweep Γ_k = λ_k^p e^{cλ_k} for heat and acoustics (Fig. 13, T21) | `heat_2d_sweep_results.npz` | heat ~4 min, acoustic ~6 min (15 workers) |
| `run_heat_c_fit.py --check` | continuous spectral fit of the heat rate c (T19, T22, Fig. 14); `--check` compares with the shipped file | `heat_c_continuous_fit.npz` | seconds |
| `heat_herfindahl.py` | Herfindahl index of the heat tail (T23, T40 rows 12–14) | stdout | seconds |
| `check_E.py [--ported]` | every printed number of T19–T23, T40, App. E and Sec. 7 | stdout | < 1 min |

The first four scripts are independent. `check_E.py` reads the shipped files by default; with `--ported` it reads the three outputs above instead. `heat_common.py` holds the shared constants, the seeds (42 + 10·room + 1), a room loader that needs no microphone signals, and the window selection.

`run_heat_2d_sweep.py --part heat` or `--part acoustic` computes one half. The other half is carried over from an existing output file (NaN if there is none).

Figures 10, 11 and 17 come from `src/visualization/fig_heat_diagnostics.py` and `fig_heat_fit_residuals.py` (room scene_00840, observation times 50 / 250 / 495 ms).

## Inputs

- `data/modal`, the paper's dataset (eigenvalues, measurement matrices, trajectories; checked first).
- The microphone signals `data/modal/scene_XXXXX/y_mics_eta1.npy`, for the acoustic half of the 2-D sweep only (`--fdtd-root` points elsewhere).
- `data/mic_subsets.npz` (the 20 canonical 4-microphone subsets), and the shipped `p_sweep/p_sweep_K50_M8.npz` and `noise_profile/noise_profile_K50_M8.npz` for the acoustic comparisons.

## Check

On the shipped files and with `--ported`: 132 values, 130 OK, 2 NOTE, 0 DIFF (exit code 0). A NOTE marks a printed statement that holds only with a qualifier:

- T40 row 9: heat p* for T ≤ 200 is 2.5–2.8 in 6 of 7 windows, but T = 20 gives 1.9.
- Section 7, "close fit at all but the earliest times": single-seed R² stays ≤ 0.83 in room 00826.

## Known limits

- **Rounding in the heat sweep.** At T ≥ 1000 the heat temporal matrix is extremely ill-conditioned: its columns grow as e^{λ T dt}. At the two smallest ridge weights (1e-6, 1e-4) the solve is dominated by BLAS rounding, so these cells change with the thread count and cannot be reproduced bit for bit. No printed number uses them: room medians agree with the shipped file to 2e-5, p* is identical at every T, and Fig. 3 is pixel-identical.
- **Acoustic half of the 2-D sweep.** It uses the released microphone signals, which match the recordings behind the paper up to GPU floating-point rounding (median relative difference in P 6e-5, max 4e-2). With both halves rerun, the Fig. 13 acoustic marker at T = 500 moves from p = 1.8 to p = 1.2 (still c = 0). This is the only row of `check_E.py` that changes with `--ported`.
- **Padding.** The 2-D sweep zero-pads every room to K = 50 modes before drawing the heat signals, as the shipped results do. Padded modes are unrecoverable constants, so the 9 rooms with K_total < 50 have P ≫ 1 (e.g. 00800, P ≈ 1157). This raises the 2-D-sweep medians above the 1-D sweep.
- **No truncation noise in the heat data.** The heat signals contain only the K retained modes plus sensor noise; the Herfindahl numbers (T23, T40 rows 12–14) are analytic statements about the discarded modes a heat measurement would add.

`check_E.py` prints the full list with the recomputed values.
