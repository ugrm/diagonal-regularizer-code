# Appendices C and D, main-text prose and the remaining tables

`check_C_D.py` recomputes every number printed in Tables T7–T15, T18, T30, T31, T33, T34 and T37–T39, and the numeric claims in the main-text prose and in the prose of Appendices C (acoustic experiments), D (learning experiments), H and I. Tables T1–T6, T19–T28, T32, T35, T36 and T40–T42 are checked by `tests/test_locked_numbers.py` and the other appendix checkers.

## Run

From the repository root, CPU only, about a minute (`make check` runs it):

    python scripts/appendix/C_D/check_C_D.py

It reads only shipped files: `data/experiments/` (p-sweep, cost, noise profile, s_room comparison, size ablation, LIR, sweep evaluation, `appendix/room_diagnostic_regression.npz`), `data/supplementary/` (geometric predictors, capacity checks), `data/results/` (M2/M3 outputs), `checkpoints/*/trajectory.npz` (M3 seed runs), and `data/modal` (room metadata). `appendix/room_diagnostic_regression.npz` comes from `scripts/appendix/A_B/room_diagnostic_regression.py`.

## Check

Each line is `table | row | printed | recomputed | status`. On the shipped files: 142 values, 124 OK, 16 OK(stored), 2 NOT CHECKED, 0 DIFF (exit code 0).

- **OK(stored)**: the value matches a stored output of trained models whose weights are not shipped (T31 M3 seed exponents; T34 M2/M3 exponents and the sweep-evaluation rows; T38 capacity check). `make train` retrains them; new weights give slightly different values.
- **NOT CHECKED**: T13 and the D.4 room counts come from a parameter-varied dataset (K = 100, M = 16, |s| = 1.29) and models that were not kept.
