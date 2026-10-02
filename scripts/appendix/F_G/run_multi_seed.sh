#!/usr/bin/env bash
# Appendix F/G multi-seed reruns: every value that depends on random draws (T27 flatness,
# T28 3D column, T41 KS gap and single-draw range, T42 flatness/3D rows) over 10 fixed
# seeds; each seed redraws the amplitudes and the sensor layouts. Outputs go to
# data/experiments/appendix/seeds/, then multi_seed_summary.py prints median [min, max].
# Run from the repository root; CPU only. About 80 min on 34 cores (three chains).
set -euo pipefail
PY=${PYTHON:-python}
D=scripts/appendix/F_G
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
export CUDA_VISIBLE_DEVICES=""
mkdir -p data/experiments/appendix/seeds

(for s in $SEEDS; do
   $PY $D/cohort_flatness.py --workers 16 --seed $s --out seeds/results_cohort_flatness_seed$s.npz
 done) > data/experiments/appendix/seeds/log_cohort.txt 2>&1 &
(for s in $SEEDS; do
   $PY $D/box3d_flatness.py --workers 8 --seed $s --out seeds/box3d_flatness_seed$s.npz
 done
 for s in $SEEDS; do
   $PY $D/matched_ktotal_rates.py --workers 10 --seed $s --out seeds/matched_ktotal_seed$s.npz
 done) > data/experiments/appendix/seeds/log_box3d_matched.txt 2>&1 &
(for s in $SEEDS; do
   $PY $D/prior_tails_qq.py --seed $s --out seeds/prior_tails_qq_seed$s.npz
 done
 $PY $D/single_draw_seed_spread.py --workers 16 --n-seeds 10 \
     --out seeds/single_draw_seed_spread_n10.npz) > data/experiments/appendix/seeds/log_f.txt 2>&1 &
wait
$PY $D/multi_seed_summary.py
