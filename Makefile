PYTHON  ?= python
CONFIG  ?= configs/default.yaml
WORKERS ?= 16
# Release that holds the dataset archive (see README, "Data")
DATA_URL ?= https://github.com/ugrm/diagonal-regularizer-code/releases/download/v1.0
DATA_PARTS := modal_dataset_v1_val modal_dataset_v1_train_a modal_dataset_v1_train_b
# Full eigenfunctions on the mesh, needed by four appendix scripts (make rooms)
export DR_ROOMS ?= data/rooms

CPU   = CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

.PHONY: help download-data figures check check-main check-appendix reproduce-core \
        appendix appendix-A-B appendix-E appendix-F-G appendix-I train train-lir \
        fdtd-signals rooms

help:
	@echo "From the shipped results (minutes, CPU):"
	@echo "  make figures          draw every figure of the paper into figures/"
	@echo "  make check            compare every printed number with the data"
	@echo "From the raw dataset (needs make download-data):"
	@echo "  make reproduce-core   recompute the P(p) sweep (Fig. 2, Table 1) and compare (~1-2 h, $(WORKERS)+ cores)"
	@echo "  make appendix         rerun the appendix experiments (A/B, E, F/G, I) and their checks (CPU, ~2 h)"
	@echo "  make train            M1-M3 sweep and LIR (GPU, ~2-3 days)"
	@echo "Regenerating inputs:"
	@echo "  make fdtd-signals     re-simulate the microphone signals of every room (GPU, ~2 h)"
	@echo "  make rooms            full eigensolve of every room into data/rooms (CPU, ~1 day, 23 GB)"

# ─── Data ────────────────────────────────────────────────────────────────────
download-data:
	@if [ -f data/modal/scene_00806/y_mics_eta1.npy ]; then echo "data/modal already present"; else \
	  curl -fL -o SHA256SUMS $(DATA_URL)/SHA256SUMS && \
	  for p in $(DATA_PARTS); do curl -fL -o $$p.tar.gz $(DATA_URL)/$$p.tar.gz || exit 1; done && \
	  sha256sum -c SHA256SUMS && for p in $(DATA_PARTS); do tar -xzf $$p.tar.gz || exit 1; done && \
	  $(PYTHON) -c "from src.utils.paths import check_paper_dataset; check_paper_dataset(); print('dataset OK')"; fi

# ─── From shipped results ────────────────────────────────────────────────────
figures:
	$(CPU) $(PYTHON) scripts/make_figures.py --output-dir figures/

check: check-main check-appendix

check-main:
	$(PYTHON) tests/test_locked_numbers.py

check-appendix:
	$(CPU) $(PYTHON) scripts/appendix/A_B/check_A_B.py
	$(CPU) $(PYTHON) scripts/appendix/E/check_E.py
	$(CPU) $(PYTHON) scripts/appendix/F_G/check_F_G.py

# ─── From the raw dataset ────────────────────────────────────────────────────
reproduce-core:
	$(CPU) $(PYTHON) scripts/07_compute_p_sweep.py --exp-dir data/experiments_repro --workers $(WORKERS)
	$(PYTHON) scripts/compare_p_sweep.py data/experiments_repro/p_sweep/p_sweep_K50_M8.npz

appendix: appendix-A-B appendix-E appendix-F-G appendix-I

appendix-I:
	$(CPU) $(PYTHON) scripts/appendix/I/aperture_rooms.py

appendix-A-B:
	$(CPU) $(PYTHON) scripts/appendix/A_B/run_berry_qq.py
	$(CPU) $(PYTHON) scripts/appendix/A_B/room_diagnostic_regression.py
	$(CPU) $(PYTHON) scripts/appendix/A_B/hd_decomposition.py
	$(CPU) $(PYTHON) scripts/appendix/A_B/verify_tracefree_identity.py
	$(CPU) $(PYTHON) scripts/appendix/A_B/aniso_sensitivity_sweep.py
	$(CPU) $(PYTHON) scripts/extract_E_op_per_room.py
	$(CPU) $(PYTHON) scripts/appendix/A_B/check_A_B.py

appendix-E:
	$(CPU) $(PYTHON) scripts/appendix/E/run_heat_extended.py --workers $(WORKERS)
	$(CPU) $(PYTHON) scripts/appendix/E/run_heat_2d_sweep.py --part both
	$(CPU) $(PYTHON) scripts/appendix/E/run_heat_c_fit.py --check
	$(CPU) $(PYTHON) scripts/appendix/E/heat_herfindahl.py
	$(CPU) $(PYTHON) scripts/appendix/E/check_E.py --ported

appendix-F-G:
	$(CPU) $(PYTHON) scripts/appendix/F_G/appendixF_inscope.py --workers $(WORKERS)
	$(CPU) $(PYTHON) scripts/appendix/F_G/misspec_priors_m3.py --m3-checkpoint data/experiments/m3_retrain_20261003
	$(CPU) $(PYTHON) scripts/appendix/F_G/misspec_population_main_regime.py --m3-checkpoint data/experiments/m3_retrain_20261003
	$(CPU) $(PYTHON) scripts/appendix/F_G/prior_tails_qq.py
	$(CPU) $(PYTHON) scripts/appendix/F_G/matched_ktotal_rates.py
	$(CPU) $(PYTHON) scripts/appendix/F_G/cohort_flatness.py
	$(CPU) $(PYTHON) scripts/appendix/F_G/d_vs_ktotal_sweep.py
	$(CPU) $(PYTHON) scripts/appendix/F_G/cohort_d2h.py
	$(CPU) $(PYTHON) scripts/appendix/F_G/box3d_flatness.py
	$(CPU) $(PYTHON) scripts/appendix/F_G/t28_t42_summary.py
	$(CPU) $(PYTHON) scripts/rectangular_nnsd.py
	PYTHON=$(PYTHON) bash scripts/appendix/F_G/run_multi_seed.sh
	$(CPU) $(PYTHON) scripts/appendix/F_G/check_F_G.py

# ─── GPU ─────────────────────────────────────────────────────────────────────
train: train-lir
	$(PYTHON) scripts/11_run_sweep.py --config configs/sweep_realdata.yaml --base_config $(CONFIG)
	$(PYTHON) scripts/eval_sweep_P.py --config $(CONFIG) --out data/experiments_repro/sweep_P_eval.npz

train-lir:
	$(PYTHON) scripts/train_lir.py --config $(CONFIG) --sweep --output-dir data/experiments_repro/lir

# ─── Regenerating inputs ─────────────────────────────────────────────────────
fdtd-signals:
	$(PYTHON) scripts/02b_fdtd_click_signals.py data/modal all

rooms:
	$(PYTHON) scripts/01_generate_rooms.py --config $(CONFIG)
