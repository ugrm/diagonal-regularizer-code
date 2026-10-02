# Why Learning Rediscovers the Closed-Form Diagonal Regularizer

Code and data for the NeurIPS 2026 paper by Jeahn Han and Pyojin Kim (Gwangju Institute of Science and Technology).

[Project page](https://ugrm.github.io/diagonal-regularizer/) · [arXiv](https://arxiv.org/abs/2609.09656)

Recovering modal amplitudes from a few microphones needs a regularizer, and its shape decides how strongly each mode is shrunk. The paper shows that when the noise from the discarded modes is isotropic, the best diagonal Tikhonov shape is the closed form Γ_k ∝ λ_k^{|s|}, set by the prior alone. On simulated rooms it costs 5.83 % against per-room tuning in the median room, and learned diagonal regularizers end up matching it.

This repository draws every figure, checks every printed number against the data, and reruns the experiments from the raw dataset.

## Install

```bash
conda create -n diagreg python=3.10 && conda activate diagreg
pip install -r requirements.txt        # figures, checks, sweeps, appendix experiments
pip install -r requirements-gpu.txt    # adds the CUDA FDTD simulator (needs an NVIDIA GPU)
```

`pdftoppm` (poppler) is optional; `scripts/make_figures.py --compare` uses it to compare figures pixel by pixel.

## Data

Two parts.

**In this repository (≈ 90 MB).** The results behind the printed numbers and figures: `data/experiments/`, `data/supplementary/`, `data/results/`, and the training curves of the M3 runs (`checkpoints/*/trajectory.npz`). `make figures` and `make check` need nothing else.

**The dataset (GitHub Release `v1.0`: three archives `modal_dataset_v1_{val,train_a,train_b}.tar.gz` with `SHA256SUMS`).** 1,000 simulated rooms (random convex polygons, 800 for training and 200 for validation; rooms 00905, 00913 and 00921 are degenerate and excluded). Each `data/modal/scene_XXXXX/` holds:

| file | content |
|---|---|
| `eigenpairs.npz` | Robin-Laplacian eigenvalues and frequencies below 1 kHz, room polygon |
| `measurement_matrix.npy` | eigenfunctions at the 8 microphones (Φ) |
| `modal_trajectories.npz` | modal amplitudes a_k(t) of the FDTD field, room damping |
| `mic_pos.npy`, `metadata.json` | microphone positions; mode count, area, time steps |
| `y_mics_eta1.npy` | FDTD microphone signals (click source) |

```bash
make download-data     # fetches, checks the SHA-256 and unpacks into data/modal/
```

Every script that reads `data/modal` first checks that it holds this dataset and refuses to run otherwise.

The microphone signals in the archive were simulated with `scripts/02b_fdtd_click_signals.py`, which runs the same solver, seed and nuisance draws as `scripts/02_run_fdtd.py` and keeps the one recording the experiments use. They agree with the recordings behind the paper up to GPU floating-point rounding: recomputing the P(p) sweep from them gives the per-room optimal exponent unchanged in 98.6 % of (room, window) cells and Table 1's tiers as 0.61 / 1.44 / 5.85 % against the printed 0.61 / 1.46 / 5.83 %.

Four appendix scripts also need the full eigenfunctions on the mesh (`make rooms`, about a day on CPU, 23 GB; set `DR_ROOMS` if you keep them elsewhere).

## Reproducing the paper

| goal | command | time | compares against |
|---|---|---|---|
| every figure | `make figures` | minutes | `--compare DIR` diffs with reference PDFs |
| every printed number | `make check` | 1–2 min | printed values, row by row |
| P(p) sweep, Fig. 2, Table 1 | `make reproduce-core` | 1–2 h on 16+ cores | the shipped sweep |
| Appendices A, B, E, F, G, I | `make appendix` | about 2 h (CPU) | each appendix's tables |
| learned models (§6, App. D) | `make train` | 2–3 days (GPU) | `sweep_P_eval.npz`, LIR results |
| microphone signals | `make fdtd-signals` | about 2 h (GPU) | the archive's signals |

`make check` runs `tests/test_locked_numbers.py` (main text) and one checker per appendix group (`scripts/appendix/{A_B,E,F_G}/check_*.py`). Each prints `table | row | printed | recomputed | status`. Each appendix folder has a `README.md` with the script order, run times and what each script reproduces.

Outputs of the long reruns go to `data/experiments_repro/` or `data/experiments/appendix/`, so the shipped results stay untouched for comparison.

### What a rerun cannot reproduce bit for bit

- **Trained models.** The M1–M3 weights are not included. `make train` retrains them; new weights give slightly different numbers than the shipped evaluation (`data/experiments/sweep_P_eval.npz`), which comes from the original runs.
- **Ill-conditioned cells.** At the two smallest ridge weights and T ≥ 1000 the heat sweep depends on BLAS rounding; no printed number uses these cells.

## Layout

```
configs/         dataset, grid and training settings
scripts/         pipeline: 01 rooms, 02 FDTD, 02b microphone signals, 05–09 analysis, 10–13 learning
scripts/appendix appendix experiments, one folder per appendix group, each with a checker
scripts/build    small builders for derived files
src/             library: geometry, physics, simulation, estimation, analysis, models, figures
data/            shipped results (the dataset unpacks into data/modal)
tests/           main-text number check
```

## Citation

```bibtex
@inproceedings{han2026learning,
  title     = {Why Learning Rediscovers the Closed-Form Diagonal Regularizer},
  author    = {Han, Jeahn and Kim, Pyojin},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## License

MIT (see `LICENSE`).
