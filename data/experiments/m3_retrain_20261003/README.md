# M3 retrain (n = 800, T in {1, 100}, seeds 42-46)

Retrained M3 weights for the Appendix F values that need trained M3 models (T26 M3 column; T41 "largest single-room movement" and "no-truncation control" rows). Trained on the paper's dataset (`data/modal`, guard `check_paper_dataset()` passed) with the repository defaults (500 epochs, 8000 samples, lr 1e-3, batch 32, K = 50, M = 8):

    python scripts/10_train_models.py --model m3_hypernet --t_value T --seed S \
        --checkpoint_dir data/experiments/m3_retrain_20261003/m3_hypernet_n800_T{T}_s{S}

Appendix F outputs computed from these weights (`appendix/`):

    python scripts/appendix/F_G/misspec_population_main_regime.py --m3-checkpoint data/experiments/m3_retrain_20261003
    python scripts/appendix/F_G/misspec_priors_m3.py --m3-checkpoint data/experiments/m3_retrain_20261003

(run with the output folder redirected to `data/experiments/m3_retrain_20261003/appendix/`). The same two files are shipped as `data/experiments/appendix/misspec_population_187.npz` and `misspec_priors.npz`, which `scripts/appendix/F_G/check_F_G.py` reads for Table T26 and the M3 rows of Table T41. Final training losses agree with the shipped `checkpoints/*/trajectory.npz` to four significant digits; the learned exponents differ per seed, as expected from the flat landscape.
