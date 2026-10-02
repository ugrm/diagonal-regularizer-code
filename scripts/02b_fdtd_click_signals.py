#!/usr/bin/env python
"""Re-simulate the microphone signals of the published dataset (GPU).

The paper's experiments use one FDTD recording per room: the click probe under the first
nuisance setting (eta1). This script follows build_scene_gpu() in scripts/02_run_fdtd.py step
by step with the same random-number stream (world, grid, eta1, eta2, microphones, probe
signals, then the eta1 click source) and stops after that first simulation. Every draw the
click/eta1 recording depends on happens before the other seven simulations of the full run,
so the signal equals the full run's (checked bit-for-bit on validation rooms); it matches the
recordings behind the paper up to GPU floating-point rounding.

Writes data/modal/scene_XXXXX/y_mics_eta1.npy, shape (1, 8, n_t), after checking that the
simulated microphone positions equal the dataset's mic_pos.npy.

Usage: python scripts/02b_fdtd_click_signals.py DATASET_ROOT {val|train|all|ID ...} [--force]
"""
import importlib.util
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
spec = importlib.util.spec_from_file_location("fdtd", os.path.join(REPO, "scripts", "02_run_fdtd.py"))
F = importlib.util.module_from_spec(spec); sys.modules["fdtd"] = F; spec.loader.exec_module(F)
from src.utils.paths import check_paper_dataset


def simulate_click(sid, solver, seed=42, fcut=F.FCUT_DEFAULT, k=F.K_DEFAULT, alpha=F.ALPHA_DEFAULT,
                   c=F.C_DEFAULT, T=0.5, gamma=5.0, pml_cells=20, save_every=100):
    """(mic_pos, y_click) of one room, or None when the scene is invalid (as in 02_run_fdtd.py)."""
    rng = np.random.default_rng(seed + sid)
    w = F.generate_world(rng, n_seg_min=F.N_SEG_MIN, n_seg_max=F.N_SEG_MAX, side_min=F.SIDE_MIN,
                         side_max=F.SIDE_MAX, kappa_min=F.KAPPA_MIN, kappa_max=F.KAPPA_MAX, c=c)
    vertices = np.asarray(w["vertices"]); kappas = w["kappas"]; source_pos = np.asarray(w["source_pos"])
    gm = F.choose_dx_dt(c, fcut, k=k, alpha=alpha)
    base_dx, base_dt = gm["dx"], gm["dt"]
    domain_x = vertices[:, 0].max() + 0.5; domain_y = vertices[:, 1].max() + 0.5
    n_t = F.time_steps(T, base_dt)
    if not F.validate_cfl_strict(c, base_dt, base_dx)[0]:
        return None
    eta1 = F.make_nuisance_config(base_dx, base_dt, c, alpha, domain_x, domain_y, T, (50.0, 150.0), rng)
    eta2 = F.make_nuisance_config(base_dx, base_dt, c, alpha, domain_x, domain_y, T, (50.0, 150.0), rng)
    if eta1.actual_cfl >= F.CFL_MAX or eta2.actual_cfl >= F.CFL_MAX:
        return None
    mic_pos = F.place_mics(vertices, F.N_MICS, rng, source_pos=source_pos)
    _ = {pn: F.make_probe(pn, n_t, base_dt, fcut, rng) for pn in F.PROBE_NAMES}   # same stream as the full run
    e = eta1
    mask = F.rasterise_room(vertices, e.nx, e.ny, e.dx)
    kfield = F.build_kappa_field(vertices, kappas, e.nx, e.ny, e.dx, mask)
    src_idx = (int(np.clip(round(source_pos[0] / e.dx), 0, e.nx - 1)),
               int(np.clip(round(source_pos[1] / e.dx), 0, e.ny - 1)))
    mic_idx = np.round(mic_pos / e.dx).astype(int)
    mic_idx[:, 0] = np.clip(mic_idx[:, 0], 0, e.nx - 1); mic_idx[:, 1] = np.clip(mic_idx[:, 1], 0, e.ny - 1)
    src = F.make_probe(F.PROBE_NAMES[0], e.n_timesteps, e.dt, fcut, rng)               # click
    if abs(e.source_phase) > 1e-10:
        s = np.fft.rfft(src); s *= np.exp(1j * e.source_phase); src = np.fft.irfft(s, n=len(src))
    r = solver.run(nx=e.nx, ny=e.ny, n_steps=e.n_timesteps, dx=e.dx, dt=e.dt, c=c, gamma=gamma + e.gamma_offset,
                   mask=mask, kappa_grid=kfield, source_pos_idx=src_idx, source_signal=src, mic_indices=mic_idx,
                   pml_cells=pml_cells, pml_strength=e.pml_strength, fcut=fcut, save_every=save_every)
    return mic_pos, (r["y_mics"] * 10 ** (e.mic_gain_db / 20.0)).astype(np.float32)


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    root = sys.argv[1]; args = [a for a in sys.argv[2:] if a != "--force"]; force = "--force" in sys.argv
    check_paper_dataset(root)
    sets = {"val": list(range(800, 1000)), "train": list(range(800)), "all": list(range(800, 1000)) + list(range(800))}
    ids = sets[args[0]] if args[0] in sets else [int(a) for a in args]
    solver = F.FDTDSolverGPU(device_id=0)
    t_all = time.time()
    for n, sid in enumerate(ids):
        d = os.path.join(root, f"scene_{sid:05d}")
        out = os.path.join(d, "y_mics_eta1.npy")
        if not os.path.isdir(d) or (os.path.exists(out) and not force):
            continue
        t0 = time.time()
        res = simulate_click(sid, solver)
        if res is None:
            print(f"[{n + 1}/{len(ids)}] scene_{sid:05d} invalid scene, skipped", flush=True); continue
        mic_pos, y = res
        ref = np.load(os.path.join(d, "mic_pos.npy"))
        if ref.shape != mic_pos.shape or not np.array_equal(ref, mic_pos):
            sys.exit(f"scene_{sid:05d}: simulated microphone positions differ from the dataset's")
        np.save(out + ".tmp.npy", y[None]); os.replace(out + ".tmp.npy", out)
        print(f"[{n + 1}/{len(ids)}] scene_{sid:05d} {time.time() - t0:.1f}s  total {(time.time() - t_all) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
