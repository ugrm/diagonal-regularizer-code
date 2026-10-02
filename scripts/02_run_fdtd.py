#!/usr/bin/env python3
"""
02_run_fdtd.py - GPU-Accelerated FDTD Dataset Generator

Uses Numba CUDA for 10-50x speedup over CPU. Single GPU processes scenes
sequentially but each scene is much faster.

Expected performance on RTX 4090:
- ~1-2 min per scene (vs ~20 min on CPU)
- 2000 scenes in ~3-6 hours (vs ~17 hours with 40 CPU workers)

Usage:
    python scripts/02_run_fdtd.py \\
        --out_dir data/fdtd \\
        --n_scenes 2000 \\
        --T 0.5 --gamma 5.0 \\
        --device 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

# Ensure imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulation.grid_rules import choose_dx_dt, grid_size_for_domain, time_steps, next_fft_friendly
from src.simulation.world_gen import generate_world, place_mics
from src.simulation.probes import make_probe
from src.simulation.boundary_robin import rasterise_room, build_kappa_field
from src.simulation.fdtd_gpu import FDTDSolverGPU


# =============================================================================
# CONSTANTS
# =============================================================================

CFL_MAX = 1.0 / np.sqrt(2)  # 0.707
OUTPUT_NX = 540
OUTPUT_NY = 540
INTERP_ORDER = 3
INTERP_METHOD = "scipy.ndimage.zoom"
C_DEFAULT = 343.0
FCUT_DEFAULT = 2000.0
K_DEFAULT = 10
ALPHA_DEFAULT = 0.5
KAPPA_MIN = 0.1
KAPPA_MAX = 10.0
N_SEG_MIN = 3
N_SEG_MAX = 10
SIDE_MIN = 2.0
SIDE_MAX = 8.0
N_MICS = 8
PROBE_NAMES = ["click", "logchirp", "mls", "multitone"]


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class WorldConfig:
    vertices: List[List[float]]
    kappas: List[float]
    source_pos: List[float]
    n_segments: int
    room_area: float
    c: float = C_DEFAULT
    kappa_per_segment: List[float] = field(default_factory=list)
    source_position: List[float] = field(default_factory=list)
    segments: List[List[int]] = field(default_factory=list)

    def __post_init__(self):
        if not self.kappa_per_segment:
            self.kappa_per_segment = self.kappas
        if not self.source_position:
            self.source_position = self.source_pos
        if not self.segments:
            n = len(self.vertices)
            self.segments = [[i, (i + 1) % n] for i in range(n)]


@dataclass
class GridConfig:
    dx: float
    dt: float
    nx: int
    ny: int
    fcut: float
    cfl_target: float
    actual_cfl: float
    k: int
    c: float
    lambda_min: float
    n_timesteps: int
    T_sim: float
    pml_thickness: int
    output_nx: int = OUTPUT_NX
    output_ny: int = OUTPUT_NY


@dataclass
class NuisanceConfig:
    dx_factor: float
    dx: float
    dt: float
    pml_strength: float
    gamma_offset: float
    mic_gain_db: float
    source_phase: float
    nx: int
    ny: int
    n_timesteps: int
    actual_cfl: float


@dataclass
class ProbeConfig:
    probe_names: List[str]
    dt: float
    fcut: float
    n_timesteps: int


# =============================================================================
# Utility Functions
# =============================================================================

def convert_numpy_types(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj


def validate_cfl_strict(c: float, dt: float, dx: float) -> Tuple[bool, float]:
    actual_cfl = c * dt / dx
    return actual_cfl < CFL_MAX, actual_cfl


def interpolate_to_output_grid(
    field: np.ndarray,
    sim_nx: int, sim_ny: int,
    target_nx: int = OUTPUT_NX,
    target_ny: int = OUTPUT_NY,
) -> np.ndarray:
    n_snaps = field.shape[0]
    zoom_x = target_nx / sim_nx
    zoom_y = target_ny / sim_ny
    output = np.zeros((n_snaps, target_nx, target_ny), dtype=np.float32)
    for i in range(n_snaps):
        output[i] = zoom(field[i], (zoom_x, zoom_y), order=INTERP_ORDER).astype(np.float32)
    return output


def make_nuisance_config(
    base_dx: float, base_dt: float, c: float, cfl_alpha: float,
    domain_x: float, domain_y: float, T: float,
    pml_range: Tuple[float, float], rng: np.random.Generator,
) -> NuisanceConfig:
    dx_factor = rng.uniform(0.94, 1.00)  # Range used for the paper dataset
    dx = base_dx * dx_factor
    dt = cfl_alpha * dx / c

    is_valid, actual_cfl = validate_cfl_strict(c, dt, dx)
    if not is_valid:
        dt = 0.99 * CFL_MAX * dx / c
        _, actual_cfl = validate_cfl_strict(c, dt, dx)

    nx = next_fft_friendly(grid_size_for_domain(domain_x, dx))
    ny = next_fft_friendly(grid_size_for_domain(domain_y, dx))
    n_t = time_steps(T, dt)

    return NuisanceConfig(
        dx_factor=float(dx_factor),
        dx=float(dx),
        dt=float(dt),
        pml_strength=float(rng.uniform(*pml_range)),
        gamma_offset=float(rng.uniform(-1.0, 1.0)),
        mic_gain_db=float(rng.uniform(-0.5, 0.5)),
        source_phase=float(rng.uniform(0, 2 * np.pi)),
        nx=nx, ny=ny, n_timesteps=n_t,
        actual_cfl=float(actual_cfl),
    )


def validate_energy_decay(energy: np.ndarray) -> Dict[str, Any]:
    result = {"pass": True, "warnings": [], "details": {}}
    e_pos = energy[energy > 0]
    if len(e_pos) == 0:
        result["pass"] = False
        result["warnings"].append("All energy values are zero or negative")
        return result

    peak = float(energy.max())
    final = float(e_pos[-1])
    result["details"]["peak_energy"] = peak
    result["details"]["final_energy"] = final

    if final >= peak and peak > 0:
        result["pass"] = False
        result["warnings"].append(f"Final energy >= peak energy")

    if peak > 0:
        decay_ratio = final / peak
        result["details"]["decay_ratio"] = decay_ratio
        if decay_ratio >= 0.5:
            result["warnings"].append(f"Insufficient decay: {decay_ratio:.2f}")

    n = len(energy)
    last_20 = energy[int(n * 0.8):]
    if len(last_20) > 0 and np.any(last_20 > 0):
        min_last_20 = float(last_20[last_20 > 0].min())
        final_vs_min = final / (min_last_20 + 1e-10)
        result["details"]["late_stage_ratio"] = final_vs_min
        if final_vs_min > 2.0:
            result["pass"] = False
            result["warnings"].append(f"Late-stage energy growth: {final_vs_min:.2f}")

    return result


# =============================================================================
# Scene Builder (GPU)
# =============================================================================

def build_scene_gpu(
    scene_id: int,
    out_dir: str,
    solver: FDTDSolverGPU,
    fcut: float, k: int, alpha: float, c: float,
    T: float, gamma: float, pml_cells: int, save_every: int,
    seed: int,
) -> Dict[str, Any]:
    """Build a single scene using GPU solver."""
    rng = np.random.default_rng(seed + scene_id)
    scene_dir = os.path.join(out_dir, f"scene_{scene_id:05d}")
    os.makedirs(scene_dir, exist_ok=True)

    validation_results = {"scene_id": f"scene_{scene_id:05d}", "passed": True, "issues": []}

    # 1. Generate World
    world_raw = generate_world(
        rng, n_seg_min=N_SEG_MIN, n_seg_max=N_SEG_MAX,
        side_min=SIDE_MIN, side_max=SIDE_MAX,
        kappa_min=KAPPA_MIN, kappa_max=KAPPA_MAX, c=c,
    )

    vertices = np.asarray(world_raw["vertices"])
    kappas = world_raw["kappas"]
    source_pos = np.asarray(world_raw["source_pos"])

    world = WorldConfig(
        vertices=vertices.tolist(), kappas=kappas,
        source_pos=source_pos.tolist(),
        n_segments=int(world_raw["n_segments"]),
        room_area=world_raw["room_area"], c=c,
    )
    with open(os.path.join(scene_dir, "world.json"), "w") as f:
        json.dump(convert_numpy_types(asdict(world)), f, indent=2)

    # 2. Grid Parameters
    grid_meta = choose_dx_dt(c, fcut, k=k, alpha=alpha)
    base_dx, base_dt = grid_meta["dx"], grid_meta["dt"]

    domain_x = vertices[:, 0].max() + 0.5
    domain_y = vertices[:, 1].max() + 0.5

    nx = next_fft_friendly(grid_size_for_domain(domain_x, base_dx))
    ny = next_fft_friendly(grid_size_for_domain(domain_y, base_dx))
    n_t = time_steps(T, base_dt)

    is_valid, actual_cfl = validate_cfl_strict(c, base_dt, base_dx)
    if not is_valid:
        validation_results["passed"] = False
        validation_results["issues"].append(f"CFL violation: {actual_cfl:.4f}")
        return validation_results

    # 3. Nuisance Pair
    eta1 = make_nuisance_config(base_dx, base_dt, c, alpha, domain_x, domain_y, T, (50.0, 150.0), rng)
    eta2 = make_nuisance_config(base_dx, base_dt, c, alpha, domain_x, domain_y, T, (50.0, 150.0), rng)

    for label, eta in [("eta1", eta1), ("eta2", eta2)]:
        if eta.actual_cfl >= CFL_MAX:
            validation_results["passed"] = False
            validation_results["issues"].append(f"{label} CFL: {eta.actual_cfl:.4f}")
            return validation_results

    # 4. Microphones (with distance constraints per spec)
    # - Mics >= 0.3m from walls
    # - Mics >= 0.5m from source
    mic_pos = place_mics(vertices, N_MICS, rng, source_pos=source_pos)
    np.save(os.path.join(scene_dir, "mic_pos.npy"), mic_pos)

    # 5. Probes
    probe_signals = {pn: make_probe(pn, n_t, base_dt, fcut, rng) for pn in PROBE_NAMES}
    np.savez_compressed(os.path.join(scene_dir, "probes.npz"), **probe_signals)

    probe_config = ProbeConfig(probe_names=PROBE_NAMES, dt=base_dt, fcut=fcut, n_timesteps=n_t)
    with open(os.path.join(scene_dir, "probes.json"), "w") as f:
        json.dump(asdict(probe_config), f, indent=2)

    # 6. Simulate (GPU)
    for eta_label, eta in [("eta1", eta1), ("eta2", eta2)]:
        edx, edt = eta.dx, eta.dt
        enx, eny, en_t = eta.nx, eta.ny, eta.n_timesteps

        mask = rasterise_room(vertices, enx, eny, edx)
        kappa_field = build_kappa_field(vertices, kappas, enx, eny, edx, mask)

        src_idx = (
            int(np.clip(round(source_pos[0] / edx), 0, enx - 1)),
            int(np.clip(round(source_pos[1] / edx), 0, eny - 1)),
        )
        mic_idx = np.round(mic_pos / edx).astype(int)
        mic_idx[:, 0] = np.clip(mic_idx[:, 0], 0, enx - 1)
        mic_idx[:, 1] = np.clip(mic_idx[:, 1], 0, eny - 1)

        gain = 10 ** (eta.mic_gain_db / 20.0)

        all_y_mics = []
        all_snapshots = []
        all_energy = []
        save_times = None
        energy_every = None

        for pi, probe_name in enumerate(PROBE_NAMES):
            src_sig = make_probe(probe_name, en_t, edt, fcut, rng)

            if abs(eta.source_phase) > 1e-10:
                src_fft = np.fft.rfft(src_sig)
                src_fft *= np.exp(1j * eta.source_phase)
                src_sig = np.fft.irfft(src_fft, n=len(src_sig))

            result = solver.run(
                nx=enx, ny=eny, n_steps=en_t,
                dx=edx, dt=edt, c=c,
                gamma=gamma + eta.gamma_offset,
                mask=mask, kappa_grid=kappa_field,
                source_pos_idx=src_idx,
                source_signal=src_sig,
                mic_indices=mic_idx,
                pml_cells=pml_cells,
                pml_strength=eta.pml_strength,
                fcut=fcut,
                save_every=save_every,
            )

            all_y_mics.append(result["y_mics"] * gain)
            all_snapshots.append(result["u_snapshots"])
            all_energy.append(result["energy"])

            if save_times is None:
                save_times = result["save_times"]
                energy_every = result.get("energy_every", 1)

        # Stack and validate
        energy_stack = np.stack(all_energy, axis=0)  # (K, n_energy_samples)
        energy_val = validate_energy_decay(energy_stack[0])
        if not energy_val["pass"]:
            for warn in energy_val["warnings"]:
                validation_results["issues"].append(f"{eta_label}: {warn}")

        # Interpolate to 540×540
        snapshots_stack = np.stack(all_snapshots, axis=0)  # (K, n_snaps, sim_nx, sim_ny)
        n_probes, n_snaps, sim_nx, sim_ny = snapshots_stack.shape

        snapshots_interp = np.zeros((n_probes, n_snaps, OUTPUT_NX, OUTPUT_NY), dtype=np.float32)
        for pi in range(n_probes):
            snapshots_interp[pi] = interpolate_to_output_grid(
                snapshots_stack[pi], sim_nx, sim_ny, OUTPUT_NX, OUTPUT_NY
            )

        final_energy = float(energy_stack[0, -1]) if energy_stack.shape[1] > 0 else 0.0

        # Per spec: snapshots shape is (n_snapshots, 540, 540) - use click probe (index 0)
        # Per spec: energy shape is (n_energy_samples,) - use click probe (index 0)
        np.savez_compressed(
            os.path.join(scene_dir, f"u_lt_{eta_label}.npz"),
            # Spec-compliant shapes (click probe only):
            snapshots=snapshots_interp[0],  # (n_snaps, 540, 540)
            energy=energy_stack[0],  # (n_energy_samples,)
            # Metadata per spec:
            save_times=save_times,
            nx=OUTPUT_NX, ny=OUTPUT_NY,
            sim_nx=sim_nx, sim_ny=sim_ny,
            dx=edx, dt=edt,
            actual_cfl=eta.actual_cfl,
            final_energy=final_energy,
            interpolation_order=INTERP_ORDER,
            interpolation_method=INTERP_METHOD,
            # Extra: all probes (not in spec, but useful)
            snapshots_all_probes=snapshots_interp,  # (K, n_snaps, 540, 540)
            energy_all_probes=energy_stack,  # (K, n_energy_samples)
            probe_names=PROBE_NAMES,
        )

        max_len = max(y.shape[1] for y in all_y_mics)
        y_stack = np.zeros((len(PROBE_NAMES), N_MICS, max_len), dtype=np.float32)
        for ki, y in enumerate(all_y_mics):
            y_stack[ki, :, :y.shape[1]] = y

        np.save(os.path.join(scene_dir, f"y_mics_{eta_label}.npy"), y_stack)

        with open(os.path.join(scene_dir, f"{eta_label}.json"), "w") as f:
            json.dump(convert_numpy_types(asdict(eta)), f, indent=2)

    # 7. Grid metadata
    grid_config = GridConfig(
        dx=base_dx, dt=base_dt, nx=nx, ny=ny,
        fcut=fcut, cfl_target=alpha, actual_cfl=actual_cfl,
        k=k, c=c, lambda_min=grid_meta["lambda_min"],
        n_timesteps=n_t, T_sim=T, pml_thickness=pml_cells,
    )
    with open(os.path.join(scene_dir, "grid.json"), "w") as f:
        json.dump(convert_numpy_types(asdict(grid_config)), f, indent=2)

    validation_results["world"] = {
        "n_segments": world.n_segments,
        "room_area": world.room_area,
        "kappa_range": [min(kappas), max(kappas)],
    }
    validation_results["grid"] = {
        "nx": nx, "ny": ny,
        "actual_cfl": actual_cfl,
        "n_timesteps": n_t,
    }

    return validation_results


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GPU-Accelerated FDTD Dataset Generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--out_dir", default="data/fdtd")
    parser.add_argument("--n_scenes", type=int, default=100)
    parser.add_argument("--fcut", type=float, default=FCUT_DEFAULT)
    parser.add_argument("--k", type=int, default=K_DEFAULT)
    parser.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    parser.add_argument("--c", type=float, default=C_DEFAULT)
    parser.add_argument("--T", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=5.0)
    parser.add_argument("--pml_cells", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--device", type=int, default=0, help="CUDA device ID")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 60)
    print("GPU FDTD Dataset Generator")
    print("=" * 60)
    print(f"Output: {args.out_dir}")
    print(f"Scenes: {args.n_scenes}")
    print(f"Device: cuda:{args.device}")

    grid_meta = choose_dx_dt(args.c, args.fcut, k=args.k, alpha=args.alpha)
    actual_cfl = args.c * grid_meta["dt"] / grid_meta["dx"]

    print(f"\nGrid: dx={grid_meta['dx']:.6f}m, dt={grid_meta['dt']:.8f}s, CFL={actual_cfl:.4f}")
    print(f"Simulation: T={args.T}s, gamma={args.gamma}, PML={args.pml_cells}")
    print(f"Output: {OUTPUT_NX}×{OUTPUT_NY}, {N_MICS} mics, {len(PROBE_NAMES)} probes")

    if actual_cfl >= CFL_MAX:
        print(f"\nERROR: CFL={actual_cfl:.4f} >= {CFL_MAX:.4f}")
        sys.exit(1)

    # Initialize GPU solver
    print(f"\nInitializing GPU solver on cuda:{args.device}...")
    solver = FDTDSolverGPU(device_id=args.device)

    # Generate scenes
    print(f"\nGenerating {args.n_scenes} scenes...")
    t_start = time.time()
    results = []

    for sid in tqdm(range(args.n_scenes), desc="Scenes", unit="scene", ncols=80):
        try:
            r = build_scene_gpu(
                scene_id=sid, out_dir=args.out_dir, solver=solver,
                fcut=args.fcut, k=args.k, alpha=args.alpha, c=args.c,
                T=args.T, gamma=args.gamma, pml_cells=args.pml_cells,
                save_every=args.save_every, seed=args.seed,
            )
            results.append(r)
        except Exception as e:
            print(f"\nError scene {sid}: {e}")
            traceback.print_exc()
            results.append({"scene_id": f"scene_{sid:05d}", "passed": False, "issues": [str(e)]})

    elapsed = time.time() - t_start

    n_passed = sum(1 for r in results if r.get("passed", False))
    n_failed = len(results) - n_passed

    # Save manifest
    manifest_path = os.path.join(args.out_dir, "manifest.jsonl")
    with open(manifest_path, "w") as f:
        for r in results:
            f.write(json.dumps(convert_numpy_types(r)) + "\n")

    print(f"\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  Total:  {len(results)}")
    print(f"  Passed: {n_passed} ({100*n_passed/len(results):.1f}%)")
    print(f"  Failed: {n_failed}")
    print(f"  Time:   {elapsed:.1f}s ({elapsed/len(results):.1f}s per scene)")
    print(f"  Rate:   {len(results)*60/elapsed:.1f} scenes/min")
    print(f"  Manifest: {manifest_path}")

    sys.exit(0 if n_failed == 0 else 1)


if __name__ == "__main__":
    main()
