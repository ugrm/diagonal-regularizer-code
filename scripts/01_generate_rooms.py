#!/usr/bin/env python
"""
Step 1: Generate room geometries, FEM meshes, and eigensolve.

For each room r = 0, ..., n_rooms-1:
  1. Generate random convex polygon (3-10 segments)
  2. Generate FEM mesh (resolution for K_max modes)
  3. Solve Robin Laplacian eigenvalue problem
  4. Save eigenpairs and metadata

Output: data/rooms/room_{r:05d}.npz + data/rooms/manifest.json

Usage:
    python scripts/01_generate_rooms.py --config configs/default.yaml
    python scripts/01_generate_rooms.py --config configs/default.yaml --rooms 10  # quick test
"""

import argparse
import gc
import json
import os
import resource
import sys
import time

import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.config import load_config
from src.utils.seeds import get_room_seed
from src.geometry.polygon import generate_world, polygon_area
from src.geometry.mesh import mesh_room, assemble_stiffness_mass, assemble_robin_boundary
from src.physics.eigensolve import solve_eigenproblem, verify_eigenpairs, estimate_n_modes_weyl


def process_one_room(room_idx, cfg, rooms_dir, verbose=True):
    """Generate geometry, mesh, and eigensolve for one room."""
    seed = get_room_seed(cfg["dataset"]["seed"], room_idx)
    rng = np.random.default_rng(seed)

    ds_cfg = cfg["dataset"]
    eig_cfg = cfg["eigensolve"]
    c = cfg["simulation"]["speed_of_sound"]
    fcut = eig_cfg["freq_max"]
    k_max = eig_cfg["k_max"]
    epw = eig_cfg["elements_per_wavelength"]

    # Step 1: Generate polygon
    world = generate_world(
        rng,
        n_seg_min=ds_cfg["segment_range"][0],
        n_seg_max=ds_cfg["segment_range"][1],
        side_min=ds_cfg["side_range"][0],
        side_max=ds_cfg["side_range"][1],
        kappa_min=ds_cfg["kappa_range"][0],
        kappa_max=ds_cfg["kappa_range"][1],
        c=c,
    )

    vertices = np.array(world["vertices"])
    segments = np.array(world["segments"])
    kappas = np.array(world["kappa_per_segment"])
    n_seg = world["n_segments"]
    area = world["room_area"]

    # Step 2: Estimate modes and mesh
    n_modes_request = estimate_n_modes_weyl(area, fcut, c, margin=1.1)
    n_modes_request = max(n_modes_request, k_max + 10)

    lambda_min = c / fcut
    h_target = lambda_min / epw
    max_area = 0.5 * h_target**2

    try:
        mesh = mesh_room(vertices, segments, max_area)
    except Exception as e:
        return {"status": "FAIL", "error": f"Meshing: {e}", "room_idx": room_idx}

    mesh_verts = mesh["vertices"]
    mesh_tris = mesh["triangles"]
    mesh_segs = mesh["segments"]
    mesh_seg_markers = mesh["segment_markers"].flatten()

    # Step 3: Assemble and eigensolve
    K_mat, M_mat = assemble_stiffness_mass(mesh_verts, mesh_tris)
    B_mat = assemble_robin_boundary(mesh_verts, mesh_segs, mesh_seg_markers, kappas)

    try:
        eigenvalues, eigenvectors = solve_eigenproblem(K_mat, B_mat, M_mat, n_modes_request)
    except Exception as e:
        return {"status": "FAIL", "error": f"Eigensolve: {e}", "room_idx": room_idx}

    # Truncate to modes below fcut
    freqs = c * np.sqrt(np.maximum(eigenvalues, 0)) / (2 * np.pi)
    mask = freqs <= fcut * 1.01
    eigenvalues = eigenvalues[mask]
    eigenvectors = eigenvectors[:, mask]
    freqs = freqs[mask]
    K_actual = len(eigenvalues)

    # Verify (skip for zero-mode rooms)
    if K_actual > 0:
        _, ortho_err = verify_eigenpairs(eigenvalues, eigenvectors, M_mat, c, verbose=False)
    else:
        ortho_err = 0.0

    # Step 4: Save
    room_id = f"room_{room_idx:05d}"
    out_path = os.path.join(rooms_dir, f"{room_id}.npz")
    np.savez_compressed(
        out_path,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors.astype(np.float32),
        mesh_nodes=mesh_verts,
        mesh_elements=mesh_tris,
        boundary_segments=mesh_segs,
        segment_markers=mesh_seg_markers,
        vertices=vertices,
        n_segments=n_seg,
        kappas=kappas,
        c=c,
        fcut=fcut,
        frequencies=freqs,
        source_pos=np.array(world["source_pos"]),
    )

    # Rooms with too few modes are physically valid but useless for experiments
    MIN_K_USEFUL = 5
    if K_actual < MIN_K_USEFUL:
        status = "DEGENERATE"
    else:
        status = "OK"

    meta = {
        "room_id": room_id,
        "room_idx": room_idx,
        "n_segments": n_seg,
        "room_area_m2": round(float(area), 4),
        "K_total": K_actual,
        "eigenfreq_max_hz": round(float(freqs[-1]), 1) if K_actual > 0 else 0.0,
        "n_mesh_nodes": len(mesh_verts),
        "n_mesh_triangles": len(mesh_tris),
        "orthonormality_error": float(ortho_err),
        "status": status,
        "seed": seed,
    }

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if verbose and (room_idx % 10 == 0 or room_idx < 5):
        print(f"  [{room_idx:4d}] {n_seg}-seg, area={area:.2f}m², "
              f"K={K_actual}, mesh={len(mesh_verts)} nodes, RSS={rss_mb:.0f}MB")

    return meta


def _rebuild_manifest_from_files(rooms_dir, n_rooms, cfg):
    """Rebuild manifest from existing .npz files (for resume after crash)."""
    manifest = []
    for r in range(n_rooms):
        room_id = f"room_{r:05d}"
        path = os.path.join(rooms_dir, f"{room_id}.npz")
        if os.path.exists(path):
            try:
                d = np.load(path)
                freqs = d["frequencies"]
                meta = {
                    "room_id": room_id,
                    "room_idx": r,
                    "n_segments": int(d["n_segments"]),
                    "room_area_m2": round(float(polygon_area(d["vertices"])), 4),
                    "K_total": len(d["eigenvalues"]),
                    "eigenfreq_max_hz": round(float(freqs[-1]), 1) if len(freqs) > 0 else 0.0,
                    "n_mesh_nodes": len(d["mesh_nodes"]),
                    "n_mesh_triangles": len(d["mesh_elements"]),
                    "orthonormality_error": 0.0,  # not stored, skip
                    "status": "OK",
                    "seed": get_room_seed(cfg["dataset"]["seed"], r),
                }
                d.close()
            except Exception as e:
                meta = {"room_id": room_id, "room_idx": r, "status": "FAIL",
                        "error": f"Corrupt file: {e}"}
        else:
            meta = None  # not yet generated
        manifest.append(meta)
    return manifest


def _save_manifest(manifest, manifest_path):
    """Write manifest atomically (tmp + rename)."""
    entries = [m for m in manifest if m is not None]
    tmp = manifest_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(entries, f, indent=2)
    os.replace(tmp, manifest_path)


def main():
    parser = argparse.ArgumentParser(description="Generate room geometries + eigensolve")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--rooms", type=int, default=None, help="Override n_rooms")
    parser.add_argument("--resume", action="store_true",
                        help="Skip rooms that already have .npz files")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_rooms = args.rooms or cfg["dataset"]["n_rooms"]
    rooms_dir = cfg["paths"]["rooms_dir"]
    os.makedirs(rooms_dir, exist_ok=True)
    manifest_path = os.path.join(rooms_dir, "manifest.json")

    # Determine which rooms to skip
    if args.resume:
        manifest = _rebuild_manifest_from_files(rooms_dir, n_rooms, cfg)
        existing = sum(1 for m in manifest if m is not None)
        print(f"Resuming: {existing}/{n_rooms} rooms already exist, "
              f"{n_rooms - existing} remaining")
    else:
        manifest = [None] * n_rooms

    print(f"Generating {n_rooms} rooms...")
    print(f"  K_max={cfg['eigensolve']['k_max']}, fcut={cfg['eigensolve']['freq_max']}Hz")
    print(f"  Output: {rooms_dir}")

    n_ok = sum(1 for m in manifest if m is not None and m["status"] == "OK")
    n_fail = sum(1 for m in manifest if m is not None and m["status"] != "OK")
    n_processed = 0
    t0 = time.time()

    for r in range(n_rooms):
        # Skip existing rooms on resume
        if manifest[r] is not None:
            continue

        meta = process_one_room(r, cfg, rooms_dir)
        manifest[r] = meta
        n_processed += 1

        if meta["status"] == "OK":
            n_ok += 1
        else:
            n_fail += 1
            print(f"  FAIL [{r}]: {meta.get('error', 'unknown')}")

        # Incremental manifest save every 10 rooms
        if n_processed % 10 == 0:
            _save_manifest(manifest, manifest_path)

        # Free memory between rooms
        gc.collect()

    elapsed = time.time() - t0
    print(f"\nDone: {n_ok}/{n_rooms} OK, {n_fail} failed, "
          f"{n_processed} processed this run ({elapsed:.0f}s)")

    # Final manifest save
    _save_manifest(manifest, manifest_path)
    print(f"Manifest: {manifest_path}")

    # Summary stats
    ok_rooms = [m for m in manifest if m is not None and m["status"] == "OK"]
    if ok_rooms:
        K_vals = [m["K_total"] for m in ok_rooms]
        areas = [m["room_area_m2"] for m in ok_rooms]
        print(f"\n  K_total: min={min(K_vals)}, max={max(K_vals)}, median={np.median(K_vals):.0f}")
        print(f"  Area: min={min(areas):.2f}, max={max(areas):.2f}, median={np.median(areas):.2f}")


if __name__ == "__main__":
    main()
