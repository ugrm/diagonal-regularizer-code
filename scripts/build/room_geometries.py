#!/usr/bin/env python
"""
Build data/modal_summary/room_geometries.npz: polygon vertices of every room.

One array per room, keyed by scene id ("scene_00800" -> (n_vertices, 2) vertex array),
taken from room_vertices in the paper's modal dataset (data/modal/scene_*/eigenpairs.npz).
Read by src/visualization/fig_delta_vs_Eop.py (room areas for the colour scale).

Usage: python scripts/build/room_geometries.py
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.paths import DATA, MODAL, check_paper_dataset  # noqa: E402

OUT_PATH = DATA / "modal_summary" / "room_geometries.npz"


def main():
    check_paper_dataset(MODAL)
    geoms = {}
    for sid in sorted(os.listdir(MODAL)):
        eig_path = MODAL / sid / "eigenpairs.npz"
        if sid.startswith("scene_") and eig_path.exists():
            geoms[sid] = np.asarray(np.load(eig_path)["room_vertices"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_PATH, **geoms)
    print(f"Wrote {len(geoms)} room geometries to {OUT_PATH}")


if __name__ == "__main__":
    main()
