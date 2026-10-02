"""Repository paths, overridable by environment variables.

DR_MODAL  modal dataset root (default data/modal): one scene_XXXXX/ directory per room
DR_ROOMS  full eigensolve output of scripts/01_generate_rooms.py (default data/rooms)
"""
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
MODAL = Path(os.environ.get("DR_MODAL", DATA / "modal"))
ROOMS = Path(os.environ.get("DR_ROOMS", DATA / "rooms"))
EXP = DATA / "experiments"
SUPP = DATA / "supplementary"


def check_paper_dataset(modal_root=MODAL):
    """Raise if modal_root is not the dataset the paper's numbers come from.

    The paper uses the published modal dataset (download with `make download-data`).
    A dataset regenerated with different simulation settings has different mode counts;
    room 00806 is a quick fingerprint (19 retained modes in the paper's dataset).
    """
    meta = Path(modal_root) / "scene_00806" / "metadata.json"
    if not meta.exists():
        raise FileNotFoundError(f"{meta} not found: run `make download-data` first")
    k = json.load(open(meta)).get("K")
    if k != 19:
        raise RuntimeError(f"{modal_root} is not the paper's dataset (scene_00806 has K={k}, expected 19)")
