"""YAML config loading with defaults."""

import os

import numpy as np
import yaml


def load_config(path="configs/default.yaml"):
    """Load YAML config file and compute derived values."""
    with open(path) as f:
        cfg = yaml.safe_load(f)

    # Compute p_grid from min/max/step
    ev = cfg.get("evaluation", {})
    p_min = ev.get("p_grid_min", 0.0)
    p_max = ev.get("p_grid_max", 6.0)
    p_step = ev.get("p_grid_step", 0.1)
    cfg["evaluation"]["p_grid"] = np.round(
        np.arange(p_min, p_max + p_step / 2, p_step), 2
    ).tolist()

    # Compute blacklist set (legacy dataset only)
    bl = cfg.get("blacklist", {}).get("scenes", [])
    cfg["blacklist"]["scene_ids"] = {f"scene_{s:05d}" for s in bl}

    return cfg
