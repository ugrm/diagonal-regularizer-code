"""Deterministic seed management."""

import numpy as np


def set_seed(seed=42):
    """Set global numpy seed."""
    np.random.seed(seed)


def get_room_seed(base_seed, room_idx):
    """Deterministic seed for room generation: base_seed + room_idx."""
    return base_seed + room_idx


def get_scene_seed(base_seed, room_idx, mic_config_idx, n_configs=5):
    """Deterministic seed for scene (mic config) generation."""
    return base_seed * 1000 + room_idx * n_configs + mic_config_idx
