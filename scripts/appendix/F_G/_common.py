"""Shared helpers for the Appendix F/G scripts (run them from the repository root).

Importing this module puts the repository root on sys.path so that `src` imports work
when a script is started as `python scripts/appendix/F_G/<name>.py`.
"""
import os
import sys
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.utils.paths import EXP  # noqa: E402

OUT = EXP / "appendix"


def out_path(name):
    """data/experiments/appendix/<name>, creating the folder (name may hold a subfolder)."""
    p = OUT / name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def stable_seed(key):
    """Deterministic 32-bit RNG seed for `key` (a str or a tuple of str/int).

    Used instead of `abs(hash(key)) % 2**32`: Python randomizes str hashing per process,
    so hash()-seeded draws cannot be reproduced. Pass Python ints, not numpy ints, so
    that repr() does not depend on the numpy version.
    """
    return zlib.crc32(repr(key).encode())


def with_seed(key, seed):
    """`key` unchanged when seed is None (the shipped single run), else `key` + (seed,).

    The --seed option of the random-draw scripts uses this, so that each seed redraws
    everything the key seeds (sensor layouts included) while seed=None reproduces the
    shipped outputs.
    """
    if seed is None:
        return key
    return (*key, int(seed)) if isinstance(key, tuple) else (key, int(seed))


def n_workers(requested):
    """Worker count: the request, capped at 16 and at the CPU count."""
    return max(1, min(int(requested), 16, os.cpu_count() or 1))
