"""
Figure: M3 Training Dynamics — Loss curves and p-hat trajectories.

Paper reference: Section 6 (Learned Methods), supplementary
Data: checkpoints/m3_hypernet_n800_T{T}_s{seed}/trajectory.npz

Panels:
    (a) Training loss vs epoch for each (T, seed) combination
    (b) p-hat trajectory vs epoch, with R² annotated at convergence

Reads trajectory.npz files from M3 n=800 checkpoints.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from src.visualization.style import (
    setup_style, save_figure, FULL_WIDTH, WONG, cli_wrapper,
    resolve_data_path, legend_bottom,
)

T_COLORS = {
    1: WONG["blue"],
    100: WONG["green"],
    1000: WONG["red"],
}


def _load_trajectories(checkpoints_dir):
    """Scan checkpoints for M3 n=800 trajectories."""
    results = {}
    pattern = os.path.join(checkpoints_dir, "m3_hypernet_n800_T*_s*/trajectory.npz")
    for path in sorted(glob.glob(pattern)):
        dirname = os.path.basename(os.path.dirname(path))
        parts = dirname.split("_")
        T = int([p for p in parts if p.startswith("T")][0][1:])
        seed = int([p for p in parts if p.startswith("s")][0][1:])
        d = np.load(path)
        if T not in results:
            results[T] = []
        results[T].append({
            "seed": seed,
            "epoch": d["epoch"],
            "train_loss": d["train_loss"],
            "p_hat": d["p_hat"],
            "p_hat_r2": d["p_hat_r2"],
            "phat_final": float(d["phat_final"]),
            "r2_final": float(d["r2_final"]),
        })
    return results


def generate(data_dir=None, output_dir="figures/", config=None):
    """Generate M3 training dynamics figure."""
    setup_style()

    # Find checkpoints directory
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    ckpt_dir = None
    for candidate in [
        os.path.join(data_dir, "..", "checkpoints") if data_dir else None,
        os.path.join(proj_root, "checkpoints"),
    ]:
        if candidate and os.path.isdir(candidate):
            ckpt_dir = candidate
            break

    if ckpt_dir is None:
        print("WARNING: No checkpoints directory found. "
              "Skipping fig_m3_training.")
        return None

    results = _load_trajectories(ckpt_dir)
    if not results:
        print("WARNING: No M3 n=800 trajectories found. "
              "Skipping fig_m3_training.")
        return None

    # The population |s| the closed form uses, read from the noise profile so
    # the reference line cannot drift from the data.
    noise = np.load(resolve_data_path("noise_profile_K50_M8.npz", data_dir),
                    allow_pickle=True)
    s_pop = float(np.median(np.abs(noise["s_per_room"])))

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH, 2.7))

    # ── Panel (a): Loss curves ────────────────────────────────────────
    ax = axes[0]
    for T in sorted(results.keys()):
        color = T_COLORS.get(T, WONG["black"])
        for i, run in enumerate(results[T]):
            label = f"$T={T}$, each seed" if i == 0 else None
            ax.plot(run["epoch"], run["train_loss"],
                    color=color, alpha=0.5, lw=0.8, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss ($P$)")
    ax.set_title("(a) M3 loss curves")
    ax.set_yscale("log")

    # ── Panel (b): p-hat trajectories ─────────────────────────────────
    ax = axes[1]
    for T in sorted(results.keys()):
        color = T_COLORS.get(T, WONG["black"])
        for i, run in enumerate(results[T]):
            label = f"$T={T}$, each seed" if i == 0 else None
            ax.plot(run["epoch"], run["p_hat"],
                    color=color, alpha=0.5, lw=0.8, label=label)

    ax.axhline(s_pop, ls="--", color=WONG["black"], lw=0.8, alpha=0.6,
               label=r"population $|\hat{s}|$")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$\hat{p}$ (power-law fit)")
    ax.set_title(r"(b) $\hat{p}$ trajectory")

    fig.tight_layout(w_pad=2.0)
    legend_bottom(fig)
    save_figure(fig, "fig_m3_training", output_dir)
    return fig


if __name__ == "__main__":
    cli_wrapper(generate)
