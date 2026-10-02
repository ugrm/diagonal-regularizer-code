"""
Figure 7 (Supplement): Individual Room Diagnostics — 6 rooms × 2 columns.

Paper reference: Appendix C, Figure 9
Data: noise_profile_K50_M8.npz, p_sweep_K50_M8.npz,
      data/rooms/room_NNNNN.npz (vertices), data/modal/scene_NNNNN/mic_pos.npy

Layout: 6×2 grid
    Col 1: Room polygon + mic positions
    Col 2: Normalized P(p) curve at T=1, M=8

Per-room noise profiles are not drawn: the aggregate noise profile is in fig01
(Appendix A) and per-room traces at this size are illegible at print resolution.

P(p) curves are normalized to P(p)/P(0) so all rooms collapse onto a
comparable basin shape, regardless of absolute P levels.

Representative rooms: 3-seg, 4-seg, 5-seg (scene_00803), 6-seg, 8-seg, 10-seg
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FixedLocator

from src.visualization.style import (
    setup_style, save_figure, FULL_WIDTH, WONG,
    resolve_data_path, cli_wrapper, enable_all_spines, legend_bottom,
)

REPRESENTATIVE_ROOMS = {
    3: "scene_00897",
    4: "scene_00930",
    5: "scene_00803",   # Confirmed in exp1 187-room set
    6: "scene_00802",
    8: "scene_00850",
    10: "scene_00900",
}


def _find_5seg_room(data_root):
    """Fallback: find first val room with 5 vertices."""
    for scene in sorted(os.listdir(data_root)):
        if not scene.startswith("scene_"):
            continue
        try:
            num = int(scene.split("_")[1])
        except (IndexError, ValueError):
            continue
        if num < 800:
            continue
        eig_path = os.path.join(data_root, scene, "eigenpairs.npz")
        if os.path.isfile(eig_path):
            eig = np.load(eig_path)
            if "room_vertices" in eig and eig["room_vertices"].shape[0] == 5:
                return scene
    return None


def _load_room_vertices(scene_id, data_dir):
    """Room polygon and microphone positions for one scene.

    The P(p) curves in this figure come from the legacy 8-microphone arena
    (p_sweep_K50_M8.npz), so the microphones drawn must be THOSE eight. The
    release data hold a different, regenerated 16-microphone layout for the
    same rooms, and drawing it next to legacy curves mislabels the figure.
    Legacy data are therefore preferred; the release layout is the fallback.

    Returns (vertices, mic_positions) or (None, None).
    """
    from src.utils.paths import MODAL
    eig_path = os.path.join(MODAL, scene_id, "eigenpairs.npz")
    mic_path = os.path.join(MODAL, scene_id, "mic_pos.npy")
    if os.path.isfile(eig_path) and os.path.isfile(mic_path):
        eig = np.load(eig_path, allow_pickle=True)
        if "room_vertices" in eig.files:
            return np.array(eig["room_vertices"]), np.load(mic_path)

    if data_dir is None:
        return None, None
    meta_path = os.path.join(data_dir, "modal", scene_id, "metadata.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        room_id = meta.get("room_id")
        if room_id:
            room_path = os.path.join(data_dir, "rooms", f"{room_id}.npz")
            mic_path = os.path.join(data_dir, "modal", scene_id, "mic_pos.npy")
            if os.path.isfile(room_path) and os.path.isfile(mic_path):
                r = np.load(room_path, allow_pickle=True)
                if "vertices" in r.files:
                    return np.array(r["vertices"]), np.load(mic_path)[:8]
    return None, None


def generate(data_dir=None, output_dir="figures/", config=None,
             T_idx=0, M_idx=1):
    """Generate Figure 9 (App C): Per-room diagnostics, 6 rooms x 2 cols."""
    setup_style()

    noise = np.load(resolve_data_path("noise_profile_K50_M8.npz", data_dir), allow_pickle=True)
    sweep = np.load(resolve_data_path("p_sweep_K50_M8.npz", data_dir), allow_pickle=True)

    exp1_ids = np.array([str(r) for r in noise["room_ids"]])
    exp1_s = noise["s_per_room"]
    exp1_map = {rid: i for i, rid in enumerate(exp1_ids)}

    exp2_ids = np.array([str(r) for r in sweep["room_ids"]])
    P_oracle = sweep["P_oracle"]
    p_values = sweep["p_values"]
    exp2_map = {rid: i for i, rid in enumerate(exp2_ids)}

    rooms = REPRESENTATIVE_ROOMS.copy()
    seg_counts = [3, 4, 5, 6, 8, 10]
    selected = [(ns, rooms[ns]) for ns in seg_counts if rooms.get(ns)]

    # The population |s| the closed form uses, read from the data.
    s_pop = float(np.median(np.abs(exp1_s)))

    n_rows = len(selected)
    # Drawn at text width and included at \linewidth: six rows of two panels
    # plus one shared legend on the right.
    # Three rows of two room-plus-landscape pairs, one legend line below.
    n_grid_rows = int(np.ceil(n_rows / 2))
    fig, grid = plt.subplots(n_grid_rows, 4, figsize=(FULL_WIDTH, 5.4),
                             gridspec_kw={"width_ratios": [1, 1.6, 1, 1.6]})
    grid = np.atleast_2d(grid)
    axes = np.array([[grid[r // 2, 2 * (r % 2)], grid[r // 2, 2 * (r % 2) + 1]]
                     for r in range(n_rows)], dtype=object)

    for row, (n_seg, rid) in enumerate(selected):
        # Col 1: Room polygon + the eight microphones the curves use.
        ax = axes[row, 0]
        verts, mic = _load_room_vertices(rid, data_dir)
        if verts is not None:
            closed = np.vstack([verts, verts[0:1]])
            ax.plot(closed[:, 0], closed[:, 1], color=WONG["black"], lw=1.2,
                    label="boundary")
            ax.fill(verts[:, 0], verts[:, 1], color="#f0f0f0", alpha=0.5)
            if mic is not None:
                ax.scatter(mic[:, 0], mic[:, 1], s=16, c=WONG["red"],
                           marker="o", zorder=3, edgecolors="white",
                           linewidths=0.5, label="microphone")
            ax.set_aspect("equal")
            enable_all_spines(ax)
            # narrow rooms take two x ticks, wide ones three; rotate so the
            # labels of a 1-metre-wide panel cannot collide
            x_range = float(verts[:, 0].max() - verts[:, 0].min())
            # two whole-metre ticks per axis; the thinnest room is 1.6 m wide
            ax.set_xticks([0, max(1, int(np.floor(verts[:, 0].max())))])
            ax.set_yticks([0, max(1, int(np.floor(verts[:, 1].max())))])
        short = rid.split("_")[1]
        ax.set_xlabel("$x$ (m)")
        ax.set_ylabel("$y$ (m)")

        # Col 2: Normalized P(p) curve = P(p) / P(0), so every room shows a
        # comparable basin shape whatever its absolute error level.
        ax = axes[row, 1]
        if rid in exp2_map:
            idx2 = exp2_map[rid]
            P_room = P_oracle[idx2, T_idx, M_idx, :]
            P_ref = P_room[0] if P_room[0] > 0 and np.isfinite(P_room[0]) else np.nan
            P_norm = P_room / P_ref if not np.isnan(P_ref) else P_room
            ax.plot(p_values, P_norm, color=WONG["blue"], lw=1.3,
                    label=r"$P(p)/P(0)$, $T{=}1$")
            p_star = p_values[np.nanargmin(P_room)]
            s_room = exp1_s[exp1_map[rid]] if rid in exp1_map else np.nan
            s_abs = abs(s_room) if not np.isnan(s_room) else np.nan
            ax.set_xlim(-0.1, 3.1)
            ax.set_ylim(0.55, 1.05)
            ylo, yhi = ax.get_ylim()
            ax.axvline(s_pop, ls="--", color="#666666", lw=0.9, alpha=0.9,
                       zorder=1, label=r"population $|\hat{s}|$")
            if not np.isnan(s_abs):
                ax.axvline(s_abs, ls="--", color=WONG["cyan"], lw=0.9,
                           alpha=0.9, zorder=2, label=r"room's $|s|$ (OLS)")
            ax.plot(p_star, ylo, marker="^", color=WONG["red"], ms=6,
                    markeredgecolor="white", markeredgewidth=0.5,
                    clip_on=False, zorder=5, ls="none",
                    label=r"room's minimum $p^\star$")
        # one title per pair, on the wider panel
        ax.set_title(f"room {short}\n{n_seg} segments")
        ax.set_xlabel("Exponent $p$")
        ax.set_ylabel(r"$P(p)\,/\,P(0)$")

    legend_bottom(fig, ncol=6, fit_width=True)
    save_figure(fig, "fig07_per_room_diagnostics", output_dir)
    return fig


if __name__ == "__main__":
    cli_wrapper(generate)
