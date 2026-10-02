#!/usr/bin/env python
"""
Per-room s_room vs population |s| comparison.

For each room, fits s_room from its own modal amplitude variances,
then compares P(s_room) to P(|s|_pop) and the per-room oracle p*.

Output: data/experiments/sroom_vs_pop/sroom_per_room.npz
        data/experiments/sroom_vs_pop/sroom_summary.json
"""

import json
import os
import sys

import numpy as np
from scipy import stats as sp_stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.analysis.noise_profile import measure_prior_s
from src.estimation.metrics import fit_power_law


def interp_P(P_curve, p_grid, p_val):
    """Linearly interpolate P(p_val) from a P(p) curve on p_grid."""
    p_val = np.clip(p_val, p_grid[0], p_grid[-1])
    idx = np.searchsorted(p_grid, p_val) - 1
    idx = np.clip(idx, 0, len(p_grid) - 2)
    frac = (p_val - p_grid[idx]) / (p_grid[idx + 1] - p_grid[idx])
    return (1 - frac) * P_curve[idx] + frac * P_curve[idx + 1]


def bootstrap_s(a, eigenvalues, K_use, n_boot=1000, skip_first=5,
                skip_last=2, max_snaps=200, rng=None):
    """Bootstrap CI for the spectral decay slope s."""
    if rng is None:
        rng = np.random.default_rng(42)
    all_snaps = list(range(skip_first, min(a.shape[1] - skip_last, max_snaps)))
    sigma_sq = np.var(a[:K_use, all_snaps], axis=1)
    ev = eigenvalues[:K_use]

    # Identify valid points (same as fit_power_law)
    valid = (ev > 0) & (sigma_sq > 0) & np.isfinite(ev) & np.isfinite(sigma_sq)
    log_x = np.log(ev[valid])
    log_y = np.log(sigma_sq[valid])
    n_pts = len(log_x)
    if n_pts < 3:
        return np.nan, np.nan

    slopes = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_pts, size=n_pts)
        lx, ly = log_x[idx], log_y[idx]
        # OLS slope
        x_mean = lx.mean()
        slopes[b] = np.sum((lx - x_mean) * (ly - ly.mean())) / np.sum(
            (lx - x_mean) ** 2
        )
    ci_lo, ci_hi = np.percentile(slopes, [2.5, 97.5])
    return ci_lo, ci_hi


def main():
    release_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    from src.utils.paths import MODAL, check_paper_dataset
    check_paper_dataset()
    modal_root = str(MODAL)
    data_dir = os.path.join(release_root, "data", "experiments")
    out_dir = os.path.join(data_dir, "sroom_vs_pop")
    os.makedirs(out_dir, exist_ok=True)

    # --- Load P_sweep (197 rooms, full landscape) ---
    ps = np.load(
        os.path.join(data_dir, "p_sweep", "p_sweep_K50_M8.npz"), allow_pickle=True
    )
    P_oracle_full = ps["P_oracle"]  # (197, 10, 3, 61)
    p_grid = ps["p_values"]         # (61,) = [0.0, 0.1, ..., 6.0]
    T_all = ps["T_values"]          # (10,)
    M_values = ps["M_values"]       # (3,) = [4, 8, 16]
    room_ids_197 = np.array([str(r) for r in ps["room_ids"]])
    M8_idx = int(np.where(M_values == 8)[0][0])

    # --- Load noise profile (187 rooms with pre-computed s_per_room) ---
    nf = np.load(
        os.path.join(data_dir, "noise_profile", "noise_profile_K50_M8.npz"), allow_pickle=True
    )
    s_per_room_187 = nf["s_per_room"]  # (187,), NEGATIVE slopes
    room_ids_187 = np.array([str(r) for r in nf["room_ids"]])
    excluded_rooms = set(str(r) for r in nf["excluded_rooms"])

    # Map 187 rooms to 197-room indices
    room_to_idx = {r: i for i, r in enumerate(room_ids_197)}

    # --- Build full 197-room s_room array ---
    K_USE = 50
    s_room = np.full(197, np.nan)
    r2_s = np.full(197, np.nan)
    s_room_ci_lo = np.full(197, np.nan)
    s_room_ci_hi = np.full(197, np.nan)
    K_used_for_fit = np.full(197, 0, dtype=int)

    rng = np.random.default_rng(42)

    # Fill from noise profile (187 rooms)
    for i, rid in enumerate(room_ids_187):
        idx = room_to_idx[rid]
        s_room[idx] = -s_per_room_187[i]  # positive |s|
        r2_s[idx] = float(nf["r2_s_full"][i])
        K_used_for_fit[idx] = K_USE

    # Compute s for excluded rooms + bootstrap CIs for ALL rooms
    print("Computing per-room s and bootstrap CIs...")
    for r, rid in enumerate(room_ids_197):
        rid_str = str(rid)
        rdir = os.path.join(modal_root, rid_str)
        if not os.path.isdir(rdir):
            continue

        eig = np.load(os.path.join(rdir, "eigenpairs.npz"))
        traj = np.load(os.path.join(rdir, "modal_trajectories.npz"))
        ev = eig["eigenvalues"].astype(np.float64)
        a = traj["a"].astype(np.float64)
        K_room = min(len(ev), K_USE)

        if rid_str in excluded_rooms:
            # Compute s from scratch for excluded rooms
            sigma_sq, s_val, r2_val = measure_prior_s(a, ev, K_room)
            s_room[r] = -s_val  # positive |s|
            r2_s[r] = r2_val
            K_used_for_fit[r] = K_room

        # Bootstrap CI for all rooms
        ci_lo, ci_hi = bootstrap_s(a, ev, K_room, n_boot=2000, rng=rng)
        s_room_ci_lo[r] = -ci_hi  # flip sign: slope is negative
        s_room_ci_hi[r] = -ci_lo

    valid = ~np.isnan(s_room)
    print(f"  Valid rooms: {valid.sum()}/197")

    # --- Target T values ---
    T_target = np.array([1, 50, 100, 500, 1000, 2100])
    T_idx = np.array([int(np.where(T_all == t)[0][0]) for t in T_target])
    n_T = len(T_target)

    # --- Population |s| ---
    s_pop = float(-np.nanmedian(s_per_room_187))  # = 1.1266

    # --- Compute P(p*), P(|s|_pop), P(s_room) for each room ---
    P_star = np.full((197, n_T), np.nan)
    p_oracle = np.full((197, n_T), np.nan)
    P_pop = np.full((197, n_T), np.nan)
    P_sroom = np.full((197, n_T), np.nan)

    for r in range(197):
        for ti, t_idx in enumerate(T_idx):
            P_curve = P_oracle_full[r, t_idx, M8_idx, :]  # (61,)

            # Oracle p* and P(p*)
            best_p_idx = int(np.argmin(P_curve))
            P_star[r, ti] = P_curve[best_p_idx]
            p_oracle[r, ti] = p_grid[best_p_idx]

            # P at population |s|
            P_pop[r, ti] = interp_P(P_curve, p_grid, s_pop)

            # P at per-room s_room
            if not np.isnan(s_room[r]):
                P_sroom[r, ti] = interp_P(P_curve, p_grid, s_room[r])

    # --- Compute relative deltas ---
    delta_pop = (P_pop - P_star) / P_star
    delta_sroom = (P_sroom - P_star) / P_star

    # --- Save .npz ---
    npz_path = os.path.join(out_dir, "sroom_per_room.npz")
    np.savez(
        npz_path,
        room_ids=room_ids_197,
        s_room=s_room,
        s_room_ci_lo=s_room_ci_lo,
        s_room_ci_hi=s_room_ci_hi,
        r2_s=r2_s,
        K_used_for_fit=K_used_for_fit,
        s_pop=s_pop,
        p_oracle=p_oracle,
        P_star=P_star,
        P_pop=P_pop,
        P_sroom=P_sroom,
        delta_pop=delta_pop,
        delta_sroom=delta_sroom,
        T_values=T_target,
    )
    print(f"Saved: {npz_path}")

    # --- Compute summary statistics ---
    summary = {
        "s_pop": round(s_pop, 4),
        "median_s_room": round(float(np.nanmedian(s_room)), 4),
        "iqr_s_room": [
            round(float(np.nanpercentile(s_room, 25)), 4),
            round(float(np.nanpercentile(s_room, 75)), 4),
        ],
        "std_s_room": round(float(np.nanstd(s_room)), 4),
        "n_valid": int(valid.sum()),
        "n_excluded": int((~valid).sum()),
        "per_T": {},
    }

    print("\n" + "=" * 70)
    print("HEADLINE NUMBERS")
    print("=" * 70)
    print(
        f"Median s_room = {np.nanmedian(s_room):.4f}, "
        f"IQR [{np.nanpercentile(s_room, 25):.4f}, "
        f"{np.nanpercentile(s_room, 75):.4f}]"
    )
    print(f"Population |s| = {s_pop:.4f}")
    print(
        f"Median |s_room - s_pop| = "
        f"{np.nanmedian(np.abs(s_room - s_pop)):.4f}"
    )
    print()

    for ti, T in enumerate(T_target):
        d_pop = delta_pop[:, ti]
        d_sr = delta_sroom[:, ti]
        v = ~np.isnan(d_sr)

        med_pop = float(np.nanmedian(d_pop[v]))
        med_sr = float(np.nanmedian(d_sr[v]))
        frac_sr_wins = int(np.sum(d_sr[v] < d_pop[v]))
        n_valid_t = int(v.sum())
        med_diff = float(np.nanmedian(d_sr[v] - d_pop[v]))

        # Gap closed: fraction of pop delta eliminated by using s_room
        gap_closed_pct = (
            ((med_pop - med_sr) / med_pop * 100) if med_pop > 1e-10 else 0.0
        )

        summary["per_T"][str(int(T))] = {
            "median_delta_pop": round(med_pop * 100, 4),
            "median_delta_sroom": round(med_sr * 100, 4),
            "sroom_wins_over_pop": frac_sr_wins,
            "n_valid": n_valid_t,
            "median_delta_diff": round(med_diff * 100, 4),
            "gap_closed_pct": round(gap_closed_pct, 2),
        }

        print(
            f"T={T:>5d}: δ(|s|)={med_pop * 100:>6.2f}%, "
            f"δ(s_room)={med_sr * 100:>6.2f}%, "
            f"gap closed: {gap_closed_pct:>5.1f}%, "
            f"s_room beats |s|: {frac_sr_wins}/{n_valid_t}"
        )

    print("=" * 70)

    # Save summary JSON
    json_path = os.path.join(out_dir, "sroom_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {json_path}")


if __name__ == "__main__":
    main()
