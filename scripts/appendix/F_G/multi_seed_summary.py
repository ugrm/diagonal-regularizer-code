#!/usr/bin/env python
"""Appendix F/G seed-dependent values over 10 fixed seeds: median [min, max] across seeds.

Reads the per-seed outputs of run_multi_seed.sh (data/experiments/appendix/seeds/) and
recomputes, per seed, every printed value that depends on random draws, with the same
statistics as check_F_G.py / t28_t42_summary.py:
  * T41: largest one-sample KS gap of the t3 / correlated noise marginals (prior_tails_qq);
    single-draw delta(|s|) at T = 100, Gaussian prior, per seed (single_draw_seed_spread);
  * T27: flatness generic (median over the 101-room cohort) / rectangle (median over the
    4 rectangles) at K_total = 75, 100, 200, 300;
  * T42: flatness at K_total = 50; K_total = 300 difference; checkpoints (of six) where
    the rectangle median lies inside the generic IQR; 3D worst cells (relative cost and
    the same cell's absolute gap);
  * T28: 3D medians over the 8 cells (landscape ratio, cost at T = 100 and 1000);
  * G.5 prose checks against the fixed 2D values.

Output: data/experiments/appendix/seeds/multi_seed_summary.npz (+ stdout)
Usage:  python scripts/appendix/F_G/multi_seed_summary.py
"""
import numpy as np
from scipy import stats

from _common import OUT, REPO
from check_F_G import rect_flat_medians, delta_interp
from t28_t42_summary import summary_2d, summary_3d

SEEDS = list(range(10))
SD = OUT / "seeds"
K_ALL = [50, 75, 100, 150, 200, 300]


def mr(v, fmt="{:.3f}"):
    v = np.asarray(v, float)
    return (fmt.format(np.median(v)) + " [" + fmt.format(v.min()) + ", "
            + fmt.format(v.max()) + "]")


def compute(sd=SD):
    """Per-seed statistics from the outputs in `sd`: (S, d2, d3s), S a dict of arrays."""
    S = {}
    # ---- F: KS gap
    ks1, ks2 = [], []
    for s in SEEDS:
        z = np.load(sd / f"prior_tails_qq_seed{s}.npz")
        ks1.append(max(stats.kstest(z[f"z_{k}"], "norm").statistic
                       for k in ["heavy_tail", "correlated"]))
        ks2.append(max(stats.ks_2samp(z[f"z_{k}"], z["z_gaussian"]).statistic
                       for k in ["heavy_tail", "correlated"]))
    S["ks_one"], S["ks_two"] = np.array(ks1), np.array(ks2)

    # ---- F: single-draw delta per seed (one realization per room, median over 20 rooms)
    pb = np.load(sd / "single_draw_seed_spread_n10.npz")
    for pr in ["gaussian", "heavy_tail", "correlated"]:
        for T in [1, 100]:
            S[f"sd_{pr}_T{T}"] = np.array([100 * delta_interp(c)[2]
                                           for c in pb[f"{pr}_T{T}_probe"]])

    # ---- G: flatness (cohort generic vs rectangles)
    gen = {K: [] for K in K_ALL}
    rect = {K: [] for K in K_ALL}
    inside, diff300 = [], []
    for s in SEEDS:
        co = np.load(sd / f"results_cohort_flatness_seed{s}.npz")
        mk = np.load(sd / f"matched_ktotal_seed{s}.npz")
        rfl = rect_flat_medians(mk)
        n_in = 0
        for K in K_ALL:
            g = co[f"flat_K{K}"]
            gen[K].append(np.median(g))
            rect[K].append(rfl[K])
            n_in += np.percentile(g, 25) <= rfl[K] <= np.percentile(g, 75)
        inside.append(n_in)
        diff300.append(abs(rfl[300] - np.median(co["flat_K300"])))
    for K in K_ALL:
        S[f"flat_gen_K{K}"], S[f"flat_rect_K{K}"] = np.array(gen[K]), np.array(rect[K])
    S["inside_iqr"], S["diff_K300"] = np.array(inside), np.array(diff300)

    # ---- G.5: 3D boxes vs fixed 2D values
    d2 = summary_2d()
    d3s = [summary_3d(sd / f"box3d_flatness_seed{s}.npz") for s in SEEDS]
    keys3 = ([f"med_flat_T{T}" for T in [1, 100, 1000]]
             + [f"med_cost_{u}_T{T}" for T in [100, 1000] for u in ["pp", "rel"]]
             + [f"{k}_{t}" for t in ["Tle100", "T1000"]
                for k in ["max_rel", "abs_at_max_rel", "max_abs"]])
    for k in keys3:
        S[f"d3_{k}"] = np.array([d[k] for d in d3s])
    cost = np.load(OUT.parent / "cost" / "cost_K50_M8.npz")
    rel2d = {T: 100 * cost["per_room_delta_rel"][list(cost["T"]).index(T)]
             for T in [1, 100, 1000]}
    S["in_2d_range"] = np.array([all(d[f"cell_rel{T}"].max() <= np.nanmax(rel2d[T]) and
                                     d[f"cell_rel{T}"].min() >= np.nanmin(rel2d[T])
                                     for T in [1, 100, 1000]) for d in d3s])
    S["ratio_2d_3d_Tle100"] = d2["max_rel_Tle100"] / S["d3_max_rel_Tle100"]
    S["ratio_2d_3d_T1000"] = d2["max_rel_T1000"] / S["d3_max_rel_T1000"]
    S["flatter_than_2d"] = np.array([[d[f"med_flat_T{T}"] < d2[f"med_flat_T{T}"]
                                      for T in [1, 100, 1000]] for d in d3s])
    # per-cell medians over seeds, then the worst cell (alternative to per-seed worst)
    for T in [1, 100, 1000]:
        S[f"cellmed_rel{T}"] = np.median([d[f"cell_rel{T}"] for d in d3s], axis=0)
        S[f"cellmed_abs{T}"] = np.median([d[f"cell_abs{T}"] for d in d3s], axis=0)
    S["cells"] = d3s[0]["cells"]
    return S, d2, d3s


def main():
    S, d2, d3s = compute()

    # ---- report
    print(f"# Appendix F/G over {len(SEEDS)} fixed seeds: median [min, max] across seeds\n")
    print("T41 KS gap, one-sample vs N(0,1), max(t3, corr):", mr(S["ks_one"]))
    print("T41 KS gap, two-sample vs Gaussian sample:      ", mr(S["ks_two"]))
    g = S["sd_gaussian_T100"]
    print("T41 single-draw delta T=100 Gaussian per seed (%):",
          ", ".join(f"{x:.2f}" for x in g), f"-> median {np.median(g):.1f}, "
          f"range {g.min():.1f}-{g.max():.1f}, max/min {g.max()/g.min():.1f}x")
    for pr in ["gaussian", "heavy_tail", "correlated"]:
        for T in [1, 100]:
            print(f"   single-draw {pr:10s} T={T:3d}: {mr(S[f'sd_{pr}_T{T}'], '{:.2f}')} %")
    print("\nT27/T42 flatness generic / rectangle:")
    for K in K_ALL:
        print(f"  K_total={K:3d}: {mr(S[f'flat_gen_K{K}'])} / {mr(S[f'flat_rect_K{K}'])}")
    print("T42 K=300 difference:", mr(S["diff_K300"]),
          "| rectangle median inside generic IQR (of six), per seed:", list(S["inside_iqr"]))
    print("\nT28 3D column (2D fixed):")
    for T in [1, 100, 1000]:
        print(f"  landscape ratio T={T}: 3D {mr(S[f'd3_med_flat_T{T}'])}  (2D "
              f"{d2[f'med_flat_T{T}']:.3f}); 3D flatter than 2D in "
              f"{int(S['flatter_than_2d'][:, [1, 100, 1000].index(T)].sum())}/10 seeds")
    for T in [100, 1000]:
        print(f"  cost T={T}: {mr(S[f'd3_med_cost_pp_T{T}'], '{:.2f}')} pp "
              f"({mr(S[f'd3_med_cost_rel_T{T}'], '{:.2f}')} %)")
    print("\nT42 3D worst cell per seed, relative % (same cell's abs pp) / largest abs pp:")
    for t in ["Tle100", "T1000"]:
        print(f"  {t}: {mr(S[f'd3_max_rel_{t}'], '{:.2f}')} % "
              f"({mr(S[f'd3_abs_at_max_rel_{t}'], '{:.2f}')} pp); largest abs "
              f"{mr(S[f'd3_max_abs_{t}'], '{:.2f}')} pp")
    print("  per-cell median over seeds, worst cell: T<=100 "
          f"{max(S['cellmed_rel1'].max(), S['cellmed_rel100'].max()):.2f} %, "
          f"T=1000 {S['cellmed_rel1000'].max():.2f} %")
    print(f"\nG.5: 2D worst T<=100 {d2['max_rel_Tle100']:.1f} %, T=1000 "
          f"{d2['max_rel_T1000']:.1f} %")
    print("  2D/3D worst ratio T<=100:", mr(S["ratio_2d_3d_Tle100"], "{:.1f}"),
          "| T=1000:", mr(S["ratio_2d_3d_T1000"], "{:.2f}"))
    print("  all 8 cells inside the 2D range, seeds:", int(S["in_2d_range"].sum()), "/ 10")

    out = SD / "multi_seed_summary.npz"
    np.savez(out, seeds=np.array(SEEDS), **S)
    print(f"\nwrote {out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
