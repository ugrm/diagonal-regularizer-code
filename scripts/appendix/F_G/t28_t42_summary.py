#!/usr/bin/env python
"""Appendix G.5 summary: Table T28 (2D polygons vs 3D boxes) and the T42 rows
"3D boxes", "3D worst cells" and "2D boundary-inclusive maxima".

2D column, from released data (no new computation):
  * in-scope rooms = anisotropy/E_op_empirical_187.npz scene_ids (= the p-sweep rooms
    with legacy K_total > 50); K_total from the legacy modal eigenvalues (DR_MODAL);
  * H = sum w_n^2 with w_n proportional to lambda_n^-|s| over n > 50 (|s| = 1.1266);
  * landscape ratio = max/min over p in [0, 3] of each room's oracle curve
    (p_sweep_K50_M8.npz, M = 8), median over rooms;
  * cost = per-room delta (cost_K50_M8.npz), median over in-scope rooms;
  * T42 maxima over all 197 rooms, T <= 100 (T in {1,5,10,20,50,100}) and T = 1000.
3D column, from box3d_flatness.npz: per cell the
median curve over the 20 realizations gives P*, P(|s| = 1.13), the relative and the
absolute gap and the flatness; T28 reports medians over the 8 cells, T42 the worst cell.

Worst-case pairing: T42 prints the 3D worst cell's relative cost with that cell's own
absolute gap, but the 2D row's absolute value is the maximum over rooms, which at
T = 1000 is a different room. Both pairings are stored here (abs_at_max_rel, max_abs).

Output: data/experiments/appendix/t28_t42_summary.npz
Usage:  python scripts/appendix/F_G/t28_t42_summary.py [--box3d PATH]
"""
import argparse

import numpy as np

from _common import out_path, OUT
from src.utils.paths import MODAL, EXP, check_paper_dataset

K_RET = 50
M_IDX_VALUE = 8
T_LOW = [1, 5, 10, 20, 50, 100]
S_HAT_3D = 1.13


def summary_2d():
    cost = np.load(EXP / "cost" / "cost_K50_M8.npz")
    psw = np.load(EXP / "p_sweep" / "p_sweep_K50_M8.npz", allow_pickle=True)
    eop = np.load(EXP / "anisotropy" / "E_op_empirical_187.npz", allow_pickle=True)
    s = float(cost["s_value"])
    rooms = [str(r) for r in psw["room_ids"]]
    K_tot, H = [], []
    for sid in rooms:
        ev = np.asarray(np.load(MODAL / sid / "eigenpairs.npz",
                                allow_pickle=True)["eigenvalues"], float)
        K_tot.append(len(ev))
        u = ev[K_RET:] ** (-s)
        H.append(np.sum(u ** 2) / np.sum(u) ** 2 if len(u) else np.nan)
    K_tot, H = np.array(K_tot), np.array(H)
    ins = np.isin(rooms, [str(x) for x in eop["scene_ids"]])
    assert np.array_equal(ins, K_tot > K_RET), "in-scope mask != K_total > 50"

    Tv = list(psw["T_values"])
    mi = list(psw["M_values"]).index(M_IDX_VALUE)
    pm = psw["p_values"] <= 3.0 + 1e-9
    flat = {}
    for T in [1, 100, 1000]:
        P = psw["P_oracle"][:, Tv.index(T), mi, :][:, pm]
        flat[T] = np.max(P, axis=1) / np.min(P, axis=1)
    Tc = list(cost["T"])
    rel, ab = cost["per_room_delta_rel"], cost["per_room_delta_P"]
    out = dict(rooms=np.array(rooms), in_scope=ins, K_total=K_tot, H=H, s_value=s,
               n_in_scope=int(ins.sum()),
               med_K_total=float(np.median(K_tot[ins])),
               med_K_total_197=float(np.median(K_tot)),
               med_H=float(np.median(H[ins])))
    for T in [1, 100, 1000]:
        out[f"med_flat_T{T}"] = float(np.median(flat[T][ins]))
        ti = Tc.index(T)
        out[f"med_cost_pp_T{T}"] = float(100 * np.median(ab[ti][ins]))
        out[f"med_cost_rel_T{T}"] = float(100 * np.median(rel[ti][ins]))
    for tag, Ts in [("Tle100", T_LOW), ("T1000", [1000])]:
        idx = [Tc.index(T) for T in Ts]
        R, A = 100 * rel[idx], 100 * ab[idx]                 # (n_T, 197)
        i, j = np.unravel_index(np.nanargmax(R), R.shape)
        k, l = np.unravel_index(np.nanargmax(A), A.shape)
        out[f"max_rel_{tag}"] = float(R[i, j])
        out[f"max_rel_{tag}_room"] = rooms[j]
        out[f"max_rel_{tag}_T"] = int(Ts[i])
        out[f"abs_at_max_rel_{tag}"] = float(A[i, j])
        out[f"max_abs_{tag}"] = float(A[k, l])
        out[f"max_abs_{tag}_room"] = rooms[l]
        out[f"max_abs_{tag}_T"] = int(Ts[k])
    return out


def summary_3d(box3d_path):
    bx = np.load(box3d_path, allow_pickle=True)
    pg = bx["p_grid"]
    i3 = np.searchsorted(pg, 3.0) + 1
    TB = [int(t) for t in bx["T_values"]]
    names = sorted({k[:-5] for k in bx.files if k.endswith("_meta")})
    cells = []
    for nm in names:
        meta = bx[f"{nm}_meta"]
        med_c = np.nanmedian(bx[f"{nm}_curves"], axis=0)
        rec = dict(cell=nm, V=float(meta[0]), K_total=int(meta[1]), H=float(meta[2]),
                   Eop=float(meta[3]), n_real=int(bx[f"{nm}_curves"].shape[0]))
        for ti, T in enumerate(TB):
            c = med_c[ti]
            j = int(np.nanargmin(c))
            P_at = np.interp(S_HAT_3D, pg, c)
            rec[f"Pstar{T}"] = float(c[j])
            rec[f"rel{T}"] = float(100 * (P_at - c[j]) / c[j])
            rec[f"abs{T}"] = float(100 * (P_at - c[j]))
            rec[f"flat{T}"] = float(np.nanmax(c[:i3]) / np.nanmin(c[:i3]))
        cells.append(rec)
    col = lambda k: np.array([r[k] for r in cells])
    out = dict(cells=np.array(names), n_cells=len(cells),
               n_real=int(min(col("n_real"))),
               V_min=float(col("V").min()), V_max=float(col("V").max()),
               K_total_min=int(col("K_total").min()), K_total_max=int(col("K_total").max()),
               med_K_total=float(np.median(col("K_total"))), med_H=float(np.median(col("H"))))
    for T in TB:
        out[f"med_flat_T{T}"] = float(np.median(col(f"flat{T}")))
        out[f"med_cost_pp_T{T}"] = float(np.median(col(f"abs{T}")))
        out[f"med_cost_rel_T{T}"] = float(np.median(col(f"rel{T}")))
        for k in ["rel", "abs", "flat", "Pstar"]:
            out[f"cell_{k}{T}"] = col(f"{k}{T}")
    for tag, Ts in [("Tle100", [T for T in TB if T <= 100]), ("T1000", [1000])]:
        R = np.array([col(f"rel{T}") for T in Ts])
        A = np.array([col(f"abs{T}") for T in Ts])
        i, j = np.unravel_index(np.argmax(R), R.shape)
        k, l = np.unravel_index(np.argmax(A), A.shape)
        out[f"max_rel_{tag}"] = float(R[i, j])
        out[f"max_rel_{tag}_cell"] = names[j]
        out[f"abs_at_max_rel_{tag}"] = float(A[i, j])
        out[f"max_abs_{tag}"] = float(A[k, l])
        out[f"max_abs_{tag}_cell"] = names[l]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--box3d", default=str(OUT / "box3d_flatness.npz"))
    args = ap.parse_args()
    check_paper_dataset()
    d2, d3 = summary_2d(), summary_3d(args.box3d)
    print("| quantity | 2D polygons | 3D boxes |")
    print("|---|---|---|")
    print(f"| cases | {d2['n_in_scope']} in-scope rooms | {d3['n_cells']} cells |")
    print(f"| median K_total | {d2['med_K_total']:.0f} | {d3['med_K_total']:.1f} |")
    print(f"| median H | {d2['med_H']:.4f} | {d3['med_H']:.4f} |")
    print("| landscape ratio T=1/100/1000 | " +
          " / ".join(f"{d2[f'med_flat_T{T}']:.3f}" for T in [1, 100, 1000]) + " | " +
          " / ".join(f"{d3[f'med_flat_T{T}']:.3f}" for T in [1, 100, 1000]) + " |")
    for T in [100, 1000]:
        print(f"| cost T={T} | {d2[f'med_cost_pp_T{T}']:.3f} pp "
              f"({d2[f'med_cost_rel_T{T}']:.2f}%) | {d3[f'med_cost_pp_T{T}']:.3f} pp "
              f"({d3[f'med_cost_rel_T{T}']:.2f}%) |")
    for tag in ["Tle100", "T1000"]:
        print(f"\n{tag}: 2D max rel {d2[f'max_rel_{tag}']:.2f}% ({d2[f'max_rel_{tag}_room']}, "
              f"T={d2[f'max_rel_{tag}_T']}, abs there {d2[f'abs_at_max_rel_{tag}']:.2f} pp); "
              f"2D max abs {d2[f'max_abs_{tag}']:.2f} pp ({d2[f'max_abs_{tag}_room']})")
        print(f"{tag}: 3D max rel {d3[f'max_rel_{tag}']:.2f}% ({d3[f'max_rel_{tag}_cell']}, "
              f"abs there {d3[f'abs_at_max_rel_{tag}']:.2f} pp); 3D max abs "
              f"{d3[f'max_abs_{tag}']:.2f} pp ({d3[f'max_abs_{tag}_cell']})")
    out = out_path("t28_t42_summary.npz")
    np.savez(out, **{f"d2_{k}": v for k, v in d2.items()},
             **{f"d3_{k}": v for k, v in d3.items()})
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
