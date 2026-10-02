"""Compare a recomputed p-sweep with the published one, per room and for the paper's headline numbers.

Usage: python scripts/compare_p_sweep.py data/experiments_repro/p_sweep/p_sweep_K50_M8.npz
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.paths import EXP, MODAL

new = np.load(sys.argv[1], allow_pickle=True)
ref = np.load(EXP / "p_sweep" / "p_sweep_K50_M8.npz", allow_pickle=True)
assert [str(r) for r in new["room_ids"]] == [str(r) for r in ref["room_ids"]], "room lists differ"
rooms = [str(r) for r in ref["room_ids"]]
p = ref["p_values"]; T = ref["T_values"]
Mi = list(ref["M_values"]).index(8)
Pn, Pr = new["P_oracle"][:, :, Mi, :], ref["P_oracle"][:, :, Mi, :]

K = np.array([json.load(open(MODAL / r / "metadata.json"))["K"] for r in rooms])
inscope = K > 50                                  # rooms with truncation noise
s_abs = float(np.median(np.abs(np.load(EXP / "noise_profile" / "noise_profile_K50_M8.npz")["s_per_room"])))  # 1.1266

d = np.abs(Pn - Pr)
print(f"rooms {len(rooms)} (in-scope {inscope.sum()}), NaN new {np.isnan(Pn).sum()} ref {np.isnan(Pr).sum()}")
print(f"per-room |dP| (M=8): median {np.nanmedian(d):.2e}, 99th pct {np.nanpercentile(d, 99):.2e}, max {np.nanmax(d):.2e}")


def headline(P):
    """Median relative cost of the closed form vs the per-room oracle, per T (in-scope rooms)."""
    out = []
    for t in range(len(T)):
        Pc = np.array([np.interp(s_abs, p, P[r, t]) for r in range(len(rooms))])
        Po = np.nanmin(P[:, t], axis=1)
        out.append(np.nanmedian(((Pc - Po) / Po)[inscope]) * 100)
    return np.array(out)


hn, hr = headline(Pn), headline(Pr)
print(f"\n|s| = {s_abs:.4f}; median relative cost (%) over in-scope rooms")
print("T      " + " ".join(f"{t:>7d}" for t in T))
print("paper  " + " ".join(f"{v:7.3f}" for v in hr))
print("repro  " + " ".join(f"{v:7.3f}" for v in hn))
tiers = lambda h: (h[T <= 50].max(), h[T <= 100].max(), h.max())
print("tiers T<=50 / T<=100 / all:  paper %.2f / %.2f / %.2f   repro %.2f / %.2f / %.2f" % (*tiers(hr), *tiers(hn)))
ps_r = p[np.nanargmin(np.nanmedian(Pr[inscope], axis=0), axis=1)]
ps_n = p[np.nanargmin(np.nanmedian(Pn[inscope], axis=0), axis=1)]
print("population p* paper", ps_r, "\n               repro", ps_n)
pr_room = p[np.nanargmin(Pr, axis=2)]; pn_room = p[np.nanargmin(Pn, axis=2)]
print(f"per-room p* identical in {np.mean(pr_room == pn_room) * 100:.1f}% of (room, T) cells; "
      f"within 0.1 in {np.mean(np.abs(pr_room - pn_room) <= 0.1 + 1e-9) * 100:.1f}%")
