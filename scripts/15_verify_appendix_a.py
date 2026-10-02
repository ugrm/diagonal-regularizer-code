#!/usr/bin/env python
"""
Step 15: Verify Appendix A quantities (anisotropy, Herfindahl, Berry).

Computes five quantities referenced in Appendix A and checks
them against expected values.

Output: prints PASS/FAIL for each check. Non-zero exit if any FAIL.
"""

import argparse
import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BLACKLIST = {"scene_00905", "scene_00913", "scene_00921"}
K_USE = 50
M_USE = 8


def _load_legacy_rooms(exp_dir):
    """Return the 187 legacy room IDs from the noise profile."""
    noise_path = os.path.join(exp_dir, "noise_profile", "noise_profile_K50_M8.npz")
    noise = np.load(noise_path, allow_pickle=True)
    return list(noise["room_ids"])


def check_E_op(modal_root, legacy_rooms):
    """||E||_op empirical via Method B: actual phi, sigma_trunc = tr(R)/M."""
    E_op = []
    for sid in legacy_rooms:
        Phi_path = os.path.join(modal_root, sid, "measurement_matrix.npy")
        traj_path = os.path.join(modal_root, sid, "modal_trajectories.npz")
        eig_path = os.path.join(modal_root, sid, "eigenpairs.npz")
        if not all(os.path.exists(p) for p in [Phi_path, traj_path, eig_path]):
            continue
        Phi = np.load(Phi_path)[:M_USE]
        a = np.load(traj_path)["a"]
        ev = np.load(eig_path, allow_pickle=True)["eigenvalues"]
        if len(ev) <= K_USE:
            continue
        sigma_sq = np.var(a[K_USE:], axis=1)
        Phi_tr = Phi[:, K_USE:]
        R = Phi_tr @ np.diag(sigma_sq) @ Phi_tr.T
        s2 = np.trace(R) / M_USE
        if s2 < 1e-30:
            continue
        E = R / s2 - np.eye(M_USE)
        E_op.append(np.max(np.abs(np.linalg.eigvalsh(E))))
    E_op = np.array(E_op)
    return float(np.median(E_op)), float(np.percentile(E_op, 95)), len(E_op)


def check_Eij_MC(modal_root, s_val, n_trials=50000, seed=42):
    """Berry model MC: E[E_ij^2] / H on 6 representative rooms."""
    rooms = [
        "scene_00850", "scene_00817", "scene_00934",
        "scene_00802", "scene_00836", "scene_00897",
    ]
    rng = np.random.default_rng(seed)
    ratios = []
    for sid in rooms:
        eig_path = os.path.join(modal_root, sid, "eigenpairs.npz")
        if not os.path.exists(eig_path):
            continue
        ev = np.load(eig_path, allow_pickle=True)["eigenvalues"]
        if len(ev) <= K_USE:
            continue
        w = ev[K_USE:] ** (-s_val)
        W = np.sum(w)
        H = np.sum(w ** 2) / W ** 2
        N_trunc = len(ev) - K_USE
        sigma2_trunc = W / M_USE

        Eij_sq = np.empty(n_trials)
        for t in range(n_trials):
            Phi = rng.normal(0, 1.0 / np.sqrt(M_USE), size=(N_trunc, M_USE))
            E_01 = (M_USE / W) * np.sum(w * Phi[:, 0] * Phi[:, 1])
            Eij_sq[t] = E_01 ** 2
        ratios.append(np.mean(Eij_sq) / H)
    return np.array(ratios)


def check_signal_dominance(modal_root, s_val):
    """Count rooms where signal dynamic range exceeds noise eigenvalue ratio."""
    n_dom, n_total = 0, 0
    for i in range(800, 1000):
        sid = f"scene_{i:05d}"
        if sid in BLACKLIST:
            continue
        paths = [os.path.join(modal_root, sid, f)
                 for f in ["eigenpairs.npz", "measurement_matrix.npy", "modal_trajectories.npz"]]
        if not all(os.path.exists(p) for p in paths):
            continue
        eig = np.load(paths[0], allow_pickle=True)
        ev = eig["eigenvalues"]
        if len(ev) <= K_USE or len(ev) - K_USE < M_USE:
            continue
        n_total += 1
        Phi = np.load(paths[1])[:M_USE]
        a = np.load(paths[2])["a"]
        sigma_sq = np.var(a[K_USE:], axis=1)
        R = Phi[:, K_USE:] @ np.diag(sigma_sq) @ Phi[:, K_USE:].T
        s2 = np.trace(R) / M_USE
        if s2 < 1e-30:
            continue
        evals = np.linalg.eigvalsh(R / s2)
        noise_ratio = evals[-1] / max(evals[0], 1e-30)
        signal_range = (ev[K_USE - 1] / ev[0]) ** s_val
        if signal_range > noise_ratio:
            n_dom += 1
    return n_dom, n_total


def check_herfindahl(modal_root, s_val):
    """Population H median across rooms with K_total > K_USE."""
    H_all = []
    for i in range(800, 1000):
        sid = f"scene_{i:05d}"
        if sid in BLACKLIST:
            continue
        eig_path = os.path.join(modal_root, sid, "eigenpairs.npz")
        if not os.path.exists(eig_path):
            continue
        ev = np.load(eig_path, allow_pickle=True)["eigenvalues"]
        if len(ev) <= K_USE:
            continue
        w = ev[K_USE:] ** (-s_val)
        H_all.append(np.sum(w ** 2) / np.sum(w) ** 2)
    return float(np.median(H_all)), len(H_all)


def _get_room_area(eig, sid, modal_root):
    """Get room area from eigenpairs (legacy) or room_geometries.npz (release)."""
    if "room_vertices" in eig:
        verts = eig["room_vertices"]
    else:
        geom_path = os.path.join(modal_root, "room_geometries.npz")
        if not os.path.exists(geom_path):
            proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            geom_path = os.path.join(proj, "data", "modal_summary", "room_geometries.npz")
        if os.path.exists(geom_path):
            geom = np.load(geom_path, allow_pickle=True)
            if sid in geom:
                verts = geom[sid]
            else:
                return None
        else:
            return None
    return float(np.abs(
        np.sum(verts[:-1, 0] * verts[1:, 1] - verts[1:, 0] * verts[:-1, 1])
        + verts[-1, 0] * verts[0, 1] - verts[0, 0] * verts[-1, 1]
    ) / 2)


def check_rho_D_Ntrunc(modal_root):
    """Spearman rho(Berry KS D, N_trunc) across rooms with K_total > K_USE."""
    ks_D_all, N_trunc_all = [], []
    for i in range(800, 1000):
        sid = f"scene_{i:05d}"
        if sid in BLACKLIST:
            continue
        eig_path = os.path.join(modal_root, sid, "eigenpairs.npz")
        mm_path = os.path.join(modal_root, sid, "measurement_matrix.npy")
        if not all(os.path.exists(p) for p in [eig_path, mm_path]):
            continue
        eig = np.load(eig_path, allow_pickle=True)
        ev = eig["eigenvalues"]
        if len(ev) <= K_USE:
            continue
        area = _get_room_area(eig, sid, modal_root)
        if area is None:
            continue
        Phi = np.load(mm_path)[:M_USE]
        Csq = (area * Phi[:, :K_USE] ** 2).ravel()
        ks_d, _ = stats.kstest(Csq, "chi2", args=(1,))
        ks_D_all.append(ks_d)
        N_trunc_all.append(len(ev) - K_USE)
    rho, pval = stats.spearmanr(ks_D_all, N_trunc_all)
    return float(rho), float(pval), len(ks_D_all)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--modal-root", default=None)
    parser.add_argument("--exp-dir", default=None)
    args = parser.parse_args()

    # Resolve paths
    proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    exp_dir = args.exp_dir or os.path.join(proj_root, "data", "experiments")

    # Modal root: the paper's dataset (data/modal, see `make download-data`)
    from src.utils.paths import MODAL, check_paper_dataset
    modal_root = args.modal_root or str(MODAL)
    check_paper_dataset(modal_root)
    print(f"Modal root: {modal_root}")

    s_val = 1.1266  # legacy |s|
    noise_path = os.path.join(exp_dir, "noise_profile", "noise_profile_K50_M8.npz")
    if os.path.exists(noise_path):
        s_per = np.load(noise_path, allow_pickle=True)["s_per_room"]
        s_val = float(abs(np.median(s_per)))

    legacy_rooms = _load_legacy_rooms(exp_dir)
    n_pass, n_fail = 0, 0
    results = []

    def report(name, value, expected, tol, unit=""):
        nonlocal n_pass, n_fail
        if isinstance(expected, str):
            ok = str(value) == expected
        else:
            ok = abs(value - expected) <= tol
        status = "PASS" if ok else "FAIL"
        if ok:
            n_pass += 1
        else:
            n_fail += 1
        results.append((name, value, expected, tol, status))
        val_str = f"{value}{unit}" if isinstance(value, str) else f"{value:.4f}{unit}"
        exp_str = f"{expected}{unit}" if isinstance(expected, str) else f"{expected:.4f}{unit}"
        print(f"  [{status}] {name}: {val_str} (expect {exp_str}, tol={tol})")

    print("=" * 70)
    print("Appendix A verification (15_verify_appendix_a.py)")
    print("=" * 70)

    # 1. ||E||_op
    print("\n1. Empirical ||E||_op (Method B: actual phi, tr(R)/M)")
    med, p95, n = check_E_op(modal_root, legacy_rooms)
    report("||E||_op median", med, 0.58, 0.01)
    report("||E||_op 95th", p95, 2.42, 0.05)
    print(f"     ({n} rooms)")

    # 2. E[E_ij^2] / H
    print("\n2. Berry MC: E[E_ij^2] / H (50,000 trials)")
    ratios = check_Eij_MC(modal_root, s_val)
    for r in ratios:
        report(f"E[Eij²]/H", r, 1.0, 0.05)

    # 3. Signal dominance
    print("\n3. Signal dominance (signal range > noise ratio)")
    n_dom, n_total = check_signal_dominance(modal_root, s_val)
    report("signal-dominated rooms", f"{n_dom}/{n_total}", "170/186", 0)

    # 4. Herfindahl
    print("\n4. Herfindahl index H (population median)")
    H_med, n_H = check_herfindahl(modal_root, s_val)
    report("H median", H_med, 0.005, 0.001)
    print(f"     ({n_H} rooms)")

    # 5. rho(D, N_trunc)
    print("\n5. Spearman rho(Berry KS D, N_trunc)")
    rho, pval, n_rho = check_rho_D_Ntrunc(modal_root)
    report("rho(D, N_trunc)", rho, 0.29, 0.02)
    print(f"     (p={pval:.2e}, {n_rho} rooms)")

    print(f"\n{'=' * 70}")
    print(f"Results: {n_pass} PASS, {n_fail} FAIL")
    print(f"{'=' * 70}")

    sys.exit(1 if n_fail > 0 else 0)


if __name__ == "__main__":
    main()
