"""
Aperture-to-wavelength numbers for three room scales (Appendix I, Table T30).

For a rectangular room with rigid (Neumann) walls, counts the modes below
f_max = 1 kHz (DC excluded), takes the K = 50th eigenvalue, and reports
    l_min = 2 pi / sqrt(lambda_K)            shortest retained wavelength
    D_ap / l_min                             aperture-to-wavelength ratio
    2 sin(pi D_ap / l_min)                   largest difference a unit plane wave
                                             at l_min shows across the aperture
for a compact array of aperture D_ap = 12.6 cm.

Usage:
    python scripts/appendix/I/aperture_rooms.py
"""

import numpy as np

C_SOUND = 343.0
F_MAX = 1000.0
K = 50
D_AP = 0.126
ROOMS = {
    "Large":   (3.45, 7.20, 2.45),
    "Compact": (1.33, 2.10, 2.47),
    "Closet":  (1.0, 1.0, 1.0),
}


def neumann_eigenvalues(Lx, Ly, Lz, f_max=F_MAX, c=C_SOUND):
    """Sorted eigenvalues pi^2 (n^2/Lx^2 + m^2/Ly^2 + p^2/Lz^2) below f_max, DC excluded."""
    thr = (2 * f_max / c) ** 2
    n = np.arange(int(Lx * np.sqrt(thr)) + 1)[:, None, None]
    m = np.arange(int(Ly * np.sqrt(thr)) + 1)[None, :, None]
    p = np.arange(int(Lz * np.sqrt(thr)) + 1)[None, None, :]
    val = ((n / Lx) ** 2 + (m / Ly) ** 2 + (p / Lz) ** 2).ravel()
    val = val[(val <= thr) & (val > 1e-14)]
    return np.sort(np.pi ** 2 * val)


def main():
    print(f"{'Room':<8} {'V (m^3)':>8} {'K_total':>8} {'l_min (m)':>10} {'D_ap/l_min':>11} {'max var.':>9}")
    for name, (Lx, Ly, Lz) in ROOMS.items():
        lam = neumann_eigenvalues(Lx, Ly, Lz)
        l_min = 2 * np.pi / np.sqrt(lam[K - 1])
        ratio = D_AP / l_min
        print(f"{name:<8} {Lx * Ly * Lz:>8.1f} {len(lam):>8d} {l_min:>10.3f} "
              f"{100 * ratio:>10.1f}% {100 * 2 * np.sin(np.pi * ratio):>8.1f}%")


if __name__ == "__main__":
    main()
