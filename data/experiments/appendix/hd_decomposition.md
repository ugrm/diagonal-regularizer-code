# H+D anisotropy decomposition (discarded band n > 50)

|s| = 1.1265513951 (canonical; using 1.13 instead changes H by 0.464% on room_00801). psi = sqrt(|Omega|) phi exactly (ARPACK mass-orthonormality), so E_x[V] = 1 and E_x[psi_n psi_m] = delta_nm are analytic identities. S, C, D use a degree-4 (6-point) triangle rule that is EXACT for the piecewise-quartic integrands; E[V] = 1 is reproduced to max |E[V]-1| = 1.8e-09 across rooms, which bounds the total numerical error. C4 for the 2.89-reconciliation uses the published lumped path. Two quadrature traps found and documented below: per-mode lumped normalization corrupts the sensor identity (E[E_ij^2] -> 0.81 H), and lumped integration inflates D by ~1.5x.

## Arena check

- This run: v2 eigensolve (`data/rooms/`), in-scope (K_total>50) n = 189. Median K_total = 334; median H = 0.0050.
- diagnostic table (eigensolve of the dataset) in-scope n = 187: median K_total = 337, median H = 0.0050 (table s_value = 1.1265513951271393).
- Common in-scope rooms n = 187: max |H_v2 - H_legacy| = 1.18e-02, median 2.17e-18 — the eigensolves agree; only near-cutoff mask tails differ.

## Per-domain decomposition (medians [IQR])

| domain | n | H | S/(2H) | C/(2H) | D/(2H) |
|---|---|---|---|---|---|
| generic polygons (v2) | 189 | 0.0050 [0.0033, 0.0101] | 0.877 [0.859, 0.897] | 1.267 [0.819, 1.831] | 2.148 [1.695, 2.708] |
| rectangles (Dirichlet, 1000 Hz) | 4 | 0.0042 [0.0039, 0.0045] | 0.625 [0.625, 0.625] | 6.411 [6.034, 7.012] | 7.036 [6.659, 7.637] |
| 3D boxes (Dirichlet, 500 Hz) | 4 | 0.0035 [0.0011, 0.0089] | 1.188 [1.188, 1.188] | 33.315 [22.394, 49.780] | 34.503 [23.581, 50.967] |
| 3D boxes (Neumann, 500 Hz) | 4 | 0.0025 [0.0011, 0.0048] | 0.926 [0.921, 0.930] | 33.809 [26.064, 44.442] | 34.735 [26.990, 45.368] |

Per-domain detail (analytic):

- rect 3x6_rect: K_total=455 H=0.003857 S/2H=0.625 C/2H=6.780 D/2H=7.405 C4_med=2.250
- rect 2x8_long: K_total=397 H=0.004237 S/2H=0.625 C/2H=7.707 D/2H=8.332 C4_med=2.250
- rect 4x4_square: K_total=402 H=0.004237 S/2H=0.625 C/2H=6.007 D/2H=6.632 C4_med=2.250
- rect 3x5_rect: K_total=375 H=0.004466 S/2H=0.625 C/2H=6.043 D/2H=6.668 C4_med=2.250
- box-D small_2.5x3x2.4: K_total=169 H=0.008856 S/2H=1.188 C/2H=14.170 D/2H=15.357 C4_med=3.375
- box-D mid_3x4x2.5: K_total=298 H=0.004521 S/2H=1.188 C/2H=25.135 D/2H=26.322 C4_med=3.375
- box-D mid_4x5x2.7: K_total=566 H=0.002387 S/2H=1.188 C/2H=41.496 D/2H=42.683 C4_med=3.375
- box-D large_6x7x3: K_total=1380 H=0.001106 S/2H=1.188 C/2H=74.631 D/2H=75.818 C4_med=3.375
- box-N small_2.5x3x2.4: K_total=304 H=0.004752 S/2H=0.916 C/2H=19.700 D/2H=20.616 C4_med=3.375
- box-N mid_3x4x2.5: K_total=493 H=0.003042 S/2H=0.929 C/2H=28.185 D/2H=29.114 C4_med=3.375
- box-N mid_4x5x2.7: K_total=861 H=0.001922 S/2H=0.923 C/2H=39.432 D/2H=40.355 C4_med=3.375
- box-N large_6x7x3: K_total=1921 H=0.001083 S/2H=0.932 C/2H=59.472 D/2H=60.404 C4_med=3.375

## Reconciling the published C4 = 2.89

S/(2H) three ways (generic rooms, median over rooms):
- (i) scalar median C4 (2.89-style): median of (C4_med-1)*H/(2H) = 0.920
- (ii) scalar 95th C4: 1.008
- (iii) actual w_n^2-weighted sum_n w_n^2(C4_n-1)/(2H): 0.877
Median per-room C4 over rooms: 2.840 (published 2.89); median w^2-weighted C4: 2.755.

## Sensor identity check (population-normalized E; 500 uniform draws (V kurtosis ~12: per-room diag-ratio sem ~0.05), M = 8)

| room | pred = M(M-1)H+MD | mean ||E||_F^2 | total ratio | offdiag/(M(M-1)H) | diag/(MD) | ||E||_op^2/pred | selfnorm ||E||_F^2/pred | max |tr selfnorm| |
|---|---|---|---|---|---|---|---|---|
| 801 | 0.182 | 0.173 | 0.951 | 0.928 | 1.012 | 0.492 | 0.907 | 1.8e-15 |
| 811 | 0.709 | 0.698 | 0.983 | 0.953 | 1.017 | 0.537 | 0.875 | 1.7e-15 |
| 817 | 3.638 | 4.064 | 1.117 | 1.009 | 1.182 | 0.711 | 0.756 | 1.6e-15 |
| 820 | 0.949 | 0.923 | 0.972 | 0.957 | 0.994 | 0.495 | 0.898 | 1.9e-15 |
| 830 | 0.354 | 0.347 | 0.979 | 0.963 | 0.995 | 0.590 | 0.874 | 2.0e-15 |
| 835 | 0.214 | 0.205 | 0.959 | 0.905 | 1.060 | 0.539 | 0.900 | 1.8e-15 |
| 840 | 0.633 | 0.672 | 1.061 | 1.084 | 1.012 | 0.517 | 0.993 | 1.8e-15 |
| 849 | 0.312 | 0.332 | 1.063 | 1.088 | 1.033 | 0.643 | 0.943 | 1.9e-15 |
| 858 | 0.519 | 0.513 | 0.989 | 1.009 | 0.969 | 0.583 | 0.886 | 1.8e-15 |
| 860 | 0.309 | 0.284 | 0.919 | 0.920 | 0.918 | 0.504 | 0.835 | 1.7e-15 |
| 868 | 0.773 | 0.756 | 0.978 | 0.962 | 1.004 | 0.502 | 0.893 | 1.7e-15 |
| 877 | 0.287 | 0.282 | 0.982 | 0.969 | 1.007 | 0.505 | 0.935 | 1.9e-15 |
| 886 | 0.657 | 0.665 | 1.011 | 1.084 | 0.950 | 0.583 | 0.882 | 1.8e-15 |
| 890 | 6.287 | 6.288 | 1.000 | 1.031 | 0.946 | 0.541 | 0.890 | 1.8e-15 |
| 895 | 0.338 | 0.356 | 1.053 | 1.113 | 0.956 | 0.580 | 0.984 | 1.8e-15 |
| 900 | 0.202 | 0.199 | 0.985 | 1.000 | 0.950 | 0.529 | 0.950 | 2.0e-15 |
| 902 | 70.064 | 68.655 | 0.980 | 0.979 | 0.983 | 0.880 | 0.799 | 2.2e-15 |
| 904 | 1.127 | 1.099 | 0.975 | 0.970 | 0.983 | 0.481 | 0.892 | 1.8e-15 |
| 915 | 0.755 | 0.714 | 0.945 | 0.914 | 1.021 | 0.429 | 0.894 | 1.8e-15 |
| 924 | 0.361 | 0.321 | 0.889 | 0.904 | 0.867 | 0.495 | 0.819 | 1.4e-15 |
| 925 | 0.905 | 0.949 | 1.049 | 1.083 | 0.991 | 0.498 | 0.997 | 1.6e-15 |
| 929 | 2.446 | 2.106 | 0.861 | 0.951 | 0.773 | 0.417 | 0.785 | 1.8e-15 |
| 934 | 0.707 | 0.663 | 0.937 | 0.993 | 0.861 | 0.480 | 0.879 | 1.8e-15 |
| 943 | 1.221 | 1.224 | 1.003 | 1.020 | 0.966 | 0.463 | 0.956 | 1.7e-15 |
| 951 | 3.765 | 3.758 | 0.998 | 0.992 | 1.018 | 0.473 | 0.939 | 1.7e-15 |
| 952 | 0.374 | 0.336 | 0.897 | 0.869 | 0.923 | 0.573 | 0.793 | 2.0e-15 |
| 961 | 1.194 | 1.183 | 0.990 | 0.968 | 1.023 | 0.509 | 0.877 | 1.9e-15 |
| 963 | 0.375 | 0.371 | 0.988 | 0.964 | 1.029 | 0.531 | 0.930 | 1.9e-15 |
| 970 | 0.281 | 0.285 | 1.014 | 1.030 | 0.989 | 0.552 | 0.933 | 2.0e-15 |
| 979 | 0.972 | 0.954 | 0.982 | 1.051 | 0.912 | 0.485 | 0.881 | 1.8e-15 |
| 989 | 1.027 | 1.021 | 0.994 | 0.969 | 1.052 | 0.453 | 0.937 | 1.7e-15 |
| 999 | 0.334 | 0.351 | 1.050 | 1.064 | 1.034 | 0.631 | 0.947 | 1.8e-15 |

Across MC rooms (medians): total ||E||_F^2/pred 0.984; off-diagonal mass / M(M-1)H = 0.974 (exact identity, hypothesis-free — MC noise only); diagonal mass / MD = 0.992 (tests the quadrature-D against sampled Var[V]); ||E||_op^2/pred = 0.513 (looseness of the Frobenius relaxation); self-normalized Frobenius mass / pred = 0.893 (the O(sqrt(H)) absorption). tr(E)=0 holds for the SELF-normalized convention only (max |tr| 2.2e-15); population-normalized tr(E) fluctuates with sd ~ sqrt(M D).

## Quadrature audit (lumped vs exact P1)

- C4 median, lumped vs exact-quartic: median |diff| = 0.0016, max = 0.0269 — the published lumped ratio is safe for C4 (both moments share the bias).
- S term, lumped vs exact: median rel |diff| = 0.000, max = 0.012.
- **Lumped quadrature is NOT safe for D**: nodal-lumped integration of squared fields exceeds the exact P1 integral by a gradient-energy term that grows with lambda (measured per-mode | |Omega| m2_lumped - 1 | median 3.62e-02), and it inflated Var[V] by ~1.5x in the first implementation. All S, C, D, C4_ex values in this report therefore use the exact degree-4 rule; E[V] = 1 is reproduced to max |E[V]-1| = 1.8e-09 across all rooms (analytic identity, so this bounds the total numerical error).
- Consistent-mass m2 spot check (3 rooms, small/anchor/large): max | |Omega| m2_M - 1 | = room_00902: 1.5e-09, room_00802: 1.6e-09 — ARPACK mass-orthonormality at f32 storage precision.

## Targeted rooms

| room | why | H | D/(2H) | C/(2H) | stored ||E||_op |
|---|---|---|---|---|---|
| 963 | worst-Berry B.4 | 0.0042 | 2.036 | 1.181 | 0.48 |
| 924 | worst-Berry B.4 | 0.0037 | 2.538 | 1.675 | 0.42 |
| 860 | worst-Berry B.4 | 0.0033 | 2.304 | 1.442 | 0.35 |
| 835 | worst-Berry B.4 | 0.0025 | 1.884 | 1.030 | 0.31 |
| 900 | worst-Berry B.4 | 0.0026 | 1.444 | 0.589 | 0.36 |
| 951 | top-5 stored E_op | 0.0507 | 1.144 | 0.218 | 3.06 |
| 890 | top-5 stored E_op | 0.0719 | 1.965 | 1.046 | 3.22 |
| 929 | top-5 stored E_op | 0.0216 | 3.594 | 2.640 | 3.51 |
| 817 | top-5 stored E_op | 0.0244 | 5.818 | 4.727 | 5.11 |
| 902 | top-5 stored E_op | 1.0000 | 0.879 | -0.000 | 7.00 |

Spearman(D/(2H), stored ||E||_op) = -0.16; Spearman(D/(2H), KS D) = -0.32 (n=187).

Rooms with D/(2H) > 1.5: 160: room_00817 (5.82), room_00919 (5.21), room_00872 (4.62), room_00896 (4.24), room_00886 (4.16), room_00987 (4.12), room_00907 (3.97), room_00952 (3.87), room_00865 (3.80), room_00932 (3.69), room_00858 (3.67), room_00824 (3.64), room_00885 (3.62), room_00929 (3.59), room_00909 (3.54)
