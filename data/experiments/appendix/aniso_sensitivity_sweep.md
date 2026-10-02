# Anisotropy sensitivity sweep — does D/(2H) excess matter for cost?

Arena: paper's modal dataset (data/modal, 8 mics); estimator = closed-form diagonal Tikhonov, released 61-p x 12-alpha grids, |s| = 1.1265513951 at the exact p column; exact population risk (no MC, no selection noise); prior and truncation weights lambda^-|s| (population); noise synthetic, temporally white (real truncation noise is temporally correlated; at T=1 the distinction vanishes). n = 24 in-scope rooms. All absolute costs are on the paper's dataset.

**Stacking check**: PASS: rows are mic-major (row m*T+t = mic m); R_MT = kron(R, I_T) (MC residual 0.236); kron(I_T, R) rejected (residual 7.110).

Sanity: tr(R) = M*s2 at every a by construction (E_hat traceless — asserted); a = 0 vs the release pipeline solver (ridge_solve on simulated draws, 3000 each):

| room | T | a | P (pipeline MC) | P (formula) | rel diff |
|---|---|---|---|---|---|
| 963 | 1 | 0.0 | 0.8417 | 0.8410 | 0.1% |
| 963 | 1 | 0.58 | 0.8414 | 0.8410 | 0.1% |
| 963 | 100 | 0.0 | 0.6483 | 0.6489 | 0.1% |
| 963 | 100 | 0.58 | 0.6485 | 0.6490 | 0.1% |
| 924 | 1 | 0.0 | 0.8017 | 0.8041 | 0.3% |
| 924 | 1 | 0.58 | 0.8012 | 0.8031 | 0.2% |
| 924 | 100 | 0.0 | 0.6461 | 0.6456 | 0.1% |
| 924 | 100 | 0.58 | 0.6456 | 0.6452 | 0.1% |

F1 per-room PSD limits: median 1.21, range [1.00, 7.00] — cells with a >= limit are skipped (reported n per cell). F2 limit 1, k2 limit 3, F3 limit 7. Empirical markers: a = 0.58 (median), 0.88 (mean), 2.42 (95th pct) of the stored ||E||_op distribution.

## T = 1

### F1_empirical

| a | n rooms | median d_rel | IQR | median d_abs (pp) | median p* |
|---|---|---|---|---|---|
| 0.0 | 24 | 0.01% | [-0.01, 0.03] | 0.009 | 1.10 |
| 0.25 | 24 | 0.01% | [-0.01, 0.03] | 0.006 | 1.10 |
| 0.58 (median) | 24 | 0.01% | [-0.01, 0.04] | 0.005 | 1.10 |
| 0.88 (mean) | 24 | 0.03% | [-0.00, 0.09] | 0.019 | 1.10 |
| 1.5 | 6 | 0.06% | [0.03, 0.09] | 0.046 | 1.10 |
| 2.42 (95th pct) | 4 | 0.13% | [0.01, 0.39] | 0.108 | 1.15 |
| 3.5 | 2 | 0.64% | [0.33, 0.95] | 0.516 | 1.25 |

### F2_balanced

| a | eig ratio | median cost (rand Q) | max-over-sampled-Q (med rooms) | aligned | median p* |
|---|---|---|---|---|---|
| 0.0 | 1.00 | 0.01% | 0.01% | 0.01% | 1.10 |
| 0.25 | 1.67 | 0.01% | 0.03% | 0.02% | 1.10 |
| 0.58 | 3.76 | 0.02% | 0.04% | 0.05% | 1.10 |
| 0.88 | 15.67 | 0.02% | 0.07% | 0.13% | 1.10 |

### k2

| a | eig ratio | median cost (rand Q) | max-over-sampled-Q (med rooms) | aligned | median p* |
|---|---|---|---|---|---|
| 0.0 | 1.00 | 0.01% | 0.01% | 0.01% | 1.10 |
| 0.25 | 1.36 | 0.02% | 0.03% | 0.01% | 1.10 |
| 0.58 | 1.96 | 0.02% | 0.04% | 0.02% | 1.10 |
| 0.88 | 2.66 | 0.02% | 0.04% | 0.01% | 1.10 |
| 1.5 | 5.00 | 0.03% | 0.16% | 0.02% | 1.10 |
| 2.42 | 17.69 | 0.05% | 0.25% | 0.08% | 1.10 |

### F3_spike

| a | eig ratio | median cost (rand Q) | max-over-sampled-Q (med rooms) | aligned | median p* |
|---|---|---|---|---|---|
| 0.0 | 1.00 | 0.01% | 0.01% | 0.01% | 1.10 |
| 0.25 | 1.30 | 0.02% | 0.02% | 0.01% | 1.10 |
| 0.58 | 1.72 | 0.02% | 0.03% | 0.02% | 1.10 |
| 0.88 | 2.15 | 0.02% | 0.04% | 0.02% | 1.10 |
| 1.5 | 3.18 | 0.03% | 0.06% | 0.03% | 1.10 |
| 2.42 | 5.23 | 0.04% | 0.17% | 0.01% | 1.10 |
| 3.5 | 9.00 | 0.06% | 0.27% | 0.02% | 1.10 |

## T = 100

### F1_empirical

| a | n rooms | median d_rel | IQR | median d_abs (pp) | median p* |
|---|---|---|---|---|---|
| 0.0 | 24 | 0.06% | [-0.02, 0.48] | 0.027 | 1.10 |
| 0.25 | 24 | 0.06% | [-0.02, 0.47] | 0.029 | 1.10 |
| 0.58 (median) | 24 | 0.06% | [-0.02, 0.45] | 0.030 | 1.10 |
| 0.88 (mean) | 24 | 0.07% | [-0.02, 0.44] | 0.035 | 1.10 |
| 1.5 | 6 | 0.30% | [0.12, 0.79] | 0.165 | 1.20 |
| 2.42 (95th pct) | 4 | 0.48% | [0.19, 1.42] | 0.280 | 1.15 |
| 3.5 | 2 | 2.03% | [1.12, 2.95] | 1.208 | 1.35 |

### F2_balanced

| a | eig ratio | median cost (rand Q) | max-over-sampled-Q (med rooms) | aligned | median p* |
|---|---|---|---|---|---|
| 0.0 | 1.00 | 0.06% | 0.06% | 0.06% | 1.10 |
| 0.25 | 1.67 | 0.06% | 0.07% | 0.06% | 1.10 |
| 0.58 | 3.76 | 0.05% | 0.08% | 0.07% | 1.10 |
| 0.88 | 15.67 | 0.05% | 0.09% | 0.05% | 1.10 |

### k2

| a | eig ratio | median cost (rand Q) | max-over-sampled-Q (med rooms) | aligned | median p* |
|---|---|---|---|---|---|
| 0.0 | 1.00 | 0.06% | 0.06% | 0.06% | 1.10 |
| 0.25 | 1.36 | 0.07% | 0.07% | 0.07% | 1.10 |
| 0.58 | 1.96 | 0.06% | 0.07% | 0.07% | 1.10 |
| 0.88 | 2.66 | 0.06% | 0.08% | 0.08% | 1.10 |
| 1.5 | 5.00 | 0.06% | 0.09% | 0.07% | 1.10 |
| 2.42 | 17.69 | 0.06% | 0.11% | 0.08% | 1.10 |

### F3_spike

| a | eig ratio | median cost (rand Q) | max-over-sampled-Q (med rooms) | aligned | median p* |
|---|---|---|---|---|---|
| 0.0 | 1.00 | 0.06% | 0.06% | 0.06% | 1.10 |
| 0.25 | 1.30 | 0.06% | 0.07% | 0.05% | 1.10 |
| 0.58 | 1.72 | 0.07% | 0.07% | 0.06% | 1.10 |
| 0.88 | 2.15 | 0.07% | 0.08% | 0.06% | 1.10 |
| 1.5 | 3.18 | 0.07% | 0.09% | 0.06% | 1.10 |
| 2.42 | 5.23 | 0.07% | 0.11% | 0.07% | 1.10 |
| 3.5 | 9.00 | 0.07% | 0.13% | 0.08% | 1.10 |

## T = 1000

### F1_empirical

| a | n rooms | median d_rel | IQR | median d_abs (pp) | median p* |
|---|---|---|---|---|---|
| 0.0 | 24 | 0.81% | [0.59, 1.98] | 0.037 | 1.20 |
| 0.25 | 24 | 0.72% | [0.54, 1.97] | 0.038 | 1.20 |
| 0.58 (median) | 24 | 0.74% | [0.41, 1.94] | 0.039 | 1.20 |
| 0.88 (mean) | 24 | 0.73% | [0.35, 1.81] | 0.040 | 1.20 |
| 1.5 | 6 | 0.63% | [0.16, 0.98] | 0.003 | 1.25 |
| 2.42 (95th pct) | 4 | 1.27% | [0.87, 11.91] | 0.130 | 1.30 |
| 3.5 | 2 | 22.03% | [11.45, 32.61] | 0.458 | 1.45 |

### F2_balanced

| a | eig ratio | median cost (rand Q) | max-over-sampled-Q (med rooms) | aligned | median p* |
|---|---|---|---|---|---|
| 0.0 | 1.00 | 0.81% | 0.81% | 0.81% | 1.20 |
| 0.25 | 1.67 | 0.82% | 0.95% | 0.78% | 1.20 |
| 0.58 | 3.76 | 0.84% | 1.13% | 0.81% | 1.20 |
| 0.88 | 15.67 | 0.89% | 1.31% | 0.72% | 1.20 |

### k2

| a | eig ratio | median cost (rand Q) | max-over-sampled-Q (med rooms) | aligned | median p* |
|---|---|---|---|---|---|
| 0.0 | 1.00 | 0.81% | 0.81% | 0.81% | 1.20 |
| 0.25 | 1.36 | 0.83% | 0.88% | 0.79% | 1.20 |
| 0.58 | 1.96 | 0.83% | 0.98% | 0.83% | 1.20 |
| 0.88 | 2.66 | 0.86% | 1.20% | 0.80% | 1.20 |
| 1.5 | 5.00 | 0.88% | 1.39% | 0.71% | 1.20 |
| 2.42 | 17.69 | 1.02% | 1.77% | 0.78% | 1.20 |

### F3_spike

| a | eig ratio | median cost (rand Q) | max-over-sampled-Q (med rooms) | aligned | median p* |
|---|---|---|---|---|---|
| 0.0 | 1.00 | 0.81% | 0.81% | 0.81% | 1.20 |
| 0.25 | 1.30 | 0.78% | 0.86% | 0.75% | 1.20 |
| 0.58 | 1.72 | 0.82% | 0.93% | 0.75% | 1.20 |
| 0.88 | 2.15 | 0.85% | 1.02% | 0.74% | 1.20 |
| 1.5 | 3.18 | 0.91% | 1.19% | 0.63% | 1.20 |
| 2.42 | 5.23 | 1.01% | 1.65% | 0.52% | 1.20 |
| 3.5 | 9.00 | 0.94% | 2.23% | 0.71% | 1.20 |

## Local perturbation fit (empirical, NOT a bound, NOT C_room)

| T | quantity | C_emp (per unit a^2) | R^2 | fit range |
|---|---|---|---|---|
| 1 | d_rel (rel) | +0.020 | 0.637 | a in [0, 0.88] |
| 1 | d_abs (pp) | +0.014 | 0.592 | a in [0, 0.88] |
| 100 | d_rel (rel) | +0.015 | 0.926 | a in [0, 0.88] |
| 100 | d_abs (pp) | +0.009 | 0.988 | a in [0, 0.88] |
| 1000 | d_rel (rel) | -0.048 | 0.198 | a in [0, 0.88] |
| 1000 | d_abs (pp) | +0.003 | 0.786 | a in [0, 0.88] |

## Money calculation — implied ||E||_op and cost, Berry vs measured D

Using E||E||_F^2 = M(M-1)H + M*D at M=8 and the MEASURED Frobenius-relaxation factor ||E||_op^2/pred = 0.513 (uniform-sensor MC, hd_decomposition.md) — this makes the implied ||E||_op an EMPIRICAL estimate, not a bound. Cost read off the F1 sweep median curve (paper dataset) at each T by interpolation in a.

| domain | H | D/(2H) | implied op (Berry) | implied op (measured) | cost@Berry T=1 | cost@meas T=1 | cost@Berry T=100 | cost@meas T=100 | cost@Berry T=1000 | cost@meas T=1000 |
|---|---|---|---|---|---|---|---|---|---|---|
| generic polygons | 0.0050 | 2.1 | 0.43 | 0.48 | 0.01% | 0.01% | 0.06% | 0.06% | 0.73% | 0.74% |
| App-G rectangles | 0.0042 | 7.0 | 0.39 | 0.60 | 0.01% | 0.01% | 0.06% | 0.06% | 0.73% | 0.74% |
| 3D boxes (small) | 0.0089 | 15.4 | 0.57 | 1.18 | 0.01% | 0.04% | 0.06% | 0.18% | 0.74% | 0.68% |
| 3D boxes (large) | 0.0011 | 75.8 | 0.20 | 0.85 | 0.01% | 0.02% | 0.06% | 0.07% | 0.74% | 0.73% |

Reading: 'moving from Berry's D = 2H to the measured D changes the implied anisotropy from X to Y and the cost of the fixed exponent from Z% to W%' — fill from the table above; the separable-domain rows use their own H so implied op values are those domains' own scales.
