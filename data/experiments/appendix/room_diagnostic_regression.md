# Per-room diagnostic regression - report

|s| = 1.1266; rooms n=197, in-scope=187, B.4-eligible=168

## Anchor validation (paper-published values)

- A1 median delta(T=1000), in-scope: **5.83%** (paper 5.82%)
- A2 rho(E_op, delta1000): **-0.30** (p=3.5e-05, n=187) (paper -0.30)
- A3 rho(KS D, delta1000), B.4 subset: **+0.19** (p=0.012, n=168) (paper +0.22, p=0.005, n=168)
- A4 rho(KS D, N_trunc), in-scope: **+0.29** (paper +0.29)
- A5 per-room KS D (chi2 variant): median **0.061** IQR [0.046,0.073] max 0.197 (paper 0.042 [0.031,0.058] max 0.12)
- A5b per-room KS D (signed vs N(0,1)): median **0.054** IQR [0.042,0.066] max 0.205; rho(D_norm, delta1000)=-0.00 (p=0.967, n=168)
- A6 H median (in-scope): **0.0050** (paper ~0.005); E_op median: **0.58** (paper 0.58)

## Head-to-head: Spearman rho(diagnostic, delta_rel(T)), in-scope rooms (n=187)

| diagnostic | T=1 | T=50 | T=100 | T=500 | T=1000 | T=2100 |
|---|---|---|---|---|---|---|
| H (Herfindahl) | -0.06 | **-0.21** | -0.10 | -0.04 | **-0.42** | -0.11 |
| ||E||_op | -0.13 | **-0.31** | **-0.21** | +0.01 | **-0.30** | -0.04 |
| Berry KS D | *-0.15* | -0.05 | -0.09 | +0.09 | *+0.16* | +0.06 |
| dyn range (lamK/lam1)^s | -0.09 | -0.12 | *-0.18* | **+0.26** | -0.04 | +0.11 |
| K_total | +0.06 | **+0.21** | +0.10 | +0.04 | **+0.42** | +0.12 |
| area | +0.06 | **+0.21** | +0.10 | +0.04 | **+0.42** | +0.11 |

(** p<0.01, * p<0.05)

## Landscape-tier (uses the P(p) sweep): Spearman rho with delta_rel(T)

| diagnostic | T=1 | T=50 | T=100 | T=500 | T=1000 | T=2100 |
|---|---|---|---|---|---|---|
| flatness maxP/minP | **+0.21** | **+0.37** | **+0.34** | **+0.21** | -0.03 | +0.07 |
| curvature P''(p*)/P(p*) | **-0.21** | **-0.34** | -0.09 | +0.04 | -0.01 | **+0.52** |
| drift |p*-|s|| | **+0.90** | **+0.90** | **+0.80** | **+0.36** | **+0.37** | **+0.32** |

## Mechanistic decomposition: delta ~ 0.5 P''(p*) (p*-|s|)^2 / P(p*)

| T | Spearman(taylor, delta) | Pearson log-log R2 | n |
|---|---|---|---|
| 1 | +0.93 | 0.87 | 187 |
| 50 | +0.92 | 0.78 | 187 |
| 100 | +0.85 | 0.75 | 187 |
| 500 | +0.39 | 0.31 | 187 |
| 1000 | +0.37 | 0.20 | 186 |
| 2100 | +0.51 | 0.69 | 184 |

## Partial Spearman (controlling for room area), delta_rel(T=1000), in-scope

| diagnostic | raw rho | partial rho | area |
|---|---|---|---|
| H (Herfindahl) | -0.42 | -0.08 (p=2.6e-01) | controlled |
| ||E||_op | -0.30 | +0.10 (p=1.7e-01) | controlled |
| Berry KS D | +0.16 | +0.05 (p=5.1e-01) | controlled |
| dyn range (lamK/lam1)^s | -0.04 | -0.03 (p=7.0e-01) | controlled |
| K_total | +0.42 | -0.04 (p=6.0e-01) | controlled |

## Multivariate (T=1000, in-scope, physics+noise diagnostics)

Rank-OLS on standardized ranks, n=187, R2=0.20. Betas:
- H: -3.84
- E_op: +0.17
- ks_D: +0.03
- dyn_range: +0.01
- K_total: -2.31
- area: -0.97

Random forest (as in D.3): in-sample R2=0.52, 5-fold CV R2=0.06+-0.14. Permutation importances:
- H: 0.240 +- 0.037
- area: 0.137 +- 0.018
- E_op: 0.125 +- 0.016
- dyn_range: 0.105 +- 0.015
- ks_D: 0.089 +- 0.009
- K_total: 0.088 +- 0.012

### Out-of-sample predictability of delta_rel at every T (RF, 5-fold CV R2)

| T | CV R2 (physics+noise features) | median delta_rel |
|---|---|---|
| 1 | -0.17 +- 0.33 | 0.57% |
| 50 | -0.11 +- 0.23 | 0.61% |
| 100 | -0.03 +- 0.09 | 1.46% |
| 500 | +0.13 +- 0.09 | 5.18% |
| 1000 | +0.06 +- 0.14 | 5.83% |
| 2100 | -0.35 +- 0.62 | 3.19% |

## Diagnostic inter-correlations (Spearman, in-scope)

- rho(area, K_total) = +1.00
- rho(H, K_total) = -1.00
- rho(E_op, K_total) = -0.83
- rho(E_op, H) = +0.83
- rho(ks_D, K_total) = +0.29
- rho(dyn_range, area) = -0.02
