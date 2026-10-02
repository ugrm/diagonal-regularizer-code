# Trace-free Frobenius identity

M = 8, 800 sensor draws/room, 30-room MC subset, exact degree-4 quadrature for D. Identity (C) is asserted numerically inside every draw (max dev < 1e-10).

## Analytic derivation (confirmed by the MC below)

`||E_tf||_F^2 = ||E^pop||_F^2 - M*dbar^2` and `E[dbar^2] = D/M` for i.i.d. sensors, so

```
E||E_tf||_F^2 = M(M-1)H + M*D - D = (M-1)(M*H + D)
```

and `R_trunc = (S/|Omega|)(E^pop + I)` gives `E^paper = (E^pop+I)/(1+dbar) - I = E_tf/(1+dbar)` exactly.

## (A) and (B): empirical / predicted, median [IQR] over 30 rooms

| weights | (A) ||E^pop||_F^2 ratio | (B) ||E_tf||_F^2 ratio | sd(dbar) measured / sqrt(D/M) |
|---|---|---|---|
| population lambda^-|s| | 1.001 [0.985, 1.023] | **1.000** [0.983, 1.023] | 0.0657 / 0.0655 |
| realized var(a) | 1.000 [0.989, 1.019] | **0.999** [0.988, 1.017] | 0.0757 / 0.0779 |

## Bound comparison, all 187 submitted in-scope rooms

| quantity | median | IQR |
|---|---|---|
| empirical ||E^paper||_op (stored) | 0.5804 | [0.3907, 1.0906] |
| paper's printed form sqrt(M(M+1)H) | 0.6017 | [0.4886, 0.8522] |
| Berry trace-free sqrt((M-1)(M*H+2H)) | 0.5933 | [0.4817, 0.8402] |
| corrected trace-free sqrt((M-1)(M*H+D)) | 0.6741 | [0.5433, 0.9427] |

- fraction of rooms with empirical ||E||_op **below** the corrected bound: **0.647** (121/187)
- fraction below the Berry trace-free bound: 0.540 (101/187)
- fraction below the paper's printed sqrt(M(M+1)H): 0.572
- median slack, corrected bound vs empirical: 1.161x (paper's printed form: 1.037x)
- median per-room ratio empirical/corrected: 0.843

## Trace-free relaxation ratio (analogue of the 0.513)

- population weights: median ||E_tf||_op^2 / [(M-1)(M*H+D)] = **0.497** [0.467, 0.537]
- realized weights: median ||E_tf||_op^2 / [(M-1)(M*H+D)] = **0.501** [0.474, 0.528]

## n for D/(2H)

- as reported in hd_decomposition.md: n = 189 (v2 in-scope, K_total > 50), median **2.148**
- on exactly the 187 submitted in-scope rooms: n = 187, median **2.155** [1.70, 2.72]
