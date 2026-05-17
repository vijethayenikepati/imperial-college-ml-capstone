# Bayesian Optimisation — Function Context Reference

> Imperial College ML Capstone | Updated after Week 7 submissions
> All functions: Maximise Y | Input space: [0, 1]^D

---

## W6 Results — 6 of 8 New All-Time Bests

| Fn | W6 input | W6 Y | Previous best | New best? | Lift |
|----|----------|------|---------------|-----------|------|
| F1 | [0.647, 0.645] | **+0.4428** | \|Y\|=0.0128 | **Yes** | 35× |
| F2 | [0.800, 0.900] | +0.1356 | +0.611 | No | — |
| F3 | [0.582, 0.512, 0.436] | **−0.00133** | −0.0348 | **Yes** | 26× |
| F4 | [0.529, 0.471, 0.350, 0.200] | −4.6880 | −4.026 | No | narrow miss |
| F5 | [0.153, 1.0, 1.0, 1.0] | **+4443.3** | +3531 | **Yes** | +26% |
| F6 | [0.528, 0.234, 0.728, 0.790, 0.011] | **−0.4816** | −0.557 | **Yes** | +13% |
| F7 | [0.150, 0.484, 0.412, 0.176, 0.296, 0.745] | **+1.8882** | +1.365 | **Yes** | +38% |
| F8 | [0.000, 0.008, 0.000, 0.007, 0.949, 0.990, 0.034, 1.000] | +9.5656 | +9.598 | No | within noise |

---

## F1 — Radiation Contamination Detection (2D)

**INITIAL_SIZE**: 10 | **Data**: `function_1/` | **GP (tuned)**: log(|Y|) fit, **ls=0.212** (MLE), alpha=1e-10

### Best known
| Source | X* | Y* |
|--------|----|----|
| **W6 result** | **[0.646970, 0.644949]** | **+0.4428** (|Y|=0.4428) |

*Objective is to maximise |Y| — strong positive readings are valid signals.*

### Weekly history
| Week | Input | Output | Note |
|------|-------|--------|------|
| W1 | [0.591837, 0.591837] | +2.82e-04 | First signal in hot zone |
| W2 | [0.0, 1.0] | 0.0 | Boundary — dead |
| W3 | [1.0, 1.0] | +1.51e-192 | Corner — dead |
| W4 | [0.639955, 0.673176] | −0.01281 | NN grad ascent |
| W5 | [0.001256, 0.001020] | +6.01e-247 | GP EI+SVM sent to dead corner |
| W6 | [0.646970, 0.644949] | **+0.4428** | GP UCB exploit — breakthrough |
| **W7** | **[0.659153, 0.612535]** | *(planned)* | MLE-tuned ls=0.21, UCB on ±0.04 grid |

### Key patterns
- Hot zone is **[0.55, 0.75]^2** — confirmed multiple times
- W4 and W6 are 0.029 apart but Y differs by 0.46 → **steep ridge** in this region
- MLE-tuned length_scale = 0.212 (vs fixed 0.1) — peak is wider than originally assumed
- Sign varies across the ridge: W4 raw Y = −0.0128, W6 raw Y = +0.443

### W8 strategy hint
If W7 at [0.659, 0.613] gives strong |Y|, the ridge extends down-and-right of W6 → tighten further. If it weakens, W6 is the local maximum → ±0.02 random search around W6.

---

## F2 — Noisy Chemical Process (2D)

**INITIAL_SIZE**: 10 | **Data**: `function_2/` | **GP**: raw Y fit, ls=0.1 (fixed), alpha=1e-4

### Best known
| Source | X* | Y* |
|--------|----|----|
| Initial data | [0.702637, 0.926564] | **+0.6112** |
| W2 (best weekly) | [1.0, 0.326531] | +0.1493 |
| W6 | [0.800, 0.900] | +0.1356 |

### Weekly history
| Week | Input | Output | Note |
|------|-------|--------|------|
| W1 | [0.0, 1.0] | +0.0249 | Boundary |
| W2 | [1.0, 0.326531] | +0.1493 | Best weekly so far |
| W3 | [1.0, 1.0] | +0.0364 | Corner degraded |
| W4 | [0.998382, 0.0004] | −0.0548 | Dead zone |
| W5 | [0.001256, 0.001021] | +0.0325 | SVM corner failure |
| W6 | [0.800, 0.900] | +0.1356 | Drifted toward initial best — first weekly probe of x2≈0.9 |
| **W7** | **[0.663558, 0.945632]** | *(planned)* | UCB ±0.05 around initial best [0.703, 0.927] |

### Key patterns
- Initial best at [0.703, 0.927] with Y=0.611 — has never been beaten or directly retested
- F2 is noisy (alpha=1e-4 in GP, suspect higher in truth) — initial best may be optimistic noise
- W6 confirmed the x2 ≈ 0.9 corridor is more productive than the x2 ≈ 0.15-0.5 region weekly queries focused on

### W8 strategy hint
W7 directly retests near the initial best. If Y > 0.4, initial best is real → tighten ±0.02. If Y < 0.2, initial best was noise → re-anchor on best of weekly queries (currently W2 at 0.149).

---

## F3 — Drug Discovery (3D)

**INITIAL_SIZE**: 15 | **Data**: `function_3/` | **GP**: raw Y fit, ls=0.3 (fixed), alpha=1e-6

### Best known
| Source | X* | Y* |
|--------|----|----|
| **W6 result** | **[0.582160, 0.512349, 0.435542]** | **−0.00133** |
| W5 (prior best weekly) | [0.524514, 0.731593, 0.220176] | −0.11825 |

*All Y values are negative; maximise = find least negative.*

### Weekly history
| Week | Input | Output | Note |
|------|-------|--------|------|
| W1 | [0.421053, 1.0, 1.0] | −0.47623 | High x3 — bad |
| W2 | [1.0, 0.0, 0.684211] | −0.19554 | |
| W3 | [0.0, 1.0, 0.684211] | −0.16026 | |
| W4 | [0.692581, 0.411593, 0.194985] | −0.12553 | |
| W5 | [0.524514, 0.731593, 0.220176] | −0.11825 | |
| W6 | [0.582160, 0.512349, 0.435542] | **−0.00133** | BREAKTHROUGH — 26× closer to zero |
| **W7** | **[0.542768, 0.479326, 0.475361]** | *(planned)* | UCB ±0.04 around W6 |

### Key patterns
- **x3 myth busted**: W6 has x3=0.436, NOT the x3→0 the earlier trend predicted. The optimum has intermediate x3
- W5 at x3=0.22 → −0.118; W6 at x3=0.44 → −0.001. The true x3 sweet spot is ~0.40–0.45
- Productive region: x1≈[0.50–0.60], x2≈[0.48–0.55], x3≈[0.40–0.48]

### W8 strategy hint
W7 probes [0.543, 0.479, 0.475]. If Y > −0.001 (positive!), the peak crosses zero — tighten further. Otherwise, ±0.03 around W6 [0.582, 0.512, 0.436].

---

## F4 — Warehouse Placement (4D)

**INITIAL_SIZE**: 30 | **Data**: `function_4/` | **GP (tuned)**: raw Y fit, **ARD** (per-dim ls), alpha=1e-4

### Best known
| Source | X* | Y* |
|--------|----|----|
| Initial data | [0.577766, 0.428772, 0.425826, 0.249007] | **−4.0255** |
| W5 (best weekly) | [0.629449, 0.425195, 0.523474, 0.108441] | −8.934 |
| W6 | [0.528947, 0.471053, 0.350000, 0.200000] | −4.6880 |

### Weekly history
| Week | Input | Output | Note |
|------|-------|--------|------|
| W1 | [0.889, 0.556, 0.778, 0.778] | −24.548 | Corners bad |
| W2 | [1.0, 0.667, 0.111, 0.778] | −28.564 | |
| W3 | [1.0, 0.667, 0.111, 0.778] | −28.564 | W2 duplicate |
| W4 | [1.0, 1.0, 1.0, 0.0] | −48.000 | Corner catastrophe |
| W5 | [0.629449, 0.425195, 0.523474, 0.108441] | −8.934 | |
| W6 | [0.528947, 0.471053, 0.350000, 0.200000] | −4.6880 | Narrowly misses initial best |
| **W7** | **[0.497320, 0.428456, 0.397541, 0.283425]** | *(planned)* | ARD-GP UCB ±0.06 around (W6+initial)/2 |

### Key patterns
- Productive basin centred near [0.55, 0.45, 0.40, 0.23]
- W6's pull-back toward this region gave −4.69 (closest weekly to initial best)
- ARD GP fit shows mild anisotropy — x1 and x4 length_scales differ from x2/x3

### W8 strategy hint
If W7 beats −4.0, tighten ±0.03 around it. If not, the basin is well-defined; try x4 in [0.25, 0.30] which is unexplored between W6 (0.20) and initial best (0.249).

---

## F5 — Chemical Yield Optimisation (4D)

**INITIAL_SIZE**: 20 | **Data**: `function_5/` | **GP (tuned)**: raw Y fit, **Matern-1.5** kernel (RBF was strongly misspecified — log-marg-lik improved by 1100+ with Matern)

### Best known
| Source | X* | Y* |
|--------|----|----|
| **W6 result** | **[0.152632, 1.000000, 1.000000, 1.000000]** | **+4443.3** |
| W4 | [0.133474, 0.977830, 0.979582, 0.970578] | +3531.1 |

### Weekly history
| Week | Input | Output | Note |
|------|-------|--------|------|
| W1 | [0.209, 0.839, 0.859, 0.882] | 984.4 | |
| W2 | [0.205, 0.878, 0.880, 0.871] | 1192.3 | |
| W3 | [0.205, 0.878, 0.880, 0.871] | 1192.3 | W2 duplicate |
| W4 | [0.133, 0.978, 0.980, 0.971] | 3531.1 | |
| W5 | [0.155, 0.906, 0.958, 0.907] | 2081.3 | Boundary regression |
| W6 | [0.153, 1.0, 1.0, 1.0] | **4443.3** | Hard boundary confirmed optimal |
| **W7** | **[0.130000, 1.000000, 1.000000, 1.000000]** | *(planned)* | x1 scan with boundary fixed (between W4 0.133 and W6 0.153) |

### Key patterns
- **Hard boundary x2=x3=x4=1.0 is optimal** — confirmed by W6's 4443 vs W4's 3531
- x1 trend with boundary off: smaller is better (0.209→0.205→0.155→0.133, Y rising)
- x1=0.153 with boundary on gave 4443 — but this is the only boundary-active sample
- **RBF kernel was wildly misspecified** for F5's boundary saturation; Matern-1.5 fits much better (log-marg-lik +1100)

### W8 strategy hint
If W7 at x1=0.130 exceeds 4443, smaller x1 is still better → try x1=0.10. If it regresses, x1∈[0.13, 0.17] is the sweet spot at boundary → exploit ±0.01.

---

## F6 — Environmental Monitoring (5D)

**INITIAL_SIZE**: 20 | **Data**: `function_6/` | **GP (tuned)**: raw Y fit, **ARD**, alpha=1e-3

### Best known
| Source | X* | Y* |
|--------|----|----|
| **W6 result** | **[0.527881, 0.234359, 0.727980, 0.790290, 0.010640]** | **−0.4816** |
| W3 (prior best) | [0.606716, 0.166787, 0.797421, 0.720111, 0.086136] | −0.5573 |

### Weekly history
| Week | Input | Output | Note |
|------|-------|--------|------|
| W1 | [0.705, 0.105, 0.764, 0.784, 0.051] | −0.696 | |
| W2 | [0.770, 0.188, 0.823, 0.739, 0.068] | −0.755 | |
| W3 | [0.607, 0.167, 0.797, 0.720, 0.086] | −0.557 | |
| W4 | [0.407, 0.005, 0.997, 0.920, 0.0] | −0.982 | Extremes punished |
| W5 | [0.658, 0.163, 0.895, 0.580, 0.179] | −0.914 | |
| W6 | [0.528, 0.234, 0.728, 0.790, 0.011] | **−0.4816** | New best |
| **W7** | **[0.503189, 0.259871, 0.688609, 0.828287, 0.000000]** | *(planned)* | ARD-GP UCB ±0.04 around W6 |

### Key patterns
- Interior optimum confirmed (Goldilocks)
- x5 near 0 is fine (W6 x5=0.011 → best; W4 x5=0.0 was bad due to OTHER extremes)
- x1 ≈ 0.50–0.62, x2 ≈ 0.17–0.25, x3 ≈ 0.70–0.80, x4 ≈ 0.72–0.83 is the productive region

### W8 strategy hint
W7 sits near the lower-x1 / lower-x3 corner of the productive region. If it improves, push x1 down further. If not, exploit ±0.02 around W6.

---

## F7 — ML Hyperparameter Tuning (6D)

**INITIAL_SIZE**: 30 | **Data**: `function_7/` | **GP (tuned)**: raw Y fit, **ARD** — **x2 and x3 are IRRELEVANT**

### ARD-fit length_scales (36 points)
| Dim | length_scale | 1/ls (importance) | Verdict |
|-----|--------------|-------------------|---------|
| x1 | 0.50 | 2.00 | Moderate |
| **x2** | **10.0** (bound) | **0.10** | **Irrelevant** |
| **x3** | **10.0** (bound) | **0.10** | **Irrelevant** |
| x4 | 0.18 | 5.69 | **Most important** |
| x5 | 0.20 | 4.89 | **Important** |
| x6 | 0.55 | 1.83 | Moderate |

### Best known
| Source | X* | Y* |
|--------|----|----|
| **W6 result** | **[0.149588, 0.483875, 0.411552, 0.175677, 0.296274, 0.745243]** | **+1.8882** |
| Initial data | [0.057896, 0.491672, 0.247422, 0.218118, 0.420428, 0.730970] | +1.365 |

### Weekly history
| Week | Input | Output | Note |
|------|-------|--------|------|
| W1 | [0.061, 0.495, 0.127, 0.142, 0.444, 0.875] | 0.679 | x6 too high |
| W2 | [0.071, 0.492, 0.233, 0.257, 0.537, 0.628] | 0.775 | |
| W3 | [0.074, 0.561, 0.359, 0.097, 0.306, 0.674] | 1.236 | |
| W4 | [0.0, 0.351, 0.312, 0.068, 0.270, 0.881] | 0.998 | |
| W5 | [0.014, 0.568, 0.309, 0.135, 0.366, 0.698] | 1.131 | |
| W6 | [0.150, 0.484, 0.412, 0.176, 0.296, 0.745] | **1.888** | BREAKTHROUGH |
| **W7** | **[0.121273, 0.478154, 0.500852, 0.195523, 0.279358, 0.735707]** | *(planned)* | ARD-tight on x1/x4/x5/x6, loose on x2/x3 |

### Key patterns
- **x2 and x3 are irrelevant dimensions** — ARD length_scales hit upper bound (10.0). Months of pattern-matching on "x2 ≈ 0.5 looks good" was coincidence
- True drivers: **x4 (most), x5, x1, x6**
- x6 must stay in [0.55, 0.80] (W1 and W4 with x6>0.87 both scored badly)
- W6 GP predicted 1.59; actual 1.888 — GP was conservative

### W8 strategy hint
If W7 improves, tighten x1/x4/x5/x6 to ±0.01 around the new best; keep x2/x3 free. If W6 remains the best, explore around it with x4 ∈ [0.15, 0.20] (the highest-importance dim) varied ±0.015.

---

## F8 — Portfolio Optimisation (8D)

**INITIAL_SIZE**: 40 | **Data**: `function_8/` | **GP (tuned)**: raw Y fit, **ARD**, alpha=1e-3

### Best known
| Source | X* | Y* |
|--------|----|----|
| Initial data | [0.056447, 0.065956, 0.022929, 0.038786, 0.403935, 0.801055, 0.488307, 0.893085] | **+9.5985** |
| W6 | [0.0, 0.008, 0.0, 0.007, 0.949, 0.990, 0.034, 1.0] | +9.5656 |
| W4 (pure corner) | [0,0,0,0,1,1,0,1] | +9.518 |

### Weekly history
| Week | Input | Output | Note |
|------|-------|--------|------|
| W1 | [0.817, 0.656, 0.259, 0.855, 0.737, 0.234, 0.839, 0.821] | 7.275 | |
| W2 | [0.856, 0.914, 0.314, 0.370, 0.710, 0.257, 0.647, 0.026] | 7.627 | |
| W3 | [0.076, 0.101, 0.383, 0.338, 0.114, 0.882, 0.615, 0.796] | 9.038 | |
| W4 | [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0] | 9.518 | Pure corner |
| W5 | [0.076, 0.101, 0.383, 0.338, 0.114, 0.882, 0.615, 0.796] | 9.038 | W3 duplicate (wasted) |
| W6 | [0.000, 0.008, 0.000, 0.007, 0.949, 0.990, 0.034, 1.000] | **9.5656** | Near-corner — initial best still wins |
| **W7** | **[0.094538, 0.026040, 0.042757, 0.006871, 0.427589, 0.761761, 0.458953, 0.864164]** | *(planned)* | Probe initial-best basin (intermediate x5, x7) |

### Key patterns
- **Two competing optima**:
  - Near-corner: [0,0,0,0,1,1,0,1] → 9.518; with slight offset → 9.566
  - Initial-best basin: intermediate x5≈0.40, x7≈0.49 → 9.598
- Initial best is **0.032 better** than W6's near-corner — small but real
- Top-3 initial points all have intermediate x5 and x7 → distinct basin
- Pure-corner hypothesis is REJECTED — the slight-offset corner hits 9.566 ceiling

### W8 strategy hint
W7 probes the initial-best basin (different from corner). If Y > 9.6, initial-best basin wins → tighten ±0.02. If Y < 9.5, near-corner basin wins → revert to ±0.02 around W6.

---

## Summary Table — Current Best Results (post-W6)

| Fn | Description | Dims | Best Y* | Source | W7 Submitted |
|----|-------------|------|---------|--------|--------------|
| F1 | Radiation | 2D | \|Y\|=0.4428 | W6 | [0.659153, 0.612535] |
| F2 | Noisy Chemical | 2D | +0.6112 | Initial | [0.663558, 0.945632] |
| F3 | Drug Discovery | 3D | −0.00133 | W6 | [0.542768, 0.479326, 0.475361] |
| F4 | Warehouse | 4D | −4.0255 | Initial | [0.497320, 0.428456, 0.397541, 0.283425] |
| F5 | Chemical Yield | 4D | +4443.3 | W6 | [0.130000, 1.0, 1.0, 1.0] |
| F6 | Environmental | 5D | −0.4816 | W6 | [0.503189, 0.259871, 0.688609, 0.828287, 0.0] |
| F7 | ML Hyperparams | 6D | +1.8882 | W6 | [0.121273, 0.478154, 0.500852, 0.195523, 0.279358, 0.735707] |
| F8 | Portfolio | 8D | +9.5985 | Initial | [0.094538, 0.026040, 0.042757, 0.006871, 0.427589, 0.761761, 0.458953, 0.864164] |

---

## GP Configuration — W7 (MLE-tuned where applicable)

| Fn | Pre-W7 (fixed) | W7 (tuned) | Tuning method | Key insight |
|----|---------------|------------|---------------|-------------|
| F1 | RBF ls=0.1, α=1e-10 | RBF ls=0.212, α=1e-10 | Marginal-likelihood | True ridge wider than assumed |
| F2 | RBF ls=0.1, α=1e-4 | RBF ls=0.1, α=1e-3 | Manual α bump | Function is noisier than alpha=1e-4 suggested |
| F3 | RBF ls=0.3, α=1e-6 | RBF ls=0.3, α=1e-4 | Manual | Single-mode peak emerging |
| F4 | RBF iso ls=0.3 | **RBF ARD** (4D) | Marginal-likelihood | Mild anisotropy |
| F5 | RBF iso ls=0.3 | **Matern-1.5** ARD | Kernel selection by MLE | RBF strongly misspecified (Δlog-lik = +1100) |
| F6 | RBF iso ls=0.3 | **RBF ARD** (5D) | Marginal-likelihood | x5 dim has different scale |
| F7 | RBF iso ls=0.3 | **RBF ARD** (6D) | Marginal-likelihood | **x2, x3 confirmed irrelevant** (ls=10 saturated) |
| F8 | RBF iso ls=0.3 | **RBF ARD** (8D) | Marginal-likelihood | Multiple basins identified |

---

## W7 Portal Strings

```
F1: >>> 0.659153-0.612535 <<<
F2: >>> 0.663558-0.945632 <<<
F3: >>> 0.542768-0.479326-0.475361 <<<
F4: >>> 0.497320-0.428456-0.397541-0.283425 <<<
F5: >>> 0.130000-1.000000-1.000000-1.000000 <<<
F6: >>> 0.503189-0.259871-0.688609-0.828287-0.000000 <<<
F7: >>> 0.121273-0.478154-0.500852-0.195523-0.279358-0.735707 <<<
F8: >>> 0.094538-0.026040-0.042757-0.006871-0.427589-0.761761-0.458953-0.864164 <<<
```
