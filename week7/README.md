# Week 7 — Hyperparameter Tuning + Exploit Round

**Surrogate:** GP with **MLE-tuned kernel** per function (replaces fixed length_scale=0.1/0.3 used in W1–W6). ARD (Automatic Relevance Determination) enabled for 4D+ functions; Matern-1.5 substituted for RBF on F5 after kernel-selection by marginal likelihood.

**Acquisition:** UCB with β=2.0 (improved functions, exploit) or β=2.5 (missed functions, slight explore). Per-function trust radii tightened around new bests.

**Key insight from tuning:** F7 has **two irrelevant input dimensions** (x2, x3) — ARD length-scales saturate at the upper bound, meaning the GP marginal likelihood prefers infinite-scale (no signal). Months of speculating that "x2 ≈ 0.5 looks good" was coincidence.

---

## Part 1 — W7 Query Submissions

### W6 results recap

| Fn | W6 Y | Previous best | Status |
|----|------|---------------|--------|
| F1 | **+0.4428** | \|Y\|=0.0128 | New best — 35× |
| F2 | +0.1356 | +0.6112 (initial) | Still chasing |
| F3 | **−0.00133** | −0.0348 | New best — 26× |
| F4 | −4.6880 | −4.0255 (initial) | Narrow miss |
| F5 | **+4443.3** | +3531 | New best — +26% |
| F6 | **−0.4816** | −0.5573 | New best — +13% |
| F7 | **+1.8882** | +1.365 (initial) | New best — +38% |
| F8 | +9.5656 | +9.5985 (initial) | Within noise |

**6 of 8 functions hit new all-time bests in W6.** W7's primary job is to *exploit-tighten* around those wins while making informed pivots on F2, F4, F8.

### W7 portal strings

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

### Per-function rationale

**F1 (2D, [0.659153, 0.612535])** — MLE-tuned length_scale rose from 0.1 to **0.212**, telling the surrogate the W4→W6 ridge is wider than originally modelled. UCB on a ±0.04 random search around W6 picked a point that extends the W4→W6 gradient direction. If the ridge continues, |Y| should grow; if W6 was the peak, this will weaken and we tighten further next week.

**F2 (2D, [0.663558, 0.945632])** — The initial best [0.703, 0.927] at Y=0.611 has never been weekly-retested. After 6 weeks of weekly queries averaging ≤0.15, the only way to know if 0.611 is a real peak or optimistic noise is to query nearby. W7 sits ±0.05 from the initial best.

**F3 (3D, [0.542768, 0.479326, 0.475361])** — W6 was a 26× breakthrough. Tight UCB exploit on a ±0.04 random search around the W6 point. The earlier "x3 → 0" hypothesis was wrong — true x3 sweet spot is ≈ 0.44.

**F4 (4D, [0.497320, 0.428456, 0.397541, 0.283425])** — ARD-GP fitted on 36 points. W6 came within 0.7 of the initial best; W7 sits midway between W6 and the initial best with ARD-guided UCB. Predicted mean = −2.50, which would beat the initial best −4.03 if accurate.

**F5 (4D, [0.130, 1.000, 1.000, 1.000])** — **Kernel selection by marginal likelihood:** Matern-1.5 beat RBF by +1100 log-units (RBF was wildly misspecified for the boundary-saturation behaviour). With x2/x3/x4=1.0 confirmed optimal, W7 scans x1 = 0.130 — bracketing W4's 0.133 and W6's 0.153. Note: the raw GP UCB pick was x1=0.259, but I manually overrode based on the prior data trend (smaller x1 → better) and the fact that x1=0.259 has never been tested at boundary.

**F6 (5D, [0.503189, 0.259871, 0.688609, 0.828287, 0.0])** — ARD-GP UCB tight around W6's new best −0.482. x5 = 0.0 (lower bound) is fine for F6 — W4 with x5=0.0 was bad because of *other* dims being at extremes, not because of x5.

**F7 (6D, [0.121273, 0.478154, 0.500852, 0.195523, 0.279358, 0.735707])** — **The biggest tuning win this week.** ARD length-scales (after MLE on 36 points):

| Dim | length_scale | 1/ls | Verdict |
|-----|--------------|------|---------|
| x4 | 0.18 | 5.69 | Most important |
| x5 | 0.20 | 4.89 | Important |
| x1 | 0.50 | 2.00 | Moderate |
| x6 | 0.55 | 1.83 | Moderate |
| **x2** | **10.0 (bound)** | **0.10** | **Irrelevant** |
| **x3** | **10.0 (bound)** | **0.10** | **Irrelevant** |

W7 tightens x1, x4, x5, x6 to ±0.02–0.03 around W6 and explores x2/x3 freely (they don't matter). GP UCB predicted mean = 1.96 — would beat W6's 1.89.

**F8 (8D, [0.094538, 0.026040, 0.042757, 0.006871, 0.427589, 0.761761, 0.458953, 0.864164])** — Pure-corner hypothesis (W4: 9.518, W6: 9.566) has hit a ceiling 0.03 below the initial best (9.598). The top-3 initial points all have **intermediate** x5 and x7, suggesting a different basin. W7 probes ±0.04 around the initial best in that basin.

---

## Part 2 — Reflection on Strategy

### 1. Which hyperparameters did you choose to tune, and why did you prioritise them?

Three categories, in order of impact on inference:

**(a) GP kernel hyperparameters — length_scale (per dimension), α (assumed noise variance), kernel family (RBF vs Matern).** These are the *most* important because they encode the surrogate's structural assumptions: how smooth the function is and what spatial scale changes matter at. With fixed length_scale=0.1, the GP was effectively a local interpolator — every point outside ±0.1 was modelled as independent, which made acquisition functions chase corners. After tuning F1 to length_scale=0.212, the surrogate finally "saw" the W4→W6 ridge as a coherent feature. For F5, kernel choice mattered more than length_scale itself — RBF assumes infinite smoothness, but F5 has a saturation at the boundary x2=x3=x4=1.0 that creates a sharp ridge. Matern-1.5 (which is once-differentiable rather than infinitely so) fit dramatically better (Δlog-marg-lik ≈ +1100). For F7, **ARD per-dimension length-scales** turned out to be the highest-leverage change — they identified x2 and x3 as irrelevant features (length-scales saturated at the upper bound), which reshapes the whole acquisition strategy.

**(b) Acquisition hyperparameters — UCB β, EI exploration margin, trust radius.** β controls the exploration–exploitation tradeoff (mean + β·σ). I prioritised tuning β by improvement signal: β=2.0 when a function improved last round (exploit harder), β=2.5–3.5 when stalled. Trust radius (the local search neighbourhood) was tightened from ±0.10 to ±0.04 around new bests after W6's six breakthroughs.

**(c) NN surrogate hyperparameters — hidden units, dropout, weight decay, learning rate, training epochs.** These I *de-prioritised*. With 16 data points in 8D, an MLP has more parameters than samples and is overwhelmingly likely to overfit no matter how carefully tuned. The NN played its role in W4–W5 as a sanity check against the GP; from W6 onward I trust the MLE-tuned GP more.

### 2. How has hyperparameter tuning changed your query strategy compared to earlier rounds?

Earlier rounds were dominated by **acquisition β tuning** — bigger β to push out of stuck regions, smaller β to exploit. Surrogate structure was held fixed. This caused two failure modes:

- *Corner-chasing*: With length_scale=0.1 in 8D, the GP posterior reverts to the prior mean almost everywhere, so UCB sees uniform σ and picks the most extreme corner that the EI hasn't blocked. F8's W4 [0,0,0,0,1,1,0,1] is the canonical example.
- *Phantom dimensions*: I spent weeks tracking "x2 ≈ 0.5 across F7 best results" without realising x2 was an irrelevant input. ARD would have flagged this in Week 3.

After W6 tuning, query strategy is:

- *Surrogate first, acquisition second.* I fit a tuned GP (MLE on length-scale, ARD where data permits, kernel selection by log-likelihood) before choosing β.
- *Dimension prioritisation.* For F7, I now allocate exploration budget unequally — tight on x1/x4/x5/x6, loose on x2/x3.
- *Calibrated trust regions.* Trust radius = α × length_scale rather than a hardcoded ±0.05. After F1's length_scale rose from 0.1 → 0.21, the effective neighbourhood for exploit naturally widened.

### 3. Which tuning method(s) did you apply, and what trade-offs did you notice?

**Marginal-likelihood maximisation (Type-II MLE)** — primary method. For each GP, I optimised log-marginal-likelihood w.r.t. length-scale, signal variance, and noise variance using L-BFGS with 10 random restarts (scikit-learn's `n_restarts_optimizer=10`). *Trade-off:* fast, principled, and the loss surface is often multimodal — restarts help but don't guarantee the global optimum. With only 16 points per function, MLE can return overconfident length-scales (e.g., x2 and x3 saturating at the upper bound 10.0 in F7 is technically MLE's verdict that "this dimension carries no signal", but in a different data regime it could just mean "we haven't varied this dimension enough to detect signal").

**Manual adjustment** — fallback for sanity-checking. After MLE picked Matern-1.5 for F5 by a huge margin, I manually inspected predictions at the boundary and confirmed the ridge was now captured properly. For F2, I manually bumped α from 1e-4 to 1e-3 because the initial best 0.611 sits far from any retest, suggesting underestimated noise. *Trade-off:* prone to confirmation bias; I might be tuning α to make the model agree with a hypothesis rather than the data.

**Grid search (kernel selection)** — applied to F5 (RBF vs Matern-1.5 vs Matern-2.5). *Trade-off:* only feasible when the hyperparameter is low-cardinality (here, 3 kernel choices). For continuous hyperparameters, MLE dominates.

**Random search (NN architecture)** — used in earlier weeks for hidden-units/dropout/weight-decay. *Trade-off:* better than grid in higher-dim hyperparameter space but expensive — and with 16 data points, even the best NN architecture overfits, so the search is fitting noise.

**Bayesian optimisation of hyperparameters** — not applied. BO-within-BO would be principled but the meta-objective (log-marg-lik) is already what MLE optimises; using GP-on-GP adds approximation error without new information.

**Hyperband** — not applicable. Hyperband shines for *multi-fidelity* settings where you can train a cheap proxy first (epoch-budget training, sub-sample data). Our queries are atomic — one query per week per function — so there's no "cheap version" to early-stop.

### 4. As the data set grows to 16 points, what limitations of your model become clearer?

**Overfitting in high dimensions.** For F8 (8D), 16 weekly points plus 40 initial = 56 total samples. An 8D unit hypercube has volume 1.0 and 256 corners; 56 points is sparse coverage. ARD length-scales for some dimensions saturate at the upper bound (meaning the GP can't tell signal from noise on that dimension). This is the MLE telling me "you don't have enough data to identify all 8 dimensions' length-scales".

**Irrelevant features become visible.** F7's ARD result (x2, x3 length-scales hit upper bound) is direct evidence that two of six dimensions carry no signal. This was *not* visible in the fixed-isotropic-kernel regime — the isotropic length_scale is dragged toward the average, masking which dimensions actually matter. With more data, I'd expect more irrelevant-dimension detections (F8 is a candidate).

**Diminishing returns of exploration.** F1 went from \|Y\|=0.0128 (W4) to 0.4428 (W6) — a 35× jump on one query. The remaining unexplored region of the [0.55, 0.75]² hot zone is tiny. The marginal information from each new query in that region is shrinking. Reciprocally, F2 and F4 are stuck because the productive regions haven't been densely sampled — the bottleneck is *coverage*, not surrogate quality.

**Model misspecification visible via kernel-selection metrics.** F5's RBF vs Matern-1.5 marginal-likelihood gap of +1100 log-units is enormous — RBF was off by orders of magnitude in its prior. Without explicitly comparing kernel families, this misspecification would have been invisible.

**Confounding from noisy observations.** F2's alpha=1e-4 implies near-noise-free observations. After 16 points without ever reproducing the initial best 0.611, the more likely explanation is that the true noise is larger than 1e-4 and the initial best was an optimistic outlier. Tuning α via LOO-CV would expose this.

### 5. How might you apply hyperparameter tuning techniques to larger data sets in future rounds of the BBO capstone or more complex models in future ML/AI projects?

For the remaining BBO weeks (W8–W12 or however long the capstone runs):

- **Refit MLE every week** — once data passes ~25 points per function, run full ARD with all length-scales free. With ~30+ points, ARD has enough signal to identify irrelevant dimensions reliably.
- **Cross-validated alpha selection** — for noisy F2, run LOO-CV across a grid of α ∈ {1e-5, 1e-4, 1e-3, 1e-2} and pick the α minimising predictive negative log-likelihood. This estimates the true noise level more honestly than MLE on small data.
- **Kernel zoo per function** — extend the kernel-selection grid: RBF, Matern-1.5, Matern-2.5, RationalQuadratic, Periodic (for any function that might have multiple peaks). Compare by marginal likelihood and by LOO-RMSE.
- **Trust-region calibration** — make trust radius a function of the local length-scale (e.g., 0.2 × ls), not a hard-coded constant.

For larger / more complex models in future ML/AI projects:

- **Bayesian optimisation of NN hyperparameters** (with libraries like Optuna, Ax, BoTorch) — for vision/NLP models where training each config costs hours, BO with surrogate models (often GP or TPE) drastically reduces compute vs random search.
- **Hyperband / ASHA** — for models where partial training is informative (NN epochs, gradient-boosted trees with fewer rounds), early-stop unpromising configs aggressively. ASHA is the production-grade variant — works in distributed setting.
- **BOHB** — Bayesian + Hyperband fusion. State-of-the-art for HP optimisation under compute budget. Use this for any HP space with >5 hyperparameters.
- **Track everything with MLflow / Weights & Biases** — every HP configuration, every metric, every random seed. Without this, retrospection on "which HP combos worked" becomes guesswork.
- **Population-based training** (DeepMind's PBT) for very long training runs (e.g., RL agents) where HPs should *adapt* over training rather than be fixed.

### 6. How does tuning in this black-box set-up prepare you to think like a professional ML/AI practitioner in real-world contexts with incomplete information?

The BBO setup mirrors the real-world ML problem almost exactly: a black box (the data-generating process), expensive evaluations (each model training, each A/B test), no ground truth, and the obligation to commit to a decision under uncertainty. Hyperparameter tuning under these constraints teaches several professional habits:

**Separate "the model thinks X" from "X is true".** Every acquisition value is a conditional probability under a particular surrogate. When the GP predicted F7 = 1.59 for W6 and the truth came back 1.89, that wasn't the GP being wrong — it was the GP being honestly uncertain. Practitioners who treat surrogate predictions as point estimates get burned; practitioners who track predicted intervals and check them against observed values build calibration over time.

**Embed sanity checks.** Marginal-likelihood gives one signal; LOO-CV gives another; SVM region classification (used in W4) gives a third. When all three agree, confidence is justified; when they disagree, prefer the more pessimistic one. The same logic applies to production ML — don't ship a model that passes only one validation metric.

**Resist confirmation bias in HP tuning.** F7's "x2 ≈ 0.5 looks good" pattern survived three weeks of pattern-matching before ARD destroyed it. The lesson: when a heuristic survives multiple weeks without being statistically tested, that's the moment to test it, not double down.

**Communicate uncertainty in decisions.** When I tell the project portal "submit F5 = [0.130, 1, 1, 1]", I'm committing one query. If asked "are you sure?", the honest answer is "GP UCB predicts mean=4444, sd=42 — I'm willing to bet the next query against the current best". In real-world ML, the equivalent is honest expected-value framing rather than overconfident point predictions.

**Build feedback loops.** Every weekly query is a controlled experiment: I made a prediction (GP mean), I got a result, I update both the model AND my meta-knowledge ("ARD on F7 revealed irrelevant dims", "RBF is misspecified for F5"). Without this loop, ML practice degenerates into trying things and remembering what worked — which is exactly what happens in teams without proper experiment-tracking infrastructure.

**Know when to stop tuning.** F1's marginal-likelihood-optimal length_scale is 0.212. I could grid-search the 4th decimal place and squeeze a tiny log-likelihood improvement, but the query budget is the binding constraint, not the surrogate's last drop of fit. Professional practitioners spend tuning effort proportional to its expected impact on the downstream decision — when the next query matters more than the next tuning iteration, stop tuning and ship.

---

## Files

- `FUNCTION_CONTEXT.md` (project root) — updated with W6 actuals, W7 plan, tuned GP configuration per function
- `outputs/hp_tune_demo.py` (session scratch) — marginal-likelihood demo for F1, F5, F7
- `outputs/w7_picker.py` (session scratch) — full W7 query selection pipeline
