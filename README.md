# Black-Box Optimisation Capstone — Imperial College London

Bayesian Optimisation of eight unknown functions using Gaussian Process surrogates and a function-specific exploration–exploitation strategy, extended in Week 4 with neural network surrogates built in TensorFlow.

---

## Project Overview

This capstone is a **Black-Box Optimisation (BBO)** challenge. I am given eight unknown functions — no equations, no gradients. The only tool available is: submit an input, receive an output. The goal is to find the input that produces the **highest possible output** for each function within a limited query budget.

Each function simulates a real-world problem — radiation detection, drug discovery, chemical yield optimisation, ML hyperparameter tuning and more. One query per function per week forces every decision to be deliberate and evidence-based.

---

## Inputs and Outputs

Each function accepts a vector of values in **[0, 1]** per dimension, submitted as a hyphen-separated string:

```
F1 (2D): 0.591837-0.591837
F5 (4D): 0.204881-0.877830-0.879582-0.870578
F8 (8D): 0.856208-0.914280-0.313812-0.369507-0.710367-0.257166-0.646775-0.025767
```

Each submission returns a **single scalar** — the function's response at that point. Outputs vary widely: F5 returns values around 1000+, F3 and F6 return negatives (higher is still better), and F1 returns near-zero unless a contamination source is nearby.

---

## Challenge Objectives

**Goal**: Maximise all eight functions. Even functions with negative outputs (F3, F6) are framed as maximisation — the goal is to bring the score as close to zero as possible.

**Key constraints:**
- One query per function per week — no re-dos
- No access to function internals, gradients or closed form
- 10–40 pre-provided data points per function to start
- Some functions are explicitly noisy (F2)
- Unknown which input dimensions matter most

---

## Technical Approach

### Weeks 1–3: Gaussian Process Surrogates

All functions use a **Gaussian Process (GP)** with a fixed RBF kernel (`length_scale=0.1`, `alpha=1e-10`). `log(|Y| + 1e-300)` is applied to Y before fitting to handle scale differences.

**Acquisition functions:**

**UCB** (most functions): `acquisition = post_mean + beta * post_std`

**EI** (F4, F8 from Week 3 — switched after consecutive poor results):
```python
improvement = post_mean - best_log_y
Z = improvement / (post_std + 1e-9)
acquisition = improvement * norm.cdf(Z) + post_std * norm.pdf(Z)
```

**Beta schedule (UCB):**

| Function | W1 | W2 | W3 | Rationale |
|----------|----|----|----|-----------|
| F1, F2, F3, F6 | 2.5 | 3.5 | 3.5 | No clear improvement — keep exploring |
| F4 | 2.5 | 3.5 | EI | Worsened two weeks running |
| F5 | 2.5 | 2.5 | 2.0 | +21% jump in W2 — exploit peak |
| F7 | 2.5 | 3.5 | 2.0 | +14% jump in W2 — exploit peak |
| F8 | 2.5 | 4.0 | EI | Marginal gains only |

### Week 4: Neural Network Surrogate (TensorFlow)

Week 4 adds a TensorFlow MLP surrogate alongside the GP:

- **Architecture**: `Sequential([Dense(64, relu, L2=1e-3), Dropout(0.1), Dense(64, relu, L2=1e-3), Dropout(0.1), Dense(1)])`
- **Training**: Custom `train_step` with `tf.GradientTape`, Adam lr=1e-3, 1500 epochs
- **Acquisition**: Dual strategy — NN gradient ascent on `tf.Variable` input (800 steps) vs GP UCB/EI on a 30k random grid; winner selected by highest NN-predicted score
- **Sensitivity**: `tf.GradientTape` used to compute `|∂output/∂x_i|` at best known point — ranks which input dimensions are most influential
- **Validation**: RBF-SVM classifies observations into top-30% (good) / bottom-70% (poor) and checks whether the proposed query falls in the good region

---

## Results Summary (Weeks 1–4)

| Function | Dims | W1 | W2 | W3 | W4 submitted |
|----------|------|----|----|----|--------------|
| F1 | 2 | 2.82e-4 | 0.0 | ~0 | 0.639955-0.673176 |
| F3 | 3 | −0.476 | −0.196 | improving | — |
| F5 | 4 | 984 | **1192** | strong | — |
| F7 | 6 | 0.679 | 0.775 | improving | — |
| F4 | 4 | −24.5 | −28.6 | switched to EI | — |
| F8 | 8 | 7.28 | 7.63 | switched to EI | — |

---

## Repository Structure

```
week1/
  data/          initial inputs/outputs (.npy) per function
  notebooks/     function_1 … function_8 notebooks
week2/
  data/
  notebooks/     same structure; weekly_log cells updated
week3/
  data/
  notebooks/
  notebooks/week3_strategy_reflection.ipynb
week4/
  data/
  notebooks/     NN surrogate (TensorFlow) + GP fallback
  notebooks/week4_strategy_reflection.ipynb
```

Each function notebook is self-contained: load data → fit surrogate → run acquisition → print submission string.

---

## Libraries

| Library | Role |
|---------|------|
| `scikit-learn` | GaussianProcessRegressor, RBF kernel, SVC |
| `scipy` | EI acquisition (norm.cdf, norm.pdf) |
| `TensorFlow / Keras` | NN surrogate (Week 4): Sequential API, GradientTape training |
| `numpy` | Array operations, grid search |

Week 5 will introduce **PyTorch** as an alternative framework for the surrogate model.

---

*Imperial College London — ML Capstone | Bayesian Black-Box Optimisation*
