---
title: "ML Interview Questions"
date: 2026-07-24
summary: >-
  My running bank of machine-learning interview questions, worked out in full —
  the bias-variance decomposition, every loss and optimizer, trees and boosting,
  classic ML, backprop by hand, CNN/RNN/Transformer, MLE statistics, and the
  causal traps behind "why did the A/B test look wrong?". Short answer first,
  then the depth to survive the follow-ups.
tags: [Interview, ML, Foundations, Bias-Variance, Regularization, Deep-Learning, Statistics, Experimentation]
---

This is my working answer bank for ML interviews. Each entry is a question I expect to get, written the way I'd want to answer it on a whiteboard: the short version first, then the depth to survive follow-ups. I pull these from my prep tracker as I go, and I've tried to write each one so I could teach it, not just recite it — because that's the level an interviewer is actually probing for.

How to read this: the **30-second answer** is what I say first, out loud. Everything after it is the material I reach into when the interviewer says *"okay, but why?"*.

---

# 1. Bias–variance tradeoff & regularization

*The classic "ML breadth" opener. What's really being checked is whether I can reason about **generalization error** — the gap between training and unseen performance — rather than reciting two definitions.*

## The 30-second answer

For a model `f̂` trained on a random dataset, the expected squared error at a test point `x` decomposes into three additive, non-negative pieces:

```
E[(y - f̂(x))²]  =  Bias[f̂(x)]²   +   Var[f̂(x)]   +   σ²
                    (underfit)         (overfit)      (irreducible)
```

- **Bias** — how far the *average* prediction (averaged over all possible training sets) is from the truth. High bias means the model class is too simple to represent the real function — it's wrong even with infinite data. This is **underfitting**, and it shows as high error on *both* train and test.
- **Variance** — how much the prediction *wobbles* as the training set changes. High variance means the model is chasing the noise of whichever particular dataset it saw. This is **overfitting**, and it shows as low train error but high test error.
- **σ² (irreducible)** — noise in the labels/data-generating process. No model can beat this floor; pretending you can is a sign you're fitting noise.

The **tradeoff**: adding capacity (more parameters, more features, deeper trees) lowers bias but raises variance; adding constraints (regularization, simpler models, more data) lowers variance but usually raises bias. Total error is a U-shape in complexity, and the job is to sit at the bottom.

```
error
  │\                          /
  │ \    variance   total →  /
  │  \____              ____/
  │       \___      ___/
  │  bias     \____/    ← sweet spot: minimum total error
  └──────────────────────────► model complexity
```

## Deriving the decomposition (be ready to sketch it)

Let the truth be `y = f(x) + ε` with `E[ε]=0`, `Var(ε)=σ²`. For a prediction `f̂(x)` (random because the training set is random), add and subtract the mean prediction `E[f̂]`:

```
E[(y - f̂)²] = E[(f + ε - f̂)²]
            = (f - E[f̂])²      ← Bias²   (systematic error)
            + E[(f̂ - E[f̂])²]  ← Variance (sensitivity to the sample)
            + σ²                ← noise    (cross terms vanish since E[ε]=0)
```

The takeaway the interviewer wants: **you cannot drive both bias and variance to zero at once** with finite, noisy data — reducing one term generally inflates the other, so you optimize their *sum*.

## How I'd actually diagnose a model

| Symptom | Diagnosis | What I'd do |
|---|---|---|
| High train error **and** high test error | **High bias** (underfit) | More capacity, better/more features, train longer, weaker regularization |
| Low train error, high test error (big gap) | **High variance** (overfit) | Regularize, get more data, simpler model, early stopping, ensembling |
| Low train **and** low test error | Good fit | Ship it |
| Low train, test error that won't drop below a floor | You've hit **irreducible error** | Improve the *data/labels*, not the model |

The tool that separates these is a **learning curve** — train and validation error plotted against training-set size (or epochs). Converged-but-high = bias; a persistent gap that shrinks with more data = variance.

## Regularization is the variance-reduction toolkit

The one line I want to land: **every one of these buys a large reduction in variance at the cost of a small increase in bias.** They all constrain the hypothesis space so the model *cannot* fit noise. That's the unifying frame — not four unrelated tricks.

### L2 (Ridge / weight decay) — add `λ Σ wᵢ²` to the loss
- **Effect:** penalizes large weights, shrinking them **smoothly toward zero** (almost never exactly zero). The gradient of the penalty is `2λw`, so every step also nudges each weight a bit back toward the origin — hence the name *weight decay*.
- **Why it cuts variance:** it caps the *effective* capacity of the model. Small weights make the learned function smoother — a small change in an input produces a small change in output — so the fitted function moves less from one training set to the next. Less wobble = less variance.
- **Bayesian view:** equivalent to a **Gaussian prior** (mean 0) on the weights; `λ` is the prior's inverse variance. MAP estimation with that prior *is* ridge regression.
- **With correlated features:** L2 spreads weight across the correlated group rather than arbitrarily picking one, which stabilizes the solution.
- **Closed form (linear case):** `w = (XᵀX + λI)⁻¹Xᵀy` — the `+λI` also makes the matrix invertible even when `XᵀX` is singular, which is a nice secondary benefit to mention.

### L1 (Lasso) — add `λ Σ |wᵢ|` to the loss
- **Effect:** drives many weights to **exactly zero** → automatic **feature selection** and a sparse, interpretable model.
- **Why it cuts variance:** zeroing a weight literally deletes a parameter, shrinking the number of effective degrees of freedom. Fewer parameters = lower variance.
- **Why sparsity happens (the geometry):** minimizing loss subject to a budget `Σ|wᵢ| ≤ t` means the elliptical loss contours expand until they first touch the constraint region. L1's region is a **diamond** whose corners lie *on the axes* — the contour almost always touches at a corner, where some coordinates are exactly 0. L2's region is a **circle** with no corners, so the touch point generically has all coordinates non-zero.
- **Bayesian view:** a **Laplace prior** (sharply peaked at 0, heavy tails).
- **Elastic Net** (`λ₁Σ|w| + λ₂Σw²`) combines both: it keeps L1's sparsity while, like L2, handling groups of correlated features gracefully instead of arbitrarily keeping one and dropping the rest.

### Dropout (neural networks)
- **Mechanism:** during training, independently zero each unit with probability `p`; at test time use all units but scale activations by `(1-p)` — or, more common, use *inverted dropout* (scale by `1/(1-p)` at train time so test time is untouched).
- **Why it cuts variance, two lenses:**
  1. **Anti-co-adaptation:** a neuron can't rely on any specific other neuron being present (it might be dropped), so each neuron is forced to learn a feature that's useful on its own. This yields more robust, redundant representations.
  2. **Implicit ensemble:** each mini-batch trains a different randomly-thinned sub-network, and all sub-networks share weights. Test-time scaling approximates **averaging an exponential number of sub-networks** — and averaging an ensemble is textbook variance reduction.
- **Practical notes:** typical `p` is 0.5 for dense layers, 0.1–0.2 for inputs; it's mostly redundant with (and often replaced by) batch norm in convnets, but still standard in transformers.

### Early stopping
- **Mechanism:** monitor validation error each epoch; stop when it stops improving (with a patience window), even though train error is still dropping.
- **Why it cuts variance:** it limits how far the weights travel from their (small) initialization. Bounding the distance traveled bounds the effective weight magnitude — which is **mathematically close to L2 regularization** (for a quadratic loss, the two are provably related). Less optimization time = less opportunity to memorize noise.
- **Bonus:** it's essentially free — no extra penalty term, no hyperparameter search over `λ`, and it doubles as your compute-saver.

## Variance reducers that aren't a penalty on the loss
- **More training data** — the single most reliable lever; it shrinks `Var[f̂]` directly and doesn't add bias. If you can get more labeled data, do that before anything clever.
- **Bagging / random forests** — average many high-variance models fit on bootstrap resamples; the averaging cancels their independent errors (see §4).
- **Data augmentation** — a domain-specific way to manufacture more effective training data (flips/crops for images, paraphrase/back-translation for text).
- **Batch normalization** — normalizes layer inputs; its use of noisy mini-batch statistics has a mild regularizing side effect.

## Follow-ups I want to be ready for
- **"Why does L1 give sparsity but L2 doesn't?"** — The diamond-vs-circle geometry above. Corners on the axes ⇒ exact zeros.
- **"Is boosting bias- or variance-reduction?"** — Primarily **bias** reduction (it sequentially fits residuals to build a strong learner from weak ones). Bagging is the variance-reduction cousin. (See §4.)
- **"What's deep double descent?"** — For very overparameterized models, test error follows the classic U-curve up to the **interpolation threshold** (where the model exactly fits the train set), then *decreases again* as you keep adding capacity. So the tidy U-curve isn't the whole modern story; enormous nets can generalize well despite fitting training data perfectly. Worth naming to show you're current.
- **"Does regularization increase bias?"** — Yes, by construction. That's the price. It's only worth paying when the variance reduction outweighs the added bias — i.e. when you're in the overfitting regime.
- **"How do you pick `λ` / dropout rate?"** — Cross-validation over a log-spaced grid; watch the validation curve, not the training loss.

---

# 2. Loss functions — cross-entropy vs MSE vs hinge

*The "when would you use each?" question. The real test is whether I connect each loss to the **task**, the **distributional assumption** behind it, and — the part that separates strong answers — what its **gradient** does during training.*

## The 30-second answer

The loss is the quantity optimization actually minimizes, so it must encode what "good" means for the task. Each common loss is the **maximum-likelihood** estimator under a specific noise model.

| Loss | Task | Per-example form | Implied noise model |
|---|---|---|---|
| **MSE** | Regression | `(y - ŷ)²` | Gaussian noise on the target |
| **Cross-entropy** | Classification (want probabilities) | `-Σ yᵢ log p̂ᵢ` | Bernoulli / Categorical target |
| **Hinge** | Classification (want a max-margin boundary) | `max(0, 1 - y·ŷ)`, `y∈{-1,+1}` | None — geometric, not probabilistic |

## Why not just use MSE for everything? (the crux)

This is the answer that impresses, because it's about gradients, not vibes.

**MSE + a sigmoid output is a bad pair for classification.** Two problems:
1. **Non-convex** loss surface in the weights ⇒ optimization can stall in bad local minima.
2. **Vanishing gradient exactly when you're most wrong.** With `ŷ = σ(z)` and MSE, the gradient carries a factor of `σ'(z)`. But `σ'(z) → 0` in the saturated tails — precisely where a *confidently wrong* prediction lives. So the most-wrong examples generate the *smallest* gradient and barely get corrected. Learning crawls.

**Cross-entropy fixes both.** Pair a sigmoid (binary) or softmax (multiclass) with CE and the `σ'(z)` term **algebraically cancels**, leaving a beautifully simple gradient:

```
∂L/∂z = p̂ - y
```

The gradient is now **proportional to the error itself**. A confidently-wrong example (`p̂≈1`, `y=0`) produces the *largest* possible gradient — exactly what you want. CE is also **convex in the logits** and is the log-likelihood of the class labels, so minimizing it = maximizing likelihood.

## MSE for regression, in full
- Penalizes error **quadratically**, so a single large error dominates the loss ⇒ **sensitive to outliers**. It corresponds to assuming Gaussian residuals; if that's roughly true, MSE is the right, efficient choice.
- **Robust alternatives when outliers are a concern:**
  - **MAE / L1 loss** `|y - ŷ|` — robust to outliers (linear penalty), but non-differentiable at 0 and gives a constant gradient magnitude, which can hurt convergence near the optimum. It estimates the **median**, not the mean.
  - **Huber loss** — quadratic within `|error| < δ`, linear beyond. Smooth near 0 *and* robust in the tails — the pragmatic default for noisy regression. `δ` sets the crossover.
  - **Log-cosh, quantile loss** — smooth-Huber-like, and (for quantile) lets you predict intervals rather than a point.

## Hinge vs cross-entropy
- **Hinge** cares only about the **margin**. Once a point is correctly classified *and past the margin* (`y·ŷ ≥ 1`), its loss is **exactly 0** and it stops influencing the boundary. That's what gives SVMs their sparse **support-vector** solution. Downside: no probabilities, and it's non-differentiable at the hinge (use subgradients).
- **Cross-entropy** never reaches 0 — it keeps pushing correct predictions to be *more* confident, so every point keeps contributing a (shrinking) gradient. Use it whenever you need **calibrated probabilities** for ranking, thresholding, or downstream decisions.

## Related losses worth naming
- **Label smoothing:** replace hard targets `(1, 0, …, 0)` with `(1-ε, ε/(K-1), …)`. Stops the model from driving logits to ±∞, improves **calibration**, and acts as mild regularization. Standard in modern image and language training.
- **Focal loss:** `-(1 - p̂)^γ · log p̂` — cross-entropy down-weighted on easy examples so training focuses on hard ones. The fix for **extreme class imbalance** (dense object detection).
- **KL divergence:** what you minimize in distillation / variational methods; CE = KL + a constant, so minimizing CE against a fixed target minimizes KL.
- **Contrastive / triplet / InfoNCE:** for representation learning — pull positives together, push negatives apart in embedding space.

## Follow-ups I want to be ready for
- **"Why does softmax pair with cross-entropy specifically?"** — Because their derivatives combine to the clean `p̂ - y` gradient and together they form the MLE for a categorical target.
- **"Handling class imbalance in plain CE?"** — Class-weighting (up-weight the rare class), resampling (over/under), or move to focal loss; otherwise the majority class dominates the gradient and the minority is ignored.
- **"MSE vs MAE — which and when?"** — MSE if residuals are ~Gaussian and outliers are rare (predicts the mean, penalizes big misses hard); MAE/Huber if the data has heavy tails or outliers (MAE predicts the median).
- **"Is accuracy a loss?"** — No — it's non-differentiable (a step function), so you can't backprop through it. You optimize a smooth surrogate (CE/hinge) and *report* accuracy/F1/AUC.

---

# 3. Optimizers & learning-rate schedules

*Almost always phrased as **"why Adam over SGD?"** The strong answer treats it as a genuine tradeoff, not a strict win, and explains AdamW.*

## The 30-second answer

Everything here is gradient descent — `θ ← θ - η·g`. The variants differ in two axes: whether they use **momentum** (a running memory of past gradients) and whether they **adapt the step size per parameter**.

| Optimizer | Update idea | State kept | Character |
|---|---|---|---|
| **SGD** | Step along the mini-batch gradient | none | Noisy, LR-sensitive, crawls in ravines — but well-understood and generalizes well |
| **SGD + Momentum** | Keep a velocity = EMA of gradients; step along velocity | 1 buffer | Damps oscillation, accelerates along consistent directions |
| **Nesterov** | Momentum but evaluate the gradient at the *look-ahead* point | 1 buffer | Slightly better-corrected momentum |
| **RMSProp** | Divide the step by a running RMS of recent gradients (per param) | 1 buffer | Adapts step size to each parameter's gradient scale |
| **Adam** | Momentum (1st moment) **+** RMSProp-style scaling (2nd moment) | 2 buffers | Fast, robust to LR choice — the default |
| **AdamW** | Adam with **decoupled** weight decay | 2 buffers | The *correct* default for training modern nets |

## Adam in detail (be able to write the update)

```
m ← β₁·m + (1-β₁)·g            # 1st moment: momentum
v ← β₂·v + (1-β₂)·g²           # 2nd moment: per-param gradient scale
m̂ = m/(1-β₁ᵗ),  v̂ = v/(1-β₂ᵗ)  # bias correction (m,v start at 0)
θ ← θ - η · m̂ / (√v̂ + ε)
```

- The **1st moment** is momentum — smooths the trajectory.
- The **2nd moment** gives each parameter its own effective learning rate: parameters with historically large gradients get *smaller* steps, small-gradient (e.g. rare-feature) parameters get *larger* steps. This is why Adam handles **sparse and unevenly-scaled** gradients so well.
- **Bias correction** matters early: `m` and `v` initialize at 0, so without the `1-βᵗ` correction the first steps would be biased toward zero.
- Defaults `β₁=0.9, β₂=0.999, ε=1e-8` work across a huge range of problems — that robustness is much of Adam's appeal.

## Why Adam over SGD — and when *not* to
- **Adam pros:** minimal LR tuning, fast early convergence, robust on messy/sparse gradients, the default for **transformers** and most non-vision deep learning.
- **Adam con:** it tends to find **sharper minima** that can **generalize slightly worse**. That's why state-of-the-art **CNN image classifiers** are still often trained with **SGD + momentum + a strong schedule** — slower and more tuning, but flatter minima and a bit better test accuracy.
- The honest one-liner: *"Adam when I want fast, reliable convergence and anything transformer-shaped; SGD+momentum when I have the tuning budget and want the last bit of generalization, classically in vision."*

## AdamW — the detail most people miss
Plain Adam implements L2 regularization by adding `λw` to the gradient. But that penalty then flows through the `√v̂` denominator, so parameters with large gradients get their weight decay **shrunk** — decay and adaptive-LR get **coupled**, and the regularization is effectively weaker and inconsistent across parameters. **AdamW decouples** them: it applies weight decay **directly to the weights** (`θ ← θ - η(m̂/(√v̂+ε) + λθ)`), independent of the gradient statistics. This small fix measurably improves generalization and is why **AdamW**, not Adam, is the real modern default.

## Learning-rate schedules
The LR is the single most important hyperparameter; how you vary it over training matters as much as its value.
- **Warmup** — start near 0 and ramp up over the first few hundred/thousand steps. Early gradients are large and noisy — and Adam's `v` estimate is unreliable before it's warmed up — so jumping straight to a high LR can diverge. Warmup is essentially mandatory for transformers.
- **Cosine annealing** — after warmup, decay the LR to ~0 along a cosine curve. Big steps early to explore the landscape, tiny steps late to settle precisely into a minimum. The modern default shape is **linear warmup → cosine decay**.
- **Step decay** — cut the LR by a factor (e.g. 10×) at fixed milestones. Classic for SGD-trained CNNs.
- **ReduceLROnPlateau** — drop the LR when validation stops improving; robust and hands-off.
- **Cyclical / warm restarts (SGDR)** — periodically jump the LR back up to escape sharp minima and explore multiple basins; can be ensembled ("snapshot ensembles").

## Follow-ups
- **"Why does momentum help?"** — In a ravine (steep across, shallow along), plain SGD zig-zags off the walls; momentum averages out the transverse oscillation and accumulates speed along the valley floor, so you reach the minimum faster.
- **"Batch size ↔ LR relationship?"** — Bigger batch = lower gradient noise = you can use a proportionally **larger LR** (the linear scaling rule), usually with a longer warmup to stay stable.
- **"Exploding/vanishing gradients in training?"** — **Gradient clipping** (cap the global gradient norm) for exploding; good init + normalization + residual connections + ReLU for vanishing (see §6).
- **"What if loss diverges to NaN?"** — LR too high, no warmup, bad init, or numerical overflow (missing `ε`, unnormalized inputs, fp16 without loss scaling). Lower the LR / add warmup first.
- **"Second-order methods?"** — Newton/L-BFGS use curvature (Hessian) for better steps but are too memory-heavy for deep nets; Adam's 2nd moment is a cheap diagonal approximation to that idea.

---

# 4. Trees & boosting

*The "classic ML" staple, and still the winner on tabular data. The single idea to nail: **bagging reduces variance, boosting reduces bias**, and gradient boosting = sequentially fitting the residuals.*

## Decision tree — the base learner
- **How it works:** recursively split the feature space along the feature/threshold that most reduces **impurity** — **Gini** or **entropy** for classification, **variance (MSE)** for regression. Each leaf predicts the majority class / mean of the points that land in it.
  - *Gini:* `1 - Σ pₖ²`; *Entropy:* `-Σ pₖ log pₖ`. Both measure how mixed a node is; the split maximizes the impurity *decrease* (information gain).
- **Strengths:** interpretable (you can read the rules), handles mixed numeric/categorical features, needs no feature scaling, captures nonlinear interactions automatically, robust to monotone feature transforms.
- **Weakness:** a single deep tree is **high variance** — small data changes flip the splits and it memorizes noise. This is exactly what the two ensemble families fix, in opposite ways.

## Random forest — bagging (variance reduction)
- **Recipe:** train many **deep** trees, each on a **bootstrap sample** of the rows, and at *each split* consider only a **random subset of features** (typically `√d` for classification). Average the predictions (vote for classification).
- **Why the feature subsampling matters:** without it, one dominant feature would be the top split in nearly every tree and the trees would be highly **correlated** — averaging correlated predictors barely reduces variance. Random feature subsets **decorrelate** the trees, so their errors are more independent and cancel on averaging.
- **Net effect:** big **variance reduction**, negligible change in bias. Adding more trees never overfits (it just converges), which makes RF a forgiving, strong baseline.
- **Free diagnostics:** **out-of-bag (OOB) error** — each tree's bootstrap leaves ~37% of rows unused, so you get a validation estimate without a holdout set.

## Gradient boosting — sequential residual-fitting (bias reduction)
- **Recipe:** start with a constant prediction. Then repeatedly fit a **new shallow tree to the negative gradient** of the loss w.r.t. current predictions (for MSE, that's just the **residuals** `y - ŷ`), and add it in scaled by a **learning rate** `η`: `F_{m} = F_{m-1} + η·h_m`.
- **Why it reduces bias:** each tree explicitly corrects the ensemble's current errors, so a sequence of weak learners composes into a strong one. The ensemble's bias falls with each round.
- **Why it can overfit:** because it keeps chasing residuals, it *will* eventually fit noise. Controls: small **learning rate** (shrinkage, e.g. 0.01–0.1), shallow trees (depth 3–8), **subsampling** rows/columns (stochastic gradient boosting), and **early stopping** on a validation set.
- **Boosting vs bagging on trees:** boosting uses *shallow, high-bias* trees and reduces bias sequentially; bagging uses *deep, high-variance* trees and reduces variance in parallel.

## XGBoost / LightGBM / CatBoost — production gradient boosting
- **XGBoost:** uses a **second-order (Newton)** Taylor expansion of the loss (gradients *and* Hessians) for better split choices, plus an explicit **regularization term** on the number of leaves and leaf weights. Level-wise (depth-wise) tree growth. The long-time Kaggle workhorse.
- **LightGBM:** **leaf-wise** growth (always split the leaf with the largest loss reduction) + **histogram binning** of continuous features (bucket values so split-finding is O(bins) not O(rows)). Dramatically faster and more memory-efficient on large data; can overfit small datasets, so cap depth / num_leaves.
- **CatBoost:** native, principled **categorical** handling (ordered target statistics) and **ordered boosting** to reduce target leakage — strong when you have many categorical features.

## The comparison that ties it together

| | Random Forest (bagging) | Gradient Boosting |
|---|---|---|
| Trees built | In parallel, independent | Sequentially, each fixes the last |
| Base tree | Deep (low bias, high variance) | Shallow (high bias, low variance) |
| Primarily reduces | **Variance** | **Bias** |
| Overfit risk | Low — more trees is safe | Higher — needs LR, depth cap, early stop |
| Tuning effort | Forgiving | More sensitive, but higher accuracy ceiling |
| Parallelism | Trivially parallel | Sequential (though split-finding parallelizes) |

## Follow-ups
- **"RF vs GBM in practice?"** — Tuned GBM usually wins on tabular accuracy; RF is the safer, near-zero-tuning baseline. I often start with RF to get a number, then push with LightGBM/XGBoost.
- **"Why must boosting use weak learners?"** — Each round only needs to be slightly better than chance; shallow trees keep per-round variance low so the sequential bias-reduction doesn't spiral into overfitting. Deep trees would overfit within a couple of rounds.
- **"Trees vs neural nets?"** — Gradient-boosted trees remain the default on **structured/tabular** data (they handle mixed types, need no scaling, and train fast); neural nets dominate **unstructured** data (images, text, audio) where representation learning matters.
- **"Feature importance — how, and the caveat?"** — Impurity-based (fast, but **biased toward high-cardinality / continuous** features), **permutation importance** (shuffle a feature, measure the accuracy drop — model-agnostic and more honest), or **SHAP** (game-theoretic, gives per-prediction attributions). Prefer permutation/SHAP for anything you'll act on.
- **"AdaBoost vs gradient boosting?"** — AdaBoost re-weights misclassified *examples* each round; gradient boosting fits the *gradient* of an arbitrary differentiable loss — the more general formulation AdaBoost is a special case of.

---

# 5. Classic ML: SVM, k-means, PCA

*The "one line each — what problem does it solve?" round. Interviewers use these three to check breadth fast, then dig into whichever I sound shakiest on.*

## SVM — the maximum-margin classifier
- **Problem it solves:** among all separating hyperplanes, find the one that **maximizes the margin** — the distance from the boundary to the nearest point of either class. Intuition: the widest "street" between classes generalizes best because it's the most robust to perturbations.
- **Support vectors:** only the points *on* the margin (and any violators) determine the boundary. Remove any other point and the solution is unchanged — the model is defined by a sparse set of examples, which is elegant and memory-efficient.
- **Soft margin (`C`):** real data isn't cleanly separable, so we allow margin violations with slack variables, penalized by `C`. **Small `C`** = wide margin, tolerate more violations = **more regularization** (higher bias, lower variance); **large `C`** = fit the training data harder (lower bias, higher variance). `C` is the bias-variance knob.
- **Kernel trick (the elegant part):** the SVM only ever needs **dot products** between points. Replace `xᵢ·xⱼ` with a **kernel** `K(xᵢ,xⱼ)` — e.g. **RBF** `exp(-γ‖xᵢ-xⱼ‖²)` or polynomial — and you implicitly operate in a very high- (even infinite-) dimensional feature space, getting a **nonlinear** boundary **without ever computing the coordinates** there. `γ` controls RBF reach: large `γ` = each point's influence is local = wiggly boundary = risk of overfit.
- **Loss connection:** the soft-margin SVM is exactly **hinge loss + L2 regularization** (§2), which is a nice unifying point to drop.

## k-means — unsupervised clustering
- **Problem it solves:** partition data into `k` clusters that minimize **within-cluster sum of squared distances** (inertia) to the cluster centers.
- **Algorithm (Lloyd's):** initialize `k` centroids → **assign** each point to its nearest centroid → **update** each centroid to the mean of its assigned points → repeat until assignments stop changing. It's coordinate descent on the inertia objective, so it converges, but only to a **local** optimum.
- **Choosing `k`:** the **elbow method** (plot inertia vs `k`, look for the diminishing-returns kink) or the **silhouette score** (how well-separated clusters are). There's no `k` that's "correct" without a criterion.
- **Failure modes (know these):** sensitive to initialization → use **k-means++** (spread initial centroids). Assumes **spherical, similarly-sized, similar-density** clusters under Euclidean distance, so it fails on elongated, nested, or varying-density shapes — reach for **DBSCAN** (density-based, finds arbitrary shapes and outliers) or **Gaussian Mixture Models** (soft, elliptical clusters via EM) there. Must **standardize** features first, since it's distance-based.

## PCA — linear dimensionality reduction
- **Problem it solves:** find a lower-dimensional linear subspace that retains the **maximum variance** of the data — used for compression, denoising, decorrelation, and 2-D/3-D visualization.
- **How it works:** the principal components are the **eigenvectors of the covariance matrix** (equivalently, the right singular vectors from the **SVD** of the mean-centered data), ordered by eigenvalue = variance explained. Project the data onto the top `k` components. The first PC is the direction of greatest variance; each subsequent PC is orthogonal to the previous and captures the most remaining variance.
- **Must-know details:**
  - **Standardize first** — PCA is scale-sensitive; a feature in large units would otherwise dominate purely because of its scale.
  - Components are **orthogonal linear combinations** of the original features, so they're decorrelated but **lose interpretability**.
  - PCA is **unsupervised** — it maximizes variance, which is not necessarily the direction that best *separates classes*. When you have labels and want discriminative axes, use **LDA** instead.
  - Choose `k` via the **explained-variance ratio** (e.g. keep enough PCs to retain 95% of variance) — the scree plot.
- **Limitation:** it's **linear**. Data on a curved manifold (a swiss roll) needs nonlinear methods — kernel PCA, **t-SNE**/**UMAP** (for visualization), or autoencoders.

## Follow-ups
- **"PCA vs autoencoder?"** — A linear autoencoder with MSE recovers the same subspace as PCA. Nonlinear autoencoders (and t-SNE/UMAP) capture curved manifolds PCA can't; the price is no closed form and less interpretability.
- **"When does k-means fail, concretely?"** — Two crescent-moon clusters: k-means splits them straight down the middle because it can only draw spherical (Voronoi) boundaries; DBSCAN gets them right.
- **"Why is the kernel trick efficient?"** — It computes the high-dimensional inner product directly via `K(·,·)` without ever materializing the high-dimensional coordinates, sidestepping the curse of dimensionality on both compute and memory.
- **"Is k-means guaranteed optimal?"** — No — only a local optimum of a non-convex objective; run it several times with different seeds (k-means++) and keep the lowest inertia.
- **"PCA for a classifier — good idea?"** — Sometimes (denoises, decorrelates, speeds training), but it can throw away *low-variance* directions that are actually *discriminative*. Validate that it doesn't hurt downstream accuracy.

---

# 6. Backprop from scratch

*The single most-tested deep-learning fundamental. Deriving the chain rule through a 2-layer MLP on paper proves I understand how training actually works, not just that I can call `.backward()`.*

## The 30-second answer

Backprop is the **chain rule applied efficiently** via **dynamic programming**. The **forward pass** computes and caches each layer's activations; the **backward pass** propagates the loss gradient from output back to input, and — crucially — **reuses** each layer's computed gradient to get the previous layer's. That reuse is why training a network costs roughly the same as *one* extra forward pass, instead of being exponential in depth.

## The 2-layer MLP, worked by hand

**Forward** (one hidden layer, softmax output, cross-entropy loss):
```
z1 = W1·x + b1          a1 = σ(z1)          # hidden pre-activation, activation
z2 = W2·a1 + b2         ŷ  = softmax(z2)    # output logits, probabilities
L  = -Σ yₖ log ŷₖ                            # cross-entropy
```

**Backward** — apply the chain rule layer by layer, output → input:
```
dz2 = ŷ - y                     # softmax + CE collapse to this clean form
dW2 = dz2 · a1ᵀ                 # outer product: (out×1)(1×hidden)
db2 = dz2
da1 = W2ᵀ · dz2                 # push the error back through W2
dz1 = da1 ⊙ σ'(z1)             # ⊙ elementwise; through the activation's local slope
dW1 = dz1 · xᵀ
db1 = dz1
```

**Update** (gradient descent): `W ← W - η·dW`, `b ← b - η·db`.

The pattern that makes it click and generalizes to any depth: **each layer receives an upstream gradient, multiplies by its own local gradient (the derivative of the operation it performed), and passes the product downstream.** Repeat per layer. A matmul's local gradient is a matmul; an elementwise activation's local gradient is elementwise; that's the entire algorithm.

## Why the details matter

- **Vanishing gradients:** the backward pass through each layer multiplies by the activation's derivative `σ'(z)`. For **sigmoid/tanh**, `σ' ≤ 0.25`, so stacking `L` layers multiplies `L` sub-1 numbers → the gradient reaching early layers is exponentially tiny → early layers barely learn. Fixes:
  - **ReLU** (`max(0,z)`) — derivative is exactly 1 for positive inputs, so it doesn't shrink the gradient (its own issue is "dead" units for negative inputs — hence LeakyReLU/GELU).
  - **Residual connections** — `y = x + f(x)` gives the gradient an additive `+1` shortcut path that never vanishes; this is what made very deep nets (ResNets, transformers) trainable.
  - **Batch/layer norm** — keep activations in a healthy range so derivatives don't saturate.
- **Exploding gradients:** the mirror case (products > 1). Fix with **gradient clipping** and careful initialization.
- **Why cache activations:** the backward pass needs the forward values (`a1`, `σ'(z1)`, the inputs to each matmul). Storing them is the classic **memory-vs-compute** tradeoff, and it's exactly what **activation/gradient checkpointing** trades the other way (recompute in the backward pass to save memory).

## Weight initialization (a common follow-up made explicit)
- **All zeros is fatal:** every neuron in a layer would compute the same thing and receive the same gradient, so they'd stay identical forever — **symmetry never breaks**. You need random init.
- **Xavier/Glorot** (for tanh/sigmoid) and **He** (for ReLU) scale the initial variance by the layer's fan-in/out so activation and gradient variance stay roughly constant across depth — preventing both vanishing and exploding signals at step 0.

## Follow-ups
- **"Why is softmax+CE's gradient exactly `ŷ - y`?"** — The softmax Jacobian and the cross-entropy derivative cancel algebraically. Same underlying reason CE is the right classification loss (§2), and it's why frameworks fuse the two into one numerically-stable op.
- **"What is autograd doing?"** — Building a computation graph on the forward pass, then applying these local-gradient rules in reverse-topological order (reverse-mode automatic differentiation). Backprop *is* reverse-mode autodiff specialized to scalar loss.
- **"LR too high vs too low?"** — Too high: steps overshoot the minimum, loss oscillates or diverges to NaN. Too low: training crawls and can stall on plateaus/saddle points.
- **"Do you update on every example?"** — Usually **mini-batches**: a batch averages the per-example gradients, trading gradient noise (which aids generalization and escaping saddles) against hardware efficiency.
- **"Batch norm's effect on backprop?"** — It normalizes each layer's inputs, smoothing the loss landscape so gradients are better-behaved and larger LRs are usable; it also injects mini-batch noise (mild regularization).

---

# 7. Deep-learning architectures: CNN, RNN/LSTM, Transformer

*The training-side architecture round. The framing that scores: for each model, **what inductive bias does it bake in, and what problem does that bias solve?***

## CNN — exploiting spatial structure
- **Core operations:**
  - **Convolution** — slide a small learned **filter/kernel** across the input, computing dot products; the same weights are used at every location (**weight sharing**). One filter detects one pattern (an edge, a texture) wherever it appears.
  - **Pooling** (max/avg) — downsample spatially, giving small **translation tolerance** and reducing compute.
- **Inductive biases:** **locality** (nearby pixels are related, so filters are small) and **translation equivariance** (a feature is detected the same way anywhere in the image). Weight sharing means far fewer parameters than a dense layer over pixels — the reason CNNs are data-efficient on images.
- **Receptive field:** the region of the input that influences a given output unit. It **grows with depth** (and with stride/pooling), so early layers respond to edges/textures and deep layers respond to object parts and whole objects.
- **Solves:** images and any grid-structured signal (spectrograms, some time series) — efficiently and with a strong prior.

## RNN / LSTM — exploiting sequential structure
- **RNN:** maintains a **hidden state** `hₜ = f(Wₓxₜ + Wₕhₜ₋₁)` passed step to step, applying the **same weights** at every timestep. Built-in bias for **order and recurrence**; handles variable-length sequences naturally.
- **The core problem:** training uses **backprop-through-time**, which multiplies by the recurrent weight repeatedly across timesteps. Repeated multiplication → **vanishing gradients** (can't learn long-range dependencies) or **exploding gradients**. Vanilla RNNs effectively forget anything more than ~10 steps back.
- **LSTM fix:** add a **cell state** `cₜ` (a memory conveyor belt) plus three **gates** — **forget** (what to erase from memory), **input** (what new info to write), **output** (what to expose as the hidden state). The cell state updates **additively** (`cₜ = fₜ⊙cₜ₋₁ + iₜ⊙c̃ₜ`), so gradients flow across many timesteps **without repeated multiplicative shrinkage** — that additive path is the whole reason LSTMs learn long-range dependencies where RNNs fail. **GRU** is a lighter 2-gate variant (merges cell/hidden and forget/input) — fewer parameters, similar performance.
- **Solves:** sequences (text, time series, audio) — historically dominant, now largely replaced by transformers for anything where you can afford attention.

## Transformer — attention over the whole sequence at once
- **Self-attention (the core op):** each token produces three vectors via learned projections — **Query, Key, Value**. Attention weights are `softmax(QKᵀ / √d_k)`, and each token's output is the weighted sum of all tokens' **Values**. So every token can attend **directly** to every other token in **one operation** — no recurrence, therefore **no distance-based gradient decay**. Long-range dependencies are as easy to learn as short ones.
- **Why divide by `√d_k`:** dot products grow in magnitude with dimension; without scaling they push softmax into saturated regions where gradients vanish. `√d_k` keeps the logits at a sane scale.
- **Multi-head attention:** run attention `h` times in parallel in different learned subspaces, then concatenate. Different heads specialize (syntax, coreference, positional patterns), letting the model attend to multiple kinds of relationships simultaneously.
- **Positional encoding:** attention is **permutation-invariant** — on its own it has no notion of token order. So you must **inject position**: fixed **sinusoidal** encodings (original), **learned** embeddings, or **rotary (RoPE)** in modern LLMs. Without it, "dog bites man" and "man bites dog" look identical.
- **Full block:** attention → residual + LayerNorm → position-wise **feed-forward network** → residual + LayerNorm. The residuals (see §6) and LayerNorm are what make deep stacks trainable.
- **Why it won:** it's **fully parallelizable** across the sequence (unlike RNNs, which must process tokens one at a time), so it trains efficiently on massive data and hardware; and it captures long-range dependencies directly. Parallel training + scaling is the whole story of modern LLMs.
- **The cost:** self-attention is **O(n²)** in sequence length `n` (every token attends to every token), which is the bottleneck every efficiency paper attacks (FlashAttention makes it IO-efficient; sparse/linear-attention variants change the complexity).

## Follow-ups
- **"CNN vs Transformer for vision?"** — A **Vision Transformer (ViT)** splits an image into patches and treats them as tokens; it beats CNNs **at scale** but needs more data/regularization because it lacks the CNN's built-in locality prior (it has to *learn* that structure).
- **"Why did Transformers replace LSTMs for NLP?"** — Parallel training (huge speedup on modern hardware) plus direct long-range attention. LSTMs are inherently sequential and still degrade over long contexts.
- **"Encoder vs decoder vs encoder-decoder?"** — **Encoder** = bidirectional self-attention, sees the whole input (BERT, understanding/classification). **Decoder** = causal (masked) attention, can only see past tokens (GPT, generation). **Encoder-decoder** = seq2seq with cross-attention (T5, translation, summarization).
- **"Self-attention vs cross-attention?"** — Self: Q, K, V all from the same sequence. Cross: Q from one sequence (e.g. the decoder), K/V from another (the encoder output) — how the decoder "reads" the source.
- **"Batch norm vs layer norm — why transformers use LayerNorm?"** — BatchNorm normalizes across the batch dimension, which is unstable for variable-length sequences and small/streaming batches; **LayerNorm** normalizes across features **per token**, independent of batch, which fits sequence models.

---

# 8. Statistics for MLE

*The backbone of every experimentation and A/B-testing question. Interviewers want **precise** definitions — the p-value one especially — because sloppy stats is where a lot of "MLEs" get exposed.*

## The core vocabulary, said precisely

- **p-value:** the probability of observing data **at least as extreme** as what you saw, **assuming the null hypothesis is true**. It is emphatically **not** the probability the null is true, **not** the probability your result happened by chance, and **not** `1 - P(alternative)`. Nailing this exact wording is half the test.
- **Null vs alternative hypothesis:** the null `H₀` is the status quo / "no effect"; the alternative `H₁` is what you're arguing for. You never "prove `H₀`" — you either **reject** it or **fail to reject** it.
- **Significance level α:** the p-value threshold below which you reject `H₀` (commonly 0.05). It's the **false-positive rate you're willing to accept**, chosen *before* the test.
- **Confidence interval (95%):** a range produced by a procedure that, across many repeated experiments, contains the true parameter **95% of the time**. Subtlety to state carefully: for any *single* computed interval the parameter is either in it or not — the "95%" is a property of the **procedure**, not a probability about that one interval. (A CI that excludes 0 corresponds to a significant result at the matching α.)
- **Type I error (false positive), rate α:** rejecting a **true** null — declaring an effect that isn't real.
- **Type II error (false negative), rate β:** failing to reject a **false** null — missing a real effect.
- **Statistical power (1 − β):** the probability of **detecting an effect that genuinely exists**. Convention: aim for **0.80**. Low power is why many "no significant difference" results are actually just underpowered.

The two-error framing in one grid:

| | `H₀` true (no real effect) | `H₀` false (real effect) |
|---|---|---|
| **Reject `H₀`** | Type I error (α) ✗ | Correct — **power** (1−β) ✓ |
| **Fail to reject** | Correct (1−α) ✓ | Type II error (β) ✗ |

## Power drives sample-size planning
Power **increases** with: larger **sample size** `n`, larger true **effect size**, lower **variance**, and a **higher α** (you accept more false positives to catch more true ones). Before running a test you do a **power analysis** to compute the `n` needed to detect the **minimum effect you'd care about (MDE)** at your chosen α and power. Skip this and an "inconclusive" test tells you nothing — you may simply not have collected enough data. This is the single most common practical stats question for MLE/experimentation roles.

## The multiple-comparisons trap
Run 20 independent tests at α = 0.05 and you expect **~1 false positive by chance alone** — `P(at least one) = 1 - 0.95²⁰ ≈ 64%`. Any time a dashboard slices an experiment across many metrics/segments, this bites. Corrections:
- **Bonferroni** — test each at `α/m`. Simple, controls the family-wise error rate, but **conservative** (kills power when `m` is large).
- **Benjamini–Hochberg** — controls the **false discovery rate** (expected fraction of false positives among rejections). Less conservative, the standard choice when you have many tests.

## Common distributions & tests to have ready
- **CLT:** the sampling distribution of the mean → Normal as `n` grows, *regardless* of the underlying distribution. This is why `t`/`z` tests on means work even for non-normal data at large `n`.
- **t-test** — compare two **means** (small-to-moderate `n`, unknown variance).
- **z-test / chi-square** — compare **proportions/rates** (CTR, conversion).
- **Mann-Whitney U / Wilcoxon** — **non-parametric**, compare distributions via ranks when data is skewed or heavy-tailed (see §9).
- **ANOVA** — compare **>2** group means at once (then post-hoc tests, with correction).

## Follow-ups
- **"p = 0.049 vs 0.051 — meaningfully different?"** — No. 0.05 is an arbitrary convention; treat the p-value as continuous evidence and weigh **effect size** and **practical significance** alongside it. A knife-edge p-value is fragile.
- **"Statistical vs practical significance?"** — With huge `n`, a trivially small, business-irrelevant effect can be highly statistically significant. Always report the **effect size and CI**, not just "p < 0.05," and decide on practical grounds.
- **"One-tailed vs two-tailed?"** — Two-tailed tests for a difference in either direction (the safe default); one-tailed only when you truly care about a single direction and commit to it in advance (it's easier to reach significance, so it's easy to abuse).
- **"Bayesian alternative?"** — Report the **posterior** / a **credible interval** ("95% probability the parameter is in this range" — which *is* the intuitive statement people wrongly attribute to CIs) or `P(treatment > control)`, often more directly decision-useful than a p-value.
- **"Frequentist vs Bayesian in one line?"** — Frequentist treats parameters as fixed and data as random (long-run frequencies); Bayesian treats parameters as random and updates a prior with data into a posterior.

---

# 9. Experimentation & causal traps

*The "why did the A/B test look wrong?" question — where MLE meets product judgment. The interviewer wants to hear the correlation-vs-causation traps **by name**, with the tell for each.*

## The traps, each with the tell and the fix

- **Simpson's paradox** — a trend visible in **every subgroup reverses** when the groups are pooled (or vice versa), because of an unbalanced lurking variable.
  - *Concrete example:* a treatment has a **higher** recovery rate than control among **mild** patients *and* among **severe** patients, yet a **lower** rate **overall** — because far more severe patients happened to receive the treatment, and severe cases recover less no matter what. The aggregate is Simpson-reversed by the case-mix.
  - *Fix:* segment by the confounding variable; don't trust the pooled number when group composition differs across arms.
- **Confounders** — a variable that influences **both** the treatment assignment and the outcome, manufacturing a spurious association (ice-cream sales "cause" drownings; the confounder is summer).
  - *Fix:* **randomization** — it balances confounders (known **and** unknown) across arms in expectation, which is the entire reason RCTs license causal claims. When you can't randomize, control via stratification, matching, regression adjustment, or causal-inference designs (IV, diff-in-diff, propensity scores).
- **A/A test** — run the "experiment" with **both arms identical**. If it shows a "significant" difference, the pipeline is broken: bad randomization/hashing, **sample-ratio mismatch**, logging bugs, or a mis-specified metric. It's the sanity check you run **before** trusting any A/B result, and it also empirically calibrates your false-positive rate.
- **Novelty effect** — a new feature gets a **temporary** bump purely because it's new and users click to explore; the lift **decays** as novelty wears off. Shipping on week-1 numbers over-credits the change.
- **Primacy effect** — the opposite: existing users are habituated to the old experience and **underperform** on the new one at first, then adapt and improve. Both novelty and primacy mean you must run the test **long enough to reach steady state** before reading it.

## The A/B testing statistics toolkit
- **t-test** — compares two group **means** (avg revenue/user, avg session length); relies on approximate normality of the mean, which the CLT gives you at large `n`.
- **Mann-Whitney U** — the **non-parametric** counterpart; compares distributions via ranks. Reach for it when the metric is **skewed or heavy-tailed** (revenue, watch time, latency), where a few whales make the mean misleading.
- **Chi-square / z-test of proportions** — for **rate** metrics (CTR, conversion, retention).
- **Guardrail metrics** — metrics that must **not regress** even if the target metric improves (latency, crash rate, revenue, unsubscribe rate, complaints). You watch them alongside the primary metric so a "win" on one dimension doesn't quietly harm the product or the business.
- **Variance reduction (CUPED)** — use pre-experiment data as a covariate to shrink metric variance, buying more power (smaller detectable effect) for the same `n`. Good to name-drop for a sophisticated answer.

## The whole workflow, in order
1. **Hypothesis** + pick the **primary metric** (and its guardrails) *before* looking at data.
2. **Power analysis** → required sample size / test duration for your MDE (§8).
3. **A/A check** the pipeline and randomization.
4. **Randomize**, then run **long enough** to clear novelty/primacy and cover weekly seasonality (usually ≥ 1–2 full weeks).
5. Verify **sample-ratio match** (did the traffic split come out as designed? a mismatch invalidates the test).
6. Run the **appropriate significance test**; **correct for multiple comparisons** across metrics/segments.
7. Read **effect size + confidence interval**, weigh practical significance against guardrails, then decide.

## Follow-ups
- **"How do you handle interference / network effects?"** — Standard user-level randomization breaks when users influence each other (social feeds, marketplaces, two-sided markets): treatment "leaks" into control. Use **cluster randomization** (randomize communities/regions) or **geo experiments** / switchback designs so arms don't contaminate each other.
- **"The peeking problem?"** — Repeatedly checking a running test and stopping the moment it hits significance massively inflates the false-positive rate (you're multiple-testing over time). Fix with a **pre-committed sample size**, or **sequential testing** methods that give always-valid p-values (e.g. mSPRT, group-sequential boundaries).
- **"Correlation vs causation, one line?"** — Only a **randomized experiment** (or a valid quasi-experimental causal design) licenses a causal claim; any observational correlation can be a confounder in disguise.
- **"Primary metric moved but a guardrail regressed — ship?"** — Usually no, unless the guardrail regression is within a pre-agreed tolerance and the tradeoff is explicitly worth it. Guardrails exist precisely to stop "wins" that hurt elsewhere.
- **"Novelty effect — how do you actually detect it?"** — Plot the treatment effect **over time**; a lift that steadily **decays** toward zero is the novelty signature. Segmenting new vs existing users also separates novelty (new users) from primacy (existing users).
