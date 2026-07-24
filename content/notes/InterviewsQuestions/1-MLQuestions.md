---
title: "ML Interview Questions"
date: 2026-07-24
summary: >-
  My running bank of machine-learning interview questions, worked out in full:
  the bias-variance decomposition, the common losses and optimizers, trees and
  boosting, classic ML, backprop by hand, CNN/RNN/Transformer, MLE statistics,
  and the causal traps behind "why did the A/B test look wrong?". Short answer
  first, then the depth to survive the follow-ups.
tags: [Interview, ML, Foundations, Bias-Variance, Regularization, Deep-Learning, Statistics, Experimentation]
---

This is my working answer bank for ML interviews. Each entry is a question I expect to get, written the way I'd want to answer it on a whiteboard: the short version first, then the depth to survive follow-ups. I pull these from my prep tracker as I go, and I've tried to write each one so I could teach it rather than just recite it, since that's the level an interviewer is actually probing for.

How to read this: the "short version" is what I say first, out loud. Everything after it is the material I reach into when the interviewer asks "okay, but why?".

---

# 1. Bias-variance tradeoff and regularization

The classic "ML breadth" opener. What's really being checked is whether I can reason about generalization error, the gap between training and unseen performance, rather than reciting two definitions.

## Short version

For a model `f̂` trained on a random dataset, the expected squared error at a test point `x` splits into three non-negative pieces:

```
E[(y - f̂(x))²]  =  Bias[f̂(x)]²   +   Var[f̂(x)]   +   σ²
                    (underfit)         (overfit)      (irreducible)
```

**Bias** is how far the average prediction (averaged over all possible training sets) sits from the truth. High bias means the model class is too simple to represent the real function, so it's wrong even with infinite data. That's underfitting, and it shows up as high error on both train and test.

**Variance** is how much the prediction wobbles as the training set changes. High variance means the model is chasing the noise of whichever dataset it happened to see. That's overfitting, and it shows up as low train error but high test error.

**σ² (irreducible)** is noise in the labels and the data-generating process. No model beats this floor, and pretending you can is a sign you're fitting noise.

The tradeoff itself: adding capacity (more parameters, more features, deeper trees) lowers bias but raises variance, while adding constraints (regularization, simpler models, more data) lowers variance but usually raises bias. Total error is a U-shape in complexity, and the job is to sit at the bottom.

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

Let the truth be `y = f(x) + ε` with `E[ε]=0` and `Var(ε)=σ²`. For a prediction `f̂(x)`, which is random because the training set is random, add and subtract the mean prediction `E[f̂]`:

```
E[(y - f̂)²] = E[(f + ε - f̂)²]
            = (f - E[f̂])²      ← Bias²   (systematic error)
            + E[(f̂ - E[f̂])²]  ← Variance (sensitivity to the sample)
            + σ²                ← noise    (cross terms vanish since E[ε]=0)
```

The takeaway an interviewer is after: with finite, noisy data you cannot drive both bias and variance to zero at once. Reducing one term generally inflates the other, so you optimize their sum, not either one alone.

## How I'd actually diagnose a model

| Symptom | Diagnosis | What I'd do |
|---|---|---|
| High train error and high test error | High bias (underfit) | More capacity, better/more features, train longer, weaker regularization |
| Low train error, high test error (big gap) | High variance (overfit) | Regularize, get more data, simpler model, early stopping, ensembling |
| Low train and low test error | Good fit | Ship it |
| Low train, test error stuck at a floor | You've hit irreducible error | Improve the data and labels, not the model |

The tool that separates these is a learning curve: train and validation error plotted against training-set size or epochs. Converged but high means bias. A persistent gap that shrinks as you add data means variance.

## Regularization is really variance reduction

The line I want to land is that every one of these buys a large drop in variance for a small rise in bias. They all constrain the hypothesis space so the model can't fit noise. Seeing them as one idea rather than four unrelated tricks is the point.

**L2 (Ridge, weight decay)** adds `λ Σ wᵢ²` to the loss. It penalizes large weights and shrinks them smoothly toward zero (almost never exactly zero). The penalty's gradient is `2λw`, so every step nudges each weight back toward the origin, which is where the name weight decay comes from. It cuts variance by capping the model's effective capacity: small weights make the learned function smoother, so a small change in an input causes a small change in output, and the fit moves less from one training set to the next. In Bayesian terms it's a Gaussian prior (mean 0) on the weights, with `λ` as the prior's inverse variance, and MAP estimation under that prior is exactly ridge regression. With correlated features L2 spreads weight across the group rather than arbitrarily picking one, which stabilizes the solution. In the linear case there's a closed form, `w = (XᵀX + λI)⁻¹Xᵀy`, and the `+λI` also makes the matrix invertible even when `XᵀX` is singular, which is a nice thing to mention.

**L1 (Lasso)** adds `λ Σ |wᵢ|`. It drives many weights to exactly zero, giving you automatic feature selection and a sparse, readable model. Zeroing a weight literally deletes a parameter, so it shrinks the effective degrees of freedom, and fewer parameters means lower variance. The geometry is worth knowing: minimizing loss subject to a budget `Σ|wᵢ| ≤ t` means the elliptical loss contours expand until they first touch the constraint region. L1's region is a diamond whose corners sit on the axes, so the contour almost always touches at a corner where some coordinates are exactly 0. L2's region is a circle with no corners, so the touch point generically has every coordinate non-zero. In Bayesian terms L1 is a Laplace prior (sharply peaked at 0, heavy tails). Elastic Net (`λ₁Σ|w| + λ₂Σw²`) combines the two, keeping L1's sparsity while handling correlated feature groups the way L2 does instead of arbitrarily keeping one and dropping the rest.

**Dropout** (for neural nets) zeros each unit independently with probability `p` during training, then at test time either uses all units scaled by `(1-p)` or, more commonly, uses inverted dropout (scale by `1/(1-p)` at train time so test time is untouched). It reduces variance in two ways. First, anti-co-adaptation: a neuron can't rely on any specific other neuron being present because that neuron might be dropped, so each one has to learn a feature that's useful on its own, which gives more robust, redundant representations. Second, it's an implicit ensemble: every mini-batch trains a different randomly-thinned sub-network, all sharing weights, and the test-time scaling approximates averaging an exponential number of those sub-networks, which is textbook variance reduction. In practice `p` is around 0.5 for dense layers and 0.1 to 0.2 for inputs. It's largely redundant with batch norm in convnets but still standard in transformers.

**Early stopping** watches validation error each epoch and stops when it quits improving (with a patience window), even while train error keeps dropping. It reduces variance by limiting how far the weights travel from their small initialization. Bounding that distance bounds the effective weight magnitude, which for a quadratic loss is provably close to L2 regularization. Less optimization time means less chance to memorize noise. It's also basically free: no extra penalty term, no search over `λ`, and it saves compute.

## Variance reducers that aren't a penalty on the loss

The most reliable lever is simply more training data, which shrinks `Var[f̂]` directly without adding bias. If you can label more data, do that before anything clever. Bagging and random forests average many high-variance models fit on bootstrap resamples so their independent errors cancel (see §4). Data augmentation is a domain-specific way to manufacture more effective data (flips and crops for images, paraphrase and back-translation for text). Batch normalization normalizes layer inputs, and its use of noisy mini-batch statistics has a mild regularizing side effect.

## Follow-ups I want to be ready for

**"Why does L1 give sparsity but L2 doesn't?"** The diamond-versus-circle geometry above. Corners on the axes give exact zeros.

**"Is boosting bias or variance reduction?"** Mostly bias reduction, since it sequentially fits residuals to build a strong learner out of weak ones. Bagging is the variance-reduction cousin. (See §4.)

**"What's deep double descent?"** For very overparameterized models, test error follows the classic U-curve up to the interpolation threshold, where the model exactly fits the training set, and then decreases again as you keep adding capacity. So the tidy U-curve isn't the whole modern story: enormous nets can generalize well despite fitting the training data perfectly. Worth naming to show you're current.

**"Does regularization increase bias?"** Yes, by construction, and that's the price. It only pays off when the variance reduction outweighs the added bias, which is to say when you're in the overfitting regime.

**"How do you pick `λ` or the dropout rate?"** Cross-validation over a log-spaced grid, watching the validation curve rather than the training loss.

---

# 2. Loss functions: cross-entropy vs MSE vs hinge

The "when would you use each?" question. The real test is whether I connect each loss to the task, to the distributional assumption behind it, and (the part that separates strong answers) to what its gradient does during training.

## Short version

The loss is the quantity optimization actually minimizes, so it has to encode what "good" means for the task. Each of the common losses is the maximum-likelihood estimator under a particular noise model.

| Loss | Task | Per-example form | Implied noise model |
|---|---|---|---|
| MSE | Regression | `(y - ŷ)²` | Gaussian noise on the target |
| Cross-entropy | Classification (want probabilities) | `-Σ yᵢ log p̂ᵢ` | Bernoulli / Categorical target |
| Hinge | Classification (want a max-margin boundary) | `max(0, 1 - y·ŷ)`, `y∈{-1,+1}` | None; it's geometric, not probabilistic |

## Why not just use MSE for everything?

This is the answer that impresses, because it's about gradients, not vibes.

MSE paired with a sigmoid output is a bad fit for classification, for two reasons. The loss surface is non-convex in the weights, so optimization can stall in bad local minima. And the gradient vanishes exactly when you're most wrong: with `ŷ = σ(z)` and MSE, the gradient carries a factor of `σ'(z)`, but `σ'(z)` goes to 0 in the saturated tails, which is precisely where a confidently wrong prediction lives. So the most-wrong examples produce the smallest gradient and barely get corrected, and learning crawls.

Cross-entropy fixes both. Pair a sigmoid (binary) or softmax (multiclass) with CE and the `σ'(z)` term cancels algebraically, leaving a clean gradient:

```
∂L/∂z = p̂ - y
```

Now the gradient is proportional to the error itself. A confidently wrong example (`p̂≈1`, `y=0`) produces the largest possible gradient, which is exactly what you want. CE is also convex in the logits and is the log-likelihood of the class labels, so minimizing it is maximizing likelihood.

## MSE for regression, in full

MSE penalizes error quadratically, so a single large error dominates the loss, which makes it sensitive to outliers. It corresponds to assuming Gaussian residuals, and when that roughly holds it's the right, efficient choice. When outliers are a real concern there are robust alternatives. MAE (L1 loss, `|y - ŷ|`) is robust because the penalty is linear, but it's non-differentiable at 0 and gives a constant gradient magnitude, which can hurt convergence near the optimum; it estimates the median rather than the mean. Huber loss is quadratic within `|error| < δ` and linear beyond, so it's smooth near 0 and robust in the tails, which makes it the pragmatic default for noisy regression, with `δ` setting the crossover. Log-cosh behaves like a smooth Huber, and quantile loss lets you predict intervals instead of a point.

## Hinge vs cross-entropy

Hinge cares only about the margin. Once a point is correctly classified and past the margin (`y·ŷ ≥ 1`), its loss is exactly 0 and it stops influencing the boundary, which is what gives SVMs their sparse support-vector solution. The downside is no probabilities, and it's non-differentiable at the hinge, so you use subgradients. Cross-entropy never reaches 0; it keeps pushing correct predictions to be more confident, so every point keeps contributing a shrinking gradient. Use it whenever you need calibrated probabilities for ranking, thresholding, or downstream decisions.

## Related losses worth naming

Label smoothing replaces hard targets like `(1, 0, …, 0)` with `(1-ε, ε/(K-1), …)`. It stops the model from driving logits to ±∞, improves calibration, and acts as mild regularization, and it's standard in modern image and language training. Focal loss, `-(1 - p̂)^γ · log p̂`, is cross-entropy down-weighted on easy examples so training focuses on the hard ones, which is the fix for extreme class imbalance (dense object detection). KL divergence is what you minimize in distillation and variational methods; since CE equals KL plus a constant, minimizing CE against a fixed target minimizes KL. Contrastive, triplet, and InfoNCE losses are for representation learning: pull positives together and push negatives apart in embedding space.

## Follow-ups I want to be ready for

**"Why does softmax pair with cross-entropy specifically?"** Their derivatives combine into the clean `p̂ - y` gradient, and together they form the MLE for a categorical target.

**"Handling class imbalance in plain CE?"** Class-weighting (up-weight the rare class), resampling (over or under), or moving to focal loss. Otherwise the majority class dominates the gradient and the minority gets ignored.

**"MSE vs MAE, which and when?"** MSE if residuals are roughly Gaussian and outliers are rare, since it predicts the mean and penalizes big misses hard. MAE or Huber if the data has heavy tails or outliers, since MAE predicts the median.

**"Is accuracy a loss?"** No. It's non-differentiable (a step function), so you can't backprop through it. You optimize a smooth surrogate like CE or hinge and report accuracy, F1, or AUC.

---

# 3. Optimizers and learning-rate schedules

Almost always phrased as "why Adam over SGD?". The strong answer treats it as a genuine tradeoff rather than a strict win, and explains AdamW.

## Short version

Everything here is gradient descent, `θ ← θ - η·g`. The variants differ on two axes: whether they use momentum (a running memory of past gradients) and whether they adapt the step size per parameter.

| Optimizer | Update idea | State kept | Character |
|---|---|---|---|
| SGD | Step along the mini-batch gradient | none | Noisy, LR-sensitive, crawls in ravines, but well-understood and generalizes well |
| SGD + Momentum | Keep a velocity (EMA of gradients); step along it | 1 buffer | Damps oscillation, accelerates along consistent directions |
| Nesterov | Momentum, but evaluate the gradient at the look-ahead point | 1 buffer | Slightly better-corrected momentum |
| RMSProp | Divide the step by a running RMS of recent gradients, per param | 1 buffer | Adapts step size to each parameter's gradient scale |
| Adam | Momentum (1st moment) plus RMSProp-style scaling (2nd moment) | 2 buffers | Fast, robust to LR choice, the default |
| AdamW | Adam with decoupled weight decay | 2 buffers | The correct default for training modern nets |

## Adam in detail (be able to write the update)

```
m ← β₁·m + (1-β₁)·g            # 1st moment: momentum
v ← β₂·v + (1-β₂)·g²           # 2nd moment: per-param gradient scale
m̂ = m/(1-β₁ᵗ),  v̂ = v/(1-β₂ᵗ)  # bias correction (m,v start at 0)
θ ← θ - η · m̂ / (√v̂ + ε)
```

The 1st moment is momentum, which smooths the trajectory. The 2nd moment gives each parameter its own effective learning rate: parameters with historically large gradients take smaller steps, and small-gradient parameters (say, rare features) take larger steps, which is why Adam handles sparse and unevenly-scaled gradients so well. Bias correction matters early on because `m` and `v` start at 0, so without the `1-βᵗ` correction the first steps would be biased toward zero. The defaults `β₁=0.9, β₂=0.999, ε=1e-8` work across a huge range of problems, and that robustness is much of Adam's appeal.

## Why Adam over SGD, and when not to

Adam's pros are minimal LR tuning, fast early convergence, robustness on messy or sparse gradients, and being the default for transformers and most non-vision deep learning. Its con is that it tends to find sharper minima that can generalize slightly worse. That's why state-of-the-art CNN image classifiers are often still trained with SGD plus momentum plus a strong schedule: slower and more tuning, but flatter minima and a bit better test accuracy. The honest one-liner is that I reach for Adam when I want fast, reliable convergence and anything transformer-shaped, and for SGD plus momentum when I have the tuning budget and want the last bit of generalization, classically in vision.

## AdamW, the detail most people miss

Plain Adam implements L2 regularization by adding `λw` to the gradient. But that penalty then flows through the `√v̂` denominator, so parameters with large gradients get their weight decay shrunk. Decay and adaptive learning rate become coupled, and the regularization ends up weaker and inconsistent across parameters. AdamW decouples them by applying weight decay directly to the weights, `θ ← θ - η(m̂/(√v̂+ε) + λθ)`, independent of the gradient statistics. That small fix measurably improves generalization, which is why AdamW rather than Adam is the real modern default.

## Learning-rate schedules

The LR is the single most important hyperparameter, and how you vary it over training matters as much as its value. Warmup starts near 0 and ramps up over the first few hundred or thousand steps; early gradients are large and noisy, and Adam's `v` estimate is unreliable before it's warmed up, so jumping straight to a high LR can diverge. Warmup is essentially mandatory for transformers. Cosine annealing then decays the LR to near 0 along a cosine curve after warmup: big steps early to explore, tiny steps late to settle into a minimum. The modern default shape is linear warmup followed by cosine decay. Step decay cuts the LR by a factor (say 10x) at fixed milestones, which is classic for SGD-trained CNNs. ReduceLROnPlateau drops the LR when validation stops improving, which is robust and hands-off. Cyclical schedules and warm restarts (SGDR) periodically jump the LR back up to escape sharp minima and explore multiple basins, and you can ensemble the snapshots.

## Follow-ups

**"Why does momentum help?"** In a ravine (steep across, shallow along), plain SGD zig-zags off the walls, while momentum averages out the transverse oscillation and builds speed along the valley floor, so you reach the minimum faster.

**"Batch size and LR relationship?"** A bigger batch means lower gradient noise, so you can use a proportionally larger LR (the linear scaling rule), usually with a longer warmup to stay stable.

**"Exploding or vanishing gradients in training?"** Gradient clipping (cap the global gradient norm) for exploding; good init, normalization, residual connections, and ReLU for vanishing (see §6).

**"What if loss diverges to NaN?"** Usually the LR is too high, there's no warmup, the init is bad, or something overflowed (missing `ε`, unnormalized inputs, fp16 without loss scaling). Lower the LR or add warmup first.

**"Second-order methods?"** Newton and L-BFGS use curvature (the Hessian) for better steps but are too memory-heavy for deep nets. Adam's 2nd moment is a cheap diagonal approximation to that idea.

---

# 4. Trees and boosting

The classic-ML staple, and still the winner on tabular data. The one idea to nail: bagging reduces variance, boosting reduces bias, and gradient boosting is sequentially fitting the residuals.

## Decision tree, the base learner

A tree recursively splits the feature space along the feature and threshold that most reduces impurity: Gini or entropy for classification, variance (MSE) for regression. Each leaf predicts the majority class or the mean of the points that land in it. Gini is `1 - Σ pₖ²` and entropy is `-Σ pₖ log pₖ`; both measure how mixed a node is, and the split maximizes the impurity decrease (information gain). Trees are interpretable (you can read the rules), handle mixed numeric and categorical features, need no feature scaling, capture nonlinear interactions automatically, and are robust to monotone feature transforms. The weakness is that a single deep tree is high variance: small changes in the data flip the splits and it memorizes noise. That's exactly what the two ensemble families fix, in opposite ways.

## Random forest, bagging (variance reduction)

Train many deep trees, each on a bootstrap sample of the rows, and at each split consider only a random subset of features (typically `√d` for classification), then average the predictions (vote for classification). The feature subsampling is the key part. Without it, one dominant feature would be the top split in nearly every tree and the trees would be highly correlated, and averaging correlated predictors barely reduces variance. Random feature subsets decorrelate the trees so their errors are more independent and cancel on averaging. The net effect is a big variance reduction with negligible change in bias, and adding more trees never overfits (it just converges), which makes a random forest a forgiving, strong baseline. You also get out-of-bag error for free: each tree's bootstrap leaves about 37% of rows unused, giving you a validation estimate without a holdout set.

## Gradient boosting, sequential residual-fitting (bias reduction)

Start with a constant prediction, then repeatedly fit a new shallow tree to the negative gradient of the loss with respect to current predictions (for MSE that's just the residuals `y - ŷ`), and add it in scaled by a learning rate `η`, so `F_m = F_{m-1} + η·h_m`. It reduces bias because each tree explicitly corrects the ensemble's current errors, so a sequence of weak learners composes into a strong one and the bias falls each round. It can overfit precisely because it keeps chasing residuals, so it eventually fits noise; you control that with a small learning rate (shrinkage, say 0.01 to 0.1), shallow trees (depth 3 to 8), subsampling of rows and columns (stochastic gradient boosting), and early stopping on a validation set. Put simply, boosting uses shallow high-bias trees and reduces bias sequentially, while bagging uses deep high-variance trees and reduces variance in parallel.

## XGBoost, LightGBM, CatBoost

XGBoost uses a second-order (Newton) Taylor expansion of the loss, gradients and Hessians, for better split choices, plus an explicit regularization term on the number of leaves and the leaf weights, with level-wise tree growth. It was the long-time Kaggle workhorse. LightGBM uses leaf-wise growth (always split the leaf with the largest loss reduction) and histogram binning of continuous features (bucket the values so split-finding is O(bins) instead of O(rows)), which makes it much faster and more memory-efficient on large data; it can overfit small datasets, so cap depth and num_leaves. CatBoost brings principled native categorical handling (ordered target statistics) and ordered boosting to reduce target leakage, and it shines when you have many categorical features.

## The comparison that ties it together

| | Random Forest (bagging) | Gradient Boosting |
|---|---|---|
| Trees built | In parallel, independent | Sequentially, each fixes the last |
| Base tree | Deep (low bias, high variance) | Shallow (high bias, low variance) |
| Primarily reduces | Variance | Bias |
| Overfit risk | Low; more trees is safe | Higher; needs LR, depth cap, early stop |
| Tuning effort | Forgiving | More sensitive, but higher accuracy ceiling |
| Parallelism | Trivially parallel | Sequential (though split-finding parallelizes) |

## Follow-ups

**"RF vs GBM in practice?"** Tuned GBM usually wins on tabular accuracy, while RF is the safer, near-zero-tuning baseline. I often start with RF to get a number, then push with LightGBM or XGBoost.

**"Why must boosting use weak learners?"** Each round only needs to be slightly better than chance, and shallow trees keep per-round variance low so the sequential bias reduction doesn't spiral into overfitting. Deep trees would overfit within a couple of rounds.

**"Trees vs neural nets?"** Gradient-boosted trees remain the default on structured, tabular data because they handle mixed types, need no scaling, and train fast. Neural nets dominate unstructured data (images, text, audio) where representation learning matters.

**"Feature importance, how and the caveat?"** Impurity-based importance is fast but biased toward high-cardinality and continuous features. Permutation importance (shuffle a feature and measure the accuracy drop) is model-agnostic and more honest, and SHAP gives game-theoretic per-prediction attributions. Prefer permutation or SHAP for anything you'll act on.

**"AdaBoost vs gradient boosting?"** AdaBoost re-weights misclassified examples each round, while gradient boosting fits the gradient of an arbitrary differentiable loss, which is the more general formulation that AdaBoost is a special case of.

---

# 5. Classic ML: SVM, k-means, PCA

The "one line each, what problem does it solve?" round. Interviewers use these three to check breadth quickly, then dig into whichever I sound shakiest on.

## SVM, the maximum-margin classifier

Among all separating hyperplanes, an SVM finds the one that maximizes the margin, the distance from the boundary to the nearest point of either class. The intuition is that the widest "street" between classes generalizes best because it's the most robust to perturbations. Only the points on the margin (and any violators) determine the boundary; remove any other point and the solution is unchanged, so the model is defined by a sparse set of support vectors, which is both elegant and memory-efficient. Real data isn't cleanly separable, so the soft margin allows violations with slack variables penalized by `C`. A small `C` means a wide margin that tolerates more violations, which is more regularization (higher bias, lower variance), while a large `C` fits the training data harder (lower bias, higher variance); `C` is the bias-variance knob. The elegant part is the kernel trick: the SVM only ever needs dot products between points, so you replace `xᵢ·xⱼ` with a kernel `K(xᵢ,xⱼ)`, say RBF `exp(-γ‖xᵢ-xⱼ‖²)` or polynomial, and you implicitly work in a very high (even infinite) dimensional feature space, getting a nonlinear boundary without ever computing the coordinates there. For RBF, `γ` controls reach: a large `γ` makes each point's influence local, giving a wiggly boundary that risks overfitting. One nice unifying point to drop is that the soft-margin SVM is exactly hinge loss plus L2 regularization (§2).

## k-means, unsupervised clustering

k-means partitions data into `k` clusters that minimize the within-cluster sum of squared distances (inertia) to the cluster centers. Lloyd's algorithm initializes `k` centroids, assigns each point to its nearest centroid, updates each centroid to the mean of its assigned points, and repeats until assignments stop changing. It's coordinate descent on the inertia objective, so it converges, but only to a local optimum. To choose `k`, use the elbow method (plot inertia against `k` and look for the diminishing-returns kink) or the silhouette score (how well-separated the clusters are); there's no "correct" `k` without a criterion. Know the failure modes: it's sensitive to initialization, so use k-means++ to spread the initial centroids; and it assumes spherical, similarly-sized, similar-density clusters under Euclidean distance, so it fails on elongated, nested, or varying-density shapes, where you'd reach for DBSCAN (density-based, finds arbitrary shapes and outliers) or Gaussian Mixture Models (soft, elliptical clusters via EM). Standardize features first, since it's distance-based.

## PCA, linear dimensionality reduction

PCA finds a lower-dimensional linear subspace that keeps the maximum variance of the data, useful for compression, denoising, decorrelation, and 2-D or 3-D visualization. The principal components are the eigenvectors of the covariance matrix (equivalently the right singular vectors from the SVD of the mean-centered data), ordered by eigenvalue, which is the variance explained; you project the data onto the top `k`. The first PC is the direction of greatest variance, and each next PC is orthogonal to the previous ones and captures the most remaining variance. A few details matter. Standardize first, because PCA is scale-sensitive and a feature in large units would otherwise dominate purely because of its scale. The components are orthogonal linear combinations of the original features, so they're decorrelated but lose interpretability. PCA is unsupervised: it maximizes variance, which isn't necessarily the direction that best separates classes, so when you have labels and want discriminative axes, use LDA instead. Choose `k` via the explained-variance ratio (keep enough PCs to retain, say, 95% of variance), reading it off the scree plot. The main limitation is that PCA is linear: data on a curved manifold (a swiss roll) needs nonlinear methods like kernel PCA, t-SNE or UMAP (for visualization), or autoencoders.

## Follow-ups

**"PCA vs autoencoder?"** A linear autoencoder with MSE recovers the same subspace as PCA. Nonlinear autoencoders (and t-SNE or UMAP) capture curved manifolds PCA can't; the price is no closed form and less interpretability.

**"When does k-means fail, concretely?"** Two crescent-moon clusters: k-means splits them straight down the middle because it can only draw spherical (Voronoi) boundaries, whereas DBSCAN gets them right.

**"Why is the kernel trick efficient?"** It computes the high-dimensional inner product directly through `K(·,·)` without ever materializing the high-dimensional coordinates, sidestepping the curse of dimensionality on both compute and memory.

**"Is k-means guaranteed optimal?"** No, it only reaches a local optimum of a non-convex objective. Run it several times with different seeds (k-means++) and keep the lowest inertia.

**"PCA before a classifier, good idea?"** Sometimes, since it denoises, decorrelates, and speeds training, but it can throw away low-variance directions that are actually discriminative. Validate that it doesn't hurt downstream accuracy.

---

# 6. Backprop from scratch

The single most-tested deep-learning fundamental. Deriving the chain rule through a 2-layer MLP on paper proves I understand how training actually works, not just that I can call `.backward()`.

## Short version

Backprop is the chain rule applied efficiently through dynamic programming. The forward pass computes and caches each layer's activations, and the backward pass propagates the loss gradient from output back to input, reusing each layer's computed gradient to get the previous layer's. That reuse is why training a network costs roughly one extra forward pass rather than something exponential in depth.

## The 2-layer MLP, worked by hand

Forward pass (one hidden layer, softmax output, cross-entropy loss):
```
z1 = W1·x + b1          a1 = σ(z1)          # hidden pre-activation, activation
z2 = W2·a1 + b2         ŷ  = softmax(z2)    # output logits, probabilities
L  = -Σ yₖ log ŷₖ                            # cross-entropy
```

Backward pass, applying the chain rule layer by layer from output to input:
```
dz2 = ŷ - y                     # softmax + CE collapse to this clean form
dW2 = dz2 · a1ᵀ                 # outer product: (out×1)(1×hidden)
db2 = dz2
da1 = W2ᵀ · dz2                 # push the error back through W2
dz1 = da1 ⊙ σ'(z1)             # ⊙ elementwise; through the activation's local slope
dW1 = dz1 · xᵀ
db1 = dz1
```

Update with gradient descent: `W ← W - η·dW`, `b ← b - η·db`.

The pattern that makes it click, and that generalizes to any depth: each layer receives an upstream gradient, multiplies by its own local gradient (the derivative of the operation it performed), and passes the product downstream. Repeat per layer. A matmul's local gradient is a matmul, an elementwise activation's local gradient is elementwise, and that's the whole algorithm.

## Why the details matter

Vanishing gradients come from the backward pass multiplying by the activation's derivative `σ'(z)` at each layer. For sigmoid or tanh, `σ' ≤ 0.25`, so stacking `L` layers multiplies `L` sub-1 numbers and the gradient reaching early layers is exponentially tiny, so those layers barely learn. The fixes: ReLU (`max(0,z)`) has a derivative of exactly 1 for positive inputs, so it doesn't shrink the gradient (its own issue is dead units for negative inputs, hence LeakyReLU and GELU); residual connections, `y = x + f(x)`, give the gradient an additive shortcut path that never vanishes, which is what made very deep nets like ResNets and transformers trainable; and batch or layer norm keeps activations in a healthy range so derivatives don't saturate. Exploding gradients are the mirror case (products greater than 1), fixed with gradient clipping and careful initialization. As for why you cache activations: the backward pass needs the forward values (`a1`, `σ'(z1)`, the inputs to each matmul), and storing them is the classic memory-versus-compute tradeoff, which is exactly what activation checkpointing trades back the other way by recomputing in the backward pass to save memory.

## Weight initialization

Initializing all weights to zero is fatal: every neuron in a layer would compute the same thing and get the same gradient, so they'd stay identical forever and symmetry never breaks. You need random init. Xavier/Glorot (for tanh or sigmoid) and He (for ReLU) scale the initial variance by the layer's fan-in and fan-out so activation and gradient variance stay roughly constant across depth, which prevents both vanishing and exploding signals at step 0.

## Follow-ups

**"Why is softmax+CE's gradient exactly `ŷ - y`?"** The softmax Jacobian and the cross-entropy derivative cancel algebraically. It's the same reason CE is the right classification loss (§2), and it's why frameworks fuse the two into one numerically-stable op.

**"What is autograd doing?"** Building a computation graph on the forward pass, then applying these local-gradient rules in reverse-topological order. That's reverse-mode automatic differentiation, and backprop is exactly reverse-mode autodiff specialized to a scalar loss.

**"LR too high vs too low?"** Too high and the steps overshoot the minimum, so the loss oscillates or diverges to NaN. Too low and training crawls and can stall on plateaus or saddle points.

**"Do you update on every example?"** Usually on mini-batches: a batch averages the per-example gradients, trading gradient noise (which helps generalization and escaping saddles) against hardware efficiency.

**"Batch norm's effect on backprop?"** It normalizes each layer's inputs, smoothing the loss landscape so gradients are better-behaved and larger LRs become usable, and it also injects mini-batch noise as mild regularization.

---

# 7. Deep-learning architectures: CNN, RNN/LSTM, Transformer

The training-side architecture round. The framing that scores: for each model, what inductive bias does it bake in, and what problem does that bias solve?

## CNN, exploiting spatial structure

The two core operations are convolution and pooling. Convolution slides a small learned filter across the input computing dot products, with the same weights used at every location (weight sharing), so one filter detects one pattern (an edge, a texture) wherever it appears. Pooling (max or average) downsamples spatially, giving small translation tolerance and cutting compute. The inductive biases are locality (nearby pixels are related, so filters are small) and translation equivariance (a feature is detected the same way anywhere in the image), and weight sharing means far fewer parameters than a dense layer over pixels, which is why CNNs are data-efficient on images. The receptive field, the region of the input that influences a given output unit, grows with depth and with stride and pooling, so early layers respond to edges and textures while deep layers respond to object parts and whole objects. CNNs solve images and any grid-structured signal (spectrograms, some time series) efficiently and with a strong prior.

## RNN and LSTM, exploiting sequential structure

An RNN keeps a hidden state `hₜ = f(Wₓxₜ + Wₕhₜ₋₁)` passed step to step, applying the same weights at every timestep, which gives a built-in bias for order and recurrence and handles variable-length sequences naturally. The core problem is that training uses backprop-through-time, which multiplies by the recurrent weight repeatedly across timesteps, so you get vanishing gradients (can't learn long-range dependencies) or exploding gradients, and vanilla RNNs effectively forget anything more than about ten steps back. The LSTM fix adds a cell state `cₜ`, a kind of memory conveyor belt, plus three gates: forget (what to erase), input (what new information to write), and output (what to expose as the hidden state). The cell state updates additively, `cₜ = fₜ⊙cₜ₋₁ + iₜ⊙c̃ₜ`, so gradients flow across many timesteps without repeated multiplicative shrinkage, and that additive path is the whole reason LSTMs learn long-range dependencies where RNNs fail. The GRU is a lighter two-gate variant (it merges cell and hidden, and forget and input) with fewer parameters and similar performance. These models solve sequences (text, time series, audio) and were historically dominant, now largely replaced by transformers wherever you can afford attention.

## Transformer, attention over the whole sequence at once

In self-attention, each token produces three vectors through learned projections: Query, Key, and Value. Attention weights are `softmax(QKᵀ / √d_k)`, and each token's output is the weighted sum of all tokens' Values, so every token attends directly to every other token in one operation, with no recurrence and therefore no distance-based gradient decay; long-range dependencies are as easy to learn as short ones. You divide by `√d_k` because dot products grow in magnitude with dimension, and without scaling they push softmax into saturated regions where gradients vanish, so `√d_k` keeps the logits at a sane scale. Multi-head attention runs attention `h` times in parallel in different learned subspaces and concatenates the results, so different heads can specialize (syntax, coreference, positional patterns) and the model attends to several kinds of relationships at once. Attention is permutation-invariant, with no notion of token order on its own, so you inject position through fixed sinusoidal encodings (the original), learned embeddings, or rotary embeddings (RoPE) in modern LLMs; without it, "dog bites man" and "man bites dog" look identical. A full block is attention, then residual and LayerNorm, then a position-wise feed-forward network, then residual and LayerNorm again, and the residuals (see §6) and LayerNorm are what make deep stacks trainable. Transformers won because they're fully parallelizable across the sequence, unlike RNNs which process tokens one at a time, so they train efficiently on massive data and hardware while capturing long-range dependencies directly; that parallel training plus scaling is the whole story of modern LLMs. The cost is that self-attention is O(n²) in sequence length, since every token attends to every token, which is the bottleneck every efficiency paper attacks (FlashAttention makes it IO-efficient; sparse and linear-attention variants change the complexity).

## Follow-ups

**"CNN vs Transformer for vision?"** A Vision Transformer (ViT) splits an image into patches and treats them as tokens. It beats CNNs at scale but needs more data and regularization because it lacks the CNN's built-in locality prior and has to learn that structure.

**"Why did Transformers replace LSTMs for NLP?"** Parallel training (a huge speedup on modern hardware) plus direct long-range attention. LSTMs are inherently sequential and still degrade over long contexts.

**"Encoder vs decoder vs encoder-decoder?"** An encoder uses bidirectional self-attention and sees the whole input (BERT, understanding and classification). A decoder uses causal, masked attention and can only see past tokens (GPT, generation). An encoder-decoder is seq2seq with cross-attention (T5, translation, summarization).

**"Self-attention vs cross-attention?"** In self-attention, Q, K, and V all come from the same sequence. In cross-attention, Q comes from one sequence (say the decoder) and K and V from another (the encoder output), which is how the decoder reads the source.

**"Batch norm vs layer norm, why do transformers use LayerNorm?"** BatchNorm normalizes across the batch dimension, which is unstable for variable-length sequences and small or streaming batches. LayerNorm normalizes across features per token, independent of batch, which fits sequence models.

---

# 8. Statistics for MLE

The backbone of every experimentation and A/B-testing question. Interviewers want precise definitions, the p-value one especially, because sloppy stats is where a lot of self-described MLEs get exposed.

## The core vocabulary, said precisely

The p-value is the probability of observing data at least as extreme as what you saw, assuming the null hypothesis is true. It is emphatically not the probability the null is true, not the probability your result happened by chance, and not `1 - P(alternative)`. Nailing this exact wording is half the test.

The null hypothesis `H₀` is the status quo or "no effect", and the alternative `H₁` is what you're arguing for. You never prove `H₀`; you either reject it or fail to reject it. The significance level `α` is the p-value threshold below which you reject `H₀` (commonly 0.05), and it's the false-positive rate you're willing to accept, chosen before the test.

A 95% confidence interval is a range produced by a procedure that, across many repeated experiments, contains the true parameter 95% of the time. State the subtlety carefully: for any single computed interval the parameter is either in it or not, so the 95% is a property of the procedure, not a probability about that one interval. (A CI that excludes 0 corresponds to a significant result at the matching `α`.)

A Type I error (false positive, rate `α`) is rejecting a true null, declaring an effect that isn't real. A Type II error (false negative, rate `β`) is failing to reject a false null, missing a real effect. Statistical power (1 − β) is the probability of detecting an effect that genuinely exists, and the convention is to aim for 0.80. Low power is why many "no significant difference" results are really just underpowered.

The two error types in one grid:

| | `H₀` true (no real effect) | `H₀` false (real effect) |
|---|---|---|
| Reject `H₀` | Type I error (α) ✗ | Correct, power (1−β) ✓ |
| Fail to reject | Correct (1−α) ✓ | Type II error (β) ✗ |

## Power drives sample-size planning

Power goes up with a larger sample size `n`, a larger true effect size, lower variance, and a higher `α` (you accept more false positives to catch more true ones). Before running a test you do a power analysis to compute the `n` needed to detect the minimum effect you'd care about (the MDE) at your chosen `α` and power. Skip that and an "inconclusive" test tells you nothing, because you may simply not have collected enough data. This is the single most common practical stats question for MLE and experimentation roles.

## The multiple-comparisons trap

Run 20 independent tests at `α = 0.05` and you expect about one false positive by chance alone, since `P(at least one) = 1 - 0.95²⁰ ≈ 64%`. Any time a dashboard slices an experiment across many metrics or segments, this bites. Two corrections: Bonferroni tests each hypothesis at `α/m`, which is simple and controls the family-wise error rate but is conservative and kills power when `m` is large; Benjamini-Hochberg controls the false discovery rate (the expected fraction of false positives among rejections), which is less conservative and the standard choice when you have many tests.

## Distributions and tests to have ready

The central limit theorem says the sampling distribution of the mean approaches Normal as `n` grows, regardless of the underlying distribution, which is why `t` and `z` tests on means work even for non-normal data at large `n`. A t-test compares two means (small-to-moderate `n`, unknown variance). A z-test or chi-square compares proportions and rates (CTR, conversion). Mann-Whitney U and Wilcoxon are non-parametric and compare distributions via ranks when data is skewed or heavy-tailed (see §9). ANOVA compares more than two group means at once, followed by post-hoc tests with correction.

## Follow-ups

**"p = 0.049 vs 0.051, meaningfully different?"** No. 0.05 is an arbitrary convention, so treat the p-value as continuous evidence and weigh effect size and practical significance alongside it. A knife-edge p-value is fragile.

**"Statistical vs practical significance?"** With huge `n`, a trivially small, business-irrelevant effect can be highly statistically significant. Always report the effect size and CI, not just "p < 0.05", and decide on practical grounds.

**"One-tailed vs two-tailed?"** Two-tailed tests for a difference in either direction, which is the safe default. Use one-tailed only when you truly care about a single direction and commit to it in advance, since it's easier to reach significance and therefore easy to abuse.

**"Bayesian alternative?"** Report the posterior or a credible interval ("95% probability the parameter is in this range", which actually is the intuitive statement people wrongly attribute to CIs), or `P(treatment > control)`, which is often more directly useful for a decision than a p-value.

**"Frequentist vs Bayesian in one line?"** Frequentist treats parameters as fixed and data as random (long-run frequencies); Bayesian treats parameters as random and updates a prior into a posterior with data.

---

# 9. Experimentation and causal traps

The "why did the A/B test look wrong?" question, where MLE meets product judgment. The interviewer wants to hear the correlation-versus-causation traps by name, with the tell for each.

## The traps, each with the tell and the fix

Simpson's paradox is when a trend visible in every subgroup reverses once the groups are pooled (or the other way around), because of an unbalanced lurking variable. A concrete example: a treatment has a higher recovery rate than control among mild patients and among severe patients, yet a lower rate overall, because far more severe patients happened to receive the treatment and severe cases recover less no matter what, so the aggregate is reversed by the case-mix. The fix is to segment by the confounding variable and not trust the pooled number when group composition differs across arms.

Confounders are variables that influence both the treatment assignment and the outcome, manufacturing a spurious association (ice-cream sales "cause" drownings; the confounder is summer). The fix is randomization, which balances confounders, known and unknown, across arms in expectation, and that's the entire reason RCTs license causal claims. When you can't randomize, you control through stratification, matching, regression adjustment, or causal-inference designs (instrumental variables, difference-in-differences, propensity scores).

An A/A test runs the "experiment" with both arms identical. If it shows a significant difference, the pipeline is broken: bad randomization or hashing, sample-ratio mismatch, logging bugs, or a mis-specified metric. It's the sanity check you run before trusting any A/B result, and it also empirically calibrates your false-positive rate.

A novelty effect is when a new feature gets a temporary bump purely because it's new and users click to explore, and the lift decays as the novelty wears off, so shipping on week-1 numbers over-credits the change. The primacy effect is the opposite: existing users are habituated to the old experience and underperform on the new one at first, then adapt and improve. Both mean you have to run the test long enough to reach steady state before reading it.

## The A/B testing statistics toolkit

A t-test compares two group means (average revenue per user, average session length) and relies on approximate normality of the mean, which the CLT gives you at large `n`. Mann-Whitney U is the non-parametric counterpart, comparing distributions via ranks; reach for it when the metric is skewed or heavy-tailed (revenue, watch time, latency), where a few whales make the mean misleading. Chi-square or a z-test of proportions covers rate metrics (CTR, conversion, retention). Guardrail metrics are the ones that must not regress even if the target metric improves (latency, crash rate, revenue, unsubscribe rate, complaints); you watch them alongside the primary metric so a win on one dimension doesn't quietly harm the product or the business. CUPED uses pre-experiment data as a covariate to shrink metric variance, buying more power (a smaller detectable effect) for the same `n`, and it's a good thing to name-drop for a sophisticated answer.

## The whole workflow, in order

1. Write the hypothesis and pick the primary metric (and its guardrails) before looking at data.
2. Run a power analysis to get the required sample size and test duration for your MDE (§8).
3. A/A check the pipeline and randomization.
4. Randomize, then run long enough to clear novelty and primacy and to cover weekly seasonality, usually at least one to two full weeks.
5. Verify the sample-ratio match: did the traffic split come out as designed? A mismatch invalidates the test.
6. Run the appropriate significance test and correct for multiple comparisons across metrics and segments.
7. Read the effect size and confidence interval, weigh practical significance against the guardrails, then decide.

## Follow-ups

**"How do you handle interference or network effects?"** Standard user-level randomization breaks when users influence each other (social feeds, marketplaces, two-sided markets), because treatment leaks into control. Use cluster randomization (randomize communities or regions) or geo experiments and switchback designs so the arms don't contaminate each other.

**"The peeking problem?"** Repeatedly checking a running test and stopping the moment it hits significance massively inflates the false-positive rate, since you're multiple-testing over time. Fix it with a pre-committed sample size, or with sequential testing methods that give always-valid p-values (mSPRT, group-sequential boundaries).

**"Correlation vs causation, one line?"** Only a randomized experiment, or a valid quasi-experimental causal design, licenses a causal claim; any observational correlation can be a confounder in disguise.

**"Primary metric moved but a guardrail regressed, do you ship?"** Usually not, unless the guardrail regression is within a pre-agreed tolerance and the tradeoff is explicitly worth it. Guardrails exist precisely to stop wins that hurt elsewhere.

**"Novelty effect, how do you actually detect it?"** Plot the treatment effect over time; a lift that steadily decays toward zero is the novelty signature. Segmenting new versus existing users also separates novelty (new users) from primacy (existing users).
