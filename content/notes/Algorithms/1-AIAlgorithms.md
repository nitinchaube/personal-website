---
title: "AI / ML Algorithms"
date: 2026-06-22
summary: "Single-source, from-scratch Python + math reference for the ML / DL / AI algorithms you get asked to derive and code in Google-level interviews — classic ML, neural nets, attention, LLM decoding, embeddings, generative models, metrics and RL."
tags: [ML, DL, AI, Algorithms, Interview]
---

This is my **single source of truth** for ML/DL/AI interview prep. In these interviews you are rarely asked to call `sklearn` or `torch` — you are asked to **derive the math and implement it**: softmax that doesn't overflow, backprop by hand, scaled dot-product attention, top-p sampling, ROC-AUC, gradient boosting.

Every block below has: the **formula**, a plain-`numpy` (or pure Python) **implementation**, a tiny **runnable example**, and the **complexity / interview gotcha**. Read top-to-bottom once, then use it as a night-before cheat-sheet.

> **The universal mental model:** almost every ML algorithm = **(1)** a model $f(x; \theta)$, **(2)** a loss $L$, **(3)** a rule to push $\theta$ down $\nabla_\theta L$. Master those three and most "implement X" questions collapse into the same template.

---

# 0. Math & Numerical Building Blocks

These show up *inside* almost every other algorithm. Interviewers love them because one missing `max`-subtraction reveals whether you actually understand numerical stability.

## Sigmoid (numerically stable)

$$
\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma'(z) = \sigma(z)\,(1 - \sigma(z))
$$

```python
import numpy as np

def sigmoid(z):
    # Naive 1/(1+exp(-z)) overflows for large negative z.
    # Handle the two regimes so exp() never sees a large positive argument.
    pos = z >= 0
    out = np.empty_like(z, dtype=float)
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out

# Example
print(sigmoid(np.array([-1000., 0., 1000.])))   # [0.  0.5  1.] — no overflow
```

## Softmax (the #1 "show me numerical stability" question)

$$
\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_j e^{x_j}} = \frac{e^{x_i - c}}{\sum_j e^{x_j - c}}, \quad c = \max_j x_j
$$

The right-hand identity is the whole trick: softmax is invariant to shifting all logits by a constant, so subtract the max to keep `exp` in range.

```python
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)   # shift for stability
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

# Example — shift invariance
logits = np.array([2.0, 1.0, 0.1])
print(softmax(logits))            # [0.659  0.242  0.099]
print(softmax(logits + 1000))     # identical
```

## LogSumExp (stable log of a sum of exps)

$$
\text{LSE}(x) = \log \sum_i e^{x_i} = c + \log \sum_i e^{x_i - c}, \quad c = \max_i x_i
$$

```python
def logsumexp(x, axis=-1):
    c = np.max(x, axis=axis, keepdims=True)
    return (c + np.log(np.sum(np.exp(x - c), axis=axis, keepdims=True))).squeeze(axis)

# log(e^1000 + e^1000) = 1000 + log(2), with no overflow
print(logsumexp(np.array([1000., 1000.])))   # 1000.693...
```

## Cross-Entropy Loss (from logits)

$$
L = -\sum_i y_i \log p_i, \qquad p = \text{softmax}(z) \;\Rightarrow\; L = \text{LSE}(z) - z_{\text{correct}}
$$

```python
def cross_entropy_from_logits(logits, labels):
    # logits: (N, C); labels: (N,) integer class ids.
    N = logits.shape[0]
    lse = logsumexp(logits, axis=1)
    correct = logits[np.arange(N), labels]
    return np.mean(lse - correct)

logits = np.array([[2.0, 0.5, 0.1], [0.2, 1.5, 0.3]])
labels = np.array([0, 1])
print(cross_entropy_from_logits(logits, labels))   # 0.385...
```

## KL Divergence (how far apart two distributions are)

$$
D_{\mathrm{KL}}(P \,\|\, Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)} \;\ge\; 0
$$

Asymmetric: $D_{KL}(P\|Q) \ne D_{KL}(Q\|P)$. Cross-entropy $= H(P) + D_{KL}(P\|Q)$, which is why minimizing CE w.r.t. the model = minimizing KL to the data distribution.

```python
def kl_divergence(p, q, eps=1e-12):
    p, q = np.asarray(p), np.asarray(q)
    return np.sum(p * np.log((p + eps) / (q + eps)))

print(round(kl_divergence([0.5, 0.5], [0.9, 0.1]), 4))   # 0.5108
```

## Activations & their derivatives (needed for backprop)

$$
\text{ReLU}(z)=\max(0,z),\quad \tanh'(z)=1-\tanh^2(z),\quad \text{GELU}(z)\approx 0.5z\!\left(1+\tanh\!\big[\sqrt{\tfrac{2}{\pi}}(z+0.044715 z^3)\big]\right)
$$

```python
def relu(z):       return np.maximum(0, z)
def relu_grad(z):  return (z > 0).astype(float)
def tanh(z):       return np.tanh(z)
def tanh_grad(z):  return 1 - np.tanh(z) ** 2
def gelu(z):       return 0.5 * z * (1 + np.tanh(np.sqrt(2/np.pi) * (z + 0.044715 * z**3)))
def sigmoid_grad(z):
    s = sigmoid(z); return s * (1 - s)
```

## Distance / similarity metrics

$$
\text{cos}(a,b)=\frac{a\cdot b}{\lVert a\rVert\,\lVert b\rVert},\qquad d_{L2}=\sqrt{\textstyle\sum_i (a_i-b_i)^2},\qquad d_{L1}=\textstyle\sum_i |a_i-b_i|
$$

```python
def euclidean(a, b): return np.sqrt(np.sum((a - b) ** 2))
def manhattan(a, b): return np.sum(np.abs(a - b))
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

print(cosine_similarity(np.array([1, 0]), np.array([10, 0])))   # 1.0 (direction only)
```

## Weight Initialization (Xavier / He)

$$
\text{Xavier: } \sigma = \sqrt{\frac{2}{n_{in}+n_{out}}} \;(\text{tanh/sigmoid}), \qquad \text{He: } \sigma = \sqrt{\frac{2}{n_{in}}} \;(\text{ReLU})
$$

```python
def init_weights(d_in, d_out, mode="he", seed=0):
    rng = np.random.default_rng(seed)
    std = np.sqrt(2/(d_in+d_out)) if mode == "xavier" else np.sqrt(2/d_in)
    return rng.normal(0, std, (d_in, d_out))
```
*Why it matters: bad init → activations/gradients vanish or explode across layers. He keeps ReLU variance ~1 per layer.*

## Numerical Gradient Checking (verify your backprop)

$$
\frac{\partial f}{\partial x_i} \approx \frac{f(x_i+\epsilon) - f(x_i-\epsilon)}{2\epsilon} \quad (\text{central difference, } O(\epsilon^2)\text{ error})
$$

```python
def grad_check(f, x, analytic_grad, eps=1e-6):
    num = np.zeros_like(x)
    for i in range(x.size):
        old = x.flat[i]
        x.flat[i] = old + eps; fp = f(x)
        x.flat[i] = old - eps; fm = f(x)
        x.flat[i] = old
        num.flat[i] = (fp - fm) / (2 * eps)
    return np.max(np.abs(num - analytic_grad))   # should be ~1e-7

x = np.array([1., 2., 3.])
print(grad_check(lambda v: np.sum(v**2), x.copy(), 2 * x))   # ~8e-10
```

---

# 1. Optimizers & Gradient Descent

**Interview framing:** "Implement SGD with momentum" / "What does Adam actually do?" Write the update rules from memory.

## Gradient Descent (batch / mini-batch / stochastic)

$$
\theta \leftarrow \theta - \eta \, \nabla_\theta L(\theta)
$$

```python
def gradient_descent(grad_fn, theta, lr=0.1, steps=100):
    for _ in range(steps):
        theta = theta - lr * grad_fn(theta)
    return theta

# minimize (x-3)^2, grad = 2(x-3)
print(round(gradient_descent(lambda x: 2*(x-3), 0.0, lr=0.1, steps=50), 4))   # ~3.0
```
*Batch = full data per step (stable, slow). SGD = one sample (noisy, fast). Mini-batch = the practical middle.*

## Momentum, RMSProp, Adam

$$
\text{Momentum: } v_t = \beta v_{t-1} + g_t,\quad \theta \leftarrow \theta - \eta v_t
$$

$$
\text{RMSProp: } s_t = \beta s_{t-1} + (1-\beta)g_t^2,\quad \theta \leftarrow \theta - \frac{\eta\, g_t}{\sqrt{s_t}+\epsilon}
$$

$$
\text{Adam: } m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t,\;\; v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2,\;\;
\hat m_t = \frac{m_t}{1-\beta_1^t},\;\; \hat v_t = \frac{v_t}{1-\beta_2^t},\;\;
\theta \leftarrow \theta - \frac{\eta\, \hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

```python
class Adam:
    """Adam = momentum (1st moment) + RMSProp (2nd moment) + bias correction."""
    def __init__(self, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m, self.v, self.t = None, None, 0

    def step(self, theta, grad):
        if self.m is None:
            self.m = np.zeros_like(theta); self.v = np.zeros_like(theta)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        m_hat = self.m / (1 - self.b1 ** self.t)     # bias correction (crucial early on)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

opt = Adam(lr=0.1); theta = np.array([0.0])
for _ in range(100):
    theta = opt.step(theta, grad=2 * (theta - 3))
print(np.round(theta, 3))   # ~[3.]
```

## Learning-Rate Schedule (linear warmup + cosine decay)

$$
\eta_t =
\begin{cases}
\eta_0 \cdot \dfrac{t}{t_{\text{warmup}}} & t < t_{\text{warmup}} \\[2ex]
\dfrac{1}{2}\eta_0\left(1 + \cos\!\Big(\pi \dfrac{t - t_{\text{warmup}}}{T - t_{\text{warmup}}}\Big)\right) & \text{otherwise}
\end{cases}
$$

```python
def lr_schedule(step, base_lr, warmup, total):
    if step < warmup:
        return base_lr * step / warmup            # warmup: ramp up to avoid early divergence
    prog = (step - warmup) / (total - warmup)
    return 0.5 * base_lr * (1 + np.cos(np.pi * prog))   # cosine: smooth decay to 0

print([round(lr_schedule(s, 0.1, 100, 1000), 4) for s in (50, 100, 1000)])  # [0.05, 0.1, 0.0]
```
*Warmup is essential for Transformers — Adam's early variance estimate is noisy; ramping LR avoids blowing up.*

---

# 2. Linear Models

## Linear Regression — Closed Form (Normal Equation)

$$
\hat\theta = (X^\top X)^{-1} X^\top y, \qquad L = \frac{1}{n}\lVert X\theta - y\rVert^2
$$

```python
def linear_regression_normal_eq(X, y):
    X = np.c_[np.ones(len(X)), X]                  # add bias column
    return np.linalg.pinv(X.T @ X) @ X.T @ y       # pinv handles singular X^T X

X = np.array([[1], [2], [3], [4]], dtype=float)
y = np.array([3, 5, 7, 9], dtype=float)            # y = 2x + 1
print(np.round(linear_regression_normal_eq(X, y), 3))   # [1. 2.]
```
*Closed form is $O(d^3)$ to invert $X^\top X$ — fine for small $d$, use GD when $d$ is large.*

## Linear Regression — Gradient Descent

$$
\nabla_\theta L = \frac{2}{n} X^\top (X\theta - y)
$$

```python
def linear_regression_gd(X, y, lr=0.01, epochs=1000):
    X = np.c_[np.ones(len(X)), X]
    theta = np.zeros(X.shape[1]); n = len(y)
    for _ in range(epochs):
        theta -= lr * (2 / n) * X.T @ (X @ theta - y)
    return theta

print(np.round(linear_regression_gd(X, y), 2))   # [0.99 2.] -> [1, 2]
```

## Logistic Regression (binary classification)

$$
p = \sigma(\theta^\top x), \qquad L = -\frac{1}{n}\sum_i \big[y_i \log p_i + (1-y_i)\log(1-p_i)\big], \qquad \nabla_\theta L = \frac{1}{n}X^\top(\hat y - y)
$$

The gradient of binary cross-entropy is the same clean form as linear regression — that's not a coincidence (both are GLMs).

```python
def logistic_regression(X, y, lr=0.1, epochs=1000):
    X = np.c_[np.ones(len(X)), X]
    theta = np.zeros(X.shape[1]); n = len(y)
    for _ in range(epochs):
        preds = sigmoid(X @ theta)
        theta -= lr * (X.T @ (preds - y) / n)
    return theta

def predict(X, theta, thresh=0.5):
    X = np.c_[np.ones(len(X)), X]
    return (sigmoid(X @ theta) >= thresh).astype(int)

X = np.array([[-2], [-1], [1], [2]], dtype=float); y = np.array([0, 0, 1, 1])
print(predict(X, logistic_regression(X, y)))   # [0 0 1 1]
```

## Regularization (Ridge L2 / Lasso L1)

$$
L_{\text{ridge}} = L + \lambda \lVert\theta\rVert_2^2 \;\Rightarrow\; +\,2\lambda\theta \text{ in grad}, \qquad
L_{\text{lasso}} = L + \lambda \lVert\theta\rVert_1 \;\Rightarrow\; +\,\lambda\,\text{sign}(\theta)
$$

*L2 → small, smooth weights (shrinkage). L1 → sparse weights, drives some to **exactly zero** (feature selection). Never regularize the bias term.*

---

# 3. Classic ML Algorithms (from scratch)

## K-Nearest Neighbors (lazy, no training)

Predict by majority vote of the $k$ closest training points under a distance metric.

```python
from collections import Counter

def knn_predict(X_train, y_train, x, k=3):
    dists = [euclidean(x, xt) for xt in X_train]
    idx = np.argsort(dists)[:k]
    return Counter(y_train[idx]).most_common(1)[0][0]

X_train = np.array([[1, 1], [1, 2], [5, 5], [6, 5]]); y_train = np.array([0, 0, 1, 1])
print(knn_predict(X_train, y_train, np.array([1.2, 1.1]), k=3))   # 0
```
*Time: $O(N d)$ per query. Gotcha: must normalize features (distance is scale-sensitive). Curse of dimensionality kills it in high $d$.*

## K-Means Clustering (Lloyd's algorithm)

$$
\min_{\{\mu_k\}} \; J = \sum_{i} \lVert x_i - \mu_{c_i}\rVert^2, \qquad c_i = \arg\min_k \lVert x_i - \mu_k\rVert^2
$$

Alternate: **assign** points to nearest centroid, then **update** centroids to the mean of their members.

```python
def kmeans(X, k, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    centroids = X[rng.choice(len(X), k, replace=False)]
    for _ in range(iters):
        dists = np.linalg.norm(X[:, None] - centroids[None], axis=2)  # assign
        labels = np.argmin(dists, axis=1)
        new = np.array([X[labels == j].mean(axis=0) if np.any(labels == j)
                        else centroids[j] for j in range(k)])         # update
        if np.allclose(new, centroids): break                          # converged
        centroids = new
    return labels, centroids

X = np.r_[np.random.default_rng(0).normal(0, 0.3, (20, 2)),
          np.random.default_rng(1).normal(5, 0.3, (20, 2))]
labels, c = kmeans(X, k=2)
print(np.round(np.sort(c.ravel())[[0, -1]], 1))   # centroids near 0 and 5
```
*Gotchas: sensitive to init (use **k-means++**), assumes spherical equal-size clusters, must pick $k$ (elbow/silhouette). It's the hard-assignment special case of GMM.*

## Naive Bayes (Gaussian)

$$
\hat y = \arg\max_c \; P(c)\prod_j P(x_j \mid c), \qquad P(x_j\mid c)=\frac{1}{\sqrt{2\pi\sigma_{c,j}^2}}\exp\!\left(-\frac{(x_j-\mu_{c,j})^2}{2\sigma_{c,j}^2}\right)
$$

Work in **log-space** (sum of logs) to avoid underflow.

```python
class GaussianNB:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.mean, self.var, self.prior = {}, {}, {}
        for c in self.classes:
            Xc = X[y == c]
            self.mean[c] = Xc.mean(axis=0)
            self.var[c] = Xc.var(axis=0) + 1e-9
            self.prior[c] = len(Xc) / len(X)
        return self

    def _log_likelihood(self, x, c):
        m, v = self.mean[c], self.var[c]
        return np.sum(-0.5 * np.log(2 * np.pi * v) - (x - m) ** 2 / (2 * v))

    def predict(self, X):
        out = []
        for x in X:
            posts = {c: np.log(self.prior[c]) + self._log_likelihood(x, c) for c in self.classes}
            out.append(max(posts, key=posts.get))
        return np.array(out)

X = np.array([[1., 1.], [1.2, 0.9], [5., 5.], [5.1, 4.8]]); y = np.array([0, 0, 1, 1])
print(GaussianNB().fit(X, y).predict(np.array([[1.1, 1.0]])))   # [0]
```
*"Naive" = assumes features conditionally independent given the class. Surprisingly strong baseline for text.*

## Decision Tree (CART)

$$
\text{Gini}(S) = 1 - \sum_k p_k^2, \qquad \text{Entropy}(S) = -\sum_k p_k \log p_k, \qquad \text{Gain} = I(S) - \sum_{c}\frac{|S_c|}{|S|}I(S_c)
$$

Greedily pick the (feature, threshold) that maximizes impurity reduction.

```python
class DecisionTree:
    def __init__(self, max_depth=3): self.max_depth = max_depth

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return 1 - np.sum(p ** 2)

    def _best_split(self, X, y):
        best = {"gain": 0}; base = self._gini(y)
        for f in range(X.shape[1]):
            for t in np.unique(X[:, f]):
                left = X[:, f] <= t
                if left.sum() == 0 or (~left).sum() == 0: continue
                child = left.mean() * self._gini(y[left]) + (~left).mean() * self._gini(y[~left])
                gain = base - child
                if gain > best["gain"]:
                    best = {"gain": gain, "f": f, "t": t, "left": left}
        return best

    def fit(self, X, y, depth=0):
        if len(np.unique(y)) == 1 or depth >= self.max_depth:
            return {"leaf": Counter(y).most_common(1)[0][0]}
        s = self._best_split(X, y)
        if s["gain"] == 0:
            return {"leaf": Counter(y).most_common(1)[0][0]}
        return {"f": s["f"], "t": s["t"],
                "L": self.fit(X[s["left"]], y[s["left"]], depth + 1),
                "R": self.fit(X[~s["left"]], y[~s["left"]], depth + 1)}

    def predict_one(self, node, x):
        if "leaf" in node: return node["leaf"]
        branch = "L" if x[node["f"]] <= node["t"] else "R"
        return self.predict_one(node[branch], x)

X = np.array([[2, 3], [1, 1], [8, 9], [9, 8]]); y = np.array([0, 0, 1, 1])
tree = DecisionTree(2); root = tree.fit(X, y)
print(tree.predict_one(root, np.array([8.5, 8.5])))   # 1
```

## Random Forest (bagging + feature randomness)

Train many trees on **bootstrap** samples (sample $n$ rows with replacement) and average/vote. Decorrelating trees (also random feature subsets per split) reduces variance.

```python
def random_forest(X, y, base_tree, n_trees=10, seed=0):
    rng = np.random.default_rng(seed); trees = []
    for _ in range(n_trees):
        idx = rng.integers(0, len(X), len(X))        # bootstrap sample
        t = base_tree(); trees.append(t.fit(X[idx], y[idx]))
    return trees

def rf_predict(trees, X):
    preds = np.array([[t.predict_one(t.root, x) for x in X] for t in trees])  # (trees, N)
    return np.array([Counter(preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])])
```
*Bagging reduces **variance** (averaging i.i.d.-ish estimators); each tree is high-variance but the ensemble is stable. Out-of-bag samples give a free validation estimate.*

## AdaBoost (adaptive boosting)

$$
\alpha_m = \frac{1}{2}\ln\frac{1-\epsilon_m}{\epsilon_m}, \qquad w_i \leftarrow w_i \, e^{-\alpha_m y_i h_m(x_i)}, \qquad H(x)=\text{sign}\!\Big(\sum_m \alpha_m h_m(x)\Big)
$$

Re-weight misclassified points up each round; weak learners (stumps) vote weighted by accuracy.

```python
def adaboost(X, y, base_stump, n=10):
    y = np.where(y <= 0, -1, 1)                       # labels in {-1, +1}
    w = np.ones(len(X)) / len(X); models = []
    for _ in range(n):
        s = base_stump().fit(X, y, sample_weight=w)
        pred = s.predict(X)
        err = np.sum(w * (pred != y)) / np.sum(w)
        err = min(max(err, 1e-10), 1 - 1e-10)
        alpha = 0.5 * np.log((1 - err) / err)
        w *= np.exp(-alpha * y * pred); w /= w.sum()  # boost hard examples
        models.append((alpha, s))
    return models
```
*Boosting reduces **bias** (sequentially fixing errors); the opposite emphasis to bagging. Sensitive to label noise/outliers.*

## Gradient Boosting (GBDT — the Kaggle/Google workhorse)

$$
F_m(x) = F_{m-1}(x) + \nu\, h_m(x), \qquad h_m \approx -\frac{\partial L}{\partial F_{m-1}} \;(\text{= residual } y - F_{m-1} \text{ for MSE})
$$

Each new tree fits the **negative gradient** (pseudo-residuals) of the loss — gradient descent in function space.

```python
def grad_boost(X, y, base_reg_tree, n=20, lr=0.1):
    pred = np.full(len(y), y.mean()); f0 = y.mean(); trees = []
    for _ in range(n):
        residual = y - pred                          # neg. gradient of 1/2 (y-F)^2
        t = base_reg_tree().fit(X, residual)
        pred += lr * t.predict(X); trees.append(t)   # shrinkage via lr
    return f0, trees

def gb_predict(f0, trees, X, lr=0.1):
    p = np.full(len(X), f0)
    for t in trees: p += lr * t.predict(X)
    return p
```
*XGBoost/LightGBM add 2nd-order (Hessian) info, regularization, and clever histogram splits. Know: shrinkage (`lr`) + many shallow trees > few deep trees.*

## PCA (eigendecomposition of the covariance)

$$
\Sigma = \frac{1}{n}X_c^\top X_c, \qquad \Sigma v_k = \lambda_k v_k, \qquad \text{components} = \text{top-}m \text{ eigenvectors by } \lambda
$$

Project onto directions of **maximum variance** (= top eigenvectors of the covariance).

```python
def pca(X, n_components):
    X = X - X.mean(axis=0)                   # 1. center (PCA needs zero mean!)
    cov = np.cov(X, rowvar=False)            # 2. covariance (d, d)
    vals, vecs = np.linalg.eigh(cov)         # 3. eigen-decompose (symmetric)
    idx = np.argsort(vals)[::-1]             # 4. sort by variance desc
    components = vecs[:, idx[:n_components]]
    return X @ components, components

X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=float)
proj, comps = pca(X, 1)
print(np.round(proj.ravel(), 2))   # 1D coords along the diagonal
```
*SVD is the numerically preferred route: `U, S, Vt = np.linalg.svd(X_centered)`; components are rows of `Vt`. Explained variance ratio $= \lambda_k / \sum \lambda$.*

## SVM (linear, hinge loss + sub-gradient descent)

$$
\min_{w,b} \; \frac{1}{2}\lVert w\rVert^2 + C\sum_i \max\big(0,\, 1 - y_i(w^\top x_i + b)\big)
$$

Maximize the margin $\frac{2}{\lVert w\rVert}$; the hinge term penalizes points inside the margin or misclassified.

```python
def linear_svm(X, y, lr=0.01, lam=0.01, epochs=1000):
    y = np.where(y <= 0, -1, 1)
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(epochs):
        for i in range(len(X)):
            if y[i] * (X[i] @ w + b) >= 1:        # correct & outside margin
                w -= lr * (2 * lam * w)
            else:                                  # inside margin / wrong
                w -= lr * (2 * lam * w - y[i] * X[i])
                b -= lr * (-y[i])
    return w, b

X = np.array([[2, 2], [3, 3], [-2, -2], [-3, -3]], dtype=float); y = np.array([1, 1, 0, 0])
w, b = linear_svm(X, y)
print(np.sign(X @ w + b))   # [1. 1. -1. -1.]
```
*Kernel trick (RBF $k(x,x')=e^{-\gamma\lVert x-x'\rVert^2}$) learns non-linear boundaries without explicit feature maps. Only support vectors (margin points) define the boundary.*

## EM for Gaussian Mixture Models (soft K-Means)

$$
\textbf{E: } r_{ik} = \frac{\pi_k \mathcal{N}(x_i\mid\mu_k,\sigma_k^2)}{\sum_j \pi_j \mathcal{N}(x_i\mid\mu_j,\sigma_j^2)} \qquad
\textbf{M: } \mu_k = \frac{\sum_i r_{ik}x_i}{\sum_i r_{ik}},\;\; \sigma_k^2 = \frac{\sum_i r_{ik}(x_i-\mu_k)^2}{\sum_i r_{ik}},\;\; \pi_k=\frac{\sum_i r_{ik}}{n}
$$

```python
def gmm_em_1d(X, k=2, iters=50, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.choice(X, k); var = np.full(k, X.var()); pi = np.full(k, 1/k)
    for _ in range(iters):
        pdf = (1/np.sqrt(2*np.pi*var)) * np.exp(-(X[:,None]-mu)**2 / (2*var))
        r = pi * pdf; r /= r.sum(axis=1, keepdims=True)        # E-step
        Nk = r.sum(axis=0)
        mu  = (r * X[:,None]).sum(axis=0) / Nk                 # M-step
        var = (r * (X[:,None]-mu)**2).sum(axis=0) / Nk + 1e-9
        pi  = Nk / len(X)
    return mu, var, pi

X = np.r_[np.random.default_rng(0).normal(0, 1, 100),
          np.random.default_rng(1).normal(5, 1, 100)]
mu, var, pi = gmm_em_1d(X, k=2)
print(np.round(np.sort(mu), 1))   # ~[0. 5.]
```
*EM never decreases the log-likelihood (monotone). E-step = soft assignment; M-step = weighted MLE.*

## DBSCAN (density-based clustering)

Grows clusters from **core points** (≥ `min_pts` neighbors within `eps`). Finds arbitrary shapes and labels outliers as noise — no need to pick $k$.

```python
def dbscan(X, eps, min_pts):
    n = len(X); labels = np.full(n, -1); visited = np.zeros(n, bool); cid = 0
    def neighbors(i): return [j for j in range(n) if np.linalg.norm(X[i]-X[j]) <= eps]
    for i in range(n):
        if visited[i]: continue
        visited[i] = True; N = neighbors(i)
        if len(N) < min_pts: continue                 # not a core point -> leave as noise
        labels[i] = cid; seeds = list(N); k = 0
        while k < len(seeds):                          # expand cluster
            j = seeds[k]; k += 1
            if not visited[j]:
                visited[j] = True; Nj = neighbors(j)
                if len(Nj) >= min_pts: seeds += Nj
            if labels[j] == -1: labels[j] = cid
        cid += 1
    return labels   # -1 = noise

X = np.r_[np.random.default_rng(0).normal(0, 0.2, (10, 2)),
          np.random.default_rng(1).normal(5, 0.2, (10, 2))]
print(len(set(dbscan(X, eps=0.7, min_pts=3)) - {-1}))   # 2 clusters
```
*Strength: non-spherical clusters + outlier detection. Weakness: one global `eps` struggles with varying density.*

---

# 4. Neural Networks from Scratch

## 2-Layer MLP with manual backprop (the canonical whiteboard question)

Forward: $z_1 = XW_1+b_1,\; a_1=\text{ReLU}(z_1),\; z_2=a_1W_2+b_2,\; p=\text{softmax}(z_2)$.
The one gradient to memorize:

$$
\frac{\partial L}{\partial z_2} = p - \text{onehot}(y), \qquad \frac{\partial L}{\partial z_1} = \big(\tfrac{\partial L}{\partial z_2}W_2^\top\big)\odot \mathbb{1}[z_1>0]
$$

```python
class TwoLayerNet:
    """Input -> Linear -> ReLU -> Linear -> Softmax -> Cross-Entropy."""
    def __init__(self, d_in, d_hidden, d_out, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2/d_in), (d_in, d_hidden)); self.b1 = np.zeros(d_hidden)
        self.W2 = rng.normal(0, np.sqrt(2/d_hidden), (d_hidden, d_out)); self.b2 = np.zeros(d_out)

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1; self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2; self.probs = softmax(self.z2)
        return self.probs

    def backward(self, y, lr=0.1):
        N = len(y)
        dz2 = self.probs.copy(); dz2[np.arange(N), y] -= 1; dz2 /= N   # softmax+CE gradient
        dW2 = self.a1.T @ dz2; db2 = dz2.sum(axis=0)
        dz1 = (dz2 @ self.W2.T) * relu_grad(self.z1)                   # chain rule
        dW1 = self.X.T @ dz1; db1 = dz1.sum(axis=0)
        self.W2 -= lr*dW2; self.b2 -= lr*db2; self.W1 -= lr*dW1; self.b1 -= lr*db1

X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float); y = np.array([0, 1, 1, 0])  # XOR
net = TwoLayerNet(2, 8, 2)
for _ in range(2000): net.forward(X); net.backward(y, lr=0.5)
print(np.argmax(net.forward(X), axis=1))   # [0 1 1 0] — solved the non-linear XOR
```

## Autograd Micro-Engine (reverse-mode, scalar — "build backprop")

Each `Value` records its parents and a local `_backward`. We topologically sort the graph and apply the chain rule in reverse. This is the heart of PyTorch in ~30 lines.

```python
import math

class Value:
    def __init__(self, data, _children=()):
        self.data = data; self.grad = 0.0
        self._backward = lambda: None; self._prev = set(_children)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        def _b(): self.grad += out.grad; other.grad += out.grad
        out._backward = _b; return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        def _b():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _b; return out

    def tanh(self):
        t = math.tanh(self.data); out = Value(t, (self,))
        def _b(): self.grad += (1 - t*t) * out.grad
        out._backward = _b; return out

    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._prev: build(c)
                topo.append(v)
        build(self); self.grad = 1.0
        for v in reversed(topo): v._backward()   # reverse-mode chain rule

a = Value(2.0); b = Value(-3.0); c = Value(10.0)
e = (a*b + c).tanh()
e.backward()
print(round(a.grad, 5), round(b.grad, 5))   # gradients of e w.r.t. a, b
```

## Batch Norm & Layer Norm

$$
\hat x = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad y = \gamma\hat x + \beta
$$

BatchNorm normalizes each feature over the **batch**; LayerNorm normalizes each example over its **features** (used in Transformers because it's batch-size independent).

```python
def batch_norm(X, gamma, beta, eps=1e-5):
    mu = X.mean(axis=0); var = X.var(axis=0)            # stats over batch dim
    return gamma * (X - mu) / np.sqrt(var + eps) + beta

def layer_norm(X, gamma, beta, eps=1e-5):
    mu = X.mean(axis=-1, keepdims=True); var = X.var(axis=-1, keepdims=True)  # over features
    return gamma * (X - mu) / np.sqrt(var + eps) + beta

print(np.round(layer_norm(np.array([[1., 2., 3.], [4., 5., 6.]]), 1.0, 0.0), 2))
```
*Gotcha: BatchNorm behaves differently in train vs eval (uses running stats at inference); LayerNorm doesn't — hence LLMs use LayerNorm/RMSNorm.*

## Dropout (inverted)

$$
\tilde x = x \cdot \frac{\text{mask}}{1-p}, \quad \text{mask}_i \sim \text{Bernoulli}(1-p)
$$

Scale by $1/(1-p)$ at **train** time so $\mathbb{E}[\tilde x]=x$ and inference needs no change.

```python
def dropout(X, p=0.5, training=True, seed=None):
    if not training or p == 0: return X
    rng = np.random.default_rng(seed)
    mask = (rng.random(X.shape) > p) / (1 - p)
    return X * mask

print(dropout(np.ones((1, 6)), p=0.5, seed=0))   # ~half zeroed, rest scaled to 2.0
```

## Convolution 2D (single channel)

$$
\text{out}[i,j] = \sum_{u,v} \text{img}[i\!\cdot\!s+u,\, j\!\cdot\!s+v]\cdot K[u,v], \qquad
\text{out size} = \left\lfloor\frac{H - K + 2P}{S}\right\rfloor + 1
$$

```python
def conv2d(image, kernel, stride=1, padding=0):
    if padding: image = np.pad(image, padding)
    kh, kw = kernel.shape; H, W = image.shape
    out_h = (H - kh) // stride + 1; out_w = (W - kw) // stride + 1
    out = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            region = image[i*stride:i*stride+kh, j*stride:j*stride+kw]
            out[i, j] = np.sum(region * kernel)
    return out

img = np.array([[0,0,1,1]]*4, dtype=float)
sobel = np.array([[-1,1],[-1,1]], dtype=float)
print(conv2d(img, sobel))   # high response at the 0->1 edge
```
*Know the output-size formula cold — it's a frequent quick-fire question. Params per conv layer: $(K\!\cdot\!K\!\cdot\!C_{in}+1)\cdot C_{out}$.*

## RNN cell & LSTM cell

$$
\text{RNN: } h_t = \tanh(W_x x_t + W_h h_{t-1} + b)
$$

$$
\text{LSTM: } f_t,i_t,o_t = \sigma(\cdot),\;\; \tilde c_t = \tanh(\cdot),\;\; c_t = f_t\odot c_{t-1} + i_t\odot \tilde c_t,\;\; h_t = o_t\odot\tanh(c_t)
$$

```python
def rnn_cell(x_t, h_prev, Wx, Wh, b):
    return np.tanh(x_t @ Wx + h_prev @ Wh + b)

def lstm_cell(x_t, h_prev, c_prev, p):
    z = np.concatenate([h_prev, x_t])
    f = sigmoid(p['Wf'] @ z + p['bf'])    # forget gate
    i = sigmoid(p['Wi'] @ z + p['bi'])    # input gate
    g = np.tanh(p['Wg'] @ z + p['bg'])    # candidate
    o = sigmoid(p['Wo'] @ z + p['bo'])    # output gate
    c = f * c_prev + i * g
    h = o * np.tanh(c)
    return h, c
```
*Why LSTM beats vanilla RNN: the **additive** cell-state path lets gradients flow without vanishing — the answer to "why do RNNs struggle with long sequences?".*

---

# 5. Transformers & Attention

*The* deep-learning interview topic. Be able to write scaled dot-product attention from memory.

## Scaled Dot-Product Attention

$$
\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

```python
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(0, 2, 1) if Q.ndim == 3 else Q @ K.T) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)   # causal / padding mask
    weights = softmax(scores, axis=-1)
    return weights @ V, weights

Q = np.array([[1., 0.]]); K = np.array([[1., 0.], [0., 1.]]); V = np.array([[10., 0.], [0., 10.]])
out, w = scaled_dot_product_attention(Q, K, V)
print(np.round(w, 2))   # [[0.67 0.33]] — attends more to the matching key
```
> **Why divide by $\sqrt{d_k}$?** Dot products have variance $\propto d_k$; without scaling, softmax saturates into a near one-hot and gradients vanish. Most-asked attention follow-up.

## Multi-Head Attention

$$
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V), \qquad \text{MHA} = \text{Concat}(\text{head}_1,\dots,\text{head}_h)\,W^O
$$

```python
def multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads):
    L, d_model = X.shape; d_k = d_model // num_heads
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    def split(M): return M.reshape(L, num_heads, d_k).transpose(1, 0, 2)
    Qh, Kh, Vh = split(Q), split(K), split(V)
    heads = [scaled_dot_product_attention(Qh[h], Kh[h], Vh[h])[0] for h in range(num_heads)]
    return np.concatenate(heads, axis=-1) @ Wo

X = np.random.default_rng(0).normal(size=(4, 8))
W = lambda: np.random.default_rng(1).normal(size=(8, 8))
print(multi_head_attention(X, W(), W(), W(), W(), num_heads=2).shape)   # (4, 8)
```
*Heads attend to different relationship types (syntax, coreference, position) in parallel subspaces.*

## Sinusoidal Positional Encoding

$$
PE_{(pos, 2i)} = \sin\!\Big(\frac{pos}{10000^{2i/d}}\Big), \qquad PE_{(pos, 2i+1)} = \cos\!\Big(\frac{pos}{10000^{2i/d}}\Big)
$$

```python
def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]; i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d_model)
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2]); pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe

print(positional_encoding(4, 6).shape)   # (4, 6)
```
*Attention is permutation-invariant — without positions it can't tell word order. Modern LLMs use **RoPE** (rotary) which encodes relative position via rotation.*

## Transformer Encoder Block

```python
def transformer_block(X, mha_weights, ff_w1, ff_w2, ln):
    attn = multi_head_attention(X, *mha_weights, num_heads=2)
    X = layer_norm(X + attn, ln['g1'], ln['b1'])          # residual + LayerNorm
    ff = gelu(X @ ff_w1) @ ff_w2                           # position-wise FFN
    X = layer_norm(X + ff, ln['g2'], ln['b2'])
    return X
```
*Two universal ingredients: **residual connections** (gradient highway) and **LayerNorm** (stable activations). Pre-LN (norm inside the residual) trains more stably than the original Post-LN.*

## KV Cache (the inference-efficiency question)

At generation step $t$, the new token must attend to all prior tokens. Recomputing every K/V is $O(N^2)$; instead cache each token's K and V once. Cache size in bytes:

$$
\text{KV bytes} = 2 \cdot L \cdot H_{kv} \cdot d_{head} \cdot N_{seq} \cdot N_{batch} \cdot \text{bytes/elem}
$$

(leading 2 = K and V; $L$ layers, $H_{kv}$ KV-heads). The KV cache — not the weights — usually caps how many concurrent users you can batch. **Multi-Query / Grouped-Query Attention** shrinks $H_{kv}$ to cut it.

---

# 6. LLM Decoding & Sampling

Given logits over the vocabulary, *how you pick the next token* is its own family of algorithms — heavily asked for LLM roles.

## Greedy & Temperature Sampling

$$
p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}} \quad (T<1 \text{ sharpens},\; T>1 \text{ flattens},\; T\!\to\!0 \text{ = greedy})
$$

```python
def greedy_decode(logits):
    return int(np.argmax(logits))            # deterministic, can loop/repeat

def temperature_sample(logits, temperature=1.0, seed=None):
    rng = np.random.default_rng(seed)
    probs = softmax(logits / temperature)
    return int(rng.choice(len(probs), p=probs))

logits = np.array([2.0, 1.0, 0.1, 0.05])
print(greedy_decode(logits), temperature_sample(logits, 0.7, seed=0))
```

## Top-k Sampling

Keep only the $k$ highest-logit tokens, renormalize, sample.

```python
def top_k_sample(logits, k=2, seed=None):
    rng = np.random.default_rng(seed)
    idx = np.argpartition(logits, -k)[-k:]
    probs = softmax(logits[idx])
    return int(rng.choice(idx, p=probs))

print(top_k_sample(np.array([2.0, 1.0, 0.1, 0.05]), k=2, seed=0))   # 0 or 1
```

## Top-p (Nucleus) Sampling — the modern default

Keep the **smallest** set of tokens whose cumulative probability $\ge p$, renormalize, sample. Adapts the candidate set to the distribution's shape.

```python
def top_p_sample(logits, p=0.9, seed=None):
    rng = np.random.default_rng(seed)
    probs = softmax(logits)
    order = np.argsort(probs)[::-1]
    cumulative = np.cumsum(probs[order])
    cutoff = np.searchsorted(cumulative, p) + 1
    keep = order[:cutoff]
    return int(rng.choice(keep, p=probs[keep] / probs[keep].sum()))

print(top_p_sample(np.array([2.0, 1.0, 0.1, 0.05]), p=0.9, seed=0))
```

## Beam Search (keep top-B sequences)

```python
def beam_search(step_logits_fn, start, beam_width=2, max_len=3):
    beams = [(start, 0.0)]                                  # (sequence, cumulative log-prob)
    for _ in range(max_len):
        candidates = []
        for seq, score in beams:
            logp = np.log(softmax(step_logits_fn(seq)))
            for tok in range(len(logp)):
                candidates.append((seq + [tok], score + logp[tok]))
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]
    return beams[0][0]

print(beam_search(lambda seq: np.array([0.1, 2.0, 0.1]), start=[0]))   # [0, 1, 1, 1]
```
*Higher-likelihood sequences (good for translation), but bland for open-ended chat — hence sampling. Length-normalize scores to avoid favoring short sequences.*

## Byte-Pair Encoding (the GPT tokenizer)

Start from characters; greedily merge the most frequent adjacent pair, repeat. Frequent words → one token, rare words → subwords (no OOV).

```python
from collections import Counter

def bpe_train(corpus, num_merges):
    vocab = [list(word) + ['</w>'] for word in corpus]
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for word in vocab:
            for a, b in zip(word, word[1:]): pairs[(a, b)] += 1
        if not pairs: break
        best = max(pairs, key=pairs.get); merges.append(best)
        vocab = [_merge_pair(w, best) for w in vocab]
    return merges

def _merge_pair(word, pair):
    out, i = [], 0
    while i < len(word):
        if i < len(word)-1 and (word[i], word[i+1]) == pair:
            out.append(word[i] + word[i+1]); i += 2
        else:
            out.append(word[i]); i += 1
    return out

print(bpe_train(["low", "lower", "lowest"], num_merges=3))   # learns ('l','o'), ('lo','w'), ...
```

---

> **Deeper dive:** for the full math and algorithm treatment of quantization — GPTQ, AWQ, SmoothQuant, GGUF k-quants, QAT/STE, NF4, BitNet — see the standalone [Quantization primer](/notes/Primers/Quantization).

---

# 7. Representation Learning & Embeddings

Google leans heavily on retrieval, recommendations and embeddings — these come up constantly.

## TF-IDF

$$
\text{tfidf}(t,d) = \text{tf}(t,d)\cdot \log\frac{N}{\text{df}(t)} \quad(\text{smoothed: } \log\tfrac{1+N}{1+\text{df}(t)} + 1)
$$

```python
def tf_idf(docs):
    vocab = sorted(set(w for d in docs for w in d)); N = len(docs)
    tf = np.zeros((N, len(vocab)))
    for i, d in enumerate(docs):
        for w in d: tf[i, vocab.index(w)] += 1
        tf[i] /= len(d)
    df = np.array([sum(w in d for d in docs) for w in vocab])
    idf = np.log((1 + N) / (1 + df)) + 1
    return tf * idf, vocab

mat, vocab = tf_idf([["cat", "sat"], ["dog", "sat"], ["cat", "dog", "cat"]])
print(mat.shape, vocab)
```
*Down-weights words common across all docs ("the", "sat") and up-weights distinctive ones. Still a strong sparse-retrieval baseline (BM25 is the refined version).*

## Word2Vec — Skip-Gram with Negative Sampling

Predict context words from a center word. Negative sampling turns the expensive softmax into binary logistic regressions against $k$ random "negative" words:

$$
L = -\log\sigma(v_c^\top v_o) - \sum_{k} \log\sigma(-v_c^\top v_{n_k})
$$

```python
def skipgram_neg_step(center_v, context_v, neg_vs, lr=0.05):
    # gradient step for one (center, +context, k negatives) tuple
    grad_center = (sigmoid(context_v @ center_v) - 1) * context_v
    new_context = context_v - lr * (sigmoid(context_v @ center_v) - 1) * center_v
    for nv in neg_vs:
        grad_center += sigmoid(nv @ center_v) * nv          # push away negatives
    return center_v - lr * grad_center, new_context

c = np.random.default_rng(0).normal(size=5)
ctx = np.random.default_rng(1).normal(size=5)
negs = np.random.default_rng(2).normal(size=(3, 5))
nc, ncx = skipgram_neg_step(c, ctx, negs)
print(nc.shape)   # (5,) updated center embedding
```
*Key idea: "you shall know a word by the company it keeps." Negative sampling makes training $O(k)$ instead of $O(V)$ per step.*

## Contrastive & Triplet Loss (metric learning)

$$
\text{Triplet: } L = \max\!\big(0,\; \lVert a-p\rVert^2 - \lVert a-n\rVert^2 + \alpha\big)
$$

$$
\text{InfoNCE: } L = -\log\frac{e^{q\cdot k^+/\tau}}{\sum_j e^{q\cdot k_j/\tau}}
$$

```python
def triplet_loss(anchor, positive, negative, margin=1.0):
    # pull anchor toward positive, push away from negative by at least `margin`
    return max(0, np.sum((anchor-positive)**2) - np.sum((anchor-negative)**2) + margin)

def info_nce(q, keys, temp=0.07):
    # keys[0] is the positive; the rest are negatives (SimCLR / CLIP style)
    logits = keys @ q / temp
    return -np.log(softmax(logits)[0])

print(triplet_loss(np.array([0.,0]), np.array([2.,0]), np.array([0.5,0])))   # 4.75 (violated)
print(round(float(info_nce(np.array([1.,0]), np.array([[1.,0],[0,1.],[-1,0.]]))), 4))
```
*Foundation of face recognition (FaceNet), CLIP, and modern self-supervised pretraining. Hard-negative mining is what makes triplet loss actually work.*

## Matrix Factorization for Recommendations

$$
\hat r_{ij} = p_i^\top q_j, \qquad \min_{P,Q}\sum_{(i,j)\in\text{obs}}(r_{ij}-p_i^\top q_j)^2 + \lambda(\lVert p_i\rVert^2 + \lVert q_j\rVert^2)
$$

```python
def matrix_factorization(R, k=2, lr=0.01, reg=0.1, steps=2000, seed=0):
    rng = np.random.default_rng(seed); m, n = R.shape
    P = rng.normal(0, 0.1, (m, k)); Q = rng.normal(0, 0.1, (n, k))
    for _ in range(steps):
        for i in range(m):
            for j in range(n):
                if R[i, j] > 0:                       # only observed ratings
                    e = R[i, j] - P[i] @ Q[j]
                    P[i] += lr * (e * Q[j] - reg * P[i])
                    Q[j] += lr * (e * P[i] - reg * Q[j])
    return P @ Q.T

R = np.array([[5,3,0,1],[4,0,0,1],[1,1,0,5],[0,0,5,4]], dtype=float)  # 0 = unobserved
print(np.round(matrix_factorization(R), 1))   # fills in the missing entries
```
*The classic collaborative-filtering model (Netflix Prize). Two-tower neural retrieval is the deep-learning descendant.*

---

# 8. Generative Models

## Variational Autoencoder (VAE)

Maximize the ELBO; the **reparameterization trick** ($z = \mu + \sigma\odot\epsilon$) makes sampling differentiable:

$$
\mathcal{L} = \underbrace{\mathbb{E}_{q(z|x)}[\log p(x|z)]}_{\text{reconstruction}} - \underbrace{D_{KL}\big(q(z|x)\,\|\,p(z)\big)}_{\text{regularizer}}, \qquad
D_{KL} = -\tfrac{1}{2}\sum_j\big(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\big)
$$

```python
def reparameterize(mu, logvar, seed=None):
    rng = np.random.default_rng(seed)
    std = np.exp(0.5 * logvar)
    return mu + std * rng.normal(size=mu.shape)        # differentiable sampling

def vae_loss(x, x_recon, mu, logvar):
    recon = np.sum((x - x_recon) ** 2)                 # or BCE for binary data
    kl = -0.5 * np.sum(1 + logvar - mu**2 - np.exp(logvar))
    return recon + kl

print(round(vae_loss(np.array([1.,0]), np.array([0.9,0.1]),
                     np.array([0.,0]), np.array([0.,0])), 3))   # 0.02
```
*Without the reparameterization trick you can't backprop through a random sample. The KL term keeps the latent space close to $\mathcal{N}(0,I)$ so you can sample new data.*

## Generative Adversarial Network (GAN)

$$
\min_G\max_D \; \mathbb{E}_{x}[\log D(x)] + \mathbb{E}_{z}[\log(1 - D(G(z)))]
$$

```python
def gan_d_loss(d_real, d_fake):   # discriminator: real->1, fake->0
    return -(np.log(d_real + 1e-9) + np.log(1 - d_fake + 1e-9))

def gan_g_loss(d_fake):           # generator: fool D (non-saturating form)
    return -np.log(d_fake + 1e-9)

print(round(gan_d_loss(0.9, 0.1), 3), round(gan_g_loss(0.1), 3))
```
*Generator and discriminator play a minimax game. Use the **non-saturating** generator loss $-\log D(G(z))$ (not $\log(1-D)$) for stronger early gradients. Mode collapse is the classic failure.*

---

# 9. Evaluation Metrics (from scratch)

Knowing *which* metric — and computing it without `sklearn` — is a frequent screen.

## Confusion Matrix → Precision / Recall / F1

$$
P = \frac{TP}{TP+FP}, \quad R = \frac{TP}{TP+FN}, \quad F_1 = \frac{2PR}{P+R}, \quad \text{Acc} = \frac{TP+TN}{N}
$$

```python
def classification_metrics(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1)); fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1)); tn = np.sum((y_pred == 0) & (y_true == 0))
    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    return dict(accuracy=(tp + tn) / len(y_true), precision=precision, recall=recall, f1=f1)

y_true = np.array([1, 1, 0, 0, 1]); y_pred = np.array([1, 0, 0, 1, 1])
print({k: round(v, 2) for k, v in classification_metrics(y_true, y_pred).items()})
```
*Precision vs recall: spam filter wants precision (don't kill real mail); cancer screen wants recall (don't miss a case). Use F1/AUC on imbalanced data, not accuracy.*

## ROC-AUC (threshold-free, rank-based)

$$
\text{AUC} = P\big(\text{score}(x^+) > \text{score}(x^-)\big) = \frac{1}{|P||N|}\sum_{i\in P}\sum_{j\in N}\mathbb{1}[s_i > s_j]
$$

```python
def roc_auc(y_true, scores):
    pos = scores[y_true == 1]; neg = scores[y_true == 0]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)   # 0.5 credit for ties
    return wins / (len(pos) * len(neg))

print(round(roc_auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])), 3))   # 0.75
```
*AUC = probability a random positive ranks above a random negative. Insensitive to threshold and class balance.*

## Regression: MSE / RMSE / MAE / R²

$$
\text{MSE}=\frac{1}{n}\sum(\hat y-y)^2,\quad \text{MAE}=\frac{1}{n}\sum|\hat y-y|,\quad R^2 = 1 - \frac{\sum(\hat y-y)^2}{\sum(y-\bar y)^2}
$$

```python
def regression_metrics(y_true, y_pred):
    err = y_true - y_pred
    mse = np.mean(err ** 2)
    ss_res = np.sum(err ** 2); ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return dict(mse=mse, rmse=np.sqrt(mse), mae=np.mean(np.abs(err)), r2=1 - ss_res / (ss_tot + 1e-12))

print({k: round(v, 3) for k, v in
       regression_metrics(np.array([3., 5., 7.]), np.array([2.8, 5.2, 6.9])).items()})
```
*MAE is robust to outliers; MSE punishes large errors harder. $R^2$ = fraction of variance explained (1 = perfect, 0 = predicting the mean).*

## NDCG (ranking quality — recsys/search)

$$
\text{DCG@k} = \sum_{i=1}^{k}\frac{2^{rel_i}-1}{\log_2(i+1)}, \qquad \text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}
$$

```python
def ndcg(relevances, k=None):
    relevances = np.array(relevances); k = k or len(relevances)
    def dcg(r): return np.sum((2**r - 1) / np.log2(np.arange(2, len(r) + 2)))
    ideal = np.sort(relevances)[::-1]
    return dcg(relevances[:k]) / (dcg(ideal[:k]) + 1e-12)

print(round(ndcg([3, 2, 3, 0, 1, 2]), 3))   # 0.949 — rewards relevant items ranked high
```
*The discount ($1/\log_2$) means relevance at rank 1 matters far more than at rank 10. Normalized so 1.0 = perfect ranking.*

---

# 10. Probability, Sampling & Bandits

## Sampling from a Categorical (inverse-CDF & Gumbel-max)

$$
\text{Gumbel-max: } \arg\max_i\big(z_i + g_i\big),\;\; g_i = -\log(-\log u_i),\; u_i\sim U(0,1) \;\sim\; \text{softmax}(z)
$$

```python
def sample_categorical(probs, seed=None):
    rng = np.random.default_rng(seed)
    return int(np.searchsorted(np.cumsum(probs), rng.random()))   # inverse CDF

def gumbel_max_sample(logits, seed=None):
    rng = np.random.default_rng(seed)
    g = -np.log(-np.log(rng.random(len(logits))))
    return int(np.argmax(logits + g))

print(sample_categorical(np.array([0.1, 0.3, 0.6]), seed=0))
```
*Gumbel-softmax (a soft, differentiable version) lets you backprop through discrete sampling.*

## Reservoir Sampling (uniform sample from an unbounded stream)

Each of $n$ items ends up retained with probability $k/n$ in a single pass, $O(k)$ memory.

```python
def reservoir_sample(stream, k, seed=0):
    rng = np.random.default_rng(seed); reservoir = []
    for i, item in enumerate(stream):
        if i < k: reservoir.append(item)
        else:
            j = rng.integers(0, i + 1)        # replace with prob k/i
            if j < k: reservoir[j] = item
    return reservoir

print(reservoir_sample(range(1000), k=3))
```
*Classic "sample from data you can't fit in memory." Provable by induction that every element has prob $k/n$.*

## Multi-Armed Bandit — ε-greedy & UCB

$$
\varepsilon\text{-greedy: explore w.p. } \varepsilon, \text{ else } \arg\max_a Q_a \qquad
\text{UCB: } a_t = \arg\max_a\Big(Q_a + c\sqrt{\tfrac{\ln t}{N_a}}\Big)
$$

UCB adds an "optimism" bonus that shrinks as an arm is pulled more — principled exploration.

```python
def epsilon_greedy_bandit(true_means, steps=1000, eps=0.1, seed=0):
    rng = np.random.default_rng(seed); k = len(true_means)
    Q = np.zeros(k); N = np.zeros(k)
    for _ in range(steps):
        a = rng.integers(k) if rng.random() < eps else int(np.argmax(Q))   # explore/exploit
        r = rng.normal(true_means[a], 1.0)
        N[a] += 1; Q[a] += (r - Q[a]) / N[a]        # incremental sample mean
    return Q

def ucb_bandit(true_means, steps=2000, c=2.0, seed=0):
    rng = np.random.default_rng(seed); K = len(true_means); Q = np.zeros(K); N = np.zeros(K)
    for t in range(1, steps + 1):
        a = int(np.argmin(N)) if 0 in N else int(np.argmax(Q + c * np.sqrt(np.log(t) / N)))
        r = rng.normal(true_means[a], 1.0); N[a] += 1; Q[a] += (r - Q[a]) / N[a]
    return int(np.argmax(Q))

print(np.round(epsilon_greedy_bandit([1.0, 1.5, 2.0]), 2))   # best arm ~2.0
print("UCB best arm:", ucb_bandit([1.0, 1.5, 2.0]))          # 2
```
*Exploration/exploitation in miniature. Thompson Sampling (Bayesian posterior sampling) is the third must-know.*

---

# 11. Sequence Models & Dynamic Programming in ML

## Viterbi Algorithm (most-likely hidden state sequence — HMM decoding)

$$
\delta_t(s) = \max_{s'}\big[\delta_{t-1}(s')\cdot a_{s's}\big]\cdot b_s(o_t) \quad(\text{work in log-space to avoid underflow})
$$

DP over (time × states): keep the best score reaching each state, plus backpointers to reconstruct the path.

```python
def viterbi(obs, start_p, trans_p, emit_p):
    T = len(obs); N = len(start_p)
    dp = np.zeros((T, N)); back = np.zeros((T, N), int)
    dp[0] = np.log(start_p) + np.log(emit_p[:, obs[0]])
    for t in range(1, T):
        for s in range(N):
            scores = dp[t-1] + np.log(trans_p[:, s])       # best predecessor
            back[t, s] = int(np.argmax(scores))
            dp[t, s] = np.max(scores) + np.log(emit_p[s, obs[t]])
    path = [int(np.argmax(dp[-1]))]
    for t in range(T-1, 0, -1):
        path.insert(0, int(back[t, path[0]]))              # backtrack
    return path

start = np.array([0.6, 0.4])
trans = np.array([[0.7, 0.3], [0.4, 0.6]])
emit  = np.array([[0.5, 0.4, 0.1], [0.1, 0.3, 0.6]])
print(viterbi([0, 1, 2], start, trans, emit))   # [0, 0, 1] most likely state path
```
*Same DP skeleton as edit distance / sequence alignment. Forward-backward (sum instead of max) gives marginal probabilities; CTC and beam-search decoding are cousins.*

---

# 12. Reinforcement Learning

## Tabular Q-Learning (off-policy TD control)

$$
Q(s,a) \leftarrow Q(s,a) + \alpha\Big[r + \gamma\max_{a'}Q(s',a') - Q(s,a)\Big]
$$

```python
def q_learning(env_step, n_states, n_actions, episodes=500,
               alpha=0.1, gamma=0.99, eps=0.1, seed=0):
    rng = np.random.default_rng(seed); Q = np.zeros((n_states, n_actions))
    for _ in range(episodes):
        s = 0; done = False
        while not done:
            a = rng.integers(n_actions) if rng.random() < eps else int(np.argmax(Q[s]))
            s2, r, done = env_step(s, a)
            Q[s, a] += alpha * (r + gamma * np.max(Q[s2]) - Q[s, a])   # Bellman update
            s = s2
    return Q

def env_step(s, a):                            # 3-state chain; action 1 = move right
    s2 = min(s + 1, 2) if a == 1 else max(s - 1, 0)
    return s2, (1.0 if s2 == 2 else 0.0), (s2 == 2)

print(np.argmax(q_learning(env_step, 3, 2), axis=1))   # policy: move right to goal
```
*Off-policy: the `max` over next actions learns the greedy policy's value while exploring with ε-greedy. SARSA uses the actually-taken next action (on-policy).*

## REINFORCE (policy gradient)

$$
\nabla_\theta J = \mathbb{E}\Big[\sum_t \nabla_\theta \log\pi_\theta(a_t\mid s_t)\,G_t\Big], \qquad G_t = \sum_{k\ge t}\gamma^{k-t} r_k
$$

```python
def reinforce_loss(log_probs, rewards, gamma=0.99):
    G, returns = 0.0, []
    for r in reversed(rewards):
        G = r + gamma * G; returns.insert(0, G)            # discounted return-to-go
    returns = np.array(returns)
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)   # baseline = variance reduction
    return -np.sum(np.array(log_probs) * returns)          # gradient ascent on expected return

print(round(reinforce_loss([np.log(0.6), np.log(0.4)], [1.0, 1.0]), 3))
```
*Push up log-prob of actions that led to high return, down for low. Subtracting a baseline reduces variance without bias — the seed of A2C/PPO.*

---

# 13. Interview Quick-Reference

**If they say "implement X", reach for:**

| Prompt | Pattern |
|---|---|
| "Softmax / numerically stable" | subtract max before exp |
| "Logistic regression" | sigmoid + BCE gradient $\frac{1}{n}X^\top(\hat y - y)$ |
| "Backprop a small net" | $\partial L/\partial z = \text{softmax}-\text{onehot}$, chain rule up |
| "Build autograd" | record parents + local `_backward`, topo-sort, reverse |
| "Self-attention" | $\text{softmax}(QK^\top/\sqrt{d_k})\,V$ |
| "Sample next token" | top-p > top-k > temperature > greedy |
| "Boosting vs bagging" | GBDT fits residuals (bias↓) / RF averages bootstraps (var↓) |
| "Evaluate a classifier" | confusion matrix → P/R/F1/AUC |
| "Cluster, no labels" | K-Means (hard) / GMM-EM (soft) / DBSCAN (density) |
| "Reduce dimensions" | PCA = top eigvecs of covariance |
| "Recommend items" | matrix factorization $\hat r = p_i^\top q_j$ |
| "Sample from a stream" | reservoir sampling |
| "Explore vs exploit" | ε-greedy → UCB → Thompson |
| "Most likely sequence" | Viterbi (DP over time × states) |

**Concepts they probe after the code:**

- **Bias–variance:** underfit = high bias (more capacity/features); overfit = high variance (regularize, more data, dropout, bagging).
- **Why $\sqrt{d_k}$ in attention:** keeps softmax out of saturation so gradients survive.
- **BatchNorm vs LayerNorm:** batch stats (train≠eval) vs per-example stats → LayerNorm for sequences/LLMs.
- **L1 vs L2:** L1 → sparsity (feature selection); L2 → small smooth weights.
- **Vanishing/exploding gradients:** residual connections, LSTM cell-state, ReLU/GELU, good init (He/Xavier), gradient clipping.
- **Bagging vs boosting:** bagging averages independent high-variance models; boosting sequentially reduces bias.
- **Generative vs discriminative:** Naive Bayes models $P(x\mid y)$; logistic regression models $P(y\mid x)$ directly.
- **On-policy vs off-policy:** SARSA learns the policy it follows; Q-learning learns the greedy one.
- **Reparameterization trick:** $z=\mu+\sigma\epsilon$ makes sampling differentiable (VAEs).

> **Closing thought:** every block here is *one model + one loss + one gradient step*. If you can name those three for any algorithm, you can derive the code on the spot — which is exactly what the interview tests.
