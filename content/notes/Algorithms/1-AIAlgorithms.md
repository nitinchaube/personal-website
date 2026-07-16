---
title: "AI / ML Algorithms"
date: 2026-06-22
summary: "Single-source, from-scratch Python + math reference for the ML / DL / AI algorithms you get asked to derive and code in Google-level interviews - classic ML, neural nets, tokenizers, attention (MHA/GQA/MLA/RoPE/MoE), LLM decoding, fine-tuning & alignment (LoRA/RLHF/DPO), embeddings & retrieval, generative models (VAE/GAN/diffusion), metrics and RL (up to PPO)."
tags: [ML, DL, AI, Algorithms, Interview]
---

This is my **single source of truth** for ML/DL/AI interview prep. In these interviews you are rarely asked to call `sklearn` or `torch` - you are asked to **derive the math and implement it**: softmax that doesn't overflow, backprop by hand, scaled dot-product attention, top-p sampling, ROC-AUC, gradient boosting.

Every block below has: the **formula**, a plain-`numpy` (or pure Python) **implementation**, a tiny **runnable example**, and the **complexity / interview gotcha**. Read top-to-bottom once, then use it as a night-before cheat-sheet.

> **The universal mental model:** almost every ML algorithm = **(1)** a model $f(x; \theta)$, **(2)** a loss $L$, **(3)** a rule to push $\theta$ down $\nabla_\theta L$. Master those three and most "implement X" questions collapse into the same template.

---

# Math & Numerical Building Blocks

These show up *inside* almost every other algorithm. Interviewers love them because one missing `max`-subtraction reveals whether you actually understand numerical stability.

## Sigmoid (numerically stable)

$$  
\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma'(z) = \sigma(z)(1 - \sigma(z))  
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
print(sigmoid(np.array([-1000., 0., 1000.])))   # [0.  0.5  1.] - no overflow
```

---

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

# Example - shift invariance
logits = np.array([2.0, 1.0, 0.1])
print(softmax(logits))            # [0.659  0.242  0.099]
print(softmax(logits + 1000))     # identical
```

---

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

---

## Cross-Entropy Loss (from logits)

$$  
L = -\sum_i y_i \log p_i, \qquad p = \text{softmax}(z) \Rightarrow L = \text{LSE}(z) - z_{\text{correct}}  
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

---

## KL Divergence (how far apart two distributions are)

$$  
D_{\mathrm{KL}}(P  Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)} \ge 0  
$$

Asymmetric: $D_{KL}(PQ) \ne D_{KL}(QP)$. Cross-entropy $= H(P) + D_{KL}(PQ)$, which is why minimizing CE w.r.t. the model = minimizing KL to the data distribution.

```python
def kl_divergence(p, q, eps=1e-12):
    p, q = np.asarray(p), np.asarray(q)
    return np.sum(p * np.log((p + eps) / (q + eps)))

print(round(kl_divergence([0.5, 0.5], [0.9, 0.1]), 4))   # 0.5108
```

---

## Activations & their derivatives (needed for backprop)

$$  
\text{ReLU}(z)=\max(0,z),\quad \tanh'(z)=1-\tanh^2(z),\quad \text{GELU}(z)\approx 0.5z\left(1+\tanh\big[\sqrt{\tfrac{2}{\pi}}(z+0.044715 z^3)\big]\right)  
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

---

## Distance / similarity metrics

$$  
\text{cos}(a,b)=\frac{a\cdot b}{\lVert a\rVert\lVert b\rVert},\qquad d_{L2}=\sqrt{\textstyle\sum_i (a_i-b_i)^2},\qquad d_{L1}=\textstyle\sum_i |a_i-b_i|  
$$

```python
def euclidean(a, b): return np.sqrt(np.sum((a - b) ** 2))
def manhattan(a, b): return np.sum(np.abs(a - b))
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)

print(cosine_similarity(np.array([1, 0]), np.array([10, 0])))   # 1.0 (direction only)
```

---

## Weight Initialization (Xavier / He)

$$  
\text{Xavier: } \sigma = \sqrt{\frac{2}{n_{in}+n_{out}}} (\text{tanh/sigmoid}), \qquad \text{He: } \sigma = \sqrt{\frac{2}{n_{in}}} (\text{ReLU})  
$$

```python
def init_weights(d_in, d_out, mode="he", seed=0):
    rng = np.random.default_rng(seed)
    std = np.sqrt(2/(d_in+d_out)) if mode == "xavier" else np.sqrt(2/d_in)
    return rng.normal(0, std, (d_in, d_out))
```

*Why it matters: bad init → activations/gradients vanish or explode across layers. He keeps ReLU variance ~1 per layer.*

---

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

# Optimizers & Gradient Descent

**Interview framing:** "Implement SGD with momentum" / "What does Adam actually do?" Write the update rules from memory.

## Gradient Descent (batch / mini-batch / stochastic)

$$  
\theta \leftarrow \theta - \eta  \nabla_\theta L(\theta)  
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

---

## Momentum, RMSProp, Adam

$$  
\text{Momentum: } v_t = \beta v_{t-1} + g_t,\quad \theta \leftarrow \theta - \eta v_t  
$$

$$  
\text{RMSProp: } s_t = \beta s_{t-1} + (1-\beta)g_t^2,\quad \theta \leftarrow \theta - \frac{\eta g_t}{\sqrt{s_t}+\epsilon}  
$$

$$  
\text{Adam: } m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t, v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2,  
\hat m_t = \frac{m_t}{1-\beta_1^t}, \hat v_t = \frac{v_t}{1-\beta_2^t},  
\theta \leftarrow \theta - \frac{\eta \hat m_t}{\sqrt{\hat v_t}+\epsilon}  
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

---

## Learning-Rate Schedule (linear warmup + cosine decay)

$$  
\eta_t =  
\begin{cases}  
\eta_0 \cdot \dfrac{t}{t_{\text{warmup}}} & t < t_{\text{warmup}} [2ex]  
\dfrac{1}{2}\eta_0\left(1 + \cos\Big(\pi \dfrac{t - t_{\text{warmup}}}{T - t_{\text{warmup}}}\Big)\right) & \text{otherwise}  
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

*Warmup is essential for Transformers - Adam's early variance estimate is noisy; ramping LR avoids blowing up.*

---

## AdamW & Gradient Clipping (what actually trains LLMs)

$$  
\text{AdamW: } \theta \leftarrow \theta - \eta\lambda\theta - \frac{\eta\hat m_t}{\sqrt{\hat v_t}+\epsilon}, \qquad \text{clip: } g \leftarrow g\cdot\min\!\left(1, \frac{c}{\lVert g\rVert}\right)  
$$

The subtlety: plain Adam with L2 penalty is *not* the same as weight decay, because Adam divides the penalty by $\sqrt{\hat v}$ too. **AdamW decouples** decay from the adaptive step - shrink weights by a flat $\eta\lambda$, then take the normal Adam step. This is the default optimizer for basically every Transformer.

```python
class AdamW:
    def __init__(self, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.01):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.m, self.v, self.t = None, None, 0

    def step(self, theta, grad, clip=None):
        if clip is not None:                              # global-norm gradient clipping
            norm = np.linalg.norm(grad)
            if norm > clip: grad = grad * (clip / norm)
        if self.m is None:
            self.m = np.zeros_like(theta); self.v = np.zeros_like(theta)
        self.t += 1
        self.m = self.b1*self.m + (1-self.b1)*grad
        self.v = self.b2*self.v + (1-self.b2)*grad**2
        m_hat = self.m / (1 - self.b1**self.t); v_hat = self.v / (1 - self.b2**self.t)
        theta = theta - self.lr * self.wd * theta         # decoupled weight decay (the "W")
        return theta - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

opt = AdamW(lr=0.1, wd=0.01); theta = np.array([5.0])
for _ in range(200): theta = opt.step(theta, grad=2*(theta-3), clip=1.0)
print(np.round(theta, 2))   # ~[3.] - clipping caps the early giant gradient
```

*Clipping by global norm (not per-element) tames the occasional exploding-gradient spike that would otherwise NaN a long training run. Typical clip value ~1.0.*

---

# Linear Models

## Linear Regression - Closed Form (Normal Equation)

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

*Closed form is $O(d^3)$ to invert $X^\top X$ - fine for small $d$, use GD when $d$ is large.*

---

## Linear Regression - Gradient Descent

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

---

## Logistic Regression (binary classification)

$$  
p = \sigma(\theta^\top x), \qquad L = -\frac{1}{n}\sum_i \big[y_i \log p_i + (1-y_i)\log(1-p_i)\big], \qquad \nabla_\theta L = \frac{1}{n}X^\top(\hat y - y)  
$$

The gradient of binary cross-entropy is the same clean form as linear regression - that's not a coincidence (both are GLMs).

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

---

## Regularization (Ridge L2 / Lasso L1)

$$  
L_{\text{ridge}} = L + \lambda \lVert\theta\rVert_2^2 \Rightarrow +2\lambda\theta \text{ in grad}, \qquad  
L_{\text{lasso}} = L + \lambda \lVert\theta\rVert_1 \Rightarrow +\lambda\text{sign}(\theta)  
$$

*L2 → small, smooth weights (shrinkage). L1 → sparse weights, drives some to **exactly zero** (feature selection). Never regularize the bias term.*

---

# Classic ML Algorithms (from scratch)

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

---

## K-Means Clustering (Lloyd's algorithm)

$$  
\min_{\mu_k}  J = \sum_{i} \lVert x_i - \mu_{c_i}\rVert^2, \qquad c_i = \arg\min_k \lVert x_i - \mu_k\rVert^2  
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

---

## Naive Bayes (Gaussian)

$$  
\hat y = \arg\max_c  P(c)\prod_j P(x_j \mid c), \qquad P(x_j\mid c)=\frac{1}{\sqrt{2\pi\sigma_{c,j}^2}}\exp\left(-\frac{(x_j-\mu_{c,j})^2}{2\sigma_{c,j}^2}\right)  
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

---

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

---

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

---

## AdaBoost (adaptive boosting)

$$  
\alpha_m = \frac{1}{2}\ln\frac{1-\epsilon_m}{\epsilon_m}, \qquad w_i \leftarrow w_i  e^{-\alpha_m y_i h_m(x_i)}, \qquad H(x)=\text{sign}\Big(\sum_m \alpha_m h_m(x)\Big)  
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

---

## Gradient Boosting (GBDT - the Kaggle/Google workhorse)

$$  
F_m(x) = F_{m-1}(x) + \nu h_m(x), \qquad h_m \approx -\frac{\partial L}{\partial F_{m-1}} (\text{= residual } y - F_{m-1} \text{ for MSE})  
$$

Each new tree fits the **negative gradient** (pseudo-residuals) of the loss - gradient descent in function space.

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

---

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

---

## LDA (Linear Discriminant Analysis - supervised projection)

Where PCA finds directions of max **variance** (unsupervised), LDA finds directions that best **separate labeled classes** - maximize between-class scatter $S_B$ relative to within-class scatter $S_W$:

$$  
\max_w \frac{w^\top S_B w}{w^\top S_W w} \;\Rightarrow\; \text{top eigenvectors of } S_W^{-1}S_B  
$$

```python
def lda_fit(X, y):
    classes = np.unique(y); mean_all = X.mean(axis=0)
    Sw = np.zeros((X.shape[1],) * 2); Sb = np.zeros((X.shape[1],) * 2)
    for c in classes:
        Xc = X[y == c]; mc = Xc.mean(axis=0)
        Sw += (Xc - mc).T @ (Xc - mc)                 # within-class scatter
        d = (mc - mean_all)[:, None]
        Sb += len(Xc) * (d @ d.T)                      # between-class scatter
    vals, vecs = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)
    return vecs[:, np.argsort(vals.real)[::-1]].real   # discriminant directions

X = np.r_[np.random.default_rng(0).normal([0,0], 0.3, (20,2)),
          np.random.default_rng(1).normal([4,4], 0.3, (20,2))]
y = np.array([0]*20 + [1]*20)
print(lda_fit(X, y).shape)   # (2, 2) - project onto column 0 for best class separation
```

*At most $C-1$ useful discriminants for $C$ classes. Doubles as a (Gaussian, shared-covariance) classifier - the generative cousin of logistic regression.*

---

## t-SNE / UMAP (non-linear visualization)

Both embed high-D points into 2-D for visualization by preserving *local neighborhoods* rather than global distances. t-SNE turns pairwise distances into probabilities in both spaces and minimizes their KL divergence; UMAP does a similar fuzzy-graph matching but is much faster and keeps more global structure.

```python
def tsne_affinities(X, sigma=1.0):
    # high-dim neighbor probabilities p_{j|i}: closer points -> higher prob
    D = np.sum((X[:, None] - X[None]) ** 2, axis=2)
    P = np.exp(-D / (2 * sigma ** 2)); np.fill_diagonal(P, 0)
    P /= P.sum(axis=1, keepdims=True)
    return (P + P.T) / (2 * len(X))                    # symmetrize
# Low-dim side uses a heavy-tailed Student-t kernel; minimize KL(P || Q) by GD on the 2-D coords.
print(np.round(tsne_affinities(X).sum(), 3))   # 1.0 - it's a probability distribution
```

*Gotchas: distances/cluster sizes between clusters are **not** meaningful, perplexity (≈ effective neighbor count) changes the picture a lot, and it's for viz - not a feature transform you feed downstream. Prefer UMAP at scale.*

---

## SVM (linear, hinge loss + sub-gradient descent)

$$  
\min_{w,b}  \frac{1}{2}\lVert w\rVert^2 + C\sum_i \max\big(0, 1 - y_i(w^\top x_i + b)\big)  
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

---

## EM for Gaussian Mixture Models (soft K-Means)

$$  
\textbf{E: } r_{ik} = \frac{\pi_k \mathcal{N}(x_i\mid\mu_k,\sigma_k^2)}{\sum_j \pi_j \mathcal{N}(x_i\mid\mu_j,\sigma_j^2)} \qquad  
\textbf{M: } \mu_k = \frac{\sum_i r_{ik}x_i}{\sum_i r_{ik}}, \sigma_k^2 = \frac{\sum_i r_{ik}(x_i-\mu_k)^2}{\sum_i r_{ik}}, \pi_k=\frac{\sum_i r_{ik}}{n}  
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

---

## DBSCAN (density-based clustering)

Grows clusters from **core points** (≥ `min_pts` neighbors within `eps`). Finds arbitrary shapes and labels outliers as noise - no need to pick $k$.

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

## Hierarchical (Agglomerative) Clustering

Start with every point its own cluster, then repeatedly **merge the two closest clusters** until you hit the target count. The full merge history is a dendrogram you can cut at any level - no need to fix $k$ up front.

```python
def agglomerative(X, n_clusters):
    clusters = [[i] for i in range(len(X))]
    def linkage(a, b):                                 # single linkage = closest pair
        return min(np.linalg.norm(X[i] - X[j]) for i in a for j in b)
    while len(clusters) > n_clusters:
        best, bi, bj = np.inf, 0, 1
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = linkage(clusters[i], clusters[j])
                if d < best: best, bi, bj = d, i, j
        clusters[bi] += clusters[bj]; clusters.pop(bj)  # merge closest two
    labels = np.zeros(len(X), int)
    for cid, c in enumerate(clusters):
        for i in c: labels[i] = cid
    return labels

X = np.r_[np.random.default_rng(0).normal(0, 0.2, (5, 2)),
          np.random.default_rng(1).normal(5, 0.2, (5, 2))]
print(len(set(agglomerative(X, 2))))   # 2
```

*Linkage choice matters: **single** (min) finds stringy clusters and chains, **complete** (max) makes compact ones, **Ward** minimizes variance (most K-Means-like). Naive cost is $O(n^3)$; $O(n^2\log n)$ with a heap.*

---

## Isolation Forest (anomaly detection)

Flip the usual framing: instead of modeling "normal," **isolate anomalies**. Build random trees that split on a random feature at a random threshold; outliers get isolated in far fewer splits, so their average path length is short → high anomaly score.

$$  
s(x) = 2^{-\frac{\mathbb{E}[h(x)]}{c(n)}}, \qquad c(n) = 2H(n-1) - \tfrac{2(n-1)}{n} \;(\text{avg path length of an unsuccessful BST search})  
$$

```python
def iforest_path(x, X, rng, depth=0, max_depth=8):
    if len(X) <= 1 or depth >= max_depth: return depth
    f = rng.integers(X.shape[1]); lo, hi = X[:, f].min(), X[:, f].max()
    if lo == hi: return depth
    split = rng.uniform(lo, hi)                         # random split
    sub = X[X[:, f] < split] if x[f] < split else X[X[:, f] >= split]
    return iforest_path(x, sub, rng, depth + 1, max_depth)

def iforest_score(x, X, n_trees=100):
    rng = np.random.default_rng(0)
    avg = np.mean([iforest_path(x, X, rng) for _ in range(n_trees)])
    n = len(X); c = 2 * (np.log(n - 1) + 0.5772) - 2 * (n - 1) / n
    return 2 ** (-avg / c)                              # ->1 = anomaly, ->0.5 = normal

Xi = np.random.default_rng(0).normal(0, 1, (100, 2))
print(round(iforest_score(np.array([0., 0]), Xi), 2),   # inlier, lower
      round(iforest_score(np.array([10., 10]), Xi), 2))  # outlier, higher
```

*Linear time, no distance metric, no density assumption - scales to high-D far better than DBSCAN/KNN-based detectors. Score > 0.5 leans anomalous.*

---

# Neural Networks from Scratch

## 2-Layer MLP with manual backprop (the canonical whiteboard question)

Forward: $z_1 = XW_1+b_1, a_1=\text{ReLU}(z_1), z_2=a_1W_2+b_2, p=\text{softmax}(z_2)$.  
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
print(np.argmax(net.forward(X), axis=1))   # [0 1 1 0] - solved the non-linear XOR
```

---

## Autograd Micro-Engine (reverse-mode, scalar - "build backprop")

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

---

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

*Gotcha: BatchNorm behaves differently in train vs eval (uses running stats at inference); LayerNorm doesn't - hence LLMs use LayerNorm/RMSNorm.*

---

## RMSNorm (what LLaMA/most modern LLMs actually use)

LayerNorm subtracts the mean *and* divides by std. RMSNorm drops the mean-centering and the bias - just rescale by the root-mean-square. Cheaper, and empirically the centering wasn't doing much:

$$  
\text{RMSNorm}(x) = \gamma \cdot \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \epsilon}}  
$$

```python
def rms_norm(x, gamma, eps=1e-6):
    return gamma * x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)

print(np.round(rms_norm(np.array([1., 2., 3.]), 1.0), 3))   # [0.463 0.926 1.389]
```

*One fewer reduction and no learned bias → a measurable speedup at LLM scale for no quality loss. This is why LLaMA, Mistral, Gemma all ship RMSNorm.*

---

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

---

## Label Smoothing (don't let the model get overconfident)

Instead of a hard one-hot target, put $1-\epsilon$ on the true class and spread $\epsilon$ across the rest. This caps the logit gap the model chases, improving calibration and generalization:

$$  
y_i^{LS} = (1-\epsilon)\,y_i + \frac{\epsilon}{C}  
$$

```python
def label_smoothed_ce(logits, labels, eps=0.1):
    N, C = logits.shape
    logp = np.log(softmax(logits, axis=1))
    onehot = np.zeros((N, C)); onehot[np.arange(N), labels] = 1
    soft = (1 - eps) * onehot + eps / C                # smoothed target
    return -np.mean(np.sum(soft * logp, axis=1))

print(round(label_smoothed_ce(np.array([[2., 0.5, 0.1]]), np.array([0])), 3))   # 0.43
```

*Used in the original Transformer and most image classifiers. Trade-off: slightly worse raw accuracy, better calibration and downstream robustness.*

---

## Convolution 2D (single channel)

$$  
\text{out}[i,j] = \sum_{u,v} \text{img}[i\cdots+u, j\cdots+v]\cdot K[u,v], \qquad  
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

*Know the output-size formula cold - it's a frequent quick-fire question. Params per conv layer: $(K\cdotK\cdotC_{in}+1)\cdot C_{out}$.*

---

## RNN cell & LSTM cell

$$  
\text{RNN: } h_t = \tanh(W_x x_t + W_h h_{t-1} + b)  
$$

$$  
\text{LSTM: } f_t,i_t,o_t = \sigma(\cdot), \tilde c_t = \tanh(\cdot), c_t = f_t\odot c_{t-1} + i_t\odot \tilde c_t, h_t = o_t\odot\tanh(c_t)  
$$

$$  
\text{GRU: } r_t,u_t = \sigma(\cdot),\ \tilde h_t = \tanh(W[r_t\odot h_{t-1}, x_t]),\ h_t = (1-u_t)\odot h_{t-1} + u_t\odot\tilde h_t  
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

def gru_cell(x_t, h_prev, p):
    z = np.concatenate([h_prev, x_t])
    r = sigmoid(p['Wr'] @ z + p['br'])                # reset gate
    u = sigmoid(p['Wu'] @ z + p['bu'])                # update gate
    cand = np.tanh(p['Wc'] @ np.concatenate([r * h_prev, x_t]) + p['bc'])
    return (1 - u) * h_prev + u * cand                # blend old state & candidate
```

*Why LSTM beats vanilla RNN: the **additive** cell-state path lets gradients flow without vanishing - the answer to "why do RNNs struggle with long sequences?". GRU merges the forget/input gates into one update gate and drops the separate cell state: fewer parameters, often the same accuracy, slightly faster.*

---

# Tokenizers

Before any model sees a single float, raw text has to become integers. *Which* subword algorithm you use decides vocab size, sequence length, and whether "unknown token" can even exist - a surprisingly deep interview topic in its own right.

## Word vs. Character vs. Subword

- **Word-level:** one token per word → tiny sequences, but huge vocab and every typo/rare word becomes `<unk>` (no generalization to unseen words).
- **Character-level:** tiny vocab, zero OOV, but sequences explode in length (attention is $O(n^2)$, so this hurts a lot) and each token carries little meaning.
- **Subword (the modern default):** frequent words stay whole, rare words split into meaningful pieces (`unhappiness` → `un`, `happi`, `ness`). Best of both: bounded vocab, bounded sequence length, no OOV.

---

## Byte-Pair Encoding (BPE) - the GPT-2/3 tokenizer

Start from characters; greedily merge the **most frequent adjacent pair** into a new symbol, repeat for a fixed number of merges. Encoding a new word just replays the learned merges in order.

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
        best = max(pairs, key=pairs.get); merges.append(best)   # greedy: just the most frequent pair
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

def bpe_encode(word, merges):
    tokens = list(word) + ['</w>']
    for pair in merges:                       # replay merges in the order they were learned
        tokens = _merge_pair(tokens, pair)
    return tokens

merges = bpe_train(["low", "lower", "lowest", "newest", "wider"], num_merges=6)
print(bpe_encode("lowest", merges))   # ['lowe', 'st</w>'] - subword split, not a full re-tokenization
```

*Time: $O(\text{merges}\cdot|\text{corpus}|)$ to train, $O(|\text{merges}|)$ to encode a word. Frequency-greedy, not likelihood-optimal - that's what Unigram fixes below.*

---

## WordPiece - the BERT tokenizer

Same merge-based skeleton as BPE, but scores pairs by **likelihood gain** instead of raw frequency, so it doesn't just glue together two very common symbols:

$$  
\text{score}(a,b) = \frac{\text{freq}(ab)}{\text{freq}(a)\cdot\text{freq}(b)}  
$$

```python
def wordpiece_score(vocab_pairs_freq, freq):
    # picks the pair whose *joint* frequency is high relative to its parts'
    # individual frequencies - favors merges that are more than coincidence.
    return {pair: f / (freq[pair[0]] * freq[pair[1]]) for pair, f in vocab_pairs_freq.items()}
```

Continuation subwords are marked with `##` (e.g. `play`, `##ing`) so detokenization knows there was no space. Encoding is **greedy longest-match-first**: repeatedly chop off the longest prefix that's in the vocab.

```python
def wordpiece_encode(word, vocab):
    tokens, start = [], 0
    while start < len(word):
        end = len(word)
        piece = None
        while end > start:                      # longest-match-first
            cand = word[start:end] if start == 0 else "##" + word[start:end]
            if cand in vocab:
                piece = cand; break
            end -= 1
        if piece is None:
            return ["[UNK]"]                     # unlike BPE, WordPiece *can* fail to cover a char
        tokens.append(piece); start = end
    return tokens

vocab = {"play", "##ing", "##er", "un", "##happy"}
print(wordpiece_encode("playing", vocab))   # ['play', '##ing']
```

*Gotcha: unlike byte-level BPE, classic WordPiece is defined over characters, so an out-of-vocab character can still produce `[UNK]` - this is exactly why GPT moved to byte-level BPE.*

---

## Unigram Language Model & SentencePiece

Runs the merge idea **backwards**: start from a huge candidate vocab (all substrings), then iteratively drop the subwords whose removal hurts the corpus log-likelihood *least*, under a unigram LM where each token has probability $p(t)$. Segmentation of a word is the Viterbi-optimal split under that LM, not a greedy left-to-right match - so the same word can even be segmented differently depending on context (used for subword regularization/data augmentation).

$$  
p(x) = \prod_{t\in\text{segmentation}(x)} p(t), \qquad \text{segmentation}^*(x) = \arg\max_{\text{segmentations}} \sum_t \log p(t)  
$$

```python
def viterbi_segment(word, token_logp, max_len=6):
    # token_logp: dict[str, float] = log p(token). DP over end positions, like Viterbi decoding.
    n = len(word)
    best = [(-np.inf, None)] * (n + 1); best[0] = (0.0, None)
    for end in range(1, n + 1):
        for start in range(max(0, end - max_len), end):
            tok = word[start:end]
            if tok in token_logp:
                score = best[start][0] + token_logp[tok]
                if score > best[end][0]:
                    best[end] = (score, start)
    path, pos = [], n
    while pos > 0:
        start = best[pos][1]; path.insert(0, word[start:pos]); pos = start
    return path

logp = {"un": -1.0, "happy": -1.5, "happi": -1.8, "ness": -1.2, "unhappy": -3.0}
print(viterbi_segment("unhappy", logp))   # ['un', 'happy'] - beats 'unhappy' whole (-3.0 < -2.5)
```

**SentencePiece** is the library that made this practical: it treats raw text as a **byte stream, whitespace included** (encoded as `▁`), so there's no separate pre-tokenization step and detokenization is lossless - this is why LLaMA/T5/Gemini-family models use it.

---

## Byte-level BPE (no `<unk>`, ever)

GPT-2/3/4's trick: run BPE over raw **UTF-8 bytes** (base vocab = 256) instead of Unicode characters. Since every possible byte is already in the base vocabulary, **every string is representable** - emoji, code, any language, even malformed text - with zero `<unk>` tokens.

```python
def to_byte_symbols(text):
    return [f"<{b:02x}>" for b in text.encode("utf-8")]   # base alphabet of 256 symbols

print(to_byte_symbols("café🙂")[:6])   # multi-byte UTF-8 chars just become 2-4 byte symbols
```

*Trade-off: rare Unicode (emoji, CJK) costs more tokens (2-4 bytes each) than a dedicated char-level vocab would, but you never crash on unseen input - a huge robustness win for a production LLM.*

---

## Special tokens, vocab size & sequence-length trade-offs

- **Reserved tokens:** `[BOS]/[EOS]` (sequence boundaries), `[PAD]` (batch padding, usually masked out of loss/attention), `[UNK]` (only needed for non-byte-level tokenizers), `[CLS]/[SEP]` (BERT-style task tokens).
- **Vocab size trade-off:** bigger vocab → shorter sequences (cheaper attention, more context per token) but a bigger, sparser embedding/softmax matrix and rarer tokens are seen less often during training. GPT-family sits around 50k–100k+; that number is itself a tuned hyperparameter.
- **Fertility:** tokens-per-word varies by language - tokenizers trained mostly on English under-serve other languages (more tokens per word there), which is a real fairness/cost issue for multilingual LLMs.
- **Interview one-liner:** *BPE = greedy frequency merges (GPT), WordPiece = greedy likelihood-ratio merges + `##` (BERT), Unigram = prune from a big vocab + Viterbi segmentation (SentencePiece/T5), byte-level BPE = BPE over bytes so OOV is structurally impossible (GPT-2+).*

---

# Transformers & Attention

*The* deep-learning interview topic. Be able to write scaled dot-product attention from memory.

## Scaled Dot-Product Attention

$$  
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V  
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
print(np.round(w, 2))   # [[0.67 0.33]] - attends more to the matching key
```

> **Why divide by $\sqrt{d_k}$?** Dot products have variance $\propto d_k$; without scaling, softmax saturates into a near one-hot and gradients vanish. Most-asked attention follow-up.

---

## Multi-Head Attention

$$  
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V), \qquad \text{MHA} = \text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O  
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

---

## MQA, GQA & MLA (shrinking the KV cache)

Standard MHA gives every query head its **own** key/value head, so the KV cache grows with $H$ heads. Three variants trade some quality for a smaller cache, which is what actually limits inference batch size / context length:

- **MQA (Multi-Query Attention):** all $H$ query heads share a **single** K/V head ($H_{kv}=1$). Max cache compression, but the biggest quality hit.
- **GQA (Grouped-Query Attention):** query heads are split into $G$ groups, each group shares one K/V head ($1 < H_{kv}=G < H$) - interpolates between MHA ($G=H$) and MQA ($G=1$). This is the deployed sweet spot (LLaMA-2/3 70B, Mistral, Gemma).
- **MLA (Multi-head Latent Attention, DeepSeek-V2/V3):** instead of fewer K/V *heads*, compress K and V for **all** heads into one shared low-rank latent vector per token, then up-project back to per-head K/V at attention time. Shrinks the cache even further than GQA while staying much closer to full-MHA quality (you cache the small latent, not $H_{kv}$ full-size K/V pairs).

$$  
H_{kv}=H\ (\text{MHA}) \;>\; G\ (\text{GQA}) \;>\; 1\ (\text{MQA}), \qquad \text{MLA: cache } c_t\in\mathbb{R}^{r},\ r \ll H\cdot d_{head}  
$$

```python
def grouped_query_attention(X, Wq, Wk, Wv, Wo, num_heads, num_kv_heads):
    # num_kv_heads == num_heads -> plain MHA; == 1 -> MQA; in between -> GQA.
    L, d_model = X.shape; d_k = d_model // num_heads
    Q, K, V = X @ Wq, X @ Wk, X @ Wv                      # K, V projected to fewer heads
    def split(M, h): return M.reshape(L, h, d_k).transpose(1, 0, 2)
    Qh, Kh, Vh = split(Q, num_heads), split(K, num_kv_heads), split(V, num_kv_heads)
    group = num_heads // num_kv_heads                      # query heads per shared KV head
    heads = [scaled_dot_product_attention(Qh[h], Kh[h // group], Vh[h // group])[0]
             for h in range(num_heads)]
    return np.concatenate(heads, axis=-1) @ Wo

rng = np.random.default_rng(0); X = rng.normal(size=(4, 8))
Wq = rng.normal(size=(8, 8))
for num_kv in [4, 2, 1]:                                   # MHA, GQA, MQA - same call, different H_kv
    Wk = Wv = rng.normal(size=(8, num_kv * 2))
    print(num_kv, grouped_query_attention(X, Wq, Wk, Wv, rng.normal(size=(8, 8)),
                                           num_heads=4, num_kv_heads=num_kv).shape)   # all (4, 8)
```

*Interview one-liner: MHA = quality ceiling; MQA = smallest KV cache but the biggest quality drop; GQA = the practical middle ground almost every open-weight model ships; MLA = compress into a shared low-rank latent instead of dropping heads, so DeepSeek keeps near-MHA quality at a smaller-than-GQA cache.*

---

## Sinusoidal Positional Encoding

$$  
PE_{(pos, 2i)} = \sin\Big(\frac{pos}{10000^{2i/d}}\Big), \qquad PE_{(pos, 2i+1)} = \cos\Big(\frac{pos}{10000^{2i/d}}\Big)  
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

*Attention is permutation-invariant - without positions it can't tell word order. Modern LLMs use **RoPE** (rotary) which encodes relative position via rotation.*

---

## RoPE (Rotary Position Embedding - what modern LLMs actually use)

Instead of *adding* a position vector, RoPE **rotates** each 2-D slice of Q and K by an angle proportional to the token's position. The dot product $q_m\cdot k_n$ then depends only on the **relative** offset $m-n$ - so the model extrapolates to longer contexts far better than fixed sinusoids.

$$  
\theta_i = \text{base}^{-2i/d}, \qquad \big(x_{2i}, x_{2i+1}\big) \mapsto \big(x_{2i}\cos m\theta_i - x_{2i+1}\sin m\theta_i,\; x_{2i}\sin m\theta_i + x_{2i+1}\cos m\theta_i\big)  
$$

```python
def rope(x, base=10000):
    # x: (seq_len, d) - rotate each (even, odd) dimension pair by pos * theta_i.
    seq_len, d = x.shape
    pos = np.arange(seq_len)[:, None]; i = np.arange(0, d, 2)[None, :]
    theta = pos / base ** (i / d)                  # angle per position & frequency
    cos, sin = np.cos(theta), np.sin(theta)
    x1, x2 = x[:, 0::2], x[:, 1::2]
    out = np.empty_like(x)
    out[:, 0::2] = x1 * cos - x2 * sin             # 2-D rotation
    out[:, 1::2] = x1 * sin + x2 * cos
    return out

print(rope(np.ones((3, 4))).shape)   # (3, 4) - applied to Q and K before the QK^T dot product
```

*Applied to Q and K only (not V), per layer, right before attention. Low frequencies handle long-range position, high frequencies fine local order. NTK/YaRN scaling stretches RoPE to extend context windows post-training.*

---

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

---

## Mixture of Experts (MoE - scale params without scaling compute)

Replace the single dense FFN with $N$ expert FFNs plus a **router** that sends each token to only its top-$k$ experts (usually $k{=}1$ or $2$). You get a model with a huge parameter count but a nearly constant per-token FLOP cost - the trick behind Mixtral, DeepSeek-V3, and the GPT-4-class "sparse" models.

$$  
g = \text{softmax}(W_g x),\qquad y = \sum_{e\in\text{top-}k(g)} \frac{g_e}{\sum_{e'} g_{e'}}\, f_e(x)  
$$

```python
def moe_layer(x, experts, W_gate, k=2):
    # experts: list of weight matrices (each an FFN); route x to its top-k.
    gate = softmax(W_gate @ x)                      # (num_experts,) routing scores
    topk = np.argsort(gate)[::-1][:k]
    weights = gate[topk] / gate[topk].sum()         # renormalize over chosen experts
    y = sum(w * (experts[e] @ x) for w, e in zip(weights, topk))
    return y, topk

rng = np.random.default_rng(0); d = 4
experts = [rng.normal(size=(d, d)) for _ in range(4)]
out, chosen = moe_layer(rng.normal(size=d), experts, rng.normal(size=(4, d)), k=2)
print(out.shape, chosen)   # only 2 of 4 experts run per token
```

*Key headache: **load balancing** - without an auxiliary loss the router collapses onto a few experts. Also, all experts' weights must live in memory even though each token uses few, so MoE trades compute for memory/bandwidth.*

---

## KV Cache (the inference-efficiency question)

At generation step $t$, the new token must attend to all prior tokens. Recomputing every K/V is $O(N^2)$; instead cache each token's K and V once. Cache size in bytes:

$$  
\text{KV bytes} = 2 \cdot L \cdot H_{kv} \cdot d_{head} \cdot N_{seq} \cdot N_{batch} \cdot \text{bytes/elem}  
$$

(leading 2 = K and V; $L$ layers, $H_{kv}$ KV-heads). The KV cache - not the weights - usually caps how many concurrent users you can batch. **MQA/GQA** (above) shrink $H_{kv}$ to cut it; **MLA** replaces the $H_{kv}$ term with a much smaller latent dimension $r$.

---

# LLM Decoding & Sampling

Given logits over the vocabulary, *how you pick the next token* is its own family of algorithms - heavily asked for LLM roles.

## Greedy & Temperature Sampling

$$  
p_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}} \quad (T<1 \text{ sharpens}, T>1 \text{ flattens}, T\to0 \text{ = greedy})  
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

---

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

---

## Top-p (Nucleus) Sampling - the modern default

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

---

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

*Higher-likelihood sequences (good for translation), but bland for open-ended chat - hence sampling. Length-normalize scores to avoid favoring short sequences.*

---

## Speculative Decoding (2-3x faster inference, same distribution)

A small **draft** model proposes $\gamma$ tokens cheaply; the big **target** model then verifies them all in a *single* forward pass. Accept each draft token with probability $\min(1, p_{\text{target}}/p_{\text{draft}})$; on the first rejection, resample from the adjusted residual and stop. The math guarantees the output is distributed *exactly* as if you'd sampled from the target model alone - pure speedup, no quality loss.

```python
def speculative_decode(draft_logits_fn, target_logits_fn, prefix, gamma=4, seed=0):
    rng = np.random.default_rng(seed)
    seq, drafted = list(prefix), []
    for _ in range(gamma):                                   # 1. draft gamma tokens cheaply
        p = softmax(draft_logits_fn(seq))
        t = int(rng.choice(len(p), p=p)); drafted.append((t, p)); seq.append(t)
    accepted = list(prefix)
    for t, p_draft in drafted:                               # 2. verify with the target model
        p_tgt = softmax(target_logits_fn(accepted))
        if rng.random() < min(1.0, p_tgt[t] / (p_draft[t] + 1e-12)):
            accepted.append(t)                               # accept
        else:
            resid = np.maximum(p_tgt - p_draft, 0); resid /= resid.sum()
            accepted.append(int(rng.choice(len(resid), p=resid)))   # resample & stop
            break
    return accepted

V = 5
draft = lambda s: np.random.default_rng(sum(s) % 100).normal(size=V)
target = lambda s: np.random.default_rng((sum(s) + 1) % 100).normal(size=V)
print(speculative_decode(draft, target, [0], gamma=4))
```

*The win comes from verifying $\gamma$ tokens in one batched target pass instead of $\gamma$ sequential ones - memory-bandwidth-bound LLM decoding loves this. Medusa / EAGLE replace the separate draft model with extra prediction heads.*

---

> **Deeper dive:** I go deep on the math and the full algorithm zoo (GPTQ, AWQ, SmoothQuant, GGUF k-quants, QAT/STE, NF4, BitNet) in the standalone [Quantization primer](/notes/Primers/Quantization).

---

# Fine-Tuning & Alignment

Pretraining gives you a model that *completes* text; this section is how you make it *useful and safe* - cheap parameter-efficient tuning, then preference alignment. The hottest LLM-interview area right now.

## LoRA / QLoRA (parameter-efficient fine-tuning)

Full fine-tuning updates all $d\times d$ weights per layer - billions of params. LoRA **freezes** the pretrained $W$ and learns a tiny low-rank update $\Delta W = \frac{\alpha}{r}AB$ with $A\in\mathbb{R}^{d\times r}, B\in\mathbb{R}^{r\times d}, r\ll d$. You train ~0.1% of the parameters and can hot-swap adapters per task.

$$  
h = xW + \frac{\alpha}{r}\,x A B, \qquad A \sim \mathcal{N}(0,\sigma^2),\; B = 0 \;\text{at init (so } \Delta W = 0)  
$$

```python
def lora_forward(x, W, A, B, alpha=8, r=4):
    return x @ W + (alpha / r) * (x @ A @ B)        # frozen W + trainable low-rank path

d_in, d_out, r = 6, 5, 2
rng = np.random.default_rng(0)
W = rng.normal(size=(d_in, d_out))                  # frozen pretrained weights
A = rng.normal(size=(d_in, r)) * 0.01               # B starts at 0 so training begins as a no-op
B = np.zeros((r, d_out))
print(lora_forward(rng.normal(size=(3, d_in)), W, A, B).shape)   # (3, 5)
```

*B init at zero means the adapter starts as identity - no shock to the pretrained model. **QLoRA** goes further: quantize the frozen base to 4-bit (NF4) and train the LoRA adapters in bf16, so a 65B model fine-tunes on a single GPU. At inference you can fold $\frac{\alpha}{r}AB$ back into $W$ for zero added latency.*

---

## RLHF (reward model + PPO)

Align a model to human preferences in three stages: **(1)** supervised fine-tune (SFT) on demonstrations; **(2)** train a **reward model** on human A/B comparisons using the Bradley-Terry loss; **(3)** optimize the policy against that reward with **PPO** (see the RL section), penalized by a KL term that keeps it near the SFT model so it doesn't reward-hack into gibberish.

$$  
L_{RM} = -\log\sigma\big(r(x, y_w) - r(x, y_l)\big), \qquad \max_\pi \; \mathbb{E}\big[r(x,y)\big] - \beta\, D_{KL}\!\big(\pi \,\|\, \pi_{SFT}\big)  
$$

```python
def reward_model_loss(r_chosen, r_rejected):
    # Bradley-Terry: chosen response should score higher than rejected.
    return -np.log(sigmoid(r_chosen - r_rejected)).mean()

def rlhf_reward(r_score, logp_policy, logp_ref, beta=0.1):
    # per-token reward handed to PPO: task reward minus KL drift from the frozen SFT model.
    return r_score - beta * (logp_policy - logp_ref)

print(round(float(reward_model_loss(np.array([2.0]), np.array([0.5]))), 4))   # low: chosen >> rejected
print(round(float(rlhf_reward(1.0, -2.0, -2.3)), 4))
```

*Why the KL leash: without it PPO drives the policy to whatever maximizes the (imperfect) reward model, producing degenerate high-reward text. RLHF is powerful but a fragile multi-model pipeline - which is exactly why DPO exists.*

---

## DPO (Direct Preference Optimization - RLHF without RL)

DPO skips the reward model and PPO entirely. A bit of algebra shows the optimal RLHF policy has a closed form, and you can optimize directly on preference pairs with a simple classification loss - the model *is* its own implicit reward. Far more stable, no sampling loop, no separate reward network.

$$  
L_{DPO} = -\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)  
$$

```python
def dpo_loss(lp_chosen, lp_rejected, ref_chosen, ref_rejected, beta=0.1):
    # lp_* = policy log-prob of the whole response; ref_* = frozen reference model's log-prob.
    logits = beta * ((lp_chosen - ref_chosen) - (lp_rejected - ref_rejected))
    return -np.log(sigmoid(logits)).mean()          # push chosen up, rejected down, vs reference

print(round(float(dpo_loss(np.array([-2.0]), np.array([-3.0]),
                           np.array([-2.5]), np.array([-2.5]))), 4))
```

*The `ref` model (frozen SFT copy) plays the role RLHF's KL penalty did - it stops the policy drifting too far. DPO (and cousins like IPO/KTO/ORPO) has largely replaced PPO-RLHF for open-weight models because it's one loss on a static dataset.*

---

# Representation Learning & Embeddings

Google leans heavily on retrieval, recommendations and embeddings - these come up constantly.

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

---

## Word2Vec - Skip-Gram with Negative Sampling

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

---

## Contrastive & Triplet Loss (metric learning)

$$  
\text{Triplet: } L = \max\big(0, \lVert a-p\rVert^2 - \lVert a-n\rVert^2 + \alpha\big)  
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

---

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

## Approximate Nearest Neighbors (LSH & HNSW - the RAG backbone)

Once you have embeddings, retrieval = "find the nearest vectors to this query." Exact search is $O(N d)$ per query - hopeless at billions of vectors. ANN indexes trade a little recall for orders-of-magnitude speed.

- **LSH (Locality-Sensitive Hashing):** hash vectors so *nearby* ones collide. Random hyperplanes give a binary code per vector (sign of $x\cdot w$); only compare within the same bucket.
- **HNSW (the production default):** a multi-layer navigable small-world graph. Greedily hop toward the query through a sparse top layer, then descend to denser layers - $O(\log N)$ hops. This is what FAISS / pgvector / most vector DBs use.

```python
def lsh_buckets(X, n_planes=8, seed=0):
    rng = np.random.default_rng(seed)
    planes = rng.normal(size=(n_planes, X.shape[1]))
    codes = (X @ planes.T > 0).astype(int)           # sign bits = which side of each hyperplane
    buckets = {}
    for i, code in enumerate(codes):
        buckets.setdefault(tuple(code), []).append(i)   # collide -> same bucket -> candidates
    return buckets, planes

X = np.random.default_rng(0).normal(size=(20, 5))
buckets, planes = lsh_buckets(X)
print(len(buckets))   # query hashes to one bucket; only rank vectors sharing it
```

*More planes → finer buckets → higher precision but lower recall (fewer collisions). Product Quantization (PQ) is the other big idea: compress vectors to bytes so billions fit in RAM. Metric matters - normalize for cosine vs raw L2.*

---

# Generative Models

## Variational Autoencoder (VAE)

Maximize the ELBO; the **reparameterization trick** ($z = \mu + \sigma\odot\epsilon$) makes sampling differentiable:

$$  
\mathcal{L} = \underbrace{\mathbb{E}*{q(z|x)}[\log p(x|z)]}*{\text{reconstruction}} - \underbrace{D_{KL}\big(q(z|x)p(z)\big)}*{\text{regularizer}}, \qquad*  
*D*{KL} = -\tfrac{1}{2}\sum_j\big(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\big)  
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

---

## Generative Adversarial Network (GAN)

$$  
\min_G\max_D  \mathbb{E}*{x}[\log D(x)] + \mathbb{E}*{z}[\log(1 - D(G(z)))]  
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

## Diffusion Models (DDPM - how Stable Diffusion / Sora / Imagen work)

Two processes. **Forward:** gradually add Gaussian noise over $T$ steps until data becomes pure noise - and thanks to a closed form, you can jump to any step $t$ in one shot. **Reverse:** train a network $\epsilon_\theta$ to *predict the noise* added at step $t$; sampling then denoises from pure noise back to data. The loss is beautifully simple - just MSE on the predicted noise.

$$  
x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon, \qquad L = \mathbb{E}_{t,x_0,\epsilon}\big[\lVert \epsilon - \epsilon_\theta(x_t, t)\rVert^2\big]  
$$

```python
def ddpm_schedule(T=100):
    betas = np.linspace(1e-4, 0.02, T)               # variance added per step
    alphas = 1 - betas
    return betas, alphas, np.cumprod(alphas)         # abar[t] = prod of alphas up to t

def q_sample(x0, t, abar, seed=0):
    # forward: jump straight to a noisy x_t (the closed form that makes training cheap)
    eps = np.random.default_rng(seed).normal(size=np.shape(x0))
    return np.sqrt(abar[t]) * x0 + np.sqrt(1 - abar[t]) * eps, eps

def ddpm_step(xt, t, eps_pred, betas, alphas, abar, seed=0):
    # reverse: one denoising step using the network's predicted noise
    mean = (xt - (1 - alphas[t]) / np.sqrt(1 - abar[t]) * eps_pred) / np.sqrt(alphas[t])
    if t == 0: return mean
    return mean + np.sqrt(betas[t]) * np.random.default_rng(seed).normal(size=np.shape(xt))

betas, alphas, abar = ddpm_schedule()
xt, eps = q_sample(np.array([1.0, -1.0]), 50, abar)
print(np.round(ddpm_step(xt, 50, eps, betas, alphas, abar), 2))   # one step back toward data
```

*Training: sample a random $t$, noise the image, ask the net to predict the noise, MSE. That's it. The score-based view says $\epsilon_\theta \propto -\nabla_{x}\log p(x_t)$ (the "score"). **Classifier-free guidance** (mix conditional & unconditional predictions) is what makes text-to-image prompts actually follow the prompt; DDIM samplers cut $T$ from 1000 to ~20 steps.*

---

# Evaluation Metrics (from scratch)

Knowing *which* metric - and computing it without `sklearn` - is a frequent screen.

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

---

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

---

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

---

## NDCG (ranking quality - recsys/search)

$$  
\text{DCG@k} = \sum_{i=1}^{k}\frac{2^{rel_i}-1}{\log_2(i+1)}, \qquad \text{NDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}  
$$

```python
def ndcg(relevances, k=None):
    relevances = np.array(relevances); k = k or len(relevances)
    def dcg(r): return np.sum((2**r - 1) / np.log2(np.arange(2, len(r) + 2)))
    ideal = np.sort(relevances)[::-1]
    return dcg(relevances[:k]) / (dcg(ideal[:k]) + 1e-12)

print(round(ndcg([3, 2, 3, 0, 1, 2]), 3))   # 0.949 - rewards relevant items ranked high
```

*The discount ($1/\log_2$) means relevance at rank 1 matters far more than at rank 10. Normalized so 1.0 = perfect ranking.*

---

## PR-AUC / Average Precision (for rare positives)

On heavily imbalanced data (fraud, retrieval), ROC-AUC looks deceptively good because true negatives dominate. The **precision-recall** curve - and its area, **average precision** - focuses on the positive class, which is what you actually care about.

$$  
\text{AP} = \sum_k \big(R_k - R_{k-1}\big)\,P_k \quad(\text{area under the precision-recall curve})  
$$

```python
def average_precision(y_true, scores):
    order = np.argsort(scores)[::-1]; y = y_true[order]
    tp = np.cumsum(y); fp = np.cumsum(1 - y)
    precision = tp / (tp + fp); recall = tp / tp[-1]
    ap, prev_r = 0.0, 0.0
    for p, r in zip(precision, recall):
        ap += p * (r - prev_r); prev_r = r          # sum precision weighted by recall gained
    return ap

print(round(average_precision(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])), 3))   # 0.833
```

*Rule of thumb: ROC-AUC for roughly balanced classes, PR-AUC when positives are rare. A random baseline scores AP = (positive rate), not 0.5.*

---

## Perplexity (the language-model metric)

Perplexity is just the exponentiated average negative log-likelihood - "how many equally-likely tokens is the model choosing among on average?" Lower is better; it's the standard intrinsic LM score.

$$  
\text{PPL} = \exp\!\left(-\frac{1}{N}\sum_i \log p(x_i \mid x_{<i})\right) = \exp(\text{mean cross-entropy})  
$$

```python
def perplexity(logits, targets):
    logp = np.log(softmax(logits, axis=1))
    nll = -logp[np.arange(len(targets)), targets].mean()
    return np.exp(nll)

print(round(perplexity(np.array([[2., 1., 0.], [0., 2., 1.]]), np.array([0, 1])), 3))   # 1.503
```

*PPL = 1 is a perfect oracle; PPL = vocab size is uniform guessing. Only comparable across models with the **same tokenizer** - byte vs word tokenization changes the denominator.*

---

## BLEU & ROUGE (text generation overlap)

**BLEU** (translation) measures n-gram **precision** with a brevity penalty; **ROUGE-L** (summarization) measures **recall** via the longest common subsequence. Both are cheap n-gram proxies - increasingly supplemented by embedding-based scores (BERTScore) and LLM judges.

```python
def bleu(candidate, reference, max_n=4):
    score = 0.0
    for n in range(1, max_n + 1):
        cand = Counter(tuple(candidate[i:i+n]) for i in range(len(candidate)-n+1))
        ref = Counter(tuple(reference[i:i+n]) for i in range(len(reference)-n+1))
        overlap = sum(min(c, ref[g]) for g, c in cand.items())
        score += 0.25 * np.log((overlap + 1e-9) / max(sum(cand.values()), 1))
    bp = min(1.0, np.exp(1 - len(reference) / max(len(candidate), 1)))   # brevity penalty
    return bp * np.exp(score)

def rouge_l(candidate, reference):
    m, n = len(candidate), len(reference); dp = np.zeros((m+1, n+1), int)
    for i in range(1, m+1):
        for j in range(1, n+1):
            dp[i, j] = dp[i-1, j-1] + 1 if candidate[i-1] == reference[j-1] else max(dp[i-1, j], dp[i, j-1])
    lcs = dp[m, n]
    if lcs == 0: return 0.0
    prec, rec = lcs / m, lcs / n
    return 2 * prec * rec / (prec + rec)            # F1 over the longest common subsequence

print(round(bleu("the cat sat on the mat".split(), "the cat sat on the rug".split()), 3))   # 0.76
print(round(rouge_l("the cat sat down".split(), "the cat sat".split()), 3))                 # 0.857
```

*The brevity penalty stops BLEU from rewarding one-word high-precision outputs. Both correlate only loosely with human judgment - know their limits, and that BERTScore / LLM-as-judge are the modern complements.*

---

# Probability, Sampling & Bandits

## Sampling from a Categorical (inverse-CDF & Gumbel-max)

$$  
\text{Gumbel-max: } \arg\max_i\big(z_i + g_i\big), g_i = -\log(-\log u_i), u_i\sim U(0,1) \sim \text{softmax}(z)  
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

---

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

---

## Multi-Armed Bandit - ε-greedy & UCB

$$  
\varepsilon\text{-greedy: explore w.p. } \varepsilon, \text{ else } \arg\max_a Q_a \qquad  
\text{UCB: } a_t = \arg\max_a\Big(Q_a + c\sqrt{\tfrac{\ln t}{N_a}}\Big)  
$$

UCB adds an "optimism" bonus that shrinks as an arm is pulled more - principled exploration.

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

## Thompson Sampling (Bayesian posterior sampling)

Keep a posterior over each arm's reward (Beta for Bernoulli rewards). Each round, **sample** a plausible mean from every arm's posterior and pull the argmax - arms you're uncertain about occasionally sample high, giving exploration for free, with no explicit bonus term. Often beats UCB empirically.

$$  
\theta_a \sim \text{Beta}(\alpha_a, \beta_a), \quad a_t = \arg\max_a \theta_a, \qquad (\alpha_a,\beta_a) \mathrel{+}= (r,\, 1-r)  
$$

```python
def thompson_bandit(true_means, steps=2000, seed=0):
    rng = np.random.default_rng(seed); K = len(true_means)
    a = np.ones(K); b = np.ones(K)                   # Beta(1,1) = uniform prior per arm
    for _ in range(steps):
        samples = rng.beta(a, b)                     # sample a plausible mean per arm
        arm = int(np.argmax(samples))               # pull the sampled-best arm
        reward = rng.random() < true_means[arm]
        a[arm] += reward; b[arm] += 1 - reward       # Bayesian update of the posterior
    return int(np.argmax(a / (a + b)))

print(thompson_bandit([0.2, 0.5, 0.8]))   # 2 - converges on the best arm
```

*The posterior automatically shrinks exploration as evidence accumulates - no schedule to tune, unlike ε or UCB's $c$. Extends cleanly to Gaussian rewards and contextual bandits.*

---

# Sequence Models & Dynamic Programming in ML

## Viterbi Algorithm (most-likely hidden state sequence - HMM decoding)

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

## Edit (Levenshtein) Distance

Minimum insert/delete/substitute edits to turn one string into another. The archetypal 2-D DP - same grid as sequence alignment, spell-check, and diff.

$$  
dp[i][j] = \min\big(dp[i{-}1][j]{+}1,\; dp[i][j{-}1]{+}1,\; dp[i{-}1][j{-}1] + \mathbb{1}[a_i \ne b_j]\big)  
$$

```python
def edit_distance(a, b):
    m, n = len(a), len(b); dp = np.zeros((m+1, n+1), int)
    dp[:, 0] = np.arange(m+1); dp[0, :] = np.arange(n+1)   # cost of pure insert/delete
    for i in range(1, m+1):
        for j in range(1, n+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i, j] = min(dp[i-1, j] + 1,          # delete
                           dp[i, j-1] + 1,          # insert
                           dp[i-1, j-1] + cost)     # match / substitute
    return dp[m, n]

print(edit_distance("kitten", "sitting"))   # 3
```

---

## Dynamic Time Warping (DTW)

Edit distance's continuous cousin: align two time series that are out of phase or run at different speeds (speech, gestures, sensor traces) by warping the time axis. Same DP grid, but the cell cost is a distance and diagonal moves aren't penalized for length.

```python
def dtw(a, b):
    n, m = len(a), len(b); D = np.full((n+1, m+1), np.inf); D[0, 0] = 0
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = abs(a[i-1] - b[j-1])
            D[i, j] = cost + min(D[i-1, j], D[i, j-1], D[i-1, j-1])   # warp: reuse a step
    return D[n, m]

print(round(dtw([1, 2, 3, 4, 5], [1, 2, 3, 6, 5]), 2))   # 2.0 - one mismatched sample
```

*Both are $O(nm)$. A Sakoe-Chiba band (limit how far the warp path strays from the diagonal) cuts DTW cost and prevents pathological alignments.*

---

# Reinforcement Learning

The through-line is the **Bellman equation**: the value of a state = immediate reward + discounted value of where you land next. When you *know* the environment's dynamics you can plan directly (value/policy iteration); when you don't, you learn from samples (Q-learning, policy gradients).

## Value Iteration & Policy Iteration (planning with a known model)

Given transition probabilities $P$ and rewards $R$, solve for the optimal value function. **Value iteration** repeatedly applies the Bellman optimality backup until $V$ converges; **policy iteration** alternates full policy *evaluation* with greedy *improvement* - fewer, heavier iterations.

$$  
V^*(s) = \max_a \Big[R(s,a) + \gamma \sum_{s'} P(s'\mid s,a)\,V^*(s')\Big]  
$$

```python
def value_iteration(P, R, gamma=0.9, tol=1e-6):
    # P: (S, A, S) transition probs, R: (S, A) rewards.
    S, A = R.shape; V = np.zeros(S)
    while True:
        Q = R + gamma * np.einsum('sap,p->sa', P, V)   # Bellman backup
        newV = Q.max(axis=1)
        if np.max(np.abs(newV - V)) < tol: break
        V = newV
    return V, Q.argmax(axis=1)                          # optimal values + greedy policy

def policy_iteration(P, R, gamma=0.9):
    S, A = R.shape; pi = np.zeros(S, int)
    while True:
        V = np.zeros(S)
        for _ in range(1000):                           # evaluate current policy
            V = np.array([R[s, pi[s]] + gamma * P[s, pi[s]] @ V for s in range(S)])
        Q = R + gamma * np.einsum('sap,p->sa', P, V)
        new_pi = Q.argmax(axis=1)                       # improve: act greedily w.r.t. V
        if np.array_equal(new_pi, pi): return V, pi
        pi = new_pi

S, A = 3, 2                                             # 3-state chain, action 1 = right
P = np.zeros((S, A, S))
for s in range(S):
    P[s, 1, min(s+1, S-1)] = 1.0; P[s, 0, max(s-1, 0)] = 1.0
R = np.zeros((S, A)); R[S-2, 1] = 1.0                  # reward for stepping into the goal
print(value_iteration(P, R)[1], policy_iteration(P, R)[1])   # [1 1 0] [1 1 0] - same policy
```

*Both converge to the same optimum (contraction mapping). Value iteration = many cheap sweeps; policy iteration = few expensive ones. Q-learning below is the model-free, sampled version of the same Bellman backup.*

---

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

---

## REINFORCE (policy gradient)

$$  
\nabla_\theta J = \mathbb{E}\Big[\sum_t \nabla_\theta \log\pi_\theta(a_t\mid s_t)G_t\Big], \qquad G_t = \sum_{k\ge t}\gamma^{k-t} r_k  
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

*Push up log-prob of actions that led to high return, down for low. Subtracting a baseline reduces variance without bias - the seed of A2C/PPO.*

---

## Actor-Critic & GAE (learned baseline)

REINFORCE's baseline can be a *learned* value function (the **critic**) instead of the batch mean. The advantage $A_t = G_t - V(s_t)$ says "how much better than expected was this action." **GAE** computes a low-variance advantage by exponentially averaging multi-step TD errors - the standard advantage estimator feeding PPO.

$$  
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \qquad \hat A_t^{GAE} = \sum_{l\ge 0}(\gamma\lambda)^l\,\delta_{t+l}  
$$

```python
def gae(rewards, values, gamma=0.99, lam=0.95):
    adv = np.zeros(len(rewards)); last = 0.0
    values = list(values) + [0.0]                       # bootstrap terminal value = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t+1] - values[t]   # TD error
        last = delta + gamma * lam * last               # exponentially-weighted accumulation
        adv[t] = last
    return adv

print(np.round(gae([1., 1., 1.], [0.5, 0.5, 0.5]), 3))
```

*$\lambda$ trades bias vs variance: $\lambda{=}0$ is one-step TD (biased, low variance), $\lambda{=}1$ is Monte-Carlo (unbiased, high variance). ~0.95 is the usual sweet spot.*

---

## PPO (Proximal Policy Optimization - the RLHF workhorse)

Policy gradients are unstable because one big step can wreck the policy. PPO takes the biggest safe step by **clipping** the probability ratio $\pi_\theta/\pi_{\text{old}}$: as long as the update stays within $[1-\epsilon, 1+\epsilon]$ it optimizes the advantage, but the clip removes any incentive to move further. Simple, robust, and the algorithm behind RLHF.

$$  
L^{CLIP} = \mathbb{E}\Big[\min\big(\rho_t \hat A_t,\; \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat A_t\big)\Big], \quad \rho_t = \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}  
$$

```python
def ppo_loss(logp_new, logp_old, adv, eps=0.2):
    ratio = np.exp(logp_new - logp_old)                 # importance ratio
    unclipped = ratio * adv
    clipped = np.clip(ratio, 1 - eps, 1 + eps) * adv    # remove incentive to step too far
    return -np.mean(np.minimum(unclipped, clipped))     # pessimistic (lower) bound

print(round(ppo_loss(np.array([-0.5, -0.7]), np.array([-0.6, -0.6]),
                     np.array([1.0, -1.0])), 4))
```

*The `min` makes it a lower bound: gains are capped by the clip, but losses aren't - so a bad action is always fully discouraged. PPO reuses each batch for several epochs (sample-efficient) and is the RL half of RLHF.*

---

# Interview Quick-Reference

**If they say "implement X", reach for:**

| Prompt                         | Pattern                                                                     |
| ------------------------------ | --------------------------------------------------------------------------- |
| "Softmax / numerically stable" | subtract max before exp                                                     |
| "Logistic regression"          | sigmoid + BCE gradient $\frac{1}{n}X^\top(\hat y - y)$                      |
| "Backprop a small net"         | $\partial L/\partial z = \text{softmax}-\text{onehot}$, chain rule up       |
| "Build autograd"               | record parents + local `_backward`, topo-sort, reverse                      |
| "Self-attention"               | $\text{softmax}(QK^\top/\sqrt{d_k})V$                                       |
| "Shrink the KV cache"          | GQA (share K/V per head group) > MQA (1 K/V head) > MLA (low-rank latent cache) |
| "Sample next token"            | top-p > top-k > temperature > greedy                                        |
| "Build a tokenizer"            | BPE merges (GPT) / WordPiece score (BERT) / Unigram+Viterbi (SentencePiece) |
| "Boosting vs bagging"          | GBDT fits residuals (bias↓) / RF averages bootstraps (var↓)                 |
| "Evaluate a classifier"        | confusion matrix → P/R/F1/AUC                                               |
| "Cluster, no labels"           | K-Means (hard) / GMM-EM (soft) / DBSCAN (density) / hierarchical (dendrogram) |
| "Reduce dimensions"            | PCA (variance) / LDA (class separation) / t-SNE-UMAP (viz)                  |
| "Detect anomalies"             | Isolation Forest (short random-tree path = outlier)                         |
| "Recommend items"              | matrix factorization $\hat r = p_i^\top q_j$                                |
| "Fast nearest neighbors"       | LSH (hyperplane hashing) / HNSW graph (vector DBs)                          |
| "Fine-tune cheaply"            | LoRA: freeze $W$, learn low-rank $\frac{\alpha}{r}AB$                       |
| "Align an LLM"                 | RLHF (reward model + PPO) or DPO (one preference loss, no RL)               |
| "Speed up LLM inference"       | KV cache → GQA/MLA → speculative decoding                                   |
| "Generate images"             | diffusion: predict the noise, denoise from $\mathcal{N}(0,I)$               |
| "Sample from a stream"         | reservoir sampling                                                          |
| "Explore vs exploit"           | ε-greedy → UCB → Thompson                                                   |
| "Plan with a known model"      | value / policy iteration (Bellman backup)                                   |
| "Policy gradient, stably"      | PPO (clip the ratio $\pi_\theta/\pi_{old}$) + GAE advantages                |
| "Fuzzy string / time match"    | edit distance (DP) / DTW (warped alignment)                                 |
| "Most likely sequence"         | Viterbi (DP over time × states)                                             |

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
- **Tokenizer choice:** BPE/WordPiece/Unigram are all subword compromises between word-level (huge vocab, OOV) and char-level (huge sequences); byte-level BPE removes `<unk>` entirely by tokenizing bytes.
- **Adam vs AdamW:** AdamW decouples weight decay from the adaptive step (L2-in-Adam is *not* true weight decay); default for Transformers.
- **LayerNorm vs RMSNorm:** RMSNorm drops mean-centering and bias - cheaper, same quality, hence LLaMA/Mistral/Gemma.
- **RLHF vs DPO:** RLHF = reward model + PPO with a KL leash (powerful, fragile, multi-model); DPO = one closed-form preference loss against a frozen reference (stable, no RL loop).
- **The four generative families:** VAE (encode→sample latent), GAN (minimax vs a discriminator), autoregressive (next-token, LLMs), diffusion (iterative denoising) - know one loss each.
- **Sinusoidal vs RoPE:** RoPE rotates Q/K so attention depends on *relative* position → better length extrapolation; the modern default.

> **Closing thought:** every block here is *one model + one loss + one gradient step*. If you can name those three for any algorithm, you can derive the code on the spot - which is exactly what the interview tests.
