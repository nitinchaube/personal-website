---
title: "PyTorch and TensorFlow Essentials"
date: 2026-08-26
summary: >-
  A side-by-side recall sheet for PyTorch and TensorFlow. Tensors, autograd,
  building models, the training loop, data pipelines, saving and loading, layers,
  reshaping, broadcasting, einsum, weight init, schedulers, gradient clipping and
  accumulation, transfer learning, custom layers and losses, mixed precision,
  distributed training, profiling, ONNX export, and the tricks that save hours.
  Written twice, one column each, with diagrams for the parts that are easier to
  see than to read.
tags: [PyTorch, TensorFlow, Keras, Autograd, Training, GPU, Deep Learning, Cheatsheet]
---

This is my recall sheet for the two frameworks I actually use. Everything is written twice, PyTorch on one side and TensorFlow (Keras) on the other, so I can look up the same idea in whichever one I am in that day. The goal is not to teach deep learning, it is to remember the exact call I need when my hands are already on the keyboard, plus the handful of tricks that separate code that runs from code that runs fast and correct.

One mental model before anything else. PyTorch is imperative and eager, so your code runs line by line like normal Python and you write your own training loop. TensorFlow through Keras is more declarative, you describe a model and call `fit`, and it compiles a graph under the hood for speed. Both can do the other style. Once you see that, most of the differences below are just spelling.

```
        PYTORCH                              TENSORFLOW / KERAS
  ----------------------              ----------------------------
  eager by default                    graph by default (inside fit / tf.function)
  you write the loop                  fit() writes the loop for you
  tensors on CPU/GPU you move         placement mostly automatic
  dim= , channels first (N,C,H,W)     axis= , channels last (N,H,W,C)
  torch.compile() for speed           @tf.function for speed
```

---

# TL;DR: a whole model, start to finish

If you only read one thing, read this. Both columns do the exact same job in the exact same order: build data, build model, train, run inference, save, reload. Everything else in this note is detail hanging off these six steps.

```
   THE LIFECYCLE (same for both)

   data  ->  model  ->  train loop  ->  inference  ->  save  ->  load
```

```python
# ============================ PyTorch ============================
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

dev = "cuda" if torch.cuda.is_available() else "cpu"

# 1. DATA
X, y = torch.randn(1000, 20), torch.randint(0, 3, (1000,))
loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

# 2. MODEL
model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 3)).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# 3. TRAIN
model.train()
for epoch in range(10):
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad()                 # clear old grads
        loss = loss_fn(model(xb), yb)   # forward + loss
        loss.backward()                 # backprop
        opt.step()                      # update weights

# 4. INFERENCE
model.eval()
with torch.no_grad():
    preds = model(X.to(dev)).argmax(1)

# 5. SAVE   and   6. LOAD
torch.save(model.state_dict(), "model.pt")
model.load_state_dict(torch.load("model.pt", map_location=dev))
```

```python
# ========================== TensorFlow ==========================
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. DATA
X = tf.random.normal((1000, 20))
y = tf.random.uniform((1000,), 0, 3, tf.int32)

# 2. MODEL
model = keras.Sequential([
    layers.Dense(64, activation="relu", input_shape=(20,)),
    layers.Dense(3),
])
model.compile(optimizer=keras.optimizers.AdamW(1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

# 3. TRAIN  (fit hides the loop above)
model.fit(X, y, batch_size=64, epochs=10, verbose=2)

# 4. INFERENCE
preds = model.predict(X).argmax(1)

# 5. SAVE   and   6. LOAD
model.save("model.keras")
model = keras.models.load_model("model.keras")
```

Read the PyTorch training block top to bottom and you see the machinery. Read the Keras one and `fit` swallows it. That single trade, explicit loop versus `fit`, is the whole personality difference between the two.

---

# Install and sanity check

```python
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
print(torch.__version__, torch.cuda.is_available())

# TensorFlow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(tf.__version__, tf.config.list_physical_devices('GPU'))
```

---

# Tensors: the basics

A tensor is an n-dimensional array that can live on a GPU and remember how it was computed. Same idea in both frameworks.

| Task               | PyTorch                                | TensorFlow                               |
| ------------------ | -------------------------------------- | ---------------------------------------- |
| From list          | `torch.tensor([1,2,3])`                | `tf.constant([1,2,3])`                   |
| Zeros / ones       | `torch.zeros(2,3)` / `torch.ones(2,3)` | `tf.zeros([2,3])` / `tf.ones([2,3])`     |
| Range              | `torch.arange(0,10,2)`                 | `tf.range(0,10,2)`                       |
| Linspace           | `torch.linspace(0,1,5)`                | `tf.linspace(0.,1.,5)`                   |
| Random normal      | `torch.randn(2,3)`                     | `tf.random.normal([2,3])`                |
| Random uniform     | `torch.rand(2,3)`                      | `tf.random.uniform([2,3])`               |
| Random ints        | `torch.randint(0,10,(2,3))`            | `tf.random.uniform([2,3],0,10,tf.int32)` |
| Identity           | `torch.eye(3)`                         | `tf.eye(3)`                              |
| Fill               | `torch.full((2,3), 7)`                 | `tf.fill([2,3], 7)`                      |
| Like another       | `torch.zeros_like(x)`                  | `tf.zeros_like(x)`                       |
| Dtype              | `x.dtype`, `x.float()`, `x.long()`     | `x.dtype`, `tf.cast(x, tf.float32)`      |
| Shape              | `x.shape` or `x.size()`                | `x.shape` or `tf.shape(x)`               |
| Number of elements | `x.numel()`                            | `tf.size(x)`                             |
| To Python scalar   | `x.item()`                             | `x.numpy()` or `float(x)`                |
| To NumPy           | `x.numpy()` (cpu first)                | `x.numpy()`                              |
| From NumPy         | `torch.from_numpy(a)`                  | `tf.convert_to_tensor(a)`                |

The one difference that bites people: a PyTorch tensor is mutable, you can assign into it in place. A plain TF tensor is immutable, you build a new one. TF's mutable thing is `tf.Variable`, which is what layer weights are.

```python
# PyTorch: in-place is normal
x = torch.zeros(3)
x[0] = 5                 # fine

# TensorFlow: use a Variable for mutation
v = tf.Variable(tf.zeros(3))
v[0].assign(5)           # fine
v.assign_add(tf.ones(3))
```

## Dtypes worth knowing

| Meaning                | PyTorch                | TensorFlow    |
| ---------------------- | ---------------------- | ------------- |
| 32-bit float (default) | `torch.float32`        | `tf.float32`  |
| 16-bit half            | `torch.float16`        | `tf.float16`  |
| bfloat16               | `torch.bfloat16`       | `tf.bfloat16` |
| 64-bit int             | `torch.int64` (`long`) | `tf.int64`    |
| bool                   | `torch.bool`           | `tf.bool`     |

Trick: label tensors for classification must be `long` (int64) in PyTorch cross entropy. Floats there give a cryptic error. In Keras use `SparseCategoricalCrossentropy` for integer labels and it just works.

---

# Devices and GPU

| Task        | PyTorch                                                | TensorFlow                                        |
| ----------- | ------------------------------------------------------ | ------------------------------------------------- |
| Pick device | `dev = "cuda" if torch.cuda.is_available() else "cpu"` | placement is automatic                            |
| Move tensor | `x = x.to(dev)`                                        | `with tf.device('/GPU:0'): ...`                   |
| Move model  | `model.to(dev)`                                        | handled by strategy / automatic                   |
| Count GPUs  | `torch.cuda.device_count()`                            | `len(tf.config.list_physical_devices('GPU'))`     |
| Memory used | `torch.cuda.memory_allocated()`                        | `tf.config.experimental.get_memory_info('GPU:0')` |
| Empty cache | `torch.cuda.empty_cache()`                             | not applicable                                    |

PyTorch makes you move things yourself, and forgetting to move either the model or the batch is the single most common beginner crash ("expected all tensors on the same device"). TensorFlow hides it, which is convenient until you need control.

Trick: set the device once and thread it through everything. Do not sprinkle `.cuda()` around, use `.to(device)` so the same code runs on CPU when there is no GPU.

Trick: TF grabs all GPU memory on the first op by default. On a shared machine turn on memory growth right after import, before any tensor exists.

```python
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
```

---

# Indexing, slicing, masking

Both index like NumPy. The useful extras are boolean masks and gather. Every example below uses this one small tensor so the outputs are easy to picture:

```
  x = [[ 1,  2,  3],
       [ 4, -5,  6],
       [-7,  8,  9]]      shape (3, 3)
```

| Task            | Call (PyTorch / TF)                             | Result                              |
| --------------- | ----------------------------------------------- | ----------------------------------- |
| Single element  | `x[1, 2]`                                       | `6`                                 |
| Row             | `x[0]`                                          | `[1, 2, 3]`                         |
| Column          | `x[:, 1]`                                       | `[2, -5, 8]`                        |
| Sub-block       | `x[0:2, 1:3]`                                   | `[[2, 3], [-5, 6]]`                 |
| Every other row | `x[::2]`                                        | `[[1,2,3], [-7,8,9]]`               |
| Last row        | `x[-1]`                                         | `[-7, 8, 9]`                        |
| Boolean mask    | `x[x > 0]` / `tf.boolean_mask(x, x>0)`          | `[1, 2, 3, 4, 6, 8, 9]` (flattened) |
| Masked fill     | `x.masked_fill(x<0, 0)` / `tf.where(x<0, 0, x)` | negatives become `0`                |
| Gather rows     | `x[[0, 2]]` / `tf.gather(x, [0,2])`             | `[[1,2,3], [-7,8,9]]`               |
| Top-k per row   | `x.topk(2).values` / `tf.math.top_k(x,2)`       | `[[3,2], [6,4], [9,8]]`             |
| Argmax per row  | `x.argmax(1)` / `tf.argmax(x,1)`                | `[2, 2, 2]`                         |
| Nonzero coords  | `x.nonzero()` / `tf.where(x!=0)`                | list of `(row, col)` pairs          |
| One-hot         | `F.one_hot(torch.tensor([0,2]),3)`              | `[[1,0,0], [0,0,1]]`                |

Trick: use masks to avoid loops. To zero out padded positions in a batch of sequences, build a `(batch, seq)` boolean mask once and apply it, rather than looping over sequences.

---

# Reshaping and moving axes

This is where I forget syntax most, so here it is in one place. The shape column is the whole point: reshaping is bookkeeping on shapes, so track the shape and the rest follows. Start from `x = torch.arange(6)`, a flat vector of `[0,1,2,3,4,5]` with shape `(6,)`.

| Task                | PyTorch                           | TensorFlow                 | Shape / result                           |
| ------------------- | --------------------------------- | -------------------------- | ---------------------------------------- |
| Reshape             | `x.view(2,3)` or `x.reshape(2,3)` | `tf.reshape(x, [2,3])`     | `(6,) -> (2,3)`: `[[0,1,2],[3,4,5]]`     |
| Flatten             | `x.flatten()` or `x.view(-1)`     | `tf.reshape(x, [-1])`      | `(2,3) -> (6,)` back to a vector         |
| Infer a dim with -1 | `x.reshape(3, -1)`                | `tf.reshape(x, [3,-1])`    | `(6,) -> (3,2)`, the 2 is inferred       |
| Add axis (front)    | `x.unsqueeze(0)` or `x[None]`     | `tf.expand_dims(x, 0)`     | `(6,) -> (1,6)`                          |
| Add axis (back)     | `x.unsqueeze(-1)`                 | `tf.expand_dims(x, -1)`    | `(6,) -> (6,1)`                          |
| Remove size-1 axis  | `x.squeeze()`                     | `tf.squeeze(x)`            | `(1,6,1) -> (6,)`                        |
| Transpose 2D        | `m.t()` or `m.T`                  | `tf.transpose(m)`          | `(2,3) -> (3,2)`, rows become cols       |
| Permute axes        | `t.permute(0,2,1)`                | `tf.transpose(t, [0,2,1])` | `(N,H,W) -> (N,W,H)`                     |
| Concatenate         | `torch.cat([a,b], dim=0)`         | `tf.concat([a,b], axis=0)` | two `(2,3)` -> `(4,3)`, stacks rows      |
| Stack (new axis)    | `torch.stack([a,b], dim=0)`       | `tf.stack([a,b], axis=0)`  | two `(2,3)` -> `(2,2,3)`, new front axis |
| Split               | `torch.chunk(m, 2, dim=0)`        | `tf.split(m, 2, axis=0)`   | `(4,3) -> two (2,3)` pieces              |
| Repeat / tile       | `v.repeat(2)`                     | `tf.tile(v, [2])`          | `[0,1] -> [0,1,0,1]`                     |

The difference to keep straight: `cat`/`concat` join along an axis that already exists (shapes stay the same rank), while `stack` adds a brand new axis (rank goes up by one).

The `view` versus `reshape` distinction in PyTorch is worth understanding, because it maps to how memory works.

```
  A tensor is a shape + strides pointing into one flat buffer.

    x = torch.arange(6).reshape(2,3)      buffer: [0 1 2 3 4 5]
                                          shape (2,3), strides (3,1)

    x.view(3,2)   -> reuses SAME buffer, just new shape/strides   (free)
    x.t()         -> shape (3,2), strides (1,3)  now NON-contiguous
    x.t().view(6) -> ERROR: view needs contiguous memory
    x.t().reshape(6) -> copies into a fresh contiguous buffer     (works)
```

Rule of thumb: `view` is free but needs contiguous memory and can fail after a transpose or permute. `reshape` always works because it copies when it has to. If `view` errors with "not contiguous", call `x.contiguous().view(...)` or just use `reshape`.

Trick: `-1` means "infer this dimension." `x.reshape(batch, -1)` flattens everything after the batch axis, the classic move before a linear layer.

---

# Broadcasting

Both follow NumPy rules. Align shapes from the right, a dimension of size 1 stretches to match, a missing dimension counts as 1.

```
        (4, 1, 8)
           (5, 8)
       -----------
    ->  (4, 5, 8)      the 1 stretches to 5, the missing axis becomes 4
```

Trick: broadcasting is how you kill loops. To add a per-channel bias to a `(N, C, H, W)` image, reshape the bias to `(1, C, 1, 1)` and add. No tiling, stays fast on the GPU.

Trick to catch silent bugs: broadcasting will happily combine `(N, 1)` with `(1, N)` into an `(N, N)` matrix when you meant elementwise. If a tensor suddenly got huge, a stray broadcast is usually why. Print shapes.

---

# Math: elementwise and reductions

| Task            | PyTorch                        | TensorFlow                              |
| --------------- | ------------------------------ | --------------------------------------- |
| Add / mul       | `a + b`, `a * b`               | `a + b`, `a * b`                        |
| Matrix multiply | `a @ b` or `torch.matmul`      | `a @ b` or `tf.matmul`                  |
| Sum             | `x.sum(dim=1)`                 | `tf.reduce_sum(x, axis=1)`              |
| Mean            | `x.mean(dim=0)`                | `tf.reduce_mean(x, axis=0)`             |
| Max / argmax    | `x.max(dim=1)`, `x.argmax(1)`  | `tf.reduce_max`, `tf.argmax`            |
| Std / var       | `x.std()`, `x.var()`           | `tf.math.reduce_std`, `reduce_variance` |
| Exp / log       | `torch.exp(x)`, `torch.log(x)` | `tf.exp(x)`, `tf.math.log(x)`           |
| Clamp / clip    | `x.clamp(0, 1)`                | `tf.clip_by_value(x, 0, 1)`             |
| Norm            | `x.norm(dim=1)`                | `tf.norm(x, axis=1)`                    |
| Softmax         | `F.softmax(x, dim=-1)`         | `tf.nn.softmax(x, axis=-1)`             |
| Where           | `torch.where(cond, a, b)`      | `tf.where(cond, a, b)`                  |

Naming split to memorize: PyTorch reductions take `dim`, TensorFlow takes `axis` and prefixes them with `reduce_`. Once that clicks the rest is obvious.

Trick: `keepdim=True` (PyTorch) / `keepdims=True` (TF) keeps the reduced axis as size 1 so the result still broadcasts back against the original. Essential for normalization: `x - x.mean(dim=-1, keepdim=True)`.

---

# einsum, the one op to remember

If you learn a single power tool, learn `einsum`. It is identical in both frameworks and replaces most matmul, transpose, and sum-of-products gymnastics with one readable string.

```python
torch.einsum('ij,jk->ik', A, B)        # matrix multiply
torch.einsum('bij,bjk->bik', A, B)     # batched matmul
torch.einsum('bhqd,bhkd->bhqk', Q, K)  # attention scores over heads
torch.einsum('ii->i', A)               # diagonal
torch.einsum('ij->ji', A)              # transpose
torch.einsum('ij->', A)                # sum everything
tf.einsum('ij,jk->ik', A, B)           # identical in TF
```

Read the string as "these input axes, produce these output axes, sum over anything that disappears." Any letter on the left but not on the right gets summed out. Same rule in NumPy, PyTorch, and TensorFlow, which is why it is worth internalizing once.

---

# Autograd: how gradients happen

This is the real conceptual difference, so it gets room and a picture.

PyTorch records operations on tensors with `requires_grad=True` as you run them, building a graph. `.backward()` walks that graph in reverse and fills each parameter's `.grad`. TensorFlow does not record by default, you open a `tf.GradientTape()` block to say "watch these ops," then ask the tape for gradients.

```
  Forward builds the graph (PyTorch) or the tape (TF):

     x ──▶ [ *w ] ──▶ h ──▶ [ +b ] ──▶ [ relu ] ──▶ y ──▶ loss
                                                             │
  Backward walks it in reverse, chain rule at each node:     ▼
     dL/dx ◀── dL/dh ◀────────────────────────────────── dL/dloss = 1

  Each parameter node accumulates dL/dparam into .grad,
  the optimizer then steps every parameter against its grad.
```

```python
# PyTorch
x = torch.tensor(3.0, requires_grad=True)
y = x**2 + 2*x
y.backward()
print(x.grad)          # dy/dx = 2x + 2 = 8

# TensorFlow
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x**2 + 2*x
print(tape.gradient(y, x))   # 8
```

Facts I keep needing:

- In PyTorch, gradients accumulate. Call `optimizer.zero_grad()` each step or old gradients pile on. Forgetting this is a classic silent bug.
- A `tf.GradientTape` is single-use by default. To call `tape.gradient` more than once make it `persistent=True`, then `del tape` when done.
- The tape watches `tf.Variable`s automatically. For a plain tensor call `tape.watch(x)` inside the block.
- Turn gradients off for inference: PyTorch `with torch.no_grad():`, TF just do not open a tape. Saves memory and time.
- Detach from the graph: `x.detach()` in PyTorch, `tf.stop_gradient(x)` in TF. Useful for targets you do not want gradients flowing through (for example in a target network).

```python
model.eval()
with torch.no_grad():
    preds = model(x)     # no graph built, faster, less memory
```

Trick: `model.eval()` and `model.train()` are not about gradients, they flip the behavior of dropout and batchnorm. Correct fast inference in PyTorch needs both `model.eval()` and `torch.no_grad()`. Keras handles this through the `training=True/False` flag it passes to layers inside `fit` and `predict`.

## `no_grad` vs `inference_mode`

`torch.inference_mode()` (PyTorch 1.9+) is a stronger, faster `no_grad`. Both switch off gradient tracking, but `inference_mode` also drops the version-counting and view bookkeeping that autograd normally keeps on every tensor, so ops and allocations are a little cheaper.

```python
model.eval()
with torch.inference_mode():      # prefer this for pure inference loops
    preds = model(x)
```

The catch: tensors created inside an `inference_mode` block are marked as "inference tensors" and error out if you later feed them into anything differentiable (re-attach them to a training graph, call `requires_grad_(True)`, and so on). `no_grad` has no such restriction, its outputs are ordinary tensors.

|                                     | `torch.no_grad()` | `torch.inference_mode()` |
| ----------------------------------- | ----------------- | ------------------------ |
| Disables gradient tracking          | yes               | yes                      |
| Disables version / view bookkeeping | no                | yes (faster)             |
| Output reusable in later autograd   | yes               | no (raises)              |
| Since                               | always            | 1.9+                     |

Rule of thumb: pure eval or serving loops where the output never re-enters training, use `inference_mode`. If the result might flow back into a differentiable computation (some RL targets, mixed train/eval code), stay on `no_grad`. TensorFlow needs neither, not opening a `GradientTape` already gives you the same effect.

## Worked example: a custom two-parameter fit, no `nn` at all

Sometimes the clearest way to see autograd is to hand-roll a tiny regression.

```python
# PyTorch: fit y = w*x + b by hand
x = torch.linspace(0, 1, 100)
y = 3*x + 2 + 0.1*torch.randn(100)
w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

for step in range(500):
    pred = w*x + b
    loss = ((pred - y)**2).mean()
    loss.backward()                # fills w.grad, b.grad
    with torch.no_grad():          # update outside the graph
        w -= 0.1 * w.grad
        b -= 0.1 * b.grad
        w.grad.zero_(); b.grad.zero_()
print(w.item(), b.item())          # close to 3, 2
```

That loop is the whole of gradient descent with nothing hidden. Everything later (optimizers, modules) is convenience on top of exactly this.

---

# Building a model

## Quick way: a stack of layers

```python
# PyTorch
model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(256, 10),
)

# TensorFlow
model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(10),
])
```

## Flexible way: subclass

Use this once the forward pass is not a straight line (skip connections, multiple inputs, custom logic).

```python
# PyTorch
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
model = Net()

# TensorFlow
class Net(keras.Model):
    def __init__(self):
        super().__init__()
        self.fc1 = layers.Dense(256, activation='relu')
        self.fc2 = layers.Dense(10)
    def call(self, x):
        return self.fc2(self.fc1(x))
model = Net()
```

Two things to remember. In PyTorch the method is `forward` and you call the module like a function (`model(x)`), never `.forward` directly because that skips hooks. In TensorFlow the method is `call`. In PyTorch you declare both input and output sizes for `Linear`. In Keras you usually give only the output units and it infers the input the first time it sees data.

## Keras functional API

TensorFlow has a third style with no direct PyTorch equal, genuinely nice for branching models.

```python
inputs = keras.Input(shape=(784,))
x = layers.Dense(256, activation='relu')(inputs)
outputs = layers.Dense(10)(x)
model = keras.Model(inputs, outputs)
model.summary()          # prints the whole architecture and param counts
```

Trick: `model.summary()` in Keras and `print(model)` in PyTorch both dump the architecture. For a proper parameter count and per-layer shapes in PyTorch, `torchinfo.summary(model, input_size=(1, 784))` is the closest match to Keras `summary`.

---

# The layers you reach for

| Layer               | PyTorch                           | TensorFlow / Keras                |
| ------------------- | --------------------------------- | --------------------------------- |
| Fully connected     | `nn.Linear(in, out)`              | `layers.Dense(out)`               |
| Conv 2D             | `nn.Conv2d(in, out, k)`           | `layers.Conv2D(out, k)`           |
| Conv transpose      | `nn.ConvTranspose2d(...)`         | `layers.Conv2DTranspose(...)`     |
| Max / avg pool      | `nn.MaxPool2d(2)`                 | `layers.MaxPooling2D(2)`          |
| Global avg pool     | `nn.AdaptiveAvgPool2d(1)`         | `layers.GlobalAveragePooling2D()` |
| Batch norm          | `nn.BatchNorm2d(c)`               | `layers.BatchNormalization()`     |
| Layer norm          | `nn.LayerNorm(dim)`               | `layers.LayerNormalization()`     |
| Dropout             | `nn.Dropout(p)`                   | `layers.Dropout(p)`               |
| Embedding           | `nn.Embedding(n, d)`              | `layers.Embedding(n, d)`          |
| LSTM / GRU          | `nn.LSTM(in, hid)`                | `layers.LSTM(hid)`                |
| Multihead attention | `nn.MultiheadAttention(d, h)`     | `layers.MultiHeadAttention(h, d)` |
| Transformer encoder | `nn.TransformerEncoderLayer(...)` | build from `MultiHeadAttention`   |
| Flatten             | `nn.Flatten()`                    | `layers.Flatten()`                |

Channel-order gotcha, drawn out because it burns everyone once:

```
  Same 32x32 RGB image, batch of 16:

  PyTorch conv expects:   (N, C, H, W) = (16,  3, 32, 32)   channels FIRST
  Keras conv expects:     (N, H, W, C) = (16, 32, 32,  3)   channels LAST
```

If you port a model and the shapes look transposed, this is why. Keras can switch with `data_format='channels_first'`, but the default trips people.

## Conv shape math (memorize the formula, not the numbers)

$$
\text{out} = \left\lfloor \frac{\text{in} + 2p - k}{s} \right\rfloor + 1
$$

with kernel $k$, padding $p$, stride $s$. "Same" padding keeps the spatial size when $s=1$. Trick: to halve spatial size, use stride 2. To keep it, use `padding = k // 2` with stride 1.

---

# Activations and losses

| Piece                      | PyTorch                  | TensorFlow                                        |
| -------------------------- | ------------------------ | ------------------------------------------------- |
| ReLU                       | `F.relu` / `nn.ReLU()`   | `tf.nn.relu` / `'relu'`                           |
| LeakyReLU                  | `F.leaky_relu`           | `layers.LeakyReLU()`                              |
| GELU                       | `F.gelu`                 | `tf.nn.gelu` / `'gelu'`                           |
| SiLU / Swish               | `F.silu`                 | `tf.nn.silu`                                      |
| Sigmoid                    | `torch.sigmoid`          | `tf.sigmoid`                                      |
| Tanh                       | `torch.tanh`             | `tf.tanh`                                         |
| Softmax                    | `F.softmax(x, dim=-1)`   | `tf.nn.softmax`                                   |
| MSE                        | `nn.MSELoss()`           | `keras.losses.MeanSquaredError()`                 |
| Cross entropy (int labels) | `nn.CrossEntropyLoss()`  | `SparseCategoricalCrossentropy(from_logits=True)` |
| Cross entropy (one-hot)    | (one-hot then NLL)       | `CategoricalCrossentropy(from_logits=True)`       |
| Binary CE                  | `nn.BCEWithLogitsLoss()` | `BinaryCrossentropy(from_logits=True)`            |

The single most important loss detail: logits versus probabilities.

- PyTorch `nn.CrossEntropyLoss` expects raw logits and integer class labels. It applies log-softmax internally. Do not put a softmax before it, that double-applies and hurts training.
- PyTorch `nn.BCEWithLogitsLoss` expects raw logits and applies sigmoid internally, which is numerically safer than `BCELoss` after a manual sigmoid. Prefer the "WithLogits" version.
- Keras losses take `from_logits=True` to do the same. No activation on the last layer means `from_logits=True`. A softmax or sigmoid on the last layer means leave it False.

Getting this wrong does not crash, it just trains badly or goes NaN, which is why it is worth burning in.

---

# Optimizers

| Optimizer      | PyTorch                            | TensorFlow                |
| -------------- | ---------------------------------- | ------------------------- |
| SGD + momentum | `optim.SGD(p, lr, momentum=0.9)`   | `SGD(lr, momentum=0.9)`   |
| Adam           | `optim.Adam(p, lr=1e-3)`           | `Adam(1e-3)`              |
| AdamW          | `optim.AdamW(p, lr, weight_decay)` | `AdamW(lr, weight_decay)` |
| RMSprop        | `optim.RMSprop(p, lr)`             | `RMSprop(lr)`             |

In PyTorch you hand the optimizer the parameters explicitly, `optim.Adam(model.parameters(), lr=1e-3)`. In Keras `compile` wires it up. AdamW is usually the right default for transformers because it decouples weight decay from the gradient update.

Trick: parameter groups let you use different learning rates per part of the model. Very common for fine-tuning, where the backbone gets a tiny rate and the new head gets a larger one.

```python
optimizer = torch.optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': 1e-5},
    {'params': model.head.parameters(),     'lr': 1e-3},
])
```

---

# Learning-rate schedules

| Schedule          | PyTorch                         | TensorFlow                        |
| ----------------- | ------------------------------- | --------------------------------- |
| Step decay        | `StepLR(opt, step_size, gamma)` | `schedules.ExponentialDecay(...)` |
| Cosine            | `CosineAnnealingLR(opt, T_max)` | `schedules.CosineDecay(...)`      |
| Reduce on plateau | `ReduceLROnPlateau(opt)`        | `callbacks.ReduceLROnPlateau()`   |
| Warmup + cosine   | `OneCycleLR(opt, max_lr, ...)`  | build a custom schedule           |

```python
# PyTorch: step the scheduler once per epoch (or per batch for OneCycle)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
for epoch in range(epochs):
    train_one_epoch(...)
    scheduler.step()
```

Trick: most schedulers step once per epoch, but `OneCycleLR` steps once per batch. Putting `scheduler.step()` in the wrong place silently ruins the schedule. Check the docstring for which one you have.

---

# Weight initialization

Keras initializes sensibly by default (Glorot uniform for Dense). PyTorch also has reasonable defaults, but people override them for specific architectures.

```python
# PyTorch: apply an init function across the whole model
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.zeros_(m.bias)
model.apply(init_weights)

# TensorFlow: pass an initializer to the layer
layers.Dense(256, activation='relu',
             kernel_initializer='he_normal',
             bias_initializer='zeros')
```

Rule of thumb: Kaiming (He) init for ReLU-family activations, Xavier (Glorot) for tanh and sigmoid. This keeps the variance of activations stable through depth so signals do not explode or vanish early.

---

# The training loop

The heart of it. PyTorch makes you write the loop, TensorFlow lets you write it or hand it off.

```
  ONE TRAINING STEP, both frameworks, same five things:

    ┌─────────────┐   ┌──────────┐   ┌─────────┐   ┌───────────┐   ┌──────────┐
    │ zero grads  │──▶│ forward  │──▶│  loss   │──▶│ backward  │──▶│  step    │
    │ (clear old) │   │ preds=   │   │ compare │   │ dL/dparam │   │ update W │
    └─────────────┘   │ model(x) │   │ to y    │   └───────────┘   └──────────┘
                      └──────────┘   └─────────┘
    repeat over every batch, then over every epoch
```

## PyTorch, the loop you write by hand

```python
model.train()
for epoch in range(epochs):
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        optimizer.zero_grad()          # 1. clear old grads
        preds = model(xb)              # 2. forward
        loss = loss_fn(preds, yb)      # 3. compute loss
        loss.backward()                # 4. backprop
        optimizer.step()               # 5. update weights
```

Memorize the order: zero, forward, loss, backward, step. Every PyTorch loop is a variation on it. The common mistakes are dropping `zero_grad`, or forgetting `.to(dev)` on the batch.

## TensorFlow, the easy way

```python
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(train_ds, validation_data=val_ds, epochs=epochs)
```

## TensorFlow, the custom loop when you need control

```python
@tf.function                          # compiles to a graph, big speedup
def train_step(xb, yb):
    with tf.GradientTape() as tape:
        preds = model(xb, training=True)
        loss = loss_fn(yb, preds)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

for epoch in range(epochs):
    for xb, yb in train_ds:
        train_step(xb, yb)
```

`@tf.function` traces your Python into a static graph on first call, then reuses it, often a large speedup. The catch is that Python side effects (prints, list appends) only run during tracing. Use `tf.print` to print inside a compiled function.

The PyTorch equivalent speedup is `torch.compile(model)` (PyTorch 2.0+), which does a similar trace and optimize in one line.

```python
model = torch.compile(model)          # PyTorch 2.x, optional, usually just works
```

---

# A full worked example, end to end

Same tiny classifier both ways, so the whole shape of a project is on one screen.

```python
# ---------- PyTorch ----------
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

dev = "cuda" if torch.cuda.is_available() else "cpu"
X = torch.randn(1000, 20); y = torch.randint(0, 3, (1000,))
loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 3)).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(10):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
    print(epoch, loss.item())

model.eval()
with torch.no_grad():
    acc = (model(X.to(dev)).argmax(1).cpu() == y).float().mean()
print("acc", acc.item())
```

```python
# ---------- TensorFlow ----------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

X = tf.random.normal((1000, 20))
y = tf.random.uniform((1000,), 0, 3, tf.int32)

model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dense(3),
])
model.compile(optimizer=keras.optimizers.AdamW(1e-3),
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(X, y, batch_size=64, epochs=10, verbose=2)
print(model.evaluate(X, y))
```

Same model, same optimizer, same loss. PyTorch spells out the loop, Keras hides it in `fit`. That is the whole personality difference in one comparison.

---

# Data pipelines

| Task             | PyTorch                                       | TensorFlow                                   |
| ---------------- | --------------------------------------------- | -------------------------------------------- |
| Wrap arrays      | `TensorDataset(X, y)`                         | `tf.data.Dataset.from_tensor_slices((X, y))` |
| Custom dataset   | subclass `Dataset` (`__len__`, `__getitem__`) | generator + `from_generator`                 |
| Batch + shuffle  | `DataLoader(ds, batch_size=32, shuffle=True)` | `ds.shuffle(1000).batch(32)`                 |
| Parallel loading | `DataLoader(..., num_workers=4)`              | `ds.map(fn, num_parallel_calls=AUTOTUNE)`    |
| Prefetch         | `DataLoader(..., pin_memory=True)`            | `ds.prefetch(AUTOTUNE)`                      |
| Repeat           | loop over epochs                              | `ds.repeat()`                                |

```python
# PyTorch custom dataset
from torch.utils.data import Dataset, DataLoader
class MyData(Dataset):
    def __init__(self, X, y): self.X, self.y = X, y
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]

loader = DataLoader(MyData(X, y), batch_size=32, shuffle=True,
                    num_workers=4, pin_memory=True)

# TensorFlow pipeline
AUTOTUNE = tf.data.AUTOTUNE
ds = (tf.data.Dataset.from_tensor_slices((X, y))
        .shuffle(1000).batch(32).prefetch(AUTOTUNE))
```

Why prefetch matters, drawn out:

```
  Without overlap:   [load b1][train b1][load b2][train b2]   GPU idle half the time
  With prefetch:     [load b1][load b2 ][load b3 ] ...
                            [train b1][train b2][train b3]     GPU never waits
```

Trick: always `prefetch` in tf.data and set `num_workers` plus `pin_memory=True` in PyTorch. Data loading on the CPU should overlap with GPU compute, otherwise the expensive GPU sits idle waiting for the next batch. Cheapest speedup most people never turn on.

Trick: `tf.data.AUTOTUNE` lets TF tune the worker count and prefetch buffer for you. Use it instead of hardcoding numbers.

---

# Saving and loading

| Task              | PyTorch                                     | TensorFlow                           |
| ----------------- | ------------------------------------------- | ------------------------------------ |
| Save weights      | `torch.save(model.state_dict(), 'm.pt')`    | `model.save_weights('m.weights.h5')` |
| Load weights      | `model.load_state_dict(torch.load('m.pt'))` | `model.load_weights('m.weights.h5')` |
| Save whole model  | `torch.save(model, 'm.pt')`                 | `model.save('m.keras')`              |
| Load whole model  | `torch.load('m.pt')`                        | `keras.models.load_model('m.keras')` |
| Resume checkpoint | dict of model + optim + epoch               | `tf.train.Checkpoint(...)`           |

The standard PyTorch advice: save the `state_dict`, not the whole model object. Saving the whole object pickles your class definition and breaks after a refactor. The `state_dict` is a plain dictionary of tensors, portable and safe.

```python
# PyTorch resume-training checkpoint
torch.save({'epoch': epoch,
            'model': model.state_dict(),
            'optim': optimizer.state_dict()}, 'ckpt.pt')

ckpt = torch.load('ckpt.pt', map_location='cpu')
model.load_state_dict(ckpt['model'])
optimizer.load_state_dict(ckpt['optim'])
```

Trick: load on a CPU-only machine with `map_location='cpu'` or PyTorch tries to place tensors on a CUDA device that is not there and crashes.

Trick: `strict=False` in `load_state_dict` lets you load a checkpoint whose keys do not perfectly match, useful when you changed the head but kept the backbone.

---

# Transfer learning and freezing

Freezing means "do not update these weights," done by turning off their gradients.

```python
# PyTorch: freeze the backbone, train a new head
for p in model.backbone.parameters():
    p.requires_grad = False
model.head = nn.Linear(512, num_classes)   # fresh, trainable
opt = torch.optim.Adam(model.head.parameters(), lr=1e-3)

# TensorFlow
base = keras.applications.ResNet50(include_top=False, weights='imagenet')
base.trainable = False
model = keras.Sequential([base,
                          layers.GlobalAveragePooling2D(),
                          layers.Dense(num_classes)])
```

Trick: the two-stage recipe that works. First freeze the backbone and train only the head until it stabilizes. Then unfreeze the top few backbone blocks and continue at a much smaller learning rate. Unfreezing everything from step one usually wrecks the pretrained features.

---

# Gradient clipping and accumulation

Two tricks that come up constantly in real training.

Gradient clipping caps the gradient norm so a single huge batch cannot blow up the weights. Standard in RNNs and transformers.

```python
# PyTorch: clip after backward, before step
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()

# TensorFlow: clip in the optimizer, or clip grads before apply
optimizer = keras.optimizers.Adam(1e-3, clipnorm=1.0)
```

Gradient accumulation simulates a big batch when it does not fit in memory. Run several small batches, sum their gradients, then step once.

```python
# PyTorch: effective batch = batch_size * accum_steps
accum = 4
optimizer.zero_grad()
for i, (xb, yb) in enumerate(loader):
    loss = loss_fn(model(xb), yb) / accum   # scale so the mean is right
    loss.backward()                          # grads accumulate
    if (i + 1) % accum == 0:
        optimizer.step()
        optimizer.zero_grad()
```

The accumulation trick leans on the exact PyTorch behavior people usually fight: gradients add up across `backward` calls. Here that is the feature.

---

# Custom layers and custom losses

```python
# PyTorch custom layer: it is just a Module
class Scale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(dim))   # a learnable weight
    def forward(self, x):
        return x * self.g

# PyTorch custom loss: any function returning a scalar tensor works
def dice_loss(pred, target, eps=1e-6):
    pred = pred.sigmoid()
    inter = (pred * target).sum()
    return 1 - (2*inter + eps) / (pred.sum() + target.sum() + eps)
```

```python
# TensorFlow custom layer
class Scale(layers.Layer):
    def build(self, shape):
        self.g = self.add_weight(shape=(shape[-1],), initializer='ones')
    def call(self, x):
        return x * self.g

# TensorFlow custom loss
def dice_loss(y_true, y_pred, eps=1e-6):
    y_pred = tf.sigmoid(y_pred)
    inter = tf.reduce_sum(y_pred * y_true)
    return 1 - (2*inter + eps) / (tf.reduce_sum(y_pred) + tf.reduce_sum(y_true) + eps)
```

Key idea: in PyTorch anything wrapped in `nn.Parameter` inside a Module is automatically tracked, moved with `.to(device)`, saved in the `state_dict`, and given gradients. In Keras, `add_weight` inside `build` does the same job.

---

# Mixed precision (faster, less memory)

Both let you run most ops in 16-bit while keeping sensitive ones in 32-bit. On modern GPUs this is close to free speed and a big memory saving.

```python
# PyTorch
scaler = torch.cuda.amp.GradScaler()
for xb, yb in loader:
    optimizer.zero_grad()
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        loss = loss_fn(model(xb), yb)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# TensorFlow: one global switch, then train normally
keras.mixed_precision.set_global_policy('mixed_float16')
```

Why the scaler exists:

```
  16-bit can represent big numbers but underflows tiny ones to ZERO.
  Small gradients would vanish, so:

    loss ── x1024 ──▶ backward ──▶ grads big enough to survive fp16
    grads ── /1024 ──▶ back to true scale ──▶ optimizer.step()
```

Keras does this scaling internally once you set the policy. Trick: keep the final softmax and loss in float32 for stability. Keras does it automatically, and PyTorch autocast already keeps the loss math in a safe dtype.

---

# Distributed and multi-GPU

| Approach         | PyTorch                               | TensorFlow                      |
| ---------------- | ------------------------------------- | ------------------------------- |
| Simple multi-GPU | `nn.DataParallel(model)` (easy, slow) | `MirroredStrategy()`            |
| Proper multi-GPU | `DistributedDataParallel` (DDP)       | `MirroredStrategy()`            |
| Multi-node       | DDP with `torchrun`                   | `MultiWorkerMirroredStrategy()` |

```python
# TensorFlow: wrap model creation in the strategy scope
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = build_model()
    model.compile(...)
model.fit(train_ds, epochs=epochs)
```

How data-parallel training actually works, since the picture makes it click:

```
  Each GPU holds a full copy of the model and a different shard of the batch.

   GPU0: forward+backward on shard0 ─┐
   GPU1: forward+backward on shard1 ─┤─▶ ALL-REDUCE (average the grads)
   GPU2: forward+backward on shard2 ─┤     every GPU ends with the same
   GPU3: forward+backward on shard3 ─┘     averaged gradient, then all step
                                           identically -> weights stay in sync
```

For PyTorch, DDP is the real answer and `DataParallel` is the toy. DDP runs one process per GPU and syncs gradients with all-reduce, which scales far better. Launch it with `torchrun --nproc_per_node=N script.py`. More setup, but it is what serious training uses.

---

# Profiling and speed

| Tool                | PyTorch                                | TensorFlow                   |
| ------------------- | -------------------------------------- | ---------------------------- |
| Built-in profiler   | `torch.profiler.profile(...)`          | TensorBoard Profiler         |
| Time a block        | wrap in profiler or `torch.cuda.Event` | `tf.profiler.experimental`   |
| Find the bottleneck | check data loading vs compute          | check `input pipeline` in TB |

Trick: before optimizing anything, find out whether you are compute-bound or data-bound. Watch GPU utilization (`nvidia-smi`). If it swings between 0 and 100 percent, your data pipeline is starving the GPU, and the fix is workers and prefetch, not a faster model. If it sits pinned near 100 percent, you are actually compute-bound and mixed precision or a smaller model is the lever.

Trick: `torch.backends.cudnn.benchmark = True` lets cuDNN pick the fastest conv algorithm for your fixed input sizes. Free speedup when your input shapes do not change between batches.

---

# TensorBoard and logging

```python
# PyTorch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/exp1')
writer.add_scalar('loss/train', loss.item(), global_step)
writer.add_histogram('fc1.weight', model.fc1.weight, epoch)

# TensorFlow, through a callback in fit
tb = keras.callbacks.TensorBoard(log_dir='runs/exp1')
model.fit(..., callbacks=[tb])
```

Launch with `tensorboard --logdir runs`. Trick: log the learning rate and gradient norms, not just loss. When training stalls, those two tell you whether the schedule died or the gradients exploded, which loss alone cannot.

---

# Callbacks and early stopping

Keras has a rich callback system built into `fit`. In PyTorch you write the same logic inline in your loop, which is the tradeoff for writing the loop yourself.

```python
# TensorFlow: declarative callbacks
callbacks = [
    keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint('best.keras', save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(patience=3),
]
model.fit(..., callbacks=callbacks)

# PyTorch: the same idea, written by hand
best = float('inf'); patience = 5; bad = 0
for epoch in range(epochs):
    val = validate(...)
    if val < best:
        best, bad = val, 0
        torch.save(model.state_dict(), 'best.pt')
    else:
        bad += 1
        if bad >= patience:
            break
```

---

# Reproducibility and seeding

```python
# PyTorch
import random, numpy as np, torch
random.seed(0); np.random.seed(0); torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.use_deterministic_algorithms(True)   # slower, fully deterministic

# TensorFlow
tf.random.set_seed(0)
tf.config.experimental.enable_op_determinism()
```

Trick: full determinism costs speed because it disables some nondeterministic GPU kernels. Turn it on when you are debugging or publishing a result, off when you are just training fast.

---

# Exporting for deployment

| Target                      | PyTorch                                 | TensorFlow                           |
| --------------------------- | --------------------------------------- | ------------------------------------ |
| Framework-native serialized | `torch.jit.script(model)` (TorchScript) | `model.save('m.keras')` / SavedModel |
| Cross-framework             | `torch.onnx.export(...)`                | `tf2onnx`                            |
| Mobile / edge               | ExecuTorch / TorchScript                | TensorFlow Lite (`.tflite`)          |
| Browser                     | ONNX Runtime Web                        | TensorFlow.js                        |

```python
# PyTorch to ONNX (portable, runs in ONNX Runtime, TensorRT, browsers)
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, 'model.onnx',
                  input_names=['input'], output_names=['logits'],
                  dynamic_axes={'input': {0: 'batch'}})
```

Trick: put the model in `eval()` before export so dropout and batchnorm bake into inference mode. Exporting in train mode ships randomness into production.

---

# Debugging tricks that actually save time

- Print shapes first, always. Most bugs are shape bugs. A wrong shape that still runs because of broadcasting is the sneakiest bug there is.
- Overfit one batch on purpose. Take a single batch and train until loss is near zero. If the model cannot memorize one batch, your model or loss wiring is broken, not your data or hyperparameters. This test has saved me more than any other.
- If a PyTorch loss is NaN, check three suspects: learning rate too high, a `log(0)` from feeding probabilities into a log, or a double softmax before cross entropy.
- `torch.autograd.set_detect_anomaly(True)` points at the exact op that produced a NaN in the backward pass. Slow, so only while hunting.
- In TF, drop `@tf.function` while debugging so you are back in eager mode and can print and set breakpoints. Add it back for speed once it works.
- Check gradients are flowing. After `backward()`, inspect `param.grad`. If it is `None` or all zeros, the graph is disconnected somewhere, often a stray `.detach()`, a `no_grad` block, or a non-differentiable op.
- Seed everything before comparing two runs, or you will chase noise.

---

# The gotchas I have actually hit

- **Forgetting `zero_grad`.** PyTorch adds new gradients onto old. Clear them every step (except when accumulating on purpose).
- **Wrong device.** Model on GPU, batch on CPU, instant crash. Move both.
- **Double activation before loss.** Softmax layer plus `CrossEntropyLoss` trains badly. Feed logits.
- **`view` on a non-contiguous tensor.** Use `reshape`, or `.contiguous().view(...)`.
- **Channels first vs last** when moving convs between the frameworks.
- **`model.eval()` forgotten at inference.** Dropout and batchnorm keep acting like training, so predictions wobble.
- **Summing raw loss tensors for logging** keeps the whole graph in memory and leaks. Accumulate with `loss.item()`, not `loss`.
- **`tf.function` and Python side effects.** Prints and appends run once at trace time. Use `tf.print` and TF ops.
- **`from_logits` mismatch** in Keras. Match it to whether the last layer has an activation.
- **Label dtype.** PyTorch cross entropy wants `long` integer labels, not float or one-hot.

---

# Metrics and evaluation

Keras computes metrics for you when you list them in `compile`. PyTorch has no built-in metrics, you either compute them by hand or use the `torchmetrics` library, which handles the tricky part of accumulating correctly across batches.

```python
# PyTorch, by hand: accuracy over a loader
model.eval()
correct = total = 0
with torch.inference_mode():
    for xb, yb in loader:
        preds = model(xb.to(dev)).argmax(1).cpu()
        correct += (preds == yb).sum().item()
        total += yb.size(0)
print(correct / total)

# PyTorch, with torchmetrics (handles multi-batch accumulation)
import torchmetrics
acc = torchmetrics.Accuracy(task="multiclass", num_classes=3)
for xb, yb in loader:
    acc.update(model(xb).argmax(1), yb)
print(acc.compute())

# TensorFlow: declare in compile, read from evaluate
model.compile(..., metrics=["accuracy",
                            keras.metrics.Precision(),
                            keras.metrics.Recall()])
loss, *scores = model.evaluate(X, y)
```

Trick: accuracy is not a mean of per-batch accuracies when the last batch is smaller. Sum correct predictions and divide by total samples, or let `torchmetrics`/Keras accumulate. Averaging the per-batch numbers quietly overweights the small final batch.

---

# Odds and ends that matter

The small tools that are easy to miss but come up in real projects.

## Buffers: non-learnable state that still travels with the model

Some tensors are part of the model but are not trained, for example batchnorm running statistics or a fixed positional encoding. In PyTorch, register them as buffers so they move with `.to(device)` and are saved in the `state_dict`, but never get gradients.

```python
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(64))  # saved, moved, no grad
```

The Keras equivalent is `self.add_weight(..., trainable=False)`.

## Hooks: peek at or edit tensors mid-forward

Hooks let you tap into a layer without changing its code, handy for grabbing activations or debugging gradients.

```python
acts = {}
def grab(name):
    def hook(module, inp, out): acts[name] = out.detach()
    return hook
model.fc1.register_forward_hook(grab("fc1"))   # acts["fc1"] fills on next forward
```

## EMA: an averaged copy of the weights for better eval

Keeping an exponential moving average of the weights and evaluating with that often gives a small, free accuracy bump. Standard in diffusion models and many vision results.

```python
ema = torch.optim.swa_utils.AveragedModel(
    model, avg_fn=lambda avg, cur, n: 0.999*avg + 0.001*cur)
# after each optimizer.step():
ema.update_parameters(model)
# evaluate with ema instead of model
```

## Activation checkpointing: trade compute for memory

For models too big to fit, activation (gradient) checkpointing drops intermediate activations during the forward pass and recomputes them during backward. You pay extra compute to cut memory a lot, which is how large models train on smaller GPUs.

```python
from torch.utils.checkpoint import checkpoint
out = checkpoint(expensive_block, x)   # activations recomputed in backward
```

Keras has no drop-in equal, the usual answer there is mixed precision plus a smaller batch.

## Two micro-optimizations worth knowing

- `optimizer.zero_grad(set_to_none=True)` frees the grad tensors instead of filling them with zeros. Slightly faster and less memory, and it is the default in newer PyTorch.
- `x.to(dev, non_blocking=True)` overlaps the CPU-to-GPU copy with compute, but only helps when the source tensor came from a `DataLoader` with `pin_memory=True`.

---

# One-glance recap

Strip everything away and the two loops are the whole story.

PyTorch, five steps, written by you:

```python
optimizer.zero_grad()
loss = loss_fn(model(xb), yb)
loss.backward()
optimizer.step()
```

TensorFlow, the same math, handed to `fit` or written inside a `GradientTape`:

```python
with tf.GradientTape() as tape:
    loss = loss_fn(yb, model(xb, training=True))
grads = tape.gradient(loss, model.trainable_variables)
optimizer.apply_gradients(zip(grads, model.trainable_variables))
```

Everything else on this page (layers, data, schedules, metrics, dtype details) hangs off those two skeletons. Learn the skeleton, look up the rest here.
