---
title: "Quantization"
date: 2026-07-08
summary: >-
  My deep-dive notes on quantizing AI/ML models, from the affine map that turns
  real weights into integers, to why LLMs are secretly hard to quantize (the
  outlier problem), the full PTQ algorithm zoo (LLM.int8, GPTQ, AWQ, SmoothQuant,
  GGUF k-quants, SpQR/QuIP#/AQLM), QAT and the straight-through estimator, KV-cache
  quantization, calibration, the real tooling and models, and how any of it
  actually makes a model faster.
tags: [Quantization, LLM, Inference, PTQ, QAT, GPTQ, AWQ, GGUF, NF4, FP8, KV-Cache, GPU, MLOps]
---

Quantization is storing a model's numbers in fewer bits than it was trained in, and doing it carefully enough that the model still works. That is the whole idea in one sentence.

Everything else in this note is about the word *carefully*. Naively rounding every weight to 4-bit wrecks a large language model. The reason it wrecks it, and the dozen tricks people invented to stop that, is where a whole subfield lives.

The thing that took me a while to internalize: quantization is not really about saving disk space. Yes, a 70B model is 140GB in FP16 and about 35GB at 4-bit, and that is why it suddenly fits on your GPU. But the real prize is **speed**. Generating one token at a time (decode) is bottlenecked on memory bandwidth, i.e. the GPU spends almost all its time reading weights out of memory and barely any time doing math. Make each weight a quarter of the size and you read a quarter of the bytes per token, so decode gets roughly 4x faster almost for free. I unpack that memory-wall argument in [Inference Engineering](/notes/LLM-And-Agents/InferenceEgineering), so I will not repeat it here.

This note is the standalone reference on quantization itself: the math, the algorithms, and the actual tools. I have kept it complete but tried to explain each piece in plain terms first, then show the math.

---

# The core idea: mapping real numbers onto a grid

## Affine (uniform) quantization

Think of it like rounding prices. If you only had whole dollars to work with, you would map every real price onto the nearest dollar, and accept a few cents of error. Quantization does the same thing, except the "dollars" are the $2^b$ integer values a $b$-bit number can hold, and you get to choose how wide one step is.

You have a tensor of real weights $W$. Pick a float range $[\alpha, \beta]$ that covers the values you care about, and the integer range $[q_{min}, q_{max}]$ your bit-width allows (so $[0, 255]$ for unsigned INT8). Two numbers define the whole mapping, the **scale** $S$ (how many real units one integer step is worth) and the **zero-point** $Z$ (which integer stands for real zero):

$$  
S = \frac{\beta - \alpha}{q_{max} - q_{min}}, \qquad Z = \text{round}\left(q_{min} - \frac{\alpha}{S}\right)  
$$

To quantize a value you divide by the step, round, and shift by $Z$. To get an approximate value back you undo those:

$$  
W_q = \text{clip}\left(\text{round}\left(\frac{W}{S}\right) + Z, q_{min}, q_{max}\right), \qquad \hat{W} = S(W_q - Z)  
$$

**A worked example.** Say your weights run from $\alpha = -0.8$ to $\beta = 1.2$, and you want unsigned INT8 ($q_{min}=0$, $q_{max}=255$).

$$  
S = \frac{1.2 - (-0.8)}{255 - 0} = \frac{2.0}{255} \approx 0.00784, \qquad Z = \text{round}\left(0 - \frac{-0.8}{0.00784}\right) = 102  
$$

Now quantize the weight $w = 0.5$:

$$  
W_q = \text{round}\left(\frac{0.5}{0.00784}\right) + 102 = 64 + 102 = 166  
$$

And read it back: $\hat W = 0.00784 \times (166 - 102) = 0.502$. So $0.5$ is stored as the single byte $166$ and comes back as $0.502$. The error, about $0.002$, is the price of the byte.

That number $\hat W$ is what the model actually computes with. It is never exactly $W$, and the whole game is choosing the range so that $\hat W$ stays close to $W$ *where it matters to the output*. Every algorithm later in this note is a different answer to "which weights matter, and how do I protect them from rounding error."

```
 real line:   α ─────────────────────────────── β
              │        w                        │
              ▼        ▼                        ▼
 grid:      q_min ─── round(w/S)+Z ── ... ───  q_max

 S sets the spacing between grid points.
 Z shifts the grid so one integer lands exactly on real 0.
 round() picks the nearest slot, clip() catches anything past the ends.
```

## Symmetric vs asymmetric

The example above is **asymmetric**: it uses a zero-point $Z$ because the range was lopsided ($-0.8$ to $1.2$ is not centered on zero).

If you instead force $Z = 0$ you get **symmetric** quantization. You center the range on zero, $[-\alpha_{max}, \alpha_{max}]$, and drop the offset entirely:

$$  
W_q = \text{clip}\left(\text{round}\left(\frac{W}{S}\right), -2^{b-1}, 2^{b-1}-1\right), \qquad S = \frac{\alpha_{max}}{2^{b-1}-1}  
$$

Redo the example symmetrically: max absolute value $1.2$, signed INT8 range $[-127, 127]$, so $S = 1.2/127 \approx 0.00945$. Then $0.5$ maps to $\text{round}(0.5/0.00945) = 53$, and reads back as $53 \times 0.00945 = 0.501$. No zero-point needed. That missing $Z$ means one less add at inference, so the integer matmul is a touch faster.

> **When do you actually need $Z$?** Trained weights sit roughly centered on zero, so a symmetric range wastes almost nothing, and weights are usually quantized symmetrically. Activations after a ReLU or GELU are one-sided (never negative). Force a symmetric range on those and you throw away half your integer slots on negative values that never occur. That is why activations often have to be asymmetric.

## What exactly are we quantizing?

People gloss over this and then get confused about why things did or did not speed up. There are three separate things you can quantize, with very different difficulty and payoff:

- **Weights.** Fixed once training is done, so you quantize them once, offline. This is the easy one, and it is what shrinks the model on disk and in memory.
- **Activations.** The tensors flowing between layers, different for every input. Quantizing these is what lets the matmul itself run in low precision (real INT8 tensor-core math), not just store the weights small.
- **KV cache.** The keys and values you keep for every past token while generating. At long context this can grow bigger than the weights, so it gets its own section below.

The shorthand you will see everywhere is **WxAy**: $x$-bit weights, $y$-bit activations (and sometimes **KVz** for the cache). Two common ones:

- **W8A8**: 8-bit weights and 8-bit activations. A true integer matmul. Good for compute-heavy serving.
- **W4A16**: 4-bit weights, 16-bit activations. The weights are stored at 4-bit but expanded back to FP16 right before the matmul, which then runs in FP16. This helps memory-bound decode (fewer weight bytes to read) but does not speed up the math. Most local LLM quantization (GPTQ, AWQ, GGUF, NF4) is W4A16.

Keep this notation in mind. Half of "why did my quantized model get faster, or not" is answered by which letters you actually touched.

## Granularity: one scale, or many?

The scale $S$ does not have to be a single number for the whole tensor. Think of it like clothing sizes. One size for everyone (per-tensor) is cheap but fits badly. A size per person (very fine-grained) fits perfectly but is expensive to store. The useful middle is a size per small group.

- **Per-tensor**: one $S$ for the entire weight matrix. Cheapest, worst accuracy, because a single big outlier anywhere forces a coarse step on everything.
- **Per-channel**: one $S_c$ per output row of the matrix, $S_c = \dfrac{\max_i |W_{c,i}|}{2^{b-1}-1}$. Now an outlier in one row only hurts that row.
- **Per-group (per-block)**: one $S$ per contiguous chunk of, say, 128 weights. This is the sweet spot at 4-bit.

| Granularity          | Extra storage           | Where it shows up                                                                      |
| -------------------- | ----------------------- | -------------------------------------------------------------------------------------- |
| Per-tensor           | 1 scale per tensor      | INT8 on well-behaved layers, many CNNs                                                 |
| Per-channel          | 1 scale per output row  | The standard baseline for INT8 weights                                                 |
| Per-group (e.g. 128) | 1 scale per 128 weights | GPTQ, AWQ, GGUF at 4-bit, because per-tensor INT4 on LLM weights is basically unusable |

There is a catch people forget: finer granularity means more scales to store, and scales are usually FP16. A group of 128 weights at 4-bit adds one 16-bit scale per 128 weights, which is $16/128 = 0.125$ extra bits per weight. That is why QLoRA even quantizes *the scales themselves* (double quantization). The metadata is not free.

## Dynamic vs static quantization

**Static**: compute the scale once, offline, and bake it in.

**Dynamic**: compute it on the fly from the actual min/max of whatever tensor is in front of you right now.

Weights are always static, since they never change at runtime. Activations are the interesting case. Dynamic activation quantization computes a fresh scale per token, needs no calibration set, and adapts to whatever comes in, at the cost of a quick max-finding pass. A lot of W8A8 serving uses **dynamic per-token** activation quantization for exactly that reason: it sidesteps the risk that your offline calibration data does not match real traffic.

## Rounding: (nearest, stochastic and learned)

The obvious choice is **round-to-nearest (RTN)**: $2.4$ becomes $2$. Deterministic, done. Almost every one-shot conversion uses it.

**Stochastic rounding** rounds up with probability equal to the fractional part, so $2.4$ rounds up to $3$ about 40% of the time and down to $2$ the other 60%. On average it lands back on $2.4$, i.e. it is unbiased: $E[\text{round}_{stoch}(x)] = x$. This matters mostly for low-precision *training*, where the same value gets rounded thousands of times and RTN's small bias would pile up. For one-shot weight quantization, where you round each weight once, RTN is fine.

Then the non-obvious one: rounding to the nearest grid point is **not actually optimal** for the model's output, and **AdaRound** showed it. Sometimes rounding a weight the "wrong" way (up when down was nearer) hurts the layer's output less, because of how that weight interacts with the others. AdaRound learns, per weight, whether to round up or down so the layer's *output* changes least. The takeaway that echoes through the rest of this note: rounding direction is a knob you can optimize, and several later methods are variations on "round smart, not near."

## Quantization error behaves like noise (SQNR)

Here is a useful way to think about the damage. Rounding error is basically small random noise added to each weight. If a step is $S$ wide, the error on any weight is somewhere in $[-S/2, +S/2]$, and its variance works out to:

$$  
\sigma_e^2 = \frac{S^2}{12}  
$$

We measure how loud the signal is compared to that noise with the **signal-to-quantization-noise ratio** (SQNR), in decibels:

$$  
\text{SQNR}*{dB} = 10 \log*{10}\left(\frac{\sigma_{signal}^2}{\sigma_e^2}\right)  
$$

The one fact to remember: **each extra bit buys you about 6 dB.** Adding a bit doubles the number of grid points, which halves the step $S$, which quarters the noise variance, which is +6 dB. So going from 4-bit to 8-bit is roughly 24 dB cleaner. (You will see this written as $\approx 6.02b + 1.76$. Trust the "6 dB per bit" slope; the $1.76$ constant is an ADC-theory detail that assumes a specific signal shape and does not always apply to a pile of weights.)

The punchline that carries the rest of the note: quantization *adds noise* to your weights and activations. You cannot delete that noise, you can only choose where it lands. GPTQ, AWQ, SmoothQuant and the rest are just different strategies for steering the noise away from the numbers the model truly depends on.

---

# Why LLMs are secretly hard: the outlier problem

Quantization is old. People shipped INT8 vision models on phones years before LLMs existed. So why did plain 8-bit break large transformers when it was fine on a ResNet?

The answer, from Dettmers' LLM.int8() work, is **outlier features**. Once a transformer gets big enough (around 6.7B parameters), a handful of activation dimensions start blowing up to huge magnitudes, sometimes 20x everything around them, and they show up in the *same* few dimensions for almost every token.

Here is why that is so destructive. Remember the scale $S$ is set by the largest value in the range. One monster value that is 20x bigger than everything else forces a huge step size, and now all the normal-sized weights get squashed into just a few grid slots near zero. Picture pricing a room full of people by net worth, in whole-million-dollar buckets: throw one billionaire into the room and everyone else rounds to "zero millions." The outlier eats all your precision.

And you cannot just drop the outliers, because they are load-bearing, the model genuinely relies on them. So this one phenomenon drives most of the algorithm zoo. Every method below is, underneath, a different way to handle outliers:

- **LLM.int8()** pulls the outlier dimensions out and keeps them in FP16, quantizing the rest.
- **SmoothQuant** shifts the outlier magnitude out of the activations and into the weights, which cope better.
- **AWQ** protects the specific weight channels that line up with large activations.
- **SpQR, QuIP#** either keep the worst weights in high precision, or rotate the whole space so no single dimension is an outlier anymore.

Hold onto the outlier picture. It is the "why" behind almost everything that follows.

---

# The precision format zoo

Before the algorithms, the number formats they target. A format is just how you split the bits between sign, exponent (dynamic range), and mantissa (precision).

| Format             | Bits      | Layout (S/E/M) | Notes                                                                       |
| ------------------ | --------- | -------------- | --------------------------------------------------------------------------- |
| FP32               | 32        | 1 / 8 / 23     | Old training default, rarely used for inference now                         |
| TF32               | 19        | 1 / 8 / 10     | NVIDIA's tensor-core format, FP32 range with less mantissa                  |
| FP16               | 16        | 1 / 5 / 10     | Narrow exponent, needs loss scaling in training                             |
| BF16               | 16        | 1 / 8 / 7      | FP32's range with a shorter mantissa, no loss scaling, the training default |
| FP8 E4M3           | 8         | 1 / 4 / 3      | More precision, less range, used for weights and forward activations        |
| FP8 E5M2           | 8         | 1 / 5 / 2      | More range, less precision, used for gradients                              |
| FP6 / FP4 (E2M1)   | 6 / 4     | 1 / 2 / 1      | Emerging on Blackwell-class hardware, still floating point                  |
| INT8 / INT4 / INT2 | 8 / 4 / 2 | fixed point    | INT8 mainstream, INT4 is where the 4-bit zoo lives, INT2 mostly research    |

Two things worth understanding beyond the table.

**1) Integer vs float at the same bit-width:** INT8 spaces its levels evenly. FP8 spaces them unevenly, dense near zero and sparse out in the tails, because of the exponent. That makes FP8 naturally better at swallowing outliers, which is a big reason it became the go-to *serving* format on H100-class hardware: near-INT8 memory savings, much less outlier pain, and native FP8 tensor cores so the math is genuinely faster too.

**2) MX / microscaling formats:** The newer OCP "MX" formats (MXFP8, MXFP6, MXFP4) are per-block by design: a block of 32 elements shares one tiny power-of-two scale, and each element is a small float like FP4. This is basically per-group quantization baked into the number format and supported directly in hardware. It is where Blackwell-era low-precision inference is heading.

### **NF4: a grid shaped like the weights**

NF4 (NormalFloat4), from the QLoRA paper, starts from one observation: trained weights are roughly a bell curve centered on zero. A uniform 4-bit grid wastes slots out in the tails where almost no weights live, and is too coarse near zero where most of them cluster. NF4 instead puts its 16 levels at the **quantiles of a standard normal**, so each slot holds an equal *slice of probability* rather than an equal slice of the number line, more resolution where the weights actually are. The 16 codebook values are:

$$  
q_i = \frac{1}{2}\Big[Q\left(\tfrac{i}{17}\right) + Q\left(\tfrac{i+1}{17}\right)\Big], \qquad i = 0, \dots, 15  
$$

where $Q$ is the quantile function (inverse CDF) of $N(0,1)$. In words: cut the bell curve into 17 equal-probability slices and take the midpoint of each. This is provably optimal for normally-distributed data. QLoRA pairs it with per-block normalization and double quantization (see granularity above).

### Ternary and binary: as low as it goes

BitNet b1.58 pushes weights down to just three values, $-1, 0, +1$, using an absolute-mean scale:

$$  
Q_w(W) = \Delta \cdot \text{RoundClip}\left(\frac{W}{\Delta + \epsilon}, -1, 1\right), \qquad \Delta = \text{mean}(|W|)  
$$

with $\text{RoundClip}(x, a, b) = \max(a, \min(b, \text{round}(x)))$ and $\epsilon$ a small guard against dividing by zero. Three states need $\log_2 3 \approx 1.58$ bits, hence the name. The catch: BitNet is not a squeeze of an existing model. It is trained at this precision from scratch (QAT, below), because you simply cannot take a normally-trained model down to ternary after the fact without it falling apart. Pure 1-bit (binary) is the original BitNet and is still mostly research.

---

# Calibration: turning data into scales

Most PTQ methods need **calibration data**: a small, unlabeled sample of representative inputs (typically 128 to 512 sequences from a corpus like C4 or WikiText) that you run through the model once to watch how the numbers behave. You are not training on it, you are measuring with it.

Turning those measurements into a scale is a real design choice, and there is a ladder from crude to clever:

- **Min-max.** Use the literal smallest and largest value. Simplest and most fragile, one outlier sets the range for everyone.
- **Percentile clipping.** Drop the extreme 0.1% first, then take min/max. You give up the true extremes for a tighter grid on the bulk.
- **MSE-optimal clipping.** Search clip ranges and keep whichever minimizes reconstruction error directly.
- **KL / entropy calibration.** TensorRT's classic INT8 recipe: pick the clipping threshold whose quantized distribution looks most like the original FP32 one (smallest KL divergence). It preserves the *shape* of the distribution, not just the extremes.
- **Second-order (GPTQ).** Accumulate $XX^\top$ over the calibration batch, capturing how the output responds to each weight, not just a range.
- **Activation-magnitude (AWQ).** Track per-channel activation averages to find which weight channels are salient.
- **Importance matrix (GGUF imatrix).** llama.cpp can build an "imatrix" from calibration data that weights each dimension's error by how much it actually matters, and it clearly improves low-bit k-quants. So the common line that "GGUF needs no calibration" is only true for the basic quants; the good ones use an imatrix.

> **The calibration-mismatch trap.** If you calibrate on generic web text, the model looks great on perplexity (which is measured on similar text) and then quietly degrades on code, math, or long-context reasoning that the calibration set never touched. Low perplexity on the calibration data is *not* proof of quality on your data. Always re-check on the real task. This is the single most common way people ship a broken quantized model without noticing.

---

# A minimal implementation, by hand

Real kernels are CUDA and C++, but the logic is exactly the affine map. Here it is in plain NumPy, quantizing a toy weight matrix per-tensor and per-channel so you can watch the granularity tradeoff turn into an actual number:

```python
import numpy as np

def quantize(W, bits=8, per_channel=False):
    q_min, q_max = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    if per_channel:
        # one scale per output row (max over each row's columns)
        amax = np.max(np.abs(W), axis=1, keepdims=True)
    else:
        amax = np.max(np.abs(W))                # one scale for the whole tensor
    scale = amax / q_max
    scale = np.where(scale == 0, 1e-8, scale)   # guard all-zero rows
    W_q = np.clip(np.round(W / scale), q_min, q_max)
    return W_q.astype(np.int8), scale

def dequantize(W_q, scale):
    return W_q.astype(np.float32) * scale

def sqnr_db(W, W_hat):
    return 10 * np.log10(np.mean(W ** 2) / np.mean((W - W_hat) ** 2))

rng = np.random.default_rng(0)
W = rng.normal(0, 0.02, size=(256, 256)).astype(np.float32)
W[10, :] *= 25   # inject one outlier row, i.e. a "salient channel"

for per_channel in (False, True):
    W_q, scale = quantize(W, bits=8, per_channel=per_channel)
    W_hat = dequantize(W_q, scale)
    tag = "per-channel" if per_channel else "per-tensor"
    print(f"{tag:12s} SQNR: {sqnr_db(W, W_hat):.1f} dB")

# per-tensor    SQNR: ~28 dB   (the outlier row forces a coarse scale on everyone)
# per-channel   SQNR: ~45 dB   (the outlier only costs its own row)
```

That ~17 dB gap, caused by a single bad row, is the whole reason per-channel and per-group exist. Real LLM weights have far more and far worse outliers than this toy, which is exactly why plain min-max is not enough at 4-bit, and why the algorithms in the next section had to be invented.

---

# The post-training quantization zoo

PTQ takes a frozen, already-trained model and quantizes it with no retraining and no gradients through the whole network, in minutes to hours instead of GPU-days. This is where most of the research has gone, because it is what you can actually do to someone else's open-weights checkpoint on one machine.

## 1) LLM.int8(): keep the outliers in FP16

The first method that made 8-bit work for big models, and the cleanest illustration of the outlier idea. It detects the handful of outlier dimensions, pulls those specific columns out, and computes them in FP16, while the other 99%+ of the matmul runs in INT8. The two partial results are added back together. So almost all the compute is 8-bit, but the load-bearing outliers never get squashed. This is exactly what `load_in_8bit=True` does in bitsandbytes. It is nearly lossless, but the FP16 side path means it is mostly a memory win, not always a speed one.

## 2) GPTQ: round using second-order information

The plain idea first: quantize the weights one column at a time, and after each column, **nudge the not-yet-quantized weights to make up for the error you just introduced.** It is like packing a suitcase and, each time you squash one item, shifting the others to keep the overall shape right.

More precisely, GPTQ treats each layer as a least-squares problem. Given calibration inputs $X$ and weights $W$, it wants the quantized $\hat W$ whose *output* is closest to the original:

$$  
\hat{W} = \arg\min_{\hat{W}}  WX - \hat{W}X_2^2  
$$

To know how to compensate, it uses the layer's Hessian $H = 2XX^\top$, which measures how sensitive the output is to each weight and how the weights interact. This comes from an older pruning method (Optimal Brain Surgeon) adapted to quantization (Optimal Brain Quantization, OBQ). The per-weight update and the error it costs are:

$$  
\delta = -\frac{w_q - \text{quant}(w_q)}{[H^{-1}]*{qq}} H^{-1}*{:,q}, \qquad \varepsilon_q = \frac{\big(w_q - \text{quant}(w_q)\big)^2}{[H^{-1}]_{qq}}  
$$

You do not need to memorize that. The point is: $\delta$ is the correction spread across the remaining weights, and $\varepsilon_q$ is how much error this weight costs. OBQ is accurate but far too slow at billions of parameters, so GPTQ makes three practical simplifications:

1. **Fixed order.** Just quantize columns left to right instead of re-sorting by error every step. At LLM scale this is nearly as good and much cheaper. (An optional `act_order` flag brings back smart ordering for a little more accuracy.)
2. **Cholesky trick.** Precompute one matrix factorization of $H^{-1}$ up front, so all the corrections fall out of it cleanly instead of repeatedly inverting a shrinking matrix (which is numerically unstable).
3. **Lazy batch updates.** Correct a block of 128 columns at once as a single big matmul, rather than many tiny updates. GPUs love big matmuls, so this is what makes GPTQ run in minutes.

```
 weight matrix, one row = one output neuron:

 col:   0   1   2   3 | 4   5   6   7 | ...
       [q] [q] [q] [q]|[·] [·] [·] [·]| ...
        └── block 1 ──┘
        quantize this block, work out the
        compensation, then push it onto every
        remaining column with one matmul ───────►
```

> **Why is this fast if the math looks so heavy?** Because GPTQ never runs a backward pass or touches the training loss. It is a forward pass over calibration data plus some linear algebra, layer by layer. It solves a small local problem per layer instead of optimizing the whole network. That is the trick behind the whole PTQ family: stay local, stay gradient-free.

## AWQ: protect the weights the activations care about

AWQ starts from a sharp observation: fewer than 1% of weight channels really drive output quality, and you find them by looking at **activation** magnitude, not weight magnitude. Channels that consistently see large activations are the ones whose weights you must keep accurate.

Protecting them costs nothing, thanks to a simple trick. In a matmul $Y = XW$ you can scale a weight channel *up* and scale the matching input *down* by the same factor, and the output is unchanged:

$$  
Y = \big(X \cdot \text{diag}(s)^{-1}\big)\big(\text{diag}(s) \cdot W\big)  
$$

But the scaled-up weights now suffer *less* rounding error, because a fixed step size is a smaller fraction of a bigger number. AWQ finds the per-channel scale $s = (\bar{|X|})^\alpha$ by a quick search over $\alpha \in [0,1]$. The best part: the activation-side scaling is folded into the previous layer at export time, so it adds zero runtime cost and stores nothing in mixed precision. It just quantizes 4-bit weights that were pre-scaled to protect the important ones. That is why AWQ is fast to apply and holds quality, and why it shows up everywhere.

## SmoothQuant: move the difficulty from activations to weights

GPTQ and AWQ are weight-only (W4A16). SmoothQuant goes after the harder prize, full **W8A8**, where activations are quantized too so you get a genuine integer matmul for compute-heavy serving. The obstacle is those activation outliers again. Weights are flat and easy; activations have the few monster channels. SmoothQuant uses the same scale-shifting trick as AWQ, but aimed the other way: it smooths the activations by pushing some of their range into the weights, which tolerate it better.

$$  
s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}, \qquad Y = \big(X \cdot \text{diag}(s)^{-1}\big)\big(\text{diag}(s) \cdot W\big)  
$$

The exponent $\alpha$ controls how much difficulty moves; $\alpha = 0.5$ is the usual sweet spot, which simplifies to $s_j = \sqrt{\max(|X_j|)/\max(|W_j|)}$. The one-liner to keep them straight: **AWQ** protects salient weights for memory savings (weight-only), **SmoothQuant** rebalances weights and activations for throughput (W8A8).

## OmniQuant: learn the clipping and the smoothing

OmniQuant is the "make the knobs learnable" step. Instead of hand-picking clip ranges and smoothing factors, it turns them into trainable parameters and optimizes them block by block with gradient descent. It stays cheap because it only trains those few extra parameters, not the model, but it squeezes out more accuracy at low bit-widths than the hand-tuned methods.

## Going below 4 bits: SpQR, QuIP#, AQLM

At 2 to 3 bits, uniform quantization breaks down and you need cleverer representations. Worth knowing by name and idea:

- **SpQR (Sparse-Quantized Representation).** Keeps the small set of worst-offending weights in high precision as a sparse side table, and quantizes the rest to 3 to 4 bits. The weight-side version of LLM.int8()'s outlier isolation.
- **QuIP / QuIP#.** Rotate the weights (with random orthogonal or Hadamard transforms) so no single coordinate is an outlier anymore, spreading the difficulty evenly, which makes even 2-bit viable. QuIP# adds a lattice codebook to pack them tightly. A genuinely different idea: instead of protecting outliers, rotate until there are none.
- **AQLM.** Represents each weight vector as a sum of vectors from learned codebooks (vector quantization, not scalar). Reaches 2-bit with surprisingly little loss, at the cost of a slower quantization process.

The through-line: scalar grids run out of room below 4-bit, so you either keep the worst weights high-precision (SpQR), rotate the outliers away (QuIP#), or switch to codebooks (AQLM).

## GGUF / llama.cpp k-quants: how most people actually run models locally

If you have run a model on your laptop via Ollama or LM Studio, you have used this, because they wrap llama.cpp. GGUF's k-quants group weights into **super-blocks of 256**, split into sub-blocks that each get their own scale, and then quantize the super-block's scales too (double quantization, to keep the metadata cheap).

The names look cryptic but decode simply. `Q4_0` / `Q4_1` are the old flat formats. The `_K` formats are k-quants, and the `_S` / `_M` / `_L` suffix says how much mixed precision they use: `_M` and `_L` keep a few error-sensitive tensors at higher bit-width. `Q4_K_M` sits around 4.8 bits per weight and is the default "good for almost everything" pick.

| Quant    | ~bits/weight | Quality vs FP16             | When you reach for it                |
| -------- | ------------ | --------------------------- | ------------------------------------ |
| `Q8_0`   | ~8.5         | Essentially lossless        | You have the VRAM and want zero risk |
| `Q6_K`   | ~6.6         | Very small loss             | High-quality local inference         |
| `Q5_K_M` | ~5.7         | Small loss                  | Good balance on mid-range hardware   |
| `Q4_K_M` | ~4.8         | Noticeable but usually fine | The default for most local setups    |
| `Q3_K_M` | ~3.9         | Real degradation            | Tight VRAM                           |
| `Q2_K`   | ~2.6         | Significant                 | Last resort when nothing else fits   |

Basic k-quants need no calibration, so you can convert a model in minutes with nothing but the checkpoint. The higher-quality path uses an imatrix from calibration data and does measurably better at the low end.

---

# Quantization-aware training (QAT)

Everything above works on a frozen model. QAT instead *trains the model to expect* the rounding it will face at inference. You reach for it when you are pushing precision so low that PTQ can no longer recover the accuracy.

## Fake quantization and the straight-through estimator

QAT inserts a "fake quantize" step into the forward pass, the same round-clip-dequantize you have seen, so the model computes with the rounded weights $\tilde W$ while the parameter you optimize stays full-precision $W$:

$$  
\tilde{W} = S\Big(\text{clip}\big(\text{round}(W/S) + Z, q_{min}, q_{max}\big) - Z\Big)  
$$

The problem: $\text{round}(\cdot)$ is a flat staircase, so its gradient is zero almost everywhere and backprop has nothing to push on. The **straight-through estimator (STE)** cheats: on the backward pass it pretends the rounding was just the identity function (as long as the value is inside the range):

$$  
\frac{\partial L}{\partial W} \approx \frac{\partial L}{\partial \tilde{W}} \cdot \mathbb{1}[\alpha \le W \le \beta]  
$$

> **Why does such a crude cheat work?** Because rounding is a small nudge, and for most values $\text{round}(x)$ is close to $x$. Treating it as the identity lets gradients flow, and the optimizer ends up parking weights in positions that survive rounding, which was the goal all along. You are not trying to differentiate the staircase, you are nudging weights to where the staircase does not hurt.

A refinement worth knowing is **LSQ (Learned Step Size Quantization)**, which makes the scale $S$ itself a trainable parameter instead of fixing it from calibration. Letting the model learn its own grid is a reliable accuracy win.

## When QAT is worth it

QAT needs your full training pipeline, real data, and real compute. So most people should not use it: for "take an open checkpoint and shrink it," PTQ (GPTQ, AWQ, GGUF) is enough, which is why those dominate. QAT earns its cost in three cases: very aggressive bit-widths where PTQ runs out of room, edge/mobile deployment where you control training and want every last point of accuracy, and native-low-precision models like BitNet that are trained ternary from scratch because there is no post-hoc path to 1.58 bits.

---

# KV-cache quantization

This one gets its own section because at long context it, not the weights, is often what actually runs you out of memory.

While generating, you cache the key and value tensors for every past token so you do not recompute them. That cache grows with context length and batch size, and past 100K tokens it can be bigger than the model itself. Quantizing it is how you fit long context and big batches.

The wrinkle: keys and values behave differently, so you treat them differently. The finding from KVQuant and KIVI is that the **key** cache has outlier *channels* (quantize it per-channel) while the **value** cache is better behaved (quantize per-token). KIVI pushes this to a 2-bit KV cache. In production the simpler, common option is an **FP8 KV cache** (vLLM, TensorRT-LLM), which roughly halves cache memory with almost no quality loss and no fancy logic. Chasing extreme context lengths, the specialized 2-bit methods buy you more, at more complexity.

The mental model: weight quantization shrinks the *model*, KV-cache quantization shrinks the *conversation*, and for long-context or high-batch serving the second is frequently the bigger lever.

---

# What actually gets faster (and what does not)

This trips people up constantly, so let me be blunt.

**Weight-only quantization (W4A16)** stores weights small but expands them back to FP16 before the matmul, which runs in FP16. So you save memory and speed up **memory-bound decode** (fewer weight bytes to stream), but you do *not* speed up **compute-bound prefill**, and you add a small expansion cost. This is GPTQ, AWQ, GGUF, NF4. Great for local inference and decode-heavy chat.

**Weight-and-activation quantization (W8A8)** quantizes activations too and runs a real integer (or FP8) matmul on the tensor cores. This speeds up the actual math, so it helps prefill and high-throughput serving. This is SmoothQuant, FP8, INT8 serving in TensorRT-LLM and vLLM. It is harder because of the activation-outlier problem, which is the whole reason SmoothQuant and FP8 exist.

So "I quantized my model, why is prefill the same speed" has a precise answer: you did W4A16, prefill is compute-bound, and you only touched the memory side. Match the letters (WxAy) to your bottleneck (decode vs prefill) and the confusion goes away.

One more reality check: the memory savings are rarely the clean 4x the bit-width suggests. Group scales, zero-points, the odd high-precision tensor, and the KV cache all sit outside the tidy quantized weights. A "4-bit" model is usually more like 4.5 to 5 effective bits once you count the metadata.

---

# Quantizing a real Hugging Face model

The theory maps onto a handful of tools you will actually run. Short versions of each.

**bitsandbytes**, the easiest path, does both 8-bit (LLM.int8()) and 4-bit (NF4), and it is what QLoRA fine-tuning sits on:

```python
from transformers import BitsAndBytesConfig, AutoModelForCausalLM

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",          # the Gaussian-quantile codebook
    bnb_4bit_use_double_quant=True,     # quantize the block scales too
    bnb_4bit_compute_dtype="bfloat16",  # expand to bf16 for the matmul
)
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb)
# for QLoRA you then freeze this 4-bit base and train small bf16 LoRA adapters on top
```

**AutoGPTQ / GPTQModel**, GPTQ, needs calibration data:

```python
from gptqmodel import GPTQModel, QuantizeConfig

cfg = QuantizeConfig(bits=4, group_size=128, desc_act=True)  # desc_act = act-order
model = GPTQModel.load(model_id, cfg)
model.quantize(calibration_examples)   # a list of tokenized calibration sequences
model.save_quantized(out_dir)
```

**AutoAWQ**, AWQ, same shape with an AWQ-specific search:

```python
from awq import AutoAWQForCausalLM

model = AutoAWQForCausalLM.from_pretrained(model_id)
model.quantize(tokenizer,
               quant_config={"w_bit": 4, "q_group_size": 128, "zero_point": True},
               calib_data=calibration_examples)
model.save_quantized(out_dir)
```

**llama.cpp**, GGUF, convert then quantize, optionally with an imatrix:

```bash
python convert_hf_to_gguf.py <hf_model_dir> --outfile model-f16.gguf
# optional: build an importance matrix from calibration text for better low-bit quants
./llama-imatrix -m model-f16.gguf -f calibration.txt -o model.imatrix
./llama-quantize --imatrix model.imatrix model-f16.gguf model-Q4_K_M.gguf Q4_K_M
```

For serving, **vLLM** and **TensorRT-LLM** load GPTQ / AWQ / FP8 checkpoints directly and run them at scale, and TensorRT-LLM is where you would do INT8 / FP8 W8A8 with SmoothQuant-style calibration for maximum throughput.

| Tool                 | Format            | Needs calibration? | Best for                                       |
| -------------------- | ----------------- | ------------------ | ---------------------------------------------- |
| bitsandbytes         | INT8 / NF4        | No                 | QLoRA fine-tuning, quick 4-bit or 8-bit loads  |
| AutoGPTQ / GPTQModel | GPTQ INT4         | Yes                | GPU-served W4A16 inference                     |
| AutoAWQ              | AWQ INT4          | Yes (light)        | GPU-served W4A16, fast to apply, holds quality |
| llama.cpp            | GGUF k-quants     | Optional (imatrix) | Local / CPU / Apple Silicon / edge             |
| TensorRT-LLM / vLLM  | FP8 / INT8 / INT4 | Depends            | Production GPU serving, W8A8 throughput        |

---

# Real quantized models you will actually meet

Grounding it in specifics, because vague gestures do not help:

- **GGUF community builds** of Llama, Mistral, Qwen and friends on Hugging Face, `Q4_K_M` as the default download, run through llama.cpp, Ollama, or LM Studio. This is most local usage.
- **GPTQ and AWQ checkpoints**, tagged `-GPTQ` or `-AWQ` on the hub, served through vLLM or TensorRT-LLM at INT4.
- **NF4 + QLoRA**: the original QLoRA paper's Guanaco models are the canonical example, a frozen NF4 base with LoRA adapters fine-tuned on a single consumer GPU. This is how most people fine-tune large models cheaply.
- **FP8 on H100**: increasingly the default *serving* format in vLLM, TensorRT-LLM, and NVIDIA NIM, near-FP16 quality at half the memory with native tensor-core speedups.
- **BitNet b1.58**: Microsoft's natively-ternary models, the frontier case for where extreme compression is headed, trained low-precision from scratch rather than squeezed after the fact.

---

# Evaluating a quantized model

You have to check the damage, and check it the right way, because the easy metric lies.

**Perplexity** (on WikiText2, C4, and so on) is cheap, standard, and a *weak* signal. It measures average next-token fit, not whether the model can still reason, code, or follow instructions. A model can hold perplexity and still lose real capability.

**Downstream task evals** (MMLU, GSM8K, HumanEval, or your own task suite) are the real signal. Run them. Do not skip them because perplexity looked fine.

**KL divergence** between the full-precision and quantized model's output distributions is the sharpest task-agnostic check:

$$  
D_{KL}\big(P_{fp16}  P_{quant}\big) = \sum_i P_{fp16}(i) \log\frac{P_{fp16}(i)}{P_{quant}(i)}  
$$

computed per-token over a held-out set. It catches distribution drift that a single averaged perplexity number smears over: if the quantized model is putting its probability mass in noticeably different places, this sees it even when perplexity does not move.

And the caveat I flagged in [Inference Engineering](/notes/LLM-And-Agents/InferenceEgineering): quantization quality is workload-dependent. A model that benchmarks fine at INT4 can fall apart specifically on long-context reasoning or code. Evaluate on *your* task, not just perplexity.

---

# Quantization beyond LLMs

LLMs get all the attention now, but quantization grew up in vision and edge deployment, and that world is worth a paragraph because the ideas transfer and the tooling is mature.

On phones and embedded devices, INT8 CNNs have been standard for years. **TensorFlow Lite** does post-training INT8 (per-axis weights, activations calibrated from a small representative dataset), plus lighter weights-only and FP16 modes. **PyTorch** has a full stack: the older eager-mode flow with observers, FX graph-mode quantization, and the newer PT2 export path, running on backends like FBGEMM (server x86) and XNNPACK (mobile ARM). **ONNX Runtime** has its own PTQ and QAT tooling. NVIDIA's classic INT8 CNN recipe (the KL-divergence calibration above) came out of exactly this world.

The differences from the LLM case are instructive. Vision models are mostly compute-bound and quantized to INT8 W8A8 to speed up the convolutions, whereas LLM decode is memory-bound and often weight-only. Vision models rarely have the dramatic activation outliers that plague large transformers, so plain per-channel INT8 usually just works without the GPTQ/AWQ/SmoothQuant machinery. If you understand the LLM story, the vision story is the easier special case.

---

# One table to bookmark

| Method / Format     | Bits (WxAy) | Memory vs FP16 | Quality             | PTQ/QAT      | Calibration? | Best for                         |
| ------------------- | ----------- | -------------- | ------------------- | ------------ | ------------ | -------------------------------- |
| BF16 / FP16         | 16          | baseline       | reference           | n/a          | n/a          | Training, max fidelity           |
| FP8 (E4M3)          | W8A8        | ~2x smaller    | Very high           | PTQ          | Light        | GPU serving on Hopper+           |
| LLM.int8()          | W8A16       | ~2x smaller    | Near-lossless       | PTQ          | No           | Easy 8-bit loads, memory         |
| SmoothQuant         | W8A8        | ~2x smaller    | High                | PTQ          | Yes          | Compute-bound INT8 serving       |
| GPTQ                | W4A16       | ~4x smaller    | High                | PTQ          | Yes          | GPU-served 4-bit decode          |
| AWQ                 | W4A16       | ~4x smaller    | High                | PTQ          | Yes (light)  | GPU-served 4-bit, fast to apply  |
| GGUF `Q4_K_M`       | ~W4.8A16    | ~3.3x smaller  | Good                | PTQ          | Optional     | Local / CPU / edge               |
| NF4 (QLoRA)         | W4A16       | ~4x smaller    | Good, great w/ LoRA | PTQ          | No           | Cheap fine-tuning on one GPU     |
| SpQR / QuIP# / AQLM | ~W2-3A16    | ~5-8x smaller  | Frontier            | PTQ          | Yes          | Squeezing below 4-bit            |
| Ternary (BitNet)    | ~1.58-bit   | ~10x smaller   | Model-dependent     | QAT (native) | n/a          | Extreme compression from scratch |

---

If you forget everything else, keep these:

- Quantization maps real numbers onto a grid. Scale sets the spacing, zero-point aligns it, and every method is a smarter way to choose the range or the rounding than plain min-max.
- The error is noise you cannot delete, only steer. Each bit buys about 6 dB, and the craft is aiming the noise away from the weights the model depends on.
- LLMs are hard because of a few huge outlier features, and almost every method is a different way to deal with them: isolate them (LLM.int8, SpQR), migrate them (SmoothQuant), protect around them (AWQ), or rotate them away (QuIP#).
- GPTQ rounds using second-order information, calibration-only and gradient-free, which is why it is both accurate and fast.
- Match the WxAy to your bottleneck: weight-only (W4A16) speeds up memory-bound decode, weight-and-activation (W8A8, FP8) speeds up compute-bound prefill and throughput.
- KV-cache quantization shrinks the conversation, not the model, and at long context it is often the bigger win.
- QAT teaches a model to survive rounding by training through it with the straight-through estimator, and it is worth the cost only at the aggressive end.

And the decision it all collapses to. Serving on GPU at scale, reach for FP8, AWQ, or GPTQ through vLLM or TensorRT-LLM. Running locally, reach for GGUF `Q4_K_M`. Fine-tuning cheaply, reach for QLoRA and NF4. Going below 4-bit, reach for the codebook and rotation methods and expect to work for it.
