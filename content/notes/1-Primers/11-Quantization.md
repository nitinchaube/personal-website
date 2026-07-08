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

Quantization is the trick of storing a model's numbers in fewer bits than it was trained in, and doing it carefully enough that the model still works. That is the whole idea in one sentence. Everything else in this note is about the word *carefully*, because a naive "just round everything to 4-bit" wrecks a large language model, and the reason it wrecks it, and the dozen different ways people have found to stop it from wrecking it, is where an entire subfield lives.

The thing that took me a while to internalize is that quantization is not really about compression. Yes, a 70B model in FP16 is 140GB and the same model at 4-bit is around 35GB, and that alone is why it fits on your GPU now. But the deeper reason we do it is speed. Decode (generating one token at a time) is memory-bandwidth-bound, i.e. the GPU spends most of its time reading weights out of memory and almost no time doing math. If you make each weight a quarter of the size, you read a quarter of the bytes per token, and decode gets roughly four times faster more or less for free. I covered that memory-wall argument in detail in [Inference Engineering](/notes/LLM-And-Agents/InferenceEgineering), so I will not re-derive it here. This note is the standalone reference on quantization itself: the math, the algorithms, and the actual tools.

I have tried to make this complete rather than short. If you read it top to bottom you should not have to open another tab.

---

# The core idea: mapping real numbers onto a grid

## Affine (uniform) quantization

You have a tensor of real-valued weights $W$ and you want to store each entry in $b$ bits. A $b$-bit integer gives you $2^b$ distinct slots, and quantization is the function that decides which slot each real number lands in, plus how you turn a slot back into an approximate real number.

Pick a float range $[\alpha, \beta]$ that covers the values you care about, and an integer range $[q_{min}, q_{max}]$ that your bit-width supports (so $[0, 255]$ for unsigned INT8, or $[-128, 127]$ for signed INT8). The **scale** $S$ and **zero-point** $Z$ are:

$$  
S = \frac{\beta - \alpha}{q_{max} - q_{min}}, \qquad Z = \text{round}\left(q_{min} - \frac{\alpha}{S}\right)  
$$

$S$ answers "how many real units is one integer step," and $Z$ answers "which integer represents real zero." You need $Z$ because $\alpha$ and $\beta$ are usually not symmetric around zero, so you need an offset to line the grid up with the float range. Quantizing a value and then reconstructing it back is:

$$  
W_q = \text{clip}\left(\text{round}\left(\frac{W}{S}\right) + Z, q_{min}, q_{max}\right), \qquad \hat{W} = S(W_q - Z)  
$$

$\hat W$ is the number the model actually computes with at inference time. It is never exactly $W$, and the entire game is choosing $\alpha, \beta$ (which fix $S, Z$) so that $\hat W$ stays close to $W$ in the places that matter to the model's output. Every algorithm later in this note is a different answer to "where do the places that matter live, and how do I protect them."

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

If you force $Z = 0$ the scheme is **symmetric**: the float range becomes $[-\alpha_{max}, \alpha_{max}]$ and the formula collapses to a plain scale-and-round with no offset:

$$  
W_q = \text{clip}\left(\text{round}\left(\frac{W}{S}\right), -2^{b-1}, 2^{b-1}-1\right), \qquad S = \frac{\alpha_{max}}{2^{b-1}-1}  
$$

The payoff is that there is no zero-point add or subtract at inference, so the integer matmul is simpler and a bit faster. **Asymmetric** keeps $Z \neq 0$ and can represent lopsided distributions exactly, at the cost of carrying that offset around.

> **When do you actually need $Z$?** Weight distributions in a trained network are close to zero-centered, so a symmetric range wastes almost nothing, and weights are usually quantized symmetrically. Activations after a ReLU or GELU are one-sided (all non-negative), and if you force a symmetric range around zero you throw away half your integer slots on values that never occur. That is why activations often have to be asymmetric.

## What exactly are we quantizing?

This is the part people gloss over and then get confused by. There are three separate things you can quantize, and they have completely different difficulty and payoff:

- **Weights.** Static, known ahead of time, quantized once, offline. This is the easy one and the one that shrinks the model on disk and in memory.
- **Activations.** The intermediate tensors flowing between layers, different for every input, so you either calibrate them offline or compute their scale on the fly. Quantizing activations is what lets you do the matmul itself in low precision (real INT8 tensor-core math), not just store weights small.
- **KV cache.** The keys and values you store for every past token during generation. At long context this cache can dwarf the weights in memory, so quantizing it is its own topic (whole section on it below).

The industry shorthand is **WxAy**, meaning $x$-bit weights and $y$-bit activations, sometimes extended with **KVz** for the cache. So:

- **W8A8** is 8-bit weights and 8-bit activations, a true integer matmul, good for compute-bound serving.
- **W4A16** is 4-bit weights but 16-bit activations, which means the weights are stored at 4-bit and dequantized back to FP16 right before the matmul, which then runs in FP16. This helps memory-bound decode (fewer weight bytes to read) but does not speed up the math itself. Most local LLM quantization (GPTQ, AWQ, GGUF, NF4) is W4A16.
- **W8A8KV8** adds an 8-bit KV cache on top.

Keep this notation in your head. Half of "why did my quantized model get faster (or not)" is answered by which letters you actually touched.

## Granularity: per-tensor, per-channel, per-group

The scale $S$ does not have to be one number for the whole tensor. You can slice it finer:

- **Per-tensor**: one $S$ for the entire weight matrix. Cheapest to store, worst accuracy, because a single outlier anywhere forces a coarse scale on everything.
- **Per-channel** (a.k.a. per-axis): one $S_c$ per output channel (per row of the weight matrix), $S_c = \dfrac{\max_i |W_{c,i}|}{2^{b-1}-1}$. An outlier in one channel now only degrades that channel.
- **Per-group / per-block**: one $S$ per contiguous chunk of, say, 32 or 128 weights inside a channel. This is the sweet spot at 4-bit.

| Granularity          | Extra storage           | Where it shows up                                                                                       |
| -------------------- | ----------------------- | ------------------------------------------------------------------------------------------------------- |
| Per-tensor           | 1 scale per tensor      | INT8 on well-behaved layers, TensorRT default for many CNNs                                             |
| Per-channel          | 1 scale per output row  | The standard baseline for INT8 weight quantization                                                      |
| Per-group (e.g. 128) | 1 scale per 128 weights | What GPTQ, AWQ, and GGUF all use at 4-bit, because per-tensor INT4 on LLM weights is basically unusable |

There is a real tradeoff here that people forget: finer granularity means more scale values to store, and those scales are usually FP16. A group size of 128 at 4-bit adds one FP16 scale per 128 weights, which is about $16/128 = 0.125$ extra bits per weight. That is why QLoRA does *double quantization* (quantizing the scales themselves) and why GGUF quantizes its super-block scales. The metadata is not free.

## Dynamic vs static quantization

**Static** means you compute the scale once, offline, from a calibration pass, and bake it in. **Dynamic** means you compute it at runtime from the actual min/max of the tensor you are looking at right now.

Weights are always static, because they do not change at runtime, so there is no reason not to precompute. Activations are the interesting case. Dynamic activation quantization (compute a fresh scale per token, per forward pass) needs no calibration set and adapts to whatever comes in, at the cost of a small per-token reduction to find the max. Static activation quantization is faster at runtime but relies on your calibration data matching production traffic. In practice a lot of W8A8 serving uses **dynamic per-token** activation quantization precisely because it sidesteps the calibration-mismatch problem.

## Rounding: nearest, stochastic, and learned

The obvious rounding is **round-to-nearest (RTN)**: $2.4$ becomes $2$, deterministic, done. It is what almost every one-shot PTQ conversion uses.

**Stochastic rounding** rounds up with probability equal to the fractional part, so $P(\lceil x\rceil) = x - \lfloor x \rfloor$, which makes it unbiased in expectation, $E[\text{round}_{stoch}(x)] = x$. RTN has a small systematic bias on any individual value that stochastic rounding removes. This matters much more for low-precision *training*, where the same value gets rounded thousands of times and the bias would pile up, than for one-shot PTQ where you round each weight once.

Then there is the non-obvious one. Round-to-nearest is *not actually optimal* for the model's final loss, and **AdaRound** proved it. The idea: for each weight, decide whether to round up or down by asking which choice hurts the layer's output less, rather than blindly picking the nearer grid point. AdaRound frames this as a per-layer optimization with a soft, learnable rounding mask and a regularizer that pushes each mask toward a hard up/down decision, minimizing the reconstruction error $WX - \hat W X^2$. It is a small idea with a big consequence: rounding direction is a degree of freedom you can optimize, and several later methods (including the spirit of GPTQ) are variations on "round smart, not near."

## Quantization error behaves like noise (SQNR)

Model the rounding error $e = W - \hat W$ as roughly uniform on $[-S/2, S/2]$. This holds when values are not piled up on grid boundaries and not dominated by a few outliers. Its variance is:

$$  
\sigma_e^2 = \frac{S^2}{12}  
$$

Define the **signal-to-quantization-noise ratio** in decibels, comparing the power of the signal to the power of the rounding noise:

$$  
\text{SQNR}*{dB} = 10\log*{10}\left(\frac{\sigma_{signal}^2}{\sigma_e^2}\right)  
$$

Each extra bit doubles the number of grid points, which halves $S$ and quarters $\sigma_e^2$, and that works out to about **+6 dB of SQNR per bit**. This is the classic "6 dB per bit" rule. (You will often see it written as $\approx 6.02b + 1.76$ dB. The $1.76$ constant comes from assuming a full-scale sine wave, which is an ADC-theory assumption that does not automatically hold for a pile of weights, so trust the per-bit slope and treat the intercept as situational.)

The punchline that carries through the rest of the note: quantization is literally adding structured noise to your weights and activations. You cannot make the noise go away, you can only decide where it lands. GPTQ, AWQ, SmoothQuant, and the rest are four different strategies for steering that noise away from the numbers the model actually depends on.

---

# Why LLMs are secretly hard: the outlier problem

Quantization is old. People shipped INT8 convolutional nets on phones years before LLMs. So why did 8-bit break large transformers when it worked fine on a ResNet?

The answer, from Dettmers' LLM.int8() work, is **emergent outlier features**. Once a transformer crosses roughly 6.7B parameters, a small number of feature dimensions in the activations start taking on huge magnitudes, sometimes 20x larger than everything around them, and they appear in the *same* few dimensions across almost all tokens. These outlier dimensions are not noise. They are load-bearing, and if you quantize straight through them, the outliers stretch the scale so far that all the normal-sized values collapse into a tiny handful of grid slots, and the model's quality falls off a cliff.

This one phenomenon explains most of the algorithm zoo. Every method below is, at some level, a different way of dealing with outliers:

- **LLM.int8()** isolates the outlier dimensions and keeps them in FP16 while quantizing the rest to INT8 (mixed-precision decomposition).
- **SmoothQuant** mathematically shifts the outlier magnitude out of the activations and into the weights, where it is easier to handle.
- **AWQ** protects the weight channels that line up with large activations.
- **SpQR, QuIP#** keep the worst weights in high precision or rotate the whole space so no dimension is an outlier anymore.

Hold onto the outlier picture. It is the "why" behind almost everything that follows.

---

# The precision format zoo

Before the algorithms, the number formats they target. A format is defined by how it splits its bits between sign, exponent (dynamic range), and mantissa (precision).

| Format             | Bits      | Layout (S/E/M) | Notes                                                                                          |
| ------------------ | --------- | -------------- | ---------------------------------------------------------------------------------------------- |
| FP32               | 32        | 1 / 8 / 23     | Old training default, rarely used for inference now                                            |
| TF32               | 19        | 1 / 8 / 10     | NVIDIA's internal tensor-core format, FP32 range with less mantissa                            |
| FP16               | 16        | 1 / 5 / 10     | Narrow exponent, needs loss scaling in training to avoid underflow                             |
| BF16               | 16        | 1 / 8 / 7      | Same range as FP32 with a truncated mantissa, no loss scaling needed, now the training default |
| FP8 E4M3           | 8         | 1 / 4 / 3      | More precision, less range, used for weights and forward activations                           |
| FP8 E5M2           | 8         | 1 / 5 / 2      | More range, less precision, used for gradients in the backward pass                            |
| FP6 / FP4 (E2M1)   | 6 / 4     | 1 / 2 / 1      | Emerging on Blackwell-class hardware, still floating point                                     |
| INT8 / INT4 / INT2 | 8 / 4 / 2 | fixed point    | INT8 is mainstream, INT4 is where the 4-bit zoo lives, INT2 is mostly research                 |

Two things are worth understanding beyond the table.

**Integer vs float at the same bit-width.** INT8 spaces its levels evenly. FP8 spaces them logarithmically (dense near zero, sparse in the tails) because of the exponent. That makes FP8 naturally better at absorbing outliers, which is a big reason FP8 became the preferred *serving* format on Hopper (H100) and later hardware: it gives you near-INT8 memory savings with much less of the outlier pain, and the hardware has native FP8 tensor cores so the matmul is genuinely faster too.

**MX / microscaling formats.** The newer OCP "MX" formats (MXFP8, MXFP6, MXFP4) are block formats: a block of 32 elements shares one small power-of-two scale (an 8-bit exponent, E8M0), and each element is a tiny float like FP4 E2M1. This is basically "per-block scaling, standardized into the number format itself and supported in hardware," and it is where Blackwell-era low-precision inference is heading. If per-group quantization is a software convention, MX is that convention baked into silicon.

### NF4: a grid shaped like the weights

NF4 (NormalFloat4), from the QLoRA paper, starts from a real observation: pretrained weights are close to zero-mean Gaussian. A uniform 4-bit grid wastes slots out in the tails where almost no weights live, and is too coarse near zero where most of them live. NF4 instead places its 16 levels at the **quantiles of a standard normal**, so each slot holds an equal amount of probability mass rather than an equal slice of the number line. The codebook levels are:

$$  
q_i = \frac{1}{2}\Big[Q\left(\tfrac{i}{17}\right) + Q\left(\tfrac{i+1}{17}\right)\Big], \qquad i = 0, \dots, 15  
$$

where $Q$ is the quantile function (inverse CDF) of $N(0,1)$. You cut the normal into 17 equal-probability boundaries and take the midpoint of each pair as a codebook value. This is information-theoretically optimal for a Gaussian source, i.e. no uniform 4-bit grid can encode more information about normally-distributed weights. QLoRA pairs NF4 with blockwise absmax normalization (each block gets its own scale so its values sit in $[-1, 1]$ before mapping to the codebook) and **double quantization**, which quantizes those per-block scales too, saving roughly another 0.4 bits per weight.

### Ternary and binary: as low as it goes

BitNet b1.58 pushes weights down to three values, $-1, 0, +1$, using an absmean scale:

$$  
Q_w(W) = \Delta \cdot \text{RoundClip}\left(\frac{W}{\Delta + \epsilon}, -1, 1\right), \qquad \Delta = \text{mean}(|W|)  
$$

with $\text{RoundClip}(x, a, b) = \max(a, \min(b, \text{round}(x)))$, $\Delta$ the mean absolute weight, and $\epsilon$ a small guard against dividing by zero. Three states need $\log_2 3 \approx 1.58$ bits, which is the "1.58" in the name. The important caveat is that BitNet is not a post-hoc squeeze of an existing model. It is trained at this precision from scratch, which is a different regime (QAT, covered below), because you cannot take a normally-trained model down to ternary after the fact without it falling apart. Pure 1-bit (binary, $-1, +1$) is the original BitNet and is mostly a research frontier.

---

# Calibration: turning data into scales

Most PTQ methods, and all static activation quantization, need **calibration data**: a small, unlabeled sample of representative inputs that you run through the model once to see how the numbers actually behave. You are not training on it, you are measuring with it. For LLM PTQ the usual amount is on the order of 128 to 512 sequences from a general corpus like C4 or WikiText.

How you turn those measurements into a scale is a real design choice, and there is a ladder of increasingly clever options:

- **Min-max.** Take the literal min and max. Simplest, fastest, and the most fragile, because one outlier sets the range for everything.
- **Percentile clipping.** Throw away the extreme 0.1% before taking min/max. You give up representing the true extremes in exchange for a tighter grid on the bulk of the values.
- **MSE-optimal clipping.** Search over candidate clip ranges and pick the one that minimizes reconstruction error directly, instead of trusting raw min/max.
- **KL / entropy calibration.** This is TensorRT's classic INT8 recipe: build a histogram of the FP32 activations, then find the clipping threshold whose quantized distribution has the smallest KL divergence from the original FP32 distribution. In other words, pick the range that distorts the *shape* of the distribution least, not just the one that fits the extremes.
- **Second-order (GPTQ).** Accumulate the full Hessian $XX^\top$ over the calibration batch, which captures how the layer's output responds to each weight, not just a range.
- **Activation-magnitude (AWQ).** Accumulate per-channel activation averages to find which weight channels are salient.
- **Importance matrix (GGUF imatrix).** llama.cpp can compute an "imatrix" from calibration data that weights the quantization error of each dimension by how much it actually contributes to activations, and it noticeably improves low-bit k-quants. So the common claim that "GGUF needs no calibration" is only true for the basic k-quants. The good ones increasingly use an imatrix.

> **The calibration-mismatch trap.** If you calibrate on generic web text, your model will look great on perplexity, because perplexity is measured on data that looks like your calibration set. Then it quietly degrades on code, math, long-context reasoning, or your specific domain, because the calibration set never exercised those. Low perplexity on the calibration distribution is not evidence of quality on your distribution. Always re-check on the real task. I cannot say this loudly enough, it is the single most common way people ship a broken quantized model without noticing.

---

# A minimal implementation, by hand

Real kernels are CUDA and C++, but the logic is exactly the affine map. Here it is in plain NumPy, quantizing a toy weight matrix per-tensor and per-channel so you can watch the granularity tradeoff turn into an actual number:

```python
import numpy as np

def quantize(W, bits=8, per_channel=False):
    q_min, q_max = -(2 ** (bits - 1)), 2 ** (bits - 1) - 1
    if per_channel:
        # one scale per output row (reduce over each row's columns)
        amax = np.max(np.abs(W), axis=1, keepdims=True)
    else:
        amax = np.max(np.abs(W))
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

That ~17 dB gap, from a single bad row, is the whole reason per-channel and per-group exist. Real LLM weights have far more and far worse outliers than this toy, which is exactly why plain min-max is not enough at 4-bit and why the algorithms in the next section had to be invented.

---

# The post-training quantization zoo

PTQ takes a frozen, already-trained model and quantizes it with no retraining, no gradients through the whole network, and calibration measured in minutes to hours instead of GPU-days. This is where nearly all the research effort has gone, because it is what you can actually do to someone else's open-weights checkpoint on one machine.

## LLM.int8(): keep the outliers in FP16

The first method that made 8-bit work for large models, and the cleanest illustration of the outlier idea. It uses **vector-wise quantization** (a separate scale per row of the activations and per column of the weights) and then does something clever: it detects the handful of outlier feature dimensions, pulls those specific columns out, and computes them in FP16, while the other 99%+ of the matmul runs in INT8. The two partial results are added back together. So it is a **mixed-precision decomposition**: almost all the compute is 8-bit, but the load-bearing outliers never get quantized. This is exactly what `load_in_8bit=True` does in bitsandbytes. It is nearly lossless, but the FP16 side path means it does not always speed things up, it is mostly a memory play.

## GPTQ: round using second-order information

GPTQ treats quantization as a per-layer least-squares problem, not a per-weight rounding rule. Given calibration inputs $X$ to a linear layer and its weights $W$, it wants the quantized $\hat W$ that changes the layer's *output* the least:

$$  
\hat{W} = \arg\min_{\hat{W}}  WX - \hat{W}X_2^2  
$$

The subtlety is that it does not quantize each weight independently. When it rounds one weight and introduces an error, it *adjusts the weights it has not quantized yet* to cancel that error out.

**The lineage.** This comes from Optimal Brain Surgeon (OBS), originally a pruning method, adapted to quantization as Optimal Brain Quantization (OBQ). OBQ quantizes one weight at a time and, after each, nudges every remaining weight in that row to compensate, using the layer's Hessian $H = 2XX^\top$. That Hessian is the second-order term of the Taylor expansion of the squared error around $W$, and notably it depends only on the inputs $X$, so the whole output layer shares it. OBQ's single-weight update and the error it costs are:

$$  
\delta = -\frac{w_q - \text{quant}(w_q)}{[H^{-1}]*{qq}} H^{-1}*{:,q}, \qquad \varepsilon_q = \frac{\big(w_q - \text{quant}(w_q)\big)^2}{[H^{-1}]_{qq}}  
$$

OBQ greedily quantizes whichever weight has the smallest $\varepsilon_q$ next, which is accurate but far too slow at billions of parameters. GPTQ makes it feasible with three moves:

1. **Fixed order.** Just quantize columns left to right in a fixed order instead of re-sorting by error every step. At LLM scale this is almost as good and hugely cheaper. (The optional `act_order` / `desc_act` flag brings back a smarter ordering, quantizing the most important columns first, for a bit more accuracy.)
2. **Cholesky reformulation.** Precompute the Cholesky factorization of $H^{-1}$ once, up front. All the row-by-row compensation terms drop out of that factor, so you never repeatedly invert a shrinking Hessian, which is where the naive version becomes numerically unstable.
3. **Lazy batch updates.** Quantize a block of columns (say 128) using the Cholesky factor, accumulate their corrections, and apply the whole thing to the rest of the matrix as one big matmul, instead of many tiny vector updates. GPUs are good at big matmuls and bad at many small updates, so this is what makes GPTQ run in minutes.

```
 weight matrix, one row = one output neuron:

 col:   0   1   2   3 | 4   5   6   7 | ...
       [q] [q] [q] [q]|[·] [·] [·] [·]| ...
        └── block 1 ──┘
        quantize this block, accumulate the
        compensation, then push it onto every
        remaining column with one matmul ───────►
```

> **Why is this fast if the math looks so heavy?** Because GPTQ never runs a backward pass or touches the training loss. It is a forward pass over calibration data plus some linear algebra per layer. It is solving a small local least-squares problem for each layer, not optimizing the whole network end to end. That is the trick to the whole PTQ family: stay local, stay gradient-free.

## AWQ: protect the weights the activations care about

AWQ starts from an empirical claim: in a trained LLM, well under 1% of weight channels dominate the output quality, and you can find them not by looking at weight magnitude (the naive guess) but by looking at **activation magnitude**. Channels that consistently see large activations are the ones whose weight precision matters.

The move that makes protecting them cost nothing extra is a plain algebraic identity. For a linear layer $Y = XW$, scale a channel's weights up by $s$ and divide the matching activations by $s$:

$$  
Y = \big(X \cdot \text{diag}(s)^{-1}\big)\big(\text{diag}(s) \cdot W\big)  
$$

The output is unchanged, but the scaled-up weight channel now has a smaller *relative* rounding error, because the same absolute quantization step is a smaller fraction of a bigger number. AWQ searches for the per-channel scale as $s = (\bar{|X|})^\alpha$, grid-searching $\alpha \in [0,1]$ on a little calibration data to minimize output error. Crucially the $\text{diag}(s)^{-1}$ on the activation side is not a runtime multiply, it is folded into the previous layer's weights (or an adjacent LayerNorm) at export time. So AWQ adds zero inference overhead and stores nothing in mixed precision, it just quantizes 4-bit weights that happen to be pre-scaled. That is why it is fast to apply and holds quality well, and why you see it everywhere on served models.

## SmoothQuant: move the difficulty from activations to weights

GPTQ and AWQ are weight-only (W4A16). SmoothQuant is after the harder prize: full **W8A8**, where you quantize activations too and get a genuine integer matmul for compute-bound serving. The obstacle, again, is activation outliers. Weights are flat and easy, activations have those few monstrous channels. SmoothQuant uses the same identity as AWQ but aimed the other way, smoothing the activations by pushing some of their range into the weights, which tolerate it better:

$$  
s_j = \frac{\max(|X_j|)^\alpha}{\max(|W_j|)^{1-\alpha}}, \qquad Y = \big(X \cdot \text{diag}(s)^{-1}\big)\big(\text{diag}(s) \cdot W\big)  
$$

The exponent $\alpha$ is the "migration strength": bigger $\alpha$ dumps more difficulty onto the weights, smaller $\alpha$ leaves more on the activations, and $\alpha = 0.5$ is the usual sweet spot, which simplifies to $s_j = \sqrt{\max(|X_j|)/\max(|W_j|)}$. The one-line way to keep AWQ and SmoothQuant straight: AWQ is weight-only and protects salient weight channels for memory savings, SmoothQuant is weight-and-activation and rebalances the joint problem for throughput.

## OmniQuant: learn the clipping and the smoothing

OmniQuant is the "make the knobs learnable" step. Instead of hand-picking clip ranges and smoothing factors, it makes the weight clipping thresholds (learnable weight clipping) and the equivalence transform (learnable equivalent transformation) into trainable parameters, and optimizes them block by block with gradient descent against the reconstruction error. It is still cheap because it only trains those few extra parameters per block, not the model, but it consistently squeezes out more accuracy at low bit-widths than the hand-tuned methods.

## Going below 4 bits: SpQR, QuIP#, AQLM

At 2 to 3 bits, uniform quantization stops working and you need cleverer representations. These are the frontier, worth knowing by name and idea:

- **SpQR (Sparse-Quantized Representation).** Finds the small set of outlier weights that cause most of the error and stores *those* in high precision as a sparse side matrix, while the rest go to 3 to 4 bits. It is the weight-side analogue of LLM.int8()'s activation-side outlier isolation.
- **QuIP and QuIP#.** Built on "incoherence processing": multiply the weights and Hessian by random orthogonal or Hadamard rotations so that no single coordinate is an outlier anymore (you spread the difficulty evenly across all dimensions), which makes even 2-bit viable. QuIP# adds a lattice codebook (the E8 lattice) to pack the rotated weights efficiently. This is a genuinely different idea from everything above: instead of protecting outliers, rotate the space until there are none.
- **AQLM (Additive Quantization of Language Models).** Represents each weight vector as a sum of vectors picked from several learned codebooks, which is vector quantization rather than scalar quantization. It reaches 2-bit with surprisingly little loss, at the cost of a heavier, slower quantization process.

The through-line: scalar uniform grids run out of room below 4-bit, so you either keep the worst weights in high precision (SpQR), rotate the outliers away (QuIP#), or switch to codebooks (AQLM).

## GGUF / llama.cpp k-quants: how most people actually run models locally

If you have run a model on your laptop through Ollama or LM Studio, you have used this, because they wrap llama.cpp. GGUF's k-quants use a hierarchical block layout: weights are grouped into **super-blocks of 256 values**, each super-block holds several sub-blocks with their own scale and min, and the super-block's own scale/min are themselves quantized. That last part is a double-quantization trick to keep the per-sub-block metadata cheap.

The naming looks cryptic but is simple once you decode it. `Q4_0` and `Q4_1` are the old flat formats (one scale per block, no super-block structure). The `_K` formats are the k-quants, and the `_S` / `_M` / `_L` suffix is how much mixed precision they use: the `_M` and `_L` variants keep certain error-sensitive tensors (commonly the attention value projection and the second feed-forward matrix) at a higher bit-width than the rest. `Q4_K_M` sits around 4.8 effective bits per weight and is the default "good for almost everything" choice.

| Quant    | ~bits/weight | Quality vs FP16             | When you reach for it                |
| -------- | ------------ | --------------------------- | ------------------------------------ |
| `Q8_0`   | ~8.5         | Essentially lossless        | You have the VRAM and want zero risk |
| `Q6_K`   | ~6.6         | Very small loss             | High-quality local inference         |
| `Q5_K_M` | ~5.7         | Small loss                  | Good balance on mid-range hardware   |
| `Q4_K_M` | ~4.8         | Noticeable but usually fine | The default for most local setups    |
| `Q3_K_M` | ~3.9         | Real degradation            | Tight VRAM                           |
| `Q2_K`   | ~2.6         | Significant                 | Last resort when nothing else fits   |

Basic k-quants need no calibration, the block stats come straight from the weights, which is why you can convert a model to GGUF in minutes with nothing but the checkpoint. The higher-quality path uses an imatrix from calibration data (see the calibration section) and does measurably better at the low end.

---

# Quantization-aware training (QAT)

Everything above operates on a frozen model. QAT instead trains the model to expect the rounding it will face at inference, which is the only real option once you go aggressive enough that PTQ can no longer recover the accuracy.

## Fake quantization and the straight-through estimator

QAT inserts a "fake quantize" op into the forward pass, the same round-clip-dequantize you have seen, and computes with $\tilde W$ instead of $W$, while the parameter you actually optimize stays full-precision $W$:

$$  
\tilde{W} = S\Big(\text{clip}\big(\text{round}(W/S) + Z, q_{min}, q_{max}\big) - Z\Big)  
$$

The problem is that $\text{round}(\cdot)$ is a staircase with zero gradient almost everywhere, so backprop has nothing to work with. The **straight-through estimator (STE)** fixes this by pretending, on the backward pass, that the rounding was the identity function inside the representable range and flat outside it:

$$  
\frac{\partial L}{\partial W} \approx \frac{\partial L}{\partial \tilde{W}} \cdot \mathbb{1}[\alpha \le W \le \beta]  
$$

> **Why does such a crude lie work?** Because rounding is a small local perturbation, and for most values $\text{round}(x)$ is close to $x$. Treating it as the identity lets gradient signal flow through, and the optimizer ends up finding weights that are *robust to being rounded*, which was the goal the whole time. You are not trying to differentiate the staircase, you are trying to nudge weights into positions where the staircase does not hurt.

A refinement worth knowing is **LSQ (Learned Step Size Quantization)**, which makes the scale $S$ itself a trainable parameter learned by gradient descent (with a carefully chosen gradient scaling), rather than fixed from calibration. Letting the model learn its own quantization grid is a reliable accuracy win in QAT.

## When QAT is worth it

QAT needs your full training pipeline, real data, and real compute. It is training, not a ten-minute post-process. So the honest answer is that most people should not use it: for "take an open checkpoint and shrink it," PTQ (GPTQ, AWQ, GGUF) is enough, which is exactly why those dominate. QAT earns its cost in three cases: very aggressive bit-widths where PTQ has run out of room (sub-4-bit), edge and mobile deployment where you control training anyway and every bit of accuracy matters, and native-low-precision models like BitNet that are trained ternary from scratch because there is no post-hoc path to 1.58 bits.

---

# KV-cache quantization

This one deserves its own section because at long context it is often the thing that actually kills you, not the weights.

During generation you cache the key and value tensors for every past token so you do not recompute them. That cache grows linearly with context length and batch size, and at 100K+ tokens it can be larger than the model weights themselves. Quantizing it is how you fit long context and large batches in memory.

The catch is that keys and values have different structure, so you treat them differently. The finding from KVQuant and KIVI is that the **key** cache has outlier *channels* (specific dimensions that are consistently large, so you quantize it per-channel) while the **value** cache is better behaved and quantizes fine per-token. KIVI pushes this to 2-bit KV cache with that asymmetric treatment. In production, the simpler and very common option is an **FP8 KV cache** (supported in vLLM and TensorRT-LLM), which roughly halves cache memory with almost no quality loss and no fancy per-channel logic. If you are chasing extreme context lengths, the specialized 2-bit methods buy you more, at more complexity.

The mental model: weight quantization shrinks the model, KV-cache quantization shrinks the *conversation*, and for long-context or high-batch serving the second one is frequently the bigger lever.

---

# What actually gets faster (and what does not)

This trips people up constantly, so let me be blunt about it.

**Weight-only quantization (W4A16)** stores weights small and dequantizes them back to FP16 right before the matmul. The matmul runs in FP16. So you save memory and you speed up **memory-bound decode** (fewer weight bytes to stream per token), but you do *not* speed up **compute-bound prefill**, and you add a small dequantization cost. This is GPTQ, AWQ, GGUF, and NF4. Great for local inference and for decode-heavy chat.

**Weight-and-activation quantization (W8A8)** quantizes activations too and runs a real integer (or FP8) matmul on the tensor cores. This speeds up the actual math, so it helps prefill and compute-bound, high-throughput serving. This is SmoothQuant, FP8, and INT8 serving in TensorRT-LLM and vLLM. It is harder to pull off because of the activation-outlier problem, which is the whole reason SmoothQuant and FP8 exist.

So "I quantized my model, why is prefill the same speed" has a precise answer: you did W4A16, and prefill is compute-bound, and you only touched the memory side. Match the letters (WxAy) to your bottleneck (decode vs prefill) and the confusion goes away.

One more reality check: the memory savings are rarely the clean 4x the bit-width suggests. Group scales, zero-points, the occasional high-precision tensor, and the KV cache all sit outside the neatly-quantized weights. A "4-bit" model is usually more like 4.5 to 5 effective bits once you count the metadata.

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
    bnb_4bit_compute_dtype="bfloat16",  # dequantize to bf16 for the matmul
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

For serving, **vLLM** and **TensorRT-LLM** load GPTQ/AWQ/FP8 checkpoints directly and run them at scale, and TensorRT-LLM is where you would do INT8/FP8 W8A8 with SmoothQuant-style calibration for maximum throughput.

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

- **GGUF community builds** of Llama, Mistral, Qwen, and friends on Hugging Face, `Q4_K_M` as the default download, run through llama.cpp, Ollama, or LM Studio. This is most local usage.
- **GPTQ and AWQ checkpoints**, tagged `-GPTQ` or `-AWQ` on the hub, served through vLLM or TensorRT-LLM at INT4 for GPU inference.
- **NF4 + QLoRA**: the original QLoRA paper's Guanaco models are the canonical example, a frozen NF4 base with LoRA adapters fine-tuned on a single consumer GPU. This is how most people fine-tune large models cheaply.
- **FP8 on H100**: increasingly the default *serving* format in vLLM, TensorRT-LLM, and NVIDIA NIM, near-FP16 quality at half the memory with native tensor-core speedups.
- **BitNet b1.58**: Microsoft's natively-ternary models, the frontier case for where extreme compression is headed, trained low-precision from scratch rather than squeezed after the fact.

---

# Evaluating a quantized model

You have to check the damage, and you have to check it the right way, because the easy metric lies.

**Perplexity** (on WikiText2, C4, and so on) is cheap and standard and a *weak* signal. It measures average next-token fit, not whether the model can still reason, code, or follow instructions. A model can hold perplexity and still lose real capability.

**Downstream task evals** (MMLU, GSM8K, HumanEval, or your own task suite) are the real signal. Run them. Do not skip them because perplexity looked fine.

**KL divergence** between the full-precision and quantized model's output distributions is the sharpest task-agnostic diagnostic:

$$  
D_{KL}\big(P_{fp16}  P_{quant}\big) = \sum_i P_{fp16}(i) \log\frac{P_{fp16}(i)}{P_{quant}(i)}  
$$

computed per-token over a held-out set. It catches distributional drift that a single averaged perplexity number smears over. If the quantized model is putting its probability mass in noticeably different places than the original, this sees it even when perplexity does not move.

And the caveat I flagged in [Inference Engineering](/notes/LLM-And-Agents/InferenceEgineering): quantization quality is workload-dependent. A model that benchmarks fine at INT4 can fall apart on long-context reasoning or code specifically. Evaluate on *your* task, not just perplexity.

---

# Quantization beyond LLMs

LLMs get all the attention now, but quantization grew up in vision and edge deployment, and that world is worth a paragraph because the ideas transfer and the tooling is mature.

On phones and embedded devices, INT8 CNNs have been standard for years. **TensorFlow Lite** does post-training INT8 (per-axis weights, activations calibrated from a small representative dataset), plus lighter "dynamic range" (weights-only) and FP16 modes. **PyTorch** has a full stack: the older eager-mode flow with `QuantStub`/`DeQuantStub` and observers, FX graph-mode quantization, and the newer PT2 export path, running on backends like FBGEMM (server x86), QNNPACK/XNNPACK (mobile ARM). **ONNX Runtime** has its own PTQ and QAT tooling. NVIDIA's classic INT8 CNN recipe (the KL-divergence calibration described earlier) came out of exactly this world.

The differences from the LLM case are instructive. Vision models are mostly compute-bound and quantized to INT8 W8A8 to speed up the actual convolutions, whereas LLM decode is memory-bound and often quantized weight-only. Vision models rarely have the dramatic activation outliers that plague large transformers, so plain per-channel INT8 usually just works without the GPTQ/AWQ/SmoothQuant machinery. If you understand the LLM story, the vision story is the easier special case.

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

# The mental model

If you forget everything else, keep these:

- Quantization is a linear map from real numbers onto a grid. Scale sets the spacing, zero-point aligns it, and every method is a smarter way to choose the range or the rounding than plain min-max.
- The error is noise you cannot delete, only steer. You get roughly 6 dB of headroom per bit, and the whole craft is aiming that noise away from the numbers the model depends on.
- LLMs are hard because of a few huge outlier features, and almost every method is a different way of dealing with those outliers: isolate them (LLM.int8, SpQR), migrate them (SmoothQuant), protect around them (AWQ), or rotate them away (QuIP#).
- GPTQ rounds using second-order information, calibration-only and gradient-free, which is why it is both accurate and fast.
- Match the WxAy to your bottleneck: weight-only (W4A16) speeds up memory-bound decode, weight-and-activation (W8A8, FP8) speeds up compute-bound prefill and throughput.
- KV-cache quantization shrinks the conversation, not the model, and at long context it is often the bigger win.
- QAT teaches a model to survive rounding by training through it with the straight-through estimator, and it is worth the cost only at the aggressive end.

And the decision it all collapses to. Serving on GPU at scale, reach for FP8, AWQ, or GPTQ through vLLM or TensorRT-LLM. Running locally, reach for GGUF `Q4_K_M`. Fine-tuning cheaply, reach for QLoRA and NF4. Going below 4-bit, reach for the codebook and rotation methods and expect to work for it.
