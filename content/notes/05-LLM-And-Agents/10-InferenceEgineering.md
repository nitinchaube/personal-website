---
title: "Inference Engineering"
date: 2026-06-22
summary: "Notes on what it actually takes to serve LLMs in production: the prefill/decode split, the KV cache, continuous batching, FlashAttention and PagedAttention, quantization, speculative decoding, multi-GPU sharding, prefill-decode disaggregation, and the MLOps around all of it."
tags: [Inference, LLM, Serving, vLLM, KV-Cache, Quantization, GPU, MLOps, Performance]
---

A model you trained once runs in production millions of times, and every one of those runs costs GPU-seconds and shows up on someone's latency dashboard. The gap between a research checkpoint and a service that holds a p99 latency SLA at a few thousand requests per second is enormous, and almost none of it is about the model. It's about how you feed the GPU.

The thing that took me a while to internalize: serving an LLM is not like serving a normal web service. A normal service is mostly stateless request/response, CPU- or network-bound and you scale it by adding boxes more or less linearly. An LLM generates one token at a time, autoregressively and each token requires reading the *entire* model out of memory. That single fact is autoregression on top of multi-gigabyte weights bends every rule you know about scaling services. The bottleneck isn't compute. Most of the time it's memory bandwidth and the engineering is a long fight to stop wasting it.

These are my notes on how production inference actually works: the two phases, the caches, the batching tricks, the compression, the sharding and the operational scaffolding.

---

# Prefill and Decode are two different machines

Everything downstream makes sense once you see that a single generation request runs in two phases with completely opposite hardware behavior.

```
PROMPT: "Write a haiku about GPUs"
        └──────────── PREFILL ────────────┘   └──────── DECODE ────────┘
        process all prompt tokens at once       generate one token at a time
        (one big forward pass)                  (one forward pass PER token)
              │                                        │   │   │   │
              ▼                                        ▼   ▼   ▼   ▼
        first token ("Silicon") ───────────────▶   "dreams" "in" "warm" ...
              ▲                                        ▲
            TTFT                                      ITL between each token
```

- **Prefill** takes the whole prompt and runs it through the model in one shot. Because you're processing many tokens simultaneously, the math is a big matrix–matrix multiply: lots of arithmetic per byte of weight you loaded. The GPU's compute units stay busy. Prefill is **compute-bound** and it's what determines your **Time To First Token (TTFT)**.
- **Decode** generates the rest, one token at a time, each one conditioned on everything before it. To produce a *single* token you still have to stream every weight in the model from HBM (the GPU's main memory) down to the compute units but you only do a thin matrix–vector multiply with it. You pay the full cost of reading the weights and do almost no math with them. Decode is **memory-bandwidth-bound**, and it determines your **Inter-Token Latency (ITL)**, sometimes called Time Per Output Token (TPOT).

## Arithmetic intensity and the roofline

Every GPU has two ceilings:

1. How fast it can do math (FLOP/s) and
2. how fast it can move bytes from HBM (bytes/s).

Which ceiling you hit depends on **arithmetic intensity** i.e. FLOPs performed per byte loaded.

$$  
\text{Arithmetic Intensity} = \frac{\text{FLOPs in the operation}}{\text{bytes moved from HBM}}  
$$

If intensity is low, you're waiting on memory and the compute units idle. If it's high, you're actually using the compute. The "ridge point" of the roofline is just `(peak FLOP/s) / (peak bandwidth)`. On an H100 that ridge sits somewhere around a few hundred FLOPs/byte depending on precision.

- **Prefill** with a long prompt has high intensity (a matrix–matrix multiply reuses each loaded weight across many tokens). It lives on the compute-bound side of the ridge.
- **Decode at batch size 1** has an intensity of roughly **1–2 FLOPs/byte**. It's nowhere near the ridge and it's pinned to the memory wall. You could double the GPU's FLOPs and decode wouldn't get faster.

The most important corollary, and the reason batching exists is the way you raise decode's arithmetic intensity is to process **more sequences at once** with the *same* loaded weights. Load the weight once, use it for 64 sequences instead of 1, and you've roughly 64×'d your intensity for free. That single insight drives almost every throughput optimization in this document.

## Why the hardware specs matter

You don't need to memorize datacenter GPU sheets, but a couple of numbers anchor the intuition:

|                  | A100 80GB (SXM)   | H100 80GB (SXM)                         |
| ---------------- | ----------------- | --------------------------------------- |
| HBM bandwidth    | ~2.0 TB/s (HBM2e) | ~3.35 TB/s (HBM3)                       |
| Compute (dense)  | 312 TFLOPS (FP16) | ~990 TFLOPS (BF16), ~1,979 TFLOPS (FP8) |
| NVLink (per GPU) | 600 GB/s          | 900 GB/s                                |

Notice what changed from A100 to H100: bandwidth went up ~1.64×, which directly speeds up memory-bound **decode**, and FP8 compute jumped a lot, which speeds up compute-bound **prefill**. The two phases benefit from two different parts of the upgrade. That's the prefill/decode split showing up even in the hardware roadmap.

---

## Metrics that actually matter are:

Before optimizing anything, you have to agree on what you're measuring and averages will lie to you.

- **TTFT( Time To First Token)**: Dominated by prefill (and by queue wait before prefill even starts). This is what makes a chat feel responsive.
- **ITL / TPOT (Inter-Token Latency):** How fast tokens stream after the first. A human reads ~5–10 tokens/sec, so an ITL under ~100 ms feels smooth; agents and tool-callers want it much lower.
- **Throughput**: Total tokens/sec across all requests, or QPS at the macro level. This is the cost metric: it's how many users one GPU pays for.
- **Goodput**: The number people forget. It's throughput *that meets your SLO*. A server doing 10,000 tok/s while violating its TTFT target is producing zero goodput. Optimize for this, not raw throughput.

And always at **percentiles**, never means. A system can have a p50 TTFT of 100 ms and a p99 of 2,500 ms and your worst-served 1% of users *are* your power users hammering it. Tail latency is the real product. In any fan-out architecture (an agent making 10 parallel sub-calls), the slowest of the 10 sets the wall-clock, so the tail compounds.

A reasonable real-time chat SLO looks like: **p99 TTFT ≤ 400 ms, p99 ITL ≤ 50 ms.** These are the most important targets before you tune because every optimization below trades one of these against another.

---

# The central tension: latency vs throughput

Here's the conflict at the heart of serving. Batching is how you get throughput? share the weight-loading cost across many sequences. But batching is also how you *hurt* latency:

1. A bigger batch means each forward pass does more work, so every token takes longer so **ITL goes up** for everyone in the batch.
2. To *form* a big batch you sometimes hold early requests in a queue waiting for friends to arrive and **TTFT goes up**.

So the same lever that maximizes your dollar-efficiency degrades your responsiveness. There's no setting that wins both; there's only the operating point you choose.

### The queuing intuition (don't run hot)

It helps to think of the engine as a queue with arrival rate $\lambda$ and service rate $\mu$. The classic M/M/1 result for average time in system is:

$$  
T_s = \frac{1}{\mu - \lambda}, \qquad \rho = \frac{\lambda}{\mu}  
$$

The exact formula is a lie for LLMs, service times aren't exponential and batching means requests are served in interfering groups, not one at a time. But the *shape* is exactly right and worth burning into your head: as utilization $\rho \to 1$, wait time goes to infinity, **non-linearly**. The last 10% of utilization costs you the most latency you'll ever pay.

```
 latency
   ^
   |                                        *
   |                                      *      ← the wall: queue blows up
   |                                   *
   |                              *
   |                        *
   |             *  *  *
   |   *  *  *
   +-------------------------------------------> utilization ρ
   0%        50%        70%        90%   100%
                          ▲
                   run here, not past it
```

The practical takeaway: to hold a tight p99, you have to **cap utilization with headroom**, often around 70% so the system can absorb bursts without the queue exploding. The 30% you "waste" is what buys your tail latency. Engineers new to this try to run the fleet at 95% to save money and then can't understand why p99 is on fire.

---

# Batching:

This is where most of the throughput comes from and it has a clear evolution.

## 1) Static batching is a trap for generation

Static batching waits for N requests, runs them as a fixed group and returns when *all* are done. For fixed-size work (image classification) it's fine. For text generation it's terrible, because output lengths vary wildly. Batch a request that emits 10 tokens with one that emits 500 and the short one finishes early but its GPU slot stays *occupied and padded* until the 500-token straggler is done. You burn most of the batch on idle padding. GPU utilization can fall below ~25% of what the hardware can do.

```
Static batch (■ = real compute, · = wasted padding):
 req A (10 tok):  ■·····························  done at 10, idle for 490
 req B (500 tok): ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■
 req C (40 tok):  ■■■■·························   done at 40, idle for 460
                  └──────── everyone waits for B ────────┘
```

## 2) Continuous (iteration-level) batching

The fix, popularized by Orca and then vLLM, is to make batching decisions **every single token step** instead of per request. After each forward pass:

1. Any sequence that just emitted `<EOS>` is **evicted** immediately and its slot freed.
2. Waiting requests are **admitted** into the freed slots on the very next step.

The batch is now a living thing, sequences join and leave mid-flight. The GPU never sits idle waiting for a straggler, because the moment a slot opens it gets refilled. This is the single biggest throughput win in modern serving, often several-fold over static batching.

```
Continuous batch (slots refill as soon as they free up):
 slot 0:  A A A ✗ E E E E E E ...      (A finishes, E moves in)
 slot 1:  B B B B B B B B B B ...
 slot 2:  C C C C ✗ F F F F F ...      (C finishes, F moves in)
          └ decisions remade every token step ┘
```

Two knobs govern the dynamic batcher:

- `max_batch_size` : the hard ceiling, set by GPU memory and the latency you'll tolerate.
- `max_queue_delay` : how long you'll hold a request to let a batch grow. Small values favor TTFT and larger ones favor throughput.

Here's the scheduler loop in spirit. Treat it as illustrative, not production code, the real ones live in C++/CUDA and are far hairier but it shows the structure: an async loop that evicts finished work, refills from a queue under a delay deadline, and offloads the GPU pass off the event loop so tokenization doesn't block it.

```python
import asyncio, time

class ContinuousBatcher:
    def __init__(self, max_batch_size, max_queue_delay_ms):
        self.max_batch_size = max_batch_size
        self.max_queue_delay = max_queue_delay_ms / 1000.0
        self.waiting = asyncio.Queue()
        self.active = []

    async def submit(self, req):
        req["done"] = asyncio.Event()
        await self.waiting.put(req)
        await req["done"].wait()          # caller suspends until generation finishes
        return req["result"]

    async def loop(self):
        while True:
            # 1. drop finished sequences (iteration-level eviction)
            self.active = [r for r in self.active if not r.get("eos")]

            # 2. refill free slots, but don't wait past the delay deadline
            slots = self.max_batch_size - len(self.active)
            deadline = time.monotonic() + self.max_queue_delay
            while slots > 0:
                timeout = max(0.0, deadline - time.monotonic())
                if timeout == 0.0 and self.waiting.empty():
                    break
                try:
                    self.active.append(await asyncio.wait_for(self.waiting.get(), timeout))
                    slots -= 1
                except asyncio.TimeoutError:
                    break

            if not self.active:
                await asyncio.sleep(0.001)   # nothing to do; don't spin the CPU
                continue

            # 3. one forward pass for the whole batch, off the event loop
            await asyncio.to_thread(self.step, self.active)
            # step() appends one token to each sequence and sets r["eos"]
            # + r["done"].set() for any that hit <EOS>
```

## 3) Chunked prefill:

Continuous batching has a wrinkle. Prefill is one big compute-heavy pass; decode is many tiny ones. If a 4,000-token prompt arrives, naively prefilling it monopolizes the GPU for one long step and every other user's decode **stalls**, their ITL spikes. You see it as periodic stutter in the token stream whenever someone submits a long prompt.

**Chunked prefill** (Sarathi-Serve and now standard in vLLM) splits a long prefill into smaller chunks and interleaves them with ongoing decode tokens in the same batch. You give up a little raw prefill throughput in exchange for smooth, predictable ITL. It's one of those settings that doesn't change your average but dramatically tightens your tail.

## 4) Preemption:

There's a subtle failure the happy-path scheduler ignores: a batch is running, all sequences are still generating, and the KV cache (next section) fills up. You can't admit anyone, and you may not even be able to *continue* everyone. The engine has to **preempt** a victim sequence and either:

- **swap** its KV cache out to CPU RAM and bring it back later, or
- **recompute** it from scratch when it's rescheduled (throw the cache away, redo the prefill).

Recompute is often cheaper than you'd think because prefill is fast, and it avoids the PCIe round-trip of swapping. Either way, this is real machinery in vLLM and it's why "out of memory" under load shows up as a latency cliff rather than a crash.

## 5) The cascading queue failure

The nastiest production incident in serving comes from unbounded queues. Picture a traffic spike: the queue grows, TTFT climbs past the client's HTTP timeout (often 30 s), the client gives up  and **retries**. But the server never learned the original request was abandoned, so it keeps processing the orphan *and* takes the retry. Effective load multiplies. Orphaned, already-dead requests fill every batch slot, real requests get pushed further back, latency climbs, more clients time out and retry. The GPU sits at 100% utilization serving nobody. The success rate is zero.

The fixes are non-negotiable for production:

- **Request deadlines.** Stamp every request with a deadline; drop it the instant it can no longer meet its SLO instead of working on a corpse.
- **Disconnect detection.** Watch for the client closing the connection and cancel the in-flight generation immediately.
- **Admission control / load shedding.** Reject (fast 429) when the queue exceeds what you can drain in time. A clean rejection is infinitely better than a slow death for everyone.

---

# The KV cache: the thing you're really managing

When generating token number 500, the model needs to attend to all 499 prior tokens. Recomputing their Key and Value projections every step would be $O(N^2)$ and ruinous. So you compute each token's K and V **once** and cache them. That cache *is* the per-request state, and managing it is most of what a serving engine does.

The size per token is:

$$  
\text{bytes/token} = 2 \times L \times H_{kv} \times d_{head} \times \text{bytes per element}  
$$

The leading 2 is for K and V; $L$ is layers, $H_{kv}$ is the number of **KV** heads, $d_{head}$ the head dimension. Multiply by sequence length and by concurrent requests and it gets large fast, the KV Cache, not the weights, is usually what limits how many users you can batch.

## 1) GQA and MQA: shrink the cache at the source

Notice the formula scales with the number of *KV* heads. The biggest single reduction isn't a serving trick at all.n it's an architecture choice baked into the model:

- **Multi-Head Attention (MHA):** every query head has its own K and V. Maximum cache.
- **Multi-Query Attention (MQA):** all query heads share **one** K/V head. Tiny cache, some quality loss.
- **Grouped-Query Attention (GQA):** the middle ground where groups of query heads share a K/V head. This is what modern models use.

Concretely, Llama-2-70B has 64 query heads but only **8** KV heads (GQA). That's an **8× smaller** KV cache than full MHA would be which means ~8× more concurrent sequences per GPU, for almost no quality cost. When someone asks why a newer model serves so much cheaper, GQA is often the answer. As a serving engineer you don't choose this, but you absolutely need to read it off the config because it dominates your memory budget.

## 2) PagedAttention:

Even with GQA, naive cache management wastes most of your memory. The old approach pre-allocated one big contiguous block per request, sized for the *maximum possible* length. A request that generates 50 tokens but reserved space for 4,096 wastes 99% of its allocation. Across many requests you get massive internal and external fragmentation. vLLM's paper measured effective utilization around 20–40%.

**PagedAttention** steals the idea of virtual memory from operating systems. Chop the KV cache into fixed-size **blocks** (e.g. 16 tokens each). A request's logical sequence of tokens maps, through a **block table**, to physically scattered blocks anywhere in GPU memory. You allocate blocks on demand as the sequence grows, so there's no over-reservation and essentially no fragmentation. It pushed effective cache utilization above ~96%, which roughly translated to serving several times more concurrent requests on the same hardware. This is the core innovation that made vLLM vLLM.

```
Logical view (what the request thinks it has):
  [tok 0..15][tok 16..31][tok 32..47]
       │           │           │        block table maps each →
       ▼           ▼           ▼
Physical GPU memory (scattered, no gaps wasted):
  ... [blk 7] ... [blk 2] ......... [blk 9] ...
```

## 3) Prefix caching:

Once the cache lives in shareable blocks, a beautiful optimization falls out. Tons of requests share a prefix, the same long system prompt, the same few-shot examples, the same retrieved document in a RAG pipeline. With block-level caching you compute that prefix's KV blocks **once** and let every later request that shares it just *point at the same physical blocks*. Reference-count them, and use copy-on-write so a request that diverges gets its own fork.

Engines track these shared prefixes with a radix tree over token sequences. The payoff is direct: for a recurring prefix the prefill is essentially **skipped**, which slashes TTFT and frees compute. In a RAG or agent system where every call carries the same 2,000-token system preamble, this is one of the highest-ROI features you can turn on.

### When the cache still won't fit

Two more levers when memory is tight:

- **KV cache quantization**: store K/V in FP8 (or lower) instead of FP16, halving the cache for a small accuracy hit. Lets you roughly double context length or batch size.
- **KV offloading**: page cold blocks out to CPU RAM (or NVMe) and pull them back when needed. Buys capacity at the cost of PCIe bandwidth; useful for very long contexts that don't fit in HBM.

---

# FlashAttention:

People conflate these two because both have "attention" in the name, but they fix different things. PagedAttention is about *where the KV cache lives in memory*. **FlashAttention** is about *how the attention math is computed*.

The naive attention computation builds the full $N \times N$ score matrix $S = QK^\top$ in HBM, softmaxes it, then multiplies by $V$. For long sequences that matrix is gigantic and — here's the key — writing it to HBM and reading it back is the actual bottleneck. The matmul is fast; shuffling the intermediate matrix to and from memory is slow.

FlashAttention is **IO-aware**. It tiles Q, K, and V into blocks small enough to fit in the GPU's on-chip SRAM, and computes the softmax **incrementally** ("online softmax") so it never materializes the full $N \times N$ matrix in HBM at all. Memory drops from $O(N^2)$ to $O(N)$ and the kernel runs much faster because it stopped thrashing memory. FlashAttention-2 improved how the work is split across the GPU's units; FlashAttention-3 exploits Hopper's async copies and FP8. You will basically always be running some FlashAttention variant under the hood — it's the default attention kernel now, and it composes with PagedAttention (the paged version reads the scattered KV blocks into the flash kernel).

---

## Making the model cheaper: quantization and distillation

Decode is memory-bound, so the most direct way to speed it up is to make the weights **smaller** — fewer bytes to stream from HBM per token. That's quantization.

### Quantization

Map high-precision weights (FP16/BF16) onto a coarser grid (INT8, INT4, FP8, FP4). Uniform quantization is:

$$  
W_q = \text{clip}\left(\text{round}\left(\frac{W}{S}\right) + Z, q_{min}, q_{max}\right), \qquad \hat{W} = S(W_q - Z)  
$$

where $S$ is a scale and $Z$ a zero-point. At INT4 you've cut weight memory ~4× versus FP16, which roughly speeds up the memory-bound decode by a similar factor — the headline reason quantization is everywhere.

Two ways to get there:

- **QAT (Quantization-Aware Training):** simulate quantization during training so the optimizer adapts to the noise. Best accuracy, but you need the training pipeline and compute. Rare for people serving open weights.
- **PTQ (Post-Training Quantization):** quantize a frozen, already-trained model. No retraining, runs in minutes to hours on a few calibration samples. This is what nearly everyone uses in production.

The two PTQ methods worth knowing:

- **GPTQ** frames quantization as layer-wise error minimization, using a second-order (inverse-Hessian) approximation to decide how to round each weight so the layer's *output* changes as little as possible. It's careful and accurate; the Cholesky decomposition and lazy batch updates are just numerical tricks to make the Hessian math stable and fast at billion-parameter scale.
- **AWQ (Activation-aware Weight Quantization)** starts from a sharp observation: a tiny fraction of weights (under ~1%) are "salient" and matter far more than the rest — and you can find them by looking at *activation* magnitudes, not weight magnitudes. AWQ scales those salient channels up before quantizing (and scales the activations down to compensate), which protects the important weights from rounding error without the cost of storing anything in mixed precision. In practice AWQ is fast to apply and holds quality well at 4-bit, which is why it shows up so often in served models.

There's also **FP8** as a first-class inference format on Hopper — it keeps a floating-point exponent so it handles outliers more gracefully than INT8, and the hardware has native FP8 tensor cores, so you get both smaller weights and faster compute.

One honest caveat: quantization quality is workload-dependent. A model that benchmarks fine at INT4 can degrade noticeably on long-context reasoning or code. Always re-evaluate on *your* task, not just perplexity.

### Distillation

Quantization shrinks the representation; distillation shrinks the **model**. You train a small "student" to imitate a big "teacher."

- **Response-based** distillation has the student match the teacher's softened output distribution. You raise the softmax temperature $T>1$ to spread probability mass, which exposes the teacher's "dark knowledge" — the relative likelihoods among *wrong* answers, which carry a lot of signal. The loss is the KL divergence between the two distributions:

$$  
q_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}  
$$

- **Feature-based** distillation goes deeper, aligning the student's *intermediate* hidden states to the teacher's (with a projection to bridge the size gap), not just the final output.

Distillation is more work than quantization and needs data, but it can produce a small model that punches well above its parameter count on a specific domain. The two compose: distill, then quantize the student.

---

## Decoding faster: speculative decoding, sampling, structure

### Speculative decoding

This one's clever. Decode is memory-bound, which means when the big model does a forward pass to make *one* token, the GPU's compute units are mostly idle — you've paid to load all the weights anyway. So why not verify *several* candidate tokens in that same pass for nearly free?

- A small, cheap **draft** model autoregressively guesses the next $K$ tokens.
- The big **target** model checks all $K$ in a **single** forward pass (it can, because it now has all $K$ candidate positions to score at once).
- A **rejection-sampling** rule decides how many to keep, and — this is the important part — it's mathematically guaranteed to produce exactly the same distribution the target model would have produced alone. No quality loss. You're not approximating; you're just doing the same sampling more efficiently.

The acceptance rule per drafted token $x$: if the target assigns it at least as much probability as the draft did ($P_{target}(x) \ge P_{draft}(x)$), accept it. Otherwise accept it with probability $P_{target}(x)/P_{draft}(x)$.

Two outcomes people routinely mix up:

- **On the first rejection**, you stop, throw away the rest of the draft, and resample *that one position* from the adjusted residual distribution $\propto \max(0, P_{target}(x) - P_{draft}(x))$. That keeps the math exact.
- **If all $K$ are accepted**, you get a free **bonus token** from the target's final position — the extra token you scored "for free" in the same pass.

When the draft is good (high acceptance rate $\alpha$), this routinely gives **2–3× faster decode** with identical output. When $\alpha$ is low, you waste compute drafting and verifying tokens you throw away, and it can be *slower* than not speculating — which is why acceptance rate is a metric you watch.

Beyond the classic draft-model setup:

- **Medusa** bolts extra prediction heads onto the target model itself, so it drafts its own future tokens — no separate model to host.
- **EAGLE** drafts at the feature level for higher acceptance.
- **Prompt-lookup / n-gram** decoding skips the model entirely for the draft: it just copies likely continuations from the prompt. Shockingly effective for summarization and code editing where the output echoes the input.

### Sampling

The forward pass gives you logits; sampling turns them into a token. This is cheap but it shapes everything users feel:

- **Greedy / temperature** — `temperature` flattens (>1) or sharpens (<1) the distribution; 0 is deterministic argmax.
- **Top-k** keeps the k most likely tokens, **top-p (nucleus)** keeps the smallest set whose probability mass exceeds p, **min-p** scales the threshold by the top token's probability. These trade diversity against coherence.
- **Repetition / presence penalties** fight loops.

It's a small part of the runtime but a large part of perceived quality, and getting it wrong (e.g. high temperature on a tool-calling agent) breaks things in ways that look like model failures.

### Structured / guided decoding

When you need the output to be valid JSON or match a schema, don't parse-and-retry — constrain the sampling. At each step you build a mask that zeroes out any token that would violate the grammar/state machine, so the model can *only* emit valid continuations. Libraries like Outlines and XGrammar compile the schema into an FSM ahead of time, so the per-token overhead is near zero. This turns "the model usually returns JSON" into "the model cannot return anything but valid JSON," which is the difference between a demo and a reliable API.

---

## Compilation and kernels: the unglamorous speedups

A surprising amount of decode time is *overhead*, not math — launching thousands of tiny CUDA kernels, Python dispatch, framework bookkeeping. Each decode step is small, so per-step overhead dominates.

- **CUDA graphs** capture the entire sequence of kernel launches for a decode step *once*, then replay the whole thing as a single unit. When each step is tiny, killing the launch overhead is one of the largest decode speedups available — vLLM uses this heavily.
- **Kernel fusion** merges adjacent operations (e.g. a matmul + bias + activation) into one kernel so intermediate results never round-trip through HBM.
- **Graph compilers** — TensorRT-LLM, `torch.compile` — do this fusion and layout selection automatically and emit hardware-tuned kernels. TensorRT-LLM tends to be the fastest on NVIDIA but the least flexible (you compile an engine per model/shape/precision); `torch.compile` is more general and lives inside PyTorch.

None of this changes the model's output. It just stops wasting cycles, and it's often a 20–50% win for nearly free.

---

## Going multi-GPU: when one accelerator isn't enough

You shard across GPUs for two reasons: the model doesn't fit, or one GPU can't keep up with demand. A 70B model in FP16 needs ~140 GB just for weights — already too big for an 80 GB H100 before you've stored a single KV block. The strategies are different tools for different problems.

### Data Parallelism (DP)

Replicate the whole model on each GPU; each is an independent, self-contained server. A stateless L7 load balancer sprays requests across replicas (least-outstanding-requests beats round-robin here, because request costs vary so much). Inference DP is simple — no gradient sync like training — and scales **throughput** linearly. But it does nothing for single-request latency, and it can't help if the model doesn't fit on one GPU in the first place.

### Tensor Parallelism (TP)

Split the *individual weight matrices* across GPUs (intra-layer sharding), the Megatron-LM way. The trick is to shard so that communication is minimized:

- Projection matrices ($W_Q, W_K, W_V$, and the MLP's first layer) are split **column-wise** — each GPU computes a slice of the output independently, no communication needed mid-way.
- Output matrices (attention's $W_O$, the MLP's second layer) are split **row-wise** — each GPU produces a *partial* result, and the partials must be summed with an **all-reduce** to reconstruct the true output.

So every transformer block costs **two all-reduces** in the forward pass (one after attention, one after the MLP). All-reduce is a blocking collective — everyone waits for everyone — so TP is brutally sensitive to interconnect speed. It's viable on NVLink (900 GB/s on H100) and a disaster over PCIe (~64 GB/s per direction), where communication swamps compute. **Rule of thumb: TP stays inside a node, across NVLink.** What you get for it: the model *and* the KV cache split $1/N$ per GPU (fits bigger models) *and* lower single-request latency (more compute on one token). It's the go-to for latency-sensitive serving of a model too big for one card.

### Pipeline Parallelism (PP)

Split by *layers* (inter-layer sharding): GPU 0 holds layers 1–20, GPU 1 holds 21–40, and so on. Activations pass from one stage to the next — point-to-point, far less communication than TP, so PP works **across nodes** (Ethernet/InfiniBand). The catch is the **pipeline bubble**: while GPU 0 works on the first layers, GPUs 1–3 sit idle waiting for it, and vice versa. You hide the bubble with **micro-batching** — slice the batch into pieces that flow through the stages staggered, keeping every stage busy on a different piece. PP is mainly about fitting big models and adding throughput, not cutting latency.

### Expert Parallelism (EP) — for MoE models

Mixture-of-Experts models (Mixtral, DeepSeek-V3) only activate a couple of "expert" FFNs per token, so they have huge parameter counts but modest per-token compute. You shard by placing different experts on different GPUs. A **router** picks experts per token, and an **all-to-all** collective shuffles each token to wherever its experts live and the results back. The challenge is *load balance* — if everyone's tokens want the same expert, that GPU melts while others idle. EP is its own discipline and increasingly central as MoE eats the frontier.

|                     | Data (DP)      | Tensor (TP)                     | Pipeline (PP)              | Expert (EP)        |
| ------------------- | -------------- | ------------------------------- | -------------------------- | ------------------ |
| Splits what         | whole replicas | weight matrices                 | layers                     | experts (MoE)      |
| Communication       | none (HTTP)    | very high (2× all-reduce/layer) | moderate (P2P activations) | high (all-to-all)  |
| Interconnect needed | Ethernet       | NVLink only                     | PCIe/InfiniBand OK         | NVLink/fast fabric |
| Memory per GPU      | 100% of model  | 1/N model + 1/N KV              | 1/N model (by layer)       | 1/N experts        |
| Primarily buys you  | throughput     | latency + fit                   | fit + throughput           | MoE fit            |

In practice you **combine** them: TP within a node, PP across nodes, DP across the whole fleet, EP for the MoE layers. The art is matching each axis to the interconnect that can afford its communication.

### Prefill–decode disaggregation

Now the idea that ties the whole document together. Prefill is compute-bound; decode is memory-bound. When they run on the *same* GPU, they fight — a long prefill stalls decode (the stutter we fixed with chunked prefill), and the two phases want different batch sizes, different parallelism, even different hardware.

**Disaggregation** (DistServe, Splitwise, Mooncake) runs them on **separate GPU pools**. A prefill pool — sized and tuned for compute and TTFT — does the prompt processing, then **ships the KV cache** over a fast interconnect to a decode pool tuned for memory bandwidth and ITL. Each pool scales independently to its own SLO, and neither interferes with the other. The cost is the KV-cache transfer between pools, which is why this needs serious networking (and is driving work on fast KV transfer like NIXL). It's the current frontier for large-scale, SLO-strict deployments — and notice it's just the prefill/decode mental model from the top of this note, taken to its logical hardware conclusion.

---

## Multi-LoRA: many fine-tunes, one base

A common real-world need: you have one base model and dozens or thousands of LoRA fine-tunes (per customer, per task). Hosting a full copy per adapter is absurdly wasteful. **Multi-LoRA serving** (S-LoRA, Punica) keeps a single copy of the base weights in memory and swaps in the tiny per-request LoRA adapters, batching requests for *different* adapters together with custom kernels (like SGMV) that apply each row's adapter on the fly. You serve thousands of fine-tunes from roughly the memory of one base model. If you're running a fine-tuning-as-a-product business, this is the architecture.

---

## The full stack: a request's life

Putting the pieces together, here's the path a single request takes through a serious serving deployment.

```
                 CLIENT  ──HTTP/gRPC──┐
                                      ▼
 ┌─────────────────────────────────────────────────────────┐
 │  L7 GATEWAY (Envoy/NGINX)                                 │
 │  TLS · auth · rate limit · canary split                   │
 └───────────────────────────┬──────────────────────────────┘
                             ▼
 ┌─────────────────────────────────────────────────────────┐
 │  ROUTER  (prefix-aware + overload protection)            │
 │  hash system prompt → replica with a WARM prefix cache    │
 │  shed load when queue > drainable depth                   │
 └───────────────────────────┬──────────────────────────────┘
                             ▼
 ┌═════════════════════════════════════════════════════════┐
 ║  SERVING RUNTIME (vLLM / TensorRT-LLM / SGLang)          ║
 ║   ┌───────────────────┐    ┌──────────────────────────┐  ║
 ║   │ continuous batcher │──▶│ speculative decode engine │ ║
 ║   │ + chunked prefill  │    │ draft → target verify     │ ║
 ║   │ + preemption       │    └──────────────────────────┘  ║
 ║   └─────────┬─────────┘                                   ║
 ║             ▼                                              ║
 ║   ┌──────────────────────────────────────────────────┐   ║
 ║   │ PagedAttention KV manager                          │  ║
 ║   │ radix-tree prefix cache · block table · COW        │  ║
 ║   └──────────────────────────────────────────────────┘   ║
 └═══════════════════════════┬═════════════════════════════┘
                             ▼  (Tensor Parallel, NVLink)
        ┌────────────┐  900GB/s  ┌────────────┐  900GB/s  ┌────────────┐
        │  GPU 0     │◀════════▶│  GPU 1     │◀════════▶│  GPU 2     │
        │ quant wgts │           │ quant wgts │           │ quant wgts │
        │ KV pages   │           │ KV pages   │           │ KV pages   │
        └────────────┘           └────────────┘           └────────────┘
```

The walk-through: the gateway terminates TLS, authenticates, and rate-limits. The router hashes the prompt's prefix and sends the request to a replica whose prefix cache is already warm (a cache hit there skips prefill entirely), and sheds load if the queue is too deep to drain in time. The runtime drops the request into the waiting queue; the continuous batcher admits it within the `max_queue_delay` window and folds it into the live batch — its prefill chunked so it doesn't stall anyone's decode. The PagedAttention manager maps the sequence to physical blocks, bumping refcounts (copy-on-write) on any prefix blocks it can share. The forward pass runs across the TP GPUs over NVLink, with an all-reduce stitching the shards back together each layer, while the speculative engine uses the otherwise-idle compute to verify drafted tokens. As tokens generate they **stream** back to the client (SSE) — you don't wait for the whole completion. On `<EOS>`, the batcher evicts the sequence, frees its KV blocks back to the pool, and closes the stream.

---

## Running it: the MLOps that's specific to GPUs

Standard web-service operations assumptions break on GPU inference. You have to relearn three things.

### Autoscaling can't use CPU or RAM

A serving process like vLLM grabs ~90% of GPU VRAM **at startup** to pre-allocate its KV block pool, so VRAM usage is flat regardless of load — useless as a scaling signal. And the CPU mostly just launches kernels, so it looks idle while the GPU is pegged. Scale on CPU/RAM and you'll never scale at all.

Autoscale on **queue depth and KV-cache utilization** instead. With KEDA + Prometheus you read runtime metrics (`vllm:num_requests_running`, queue depth) and size the fleet by capacity per replica:

$$  
\text{replicas} = \left\lceil \frac{\text{total pending + active requests}}{\text{requests one replica can serve within SLO}} \right\rceil  
$$

The best **leading** indicator is KV-cache utilization: once it crosses ~75–80%, the engine is about to stop admitting new sequences and your tail is about to spike — so scale out *before* it hits the wall, not after. Scaling on lagging latency metrics means you only react once users are already hurting.

### Cold starts are brutal

Scaling out means loading 140 GB of weights from object storage, through the CPU, across PCIe, into HBM. Done naively (`torch.load` unpickling into Python objects) it allocates huge intermediate buffers, risks CPU OOM, and takes minutes — long enough that your autoscaler is always behind the curve.

The fix is `.safetensors` + **memory-mapping**. Stage weights on fast local storage (tmpfs/NVMe), `mmap()` the file into virtual memory, and use `cudaHostRegister` to pin it so the GPU can **DMA** the weights directly across PCIe with no intermediate CPU copy. Zero-copy loading turns minutes into seconds, which is what makes responsive autoscaling possible at all. (Warm pools and snapshotting help too.)

### Rollouts when "correct" is undefined

You can't unit-test an LLM the way you test a function — generation is non-deterministic and "right" is fuzzy. So you de-risk deploys with traffic, not assertions:

- **Shadow deployment:** mirror a slice of real traffic to the new version *asynchronously*. Users only ever see the old version's response; an offline pipeline compares TTFT, ITL, and quality/perplexity of the shadow. Catches regressions with zero user risk.
- **Canary release:** route a small % of *live* traffic to the new version and watch the metrics before ramping. Critically, keep **session stickiness** — a user mid-conversation must not bounce between versions, or their prefix cache misses and quality wobbles visibly.

### Observability: the metrics to actually watch

- **TTFT & ITL histograms at p50/p95/p99** — the SLO truth. Everything else is diagnosis.
- **KV-cache utilization** (`vllm:gpu_cache_usage_perc`) — tells you when you're memory-starved vs compute-starved, and it's your scale-out trigger.
- **GPU SM utilization** (DCGM exporter) — read it *together* with queue depth. High SM + low queue = healthy compute-bound work. Low SM + high queue = a bottleneck *outside* the GPU (tokenization, the Python event loop, scheduling) — you're starving the GPU, and adding GPUs won't help.
- **Speculative acceptance rate $\alpha$** — if the draft model drifts and $\alpha$ drops below ~0.5, speculation is *costing* you compute. Alert on it and be ready to turn speculation off.
- **Batch size and preemption/swap counts** — rising preemptions mean you're memory-bound and should scale or shrink context limits.

---

## A note on hardware beyond NVIDIA

NVIDIA dominates, but it's worth knowing the alternatives exist because they change the cost math: **AMD MI300X** (massive 192 GB HBM, ROCm + vLLM support), **Google TPUs** (great for both training and batch inference if you live in JAX/XLA), **AWS Inferentia/Trainium** (cheaper per token in AWS), and latency specialists like **Groq** and **Cerebras** that target extreme tokens/sec for decode. The principles in this document are vendor-neutral — prefill vs decode, memory-bound decode, batching, KV management — but the specific knobs and the best-value chip shift with the workload.

---

## What actually breaks in production

From most common to most subtle:

- **Running utilization too hot.** Someone caps the fleet at 95% to save money, and p99 detonates the first time traffic blips. The queue is non-linear; leave headroom.
- **Unbounded queues + client retries** → the cascading failure. 100% GPU, 0% success. Fix with deadlines, disconnect detection, and load shedding.
- **Long prompts stalling everyone's decode.** Stutter in the token stream whenever a big prompt lands. Turn on chunked prefill.
- **KV cache, not weights, being the real limit.** You sized for the model and forgot the cache scales with batch × context. Check the KV math; reach for GQA models, PagedAttention, KV quantization.
- **Quantization that benchmarks fine and fails on your task.** INT4 perplexity looks great, then long-context reasoning or code degrades. Re-evaluate on *your* workload, not a generic metric.
- **TP over PCIe.** Someone enables tensor parallelism across GPUs without NVLink and throughput *drops* because all-reduce dominates. Keep TP inside the node.
- **Speculation with a bad draft.** Acceptance rate quietly falls, and your "optimization" is now overhead. Monitor $\alpha$.
- **Cold starts losing the autoscaling race.** Minutes-long loads mean you're always scaled for yesterday's traffic. Memory-map weights; keep a warm pool.
- **Scaling on the wrong metric.** CPU/RAM-based HPA never fires for GPU work. Scale on queue depth and KV utilization.
- **Prefix cache thrash from version hopping.** A canary without session stickiness blows the prefix cache and quality wobbles. Pin sessions.

---

## The mental model

If you remember one thing: **prefill is compute-bound, decode is memory-bound, and almost every technique here is a way to stop wasting memory bandwidth during decode.**

- Batching raises decode's arithmetic intensity by sharing loaded weights across sequences — that's continuous batching.
- The KV cache is the per-request state, and managing it (GQA to shrink it, PagedAttention to pack it, prefix caching to share it, quantization to compress it) is most of the engine.
- Quantization shrinks the bytes you stream per token; FlashAttention and CUDA graphs stop wasting the cycles around the math; speculative decoding uses idle compute to verify multiple tokens per memory load.
- Sharding (DP/TP/PP/EP) and disaggregation are the same ideas projected onto many GPUs, each matched to the interconnect that can afford its communication.
- And the operations — autoscale on queues and KV, scale before the wall, shed load, deploy with shadow traffic — exist because GPUs don't behave like the stateless services your old playbook assumes.

The recurring engineering question is the same one as everywhere else: which part of the hardware are you wasting right now, and what's the cheapest way to stop?
