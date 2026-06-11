---
title: "Machine Learning SD"
date: 2026-06-06
summary: "Understanding of System Design in AIML."
tags: [System Design, Agents, LLM, LLMOps, AI]
---

# The ML Design Framework:

1. **Clarify the Ask:** Define the core business goal, translate it into a specific ML problem (e.g., classification vs. ranking), and establish strict scale and latency constraints before touching any models.
2. **Define Data Strategy:** Identify your data sources, establish how you get ground truth labels (implicit vs. explicit), and split your datasets by time to avoid data leakage.
3. **Engineer Features:** Build user, item, and context features, use embeddings for high-cardinality IDs and guarantee consistency to prevent train/serve skew.
4. **Select the Model:** Always start with a simple, interpretable baseline (like logistic regression or GBDT) and clearly justify the trade-offs if upgrading to deep learning.
5. **Evaluate Offline:** Choose metrics specifically tailored to your problem (e.g., NDCG for ranking, AUC for classification) and check performance across individual user segments, not just global averages.
6. **Evaluate Online:** Run strict A/B tests with primary and guardrail metrics, using canary or shadow deployments to capture real user feedback safely.
7. **Design the Serving Architecture:** Decide between batch and real-time scoring, and use a two-stage "candidate generation → ranking" pipeline to handle large catalogs within latency budgets.
8. **Monitor and Maintain:** Set up alerts to track data drift and concept drift, monitor for performance decay over time, and establish a regular retraining cadence.

General System Design asks *"can the system handle the load correctly?"* ML SD adds *"...and is the model's output actually good, and will it stay good as the world changes?"* which is why **eval, features, and drift monitoring** are the three steps with no general-SD equivalent.

A useful thing to internalize: in production, the data and features explain far more of the outcome than the model does. Two teams running the same gradient-boosted tree often land in very different places, and the difference usually traces back to cleaner labels or features that were not corrupted between training and serving.

Two canonical reads that frame everything below: Google's [Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml) (Rule #1: don't be afraid to launch without ML) and [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) (the model is the small box; the pipelines around it are the system).

**Map of this document:**

- **Part I, Foundations:** Data · Features and the Feature Store · Train/Serve Skew and Point-in-Time Correctness
- **Part II, The Retrieval Stack:** Embeddings · Vector Search and ANN · Hybrid Retrieval
- **Part III, Evaluation and Experimentation:** Offline Metrics · Online Metrics and the Gap · A/B Testing
- **Part IV, Classic ML Systems:** Recommender Systems · Ranking Deep Dive · Model Serving · MLOps · Monitoring and Drift
- **Part V, LLM Systems:** Transformer Essentials · Distributed Training · Fine-Tuning and Alignment · Build Decision · RAG · LLM Evaluation · Inference Optimization · Test-Time Compute · Safety and Guardrails
- **Part VI, Agentic Systems:** Core Agent Architecture · Tool Design and the Harness · Multi-Agent · Agent Evaluation · Agent Safety · Agent Economics
- **Part VII, Application:** Worked Design Blueprints · Estimation Cheat Sheet · Reading List

---

# Data

**Sources.** Activity logs (clicks, watch-time) are huge, cheap, and biased. Transactional data (orders, payments) is high quality but sparse. Human annotation is accurate but slow, used when no behavioral signal exists. Synthetic or model-labeled data scales cheaply but amplifies bias.

**Feedback loop / exposure bias.** Logs only contain items the current system showed, so a recommender that never surfaces an item never learns it could be good, and the popular items keep winning. Fix with:

- **Exploration** (epsilon-greedy, bandits, randomized serving on a traffic slice) to collect unbiased data.
- **Inverse-propensity weighting**: weight each example by `1 / P(shown)`, so a rarely-shown item that still got engagement counts as stronger evidence.

**Labels.** Implicit labels (clicks, dwell) are abundant but noisy and position-biased; explicit labels (ratings, annotations) are clean but scarce and self-selected. Key issues:

- The label must be **defined**, not assumed. A raw click is weak (clickbait); "clicked and stayed 30+ seconds" encodes satisfaction.
- Labels can be **delayed**. "Churn in 30 days" is only knowable 30 days later, which caps retraining freshness.
- Human labels disagree: use multiple annotators, inter-annotator agreement (Cohen's kappa), and gold questions.

**Class imbalance** (fraud ~0.1%, CTR ~1%). Accuracy is meaningless here, since "always predict negative" scores 99.9%. Use PR-AUC or recall at a fixed threshold. Then, in order:

1. Adjust the loss: **class-weighted** or **focal loss** (down-weights easy negatives).
2. Resample: **undersample** the majority (cheap at scale) or **SMOTE** the minority.
3. **Recalibrate** afterward (Platt / isotonic). Undersampling inflates predicted probabilities, which breaks any calibrated use such as ad bidding or expected-value ranking.

**Splits.** Split **by time**, never randomly (train on weeks 1 to 8, test on week 10). Random splitting leaks the future and gives an offline number that collapses in production. Use **stratified** splits under imbalance, and **negative sampling** (random, popularity-weighted, or **hard negatives**) when negatives are effectively infinite, as in retrieval.

---

# Features and the Feature Store

**Types.** User, item, and context features are the obvious ones. The two that are underweighted:

- **Interaction features** (user × item, e.g. how often this user buys this category) are often the single strongest signal.
- **Windowed aggregates** ("clicks in last 1h / 24h / 7d") are powerful but the most prone to skew, since their value depends on when they are computed.

Use **embeddings** for high-cardinality IDs (user, product) instead of one-hot, so similar entities generalize.

**Feature store.** Its job is to define each feature **once** and serve the **same value at training and serving time**, plus versioning and freshness. Without it, the training feature (written in a notebook) and the serving feature (reimplemented in production) drift apart. It is built as two stores from one set of transformations:

| | Offline store | Online store |
|---|---|---|
| Computed by | Batch (Spark, SQL), hourly to nightly | Streaming / on-request (Flink) |
| Backed by | Warehouse, parquet | Low-latency KV (Redis, DynamoDB) |
| Latency | Seconds to minutes (fine for training) | Single-digit ms (serving budget) |
| Used for | Training and batch scoring | Real-time serving |
| Example | "Avg order value, nightly" | "Items in cart now", "clicks in last 5 min" |

**Batch vs streaming features: freshness vs cost.** Batch is cheap and simple but stale by hours; streaming is seconds-fresh but operationally expensive (stateful stream jobs, exactly-once concerns). The design rule: pay for streaming only where freshness changes the prediction (in-session intent, fraud velocity counts), and keep slow-moving profile features in batch. (Feast, Tecton, Vertex / SageMaker FS.)

---

# Train/Serve Skew and Point-in-Time Correctness

Both are silent failures: offline metrics look healthy while production quietly underperforms, with nothing in the logs.

**Train/serve skew** is when a feature is computed differently at train vs serve, so inference inputs do not match the training distribution. Three causes:

- **Code skew**: training feature in Spark, serving feature reimplemented in Go, and the two disagree on rounding, nulls, or timezone.
- **Data skew**: training reads a clean warehouse, serving reads a live API with different freshness or units.
- Time-travel skew, covered next.

It is dangerous because nothing crashes, the feature is just wrong. Prevent it with a feature store (define once); detect it by logging served features and comparing their distribution against training.

**Point-in-time correctness** means a training row for an event at time T must use only data available before T. Classic leak: label is "clicked at 2pm" and the feature is "clicks today," which already contains the 2pm click, so the model reads the answer out of its input. Offline accuracy looks near perfect and the model is useless live, because at 2pm in production you only know clicks up to 2pm. The correct feature is "clicks strictly before 2pm." Enforcing this at scale needs an **as-of join** (join the feature value as it was at the event's timestamp); a naive key join returns the latest value and leaks. The offline store performs these joins.

Point-in-time is about **time** (no future data); skew is about **consistency** (same computation both places). Both enforce one rule: training must mirror serving. The feature store is the common defense, since define-once transforms close the consistency gap and as-of joins close the time gap.

---

# Embeddings

**What they are.** An embedding is a fixed-length dense vector (typically 256 to 1536 dimensions) that encodes the meaning of an item (text, image, user, product) such that semantically similar items land close together in the vector space. They are produced by a learned encoder (e.g. a sentence-transformer for text, a two-tower model for users and items).

**What they encode.** Direction carries meaning, magnitude often does not. Two reviews that say the same thing in different words point the same way even if one is longer. This is why for most retrieval you compare **angle**, not raw distance.

**Similarity measures.**

- **Cosine similarity** (the default): measures the angle between vectors, ignoring magnitude.

  `cos(a, b) = (a · b) / (||a|| * ||b||)`, range `[-1, 1]`, where 1 means identical direction.

- **Dot product** `a · b`: faster (no normalization) and the standard choice when the model was trained with it, but it is sensitive to magnitude, so a longer document can score higher just for being longer.

- **Euclidean (L2) distance** `||a - b||`: used for some image and clustering tasks.

**Key identity:** if all vectors are L2-normalized to unit length, cosine, dot product, and Euclidean distance all rank neighbors identically. The production pattern is therefore **normalize once at index time, then use plain dot product at query time** to get cosine ranking at maximum speed.

**Training signal.** Text embedding models are trained with **contrastive learning** (pull positive pairs together, push negatives apart, InfoNCE loss); recommendation embeddings come from two-tower training on interactions. The embedding is only as good as the pairs it was trained on, which is why domain fine-tuning of embeddings is one of the highest-ROI levers in retrieval quality.

---

# Vector Search and Approximate Nearest Neighbors (ANN)

**The problem.** Given a query embedding, find the top-k most similar vectors out of millions or billions. Exact search is `O(N * d)` per query, far too slow at scale. ANN trades a small amount of accuracy for orders-of-magnitude speedup.

**The metric: recall@k.**

`recall@k = (true top-k neighbors found by ANN) / k`

You almost never need exact results; tune the index to a recall target (commonly 0.9 to 0.99) at the lowest possible latency.

## HNSW (Hierarchical Navigable Small World)

The most widely used ANN index ([Malkov & Yashunin 2016](https://arxiv.org/abs/1603.09320)):

- Build a **multi-layer graph**: top layer sparse with long-range links, each lower layer denser, bottom layer contains every point. A skip list generalized to a graph.
- **Search** starts at the top, greedily hops toward the query, drops a layer, repeats. Long links cover distance fast; short links refine precisely. Roughly `O(log N)` per query.

Two knobs:

- `M`: links per node. Higher means better recall, more memory.
- `efSearch`: candidate list size at query time. Higher means higher recall, higher latency. This is the main production dial on the recall vs latency curve.

## The three-way trade-off

You cannot maximize recall, latency, and memory at once:

| Lever | Effect |
|---|---|
| Raise `efSearch` / `M` (HNSW) | Recall up, latency and memory up |
| **Quantization** (PQ, scalar) | Memory down a lot, recall down slightly |
| More shards / replicas | Latency down (parallelism), infra cost up |

**Product Quantization (PQ)** splits each vector into sub-vectors and replaces each with a codebook id, cutting memory 8x to 32x at a modest recall cost. This is how billions of vectors fit in RAM.

**IVF (inverted file)** is the other major family: cluster vectors into coarse cells, search only the closest `nprobe` cells. Often combined with PQ (IVF-PQ in FAISS). HNSW wins on recall/latency at moderate scale; IVF-PQ wins on memory at billion scale.

## Google ScaNN

Google's ANN library for very large scale ([Guo et al. 2020, anisotropic vector quantization](https://arxiv.org/abs/1908.10396)). Key idea: weight quantization error in the directions that most affect the dot product (the ranking you actually care about) rather than minimizing raw reconstruction error. Pipeline: **partition (coarse cluster) → quantize → re-score top candidates with full precision**. Higher recall at the same memory budget for maximum-inner-product search.

**Operational concerns an interviewer will probe:** index build time and immutability (HNSW inserts are cheap, deletes are not; many systems rebuild periodically), **freshness** (new items need to be searchable in seconds: keep a small exact-search "fresh" buffer merged at query time, compact into the main index later), **sharding** (shard by vector, fan out the query, merge top-k), and **filtered search** (pre-filter shrinks the candidate space but breaks graph connectivity; post-filter wastes recall; real engines use hybrid filtered-ANN).

---

# Semantic vs Keyword vs Hybrid Retrieval

**Keyword (lexical), e.g. BM25.** Matches exact terms. Fast, cheap, interpretable, and excellent for rare tokens (product codes, error IDs, names, acronyms):

`BM25 = Σ IDF(qᵢ) * [ f(qᵢ, D) * (k₁ + 1) ] / [ f(qᵢ, D) + k₁ * (1 - b + b * |D| / avgdl) ]`

where `f(qᵢ, D)` is term frequency, `|D|` is document length, `avgdl` is average length, `k₁` and `b` are constants. Weakness: zero understanding of meaning; "laptop" will not match "notebook computer."

**Semantic (vector).** Matches meaning via embeddings + ANN. Handles synonyms, paraphrase, intent. Weakness: can miss exact identifiers (an obscure SKU has no good embedding).

| | Keyword (BM25) | Semantic (vector) |
|---|---|---|
| Matches | Exact terms | Meaning, synonyms, intent |
| Strong on | Rare tokens, IDs, names, code | Paraphrase, vague queries |
| Weak on | Synonyms, vague phrasing | Exact rare identifiers |
| Cost | Very cheap | Embedding + ANN compute |

**Hybrid (BM25 + vector).** Run both, fuse the results, because the failure modes are complementary:

- **Reciprocal Rank Fusion (RRF)** ([Cormack et al. 2009](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)): combine by rank, not raw score, avoiding score normalization: `RRF(d) = Σ 1 / (k + rankᵢ(d))`, `k ≈ 60`.
- **Weighted score fusion:** normalize each retriever's scores, weighted sum, tune per domain.

The enterprise stack: **BM25 and ANN in parallel → RRF fusion → cross-encoder reranker** for final precision. This is the candidate-generation then ranking pattern applied to retrieval.

---

# Offline Evaluation Metrics

Pick the metric that matches the **product decision** the model drives, not the one easiest to compute.

## Classification: AUC and friends

**ROC-AUC** = the probability that a random positive scores higher than a random negative:

`AUC = P(score(random positive) > score(random negative))`

0.5 is random, 1.0 perfect. Threshold-free, measures pure ranking ability. Two caveats:

- Under heavy imbalance ROC-AUC looks deceptively high (easy negatives dominate). Use **PR-AUC**, which focuses on the rare positive class.
- AUC says nothing about **calibration**. A model can rank perfectly with wrong probabilities; anything multiplying probability by money (bidding, EV) needs calibration, checked with reliability curves or **expected calibration error** (bucket predictions, compare mean predicted vs observed rate per bucket, average the gaps).

## Ranking: precision/recall@k, MAP, NDCG

- **Precision@k** = fraction of top k that is relevant ("of the 10 shown, how many were good"). For fixed display sizes.
- **Recall@k** = fraction of all relevant items in the top k. The metric for **candidate generation**, whose only job is to not lose good items before the ranker.
- **MAP** averages precision at each relevant position, then over queries. Rewards early placement; relevance is binary.
- **MRR** = mean of `1 / rank of the first relevant result`; the metric when only the first hit matters (question answering, navigational search).
- **NDCG** is the standard for graded relevance:

  `DCG@k = Σᵢ (2^relᵢ - 1) / log₂(i + 1)`, `NDCG@k = DCG@k / IDCG@k ∈ [0, 1]`

  Two design choices encoded: highly relevant items worth exponentially more (`2^rel`), and lower positions discounted logarithmically because users see them less.

**The mapping: recall@k for retrieval, NDCG@k for ranking, MRR for first-hit tasks, PR-AUC for rare events, ROC-AUC for balanced comparison.** Always slice by segment (new users, rare categories, languages); a healthy global average routinely hides a broken segment.

---

# Online Metrics and the Offline-Online Gap

Three layers of online metrics:

- **Direct engagement:** CTR (clicks / impressions), watch-time, conversion. Fast to read, gameable (clickbait raises CTR while making the product worse).
- **Session level:** session length, task success, bounce, abandonment.
- **Long-term:** retention, churn, LTV. What the business actually cares about, but slow to move and detect.

The standard failure is optimizing the fast metric and damaging the slow one. The defense is a declared **metric hierarchy**: one primary metric, supporting metrics for diagnosis, guardrails that must not regress.

**Why offline winners lose online** (structural, not bad luck):

- **Logs are biased by the old policy.** Offline eval answers "would the new model re-rank the old system's results better," not "what happens when it chooses for itself."
- **Feedback loops do not exist offline.** Online, the model changes behavior, which changes the data.
- **Metric mismatch.** NDCG on historical labels is a proxy for satisfaction, and proxies diverge under pressure.
- **Serving reality.** A model 2% better but 80ms slower can lose the A/B, because slow results lose more engagement than ranking quality gains.

Operational rule: offline metrics are a **filter** deciding which candidates earn an online test; the A/B decides what ships.

---

# A/B Testing

The goal is causal inference: did the new model **cause** the change.

**Randomization.** Assign by deterministic hash: `hash(user_id, experiment_salt) % 100`. Stable across sessions, independent of user properties. Randomize at the **user** level by default; session-level contaminates cross-session effects like retention. Watch for **interference** (treatment users affecting control: feeds, marketplaces, pricing); fix with coarser units: geo randomization, time periods, switchback designs.

**Reading the result.** Define the primary metric and sample size **before** starting (power calculation for the minimum detectable effect). Run at least one full week, usually two, to cover weekday/weekend cycles. Do not stop the first time p < 0.05 appears; peeking inflates false positives severely (use sequential testing if you must peek).

**Guardrail metrics.** Metrics not allowed to regress regardless of the primary: p99 latency (not just mean), error rate, crash rate, revenue, retention. Launch rule: primary improved **and** all guardrails held.

**Novelty effect.** New things get extra engagement because they are new, not better; the mirror is **primacy** (users perform worse until they relearn). Both decay over days to weeks. Defenses:

- Plot the treatment effect **by day** and wait for it to stabilize, instead of reading one cumulative number.
- Compare **new users** (no novelty baseline) against existing users; lift that exists only for existing users is probably novelty.
- For high-stakes launches, hold out a small long-term control group after shipping.

**Rollout pattern:** **shadow mode** (score traffic, log, do not show) → **canary** (1 to 5% watching guardrails) → powered A/B → gradual ramp with automatic rollback wired to guardrail alerts.

---

# Recommender Systems: Two-Tower, Candidate Generation, Ranking

The defining constraint: score a catalog of millions against a user in under ~100ms. No model can score everything per request, so every large recsys is a **funnel** (canonical reference: [Deep Neural Networks for YouTube Recommendations, Covington 2016](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/)):

`millions → candidate generation (recall-focused) → hundreds → ranking (precision-focused) → tens → re-ranking (business logic) → shown`

## Two-tower retrieval

The standard candidate generator ([Yi et al. 2019, sampling-bias-corrected two-tower](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/)):

- A **user tower** encodes user features into a vector `u`; an **item tower** encodes item features into `v`. Score = `dot(u, v)`.
- Trained so that interacted pairs score high, typically with **in-batch negatives**: other items in the same training batch serve as negatives (cheap, but popularity-biased, so apply a popularity correction, e.g. subtract `log P(item)` from the logit).
- The towers never see each other's raw features (no user × item cross features). That restriction is the price you pay for the serving trick below.

**Why it scales:** the towers are decoupled. Precompute all item vectors offline, load them into an ANN index (HNSW / ScaNN), compute the user vector once per request, and the top-k retrieval is one ANN query. Personalized retrieval over millions of items in milliseconds.

In practice you run **multiple candidate sources** in parallel (two-tower, item-to-item co-occurrence, popular-in-region, fresh content, subscriptions) and union them, because no single source covers all recommendation intents.

## The ranker

A second model scores only the few hundred candidates, so it can afford everything the towers gave up: user × item cross features, rich context, deeper architecture ([Wide & Deep, Cheng 2016](https://arxiv.org/abs/1606.07792): a wide linear part memorizes specific cross features while the deep part generalizes). Optimized for precision (NDCG), not recall.

## Re-ranking

Final pass applying what pure relevance misses: **diversity** (no five near-identical items, e.g. MMR: `score = λ·relevance - (1-λ)·max_similarity_to_already_picked`), business rules (sponsored slots, compliance filters), freshness boosts, and **exploration** slots for collecting unbiased data.

## Cold start

New users and new items have no interaction history. Mitigations:

- **Content features over IDs:** towers built on attributes (category, text, creator, demographics) produce reasonable embeddings for unseen entities; an ID-only model cannot.
- **Exploration with bandits:** deliberately show new items to a small traffic slice to buy data, accepting short-term engagement cost for long-term catalog coverage.
- Plus: popularity priors as fallback, and onboarding questionnaires that bootstrap an interest vector.

---

# Ranking Deep Dive: LTR, Position Bias, Calibration

**Learning to Rank (LTR)** has three formulations:

- **Pointwise:** predict each item's score independently (CTR as binary classification). Simple, but ignores that ranking is relative.
- **Pairwise:** learn "A above B" from preference pairs (RankNet, LambdaRank). Matches the actual task better.
- **Listwise:** optimize the list-level metric (NDCG) directly (LambdaMART, still the GBDT workhorse of search ranking; see [Burges 2010, From RankNet to LambdaRank to LambdaMART](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/)).

**Position bias.** Users click the top result partly because it is on top, so naive training learns "position 1 is relevant" instead of true relevance. Mitigations:

- Train with **position as a feature**, then serve with it fixed to a constant, so the model cannot lean on it at inference.
- **Counterfactual LTR:** weight clicks by inverse propensity of their position, `1 / P(examined at position p)`, estimated from randomization experiments or click models.

**Calibration.** Ads and any expected-value ranking need true probabilities, not just correct order, because the score gets multiplied by a bid: `EV = P(click) × bid`. A miscalibrated model misprices every auction. Calibrate post-hoc (Platt / isotonic) and monitor calibration online.

**Multi-task ranking.** Modern rankers predict several objectives at once (click, like, share, watch-time, dissatisfaction) with a shared backbone and per-task heads (shared-bottom, or **MMoE** ([Ma 2018](https://dl.acm.org/doi/10.1145/3219819.3220007)): expert subnetworks gated per task, which handles conflicting objectives better). The final score is a tuned weighted blend: `score = w₁P(click) + w₂E[watch] + w₃P(like) - w₄P(report)`. Those weights are product policy, not learned, and are tuned through A/B tests.

---

# Model Serving

**Latency vs throughput.** Larger batches use the GPU efficiently (throughput up) but each request waits for the batch (latency up). **Dynamic batching** is the standard compromise: hold requests for a few ms, batch whatever arrived, cap the wait. Tune batch size and max-wait against the p99 budget.

**Parallelism.**

- **Data parallelism:** replicate the model, split traffic. Solves throughput, not model size.
- **Model parallelism:** split one model across GPUs when it cannot fit on one: **tensor parallel** (split each layer's matrices, fast interconnect needed) and **pipeline parallel** (split by layers, adds bubble overhead).

**Making the model cheaper.**

- **Quantization:** FP32 → FP16/BF16 (free in practice) → INT8/INT4 (small accuracy cost, big memory and speed win). Rule of thumb for memory: `params × bytes/param`, so 70B at FP16 = 140GB, at INT4 ≈ 35GB.
- **Distillation** ([Hinton 2015](https://arxiv.org/abs/1503.02531)): train a small student to match a large teacher's outputs; serve the student. Most of the quality at a fraction of the cost.
- **Predict-and-cache:** if inputs repeat and predictions are stable (a user's daily recommendations), precompute in batch and serve from a KV store; fall back to live scoring only for cache misses.

---

# MLOps: The Training Pipeline and Model Lifecycle

The model is one box; the system around it is what an interviewer wants to see you design.

`logs → dataset build (point-in-time joins) → data validation → train → offline eval gate → registry → shadow → canary → full rollout → monitoring → (drift trigger) → retrain`

- **Data validation:** schema and distribution checks on every training batch (types, ranges, null rates, drift vs last run). Most "model bugs" are silent data bugs; this is the cheapest place to catch them.
- **Orchestration:** the pipeline is a DAG (Airflow / Kubeflow / Vertex Pipelines), fully automated so retraining is a button, not a project.
- **Experiment tracking and reproducibility:** every run pins data snapshot, code commit, config, and seed. If you cannot reproduce a model, you cannot debug it.
- **Model registry:** versioned artifacts with lineage (which data, which code) and stage labels (staging / production / archived). The prerequisite for one-click rollback.
- **CI/CD for models:** code passes unit tests; models pass **eval gates** (offline metrics above threshold, slice checks, latency budget, calibration) before promotion. A model that fails the gate never reaches shadow.
- **Retraining triggers:** scheduled cadence plus event-driven (drift alarm, metric decay). Continual learning (per-hour online updates) buys freshness at the cost of stability safeguards; most systems do fine with daily or weekly retrains.

---

# Monitoring and Drift

Models do not crash when they go stale; they degrade silently. Monitoring is the production answer to "is the model still good."

- **Data drift:** input distribution `P(x)` shifts (new demographics, a new device type, an upstream pipeline change). Detect by comparing live feature distributions against a training reference: PSI, KL divergence, or simple percentile alarms per feature.
- **Concept drift:** the relationship `P(y|x)` changes (user tastes shift, fraud adapts). The model is seeing familiar inputs but the right answer moved. Detect via online metric decay against a holdout or shadow baseline.
- **Label lag:** when true labels arrive late (churn, fraud confirmation), monitor **proxy signals** (prediction distribution shape, score percentiles, feature drift) as the early-warning layer.

**Operational responses:** scheduled retraining cadence (and event-triggered retrains on drift alarms), **shadow deployment** of the retrained model to validate before exposure, **canary** rollout, and one-click **rollback** to the previous model version (model registry with versioned artifacts is the prerequisite).

---

# Transformer Essentials for System Design

You do not need to derive the architecture in an SD round, but several serving and scaling behaviors are direct consequences of it ([Attention Is All You Need, Vaswani 2017](https://arxiv.org/abs/1706.03762)).

**Attention.** `Attention(Q, K, V) = softmax(QK^T / √d_k) V`. Every token attends to every other token, which is the source of two facts you reason from constantly: prefill cost is **quadratic** in sequence length, and decoding needs the K and V of all previous tokens (hence the KV cache).

**Multi-head variants and the KV cache.** Multi-head attention (MHA) keeps separate K/V per head; **multi-query (MQA)** shares one K/V across all heads; **grouped-query (GQA)** ([Ainslie 2023](https://arxiv.org/abs/2305.13245)) shares K/V within head groups. This is purely a KV-cache-size vs quality trade, and GQA is the production default (typically 8x cache reduction, negligible quality loss).

**FlashAttention** ([Dao 2022](https://arxiv.org/abs/2205.14135)). Exact attention computed in tiles that stay in fast on-chip SRAM, never materializing the n×n attention matrix in GPU memory. Memory goes from O(n²) to O(n) and speed improves 2 to 4x. The reason long contexts are tractable at all.

**Mixture of Experts (MoE).** Replace each FFN with N expert FFNs plus a learned router; each token activates only the top-k (1 or 2) experts ([Switch Transformer](https://arxiv.org/abs/2101.03961), [Mixtral](https://arxiv.org/abs/2401.04088)). Result: parameter count grows ~Nx while FLOPs per token stay nearly flat. The serving catch: **all experts must be resident in memory** even though few are active, so MoE trades cheap compute for expensive memory, and load-balancing the router (aux losses) is a real training problem.

**Scaling laws.** Training compute ≈ `6 × N × D` FLOPs (N params, D tokens). [Chinchilla](https://arxiv.org/abs/2203.15556): for a fixed compute budget, optimal D ≈ 20 × N, which is why modern models train on far more tokens than the old "bigger model" instinct suggested, and why inference-optimized models (small, overtrained) make economic sense when serving cost dominates training cost.

---

# Distributed Training

Why it exists: a 70B model does not fit on one GPU even for inference, and training needs roughly **16 to 20 bytes per parameter** with Adam (FP16 weights 2 + FP16 gradients 2 + FP32 master weights 4 + Adam moments 8 ≈ 16 bytes), so 70B training state ≈ **1.1TB+**, before activations.

- **Data parallelism:** replicate the model on every GPU, split the batch, all-reduce gradients each step. Solves throughput; requires the model (and optimizer state) to fit per GPU.
- **ZeRO / FSDP** ([Rajbhandari 2019](https://arxiv.org/abs/1910.02054)): keep data parallelism but **shard** the optimizer states (stage 1), gradients (stage 2), and parameters (stage 3) across the data-parallel ranks, gathering shards just-in-time. Eliminates the per-GPU memory wall with modest communication overhead. PyTorch FSDP is the same idea.
- **Tensor parallelism:** split each layer's weight matrices across GPUs (each computes a slice of every matmul). Needs very fast interconnect (NVLink), so it stays **within a node**.
- **Pipeline parallelism:** split the model **by layers** across nodes; micro-batches flow through to keep stages busy (the idle time is the "bubble," shrunk by more micro-batches).
- **3D parallelism:** tensor (intra-node) × pipeline (inter-node) × data (across the rest) combined for frontier-scale training.

Supporting techniques: **mixed precision** (BF16 compute with FP32 master weights), **gradient checkpointing** (drop activations, recompute in backward: trade ~30% compute for large memory savings), **gradient accumulation** (simulate large batches), and **frequent checkpointing** for fault tolerance, because at thousands of GPUs hardware failure during training is routine, not exceptional.

---

# Fine-Tuning and Alignment

**SFT (supervised fine-tuning).** Continue training on (prompt, response) pairs to teach format, style, domain behavior. The base recipe behind every chat model.

**LoRA** ([Hu 2021](https://arxiv.org/abs/2106.09685)). Freeze the base weights W; learn a low-rank update `ΔW = B·A` (rank r ≈ 8 to 64). Trains under 1% of parameters, fits on small hardware, and the adapter is a swappable few-hundred-MB file. **QLoRA** ([Dettmers 2023](https://arxiv.org/abs/2305.14314)) backpropagates through a 4-bit quantized base, fine-tuning a 70B on a single GPU node. The serving consequence matters in design rounds: **multi-LoRA serving** hosts one base model with per-tenant adapters hot-swapped per request, which is how you offer "custom models" to thousands of customers without thousands of model replicas.

**Preference tuning.**

- **RLHF** ([InstructGPT, Ouyang 2022](https://arxiv.org/abs/2203.02155)): collect human pairwise preferences, train a reward model, optimize the policy against it with PPO (with a KL penalty to stay near the SFT model). Powerful, complex, unstable.
- **DPO** ([Rafailov 2023](https://arxiv.org/abs/2305.18290)): a closed-form loss directly on preference pairs, no reward model, no RL loop. The default first choice today.
- **Constitutional AI / RLAIF** ([Bai 2022](https://arxiv.org/abs/2212.08073)): Anthropic's recipe; AI feedback graded against written principles replaces most human preference labeling.

**Design-round decision:** fine-tune for **behavior** (consistent format, tone, domain dialect, distilling an expensive prompt into the weights to cut per-request tokens); never fine-tune to inject **facts** (that is RAG). Re-run safety evals after any fine-tune: tuning on narrow data measurably erodes alignment.

---

# LLM Systems: Fundamentals and the Build Decision

**Tokenization.** Models read subword tokens (BPE), roughly 0.75 English words per token. All costs (latency, money, context) are measured in tokens.

**Context window.** The model's working memory per request. Attention cost grows with context length (quadratic at prefill), and long contexts degrade retrieval quality inside the prompt (see "lost in the middle"), so bigger windows are not automatically better.

**Sampling controls.** Temperature scales logits before softmax: `softmax(z / T)`. Low T sharpens (deterministic, factual tasks), high T flattens (creative tasks). **Top-p (nucleus)** samples from the smallest token set whose cumulative probability exceeds p, cutting the long garbage tail. **Constrained decoding** (grammar / JSON-schema enforcement at the sampler) guarantees syntactically valid structured output, which is what makes tool calling reliable.

**The build decision: prompt vs RAG vs fine-tune.** The most common LLM design question, resolved by what is actually missing:

| Missing | Use | Why |
|---|---|---|
| Nothing much (capability exists) | **Prompting** | Cheapest, fastest to iterate, no training |
| **Knowledge** (private, fresh, factual) | **RAG** | Updatable instantly, sources citable, no retraining for new facts |
| **Behavior** (format, style, domain dialect) | **Fine-tuning** | Bakes in consistent behavior; does NOT reliably add facts |

Rules of thumb: knowledge problems are RAG problems, behavior problems are fine-tuning problems, and you exhaust prompting before paying for either. They compose: a fine-tuned model inside a RAG pipeline is common.

---

# RAG End-to-End

The pipeline:

`ingest → chunk → embed → index → retrieve → rerank → generate (→ cite)`

**Chunking.** Documents must be split before embedding, and chunk size is a real trade-off: too small loses surrounding context, too large dilutes the embedding (one vector averaging many topics matches nothing well). Strategies: **fixed-size with overlap** (baseline), **sentence/paragraph-aware** (respects natural boundaries), **recursive** (split by structure: headers, then paragraphs, then sentences, until chunks fit). Typical starting point: 300 to 800 tokens with 10 to 20% overlap, tuned by retrieval eval, not intuition.

**Retrieval.** Hybrid (BM25 + vector + RRF) as covered above, because user queries contain both exact identifiers and vague intent. Common upgrades: **query rewriting** (decompose multi-part questions, expand acronyms), **HyDE** (embed a hypothetical answer instead of the question), and **metadata filtering** (tenant, date, source) pushed into the vector store. For relationship-heavy corpora, **GraphRAG** ([Edge 2024](https://arxiv.org/abs/2404.16130)) builds an entity graph + community summaries so "summarize across many documents" questions work.

**Lost in the middle** ([Liu 2023](https://arxiv.org/abs/2307.03172)). Models attend best to the **beginning and end** of the context; facts buried in the middle of a long prompt are measurably more likely to be ignored. Consequences: do not stuff 50 chunks into the window because "it fits," and **order retrieved chunks so the most relevant sit first or last**.

**Reranking.** The retriever (bi-encoder: query and document embedded separately) is fast but coarse. A **cross-encoder** reads the query and document together through one model and scores the pair with full token-level interaction: far more accurate, far too slow to run on the whole corpus. So: retrieve top 50 to 100 cheaply, cross-encode that set, keep the top 5 to 10 for the prompt. Retrieval is for recall; the reranker is for precision.

**Generation.** Instruct the model to answer **only from the provided context** and to say "not found" rather than guess; require citations to chunk ids so answers are verifiable.

**Agentic retrieval (the deep-research pattern).** One-shot RAG retrieves once and answers. Agentic retrieval puts the retriever inside an agent loop: search, read, notice what is missing, rewrite the query, search again, then synthesize. This is how Anthropic, OpenAI, and Google build their "deep research" products, and it dominates on hard multi-hop questions at 10 to 100x the token cost. Design with a budget cap and use it only above a difficulty threshold.

**Freshness and permissions, the two enterprise killers:** the index must update within minutes of document changes (streaming ingest, tombstone deletes), and retrieval must enforce **document-level ACLs at query time** (filter by the caller's permissions inside the vector search, never after generation), because a RAG system that leaks one unauthorized document is dead.

---

# LLM and RAG Evaluation

A RAG answer can fail in two independent places, so evaluate them separately (framework: [RAGAS](https://arxiv.org/abs/2309.15217)):

**Retrieval quality** (did the right chunks arrive): **context precision** (fraction of retrieved chunks that are relevant) and **context recall** (fraction of needed information that was retrieved). A generation fix cannot recover from a retrieval miss.

**Generation quality** (did the model use them well):

- **Faithfulness:** is every claim in the answer supported by the retrieved context? This is the anti-hallucination metric. Operationally: decompose the answer into atomic claims, check each against the context, score = supported / total.
- **Answer relevancy:** does the answer actually address the question (an answer can be faithful to context and still off-topic).

**LLM-as-judge** ([Zheng 2023, MT-Bench](https://arxiv.org/abs/2306.05685)). Human eval does not scale, so a strong LLM grades outputs against a rubric or pairwise ("which answer is better"). Known biases you must design around:

- **Position bias:** favors the first answer in pairwise comparison. Fix: judge both orders, average.
- **Verbosity bias:** favors longer answers. Fix: rubric explicitly scores conciseness or length-controls.
- **Self-preference:** favors text from its own model family. Fix: use a different model as judge.

**Validate the judge before trusting it:** grade a sample with humans, measure judge-human agreement, and only automate once agreement is high. An unvalidated judge is just a second opinion of unknown quality. Pin the judge model and rubric version, or your eval numbers move when the judge changes.

**The eval system, not just the metric:** a versioned golden dataset (curated + hard cases + past production failures), run on every change (prompt, model, retrieval config) like a unit-test suite in CI, plus online sampling of production traffic graded by the judge for continuous quality monitoring.

---

# LLM Inference Optimization

**Two phases with different bottlenecks.** **Prefill** (process the prompt) is compute-bound and parallel across tokens. **Decode** (generate token by token) is memory-bandwidth-bound: each new token re-reads all the weights. This split explains most serving behavior, including why time-to-first-token (TTFT) and tokens-per-second are tuned differently.

**KV cache.** At each decode step, attention needs the keys and values of all previous tokens. Recomputing them each step would be quadratic, so they are cached. The cost is memory:

`KV bytes per token = 2 (K and V) × n_layers × d_model × bytes_per_value`

For a 70B-class model (80 layers, d_model 8192, FP16): `2 × 80 × 8192 × 2 ≈ 2.6 MB per token`, so a 4k-token conversation holds ~10GB of KV cache for one request. This is why **GQA** (8x reduction) and KV quantization exist, and why concurrency, not weights, often limits an inference server.

**Continuous batching** ([vLLM / PagedAttention, Kwon 2023](https://arxiv.org/abs/2309.06180)). Naive batching waits for the whole batch to finish, but sequences end at different times. Continuous batching admits and retires sequences at every decode step, keeping the GPU full. **PagedAttention** allocates KV cache in small pages (like virtual memory) instead of contiguous max-length blocks, eliminating fragmentation. Together: 2 to 4x throughput. This is what vLLM/TGI give you out of the box.

**Speculative decoding** ([Leviathan 2022](https://arxiv.org/abs/2211.17192)). A small draft model proposes k tokens cheaply; the large model verifies all k in **one** forward pass (verification is parallel, generation is not), keeping the longest accepted prefix. Output is provably identical to the large model alone, at 2 to 3x speed when the draft is usually right.

**Prefix / prompt caching.** Identical prompt prefixes (system prompt, few-shot examples, conversation history) can reuse their KV cache across requests, making prefill for the shared part nearly free. Design consequence: put the stable content first and the variable content last, and never rewrite the early prompt mid-session.

**GPU sizing for a 70B model (the estimation drill).**

- Weights: FP16 = `70B × 2 bytes = 140GB` → does not fit one 80GB GPU; needs 2x for weights alone, **4x A100/H100 80GB** for realistic serving with KV cache headroom (tensor parallel).
- INT4 quantized: `≈ 35GB` → fits a single 80GB GPU with room for cache.
- Embedding storage drill: `1B vectors × 768 dims × 4 bytes = ~3TB` FP32 (1.5TB FP16, ~64GB with PQ at 64 bytes/vector). This is why quantization is not optional at billion scale.

---

# Test-Time Compute and Reasoning Models

The third scaling axis. After pretraining compute and data, the frontier discovered that **spending more compute at inference** reliably buys accuracy on reasoning tasks ([Snell 2024](https://arxiv.org/abs/2408.03314): test-time compute can outperform a 14x larger model on the right problems). Every major lab now ships this: OpenAI's o-series, DeepSeek-R1, Gemini thinking, Claude extended thinking.

**The technique ladder, cheap to expensive:**

- **Chain-of-thought:** ask for reasoning before the answer. Nearly free, large gains on multi-step problems.
- **Self-consistency** ([Wang 2022](https://arxiv.org/abs/2203.11171)): sample k reasoning chains at temperature, take the **majority vote** of the final answers. Cost × k, accuracy up because independent errors cancel; the answer agreement rate doubles as a confidence signal.
- **Best-of-n with a verifier:** generate n candidates, score each with a verifier or **process reward model**, return the best. The PRM version ([Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)) scores each reasoning step, catching chains that wander early.
- **Reasoning models:** long chain-of-thought trained in with RLVR ([DeepSeek-R1](https://arxiv.org/abs/2501.12948)) so the model natively "thinks" before answering, with a **thinking budget** exposed as an API dial (Claude extended thinking, Gemini thinking budgets).

**Design consequences (what an interviewer wants you to say):**

- Thinking tokens are decode tokens: latency and cost scale with the budget, so **route by difficulty**: no thinking for easy queries, large budgets only for hard ones. A difficulty classifier in front of the router pays for itself immediately.
- In agents, thinking helps most at **decision points**: before irreversible actions, after surprising tool results, when planning. Anthropic exposes exactly this as interleaved thinking between tool calls.
- Diminishing returns are real: accuracy vs thinking-budget curves flatten, so the budget is a tunable knob on a cost/quality curve, not "more is always better."
- **Distill the reasoning:** R1-style pipelines train small models on the reasoning traces of large ones, recovering much of the quality at a fraction of the serving cost.

---

# LLM Safety and Guardrails

**Jailbreak vs prompt injection, the distinction that matters.** A **jailbreak** is the *user* attacking the model's own policy ("ignore your rules"). **Prompt injection** is *third-party content* attacking the *system*: instructions hidden in a webpage, email, or document that the model processes as if they came from the operator. Injection is the dangerous one for agents and RAG, because the attacker does not need access to the prompt, only to content the system will read.

**Layered defense (no single layer works):**

- **Input layer:** moderation classifier on user input, PII detection/redaction before anything is logged or embedded. Against jailbreaks specifically, **Constitutional Classifiers** ([Anthropic 2025](https://arxiv.org/abs/2501.18837)): lightweight input/output classifiers trained on constitution-generated data, which held against thousands of hours of human red-teaming.
- **Prompt layer:** clear separation of instructions from data (delimiters, structured roles); useful hygiene but weak alone, since models do not reliably honor the separation.
- **Output layer:** moderation on generations, groundedness check against retrieved context (catches hallucination), regex/classifier scan for secrets and PII before the response leaves the system.
- **Action layer (the strong one):** tool allow-lists, least-privilege credentials per session, human approval for irreversible actions (sending money, deleting data, external messages), sandboxed execution for generated code.

**Operate it like security, not like a feature:** maintain a red-team suite of known attacks, measure **ASR (attack success rate)** before and after every defense change, and re-run it on every model upgrade, because safety behavior shifts between model versions.

---

# Agentic AI Systems

**The core problem: error compounding.** An agent that is 99% reliable per step succeeds at a 50-step task `0.99^50 ≈ 61%` of the time; at 95% per step, `0.95^50 ≈ 8%`. Long-horizon reliability is THE hard problem, and most agent architecture is an attempt to fight `p^N`: fewer steps, better steps, or recovery from bad steps.

**What each frontier lab's signature move is** (interview-ready attribution):

| Lab | Signature contributions to agents |
|---|---|
| **Anthropic** | Context engineering doctrine, orchestrator-worker multi-agent, MCP, computer use, constitutional classifiers, interleaved/extended thinking |
| **Google / DeepMind** | CaMeL injection defense, A2A protocol, ScaNN retrieval infra, Gemini thinking budgets |
| **OpenAI** | Test-time compute at scale (o-series), function calling as the tool-use standard, deep research |
| **DeepSeek** | R1: RLVR at scale with an open recipe, reasoning distillation |
| **Meta** | Open weights (Llama) powering most self-hosted agents, Toolformer lineage for tool use |
| **Cognition / Manus** | Single-agent context discipline, file system as memory, KV-cache-aware agent design |

**Workflow vs agent** ([Anthropic, Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents)). A workflow is a fixed DAG with LLM calls inside (predictable, debuggable, cheap); an agent chooses its own next action in a loop (flexible, expensive, compounding errors). Default to workflows; use an agent only when the path genuinely cannot be known in advance. Most production "agents" should be workflows.

**The core loop is ReAct** ([Yao 2022](https://arxiv.org/abs/2210.03629)): interleave reasoning, tool calls, and observations until done. **CodeAct** ([Wang 2024](https://arxiv.org/abs/2402.01030)) upgrades the action space from JSON tool calls to executable code, which composes operations in one step (fewer steps, less compounding). **Computer use** (Anthropic) extends the action space to screenshots, clicks, and keystrokes, benchmarked by [OSWorld](https://arxiv.org/abs/2404.07972) and [WebArena](https://arxiv.org/abs/2307.13854); grounding (clicking the right pixel) and recovery from unexpected UI states are the hard parts. Tool-use foundations: [Toolformer](https://arxiv.org/abs/2302.04761), [Gorilla](https://arxiv.org/abs/2305.15334).

**Context engineering** ([Anthropic, Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)). Context is the scarce resource. Long-running agents accumulate history that overflows the window and rots (stale tool outputs, dead ends). Strategies: **keep / evict / summarize** per item, **compaction** (summarize old turns into a running state note, with paging between in-context and external memory, [MemGPT, Packer 2023](https://arxiv.org/abs/2310.08560)), and **sub-agent isolation** (give a subtask a fresh clean context, return only the conclusion).

**Memory architectures.** Episodic (what happened), semantic (facts learned), procedural (how to do things, skills, [Voyager](https://arxiv.org/abs/2305.16291)). Backed by external stores: vector DB for recall, graph memory (GraphRAG / Zep) for entity relationships. The classic demonstration of memory + reflection driving believable long-horizon behavior: [Generative Agents, Park 2023](https://arxiv.org/abs/2304.03442).

**Improving the steps.** **Reflexion** ([Shinn 2023](https://arxiv.org/abs/2303.11366)): after a failure, the agent writes a self-critique into memory and retries. **LATS** ([Zhou 2023](https://arxiv.org/abs/2310.04406)): tree search over action sequences instead of a single greedy trajectory, buying reliability with tokens. Both are test-time compute applied to agents.

## Tool Design and the Agent Harness

The insight from [SWE-agent](https://arxiv.org/abs/2405.15793): the **agent-computer interface (ACI) matters as much as the model.** The same model jumps double-digit points on the same tasks when its tools are designed for an LLM instead of for a human. The harness (tools, prompts, context management around the model) is engineering you own; the labs treat it as the highest-leverage layer ([Anthropic, Writing effective tools for agents](https://www.anthropic.com/engineering/writing-tools-for-agents)):

- **Few, well-scoped tools beat many overlapping ones.** Every tool schema consumes context and adds a wrong-choice opportunity. Consolidate (one `search` with a source parameter, not five searches), and **namespace** related tools so they group legibly.
- **Descriptions are prompts.** The tool description is the only documentation the model gets; write it like you would explain the tool to a new hire, including when NOT to use it.
- **Token-efficient returns.** Return compact, semantically useful results (ids + names + one-line summaries, paginated), never raw dumps; a 50k-token tool result destroys the context budget that later steps need.
- **Errors must teach.** A failed call should return what went wrong and what to try instead ("date must be YYYY-MM-DD"), because the error message is the only feedback the model gets before retrying. Make tools idempotent so retries are safe.
- **Progressive disclosure** ([Anthropic, Agent Skills](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)): do not front-load every instruction; store capability docs (skills) on disk with one-line summaries in context, and let the agent read the full doc only when the task needs it. Scales capability without scaling the prompt.
- **The file system as memory** ([Manus, context engineering lessons](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)): write intermediate state, plans, and large artifacts to files and re-read on demand, keeping the context **append-only** (never rewrite earlier turns) so the KV cache stays valid across steps. Append-only context is simultaneously a correctness practice and a 10x cost lever.

## Multi-Agent Systems

**Single vs multi, the real trade-off:** orchestrator-worker ([Anthropic, multi-agent research system](https://www.anthropic.com/engineering/built-multi-agent-research-system)) improves accuracy on parallelizable research-type tasks at ~15x token cost; the counterargument ([Cognition, Don't Build Multi-Agents](https://cognition.ai/blog/dont-build-multi-agents)) is that splitting context across agents loses the shared state coding-type tasks need, so a single agent with great context management wins there. The deciding question: **is the task parallelizable into independent subtasks, or does it need one coherent evolving state?**

**Patterns:** orchestrator-worker (planner delegates to parallel workers), planner-executor (separate the plan from the doing), reflection/critic (a second model reviews before commit), debate (independent answers, then reconcile). Sub-agents also serve as **context firewalls**: a worker burns 50k tokens exploring and returns a 200-token conclusion to the orchestrator.

**Coordination hazards (where multi-agent designs die):** two agents editing the same artifact, stale views of shared state, and lost updates. Mitigations are classic distributed-systems moves: single-writer ownership per artifact, message passing over shared memory, and checkpointed state so a crashed worker resumes instead of restarting.

**Interoperability:** [MCP](https://modelcontextprotocol.io) standardizes agent-to-tool connection (one server per tool source, any client); [A2A](https://github.com/a2aproject/A2A) targets agent-to-agent interop across vendors. With hundreds of tools, **tool retrieval** (search over tool descriptions, expose only the relevant subset per step) replaces stuffing every schema into context.

## Agent Evaluation

- **pass@k vs pass^k.** pass@k = P(at least one of k tries succeeds): measures *capability*. **pass^k** = P(all k i.i.d. tries succeed) ≈ p̂^k: measures *reliability/consistency*, introduced by [τ-bench](https://arxiv.org/abs/2406.12045) because a customer-facing agent that succeeds 4 times out of 5 is a liability, not a success.
- **Outcome vs process reward.** Outcome (did the task succeed) is cheap to verify but gives no credit assignment over a long trajectory; process reward (was each step good, PRM, [Let's Verify Step by Step, Lightman 2023](https://arxiv.org/abs/2305.20050)) supervises the trajectory but is expensive to label.
- **RLVR** (RL with verifiable rewards): train only on tasks where success is mechanically checkable (tests pass, answer matches), eliminating reward-model gaming. The [DeepSeek-R1](https://arxiv.org/abs/2501.12948) recipe.
- **Trajectory eval in practice:** prefer **final-state verification** (check the database/file system actually changed correctly) over judging the transcript; add stepwise checks for tool-call validity, loop detection (same action repeated), and recovery rate after a failed step.
- **Benchmarks:** [SWE-bench Verified](https://arxiv.org/abs/2310.06770) (real GitHub issues), [GAIA](https://arxiv.org/abs/2311.12983) (multi-step tool reasoning), [τ-bench](https://arxiv.org/abs/2406.12045) (tool agent + simulated user), [WebArena](https://arxiv.org/abs/2307.13854) / [OSWorld](https://arxiv.org/abs/2404.07972) (web and computer use).
- **Observability:** trace every step as spans ([OpenTelemetry GenAI conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/); Langfuse / Phoenix as implementations). You cannot debug a 30-step failure from the final answer; the trace is the debugging surface and the future eval dataset.

## Agent Safety

**Prompt injection** is the #1 unsolved agent-security problem. The **lethal trifecta** ([Willison](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/)): an agent with (1) access to private data, (2) exposure to untrusted content, and (3) the ability to exfiltrate can be hijacked by instructions hidden in content it reads. Remove any one leg and the attack collapses. **CaMeL** ([DeepMind 2025](https://arxiv.org/abs/2503.18813)) is the by-design defense: a privileged LLM writes a plan as code from the trusted request, a quarantined LLM parses untrusted data into typed values that **cannot introduce new actions**, and a capability system tracks data provenance through execution. Measure all of it with **ASR** against a red-team suite, before and after defenses.

## Agent Economics

Agents are token furnaces; the levers: **KV-cache reuse across steps** (stable prompt prefix makes each step's prefill nearly free; append-only context, never rewrite the system prompt mid-trajectory), **model routing** (cheap model for easy steps, escalate on failure), **parallel tool calls**, and **durable execution** (checkpoint the trajectory so a crash resumes instead of re-running; Temporal-style workflow engines are increasingly the backbone under long-running agents).

---

# Worked Design Blueprints

The four designs that cover most AIML SD interviews. Each runs the same framework; what changes is where the hard part lives.

## 1. Video recommendation feed (YouTube/TikTok style)

- **Clarify:** optimize long-term satisfaction proxied by watch-time + explicit signals, not raw CTR; ~10⁸ users, ~10⁸ items, <200ms end-to-end.
- **Architecture:** multi-source candidate gen (two-tower ANN + item-to-item + fresh + followed) → union ~1000 → multi-task ranker (MMoE heads: click, watch, like, dissatisfaction) → blended score → re-rank (diversity MMR, freshness, exploration slots).
- **Data/features:** implicit labels with thresholds (watch > 30s), windowed aggregates from the feature store, position-debiased training.
- **Eval:** recall@k for candidate gen, NDCG for ranker offline; A/B on retention-leaning metrics with CTR and p99 latency as guardrails.
- **Hard parts to lead with:** feedback loops (exploration budget), cold-start items (content towers + bandits), train/serve skew on the aggregate features, calibration across heads.

## 2. Enterprise RAG assistant (internal knowledge, with citations)

- **Clarify:** correctness and trust over fluency; must cite; must respect per-document permissions; freshness within minutes.
- **Architecture:** streaming ingest → recursive chunking (300 to 800 tokens, overlap) → embeddings + BM25 dual index → hybrid retrieve (RRF) with **ACL filter inside the query** → cross-encoder rerank to top 5 to 10 → grounded generation with chunk-id citations → groundedness check on output.
- **Eval:** golden dataset in CI (context precision/recall, faithfulness, answer relevancy via validated LLM judge); online sampling graded continuously.
- **Hard parts:** permission enforcement at query time, index freshness (tombstones, reingest), lost-in-the-middle (order chunks), and the failure split (retrieval miss vs generation hallucination need different fixes).

## 3. LLM inference serving platform

- **Clarify:** target TTFT (~200ms) and tokens/sec per user; multi-tenant; cost per 1M tokens is the business metric.
- **Architecture:** request router (model routing: small model default, escalate on complexity; thinking budget by difficulty) → continuous-batching engine (vLLM, PagedAttention) → tensor-parallel replicas for the big model → prefix cache for shared system prompts → multi-LoRA for per-tenant customization → streaming responses.
- **Capacity:** weights memory (140GB FP16 for 70B → 4x80GB), KV budget per concurrent stream (~2.6MB/token, GQA), batch size vs p99 trade.
- **Levers ranked:** quantization (INT8/4) → continuous batching → prefix caching → speculative decoding → distillation to a smaller default model.
- **Hard parts:** decode is bandwidth-bound so GPU "utilization" lies, long-prompt tenants starve the batch (need fair scheduling), reasoning-model thinking budgets blow up decode time, and cache invalidation on model/prompt version bumps.

## 4. Multi-tenant agent platform ("lots of agents")

- **Clarify:** what tasks, how long-horizon, what blast radius per action; reliability target per task (pass^k, not a demo).
- **Architecture:** stateless agent runtime (LLM loop) + **durable state store** (trajectory checkpoints, resumable) + tool gateway (MCP servers, allow-lists, per-session least-privilege credentials) + sandboxed code execution + memory tiers (in-context → compacted summary → vector/graph store, files as working memory) + trace pipeline (OTel spans) feeding the eval system.
- **Orchestration:** workflows by default; orchestrator-worker only for parallelizable tasks; sub-agents as context firewalls; single-writer rule on shared artifacts.
- **Eval:** offline benchmark slice (τ-bench style) in CI with pass^k, final-state verification; online intervention/escalation rate; red-team ASR tracked per release.
- **Safety/economics:** lethal-trifecta analysis per tool grant, CaMeL-style separation for untrusted content, human approval on irreversible actions; KV-cache-friendly append-only prompts, model routing, parallel tool calls.
- **Hard parts to lead with:** error compounding (p^N) drives everything: fewer steps (CodeAct), better steps (thinking at decision points, process eval), recovery (checkpoints, Reflexion), and the eval harness that proves any of it moved.

---

# Estimation Cheat Sheet

The numbers to carry into any estimation round:

- **Model memory (inference):** `params × bytes/param`. 70B: FP16 = 140GB, INT8 = 70GB, INT4 = 35GB. GPU = 80GB (A100/H100).
- **Training memory:** ~16 to 20 bytes/param with Adam (weights + grads + FP32 master + moments) → 70B ≈ 1.1TB+ of state → sharding is mandatory.
- **Training compute:** `FLOPs ≈ 6 × params × tokens`. Chinchilla-optimal data ≈ 20 tokens/param. A100 ≈ 312 TFLOPS BF16, H100 ≈ ~1000 TFLOPS; assume 30 to 50% utilization (MFU).
- **KV cache:** `2 × n_layers × d_model × bytes × tokens`. 70B-class ≈ 2.6 MB/token FP16 (≈ 0.3 MB with GQA). 4k context ≈ 10GB/request.
- **Embeddings at rest:** `count × dims × bytes`. 1B × 768 × 4B = 3TB FP32; PQ at 64B/vector = 64GB.
- **Tokens:** ~0.75 words/token. A page ≈ 500 to 800 tokens.
- **Throughput shape:** prefill compute-bound, decode bandwidth-bound; continuous batching 2 to 4x; speculative decoding 2 to 3x; prefix caching makes shared-prefix prefill ~free.
- **Funnel shape:** millions → ANN retrieval (≈10ms) → hundreds → ranker (≈20 to 40ms) → tens → re-rank, all inside a ~100ms budget.

---

# Reading List (canonical sources, in study order)

**ML systems:** [Rules of ML](https://developers.google.com/machine-learning/guides/rules-of-ml) · [Hidden Technical Debt](https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) · [YouTube Recommendations (Covington 2016)](https://research.google/pubs/deep-neural-networks-for-youtube-recommendations/) · [Two-tower retrieval (Yi 2019)](https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/) · [Wide & Deep](https://arxiv.org/abs/1606.07792) · [MMoE](https://dl.acm.org/doi/10.1145/3219819.3220007) · [HNSW](https://arxiv.org/abs/1603.09320) · [ScaNN](https://arxiv.org/abs/1908.10396) · [Chip Huyen's blog](https://huyenchip.com/blog/)

**LLM core:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762) · [Chinchilla](https://arxiv.org/abs/2203.15556) · [FlashAttention](https://arxiv.org/abs/2205.14135) · [GQA](https://arxiv.org/abs/2305.13245) · [Mixtral (MoE)](https://arxiv.org/abs/2401.04088) · [LoRA](https://arxiv.org/abs/2106.09685) · [QLoRA](https://arxiv.org/abs/2305.14314) · [InstructGPT (RLHF)](https://arxiv.org/abs/2203.02155) · [DPO](https://arxiv.org/abs/2305.18290) · [Constitutional AI](https://arxiv.org/abs/2212.08073) · [vLLM/PagedAttention](https://arxiv.org/abs/2309.06180) · [Speculative decoding](https://arxiv.org/abs/2211.17192) · [Lost in the Middle](https://arxiv.org/abs/2307.03172) · [GraphRAG](https://arxiv.org/abs/2404.16130) · [RAGAS](https://arxiv.org/abs/2309.15217) · [LLM-as-judge / MT-Bench](https://arxiv.org/abs/2306.05685) · [Self-consistency](https://arxiv.org/abs/2203.11171) · [Scaling test-time compute (Snell 2024)](https://arxiv.org/abs/2408.03314) · [Constitutional Classifiers](https://arxiv.org/abs/2501.18837)

**Agents:** [Building Effective Agents (Anthropic)](https://www.anthropic.com/engineering/building-effective-agents) · [ReAct](https://arxiv.org/abs/2210.03629) · [CodeAct](https://arxiv.org/abs/2402.01030) · [SWE-agent (ACI)](https://arxiv.org/abs/2405.15793) · [Writing effective tools for agents (Anthropic)](https://www.anthropic.com/engineering/writing-tools-for-agents) · [Agent Skills (Anthropic)](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) · [Context engineering (Anthropic)](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) · [Context engineering lessons (Manus)](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus) · [MemGPT](https://arxiv.org/abs/2310.08560) · [Generative Agents](https://arxiv.org/abs/2304.03442) · [Voyager](https://arxiv.org/abs/2305.16291) · [Reflexion](https://arxiv.org/abs/2303.11366) · [LATS](https://arxiv.org/abs/2310.04406) · [Multi-agent research system (Anthropic)](https://www.anthropic.com/engineering/built-multi-agent-research-system) · [Don't Build Multi-Agents (Cognition)](https://cognition.ai/blog/dont-build-multi-agents) · [MCP](https://modelcontextprotocol.io) · [A2A](https://github.com/a2aproject/A2A) · [τ-bench](https://arxiv.org/abs/2406.12045) · [SWE-bench](https://arxiv.org/abs/2310.06770) · [GAIA](https://arxiv.org/abs/2311.12983) · [OSWorld](https://arxiv.org/abs/2404.07972) · [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) · [DeepSeek-R1](https://arxiv.org/abs/2501.12948) · [Lethal trifecta (Willison)](https://simonwillison.net/2025/Jun/16/the-lethal-trifecta/) · [CaMeL](https://arxiv.org/abs/2503.18813) · [OTel GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
