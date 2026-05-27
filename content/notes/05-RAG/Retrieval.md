---
title: "Retrieval Mechanisms in RAG"
date: 2026-05-27
summary: "A practical deep-dive into how retrieval actually works in RAG systems — from BM25 and dense vectors to hybrid fusion, re-ranking, graph traversal, and agentic correction loops."
tags: [RAG, Retrieval, VectorDB, LLM, AI]
---

If you've ever shipped a RAG pipeline and watched it confidently hallucinate something that was sitting right there in your documents — this note is for you.

Retrieval is not "just a vector search." It is a layered evidence pipeline that decides *what facts the LLM is allowed to see*. Get it wrong and your generator is working with garbage. Get it right and even a small model can reason well over large corpora.

This is not a survey paper. It's a working note on what the stages actually do, why they exist, and what breaks without them.

---

## How the full pipeline fits together

Every production RAG system follows the same skeleton, even if individual pieces look different. The user's natural-language query enters on the left; grounded, cited text comes out on the right. Between the two are a series of decisions about *what counts as evidence*.

```
┌─────────────┐    ┌──────────────────┐    ┌──────────────────────┐
│  User Query │───▶│  Query Rewriting │───▶│  Candidate Retrieval │
└─────────────┘    │  (expand, route, │    │  (sparse + dense +   │
                   │   decompose)     │    │   filters)           │
                   └──────────────────┘    └──────────┬───────────┘
                                                      │
                                           ┌──────────▼───────────┐
                                           │    Re-ranking         │
                                           │  (cross-encoder,      │
                                           │   LLM-based)          │
                                           └──────────┬───────────┘
                                                      │
                                           ┌──────────▼───────────┐
                                           │  Context Assembly    │
                                           │  (dedup, compress,   │
                                           │   citations)         │
                                           └──────────┬───────────┘
                                                      │
                                           ┌──────────▼───────────┐
                                           │     LLM Generator    │
                                           │   + Confidence Check │
                                           └──────────┬───────────┘
                                                      │
                                           ┌──────────▼───────────┐
                                           │  Corrective Loop     │
                                           │  (if weak, retry)    │
                                           └──────────────────────┘
```

Each stage exists because the previous stage has a known blind spot:

- **Query rewriting** exists because users phrase questions in ways that don't match how the corpus is written. "How do I cancel?" rarely matches a doc titled "Subscription Lifecycle Management."
- **Candidate retrieval** exists because you can't afford to score every document with an expensive model. You need a cheap, recall-oriented first pass.
- **Re-ranking** exists because cheap retrieval is imprecise. It returns roughly the right neighborhood, but the top result is often wrong.
- **Context assembly** exists because shoving raw chunks into a prompt wastes tokens and confuses the model. You need to deduplicate, order, and compress.
- **Confidence checking and corrective loops** exist because all of the above can silently fail, and a single LLM call has no way to know.

The key insight is that **retrieval and generation are separate quality bars**. You can evaluate retrieval before a single LLM token is generated. Most RAG failures are retrieval failures dressed up as generation failures.

---

## The math that drives retrieval

You don't need to implement these from scratch, but understanding them tells you *when* each method breaks and why combining them is not redundant.

### BM25 — lexical relevance

BM25 is the workhorse of sparse retrieval, and forty years of information retrieval research keeps coming back to it because it just works. Given a query $q$ and document $d$ over corpus $D$, BM25 scores their relevance as:

$$
\text{BM25}(d, q) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}
$$

where the **inverse document frequency** rewards rare terms over common ones:

$$
\text{IDF}(t) = \log\left( \frac{|D| - n_t + 0.5}{n_t + 0.5} + 1 \right)
$$

Reading this piece by piece:

- $f(t, d)$ — how many times term $t$ appears in document $d$. More occurrences are better, but not linearly.
- $n_t$ — number of documents in the corpus containing $t$. The more common a term is across the corpus, the smaller its contribution. The word "the" gets near-zero weight; a rare product code gets a large weight.
- $|d|$ — length of document $d$ in tokens.
- $\text{avgdl}$ — average document length in the corpus.
- $k_1 \in [1.2, 2.0]$ — term saturation parameter. As $f(t,d)$ grows, the contribution flattens. A document mentioning "kubernetes" twenty times isn't ten times more relevant than one mentioning it twice.
- $b \approx 0.75$ — length normalization. Without this, long documents win just by accumulating more term hits. With $b = 1$, length is fully normalized; with $b = 0$, length is ignored entirely.

**When it shines:** exact product codes, legal clauses, API names, error messages, identifiers, and any domain where the *exact word matters*. If a user searches for "CVE-2024-3094" you want documents that contain that exact string, not documents that semantically resemble "software vulnerability."

**Where it fails:** "ML model" vs "machine learning algorithm" — BM25 scores these as zero overlap because they share no surface terms. It has no concept of synonyms, paraphrases, or related concepts.

---

### Dense similarity — semantic retrieval

Bi-encoders solve BM25's blind spot by mapping query and document into the same continuous vector space. Two passages that mean the same thing land near each other in that space, regardless of whether they share any words.

Given a query embedding $\mathbf{q} \in \mathbb{R}^m$ and a document embedding $\mathbf{d} \in \mathbb{R}^m$, similarity is typically computed as the cosine of the angle between them:

$$
\cos(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\|\mathbf{q}\| \, \|\mathbf{d}\|}
$$

When the vectors are already unit-normalized (which most embedding APIs do for you), this reduces to a plain dot product, also called **Maximum Inner Product Search** (MIPS):

$$
s(\mathbf{q}, \mathbf{d}) = \mathbf{q}^\top \mathbf{d}
$$

The crucial property is that **document embeddings can be precomputed**. At ingest time, you push every chunk through the encoder once. At query time, you only encode the query and then run approximate nearest-neighbor (ANN) search — HNSW, IVF, or product-quantized variants — against the precomputed index. ANN trades a small amount of recall for orders of magnitude more speed.

**When it shines:** paraphrase-heavy queries ("how do I get my money back" ↔ "refund policy"), concept matching, cross-lingual retrieval (one model can map English queries to German documents), and queries where users don't know the exact terminology.

**Where it fails:** distinctions that depend on a single token. "Use version 2.0, not 2.1" — the model sees both versions as semantically similar and may surface the wrong one. Negation is also a famous failure mode: "documents not about cats" still returns cat documents.

---

### Reciprocal Rank Fusion — merging rank lists without score calibration

When you run both sparse and dense retrieval, you have two ranked lists. Their raw scores live on completely different scales — BM25 might return values in $[0, 40]$, cosine similarity in $[-1, 1]$ — so you can't just add them. **Reciprocal Rank Fusion** sidesteps the problem by only using ranks:

$$
\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}
$$

where:

- $R$ is the set of retrievers being fused (BM25, dense, etc.)
- $\text{rank}_r(d)$ is the position of document $d$ in retriever $r$'s output (1, 2, 3, …)
- $k$ is a smoothing constant, almost always set to $k = 60$ in production

The smoothing constant matters more than it looks. With small $k$, the top few ranks dominate. With larger $k$, the fusion is more democratic — documents that appear in *both* lists at moderate ranks (say, 5th and 8th) start beating documents that rank 1st in only one list. **This is the entire reason hybrid retrieval works**: documents agreed on by multiple retrievers are usually the right answer.

The other property worth knowing: RRF requires no score normalization, no calibration, no learning. You can plug in any number of retrievers and it just works. This is why almost every production RAG system uses it.

---

### ColBERT-style late interaction — token-level matching at scale

Bi-encoders collapse an entire document into one vector. This averages away fine-grained details — a long document about Python that mentions Rust in one paragraph looks, in vector space, like a document mostly about Python. ColBERT keeps a vector *per token* and scores via maximum similarity at the token level:

$$
s(q, d) = \sum_{i} \max_{j} \; \mathbf{q}_i^\top \mathbf{d}_j
$$

Read this as: for each query token $i$, find the document token $j$ that best matches it (the "MaxSim" operation), then sum those best-match scores across all query tokens.

Why this matters: each query term gets to "find its best evidence" in the document independently. A rare keyword that appears once in a long document still contributes its full weight, instead of being averaged down. This recovers some of the precision of BM25 while preserving the semantic flexibility of embeddings.

The cost is storage — you store many vectors per document instead of one — and slightly more compute at query time. ColBERT and its successors (PLAID, ColBERTv2) compress these multi-vectors aggressively to keep the index practical.

**When to use it:** domains where a specific term's presence is decisive but you also need paraphrase robustness. Technical documentation, scientific search, and legal retrieval often benefit.

---

## The three retrieval families

These three families are not competing options. They are complementary primitives, and most serious systems use at least two.

### Sparse retrieval (lexical precision)

BM25 and inverted-index queries — Elasticsearch, OpenSearch, Solr, Tantivy. The index maps each term to a posting list of documents that contain it; scoring is fast because you only look at documents containing query terms.

| Property | Detail |
|---|---|
| **Best for** | Exact terms, entity names, IDs, code identifiers, compliance phrases, error messages |
| **Weakness** | No semantic generalization — synonyms and paraphrases score zero |
| **Latency** | Sub-millisecond to single-digit milliseconds at scale |
| **Index cost** | Cheap to build, cheap to update, cheap to reason about |
| **Failure mode** | Misses obviously relevant documents that happen to use different vocabulary |

Sparse retrieval is dramatically underrated in modern RAG stacks. Many teams reach for vectors immediately and skip BM25, then wonder why their system can't find exact phrases.

---

### Dense retrieval (semantic recall)

Bi-encoder model produces fixed-length vectors; an ANN index (HNSW, IVF-PQ, ScaNN) finds nearest neighbors. Pinecone, Weaviate, Qdrant, Milvus, pgvector, Chroma — they all implement variants of this.

| Property | Detail |
|---|---|
| **Best for** | Concept matching, paraphrase-heavy queries, multilingual retrieval, "vibes-based" search |
| **Weakness** | Misses exact wording constraints; struggles with negation, numbers, identifiers |
| **Latency** | 5–50 ms depending on ANN parameters and corpus size |
| **Index cost** | Embedding compute at ingest; storage scales with vector dim and corpus size |
| **Failure mode** | Confidently retrieves topically related but factually wrong documents |

The embedding model matters enormously. A general-purpose model trained on web text will perform badly on legal contracts, medical notes, or code. Either fine-tune the embeddings on your domain or pick a domain-specific embedding model.

---

### Hybrid retrieval (the production default)

Run sparse and dense in parallel, fuse the results via RRF. This buys you both lexical precision and semantic recall in one query. The cost is running two retrievers and one fusion step, which is almost always worth it.

```
Query
  ├── BM25 → [d3, d7, d1, d9, ...]
  └── Dense ANN → [d7, d2, d3, d5, ...]
        ↓
       RRF fusion
        ↓
    [d7, d3, d2, d1, ...]   ← merged rank list
```

In published benchmarks and most production deployments, hybrid retrieval beats either retriever alone by 5–15% on recall and noticeably more on precision. The exception is highly homogeneous corpora where one or the other dominates — but even then, hybrid rarely *hurts*.

---

## Re-ranking: the second stage that earns its latency

First-pass retrieval (sparse, dense, or hybrid) is optimized for recall — you want the right answer to be *somewhere* in the top 100, even if it's not first. Re-ranking is the opposite — optimized for precision, taking those 100 candidates and identifying the 5–10 that actually belong in the prompt.

The standard approach is a **cross-encoder**. Where a bi-encoder embeds query and document independently, a cross-encoder takes them together as one input and produces a single relevance score:

$$
\text{score}(q, d) = f_\theta([\text{CLS}] \, q \, [\text{SEP}] \, d \, [\text{SEP}])
$$

Because the model attends jointly over both texts, it can model fine-grained dependencies — "does this passage actually answer the question, or just contain the same words?" — that bi-encoders fundamentally cannot capture.

The catch is that cross-encoders can't be precomputed. Every (query, document) pair must be encoded fresh at query time. This is roughly 10–100× more expensive per pair than bi-encoder scoring.

**Rule of thumb:** only re-rank after you've pruned to ≤ 200 candidates. Re-ranking 1000 documents in real-time will blow your latency budget. Re-ranking 50 is essentially free.

**Modern variants:** LLM-as-reranker (use a small instruction-tuned model with a prompt like "rate the relevance from 1–5"), and listwise rerankers that score all candidates together and exploit cross-candidate signals.

---

## Query transformation: making retrieval smarter before you search

The query the user types is rarely the optimal retrieval query. They use pronouns, omit context, mix multiple intents, or phrase things conversationally. Query transformation rewrites the user's input into something the retrieval system can actually match against.

| Technique | What it does | When to use |
|---|---|---|
| **HyDE** | Generate a hypothetical ideal answer with an LLM, embed *that*, retrieve passages similar to the hypothetical answer | Abstract or indirect questions where the query phrasing differs sharply from how the corpus is written |
| **Multi-query / RAG-Fusion** | Expand the query into N diverse reformulations, run retrieval on each, fuse the results | Broad queries with multiple valid phrasings, or when you want to maximize recall |
| **Step-back prompting** | First retrieve the general principle, then retrieve the specific detail | Reasoning-heavy queries where the answer depends on a concept that isn't named in the question |
| **Query decomposition** | Split a compound query into independent sub-questions, retrieve for each, combine | Multi-hop or comparative queries ("How does X differ from Y, and why?") |
| **Query routing** | Classify the query and dispatch it to the right index (code, docs, graph, image, policy KB) | Multi-index systems serving multiple data sources |

**HyDE in detail:** the failure case is that user queries and document text live in different stylistic registers. A user types "fix slow login" but the docs say "Troubleshooting authentication latency." HyDE bridges this by asking an LLM "write the ideal passage that would answer this query," embedding that hypothetical passage, and retrieving real passages near it. You're effectively translating from query-space to document-space before searching.

**A practical transformation pipeline:**

1. **Normalize.** Expand acronyms, standardize units, resolve typos, strip stop-phrases like "can you tell me." Cheap and high-impact.
2. **Detect intent.** Is this a factual lookup, a procedure request, a comparison, a diagnostic, a definition? Different intents benefit from different retrieval strategies.
3. **Expand to N reformulations.** Use an LLM to generate 3–5 paraphrases or sub-questions. Diversity matters more than fluency here.
4. **Execute in parallel.** Fire all the reformulations against the retrievers concurrently.
5. **Deduplicate and re-rank globally.** Merge results, drop duplicates by document ID, then run the re-ranker over the global candidate pool.

This stage often has a higher ROI than tuning the retriever itself.

---

## Chunking and indexing: the decisions that haunt you later

Most retrieval failures trace back to chunking decisions made at index time. The default "split into 512-token windows" pattern works surprisingly often but has well-known failure modes.

### Chunking strategies

| Strategy | Trade-off |
|---|---|
| **Fixed-size token windows** | Predictable and easy to reason about, but cuts mid-sentence and mid-paragraph in ways that destroy meaning |
| **Semantic boundary chunking** | Respects paragraph and section breaks, produces coherent chunks, but harder to tune for length |
| **Sliding window with overlap** | Reduces information loss at boundaries by overlapping chunks (e.g. 256-token stride with 512-token window), at the cost of a larger index |
| **Parent-child retrieval** | Index fine-grained chunks (200–400 tokens) for precise retrieval, but return the full parent section (1000–2000 tokens) to the LLM |
| **Document-level + section-level dual index** | Two layers of retrieval — find the right document first, then the right passage within it |

The parent-child pattern deserves special attention. The tension at index time is between **precision** (smaller chunks = more specific matches) and **context** (larger chunks = more information for the LLM to reason over). Parent-child resolves this by decoupling the two: index small for matching accuracy, return large for generation coherence.

### Contextualized chunk embeddings

Short chunks often have ambiguous meaning on their own. Consider a chunk that reads:

> "The maximum is 100 per minute."

Maximum of what? Per minute of what? On its own this is useless. Before embedding, prepend local metadata so the model knows the context:

```
[Source: Stripe API Docs | Section: Rate Limiting | Endpoint: /v1/charges]

The maximum is 100 per minute.
```

This dramatically improves disambiguation for short or context-poor passages. The metadata is *not* what gets returned to the LLM at retrieval time — it just sharpens the embedding. Anthropic's "contextual retrieval" technique formalizes this and reports 35%+ recall improvement on some benchmarks.

### Hierarchical retrieval (RAPTOR-style)

For long-form corpora, build recursive summaries over document clusters and store them at multiple levels of the index. Each leaf is a raw chunk; each internal node is an LLM-generated summary of its children, recursively up to a few top-level summaries per document.

At query time, retrieve across all levels:

- High-level summary nodes answer broad queries ("what is this document about?")
- Mid-level nodes answer scoped queries ("what does this section cover?")
- Leaf nodes answer specific factual questions ("what is the maximum retry count?")

This works well for technical books, long reports, and multi-document knowledge bases where pure flat chunking misses the forest for the trees.

### Metadata filters (apply before vector search)

Hard filters are the cheapest performance and quality win in any retrieval system. Apply them *before* the ANN search runs:

- **Tenant / permission scoping.** Multi-tenant SaaS systems must scope by tenant ID. Don't rely on post-filtering — apply at index-query time so you never even score the wrong tenant's data.
- **Time range.** "Only documents from the last 90 days" filters out stale results before ranking.
- **Language.** Don't return French docs for an English query unless the user opted in.
- **Document status.** Published vs draft, current vs archived, active vs deprecated.
- **Domain or product area.** When one index serves multiple products, filtering by product is faster and more accurate than letting the embedding model learn the distinction.

Filtering post-retrieval is wasteful and often *wrong* — you can score all your top-k results, only to find none of them pass the filter, leaving you with empty results when relevant documents existed further down the rank list.

---

## GraphRAG: when relationships matter more than proximity

Vector retrieval finds documents that are *semantically close* to a query. Graph retrieval finds documents that are *logically connected* through a chain of facts. These are different problems, and trying to solve one with the other fails in characteristic ways.

**The process at a high level:**

1. **Extract** entities and relations from your corpus during indexing. This is typically done with an LLM pass that emits (subject, predicate, object) triples.
2. **Build** a knowledge graph $G = (V, E)$ where vertices $V$ are entities and edges $E$ are typed relationships.
3. **Resolve** the query's entities and intent at query time — which nodes does the user actually care about?
4. **Traverse** the relevant subgraph via expansion or path search around those nodes.
5. **Serialize** the subgraph evidence — selected nodes, edges, and supporting text — into a grounded context for the LLM.

A useful way to score paths through the graph combines node-level relevance and edge-level confidence:

$$
\text{score}(\text{path}) = \sum_{i} \alpha_i \cdot \text{nodeRel}(v_i) + \sum_{j} \beta_j \cdot \text{edgeConf}(e_j)
$$

where $\alpha_i$ and $\beta_j$ are weighting coefficients you tune empirically, $\text{nodeRel}(v_i)$ measures how relevant a node is to the query, and $\text{edgeConf}(e_j)$ measures how confidently the relation was extracted in the first place. Paths with weakly extracted edges shouldn't dominate the answer.

**When to reach for GraphRAG:**

- **Multi-hop questions** that require chaining facts: "What companies does the CEO of Company X also sit on the board of?" — pure vector retrieval finds documents about Company X's CEO, but cannot follow the link to the other companies.
- **Provenance questions** where the user wants to understand how facts are connected: "How does regulation A affect requirement B via standard C?"
- **Entity-heavy domains** with rich relational structure — biomedical literature, legal precedent, supply chain, financial networks, knowledge management.

**When not to:** pure factual lookup with single-hop queries. GraphRAG adds significant indexing complexity (entity extraction, deduplication, ontology maintenance). If you're answering "what's our refund policy?" you don't need a graph.

In practice, the most robust systems combine GraphRAG with vector retrieval: use graph traversal to find entities and relationships, then use vector search within the resulting subgraph to find specific evidence passages.

---

## Multimodal retrieval

Text isn't the only input modality anymore. Modern systems routinely deal with images, document layouts, audio, and tables.

### Cross-modal text-image retrieval (CLIP-style)

CLIP, SigLIP, and similar models are trained to project text and images into a shared embedding space, where matching pairs land near each other. The same query can retrieve across modalities:

- Text query → image candidates ("find me the diagram showing the auth flow")
- Image query → text/document candidates ("find docs related to this screenshot")
- Image query → image candidates (visual similarity search)

```
User query (text)
    │
    ▼
Text Encoder ──────────────────┐
                               ▼
                     Shared Embedding Space
                               ▲
Image Encoder ─────────────────┘
    │
    ▲
Document images, photos
```

Works well for product catalogs, design assets, documentation with screenshots, and any domain where a picture conveys what words can't.

### Document-native retrieval (ColPali-style)

Traditional pipelines extract text from PDFs and lose almost everything else — table structure, equations, figure context, spatial layout, font emphasis. ColPali and its successors instead render each page as an image and embed the page directly using a vision-language model.

The retrieved unit is a *page image*, which preserves layout. The downstream LLM (a VLM in this case) can then see tables, charts, and equations as they appear, not as garbled OCR text.

**Critical for:** scientific papers (equations, plots), financial reports (tables, charts), invoices (forms), slide decks (visual hierarchy), engineering drawings.

### Audio-native retrieval

Most audio RAG pipelines transcribe with ASR and then retrieve against the transcript. This works but discards everything non-lexical: speaker identity, prosody (the "Right?" vs "Right." problem), emotional tone, background noise, music cues, silence patterns.

Audio-native retrieval embeds raw waveforms or spectrogram features directly into a vector space. You can then query for "the part where the customer sounded angry" or "the segment with applause" — things ASR cannot represent.

---

## Corrective and agentic retrieval loops

Static one-shot retrieval is a ceiling. The best systems evaluate what they retrieved and decide whether to try again. This is the difference between a system that fails silently and one that fails *loudly* with a retry plan.

```
Initial retrieval
      │
      ▼
Relevance / coverage check
      │
   ┌──┴──────────────────┐
   │ Good                │ Weak
   ▼                     ▼
Generate           Query rewrite
   │                + Extra retrieval
   ▼                     │
Faithfulness check  ◀────┘
   │
   ├── Pass → Final answer
   └── Fail → Retry with corrective retrieval
```

| Pattern | What it does | When it triggers |
|---|---|---|
| **CRAG** (Corrective RAG) | Quality-evaluate retrieved context with a small classifier; trigger fallback retrieval from an alternate index or web search on low confidence | After initial retrieval, before generation |
| **Self-RAG** | Model learns *when* to retrieve, *how* to critique evidence, and *whether* to revise its draft answer | Multiple times during a single response |
| **FLARE** | Actively retrieves again when generation uncertainty rises mid-sentence — the model pauses when its token probabilities flatten and goes back for more evidence | During generation, dynamically |
| **Agentic orchestration** | A planner agent selects tools and retrievers dynamically based on intermediate findings | Throughout the entire query lifecycle |

**The economic argument:** corrective loops are expensive. Each loop adds 1–3 seconds of latency and multiple LLM calls. You don't want them on every query. The right design triggers them *only* when a cheap confidence signal indicates the initial retrieval was weak — low max-score from the retriever, low coverage of query entities, low cross-encoder confidence on the top result, or low faithfulness score on the draft answer.

Done right, corrective loops dramatically increase precision on the hard 10% of queries without taxing latency on the easy 90%.

---

## Evaluation: measure retrieval before generation

Don't mix retrieval quality into generation quality metrics. Evaluate them separately so you know which stage to blame when something goes wrong.

### Retrieval metrics

| Metric | What it measures |
|---|---|
| **Recall@k** | Fraction of queries where a relevant document appears in the top $k$ results. Answers: "are we finding the right document at all?" |
| **Precision@k** | Relevance density within the top $k$ results. Answers: "how much noise are we feeding the LLM?" |
| **MRR** (Mean Reciprocal Rank) | $\text{MRR} = \frac{1}{|Q|} \sum_q \frac{1}{\text{rank}_q}$ — emphasizes finding the right answer first |
| **nDCG@k** | Graded relevance with logarithmic rank discounting — handles "somewhat relevant" vs "perfectly relevant" |
| **Coverage** | For multi-part queries, what fraction of the required evidence appears in the retrieved set? |

**MRR in detail:**

$$
\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q}
$$

A correct answer at rank 1 contributes $1.0$; rank 2 contributes $0.5$; rank 10 contributes $0.1$. The metric punishes you harshly for not surfacing the right answer at the top.

**nDCG@k** generalizes this with graded relevance — useful when you have human judgments on a scale like 0–4 instead of just binary relevant/not-relevant.

### Generation-grounding metrics

| Metric | What it measures |
|---|---|
| **Faithfulness** | What fraction of claims in the answer are actually supported by the retrieved evidence? |
| **Citation correctness** | Does each citation actually point to the span it claims to? |
| **Answer completeness** | Are all required sub-questions in the query addressed? |
| **Hallucination rate** | Inverse of faithfulness — how often does the model invent facts not in the evidence? |

**A common mistake:** evaluating only end-to-end answer correctness and assuming retrieval is fine when answers look right. It often isn't. You might be getting right answers despite bad retrieval (the LLM has the fact memorized from pre-training), or wrong answers despite good retrieval (the prompt confused the model). Separate the metrics, separate the diagnoses.

---

## Production patterns and failure modes

### Latency vs. quality budget

A real production pipeline is layered by cost:

```
Fast (always)      ── ANN vector search + metadata pre-filters
Medium (selective) ── BM25 + RRF fusion on top candidates
Slow (last mile)   ── Cross-encoder re-ranking on top 50–200
Cache              ── Embeddings, hot queries, frequent top-k contexts
```

The principle: never run expensive operations on more candidates than necessary. ANN gives you 100 candidates in 10 ms; fuse with BM25 in 5 ms; re-rank with a cross-encoder for 100 ms. If you re-ranked 10,000 candidates, you'd spend 10 seconds and get marginal improvement.

### Common failure modes

- **Poor chunk boundaries.** A definition and its example get split into two chunks; neither scores high enough alone. Fix: semantic chunking, sliding overlap, parent-child retrieval.
- **Missing metadata filters.** Retrieval returns documents from the wrong tenant, wrong product, or wrong time window. Fix: filter before ANN search, never after.
- **Over-aggressive compression.** A context summarizer deletes the critical number, date, or condition. Fix: extract-don't-summarize for factual content; keep raw spans for cited claims.
- **Domain-mismatched embeddings.** General-purpose model trained on web text is used on clinical notes or legal contracts. Fix: domain-specific embedding model or fine-tune your own.
- **Citation mismatch.** Context window reordering causes citation indices to point to the wrong source. Fix: cite by stable document/span IDs, not by positional index.
- **Single-retriever bias.** Pure vector search misses exact-match queries; pure BM25 misses paraphrases. Fix: hybrid retrieval is the default for a reason.
- **Stale index.** Documents updated upstream but embeddings never refreshed. Fix: track content hashes; re-embed on change.

### Minimal robust default stack

If you're starting fresh, this configuration handles most production cases competently:

1. **Hybrid retrieval** — BM25 + dense ANN in parallel
2. **RRF fusion** — merge the rank lists without score-normalization headaches
3. **Cross-encoder re-ranker** — apply on top 100 candidates
4. **Parent-child context assembly** — small chunks for matching, full sections for generation, deduplicated
5. **Grounded answer + citation verifier** — check each claim against the cited span before returning
6. **Corrective loop** — triggered only on low-confidence cases, not on every query

You can extend this with GraphRAG, multimodal indexes, agentic orchestration, and HyDE-style query transformation as your data and use cases demand. But this six-step stack is the floor, not the ceiling.

---

## Quick design checklist

Before shipping a retrieval pipeline, walk through each of these:

- [ ] Chunking aligned with semantic boundaries, not arbitrary token counts?
- [ ] Permission and metadata filters applied *before* ANN search, not after?
- [ ] Both sparse and dense retrievers contributing meaningfully (measured by individual recall, not just combined)?
- [ ] Re-ranking applied only after candidate pruning to ≤ 200 results?
- [ ] Citations traceable to exact source spans, not just document IDs?
- [ ] Corrective loop defined with a *concrete* confidence threshold, not a vague heuristic?
- [ ] Retrieval evaluated *independently* from generation quality?
- [ ] Embedding model matches the domain of the corpus?
- [ ] Index re-embedded on content changes (no stale vectors pointing to updated docs)?

---

## The mental model

> **Retrieval is not one algorithm. It is a layered evidence system that combines lexical precision, semantic recall, structural reasoning, and continuous self-correction — each layer exists because the previous one has a known blind spot.**

Vector search finds what is *semantically close*. BM25 finds what is *lexically exact*. Re-ranking finds what is *actually relevant* given the full query context. Graph traversal finds what is *logically connected*. Corrective loops find what was *missed the first time*.

Strip out any layer and you've taken on a specific class of failure. The engineering question is always: which failure modes can your application tolerate, and which ones will your users punish you for?
