---
title: "Retrieval Mechanisms in RAG"
date: 2026-05-27
summary: "A practical deep-dive into how retrieval actually works in RAG systems, from BM25 and dense vectors to hybrid fusion, re-ranking, graph traversal, and agentic correction loops."
tags: [RAG, Retrieval, VectorDB, LLM, AI]
---

There's a particular kind of frustration that comes from watching a RAG pipeline confidently answer a question with a hallucinated fact when the correct answer was sitting in the knowledge base the entire time. The document was there. The retriever just didn't find it, or found it but ranked it fifth, or found it and then the context builder threw it away.

Most of the time when people say "my RAG is hallucinating," they mean their *retrieval* failed. The LLM is not the problem. It's faithfully generating text conditioned on bad evidence.

These are my notes on how the retrieval side actually works: the math, the failure modes, what I'd actually build.

---

## The pipeline, end to end

The overall shape of every RAG system is the same, even if the internals differ wildly:

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

What I find useful is to think of each stage as a response to a known failure in the previous one. Query rewriting exists because users rarely phrase things the way the corpus is written. "How do I cancel?" doesn't match a doc titled "Subscription Lifecycle Management." Candidate retrieval exists because scoring every document with an expensive model is impractical, so you do a cheap pass first. Re-ranking exists because the cheap pass is imprecise; it finds the right neighborhood but the top result is often wrong. Context assembly exists because dumping raw chunks into a prompt wastes tokens and genuinely confuses the model. And corrective loops exist because everything before them can fail quietly.

The most important thing here is that **retrieval and generation are separate quality problems**. You can evaluate retrieval before writing a single prompt. You can measure whether the right document was retrieved independently from whether the LLM phrased the answer well. Most teams conflate these and end up tuning the wrong thing.

---

## The math behind retrieval

I think it's worth understanding these formulas not to implement them, but to know *when* each approach breaks. That's what the math actually tells you.

### BM25

BM25 has been around since the 1990s and information retrieval research keeps circling back to it. Given a query $q$ and document $d$ across a corpus $D$:

$$
\text{BM25}(d, q) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}
$$

The IDF part rewards uncommon terms:

$$
\text{IDF}(t) = \log\left( \frac{|D| - n_t + 0.5}{n_t + 0.5} + 1 \right)
$$

Walking through the pieces: $f(t,d)$ is how often term $t$ appears in the document. $n_t$ is how many documents in the corpus contain that term, so common words like "the" get near-zero weight while rare identifiers get high weight. $k_1$ (usually 1.2 to 2.0) is a saturation parameter: a document mentioning "kubernetes" twenty times isn't ten times more relevant than one mentioning it twice. At some point more occurrences stop adding information. $b \approx 0.75$ handles length normalization so long documents don't just win by accumulating more hits.

It works extremely well for exact terms, product codes, error messages, and identifiers. If someone searches for a CVE number, you want the document containing that exact string, not the one that's "semantically similar to a vulnerability report." Where it completely falls apart is paraphrases and synonyms. "ML model" vs "machine learning algorithm" scores as zero overlap because they share no surface terms.

---

### Dense retrieval

Bi-encoders map query and document into the same vector space independently. Two passages that mean the same thing end up nearby regardless of shared vocabulary.

Given query embedding $\mathbf{q} \in \mathbb{R}^m$ and document embedding $\mathbf{d} \in \mathbb{R}^m$:

$$
\cos(\mathbf{q}, \mathbf{d}) = \frac{\mathbf{q} \cdot \mathbf{d}}{\lVert\mathbf{q}\rVert \, \lVert\mathbf{d}\rVert}
$$

When vectors are unit-normalized (most APIs do this for you), this reduces to a plain dot product, also called MIPS:

$$
s(\mathbf{q}, \mathbf{d}) = \mathbf{q}^\top \mathbf{d}
$$

The key property is that document embeddings are computed once at ingest time. At query time you only encode the query and run approximate nearest-neighbor search (HNSW, IVF-PQ) against the precomputed index. ANN gives up a tiny amount of recall for orders-of-magnitude speed.

Dense retrieval handles paraphrases well, multilingual queries, and cases where users don't know the exact terminology. It struggles with single-token distinctions, like "version 2.0 not 2.1", because the model sees those as semantically similar. Negation is a classic failure: "documents not about cats" still returns cat documents.

---

### Reciprocal Rank Fusion

When you run BM25 and dense retrieval in parallel, you have two ranked lists on completely different score scales. BM25 might give values in $[0, 40]$, cosine similarity in $[-1, 1]$. You can't just add them. RRF sidesteps this by ignoring raw scores entirely and only looking at ranks:

$$
\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}
$$

where $k = 60$ is a smoothing constant and $\text{rank}_r(d)$ is the document's position in retriever $r$'s output.

The smoothing constant is more important than it looks. Small $k$ means top-ranked documents dominate. With $k = 60$, documents that appear in *both* lists at moderate positions start beating documents that top only one list, which is usually what you want. The whole reason hybrid retrieval works is that documents multiple retrievers agree on are genuinely more likely to be correct than documents one retriever loves and the other ignores.

No normalization, no calibration, no training. You can add any retriever and it just works.

---

### ColBERT and late interaction

Bi-encoders compress an entire document into one vector. That averaging loses information. A long document about Python that mentions Rust once looks, in vector space, almost identical to a document that's purely about Python. ColBERT keeps one vector *per token* and scores via maximum similarity across token pairs:

$$
s(q, d) = \sum_{i} \max_{j} \; \mathbf{q}_i^\top \mathbf{d}_j
$$

For each query token $i$, it finds the document token $j$ that best matches it, then sums those best-match scores across all query tokens. A rare keyword appearing once in a long document still gets to contribute its full weight rather than being averaged away. You recover some of BM25's precision while keeping the semantic flexibility of embeddings.

The tradeoff is storage. You store many vectors per document instead of one, plus slightly more compute at query time. ColBERTv2 and PLAID compress these aggressively enough to make it practical. Worth considering for technical documentation, scientific search, or legal retrieval where a single term being present or absent changes the answer.

---

## Three retrieval families

These aren't competing options. They're complementary, and most real systems use at least two.

**Sparse retrieval** (BM25 and inverted index, via Elasticsearch or Tantivy) is fast, cheap to reason about, and extremely good at exact terms. It's dramatically underrated in modern stacks. Teams reach straight for vector DBs, then wonder why their system can't find exact phrases or product identifiers.

**Dense retrieval** (bi-encoder plus ANN, via Pinecone, Qdrant, or pgvector) handles paraphrases, concept matching, and multilingual queries. It's what people usually mean when they say "vector search." The embedding model matters a lot more than people expect. A general-purpose model trained on web text will perform badly on clinical notes or legal contracts. Either pick a domain-specific model or plan to fine-tune.

**Hybrid** runs both in parallel and merges via RRF. In most benchmarks and deployments this beats either retriever alone by 5 to 15% on recall and more on precision. The exception is very homogeneous corpora where one approach clearly dominates, but even then hybrid rarely hurts, and it only costs you one extra query.

```
Query
  ├── BM25 → [d3, d7, d1, d9, ...]
  └── Dense ANN → [d7, d2, d3, d5, ...]
        ↓
       RRF fusion
        ↓
    [d7, d3, d2, d1, ...]
```

Default to hybrid unless you have a specific reason not to.

---

## Re-ranking

First-pass retrieval is optimized for recall. You want the right document *somewhere* in the top 100, even if it's not first. Re-ranking flips this priority and asks: of these 100 candidates, which ones actually belong in the prompt?

A cross-encoder takes query and candidate together as a single input and produces a relevance score:

$$
\text{score}(q, d) = f_\theta([\text{CLS}] \, q \, [\text{SEP}] \, d \, [\text{SEP}])
$$

Because it attends over both simultaneously, it can answer "does this passage actually answer the question, or does it just contain the same words?" This is a distinction bi-encoders cannot make because they encode query and document independently. The cost is that cross-encoders cannot be precomputed. Every pair gets scored fresh at query time, which is 10 to 100 times more expensive per candidate than ANN scoring.

The practical rule: re-rank after you've already pruned to 200 candidates or fewer. Re-ranking 50 candidates is cheap. Re-ranking 1,000 will blow your latency budget and the improvement is marginal.

Newer variants include using a small instruction-tuned LLM as a reranker (prompt it to rate relevance 1 to 5) or listwise rerankers that score all candidates together to exploit cross-candidate signals.

---

## Query transformation

The query a user types is usually not the best query to send to the retriever. People use pronouns, skip context, mix intents, or phrase things the way a question sounds rather than the way an answer reads. Rewriting before retrieval often has higher ROI than tuning the retriever itself.

**HyDE** is the most interesting technique here. The failure case it targets: user queries and document text live in different stylistic registers. The user writes "fix slow login" but the docs say "Troubleshooting authentication latency." HyDE generates a hypothetical answer to the query using an LLM, not to answer the question, but to produce text that *looks like* the correct answer. Then it retrieves real passages near that hypothetical in embedding space. You're effectively translating from query-space into document-space before searching.

**Multi-query / RAG-Fusion** generates N diverse reformulations of the same question, runs retrieval on each, and fuses the results. Useful when you want to maximize recall on broad queries that have multiple valid phrasings.

**Query decomposition** splits a compound question into independent sub-questions. "How does X differ from Y, and why did they make different choices?" is actually two separate retrieval problems. Treating them as one usually means you get evidence for one half and miss the other.

**Query routing** dispatches to the right index. If you have a code index, a docs index, a policy KB, and an image index, routing the query to the right one saves latency and improves precision. A question about a code error should go to the code index, not the policy KB.

A simple pipeline that works in practice:

1. Normalize the query: expand abbreviations, fix obvious typos, strip filler phrases.
2. Detect intent: is this a factual lookup, a procedure, a comparison, a debug question?
3. Generate 3 to 5 reformulations (diversity beats fluency here).
4. Run retrievals in parallel.
5. Deduplicate by document ID, then re-rank the merged pool globally.

---

## Chunking: the decision you can't easily undo

Most retrieval failures I've seen trace back to chunking. The default "split every 512 tokens" approach works often enough to feel fine, then fails in specific ways that are hard to diagnose.

Fixed token windows are predictable but cut mid-sentence and mid-paragraph. A definition and its example end up in different chunks; neither scores well alone. Semantic boundary chunking respects paragraph breaks and produces more coherent units, but takes more effort to tune.

Sliding overlap, say a 512-token window with a 256-token stride, reduces boundary loss at the cost of a larger index. Worth the tradeoff for dense factual content.

**Parent-child retrieval** is the pattern I keep coming back to. You index fine-grained chunks (200 to 400 tokens) for precise matching, but when a chunk scores well, you return its full parent section (1,000 to 2,000 tokens) to the LLM. Precision at retrieval time, coherence at generation time. It decouples the two constraints rather than compromising between them.

### Contextualizing embeddings

Short chunks are often contextually useless on their own. Consider this chunk:

> "The maximum is 100 per minute."

Maximum of what? Per what minute? Before embedding, prepend metadata:

```
[Source: Stripe API Docs | Section: Rate Limiting | Endpoint: /v1/charges]

The maximum is 100 per minute.
```

The metadata doesn't get returned to the LLM. It just makes the embedding more specific. Anthropic published on this ("contextual retrieval") and reported 35%+ recall improvement on some benchmarks. It's cheap to implement.

### Hierarchical indexing (RAPTOR-style)

For long-form corpora like technical books, long reports, and large document collections, flat chunking misses the document's structure entirely. RAPTOR builds recursive LLM-generated summaries over document clusters and stores them at multiple levels. Broad queries hit the summary nodes; specific queries hit the leaf chunks. You get answers at the right level of granularity without tuning a single threshold.

### Metadata filters

Apply filters *before* the ANN search, not after. This is one of the most common mistakes I see.

If you filter post-retrieval, you might score your top 100 candidates, find none of them pass the filter, and return nothing. The relevant document was sitting at rank 101 the entire time. Filtering before search constrains the search space so the ANN only looks at documents the user is actually allowed to see. For multi-tenant systems this also matters for correctness, not just performance.

---

## GraphRAG

Vector retrieval finds documents that are *semantically close*. Graph retrieval finds documents that are *logically connected* through a chain of facts. These are genuinely different problems.

The process: extract entities and (subject, predicate, object) triples from your corpus during indexing, build a graph $G = (V, E)$, then at query time identify which entities the user cares about and traverse the relevant subgraph.

Scoring a path through the graph combines node relevance and edge confidence:

$$
\text{score}(\text{path}) = \sum_{i} \alpha_i \cdot \text{nodeRel}(v_i) + \sum_{j} \beta_j \cdot \text{edgeConf}(e_j)
$$

where $\text{edgeConf}$ reflects how confidently the relation was extracted. Paths built on weakly extracted edges shouldn't dominate the answer.

GraphRAG makes a real difference for multi-hop questions that require chaining facts. "What companies does Company X's CEO also sit on the board of?" Pure vector retrieval finds documents about that CEO but cannot follow the link to the other companies. It also helps with provenance questions where the user wants to understand how facts connect, not just what the facts are.

The honest counterpoint: it adds real indexing complexity. Entity extraction, deduplication, ontology maintenance. If you're answering "what's our refund policy?" you do not need a graph. Start without it and add it when you have clear evidence that multi-hop reasoning is failing.

---

## Multimodal retrieval

Most RAG discussions assume text input and text retrieval. The interesting problems are elsewhere.

**Cross-modal retrieval** (CLIP, SigLIP) projects text and images into a shared embedding space. A text query can retrieve images; an image query can retrieve documents. Useful for product catalogs, design assets, docs with screenshots, and anywhere a diagram communicates something that text alone doesn't.

**Document-native retrieval** (ColPali-style) is the more interesting pattern for document-heavy systems. The traditional pipeline extracts text from PDFs with OCR and loses almost everything else: table structure, equations, font emphasis, spatial layout. ColPali renders each page as an image and embeds the page directly. The LLM downstream is a VLM that sees the page as it appears, tables included. For scientific papers, financial reports, and invoices, this is a significant quality improvement over OCR-based pipelines.

**Audio-native retrieval** embeds raw waveforms or spectrogram features rather than ASR transcripts. You can query for "the segment where the customer sounded frustrated," something ASR cannot represent because it throws away prosody, speaker tone, and silence. Niche, but matters for call center analysis and media search.

---

## Corrective loops

One-shot retrieval has a ceiling. The systems that work well in production evaluate what they retrieved and decide whether to try again.

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

CRAG uses a small classifier to evaluate retrieved context quality and triggers fallback retrieval (web search, alternate index) when confidence is low. Self-RAG trains the model itself to decide when to retrieve, critique its own evidence, and revise. FLARE retrieves mid-generation when token probabilities flatten; the model essentially notices it's about to guess and goes back for more evidence.

These patterns are genuinely useful but expensive. Each corrective loop adds 1 to 3 seconds of latency and more LLM calls. The right design is to trigger them on cheap confidence signals rather than every query: low max-score from the retriever, low coverage of named entities from the original question, low cross-encoder confidence on the top result. If you can identify the hard 10% of queries and apply corrective retrieval to those, you get most of the benefit without the tail latency on the easy cases.

---

## Evaluation

Evaluate retrieval and generation separately. This sounds obvious and almost nobody does it at first.

**Recall@k:** does a relevant document appear in the top $k$? This is the floor. If you can't answer yes, the LLM never had a chance.

**Precision@k:** how many of the top $k$ are actually relevant? This is the noise you're feeding directly to the LLM.

**MRR (Mean Reciprocal Rank):** for each query, the score is $\frac{1}{\text{rank}_q}$ where $\text{rank}_q$ is the position of the first correct result. Average over all queries:

$$
\text{MRR} = \frac{1}{|Q|} \sum_{q \in Q} \frac{1}{\text{rank}_q}
$$

Rank 1 = 1.0, rank 2 = 0.5, rank 10 = 0.1. It penalizes you hard for not getting the right document at the top.

**nDCG@k** handles graded relevance, for when you have human judgments on a 0 to 4 scale rather than binary relevant/not. The logarithmic rank discounting means being wrong at rank 2 hurts more than being wrong at rank 8.

The trap is evaluating only end-to-end answer correctness. You'll see a query get a correct answer and assume retrieval is fine, but the LLM might have that fact memorized from pre-training. You'll see a wrong answer and blame generation, but the retriever might have returned garbage. Separate the metrics and you can actually diagnose what's broken.

---

## What actually fails in production

From most common to most subtle:

**Bad chunk boundaries** split a definition from its example, or a condition from what it applies to. Neither half scores well enough to be retrieved. Fix: semantic chunking, sliding overlap, or parent-child.

**Missing pre-filters** let retrieval return documents from the wrong tenant, wrong product, or wrong time window. The usual symptom is confidently wrong answers. Fix: apply filters before ANN search, not after.

**Over-aggressive compression** has a context summarizer delete the critical number, date, or conditional clause. Fix: for factual content, extract rather than summarize; keep raw spans for anything that will be cited.

**Domain-mismatched embeddings** are the one people underestimate. A general web-text model used on clinical notes or legal contracts. The embeddings look fine until you start measuring recall on the actual domain. Fix: domain-specific model or fine-tune.

**Citation mismatch** happens when you reorder the context window and citations were tied to positions rather than stable IDs. Fix: cite by document ID and span hash, not by positional index.

**Single-retriever bias.** Pure vector search misses exact-match queries; pure BM25 misses paraphrases. Fix: hybrid retrieval is the default for a reason.

**Stale index.** Documents were updated upstream but the embeddings were never refreshed. Old vector, new document, wrong retrieval. Fix: track content hashes, re-embed on change.

---

## A reasonable starting stack

If you're building from scratch, this handles most cases:

1. Hybrid retrieval (BM25 plus dense ANN in parallel)
2. RRF fusion (no calibration needed)
3. Cross-encoder re-ranker on top 100 candidates
4. Parent-child context assembly with deduplication
5. Citation verifier that checks each claim against the cited span
6. Corrective loop triggered on low confidence signals, not every query

Add GraphRAG when multi-hop reasoning fails. Add multimodal when your corpus has layout-dependent content. Add HyDE or query decomposition when query-document register mismatch is measurable. Don't add everything at once.

---

## The actual mental model

Each retrieval layer exists because the previous one has a specific, known blind spot. BM25 for exact terms. Dense vectors for paraphrases. Re-ranking because first-pass retrieval optimizes for the wrong thing. Graph traversal because proximity in embedding space is not the same as logical connection. Corrective loops because all of this still fails quietly sometimes.

The engineering question is just: which failure modes can you tolerate, and which ones will your users notice?
