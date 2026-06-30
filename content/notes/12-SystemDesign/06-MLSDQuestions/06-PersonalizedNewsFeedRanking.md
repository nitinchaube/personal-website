---
title: "Personalized News Feed Ranking"
date: 2026-06-22
summary: "How I would design a personalized news feed and ranking system in an ML system design interview."
tags: [System Design, Recommendations, Ranking, Retrieval, ML, Feed]
---

This is my full answer to the classic ML system design question "design a personalized news feed." I have written it the way I would actually talk through it on a whiteboard: open by scoping the problem, draw the funnel, then go deep on retrieval, ranking, re-ranking, scaling, evaluation, and the hard parts. The news angle matters, so I call out everywhere it differs from a video or product feed.

---

## Q) Design a personalized news feed / ranking system

The way I want to run this is: spend the first two minutes pinning down scope and the objective, then draw the high-level funnel on the board, then go as deep as you want into any stage. Most of the interesting decisions live in retrieval and ranking, so I will spend my time there, but I will also cover scaling, evaluation, and the news-specific traps like freshness and integrity.

## Clarify and pin the objective

Before any modeling, I want three things settled, because they change the whole design.

**What surface and product are we building?** I am going to assume a personalized home feed in a news app: a single ranked stream of articles and posts pulled from many sources, the way Google News, Apple News, or a social feed works. That is different from "show me more like this article I am reading," which is a related-content problem leaning on the current article.

**What are we optimizing for?** This is the question I most want to get right, and I will push back gently on "engagement." If I optimize raw click-through, I will learn to serve clickbait, which spikes short-term clicks and quietly destroys trust and retention. For news specifically that is genuinely dangerous, because the product's value is people feeling informed, not tricked. So my north star is long-term, meaningful engagement: completed and dwelled reads, follows, returns the next day, with strong guardrails on explicit negative signals (hide, "see less," report) and on content quality. Clicks are an input, never the target.

**What is the scale and the latency budget?** I will assume tens of millions of daily users and a live corpus on the order of a few million articles, but with a crucial twist that defines this problem: the corpus churns constantly. Thousands of new articles publish every minute, and most of them are stale within a day. News has a shelf life measured in hours, not months. The feed has to come back in roughly 100 to 200 ms.

That freshness constraint is the single biggest thing that separates a news feed from a video or product recommender, and I will keep coming back to it: the index is never settled and almost every item is effectively cold start.

## The whiteboard: end-to-end architecture

Here is the system I would draw. The core idea is a funnel, because I cannot score a few million live articles with a heavy model in 150 ms. Each stage trades recall for precision as the candidate set shrinks.

```
                         ┌────────────────────────────────────────────────┐
                         │              OFFLINE / NEARLINE                 │
                         │  • train two-tower retrieval + ranking models   │
                         │  • embed every new article, refresh ANN index   │
                         │  • build feature store (user · article · ctx)   │
                         └────────────────────┬───────────────────────────┘
                                              │ models + ANN index + features
   ───────────────────────────────────────────┼──────────────────────────────── online
                                              ▼
   user opens   ┌─────────┐ user vec  ┌──────────────┐ ~1-5k  ┌──────────┐ ~300  ┌────────────┐
   the app  ──▶ │ FEATURE │ ────────▶ │  RETRIEVAL   │ cands  │ RANKING  │ cands │ RE-RANK    │──▶ feed
                │  FETCH  │           │ multi-source │ ─────▶ │  MMoE    │ ────▶ │ diversity, │
                └────┬────┘           │ + two-tower  │        │ multitask│       │ dedup,     │
                     ▲                │   ANN lookup │        └────┬─────┘       │ safety     │
                     │                └──────────────┘             │            └─────┬──────┘
                ┌────┴────┐                                        │ log               │
                │ FEATURE │◀───────────────────────────────────────┴───────────────────┘
                │  STORE  │        impressions + engagement events  ──▶  training data
                └─────────┘
```

Reading it left to right on the online path: a request comes in, I fetch features for the user and context, retrieval pulls a few thousand candidates from several sources, ranking scores them with a heavy multi-task model down to a few hundred, re-ranking turns that into a final slate, and every impression plus the engagement that follows is logged back as training data. The offline and nearline half keeps the models, the article embeddings, and the feature store fresh.

Three properties I would point out on this diagram:

- **The funnel is a latency necessity, not a preference.** I will justify the candidate counts with real numbers in the scaling section.
- **The logging loop is part of the architecture.** What I show determines what I learn from, which is the root of feedback loops and filter bubbles, so I design for it on purpose.
- **The offline/online split is where freshness lives or dies.** For news, the "nearline" path that ingests and embeds brand-new articles within minutes is doing the heaviest lifting.

Now I will walk each stage.

## Candidate generation and retrieval

The goal of this stage is recall and speed, not precision. I want to go from a few million live articles to a few thousand good candidates in single-digit milliseconds. I deliberately do not use one source, I blend several, because each covers a different failure mode.

**The sources I would blend:**

- **Followed and subscribed:** sources, topics, and authors the user explicitly follows. High precision, but narrow, and it creates a bubble if used alone.
- **Social and graph signals:** articles shared or engaged with by people the user follows. Strong for relevance and for distributing breaking stories.
- **Trending and breaking:** high-velocity stories right now, globally and per region. This is the path that lets a major event reach everyone fast, even people whose learned taste would not surface it.
- **Geo-local:** local news for the user's region, which generic relevance models systematically under-serve.
- **Learned retrieval (the workhorse):** a two-tower model with an ANN index, which is where most of the personalized recall comes from.

**The two-tower retrieval model.** One tower encodes the user (their reading history as a sequence, followed topics, geo, language, device, time of day). The other tower encodes the article (a text embedding of the title and body from a language model, plus source, topic, named entities, freshness, and whether it has media). I train the towers so that the dot product of the user vector and the article vector predicts meaningful engagement.

```
   USER TOWER                              ARTICLE TOWER
 ┌──────────────┐                        ┌────────────────────┐
 │ history seq  │                        │ text embedding     │
 │ followed     │                        │ (title + body)     │
 │ topics       │                        │ source, topic,     │
 │ geo / lang   │                        │ entities           │
 │ device, time │                        │ freshness, media   │
 └──────┬───────┘                        └─────────┬──────────┘
        ▼                                          ▼
   user vector u  ───────  score = dot(u, a)  ───── article vector a
        │                   high score = relevant            │
   computed per request                       precomputed offline,
                                              stored in the ANN index
```

The reason the two-tower split matters so much: because the user and the article never interact until that final dot product, I can embed every article offline the moment it publishes and put it into an Approximate Nearest Neighbor index like ScaNN or HNSW. At request time I embed the user once and do a single ANN lookup, which is what lets me hit the latency budget over a corpus I could never score item by item. I pay for that with a little accuracy, since the towers cannot cross-reference features, but recovering that accuracy is exactly what the ranking stage is for.

**The news-specific reason the article tower must be content-based.** In a video recommender you can lean on collaborative signal, because a popular video accumulates watch history. In news, the article that matters most is often five minutes old and has zero engagement history. So the article tower has to produce a sensible embedding from content alone: the language-model text embedding, the source, the topic, the entities. This is not a nice-to-have, it is the only way a breaking story is retrievable at all. It also means the ANN index is being appended to continuously, and I filter hard on recency at lookup time so I am not retrieving last week's news.

**Training and negative sampling.** I train this as a retrieval problem with a sampled-softmax (in-batch softmax) loss, not a pointwise classifier. A positive is a (user, article) pair where the user genuinely engaged, which I define as a meaningful read (dwelled or completed), not a bare click, so I do not bake clickbait into retrieval from the start. For each positive, the other articles in the batch act as negatives. Two subtleties I would mention without being asked:

- **In-batch plus hard negatives.** In-batch negatives are almost all trivially easy, a random article the user would never read, so the model learns coarse topic separation but not fine distinctions. I mix in a small fraction of hard negatives, articles that are semantically close or were shown-but-skipped, so the model learns the boundary that actually matters. Too many hard negatives makes training unstable, so it is a small fraction.
- **The logQ correction.** Popular articles show up as in-batch negatives far too often, so I subtract the log sampling probability (the logQ correction) to avoid over-penalizing head content.

## Ranking

Now I am down to a few thousand candidates, so I can afford a much richer model, and I deliberately do the thing retrieval structurally could not: real cross features between the user and each specific article. There is no precompute restriction here, so I let user and article interact directly.

**Features I would feed it:** the user's reading-history sequence (run through a small transformer or pooled), the candidate's content embeddings, explicit user-by-article crosses such as "how much of this source has the user read before" and topic affinity, language and geo match, device, time of day, and crucially the article's age and the source's quality and credibility signals. Once you concatenate all those embeddings the input is genuinely wide, which is exactly why per-candidate cost is non-trivial.

**Multi-task with MMoE.** A good feed item is not one number, so I do not predict one. I jointly predict expected dwell time, probability of a completed read, P(like), P(share), P(follow the source), and the negative signals P(hide), P(see-less), and P(report). The clean way to predict several objectives at once without them fighting is a Multi-gate Mixture-of-Experts: a shared pool of expert sub-networks, plus a separate gating network per task that learns its own soft mix over those experts.

```
        shared input  (user × article cross features, history, context)
                                  │
        ┌──────────┬──────────────┼──────────────┬──────────┐
        ▼          ▼              ▼              ▼          ▼
     expert1    expert2        expert3        expert4    expert5      shared expert pool
        └────┬─────┴──────┬───────┴──────┬───────┴────┬────┘
   gate:dwell│   gate:read │   gate:share │   gate:hide │              one gate per task
        ▼            ▼            ▼            ▼
   E[dwell]   P(complete)    P(share)    P(hide/report)               task heads
        └────────────┴──────────┬─┴────────────┘
                                ▼
                 score = blend(heads)   →  sort candidates
```

The intuition for why MMoE and not something simpler: dwell time and "not interested" genuinely pull the representation in opposite directions, so a plain shared-bottom network suffers negative transfer when tasks conflict, while fully separate per-task models cost N times to train and serve. MMoE sits in between: experts are shared where the tasks agree, gates specialize where they disagree, and it is still one forward pass, which matters enormously at feed scale.

**The dwell-time head specifically.** Dwell time is awkward to regress directly because most impressions have near-zero dwell (no click), so the target is heavily zero-inflated. The trick I like is weighted logistic regression: train a classifier where positive examples are weighted by their observed dwell and negatives by 1. The learned odds of that weighted classifier approximate expected dwell time, so at serving I exponentiate the logit to recover an E[dwell] estimate. Clean classification loss, continuous dwell-shaped output.

**Stripping out position bias.** My labels are contaminated by where things were shown: an article near the top gets clicked partly because it was near the top, not purely because it was relevant. If I ignore that, the model cheerfully learns "position 1 is great," which is circular. So I add a shallow side-tower that takes position, device, and context, and its output is added to the main relevance logit during training. That forces the main network to explain only the residual, the actual relevance, and at serving I drop the side-tower (or pin position to zero) so I rank on debiased relevance.

**The freshness trick (example age).** Models trained on logged data are biased toward older articles, simply because older articles have had more time to accumulate engagement. The fix is to feed the age of the training example (how old the article was at the moment of that impression) as a feature. The model then learns the real shape of how news engagement peaks and decays with age. At serving I set that feature to near zero, which effectively asks "predict engagement as if this just published," removing the staleness bias and favoring fresh content. For news this is not a minor tweak, it is central, because recency is one of the strongest real signals.

**Collapsing the heads into one score.** Ranking has to produce one ordering, so I fold the calibrated heads into a single expected-value score, something like:

```
score = P(complete) · E[dwell]
        · (1 + w1·P(like) + w2·P(share) + w3·P(follow))
        − w4·P(hide) − w5·P(report)
        · quality_multiplier(source_credibility)
```

The weights are not learned by the model. They are product knobs, tuned through online A/B tests, because they encode a business and trust trade-off that no offline loss can settle: how much we value time-spent versus an explicit "I liked this" versus, very importantly for news, not annoying people or amplifying low-quality sources. Keeping them explicit is what lets the team dial toward satisfaction and integrity without retraining the stack. For that multiplication to mean anything, the heads have to be calibrated true probabilities, so I monitor calibration with reliability diagrams and ECE and correct with Platt or isotonic scaling.

## Re-ranking and the final slate

Ranking scores each item on its own. The last pass fixes the things that only make sense when you look at the whole slate, and for news this stage carries unusually heavy weight.

- **Deduplication and event clustering.** The same story is published by fifty outlets. I cluster candidate articles by the underlying event (using their content embeddings and shared entities) and show one representative per event, picking the most credible or most complete source. Without this, a big news day fills the entire feed with the same headline.
- **Diversity.** A pure relevance sort returns ten near-identical politics articles. I apply diversity-aware re-ranking over the top candidates with something like Maximal Marginal Relevance or a Determinantal Point Process, which explicitly rewards picking items dissimilar from what is already on the slate, across topic, source, and format. For news there is also a viewpoint-diversity and filter-bubble concern, which is a real product and societal question, not just an engagement one.
- **Freshness and breaking-news boost.** A separate fast path can inject high-velocity breaking stories that the slower pipeline has not caught up to yet.
- **Integrity and safety filters.** Demote or remove misinformation, borderline content, and known low-quality or clickbait sources. This is a hard gate, and for news it is first-class, because the cost of amplifying a false story is far higher than showing a mediocre cat video.
- **Already-seen and fatigue control.** Filter articles the user already read, and cap repetition of a topic or source so the feed does not feel like it is nagging.
- **Business rules.** Source caps, ad or sponsored slotting, regional compliance.

This pass sits after ranking precisely because it is slate-level: every decision depends on the whole result set, not a single item's score.

## Scaling and the numbers

Interviewers like to see that the architecture is forced by arithmetic, not chosen by taste. So let me put volumes on it.

**Why ranking only gets a few hundred candidates.** Say the feed must return in about 150 ms end to end. After retrieval, feature fetching, network hops, and the re-ranking pass, the ranking model itself realistically gets maybe 40 to 50 ms. Batching all candidates into one forward pass on an accelerator, scoring a single candidate costs on the order of tens of microseconds. So the constraint is:

```
N_candidates × cost_per_candidate ≤ ranking_budget
300 × ~50 µs ≈ 15 ms   → fits comfortably
```

Flip it around: if retrieval handed ranking the full live corpus of a few million articles, the same model would need on the order of `10^6 × 50 µs ≈ 50 seconds` per request. That is not a tuning problem, it is orders of magnitude off, and that single comparison is the whole reason the heavy model lives behind the funnel and never touches the full corpus. Retrieval's only job is to make ranking physically possible.

**Fleet throughput.** Tens of millions of daily users opening the app several times a day is on the order of `10^8` to `10^9` requests per day, which averages to low-thousands QPS and several times that at peak. At a few hundred candidates each, the ranking layer is doing roughly:

```
~10^4 requests/s × 300 candidates ≈ 3×10^6 scorings/second
```

Each scoring is a few dense layers over a wide feature vector, so the aggregate lands in the hundreds-of-GFLOP/s to low-TFLOP/s range, which is why ranking runs on a sharded accelerator fleet and why "just one more layer" is a real recurring cost, not a free accuracy win.

**Memory is dominated by embeddings, not weights.** People assume the network weights are the big object. They are not, the embedding tables are. A few million live articles is smaller than a video catalog, but once you add user IDs, sources, topics, and entities, the high-cardinality embedding tables still dominate and get sharded across parameter servers, looked up by ID, while the comparatively tiny dense network is replicated everywhere. The news wrinkle is churn: article embeddings are constantly being added and aged out, so the table and the ANN index are moving targets.

**Training data volume.** Every feed I serve logs impressions. With tens of items shown per session at thousands of QPS, that is on the order of `10^5` to `10^6` impressions per second, billions of rows per day. Meaningful positives (completed reads) are a small fraction, so I heavily downsample negatives and keep the rare positives, which both balances the labels and keeps the training set manageable, refreshed continuously.

**The retraining and re-indexing cadence, which is the tight clock for news.** Two separate clocks, and getting them consistent is subtle.

- The ranking model is retrained frequently, often daily or continuously, because interest and the news cycle drift fast.
- The retrieval embeddings and ANN index are rebuilt and appended near-real-time, because a fresh article that is not in the index is invisible. There is a consistency trap here: if I update the article tower but serve user vectors from an older tower version, the dot products are meaningless. So the two towers must be versioned together, and the index rebuilt whenever the article tower changes. In the gap between rebuilds, brand-new articles are reached through the non-learned sources (trending, followed, breaking), so fresh uploads are never fully invisible.

## Evaluation

I think about this in two layers, and I am explicit that offline metrics only decide what is worth testing, never whether to ship.

**Offline.** For retrieval I track recall@k, "did the items the user actually engaged with show up in the candidate set." For ranking I track NDCG and AUC, plus calibration (reliability diagrams, ECE), because I am combining heads and miscalibration silently corrupts the blend. Because the logged data was produced by the old policy, I use inverse-propensity weighting and counterfactual estimates to reduce the bias from only observing what the previous system chose to show.

**Online.** The real decision is an A/B test, and for news I care about long-horizon outcomes: next-day and next-week return rate, sessions per user, and meaningful reads, not day-one clicks. I run guardrail metrics in parallel so I do not quietly win on time-spent while losing on trust: hide and report rates, source and topic diversity, complaint rates, and the share of low-quality sources surfaced. I run it long enough to actually see retention, because the failure mode I most want to catch is a model that boosts clicks today and erodes the product over weeks.

**The classic gotcha I would name.** Offline metric improves but the A/B test is flat or negative. Usual suspects: metric mismatch (I optimized AUC but the business cares about retention), feedback-loop and distribution shift (offline replays the old policy's distribution, the new model shifts what gets shown), residual position or selection bias inflating offline numbers, and novelty effects in the experiment. The takeaway I state plainly: offline gates candidates, the online test on long-term engagement with guardrails is the decision.

## The hard problems

This is where I would spend my remaining time, because it is where news gets genuinely harder than a generic recommender.

**Cold start, which in news is the default, not the exception.** Every article is born cold, so I do not rely on collaborative signal at all for retrieval, I rely on the content-based article tower, which is exactly why I built it that way. A new user is the harder case, since the user tower has almost no history, so I fall back to context (geo, language, device, time) and popularity priors, lean on any onboarding signals (topics they picked, first few reads), and use exploration to learn their taste fast.

**Feedback loops and filter bubbles.** Recommendations only learn from what they already showed, and for news the failure mode is a literal filter bubble, which is a real-world harm, not just a metrics problem. Two levers. First, exploration: I do not serve purely greedy top-K, I inject exploration with epsilon-greedy or, better, a contextual bandit (Thompson sampling or UCB) so under-shown items, new sources, and new users actually get impressions and generate training data. Second, propensity logging: I log the score or probability with which each item was shown so I can do inverse-propensity weighting in training and offline evaluation, which corrects for the non-uniform logging policy. Diversity in the re-rank pass is the third lever, and for news it doubles as a viewpoint-diversity safeguard.

**Freshness and breaking news.** Covered above, but to summarize the levers: the example-age feature so the model learns recency preference, hard recency filtering at retrieval, near-real-time index updates, a fast breaking-news injection path, and trending as a non-learned source so a major event is not gated on the model catching up.

**Integrity, misinformation, and clickbait.** This is the part that makes news ranking ethically loaded. Source credibility is a ranking feature and a re-rank gate, clickbait is suppressed by optimizing for completed reads and explicit negative feedback rather than clicks, and misinformation gets hard filters from a separate integrity classifier. I would say openly that this is never fully solved and needs human review and policy alongside the model.

**Calibration and head-blending.** Because I combine P(complete), P(like), P(hide), and the rest into one score, the heads must be on a comparable, true-probability scale, or one inflated head dominates the blend for the wrong reasons. I check it with reliability diagrams and ECE and fix drift with Platt or isotonic scaling on held-out data, and I keep watching it, because calibration drifts as the distribution shifts.

---

## Follow-up questions (the deep dives)

**Q: News changes by the minute. How do you actually keep the system fresh end to end?**  
Three things working together. The retrieval index is updated near-real-time: a streaming pipeline ingests each new article, the content-based article tower embeds it, and it is appended to the ANN index within minutes, so it is retrievable almost immediately. The ranking model uses the example-age feature so it actively prefers recent content rather than being biased toward older articles with more accumulated engagement. And trending plus breaking-news sources are non-learned paths that surface high-velocity stories instantly, covering the gap before the learned components catch up. The subtle trap is tower versioning: user and article towers must be the same version, or the dot products are noise, so I rebuild the index whenever the article tower changes.

**Q: The same story comes from fifty outlets. How do you avoid a feed full of duplicates?**  
Event-level deduplication in the re-rank pass. I cluster candidate articles by the underlying event using their content embeddings and shared named entities, then show one representative per cluster, choosing the most credible and complete source. The clustering can run nearline so the event groups are mostly precomputed, with a light online pass over the candidate slate. This is a slate-level decision, which is why it lives in re-ranking and not in the per-item ranker.

**Q: Why is the article tower content-based instead of using engagement features like a video recommender?**  
Because in news the highest-value item is usually minutes old with zero engagement history, so any embedding that depends on collaborative signal would leave breaking stories unretrievable exactly when they matter most. A content-based tower (language-model text embedding, source, topic, entities, freshness) gives a sensible vector from the moment of publication. The cost is that I lean more on the ranking stage and on explicit feedback to refine quality, but content-based retrieval is non-negotiable for a corpus where everything is cold.

**Q: How do you stop the feed from becoming a filter bubble?**  
I treat it as a design goal, not an accident. Exploration (a contextual bandit rather than greedy top-K) makes sure under-shown topics and sources get impressions. Diversity-aware re-ranking (MMR or a DPP) explicitly rewards topic, source, and viewpoint variety in the slate. And I log propensities so training and evaluation can correct for the policy's own bias instead of compounding it. I would also flag that pure engagement optimization tends toward bubbles, so the explicit product weights in the score (and guardrail metrics in the A/B test) are tuned to value diversity and trust, not just time-spent.

**Q: How do you handle clickbait and low-quality sources without hand-maintaining a blocklist?**  
The first defense is the objective: I optimize for completed and dwelled reads and weight explicit negative signals (hide, see-less, report) heavily, so a headline that gets clicks but no real reads and lots of hides scores badly on its own. On top of that, source credibility is a feature in ranking and a multiplier in the final score, learned from aggregate quality and feedback signals, and a separate integrity classifier provides a hard gate in re-ranking for misinformation and borderline content. Blocklists exist but are the last layer, not the main mechanism.

**Q: Why a two-stage funnel at all? Why not one good model?**  
Pure arithmetic. A model rich enough to be a good ranker costs on the order of tens of microseconds per candidate, and I have a few million live articles and about 150 ms. Scoring the whole corpus would take tens of seconds per request, which is orders of magnitude over budget. So I split it: a cheap two-tower retriever that precomputes article embeddings offline and does one ANN lookup to get from millions to a few thousand, then the expensive cross-feature ranker on those few thousand. The funnel is what makes a heavy model affordable at all.

**Q: How do you debias the training labels, given position and selection effects?**  
Two mechanisms. For position bias, a shallow side-tower takes position and context during training and absorbs the "shown at the top" component, so the main tower learns only residual relevance, and at serving I drop it or pin position to zero. For selection bias (I only observe outcomes for items the old policy chose to show), I log the propensity each item was shown with and apply inverse-propensity weighting in training and offline evaluation. Neither is perfect, which is why the online A/B test, not the offline number, is the real decision.

**Q: A breaking story needs to reach everyone in minutes. Does the personalized stack handle that?**  
Not on its own fast enough, and I would not force it to. The learned retrieval path needs the article embedded and indexed first, and the ranker needs some signal. So breaking news rides the non-learned trending and breaking sources, which can inject a high-velocity story into candidate generation immediately, with a freshness boost in re-ranking. Personalization still decides where it lands in each user's feed, but the reachability of the story does not wait on the learned components. This separation of "can it be surfaced" from "how is it ranked for you" is the key design choice.

**Q: Your A/B test shows more time-spent but more hides and reports. Ship it?**  
No, and this is exactly why I run guardrail metrics alongside the primary one. More time-spent with rising hides and reports is the signature of optimizing engagement at the expense of trust, which for a news product is a long-term loss even if day-one numbers look good. I would dig into which segments and sources drove the time-spent gain, suspect clickbait or outrage amplification, and either re-tune the product weights in the score to penalize negative feedback harder or fix the underlying head before shipping. The whole reason satisfaction and integrity guardrails exist is to stop this kind of win from going out.
