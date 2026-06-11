---
title: "Youtube Videos Recommendations"
date: 2026-06-11
summary: "This writeup containes System design interview quesitons in AIML field."
tags: [System Design, Agents, LLM, LLMOps, AI]
---

## Q1) Design a YouTube Video Recommendation System? (Homepage and optimization for engagement)

- First - Which surface are we talking about? The homepage feed and the watch next sidebar behave pretty differently, since watch next leens heavily on the video you are currently watching. And second, what are we actually optimizing for *(Push them toward long-term user satisfaction / watch time, not CTR as clickbait optimizes CTR and hurts retention)*? Because that choice drives the whole design.
- So homepage recommendations, and I am going to push back gently on "engagement". I'd optimize for long-term watch time and satisfaction rather than raw click through. If we optimize for pure CTR then we just learn to serve clickbait, which spikes short term clicks but hurts retention. So my north-star is expected watch time with guardrails on user satisfaction signals. Is this framing correct for the question?
- The Single most important constraint here is scale. We have got on the order of billions of videos and we  need to return recommendations in tens of milliseconds. That immediately rules out scoring the whole corpus with a heavy model per request. So my whole architecture is going to be a funnel: a cheap, high-recall retrieval stage that takes billions of videos down to maybe a few hundred, and then an expensive, high-precision ranking stage on top of that. Let me walk through the flow and then go deep wherever you would like.
- Starting with candidate generation, The goal here is recall and speed not precision. I dont want one source, I want several blended together: subsriptions, co-watch signals, trending, and more importand a learned retrieval model. The workhouse is a two tower model, One encode the user - their history, context, demographics and other encodes the video. I train them so the dot product of the two embeddings predicts engagement.  The best part is I can embed every video offline and build an ANN index something like ScaNN. Then at serve time I embed the user once and do an ANN lookup, which is how i can hit the latency budget. ONe thing I would be careful about here is negative sampleing. I would use in batch negatives plus some hard negatives because if i only sample easy negatives and model never learns fine distinction.
- I am using two towers and not one model because a single cross-feature model would force me to score every candidate against the user at request time, which doesn't scale to billions. The two-tower split is specifically what lets me precompute video embeddings and push the heavy lifting offline. I pay for that with a little accuracy, since the towers can't interact until the final dot product but that's exactly the tradeoff condidate generation is supposed to make. I will recover the accuracy in the ranking stage.
- Now, moving to ranking, i am only looking at a few hundred condidates, so I can afford a much richer model. Here I bring in cross features between user and video, the watch-history sequence, video age, language match and so on. I would frame this as multi task learning. I am not just predicting one number , I'm predicting watch time, likes , shares and negative signals like "not interested" jointly. A clean way to do that is a MOE setup, MMoE, so the tasks share representation without stepping on each other. Then I combine those heads into a single expected-value score for ranking.
- Two things I'd call out in ranking that interviewers usually like. One is position bias: a video that got clicked might just have been at the top, so I'd add a shallow side-tower that models position explicitly, which frees the main model to learn actual relevance. The other is that for the watch-time objective specifically, weighted logistic regression where positives are weighted by watch time is a clean trick.
- After ranking there's a re-ranking pass for the things a pure relevance model won't give me: diversity so I'm not showing ten near-identical videos, freshness, filtering out already watched content and policy and safety filters.
- For evaluation, offline i'd track recall@k for retrieval and NDCG or AUC for ranking , but those only get me a candidate worth shipping, the real decision is an online A/B test on watch time and retention, with guardrail metrics so I don't quietly hurt the user experience.
- And the last thing I'd want to flag is the hard problems. Code Start: new videos are actually handled okay by the two tower model because the video tower uses content features, but new users i'd fall back to popularity and context, and i'd add some exploration, a bandit or epsilon-greedy so fresh content and new users actually get impressions. That also helps with the feeback-loop problem whre the system just reinforces whatever it already shows. And for freshness specifically, ther ea neat trick which is feeding the age of each training example as a feature, then setting it near zero at serve time, so the model learns recency preference instead of being biased towerd older videos that have has more time to accumulate watches.

### Deep dive: the ranking phase

Ranking is the part I actually care about getting right, and the honest reason I'm even *allowed* to run an expensive model here is a numbers game. So let me put real volumes on it first, because the scale is what justifies every modeling choice that follows.

**Why I only get a few hundred candidates - the latency math.**

Retrieval hands ranking a small set, call it roughly 500 videos per request. That number isn't pulled out of the air; it falls straight out of the latency budget. Say the homepage has to come back in about 100 ms end to end. After retrieval, feature fetching, network hops, and the re-ranking pass, the ranking model itself realistically gets maybe 40-50 ms. If I batch all the candidates into one forward pass on a GPU or TPU, scoring a single candidate costs me on the order of tens of microseconds. So the constraint is basically:

```
N_candidates × cost_per_candidate ≤ ranking_budget
500 × ~50 µs ≈ 25 ms   → fits
```

Now flip it the other way. If retrieval had handed me the full catalog of a billion videos, the same model would need about `10^9 × 50 µs ≈ 50,000 seconds` per request. That's not a tuning problem, it's nine orders of magnitude off - and that one comparison is the entire reason the heavy model lives behind the funnel and never touches the whole corpus. Retrieval's only job is to make ranking's job physically possible.

**Now scale from one request to the whole fleet.**

Homepage traffic is something like a few hundred thousand requests per second at peak. Quick sanity check on that: a couple billion users, each opening the app a handful of times a day, is on the order of `10^10` requests/day, which averages to `~10^5` QPS and a few times that at peak. At ~500 candidates each, the ranking layer is doing roughly:

```
3×10^5 requests/s × 500 candidates ≈ 1.5×10^8 scorings/second
```

Each of those scorings is a few dense layers over a feature vector a few thousand dimensions wide - call it low-single-digit MFLOPs per candidate - so the aggregate lands in the **PFLOP/s** range. That's why ranking runs on a sharded accelerator fleet, and why adding "just one more layer" or "just one more feature" is a real, recurring hardware cost rather than a free accuracy win. This number is the thing I keep in my head the whole time I'm designing the model.

**Memory is a separate scaling story, and it's the embeddings.**

People assume the network weights are the big object; they aren't. The embedding tables dominate. If I have `~10^9` videos and give each a modest 64-dim embedding at 4 bytes:

```
10^9 × 64 × 4 bytes ≈ 2.5×10^11 bytes ≈ 250 GB
```

…just for video-ID embeddings, before user IDs, channels, or any other high-cardinality categorical. That obviously doesn't fit on a single host, so embedding tables get **sharded across parameter servers** and looked up by ID, while the comparatively tiny dense network is replicated everywhere. Knowing the embeddings are what blow up memory is what tells you where this system actually gets hard to operate.

**And the training data volume.**

Every page I serve logs impressions. If ~~20-30 items are actually shown per session at `~~10^5`QPS, that's on the order of`~10^6-10^7` impressions per second, which is hundreds of billions of training rows per day. I don't train on all of it raw - positives (meaningful watches) are a small fraction of impressions, so I heavily **downsample negatives** and keep the rare positives, which both balances the labels and keeps the training set to a manageable few billion examples per day, refreshed continuously.

**What the model actually does with those candidates.**

Because I'm down to a few hundred, I can finally do the thing retrieval structurally couldn't: real **cross features** between the user and each specific video. The two-tower retriever was forced to keep the user and video apart until a single dot product so it could precompute everything offline - here there's no such restriction, so I let them interact directly. Concretely I feed the watch-history sequence (run through a small transformer, or just pooled), the candidate's content embeddings, explicit user×video crosses like "what fraction of this channel has this user finished", language and geo match, device, time of day, and the video's age. Once you concatenate all those embeddings the input is genuinely wide, which is exactly why per-candidate cost is non-trivial and ties back to the FLOP math above.

**Predicting many things at once with MMoE.**

"A good recommendation" isn't one number, so I don't predict one. I jointly predict expected watch time, P(like), P(share), P(complete), and negative signals like P(not_interested) or P(report). The clean way to do that without the objectives fighting is a **Multi-gate Mixture-of-Experts (MMoE)**: a shared pool of expert sub-networks, plus a separate gating network per task that learns its own soft mix over those experts. The intuition is that watch-time and "not interested" genuinely pull the representation in opposite directions - a plain shared-bottom network suffers *negative transfer* when that happens, while fully separate per-task models cost N× to train and serve. MMoE sits in between: experts are shared where the tasks agree, gates specialize where they don't, and it's still **one forward pass**, which matters enormously given that `1.5×10^8` scorings/second figure.

**The watch-time head, specifically.**

Watch time is annoying to regress head-on, because most impressions have zero watch time (no click), so the target is heavily zero-inflated and skewed. The trick I like is **weighted logistic regression**: train a classifier where positive examples are weighted by their observed watch time and negatives by 1. The learned odds of that weighted classifier turn out to approximate the expected watch time, so at serving I just take `e^logit` to recover an `E[watch_time]` estimate. I get a clean classification loss but a continuous, watch-time-shaped output.

**Stripping out position bias.**

My labels are contaminated by *where* things were shown - an item near the top gets clicked partly because it was near the top, not purely because it was relevant. Ignore that and the model cheerfully learns "position 1 is great," which is circular. So I add a shallow **side-tower** that takes position (plus device and context), and its output is added to the main relevance logit during training. That forces the main network to explain only the *residual* - the actual relevance - and at serving I drop the side-tower (or pin position to 0) so I rank on debiased relevance.

**Collapsing the heads into one score to sort by.**

Ranking ultimately has to produce one ordering, so I fold the calibrated heads into a single expected-value score, something like:

```
score = P(watch) · E[watch_time] · (1 + w₁·P(like) + w₂·P(share)) − w₃·P(not_interested)
```

The part worth emphasizing is that those weights are **not** learned by the model - they're product knobs, tuned through online A/B tests, because they encode a business trade-off (how much do we value raw time-spent vs. an explicit "I liked this" vs. not annoying people) that no offline loss can settle for us. Keeping them explicit is what lets the team dial toward satisfaction without retraining the stack. And for that multiplication to mean anything the heads have to be **calibrated**: if `P(watch)` runs systematically high, it silently dominates the product, so I monitor calibration with reliability diagrams / ECE and correct with Platt or isotonic scaling.

### Follow-up questions (deep dives the interviewer will push on)

**Q: How exactly do you train the two-tower retrieval model, and where do labels come from?**  
It's trained as a retrieval/contrastive problem, not a pointwise classifier. A positive is a (user, video) pair where the user actually engaged, where I define "engaged" as a meaningful watch (e.g. watched past some fraction or some seconds), not just a click, so I don't bake clickbait into retrieval. The loss is sampled softmax / in-batch softmax: for each positive, the other videos in the batch act as negatives, and I maximize the dot product of the matching pair relative to those negatives. Labels are entirely implicit feedback logged from production. The one subtlety is the **sampled-softmax correction**: popular videos appear as in-batch negatives far too often, so I apply a logQ correction (subtract the log sampling probability) to avoid over-penalizing head content.

**Q: You mentioned in-batch plus hard negatives. Why not just in-batch negatives?**  
In-batch negatives are almost all trivially easy (a random video the user would never watch), so the model learns coarse topic separation but not fine distinctions, e.g. between two similar cooking videos. Hard negatives (items that are semantically close but were not engaged with, or impressed-but-skipped) force the model to learn the fine boundary that actually matters at retrieval time. The risk is going too hard: if every negative is brutally hard, training gets unstable and you can teach the model the wrong thing, so I'd mix a small fraction of hard negatives in with the easy ones.

**Q: Why MMoE for ranking instead of just training a separate model per objective?**  
Separate models are clean but expensive: you pay N times the serving cost and N times the maintenance, and you can't share signal across tasks. A shared-bottom network shares everything, which is cheap but causes negative transfer when objectives conflict (watch-time vs. "not interested" pull representations in opposite directions). MMoE is the middle ground: a set of shared expert networks plus a per-task gating network, so each task learns its own soft combination of experts. Tasks share what's useful and specialize where they conflict, and it's still a single forward pass.

**Q: How do you collapse the multiple prediction heads into one ranking score?**  
I combine them into a single expected-value score, something like a weighted product/sum of the calibrated heads, e.g. `score = P(watch) · E[watch_time] · (1 + w₁·P(like) + w₂·P(share)) − w₃·P(not_interested)`. The weights are not learned by the model; they're a product decision tuned via online A/B tests because they encode the trade-off between time-spent and satisfaction. Keeping them as explicit knobs is what lets the product team re-balance (e.g. push satisfaction harder) without retraining.

**Q: Why does the predicted-watch-time trick use weighted logistic regression?**  
I want to predict expected watch time but train with a clean classification setup. So I train logistic regression on impressions where positives (clicked/watched) are weighted by their observed watch time and negatives by 1. The learned odds of that weighted classifier approximate `E[watch_time]`, so at serving I exponentiate the logit (`e^logit`) to recover an expected-watch-time estimate. This sidesteps the messy regression on a heavily skewed, zero-inflated watch-time target.

**Q: Position bias keeps coming up. How does the side-tower actually fix it?**  
The training data is biased: an item got clicked partly *because* it was shown at position 1, not purely because it was relevant. If I ignore that, the model learns "things at the top are good." So I add a shallow tower that takes position (and device/context) as input and predicts the bias component, added to the main relevance logit during training. The main tower is then forced to explain the *residual*, meaning actual relevance. At serving time I drop the position tower (or set position to a constant/0), so I rank on debiased relevance only.

**Q: Recommendations create feedback loops where the model only learns from what it already showed. How do you break that?**  
Two levers. First, **exploration**: I don't serve purely greedy top-K; I inject exploration via epsilon-greedy or, better, a contextual bandit (Thompson sampling / UCB) so under-shown items and new users actually get impressions and generate training data. Second, **logging the propensity**: I log the score/probability with which each item was shown so I can do inverse-propensity weighting in training and offline evaluation, which corrects for the fact that the logging policy is non-uniform. Without these, the system just amplifies its own past decisions and popularity bias compounds.

**Q: Walk through cold start more concretely. New video vs new user.**  
New *video* is the easier case: the two-tower video tower is content-based (title, thumbnail, audio/visual embeddings, channel, topic), so it gets a reasonable embedding from day one without any watch history, which is a deliberate reason to keep the video tower content-driven. New *user* is harder because the user tower has almost no history, so I fall back to context (geo, language, device, time) and popularity priors, then lean on exploration to learn their taste fast. I'd also use any onboarding signals (topics they picked, first few watches) to warm up the embedding quickly.

**Q: What's the "example age" freshness trick and why does it work?**  
Models trained on logged data are biased toward older videos simply because older videos have had more time to accumulate watches. The fix is to feed the **age of the training example** (how old the video was when that impression happened) as an input feature. The model learns the real shape of how engagement decays/peaks with age. Then at serving I set that feature to zero (or near-zero), which effectively asks the model "predict engagement as if this just came out," removing the staleness bias and favoring fresh content.

**Q: Your offline metric improved but the online A/B test was flat or negative. What happened?**  
This is the classic offline/online gap, and there are a few usual suspects. (1) **Metric mismatch**: I optimized AUC/NDCG but the business cares about long-term retention, which a one-shot offline metric can't capture. (2) **Feedback-loop / distribution shift**: offline eval replays the old logging policy's distribution; the new model shifts what gets shown, so offline gains don't transfer. (3) **Position/selection bias** in the offline set inflating numbers. (4) **Novelty/primacy effects** in the experiment. The takeaway I'd state explicitly: offline metrics gate *what's worth testing*, but the A/B test on watch time + retention with guardrails is the real decision, and I'd run it long enough to see retention, not just day-1 clicks.

**Q: How do you keep the system fresh in production? Walk me through the retraining and re-indexing cadence.**  
Two separate clocks. The ranking model is retrained frequently (often daily or continuously / incrementally) because user interest and trends drift fast. The retrieval embeddings + ANN index also need rebuilding, and there's a subtle consistency trap: if I update the video tower but serve stale user embeddings (or vice-versa), the dot products are meaningless. So user and video towers must be versioned together, and the ANN index rebuilt whenever the video tower changes. For brand-new videos in the gap between index rebuilds, I rely on the non-learned candidate sources (trending, subscriptions) so fresh uploads aren't invisible.

**Q: How do you ensure diversity in the final re-ranking pass?**  
A pure relevance sort produces ten near-identical videos, which kills the session. I'd do diversity-aware re-ranking on the top candidates, using something like MMR (maximal marginal relevance) or a DPP (determinantal point process) that explicitly rewards picking items that are dissimilar from what's already in the slate. The objective becomes relevance minus a redundancy penalty over embeddings/topics. This sits *after* ranking because it's a slate-level concern: it depends on the whole result set, not a single item's score.

**Q: Predicted probabilities need to be calibrated. Why does that matter here, and how do you do it?**  
Because I'm *combining* heads (P(watch), P(like), etc.) into one score, the heads must be on a comparable, true-probability scale. If P(watch) is systematically inflated, it dominates the blend for the wrong reasons. Calibration also matters anywhere a raw probability is used downstream (e.g. expected-value blending or bidding-style logic). I'd check it with reliability diagrams / ECE and fix miscalibration with Platt scaling or isotonic regression on a held-out set, and watch it over time since calibration drifts with distribution shift.
