# 30-Day Plan — Google AI/ML Engineer (Interview-First, Two-Track)

> **Your situation:** ~95 DSA problems already solved across every pattern + a strong concurrency notebook + an LLM-leaning system-design notebook. Your gap is **not coverage — it's speed/retention on DSA, and depth + reps on AI/ML + agentic system design.** This is built for a **6+ hr/day full sprint, 6 days a week, Sundays off.**

> **Target:** Clear the coding loop (medium ≤25 min, little-hard DP ≤40 min) **and** be genuinely strong on ML / GenAI / Agentic system design, with a deployed agentic project to show.

> **Calendar:** 30 training days, **Fri Jun 5 → Thu Jul 9 2026.** Sundays (Jun 7, 14, 21, 28, Jul 5) are **rest days — non-negotiable.** The tracker auto-highlights TODAY and shows a 🌿 rest banner on Sundays.

---

## Resources — one per area, do not add more

| Area | The ONE Resource |
|---|---|
| **Coding patterns + problems** | **NeetCode 150** — neetcode.io/practice |
| **General system design** | **Hello Interview** — hellointerview.com/learn/system-design |
| **ML system design** | **"Machine Learning System Design Interview"** — Ali Aminian & Alex Xu (book) |
| **LLM / GenAI + RAG** | **"Hands-On Large Language Models"** — Alammar & Grootendorst (O'Reilly) |
| **Agentic build (project)** | **Google ADK docs** — google.github.io/adk-docs |
| **Agent design patterns (reading)** | **Anthropic — "Building Effective Agents"** (one essay) |
| **Concurrency** | Your own `Concurrency/concurrency.ipynb` |

Tabs only. No "best 12 courses." If it's not in this table, it's a distraction for the next 30 days.

---

## The Daily Loop (~6 hrs, 6 days/week)

| Block | Time | What |
|---|---|---|
| **1. DSA Warm-up — cold re-solve** | 30 min | Re-solve the named old problems **cold, on a timer**. This is your spaced-repetition queue. |
| **2. DSA — new problems, timed** | 2 hr | 3–4 problems, each **cold with a 25-min cap**. Solution only *after* the timer, then re-code clean into `ConceptProblem.ipynb`. |
| **3. ML / AI System Design** | 90 min | One topic or one design from the curriculum (ML-SD → LLM/RAG → Agentic). |
| **4. Agentic AI Build + GenAI depth** | 90 min | Move the project forward + the day's GenAI reading. |
| **5. Behavioral / Googleyness** | 20 min | Even-numbered days only. STAR stories, "Why Google", Googleyness prompts. |
| **6. Log & Review Queue** | 10 min | Mark each problem ✅/⚠️/❌. Every ⚠️/❌ becomes tomorrow's warm-up. Note cold-start time. |

**The single rule that builds skill:** *always start cold on a timer.* Reading solutions feels productive and builds nothing. The struggle is the rep. You already have coverage — you need reps and speed.

---

## The 5 Blocks (6 training days each)

Within every block: Days 1–4 learn, **Day 5 = mock**, **Day 6 = consolidate + checkpoint**. Concurrency gets a re-read on each checkpoint day.

- **Block A (D1–6) · Arrays/Search + ML-SD foundations** — hashing, two-pointer, sliding window, stack, binary search, matrix gap. ML-SD framework, data/features, embeddings & vector search (HNSW/ScaNN), metrics, A/B. Build: ADK setup → RAG retriever.
- **Block B (D7–12) · Lists/Trees/Heaps/Backtracking + Recsys** — feature stores, model serving, ranking systems, eval/monitoring. Build: tool routing, memory, guardrails, tracing.
- **Block C (D13–18) · Graphs/Greedy/1D-DP + LLM/RAG** — LLM systems, RAG end-to-end, LLM eval (faithfulness, LLM-as-judge), inference optimization (KV cache, batching, vLLM). Build: eval harness, reranking/hybrid, cost control.
- **Block D (D19–24) · 2D & Hard DP + Agentic systems** — grid/subsequence/knapsack/state-machine/hard DP. Agent architectures, memory/state, agent evals & safety, GenAI infra. Build: multi-agent, long-term memory, trajectory eval, **deploy**.
- **Block E (D25–30) · Mocks + Polish** — pattern-matching DP, speed sets, **2 full mock days**, behavioral mock, design rapid-fire, portfolio write-up/demo, taper.

---

## The Agentic Build — FRONTIER level (your depth signal)

> Full detail + reading list + benchmarks in **[AGENTIC_FRONTIER.md](AGENTIC_FRONTIER.md)**.

This track is **not** a RAG bot — it tackles the four unsolved problems big labs are fighting: (1) long-horizon reliability + context engineering, (2) agent RL + evaluation, (3) multi-agent orchestration + tool ecosystems, (4) agent safety/security + inference economics. You **read the real papers** and **build one hard scoped system** that demonstrates all four.

**The build: "Aegis"** — a long-horizon, multi-agent, benchmark-evaluated, injection-hardened agent, built **eval-first** (real pass@1 by Day 2):
- **A:** scaffold + ReAct baseline + CodeAct sandbox + trajectory logging → **baseline pass@1** on a τ-bench / SWE-bench-lite slice.
- **B:** context **compaction** (MemGPT) + memory hierarchy + **orchestrator-worker** + single-vs-multi A/B + tool retrieval + **MCP**.
- **C:** **process eval** (PRM) + rubric LLM-judge + **Reflexion** + scoped **LATS** search + **RLVR** reward + skill library.
- **D:** prompt-injection **red-team (ASR)** + **CaMeL** defense + sandboxing + **inference economics** (caching/routing/parallel) + durable execution + **deploy**.
- **E:** technical report (baseline→final pass@1, cost/task, ASR before/after) + demo + defend every choice.

By Day 30 you have a deployed agent with **before/after numbers** (accuracy up, attack-success-rate down, cost/task down) — a staff-level story, not a toy.

---

## DSA gaps this plan deliberately fills (you had broad coverage but skipped these)

Grid/2D DP (Unique Paths, Min Path Sum, Maximal Square, LCS), prefix sum (Subarray Sum K via Product-Except-Self family), BFS-by-levels (Rotting Oranges, Word Ladder, Alien Dictionary), greedy (Jump Game, Gas Station), and pattern-matching DP (Regex/Wildcard). Block D is the 2D & hard-DP ladder that turns "little-hard DP" from scary into routine.

---

## How to know you're improving (track these, not problem count)

1. **Cold-start time** — minutes to the correct approach. Target <5 min for medium by Block E.
2. **Clean-solve rate** — % of new problems solved ✅ without the solution. Target 60%+ by Block E.
3. **+7-day review pass rate** — % you re-solve clean on review. Target 80%+. This number *is* your retention.
4. **Design fluency** — can you run the framework cold on recsys, RAG, an agent platform, and inference serving? Target: all four, by Day 26.

If clean-solve rate stalls, slow down — fewer problems, but solved cold. Reps beat coverage.

---

## One-paragraph version

Six days a week, ~6 hrs, Sundays off. Each day: cold-timer warm-up (retention) → 3–4 new DSA problems fully timed before any solution (the real learning) → one ML/AI system-design topic → push the agentic build forward → log so the review queue stays fed; behavioral on even days. Five blocks: A arrays/search+ML-SD foundations, B lists/trees+recsys, C graphs/DP+RAG/evals, D hard-DP+agentic systems+deploy, E mocks+polish. Seven resources, one per area. The discipline that makes you good is **starting cold on a timer** — and protecting your Sunday so the gains stick.
