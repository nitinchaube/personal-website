---
title: "Tau-bench paper"
date: 2026-05-31
summary: "Summary of tau bench paper"
tags: [Evaluation, Agents, LLM, LLMOps, AI]
---

# Paper: tau-bench - A Benchmark for Tool-Agent-User Interaction in real world domains

[https://arxiv.org/pdf/2406.12045](https://arxiv.org/pdf/2406.12045)

AI is changing. We aren't just using it to write emails or answer questions anymore; we're asking it to *do* things. These "action-taking" AIs (often called agents) can now book flights, process refunds, and manage databases. But there is a big problem: how do we know if we can trust them?

Old tests were like multiple-choice quizzes. They checked if the AI *sounded* smart, but they couldn't test if the AI was actually following complex business rules or successfully completing a multi-step task. To fix this, researchers created tau-bench. It's a testing ground that puts AIs through realistic, messy customer service scenarios to see if they can actually get the job done reliably.

Before tau-bench, most tests evaluated AIs in a vacuum. You would give the AI a prompt, and it would spit out an action. But the real world doesn't work like that. Customers change their minds, give incomplete information, and ask confusing questions.

Even worse, the old way of grading these AIs relied on using *another* AI as a "judge". This AI judge would read the chat transcript and say, "Yep, the AI told the customer their refund was processed, so it passes!" The problem is, the AI might have hallucinated the success message without actually pressing the "refund" button in the system. This is called a "false success," and it happens a lot.

## Tau-bench working:

Instead of simple quizzes, tau-bench drops the AI into a simulated world made of three parts:

1. **The fake customer (user simulator):** a separate AI told to act like a normal, sometimes frustrating human. It doesn't hand over all the details up front, so the tested AI has to ask clarifying questions.
2. **The agent:** the AI being tested. It gets a strict policy document to follow and a set of tools it can call (cancel an order, search for a flight, issue a refund).
3. **The database:** a hidden backend that tracks the "real" world, like how many blue shirts are in stock or which seats are free on a flight.

When the conversation ends, tau-bench ignores what the agent *said* and looks straight at the database. It compares what actually changed against a hand-written "golden" answer key: was the refund processed, was the amount exactly right, did the item go back into inventory?

If the database matches the answer key exactly, the agent passes. One wrong fee or one wrong account and it scores zero. There is no partial credit, because in production a "mostly correct" refund is still a broken refund.

## The task, written formally

Each task is modeled as a **partially observable Markov decision process (POMDP)**. "Partially observable" is the important part: the agent can never see the true state of the world directly, it only learns about it through tool responses and what the user says.

A task is the tuple:

$$  
(S, A, O, T, R, U)  
$$

Reading it piece by piece:

- $S$ - the set of all possible world states.
- $A$ - the actions the agent can take.
- $O$ - the observations it can receive back.
- $T$ - the transition function, $T: S \times A \to S \times O$. Given a state and an action, it returns the new state and what the agent observes.
- $R$ - the reward, $R: S \to [0, 1]$. It scores the final state.
- $U$ - the space of task instructions given to the user (hidden from the agent).

The world has two halves that the agent juggles at the same time, the database and the user:

$$  
S = S_{db} \otimes S_{user}, \qquad  
A = A_{db} \cup A_{user}, \qquad  
O = O_{db} \cup O_{user}  
$$

Here $\otimes$ just means the full state is the database state *paired with* the user state, and $\cup$ means the agent's actions and observations are the union of the two channels (it can either call a tool or talk to the user).

The two halves behave very differently:

- **Database side.** $S_{db}$ is the hidden database. The agent changes it only by calling tools, and the result is **deterministic**, the same call on the same database always produces the same outcome:

$$  
T_{db}: (s_{db}, a_{db}) \mapsto (s'*{db}, o*{db})  
$$

- **User side.** $S_{user}$ is the hidden task instruction plus the conversation so far. When the agent sends a message, a language model samples the user's reply, so this side is **stochastic**, the same message can get different responses:

$$  
T_{user}: (s_{user}, a_{user}) \mapsto (s'*{user}, o*{user})  
$$

The episode ends when the user simulator outputs `###STOP###`, and only then is the agent scored.

### How a task is built:

Every task has two pieces: a hidden instruction $u_i$ that tells the user simulator who it is and what it wants, and a golden annotation that lists the exact database writes (and any facts the agent must tell the user). The instructions are written carefully so that, under the domain policy, only **one** final database state is correct. If a detail like the payment method were left open, different runs could end in different valid states and the task would be impossible to grade automatically.

The benchmark ships two domains:

| Domain    | Databases                                  | Write tools | Tasks |
| --------- | ------------------------------------------ | ----------- | ----- |
| τ-retail  | 500 users, 50 products, 1,000 orders       | 7           | 115   |
| τ-airline | 500 users, 300 flights, 2,000 reservations | 6           | 50    |

## How a run is scored:

tau-bench grades the outcome, not the path the agent took. A single run earns a reward of either 0 or 1, and it is the product of two checks:

$$  
r = r_{action} \times r_{output}  
$$

Because both factors are 0 or 1, multiplying them acts as a logical AND: the run only scores 1 if *both* the database and the spoken answer are correct.

**Database check ($r_{action}$).** Let $s_{db}^{(T)}$ be the database when the episode ends and $s_{db}^{*}$ be the golden database for the task.

$$  
r_{action} = \mathbb{1}\big[s_{db}^{(T)} = s_{db}^{*}\big]  
$$

The notation $\mathbb{1}[\cdot]$ is the indicator function: it is 1 when the condition inside is true and 0 otherwise. So $r_{action}$ is 1 only when the final database is identical to the golden one. Read-only calls the agent made along the way don't matter, only the end state does. For example, a return task might require exactly this one write and nothing else:

```text
return_delivered_order_items(
  order_id="#W2890441",
  item_ids=["2366567022"],
  payment_method_id="credit_card_1061405"
)
```

Any extra write, a wrong argument, or a missing write sets $r_{action} = 0$.

**Communication check ($r_{output}$).** Some tasks also require the agent to tell the user specific facts, like a refund amount. Let the required strings be $o_1, \ldots, o_m$.

$$  
r_{output} = \prod_{j=1}^{m} \mathbb{1}\big[o_j \text{ appears in some agent message}\big]  
$$

The product runs over every required string. Each factor is 1 if that string shows up somewhere in the agent's messages and 0 if it doesn't. Since a single 0 makes the whole product 0, the agent has to mention *all* of them, for instance both `54.04` and `41.64` for a two-item refund.

One honest caveat the authors flag: $r = 1$ is necessary but not always sufficient. An agent could, say, issue a return without asking for confirmation and still match the database. They accept this because a rule-based check is fast, objective, and already hard enough to expose real weaknesses.

When a task is run once, the benchmark reports the average reward over all $N$ tasks:

$$  
\bar{r} = \frac{1}{N} \sum_{i=1}^{N} r_i  
$$

This is simply the fraction of tasks the agent got fully right, also called pass@1.

## Measuring reliability: pass^k and pass@k

pass@1 has a blind spot. Agents are random (sampling temperature, different user phrasings), so a task that passes once might fail the next time. One green run tells you very little. What matters in production is **consistency**: can the agent solve the *same* task correctly again and again?

To measure this, tau-bench runs each task $n$ times and counts how many of those runs succeeded. Call that count $c$ (out of $n$ trials). From $n$ and $c$ it computes two metrics that answer opposite questions.

Both use the "choose" operator $\binom{n}{k}$ ("n choose k"), which counts how many ways you can pick a group of $k$ items from $n$.

### pass^k - the chance it never fails

Pick $k$ of the $n$ runs at random. What's the probability they are **all** successes? That is pass^k:

$$  
pass^k = \frac{\binom{c}{k}}{\binom{n}{k}}  
$$

The bottom counts every way to choose $k$ runs out of $n$; the top counts the ways to choose $k$ runs out of the $c$ that succeeded. The ratio is the chance a random group of $k$ runs is all wins. The benchmark averages this across all tasks.

If you assume each run succeeds independently with probability $p = c/n$, this simplifies to the intuition that reliability multiplies:

$$  
pass^k \approx p^k  
$$

That exponent is brutal. At $p = 0.6$:

| $k$ | $p^k$ |
| --- | ----- |
| 1   | 0.60  |
| 4   | 0.13  |
| 8   | 0.017 |

An agent that looks fine at 60% on one try has under a 2% chance of getting eight in a row right. This is the consistency cliff, and it is why being smart isn't the same as being reliable.

### pass@k - the chance it succeeds at least once

This is the metric coding benchmarks use, where you can try many times and keep any answer that works. Pick $k$ runs; what's the probability **at least one** succeeds?

$$  
pass@k = 1 - \frac{\binom{n - c}{k}}{\binom{n}{k}}  
$$

The fraction here is the chance that all $k$ picks come from the $n - c$ failures, in other words the chance of getting *zero* successes. One minus that is the chance of getting at least one. pass@k rewards "eventually finds a working answer"; pass^k rewards "never gets it wrong."

At $k = 1$ the two metrics meet, and both equal the average reward:

$$  
pass^1 = pass@1 = \bar{r}  
$$

### A worked example:

Say a task is run $n = 8$ times and $c = 5$ of them succeed.

Single-run success rate:

$$  
pass^1 = \frac{5}{8} = 0.625  
$$

Chance that 4 randomly chosen runs are all wins:

$$  
pass^4 = \frac{\binom{5}{4}}{\binom{8}{4}} = \frac{5}{70} \approx 0.07  
$$

Chance that at least one of 4 runs is a win (you can't pick 4 failures out of only 3):

$$  
pass@4 = 1 - \frac{\binom{3}{4}}{\binom{8}{4}} = 1 - 0 = 1.0  
$$

Same agent, same eight runs: 63% on a single try, but only a 7% shot at four clean wins in a row, while four tries are almost certain to land at least one. That spread is exactly why tau-bench reports pass^k instead of trusting a single number.

The official code mirrors this directly. It marks a run successful when the reward is essentially 1, then averages $\binom{c}{k} / \binom{n}{k}$ over the tasks:

```python
def is_successful(reward: float) -> bool:
    return (1 - 1e-6) <= reward <= (1 + 1e-6)

# for each task: c = number of successful runs out of n trials
pass_hat_k = mean(comb(c, k) / comb(n, k) for c in c_per_task.values())
```

##

The whole point of tau-bench is that it refuses to trust what the agent says. An LLM judge reading the transcript can be fooled by a confident "your refund is processed"; checking the database cannot. By writing tasks that have exactly one correct end state, the authors turn a fuzzy, subjective grading problem into a plain equality check, then layer pass^k on top to measure whether the agent can do it reliably rather than once by luck. The math is deliberately simple. The difficulty lives in the multi-turn conversation, the policy reasoning, and the noise of a simulated user.
