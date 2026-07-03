---
title: "Papers"
date: 2026-06-27
summary: "Short, easy to digest summaries of papers I've read on LLMs and agents"
tags: [LLM, Agents, AI]
---

# Agent Memory

Papers about how agents remember things across long conversations, manage limited context, and build up knowledge over time.

## Generative Agents: Interactive Simulacra of Human Behavior

Published: 2023-04-07 | [https://arxiv.org/abs/2304.03442](https://arxiv.org/abs/2304.03442)

This is the famous "Smallville" paper, a tiny town of 25 agents living in a sandbox, each one powered by an LLM. They wake up, cook breakfast, go to work, gossip, and at one point even organize a Valentine's Day party on their own. The fun part is the emergent behavior, but the actual contribution is how they made the agents *remember* and act consistently over days of simulated time, which is the memory problem all over again.

The core idea is the **memory stream**. Every agent keeps a long running list of everything it has experienced, written as plain English observations with a timestamp, things like "Sam is making coffee" or "the cafe is closed". This list just keeps growing, so the real question becomes: out of thousands of memories, which handful do you actually load into the prompt when the agent needs to decide what to do next? You obviously can't dump all of it in.

So they retrieve memories by scoring each one on three things and adding the scores up:

- **Recency** - how recently was this memory accessed? Older memories decay (they use an exponential decay), so fresh stuff ranks higher.
- **Importance** - how big a deal is this memory? They literally ask the LLM to rate each memory 1 to 10 when it's created. "I ate breakfast" scores low, "I had a fight with my roommate" scores high.
- **Relevance** - how related is this memory to what's happening right now? This is just embedding similarity between the memory and the current situation.

You add up the three scores, grab the top ones, and that's what goes into the context. Simple, but it works surprisingly well because it balances "what happened lately", "what matters", and "what's relevant" all at once.

The second clever bit is **reflection**. If agents only ever stored raw observations, they'd never form higher-level opinions, they'd know "Klaus read a book on Tuesday" and "Klaus read a book on Wednesday" but never conclude "Klaus is really into research". So periodically, once enough importance has piled up, the agent pauses and asks itself questions like "what can I conclude about the people and things I've seen?". The LLM generates these higher-level takeaways, called reflections, and writes them *back into the memory stream* as new memories. Those reflections can then be retrieved later just like anything else, and even feed into deeper reflections. So the agent slowly builds an understanding of its world instead of just a flat log of events.

Put together, the loop is: observe and store everything → retrieve the relevant bits by recency + importance + relevance → reflect to compress raw events into insights → use all of it to plan and act. A lot of today's agent memory systems (importance scoring, summarizing old memories into insights) are basically descendants of this paper.

## MemGPT: Towards LLMs as Operating Systems

Published: 2023-10-12 | [https://arxiv.org/abs/2310.08560](https://arxiv.org/abs/2310.08560)

LLMs have a context window, and once a conversation or document gets bigger than that window, the model just can't see the older stuff anymore. You can stretch the window, but it's expensive and even then it's still finite, so eventually you hit the same wall again. MemGPT's idea is: stop trying to make the window bigger, instead manage what's inside it, the same way an OS manages RAM.

A computer doesn't keep your entire hard drive loaded in RAM. It keeps the active stuff in RAM and pages everything else to disk, swapping data in and out as needed. MemGPT does the same thing for an LLM:

- **Main context** is the actual prompt that gets sent to the model, this is like RAM. It's small and limited.
- **External context** is everything else, stored outside the prompt, this is like disk. It can be huge.

The model itself decides what to move between the two. It's given function calls like `core_memory_append`, `archival_memory_search`, `conversation_search`, and it can call these on its own, mid-conversation, to save something for later or pull something back in when it's relevant. So instead of you manually managing memory, the LLM manages its own memory the way a process running on an OS would page things in and out.

This is what lets MemGPT-based agents hold conversations that go on for way longer than the context window would normally allow, or read documents bigger than the window, without forgetting earlier details, because the important bits get explicitly saved to external memory and recalled when needed instead of just falling off the edge of the window.

It's basically the same insight behind a lot of today's agent memory systems, the model shouldn't need infinite context, it just needs to know how to manage finite context well.

**How it actually works in code.** MemGPT isn't a new model, it's a loop wrapped around a normal function-calling LLM. The memory stores are just data structures, "main context" is the string you stuff into the prompt and "external context" is a database/vector store. The model is told these stores are functions it can call, and MemGPT intercepts those calls, runs them, and feeds the result back in. Here's the whole idea in pseudo code:

```python
# Main context (RAM) -> must fit in the window
system_prompt          # fixed instructions
core_memory            # small editable scratchpad (persona + facts about user)
messages               # recent conversation, capped

# External context (DISK) -> unlimited, lives outside the prompt
archival_db            # vector store of saved facts
recall_db              # full message history that scrolled out

# Functions the LLM can call to manage its own memory
TOOLS = [core_memory_append, core_memory_replace,
         archival_memory_insert, archival_memory_search,
         conversation_search]

def step(user_message):
    messages.append(user_message)

    while True:
        # if the window is filling up, page old messages out to disk first
        if token_count(messages) > LIMIT:
            old = messages[:HALF]
            recall_db.insert(old)                  # save to disk, nothing lost
            summary = llm.summarize(old)           # compress what's left behind
            messages = [summary] + messages[HALF:] # free up RAM

        prompt   = system_prompt + core_memory + messages
        response = llm.chat(prompt, functions=TOOLS)

        if response.is_function_call:
            # MemGPT runs the function itself, the LLM never touches the db
            result = execute(response.call)        # e.g. search vector store
            messages.append(result)                # feed result back in
            continue                               # loop again with new info
        else:
            return response.text                   # plain reply -> done


def execute(call):
    if call.name == "archival_memory_search":
        return archival_db.search(call.args.query, top_k=5)
    if call.name == "core_memory_append":
        core_memory[call.args.section] += call.args.content
        return "saved to core memory"
    # ...and so on for the rest
```

A few things worth calling out:

- That `continue` is the key bit. A single user turn can trigger several internal loops, the model searches archival memory, gets results back, maybe searches again, then finally writes a reply. The user only ever sees that last text message.
- The LLM never touches the database directly. It just emits a function call (normal tool use), and MemGPT parses it, executes it in code, and pastes the result back as the next message.
- Paging happens on "memory pressure". When `messages` gets too big, the oldest ones are flushed to `recall_db` and replaced with a summary. MemGPT even injects a *"warning: context is 70% full"* system message, which is the model's cue to call `core_memory_append` or `archival_memory_insert` and save anything important before it scrolls off. That's the self-managing part.

So the whole trick is: describe memory as tools, let the LLM call them, intercept and execute, feed results back, loop. No retraining, no bigger window.

---

# Reasoning & Verifiers

Papers on getting LLMs to reason reliably, and on the separate models that *check* that reasoning instead of trusting the final answer.

## Let's Verify Step by Step

Published: 2023-05-31 | [https://arxiv.org/abs/2305.20050](https://arxiv.org/abs/2305.20050)

When you ask an LLM to solve a hard math problem, it writes out a chain of reasoning and then gives an answer. The annoying thing is that the model can get the **right answer for the wrong reasons**, or make one silly slip in the middle of otherwise perfect work. If you want to actually *trust* these solutions, you need a second model, a **verifier**, whose only job is to look at a solution and score how likely it is to be correct. Then you can sample many solutions from the main model and let the verifier pick the best one. This paper is about how you should *train* that verifier.

There are two ways to give the verifier feedback, and the whole paper is a head-to-head between them:

- **Outcome supervision (ORM, Outcome-supervised Reward Model).** You only look at the final answer. If the final number is right, the whole solution is labeled "good"; if it's wrong, "bad". Cheap, because you just need the answer key, no human reads the steps.
- **Process supervision (PRM, Process-supervised Reward Model).** A human reads the solution **one step at a time** and labels each step as *correct*, *incorrect*, or *neutral*. The feedback is fine-grained: it points at exactly which line the reasoning went wrong.

```
                        Problem + model's solution

  OUTCOME (ORM)                          PROCESS (PRM)
  ─────────────                          ─────────────
  Step 1 ..........                      Step 1 ..........  ✓ correct
  Step 2 ..........                      Step 2 ..........  ✓ correct
  Step 3 .......... (has a bug)          Step 3 .......... ✗ incorrect  ← caught here
  Step 4 ..........                      Step 4 ..........  – neutral
  Final answer: 42                       Final answer: 42
        │                                      │
        ▼                                      ▼
  one label for the whole thing          one label PER step
  "answer correct → all good"            "step 3 is where it broke"
```

**Why outcome supervision is sneaky-bad:** Imagine step 3 has an error but two errors cancel out and the final answer still comes out to 42. Outcome supervision sees "answer correct" and happily labels the *entire* solution, including the broken step 3, as good. So the ORM learns from mislabeled reasoning. Process supervision doesn't have this problem, it rewards the model for reasoning that is *actually* sound, not reasoning that merely stumbles into the right answer.

**The headline result:** the process-supervised verifier wins, and not by a little. On the MATH dataset, when you sample 1,920 solutions per problem and use the verifier to pick the best one, the PRM solves **78.2%** of problems versus the ORM's lower rate. Process supervision is both **more reliable** *and* more aligned, you're rewarding the behavior you actually want (correct thinking) rather than a proxy for it (correct final answer).

### How the verifier scores a whole solution

The PRM outputs a correctness probability at *every step*. To turn those per-step scores into one number for the whole solution, they take the **product** of the step probabilities, which is the same as asking "what's the probability that *every single step* is correct?".

```python
# prm(step) -> probability in [0, 1] that this step is correct,
# given the problem and all steps before it.

def score_solution(problem, steps, prm):
    p_correct = 1.0
    for i, step in enumerate(steps):
        prefix = steps[:i]                      # everything before this step
        p_correct *= prm.step_prob(problem, prefix, step)
    return p_correct                            # score for the ENTIRE solution

def best_of_n(problem, generator, prm, n=1920):
    # sample many candidate solutions, keep the one the verifier trusts most
    candidates = [generator.sample(problem) for _ in range(n)]
    return max(candidates, key=lambda s: score_solution(problem, s.steps, prm))
```

One nice property: because a single wrong step drags the product toward zero, the PRM is naturally harsh on solutions with any broken step, exactly what you want.

Human step-by-step labeling is expensive, so they don't label random solutions, they label the ones that are most **informative**. Concretely: for each problem, generate many solutions, and preferentially send to humans the *convincing wrong-answer* solutions, ones the current model rates highly but that actually reach the wrong final answer. Those are exactly the cases where the model is confidently fooling itself, so a human label there teaches the most per dollar. This active-learning trick made the data collection several times more label-efficient.

---

# Self-Improvement & Reflection

Papers on agents that learn from their own mistakes *within* a task, retrying and getting better without any weight updates.

## Reflexion: Language Agents with Verbal Reinforcement Learning

Published: 2023-03-20 | [https://arxiv.org/abs/2303.11366](https://arxiv.org/abs/2303.11366)

Normally when you want an agent to get better at a task, you fine-tune it: run it, compute a reward, backprop, update the weights. That's slow and expensive, and you can't do it on the fly. Reflexion asks: what if the agent improves by **talking to itself** instead of by updating weights? The agent tries a task, fails, writes a short note to itself about *why* it failed, and keeps that note around so its next attempt is smarter. The "learning" happens entirely in natural language in the context window, which is why the paper calls it **verbal reinforcement learning**.

The system is three cooperating pieces:

- **Actor** : The LLM that actually does the task (picks actions, writes code, answers the question). Usually a ReAct- or CoT-style agent.
- **Evaluator** : Decides whether the attempt succeeded. This can be a hard signal (unit tests passed / task completed) or the model grading itself.
- **Self-Reflection** : The key part. Given the failed trajectory *and* the evaluator's verdict, this LLM writes a short verbal lesson: "I assumed the key was in the drawer, but I never actually opened it. Next time, open containers before searching them."

Those reflections get stored in **memory** and pasted into the prompt on the next attempt, so the agent literally reads its own past lessons before trying again.

```
        ┌──────────────────────────────────────────────┐
        │                                                │
        ▼                                                │
   ┌─────────┐   trajectory   ┌───────────┐   verdict    │
   │  ACTOR  │───────────────▶│ EVALUATOR │──────────┐   │
   │ (tries) │                │ (pass /   │          │   │
   └─────────┘                │  fail?)   │          ▼   │
        ▲                     └───────────┘   ┌──────────────┐
        │                                     │ SELF-REFLECT │
        │  reads past lessons                 │  "why did I  │
   ┌──────────┐   append reflection           │   fail?"     │
   │  MEMORY  │◀──────────────────────────────└──────────────┘
   │(lessons) │
   └──────────┘        loop until success or max attempts
```

Here's the whole loop in pseudo code:

```python
def reflexion(task, actor, evaluator, reflect, max_trials=5):
    memory = []                                  # long-term: verbal lessons
    for trial in range(max_trials):
        # actor conditions on every lesson learned so far
        trajectory = actor.run(task, lessons=memory)
        passed, feedback = evaluator.check(trajectory)
        if passed:
            return trajectory                    # solved it
        # turn this failure into a natural-language lesson and remember it
        lesson = reflect(task, trajectory, feedback)
        memory.append(lesson)                    # available on the next trial
    return trajectory                            # best effort after N tries
```

The clever bit is that `memory` carries *across attempts*. Trial 1 fails and produces a lesson; trial 2 reads that lesson and avoids the same mistake, maybe failing in a new way and adding a second lesson; and so on. It's trial-and-error, but the "error signal" is a paragraph of self-critique instead of a gradient.

It works well precisely where a plain agent gets stuck in a loop repeating the same mistake. Reflexion showed big gains across three very different kinds of tasks, decision-making (AlfWorld), reasoning (HotPotQA), and coding (it hit **91% pass@1 on HumanEval**, beating the base model by a wide margin), all without touching the model's weights.
