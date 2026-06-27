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
