---
title: "AgenticAI Methods"
date: 2026-05-27
summary: "List of many methods used in agentic AI to make it use at its best."
tags: [React, CodeAct]
---

## ReAct

The thing that makes a language model an *agent* rather than a glorified autocomplete is the loop between thinking and doing. ReAct (**Rea**soning + **Act**ing) is the pattern that made this loop explicit. Instead of asking the model to reason internally and then spit out a final answer, you interleave reasoning traces with concrete actions and let real-world observations feed back into the next thought.

The loop is built from three primitives that repeat until the model decides it's done:

- **Thought**: the model reasons in natural language about what it knows and what it needs next ("I should look up the capital first").
- **Action**: it picks a tool and arguments to execute against the world (`search("capital of France")`).
- **Observation**:  the tool result comes back and gets appended to the context, grounding the next thought in something real instead of the model's priors.

Here's what a single trajectory actually looks like when serialized into the prompt:

```text
Question: What is the population of the capital of France?

Thought: I need the capital of France first.
Action: search("capital of France")
Observation: Paris is the capital of France.

Thought: Now I need the population of Paris.
Action: search("population of Paris")
Observation: Paris has about 2.1 million people.

Thought: I have enough to answer.
Action: finish("About 2.1 million people.")
```

And the orchestration that drives it is just a bounded loop around the model:

```python
def react(question, tools, model, max_steps=8):
    context = f"Question: {question}\n"
    for _ in range(max_steps):
        step = model(context)          # model emits Thought + Action
        context += step
        if "finish(" in step:
            return parse_answer(step)
        name, args = parse_action(step)
        observation = tools[name](args)  # run the chosen tool
        context += f"\nObservation: {observation}\n"
    return "Stopped: step limit reached."
```

A few things worth calling out about why this works and where it bites:

- **It fixes two failure modes at once:** Pure chain-of-thought hallucinates because it never checks itself against the world; pure tool-use fumbles because it acts reactively with no plan. ReAct forces a *why* before every action.
- **It's debuggable: **When an agent derails, you read the thoughts and see exactly where reasoning diverged from reality.
- **It's expensive:** Every step is a full round-trip through the model, so latency and token cost grow with the trajectory.
- **It needs guardrails:** A poorly-bounded loop will happily burn your budget chasing a query it can't satisfy — always cap `max_steps` and provide a graceful exit.

## CodeAct

CodeAct keeps the same Thought → Action → Observation skeleton but changes the *medium* of the action. In classic ReAct an action is a structured tool call: a function name plus JSON arguments chosen from a fixed menu. That works, but it gets awkward the moment you need composition, calling three tools, feeding one's output into another, looping over a list, or branching on a condition. You end up emitting an ungainly chain of discrete calls and the orchestration logic leaks into your framework.

CodeAct's insight is that we already have a language built for composition: code. Instead of one JSON call, the agent writes a snippet of (usually) Python, an interpreter runs it, and the stdout / return value / exception becomes the observation.

Compare the two action spaces directly. The JSON-style ReAct action handles one call at a time:

```json
{ "tool": "get_weather", "args": { "city": "Paris" } }
```

The equivalent CodeAct action composes tools, loops, and handles errors in a single turn:

```python
cities = ["Paris", "Tokyo", "Cairo"]
results = {}
for c in cities:
    try:
        w = get_weather(c)          # tool exposed as a normal function
        results[c] = w["temp_c"]
    except ToolError:
        results[c] = None

hottest = max((c for c in results if results[c]), key=results.get)
print(f"Hottest city: {hottest} at {results[hottest]}C")
```

That observation (`Hottest city: Cairo at 34C`) flows back into the next thought, exactly as in ReAct.

Why I reach for it, and the catch:

- **Expressiveness:** One action can call multiple tools, store intermediate state in variables, loop, do arithmetic, parse JSON, and recover from errors with `try/except`.
- **It plays to the model's strengths:** LLMs have seen enormous amounts of code in pretraining, so they're often better at expressing intent as a program than as brittle JSON; reported results show fewer steps and higher success on complex tasks versus JSON action spaces.
- **You're executing model-generated code:** A sandbox is non-negotiable, isolate the runtime, cap execution time and memory and whitelist what the code can touch (filesystem, network, available functions).

In short, CodeAct is ReAct where the action space is "anything you can write in a constrained Python sandbox" rather than "one of these N tools" and the sandbox is what keeps that power from becoming a liability.
