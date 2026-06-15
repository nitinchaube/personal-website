---
title: "Core of AI Agents"
date: 2026-05-27
summary: "A clear, structured guide to the main methods used to build capable AI agents: planning, memory, and tool use."
tags: [Agents, Planning, Memory, Tools, ReAct, CodeAct]
---

# Overview

In an AI agent, the language model (like GPT, Claude, or Gemini) acts as the **brain**. It understands what you ask and decides what to do next. But a brain alone cannot get real work done. It cannot remember things for a long time, and it cannot reach out into the world to look things up or take actions.

To fix this, an agent surrounds the model with three helper parts. The brain focuses on understanding and decision-making, while the helpers handle memory and real-world actions.

| Component    | What it does                                                                          | Common methods                                                                                               |
| ------------ | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **Planning** | Breaks a big task into smaller steps, and changes the plan when something goes wrong. | Chain of Thought (CoT), Tree of Thoughts (ToT), LLM+P, Self-Discover, ReAct, Reflexion, Quiet-STaR           |
| **Memory**   | Keeps track of what is happening now and remembers useful information from the past.  | In-context learning (short-term), vector databases / MIPS (long-term), MemGPT, token pruning                 |
| **Tool Use** | Reaches into the outside world to find missing information or perform specific tasks. | API routing (HuggingGPT), declarative compilation (DSPy), code execution, GUI navigation (OSWorld, WebArena) |

When you combine the brain with **planning**, **memory**, and **tool use**, you can give the agent a broad, high-level command. The agent then figures out the steps, recalls the information it needs, and uses the right tools to finish the job on its own.

A raw language model has real limits:

- Its knowledge is **frozen** at the end of training, so it does not know recent events.
- It has a **finite context window**, so it can only "see" a limited amount of text at once.
- It can **hallucinate**, confidently stating things that are not true.
- It struggles with **exact math** and other tasks that need deterministic, step-by-step computation.

The three components above exist to work around these limits. Planning gives the model structure, memory gives it persistence, and tools give it access to the real world and reliable computation.

---

# Planning

A hard task usually has many steps. A good agent must plan those steps ahead of time, adapt when a step fails, and pick the right way to think about each problem. Planning is how the agent closes the gap between where it is now and the goal it wants to reach.

Planning has two main parts: **task decomposition** (breaking the work into pieces) and **self-reflection** (learning from mistakes and adjusting).

## Task Decomposition

Task decomposition means breaking one big, overwhelming goal into smaller, manageable subgoals.

### 1) Chain of Thought (CoT)

Chain of Thought is the simplest and most common technique. You ask the model to "think step by step," which makes it spend more effort breaking a hard task into smaller steps. This improves accuracy and also lets you see how the model reasoned.

The weakness: CoT follows a single straight line of reasoning. If an early step is wrong, the whole chain falls apart.

### 2) Tree of Thoughts (ToT)

Tree of Thoughts improves on CoT by exploring several ideas at each step instead of just one. It creates a branching tree of possible reasoning paths.

- The agent searches this tree using Breadth-First Search (BFS) or Depth-First Search (DFS).
- Each branch is scored by a judge, often a second model prompt or a majority vote.
- Bad branches are pruned, and the agent can backtrack when it hits a dead end.

### 3) LLM+P (Using an External Planner)

LLM+P hands the hard planning over to a classic, reliable planning algorithm instead of trusting the model to stay logical over many steps. It uses the Planning Domain Definition Language (PDDL) as a bridge:

1. The model turns the natural-language problem into a "Problem PDDL."
2. A classical planner produces a mathematically correct plan from an existing "Domain PDDL."
3. The model translates that plan back into plain-language steps the agent can follow.

### 4) Self-Discover

Different tasks need different ways of thinking. A math proof, a creative story, and a debugging session each call for a different reasoning structure. Self-Discover lets the model design its own reasoning structure for a task before it starts.

It works in two stages:

1. **Compose:** The model is given a set of basic reasoning building blocks (for example, "use critical thinking," "break into subtasks," "check edge cases"). It picks the most useful ones and combines them into a clear, task-specific reasoning plan (often written as JSON).
2. **Solve:** The agent follows that self-made plan while answering.

This combines the strengths of many reasoning styles at once, costs only a few extra steps per task, and has improved results on hard benchmarks (like BigBench-Hard and MATH) by up to 32% over plain CoT.

## Self-Reflection

Planning is fragile if the agent cannot react to surprises. Self-reflection lets an agent improve over time by reviewing its past actions and fixing its mistakes. This matters most in real tasks where trial and error are unavoidable.

### 1) ReAct (Reasoning + Acting)

ReAct combines reasoning and acting in a single loop. The model alternates between thinking in plain language and taking actions in the environment (for example, calling a search API). The loop repeats:

`Thought` → `Action` → `Observation`

The `Thought` step lets the agent plan its next move, and the `Observation` feeds real-world results back in. (A full walkthrough with code is in the [Reasoning–Acting Patterns](#reasoning-acting-patterns-react-and-codeact) section below.)

### 2) Reflexion

Reflexion adds memory and self-criticism on top of ReAct, using a simple reinforcement learning setup with a basic pass/fail reward. After each action, the agent checks a heuristic to catch two failure patterns:

- **Inefficient planning:** the agent takes too long without making progress.
- **Hallucination:** the agent repeats the same action and gets the same result over and over.

When it fails, the agent stops, writes a short plain-language "reflection" about what went wrong, and saves that note in memory. On the next try, the note guides it to do better, acting like text-based lessons learned.

### 3) Chain of Hindsight (CoH) and Algorithm Distillation (AD)

Instead of relying only on reflection notes in the prompt, these methods bake self-improvement into the model's weights through training:

- **Chain of Hindsight (CoH):** The model is trained on sequences of outputs that get better and better, each labeled with human feedback. To stop it from simply copying the feedback, some past tokens are randomly hidden during training, forcing it to truly learn how to refine its own answers.
- **Algorithm Distillation (AD):** The model is trained on long sequences of reinforcement learning episodes. It learns a general "how to improve" policy that works in new situations using only its context window, without updating weights at run time.

### 4) Quiet-STaR

ReAct and Reflexion make the thinking visible as text. But humans often think silently. Quiet-STaR teaches a model to "think before speaking" by generating private reasoning at every word position, marked with special tokens (`<|start_of_thought|>` and `<|end_of_thought|>`).

- During training, it generates many possible internal thoughts in parallel.
- A special "mixing head" checks whether each internal thought actually helped predict the correct next text.
- Using a REINFORCE-style reward, the model learns to think only when it helps, creating an always-on background reasoning module that boosts performance with no task-specific tuning.

The catch is speed: generating a thought for every token is slow. **Fast Quiet-STaR** fixes this with curriculum learning that gradually reduces the number of thought tokens until the model internalizes the reasoning and runs at normal speed.

## Reasoning–Acting Patterns: ReAct and CodeAct

This section shows the two main "loop" patterns in practice, with concrete examples.

### ReAct in Detail

What turns a language model into an *agent* (rather than fancy autocomplete) is the loop between thinking and doing. ReAct makes this loop explicit. Instead of reasoning silently and giving one final answer, the agent mixes reasoning with real actions and lets real-world results shape its next thought.

The loop has three repeating parts:

- **Thought:** the model reasons about what it knows and what it needs next ("I should look up the capital first").
- **Action:** it picks a tool and arguments to run (`search("capital of France")`).
- **Observation:** the tool result comes back and is added to the context, grounding the next thought in something real.

Here is what one trajectory looks like inside the prompt:

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

The code that drives it is just a bounded loop around the model:

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

Why it works, and where it bites:

- **It fixes two problems at once.** Pure chain-of-thought hallucinates because it never checks itself against reality. Pure tool-use fumbles because it acts with no plan. ReAct forces a *why* before every action.
- **It is easy to debug.** When an agent goes off track, you can read its thoughts and see exactly where reasoning broke down.
- **It is expensive.** Every step is a full round-trip through the model, so cost and latency grow with the number of steps.
- **It needs guardrails.** A loop with no limits will happily burn your budget chasing an impossible query, so always cap `max_steps` and give it a clean way to exit.

### CodeAct

CodeAct keeps the same Thought → Action → Observation loop but changes what an action *is*. In classic ReAct, an action is one structured tool call: a function name plus JSON arguments from a fixed menu. That works, but it gets clumsy when you need to combine steps, call several tools, feed one result into another, loop over a list, or branch on a condition.

CodeAct's idea: we already have a language built for combining steps, which is code. Instead of one JSON call, the agent writes a short snippet (usually Python), an interpreter runs it, and the output (or error) becomes the observation.

A JSON-style ReAct action handles one call at a time:

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

That output (`Hottest city: Cairo at 34C`) flows back into the next thought, exactly as in ReAct.

Why it is useful, and the catch:

- **It is far more expressive.** One action can call multiple tools, store intermediate values, loop, do math, parse data, and recover from errors with `try/except`.
- **It plays to the model's strengths.** Models have seen huge amounts of code in training, so they often express intent more reliably as a program than as brittle JSON. Reported results show fewer steps and higher success on complex tasks.
- **You are running model-generated code.** A sandbox is non-negotiable: isolate the runtime, cap execution time and memory, and whitelist what the code can touch (files, network, available functions).

In short, CodeAct is ReAct where the action space is "anything you can write in a safe Python sandbox" instead of "one of these N tools," and the sandbox is what keeps that power safe.

---

# Memory

How well an agent works over long stretches of time depends almost entirely on its memory. We can map the memory in an agent onto the memory in human thinking.

## Types of Memory

Agent memory comes in three kinds, each with a different purpose and time scale:

1. **Sensory memory:** the raw, just-arrived inputs (text, images, audio) before the model fully understands them. It holds brief impressions of what just came in.
2. **Short-term memory (STM):** also called working memory. This is what the agent is actively aware of right now: the current instructions, its ongoing reasoning, and recent observations. In an LLM, this lives in the **context window**, so it is strictly limited in size.
3. **Long-term memory (LTM):** the ability to store and recall large amounts of facts, events, and procedures indefinitely. This is done by moving data out of the context window into external storage, usually a **vector database**, and pulling it back when needed.

## Managing Limited Short-Term Memory

Short-term memory fills up fast. An agent cannot keep a multi-day log in its context window without problems: attention gets diluted, the model gets "lost in the middle," and costs climb. If the context window is like a bandwidth limit, dumping raw history into it crowds out the agent's ability to reason about new input.

### MemGPT (Operating-System-Style Memory)

MemGPT treats the model's memory the way an operating system treats RAM and disk. It sets up two tiers:

- **Main Context:** small and fast, like RAM.
- **Archival Memory:** unlimited but slower, like disk.

When short-term memory nears its limit, MemGPT does not just chop off old text. Instead it triggers an interrupt: it summarizes the oldest messages, moves the raw details out to archival storage (using a first-in-first-out queue), and injects the compact summary back into the active context. Importantly, the agent learns to manage its own memory: it writes new facts to archival storage and searches to pull old facts back into "RAM" when it needs them.

### Token Compression and Pruning

A second approach is to trim redundant text before it ever takes up short-term memory:

- **LLMLingua** looks at how much attention each token receives across the model's layers. It keeps high-attention (meaning-rich) tokens and drops low-attention ones (repetitive syntax, filler). This can reach up to 20x compression with only about 1.5% loss in reasoning quality, saving both time and money.
- **AttentionRAG** scores each retrieved chunk for how relevant it is to the current task and aggressively drops low-relevance chunks before they enter the context window.

## Long-Term Memory and MIPS

Long-term memory is powered by an external vector database. It stores text chunks, documents, and logs as high-dimensional **embedding vectors**. To recall something, the agent turns a query into an embedding and searches for the closest matches.

Comparing the query against every vector exactly (with dot product or cosine similarity) is too slow at large scale. So vector stores use **Approximate Nearest Neighbour (ANN)** algorithms to perform **Maximum Inner Product Search (MIPS)**. Algorithms like Locality-Sensitive Hashing (LSH), Hierarchical Navigable Small World (HNSW) graphs, and ScaNN trade a tiny bit of accuracy for huge, sub-millisecond speedups. This lets the agent instantly recall relevant facts across millions of documents.

---

# Tool Use

A model's weights are frozen after training, so it cannot know recent facts and is weak at exact computation. **Tool use** (also called function calling) lets the agent call external APIs to get real-time data, run code, and fetch private or proprietary information.

## 1) MRKL: Routing to Expert Modules

Early tool use relied on careful prompting: the model was given a list of tool descriptions and asked to format its output to trigger them. The MRKL (Modular Reasoning, Knowledge and Language) architecture used the model as a **router** that sends questions to specialized expert modules. These modules can be neural (other models) or symbolic (a calculator, currency converter, weather API). For example, instead of guessing arithmetic, the model extracts the numbers and routes them to a real calculator.

## 2) HuggingGPT: Picking the Right Model for Each Task

HuggingGPT turns the model into a coordinator that runs a four-stage workflow:

1. **Task planning:** parse the user request into structured tasks with dependencies and arguments.
2. **Model selection:** search a repository (like Hugging Face) and pick the best expert model for each task (for example, an object-detection model for an image).
3. **Task execution:** run the chosen models.
4. **Response generation:** combine the results into a clear, plain-language answer.

## 3) Toolformer and the API-Bank Benchmark

**Toolformer** is fine-tuned in a self-supervised way to weave API calls naturally into its text. It learns to predict when a call is needed, pauses, waits for the result, and continues.

The **API-Bank** benchmark measures tool skills at three levels:

- **Level 1:** call an API correctly and extract the right parameters.
- **Level 2:** search documentation to find and learn a new API.
- **Level 3:** chain multiple API calls together to answer an ambiguous request.

## 4) DSPy: Compiling Pipelines Instead of Hand-Writing Prompts

As tool pipelines grow, hand-crafting prompts for every interaction becomes fragile and hard to scale. **DSPy (Declarative Self-improving Python)** shifts the approach from prompt engineering to **declarative compiling**.

Instead of hard-coding prompt templates, you describe *what* you want and let DSPy figure out the prompts:

| DSPy concept     | Role in the pipeline                                                                                                                                      |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Signatures**   | Plain-language type definitions that say what a step should do (for example, "take a question, return a SQL query"), replacing hand-written instructions. |
| **Modules**      | Reusable components (like `ChainOfThought` or `ReAct`) that carry out the signatures, similar to neural network layers.                                   |
| **Teleprompter** | The compiler that optimizes the whole pipeline to maximize a chosen metric.                                                                               |

The key innovation is the **Teleprompter** compiler. Given a metric (such as accuracy or successful tool execution), it runs the pipeline over training inputs, collects good traces, and then either generates the most effective few-shot prompts automatically or fine-tunes smaller open-source models (like Llama or T5) to do the job, without needing huge proprietary models. **DSPy Assertions** add strict rules on outputs: if the agent tries to pass an invalid parameter, the pipeline catches the error and feeds it back for self-correction before execution.

## 5) GUI Navigation: WebArena and OSWorld

Real autonomy means working in messy human interfaces, not just clean APIs.

- **WebArena** tests agents on realistic web tasks (managing repositories, booking flights) using the page's HTML structure (the DOM) or accessibility tree. Interestingly, agents that skip the visual interface and call web APIs directly outperform visual browsing agents by about 15%, because API responses are more predictable.
- **OSWorld** goes further, testing agents on real operating systems (Ubuntu, Windows, macOS) with real apps (LibreOffice, Chrome, VS Code). It checks the actual machine state after execution rather than trusting a model to judge success. The results are sobering: humans succeed at about 72% of these tasks, while leading agents initially scored under 12%. The biggest problem is not high-level reasoning but **visuomotor grounding**, meaning turning intent into precise clicks at the right pixel.

---

# Challenges

Despite fast progress, deploying autonomous agents in production still faces persistent problems.

### 1)Latency and Inefficiency

Agents are often slow. On the OSWorld benchmark, a task that takes a human under 30 seconds can take an agent over 12 minutes. Most of that time (75% to 94%) is spent on the back-and-forth API calls for planning and reflection. Worse, as the task goes on, the growing context makes each step up to three times slower than the first. Even top agents need 1.4 to 2.7 times more actions than a human, a sign of trial-and-error rather than efficient navigation.

### 2) Context Window Dilution

Limited context length caps how much history, documentation, and scratchpad reasoning an agent can hold at once. Even with million-token windows, filling them hurts the model's ability to find facts buried in the middle (the "lost in the middle" problem) and sharply raises cost. This is why active memory management (MemGPT, token pruning) is essential for long-running agents.

### 3) Visuomotor Grounding Failures

As OSWorld shows, agents often reason correctly but execute poorly. An agent may correctly decide to click "Submit" but output the wrong pixel coordinates and fail. Over 75% of OSWorld failures come from these spatial mistakes. Turning intent into precise clicks remains a major hurdle outside of clean API environments.

### 4) Output Instability and Reward Hacking

Because agents generate text to format tool calls and plans, small changes in temperature or wording can cause parsing errors that break the pipeline. In reinforcement learning loops, agents are also prone to **reward hacking**: exploiting flaws in the feedback to earn a reward without truly doing the task. For example, an agent might edit a unit test to always pass instead of fixing the actual bug. Strong alignment safeguards are essential before agents get unsupervised write access to important systems.

### 5) Information Asymmetry in Multi-Agent Systems

When several agents collaborate, each may only have access to its own user's private data. They cannot simply pool everything into one shared model because of privacy limits. For tasks like scheduling a meeting across a company, agents must negotiate and share only the minimum necessary information across a network, without leaking private context. Building frameworks that handle this safely is an open challenge.

# References

## Foundational Overviews

- [LLM Powered Autonomous Agents — Lil'Log](https://lilianweng.github.io/posts/2023-06-23-agent/)
- [Prompt Engineering — Lil'Log](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/)
- [Lil'Log Posts](https://lilianweng.github.io/posts/)
- [LLM Powered Autonomous Agents — Pelayo Arbués](https://www.pelayoarbues.com/literature-notes/Articles/LLM-Powered-Autonomous-Agents)
- [LLM-based Agents: Single and Multi-Agent Systems](https://tieukhoimai.me/blog/llms-based-agents)
- [How LLM Agents are Unlocking New Possibilities — WIZ.AI](https://www.wiz.ai/how-llm-agents-are-unlocking-new-possibilities/)
- [AI Agents Overview (UCSD)](https://cseweb.ucsd.edu/~yiying/cse291a-fall25/reading/ai-agents.pdf)
- [The Rise of Agentic AI: A Technical Deep Dive](https://medium.com/@brian-curry-research/the-rise-of-agentic-ai-a-technical-deep-dive-into-autonomous-ai-systems-in-2025-c2a9355252dd)
- [Awesome-AI-Agents](https://github.com/Jenqyang/Awesome-AI-Agents)
- [awesome-llm-powered-agent](https://github.com/hyp1231/awesome-llm-powered-agent)

## Planning and Reasoning

- [Self-Discover: LLMs Self-Compose Reasoning Structures (arXiv)](https://arxiv.org/html/2402.03620v1)
- [Self-Discover (Jay Pujara, PDF)](https://www.jaypujara.org/pubs/2024/zhou-neurips24/zhou-neurips24.pdf)
- [Self-Discover (OpenReview)](https://openreview.net/forum?id=BROvXhmzYK)
- [Auto-Evolve: Self-Reasoning Framework (arXiv)](https://arxiv.org/html/2410.06328v2)
- [Quiet-STaR (arXiv HTML)](https://arxiv.org/html/2403.09629v1) · [Quiet-STaR (arXiv abstract)](https://arxiv.org/abs/2403.09629) · [Quiet-STaR (Hugging Face)](https://huggingface.co/papers/2403.09629)
- [Fast Quiet-STaR: Thinking Without Thought Tokens](https://aclanthology.org/2025.findings-emnlp.1020.pdf)

## Memory

- [Working Memory in LLMs: The Context Window as Cognitive Architecture — Atlan](https://atlan.com/know/working-memory-llms/)
- [MemGPT](https://research.memgpt.ai/)
- [MemGPT architecture notebook](https://github.com/FareedKhan-dev/all-agentic-architectures/blob/main/docs/architectures/31_memgpt.ipynb)
- [RAG with Milvus and LangChain](https://github.com/milvus-io/bootcamp/blob/master/integration/langchain/rag_with_milvus_and_langchain.ipynb)
- [ColBERT (RAG from scratch notebook)](https://github.com/labdmitriy/llm-rag/blob/master/notebooks/rag-from-scratch/14-colbert.ipynb)

## Tool Use and Frameworks

- [DSPy: Compiling Declarative LM Calls (arXiv)](https://arxiv.org/abs/2310.03714) · [DSPy (OpenReview)](https://openreview.net/pdf?id=PFS4ffN9Yx)
- [DSPy Assertions: Computational Constraints for Self-Refining Pipelines](https://arxiv.org/abs/2312.13382)
- [LangGraph — LangSmith](https://smith.langchain.com/public/467d535b-1732-46ee-8d3b-f44d9cea7efa/r)
- [Beyond Browsing: API-Based Web Agents (arXiv)](https://arxiv.org/html/2410.16464v3)

## Benchmarks and Case Studies

- [OSWorld (NeurIPS Poster)](https://neurips.cc/virtual/2024/poster/97468)
- [OSWorld: Desktop AI Agents Succeed on 12% of Tasks Where Humans Succeed on 72%](https://beancount.io/bean-labs/research-logs/2026/06/15/osworld-benchmarking-multimodal-agents-real-computer-environments)
- [OSWorld-Human: Benchmarking the Efficiency of Computer-Use Agents (arXiv)](https://arxiv.org/html/2506.16042v1) · [OSWorld-Human (OpenReview)](https://openreview.net/pdf?id=sV3n6mYy7J)
- [SWE-bench: Can Language Models Resolve Real-World GitHub Issues? (OpenReview)](https://openreview.net/pdf?id=VTF8yNQM66) · [SWE-bench (GitHub)](https://github.com/swe-bench/SWE-bench) · [SWE-bench (Semantic Scholar)](https://www.semanticscholar.org/paper/SWE-bench%3A-Can-Language-Models-Resolve-Real-World-Jimenez-Yang/94a5f96308729e31c1ffbc0f0618db87795092fe) · [SWE-bench Lite](https://www.swebench.com/lite.html)
- [The AI Scientist-v2 (GitHub)](https://github.com/sakanaai/ai-scientist-v2) · [The AI Scientist-v2 (arXiv PDF)](https://arxiv.org/pdf/2504.08066) · [The AI Scientist-v2 (Sakana AI PDF)](https://pub.sakana.ai/ai-scientist-v2/paper/paper.pdf)
- [Evaluating Sakana's AI Scientist Toward 'Artificial Research Intelligence' (arXiv)](https://arxiv.org/html/2502.14297v3) · [Evaluating Sakana's AI Scientist (arXiv PDF)](https://arxiv.org/pdf/2502.14297)
- [Autonomous Agents for Collaborative Task under Information Asymmetry (NeurIPS)](https://papers.nips.cc/paper_files/paper/2024/file/0534abc9e6db91683d82186ef0d68202-Paper-Conference.pdf)
