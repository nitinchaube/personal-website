---
title: "LLM Alignment with RL"
date: 2026-09-06
summary: "An end-to-end, math-backed guide to aligning LLMs: the three training phases, SFT and PEFT, reward modeling, the RL loop, PPO with token-level KL control, the preference-optimization family (DPO, ORPO, KTO, RLOO, GRPO), Constitutional AI, and verifiable-reward RL with graders and environments."
tags: [Alignment, RLHF, SFT, PEFT, LoRA, Reward-Model, PPO, DPO, ORPO, KTO, GRPO, RLOO, Constitutional-AI, RLVR, KL-Divergence, Chat-Templates]
---

---

# TL;DR

Alignment is the process of turning a raw next-token predictor into a model that is **helpful, honest, and harmless** while still following instructions. It happens in three phases.

1. **Pre-training** teaches language and world knowledge by predicting the next token on trillions of tokens. The model can complete text but cannot hold a conversation or follow instructions reliably.
2. **Supervised Fine-Tuning (SFT)** shows the model many high-quality `(prompt, ideal answer)` pairs so it learns the shape of a good response: instruction following, tone, and format. SFT teaches the model **what a good answer looks like**, but it only ever sees positive examples.
3. **Preference / RL post-training** teaches the model **which of two answers is better** using human or AI preferences. This is where safety guardrails, honesty, and nuance come from.

The core RL recipe is **RLHF**: train a **reward model** on human preference pairs, then use **PPO** to push the policy toward high reward while a **KL penalty** stops it drifting away from the SFT model. Because PPO is heavy and unstable, a family of simpler methods removes pieces of the pipeline: **DPO** deletes the reward model and RL loop entirely, **ORPO** even deletes the separate SFT stage, and **GRPO / RLOO** keep online RL but delete the value network. A parallel track, **RLVR** (RL with Verifiable Rewards), replaces the learned reward model with a **deterministic grader** (unit tests, a math checker, a rubric) and is what powers modern reasoning models.

```
 Pre-training  →   SFT   →   Preference / RL post-training
  (knowledge)   (format)     (judgment, safety, honesty)
   trillions     ~10k-1M       pairs, rewards, or verifiers
   of tokens      pairs
```

---

# 1. What Alignment Means

A pre-trained LLM maximizes the probability of the next token given human text. That objective is not the same as "be useful to the person asking." Alignment closes that gap along four axes.

| Alignment axis    | Question it answers             | Example of failure                                  |
| ----------------- | ------------------------------- | --------------------------------------------------- |
| **Instructional** | Did it do what was asked?       | Ignores "answer in one word" and writes a paragraph |
| **Behavioural**   | Did it act safely and honestly? | Gives dangerous or fabricated instructions          |
| **Style**         | Does the tone and format fit?   | Cold, robotic, or wrong format                      |
| **Value**         | Does it respect human values?   | Biased, offensive, or manipulative output           |

### 1.1 Common alignment failures

These are the concrete symptoms that post-training tries to remove.

- **Hallucination:** the model states false facts with confidence. Fluency is not accuracy.
- **Sycophancy:** the model agrees with the user to please them ("the customer is always right"), even when the user is wrong. This is often *caused* by naive preference training, because raters reward agreeable answers.
- **Over-refusal:** the model refuses safe requests because refusal was over-rewarded during safety tuning.
- **Unsafe content:** the model produces harmful, toxic, or policy-violating text.

Alignment is a balancing act. Push too hard on safety and you get over-refusal. Push too hard on helpfulness and you get unsafe content or sycophancy. The whole toolkit below exists to find that balance.

### 1.2 The three phases in one picture

```
┌──────────────┐   ┌───────────────────┐   ┌──────────────────────────┐
│ PRE-TRAINING │ → │  SFT (supervised)  │ → │ PREFERENCE / RL POST-TRAIN│
├──────────────┤   ├───────────────────┤   ├──────────────────────────┤
│ objective:   │   │ objective:         │   │ objective:                │
│  next token  │   │  imitate ideal     │   │  maximize preference /    │
│              │   │  answers           │   │  reward, stay near SFT    │
│ data:        │   │ data:              │   │ data:                     │
│  raw web     │   │ (prompt, answer)   │   │ (prompt, chosen, rejected)│
│              │   │                    │   │  or (prompt, verifier)    │
│ gives:       │   │ gives:             │   │ gives:                    │
│  knowledge   │   │  format + follow   │   │  judgment, safety, honesty│
└──────────────┘   └───────────────────┘   └──────────────────────────┘
```

| Phase            | Strengths                                            | Limitations                                                                                                                                          |
| ---------------- | ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| Supervised (SFT) | Instruction following, tone, format, style           | Only sees "good" examples, so it cannot learn to *avoid* bad ones. Hallucination and safety are barely improved. Tends toward template-like answers. |
| Preference / RL  | Learns nuanced behaviour, safety guardrails, honesty | Reward model can latch onto superficial cues (length, formatting), inherits rater bias, and can be reward-hacked.                                    |

---

# 2. Supervised Fine-Tuning (SFT)

SFT is the bridge from "text completer" to "assistant." You collect a dataset of high-quality demonstrations and train the model to reproduce them with the ordinary language-modeling loss, but computed **only on the answer tokens**.

### 2.1 The objective

Given a prompt $x$ and its target response $y = (y_1, \dots, y_T)$, SFT minimizes the negative log-likelihood of the response:

$$
\mathcal{L}_{\text{SFT}} = - \sum_{t=1}^{T} \log \pi_\theta\big(y_t \mid x, y_{<t}\big)
$$

$x$ is the prompt, $y_t$ is the $t$-th token of the target answer, $y_{<t}$ is all answer tokens before it, and $\pi_\theta(y_t \mid \dots)$ is the probability the model gives the correct next token. The sum over $t$ walks through the whole answer; the minus sign turns "maximize the probability of the right tokens" into a loss to minimize. In short: **at every position, reward the model for putting high probability on the token that actually came next in the demonstration.**

The key implementation detail is **loss masking**: the prompt tokens $x$ are fed as context but their positions are masked out of the loss (set to a label of $-100$ in most frameworks). We do not want the model to learn to generate the user's question; we only want it to learn the answer.

```
tokens:   [ <user> How do I sort a list? </user> <assistant> Use sorted(). </assistant> ]
labels:   [  -100   -100 ...          -100        -100        Use sorted() . </assistant> ]
                    ^ prompt: masked (context only)        ^ response: contributes to loss
```

### 2.2 Converting pre-trained weights into a chat model

A base checkpoint has the transformer weights but no notion of "turns" or "roles." Turning it into an SFT starting point involves three practical steps.

1. **Load the base weights** (for example `Llama-3-8B`, not the `-Instruct` variant). These are the frozen product of pre-training.
2. **Add or activate special tokens** for roles and turn boundaries, such as `<|system|>`, `<|user|>`, `<|assistant|>`, and an end-of-turn token like `<|eot_id|>`. If these tokens are new, the embedding matrix is resized and the new rows are trained from scratch during SFT.
3. **Pick a precision and format.** Weights ship as `fp32`, `fp16`, or `bf16`. For fine-tuning, `bf16` is the common default because it has the dynamic range of `fp32` with half the memory. Quantized formats (`int8`, `nf4`) are used for memory-constrained PEFT (see QLoRA in Section 4).

> A base model predicts the most likely continuation of internet text. After SFT it predicts the most likely continuation of an *assistant transcript*. Same machinery, different data distribution.

### 2.3 Formatting chat templates

The model can only learn turn structure if every training example is serialized into one consistent string. That serialization is the **chat template**, usually a Jinja template shipped with the tokenizer. A list of message dicts becomes a single formatted string.

Input (structured):

```python
messages = [
    {"role": "system", "content": "You are a terse assistant."},
    {"role": "user", "content": "Capital of France?"},
    {"role": "assistant", "content": "Paris."},
]
```

Output (serialized, ChatML-style):

```
<|im_start|>system
You are a terse assistant.<|im_end|>
<|im_start|>user
Capital of France?<|im_end|>
<|im_start|>assistant
Paris.<|im_end|>
```

Rules that matter in practice:

- **Use the tokenizer's own template.** `tokenizer.apply_chat_template(messages, tokenize=False)` guarantees your training format matches what the model expects at inference. Hand-rolling the format is the single most common cause of a broken SFT.
- **Training vs generation.** At training time you include the assistant reply and end token. At inference you call `apply_chat_template(..., add_generation_prompt=True)`, which emits everything up to `<|im_start|>assistant\n` and stops, so the model completes the turn.
- **A template mismatch between training and inference silently degrades quality**, because the model is now off-distribution.

### 2.4 SFT limitations (why we need more)

SFT can only imitate. If a behaviour is not in the demonstration set, the model will not learn it, and it has no signal about what a *bad* answer looks like. It cannot learn "prefer honesty over a confident guess" from positive examples alone. That is the job of preference training.

---

# 3. Parameter-Efficient Fine-Tuning (PEFT)

Full fine-tuning of a 70B model updates all 70B parameters and needs optimizer state for every one of them, which is hundreds of gigabytes. PEFT freezes the base model and trains a tiny number of new parameters instead, cutting memory and storage by orders of magnitude with almost no loss in quality.

### 3.1 LoRA: the core idea

**LoRA (Low-Rank Adaptation)** freezes each weight matrix $W_0 \in \mathbb{R}^{d \times k}$ and learns a low-rank update beside it:

$$
W = W_0 + \Delta W = W_0 + \frac{\alpha}{r}\, B A, \qquad A \in \mathbb{R}^{r \times k},\; B \in \mathbb{R}^{d \times r},\; r \ll \min(d,k)
$$

$W_0$ is the original frozen weight matrix; $\Delta W = BA$ is the small learned change placed next to it. $A$ squeezes the input down to $r$ numbers and $B$ expands it back, so their product $BA$ has the same shape as $W_0$ but is built from far fewer parameters. $r$ is the **rank** (the width of that bottleneck, often 8 to 64) and $\alpha$ is a scaling factor, so the effective strength of the update is $\alpha / r$. Because $r$ is tiny, the trainable-parameter count drops from $d \times k$ down to $r(d + k)$. Only $A$ and $B$ ever receive gradients; $W_0$ never moves.

```
        frozen                    trainable (tiny)
   x ──►  W0  ──►(+)──► h          x ──► A ──► B ──►(scaled by α/r)──► added to h
                  ▲                       r×k   d×r
                  └──────────────────────────────┘
```

At inference you can **merge** $BA$ back into $W_0$ so there is zero added latency, or keep adapters separate and hot-swap them per task.

### 3.2 QLoRA

**QLoRA** goes further: it quantizes the frozen base model to 4-bit (`nf4`, a normal-float format) and trains LoRA adapters in `bf16` on top. Gradients flow through the frozen 4-bit weights but never update them. This lets a 65B model fine-tune on a single 48 GB GPU. Two supporting tricks are **double quantization** (quantizing the quantization constants) and **paged optimizers** (spilling optimizer state to CPU on memory spikes).

| Method  | What is trained               | Base precision | Relative memory |
| ------- | ----------------------------- | -------------- | --------------- |
| Full FT | All weights + optimizer state | bf16           | Highest (1x)    |
| LoRA    | Small adapters                | bf16           | Low (~0.3x)     |
| QLoRA   | Small adapters                | 4-bit (nf4)    | Lowest (~0.1x)  |

Use full FT when you have the compute and want maximum quality; LoRA as the default for most task adaptation; QLoRA when GPU memory is the binding constraint.

---

# 4. SFTTrainer Implementation

The Hugging Face TRL library wraps everything above in `SFTTrainer`. A minimal, correct setup:

```python
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig

model_id = "meta-llama/Llama-3.1-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="bfloat16")

# Demonstration data: each row has a "messages" list.
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

sft_config = SFTConfig(
    output_dir="sft-llama3",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,   # effective batch = 32
    learning_rate=2e-4,              # higher LR is fine for LoRA
    num_train_epochs=1,
    bf16=True,
    packing=True,                    # pack short samples to fill the context
    max_seq_length=2048,
)

trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset,
    peft_config=peft_config,         # omit for full fine-tuning
    processing_class=tokenizer,
)
trainer.train()
```

What the trainer handles for you:

- **Chat templating:** it calls `apply_chat_template` on the `messages` field automatically.
- **Loss masking:** with `assistant_only_loss=True` (or a completion-only collator) it masks prompt tokens so loss is computed only on assistant turns.
- **Packing:** `packing=True` concatenates multiple short examples into one `max_seq_length` block, avoiding wasted compute on padding. Attention is masked so packed examples do not attend across boundaries.

---

# 5. Reward Model Training (RM)

Once SFT gives a competent baseline, we need a signal for *which* of several answers people prefer. Humans cannot label an absolute score reliably, but they *can* compare two answers. A **reward model** turns those comparisons into a scalar score $r_\phi(x, y)$ that the RL stage will maximize.

### 5.1 Pairwise preference data layouts

The atomic unit is a triple: one prompt, a preferred (chosen) response, and a dispreferred (rejected) response.

```
{
  "prompt":   "Explain gravity to a five year old.",
  "chosen":   "Gravity is the invisible pull that keeps you on the ground...",
  "rejected": "Gravity is a fundamental interaction described by general relativity..."
}
```

Variations you will meet:

- **Binary pairs** (chosen vs rejected): the standard layout above.
- **K-way rankings:** raters order $K$ responses; this is expanded into $\binom{K}{2}$ pairs for training.
- **Scored / Likert data:** each response gets a rating (for example 1 to 7); pairs are derived by comparing scores.
- **AI feedback (RLAIF):** the "human" labeler is replaced by a strong LLM judge that outputs the preference, which is far cheaper and the basis of Constitutional AI (Section 11).

### 5.2 Training a classifier backbone

The reward model is usually the **SFT model with its language-modeling head replaced by a scalar head**: a single linear layer that maps the final hidden state of the last token to one number.

```
prompt + response ──► transformer (init from SFT) ──► hidden state of last token ──► Linear(d → 1) ──► r_φ(x,y)
```

Starting from the SFT checkpoint matters, because the reward model needs the same understanding of language and instructions that the policy has. Only the scalar head is new; the backbone is fine-tuned.

### 5.3 The margin loss objective

The reward model is trained so the chosen response scores higher than the rejected one. Using the **Bradley-Terry** model of pairwise preference, the probability that $y_w$ (chosen/winner) beats $y_l$ (rejected/loser) is the logistic function of the score gap:

$$
P(y_w \succ y_l \mid x) = \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}
$$

Minimizing the negative log-likelihood gives the reward-model loss:

$$
\mathcal{L}_{\text{RM}} = - \mathbb{E}_{(x, y_w, y_l)} \Big[ \log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big) \Big]
$$

| Symbol                            | Plain meaning                                                                                |
| --------------------------------- | -------------------------------------------------------------------------------------------- |
| $x$                               | the prompt                                                                                   |
| $y_w$, $y_l$                      | the winning (chosen) and losing (rejected) response                                          |
| $r_\phi(x, y)$                    | the scalar score the reward model gives response $y$ for prompt $x$                          |
| $r_\phi(x, y_w) - r_\phi(x, y_l)$ | the **margin**: how much higher the winner scores than the loser                             |
| $\sigma(\cdot)$                   | the sigmoid, squashing the margin into a probability between 0 and 1                         |
| $\mathbb{E}_{(x, y_w, y_l)}$      | average over all preference triples in the dataset                                           |
| the minus sign                    | we minimize loss, which means maximizing the log-probability that the winner beats the loser |

For every labeled pair, compute how much more the model likes the winner than the loser, pass that gap through a sigmoid to turn it into "probability the winner is better," and train so that probability is close to 1. The loss only cares about the **difference** in scores, not their absolute values, so it pushes the winner's score above the loser's until the margin is large enough that $\sigma$ saturates. This is why reward-model scores have an arbitrary offset and only relative comparisons are meaningful.

A common variant adds an explicit **margin** $m$ (larger when the human said "much better" than "slightly better"):

$$
\mathcal{L}_{\text{RM}}^{m} = - \mathbb{E}\Big[ \log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l) - m\big) \Big]
$$

The failure mode to watch: reward models learn **superficial correlates** of quality such as length, markdown formatting, or a confident tone, because those correlate with human preference in the data. This is the root of reward hacking downstream.

---

# 6. RL Foundations for LLMs

Reinforcement learning is **learning by trial and error through rewards**. Before diving into specific algorithms, map the standard RL vocabulary onto language models.

| RL concept              | In an LLM                                                                             |
| ----------------------- | ------------------------------------------------------------------------------------- |
| **Agent**               | The LLM (the policy)                                                                  |
| **Environment**         | The task plus whatever returns a reward (a reward model, a grader, or a tool sandbox) |
| **State**               | The prompt plus the tokens generated so far                                           |
| **Action**              | The next token to emit                                                                |
| **Reward**              | The score for the completed response (often only at the end)                          |
| **Policy** $\pi_\theta$ | The probability distribution over the next token                                      |

### 6.1 The RL loop

```
   ┌────────────────────────────────────────────────┐
   │ 1. OBSERVE   model sees the prompt (state)      │
   │ 2. ACT       generate a response (actions)      │
   │ 3. REWARD    RM / evaluator scores the response │
   │ 4. LEARN     update the policy toward reward    │
   │ 5. REPEAT    thousands of times                 │
   └───────────────────────┬────────────────────────┘
                           │ loop
                           ▼
                 better-aligned policy
```

Step 4, "update the policy toward reward," is where all the real machinery lives, and it works through the **policy gradient**. The idea is simple to state: after the model generates a response and receives a reward, nudge its parameters so that the tokens which led to a high reward become more probable next time and the tokens which led to a low reward become less probable, with each nudge sized by how large the reward was. Good responses get their token choices reinforced, bad responses get theirs suppressed, and over thousands of rounds the policy shifts toward whatever earns reward. This bare version, "increase log-probability in proportion to reward," is the classic algorithm called **REINFORCE**, and every algorithm later in this note (PPO, RLOO, GRPO) is a more careful, lower-variance version of exactly this one move.

Two pieces of notation follow from this. The thing the model is really trying to maximize is the **return**, meaning the total reward collected from a given token onward to the end of the sequence; for LLMs the return is usually just the single response-level score handed out at the last token, plus the per-token penalties described later. And because any one sampled response is noisy (the same prompt can produce a great answer or a poor one by chance), the training objective is written as an **expectation**, the symbol $\mathbb{E}$, which just means the probability-weighted average over many sampled responses rather than the result of any single one. When you see $\mathbb{E}$ in the losses below, read it as "averaged over lots of samples."

### 6.2 Exploration in LLMs

RL only improves if the agent tries new things and discovers which earn reward. For an LLM, **exploration means sampling different token sequences** rather than always taking the greedy path. The sampling controls are the exploration knobs:

- **Temperature:** higher temperature flattens the distribution, so the model takes more varied paths.
- **Top-p (nucleus) and top-k:** sample from the smallest set of tokens covering probability mass $p$, or from the top $k$ tokens.

Too little exploration and the model never finds better behaviours; too much and the rollouts are noisy garbage. Group-based methods (Section 9) lean on this by sampling several completions per prompt and comparing them.

This raises one more distinction that shapes the entire method zoo later on: whether an algorithm learns from responses the current model generates for itself, or from a fixed pile of responses collected earlier. A method is **on-policy** when it must keep sampling fresh completions from the very model it is training, so its data always reflects the model's current behaviour (PPO, RLOO, and GRPO all work this way). A method is **off-policy**, or **offline**, when it learns from a static dataset of responses that were produced by some other model or an earlier version and never regenerated during training (DPO is the classic example). On-policy learning genuinely explores and adapts, which is powerful but expensive because generation runs in the training loop; offline learning is cheap and stable but can only ever squeeze signal out of the data it was handed, and it drifts as the model moves away from whatever produced that data.

---

# 7. Proximal Policy Optimization (PPO)

PPO is the classic RL algorithm that turns the reward model into policy updates. The policy is the LLM itself; each token is an action, the prompt-plus-generated-text is the state.

### 7.1 The four models in the loop

Full RLHF with PPO juggles four networks. Understanding memory (Section 8) starts here.

| Model                                   | Role                                                      | Trained?    |
| --------------------------------------- | --------------------------------------------------------- | ----------- |
| **Policy / Actor** $\pi_\theta$         | The LLM being aligned; generates responses                | Yes         |
| **Reference (base)** $\pi_{\text{ref}}$ | Frozen copy of the SFT model; the KL anchor               | No (frozen) |
| **Reward model** $r_\phi$               | Scores completed responses                                | No (frozen) |
| **Value / critic** $V_\psi$             | Estimates expected future reward per token, for advantage | Yes         |

### 7.2 The PPO loop and rollout

From the **rollout phase** we obtain `(state, action, reward)`, that is `(prompt, completion, reward)` tuples, together with the **log-probabilities** of each token and the **value** estimates (expected future reward per token).

```
        ┌─────────────────────────────────────────────────────────┐
        │  1. Sample prompts x from the dataset                    │
        │  2. Policy π_θ generates responses y (rollout)           │
        │  3. Reward model scores: r = r_φ(x, y)                   │
        │  4. Per-token KL penalty vs reference model              │
        │  5. Critic V_ψ estimates value; compute advantage (GAE)  │
        │  6. PPO clipped update on π_θ and V_ψ                    │
        └───────────────────────────┬─────────────────────────────┘
                                    │ repeat
                                    ▼
                         better-aligned policy
```

Pseudocode:

```
initialize policy parameters θ (from the SFT model)
for k = 0, 1, 2, ...:
    D_k  = collect completions by sampling the policy (rollout)
    r_k  = score each completion with the reward model r_φ
    Â    = compute advantages using the critic V_ψ (GAE) and the KL term
    θ    = update policy by maximizing the clipped surrogate objective
    ψ    = update the critic to better predict returns
```

### 7.3 The advantage: how good was this token?

Before the objective, we need the **advantage** $\hat{A}_t$, which answers "was token $a_t$ better or worse than what we normally expect from this state?" It is the reward actually received minus the critic's prediction:

$$
\hat{A}_t = \underbrace{R_t}_{\text{actual return from here}} - \underbrace{V_\psi(s_t)}_{\text{critic's predicted value}}
$$

- If $\hat{A}_t > 0$, the token did **better** than expected, so make it more likely.
- If $\hat{A}_t < 0$, the token did **worse** than expected, so make it less likely.
- Subtracting the critic's baseline $V_\psi$ is what keeps the gradient low-variance; without it, every token in a good response would be reinforced equally, even the mediocre ones.

In practice PPO uses **GAE (Generalized Advantage Estimation)**, a smoothed average of these differences over future tokens that trades a little bias for much lower variance. The takeaway: **advantage = "surprisingly good or bad," and that surprise is what the update chases.**

### 7.4 The clipped surrogate objective

Now the update rule. Let $\rho_t(\theta) = \dfrac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$ be the **probability ratio**: how much more (or less) likely the new policy makes this token compared to the policy that generated the rollout. PPO maximizes:

$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t \Big[ \min\big( \rho_t \hat{A}_t,\; \text{clip}(\rho_t, 1 - \epsilon, 1 + \epsilon)\, \hat{A}_t \big) \Big]
$$

| Piece                                         | Plain meaning                                                                                                                           |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| $\rho_t$                                      | ratio of new-policy to old-policy probability for the token; $\rho_t = 1$ means "unchanged," $\rho_t = 1.5$ means "50% more likely now" |
| $\hat{A}_t$                                   | the advantage from Section 7.3: positive if the token was good, negative if bad                                                         |
| $\rho_t \hat{A}_t$                            | the raw REINFORCE-style push: increase probability of good tokens, decrease bad ones                                                    |
| $\text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)$ | the same ratio but forced to stay inside $[0.8, 1.2]$ when $\epsilon = 0.2$                                                             |
| $\min(\cdot, \cdot)$                          | take the **more pessimistic** of the two, so the model never gets rewarded for moving the probability too far in one step               |

Clipping is the "proximal" (meaning "stay nearby") part. If an update would change a token's probability by more than $\pm 20\%$, the objective flattens out and the gradient for that term vanishes, so a single batch cannot swing the policy wildly. The $\min$ makes this one-sided in the safe direction: gains from over-shooting are capped, but the penalty for a bad token is never softened. This is the trick that made policy-gradient RL stable enough to use on billion-parameter models.

### 7.5 The token-level KL-divergence penalty

Maximizing reward alone is dangerous: the policy will drift into strange, off-distribution text that happens to fool the reward model (reward hacking), or forget its language ability. The fix is a **per-token KL penalty** that keeps the policy close to the frozen reference. KL divergence is the standard way to measure how far one probability distribution sits from another, so here it measures, token by token, how far the updated policy's next-token distribution has moved away from the reference model's next-token distribution; the penalty grows as that gap grows. The effective reward at each token becomes:

$$
R_t = \underbrace{r_\phi(x, y)\cdot \mathbb{1}[t = T]}_{\text{RM score at final token}} \;-\; \beta \, \underbrace{\log \frac{\pi_\theta(y_t \mid x, y_{<t})}{\pi_{\text{ref}}(y_t \mid x, y_{<t})}}_{\text{KL per token}}
$$

| Symbol                                     | Plain meaning                                                                                            |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------- |
| $r_\phi(x, y)$                             | the reward model's score for the whole finished response                                                 |
| $\mathbb{1}[t = T]$                        | an indicator that is 1 only at the **last** token $T$, so the RM score is added just once, at the end    |
| $\pi_\theta(y_t \mid \dots)$               | probability the current policy assigns to the token it emitted                                           |
| $\pi_{\text{ref}}(y_t \mid \dots)$         | probability the frozen SFT model would have assigned to that same token                                  |
| $\log \frac{\pi_\theta}{\pi_{\text{ref}}}$ | the per-token KL term: positive when the policy is more confident than the reference, so it costs reward |
| $\beta$                                    | the strength of the leash tying the policy to the reference                                              |

The reward model gives one score at the very end of the sequence. At every token along the way, we subtract $\beta$ times how far the policy has strayed from the reference on that token. If the policy makes a token much more likely than the reference did, it pays a penalty right there. The coefficient $\beta$ is the knob:

- $\beta$ too small → the policy chases reward, hacks the RM, and produces degenerate text.
- $\beta$ too large → the policy barely moves from SFT and learns nothing new.

Many implementations make $\beta$ **adaptive**, raising it when the measured KL exceeds a target and lowering it when the model is too conservative. This single term is the heart of "align but do not break the model."

---

# 8. Memory Footprint Engineering

PPO is expensive because those four models (Section 7.1) can all sit in GPU memory at once, and two of them (policy, critic) also carry gradients and optimizer state. Managing this is a large part of practical RLHF engineering.

### 8.1 Where the memory goes

For a model with $P$ parameters trained in mixed precision with Adam, the rough per-model budget is:

$$
\text{Memory} \approx \underbrace{2P}_{\text{bf16 weights}} + \underbrace{2P}_{\text{gradients}} + \underbrace{12P}_{\text{Adam: fp32 copy + 2 moments}} + \text{activations}
$$

That is roughly **16 bytes per trainable parameter** before activations. The frozen reference and reward models only need their 2 bytes of weights each (no gradients, no optimizer state).

### 8.2 The main levers

- **PEFT for the policy (LoRA):** only adapters carry gradients and optimizer state, so the expensive $16P$ term shrinks to the adapter size. This is the biggest single win.
- **Shared backbone:** attach the value head on top of the same backbone as the policy or reward model, so you do not pay for a fifth full transformer.
- **Reference-free methods:** DPO and friends (Section 9) delete the reward and critic models entirely. GRPO deletes just the critic.
- **Quantization:** load frozen reference and reward models in 8-bit or 4-bit.
- **Gradient checkpointing:** recompute activations in the backward pass instead of storing them, trading compute for memory.
- **Offloading (ZeRO / FSDP):** shard or offload optimizer state and gradients across GPUs or to CPU.

| Recipe     | Models in memory   | Trainable              | Relative cost |
| ---------- | ------------------ | ---------------------- | ------------- |
| Full PPO   | 4                  | policy + critic        | Highest       |
| PPO + LoRA | 4 (2 frozen fully) | adapters + critic head | Medium        |
| GRPO       | 3 (no critic)      | policy                 | Lower         |
| DPO / ORPO | 1-2 (policy, ref)  | policy                 | Lowest        |

---

# 9. The Preference-Optimization Family

PPO works but is heavy and finicky. A family of methods trades some of its power for simplicity by removing pieces of the pipeline. Start with DPO, the one that changed the field.

## 9.1 Direct Preference Optimization (DPO)

DPO is the insight that you do **not need a separate reward model or an RL loop at all**. It shows that the RLHF objective (maximize reward under a KL constraint) has a closed-form optimal policy, and you can rearrange that solution to train directly on the preference pairs with a simple classification loss.

The KL-constrained reward-maximizing policy satisfies:

$$
\pi^*(y \mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x)\, \exp\!\Big(\tfrac{1}{\beta} r(x, y)\Big)
$$

Solving this for the reward gives $r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$. In other words, **the policy itself is an implicit reward model**. Substitute this back into the Bradley-Terry preference loss and the intractable partition function $Z(x)$ cancels, because it depends only on $x$ and appears in both the chosen and rejected terms.

$$
\mathcal{L}_{\text{DPO}} = - \mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\!\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} \;-\; \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]
$$

| Symbol                                                             | Plain meaning                                                                                                                                     |
| ------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| $\pi_\theta(y \mid x)$                                             | probability the model being trained assigns to response $y$                                                                                       |
| $\pi_{\text{ref}}(y \mid x)$                                       | probability the frozen SFT model assigns to the same response                                                                                     |
| $\log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}$ | the **chosen log-ratio**: how much more likely the policy now makes the good answer versus the reference. This is DPO's implicit reward for $y_w$ |
| $\log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}$ | the **rejected log-ratio**: the same implicit reward for the bad answer                                                                           |
| $\beta$                                                            | how hard the policy may pull away from the reference (same role as the KL coefficient in PPO)                                                     |
| $\sigma(\cdot)$, minus sign                                        | as in the reward model: turn the reward gap into a probability and train it toward 1                                                              |

DPO treats "how much more the policy likes an answer than the reference does" as a built-in reward, so no separate reward model is needed. The loss widens the gap between the chosen and rejected implicit rewards: **push the chosen log-ratio up and the rejected one down.**

On $\beta$: suppose the policy raises the chosen answer's probability and the sigmoid input is large and positive. The loss is already near zero, so gradients shrink and training stops pushing that pair. Small $\beta$ scales that input down, letting the policy wander far from the reference before the loss is satisfied; large $\beta$ scales it up, so even a small move satisfies the loss and the policy stays close to the SFT model. It is the same "align but do not break the model" leash that the KL term provides in PPO, folded directly into the loss.

```
    reference (frozen)          policy (trained)
   log π_ref(y_w|x) ─┐        log π_θ(y_w|x) ─┐
                     ├─ ratio_w ──────────────┤
   log π_ref(y_l|x) ─┘        log π_θ(y_l|x) ─┘
                                              │
                     β·(ratio_w − ratio_l) ──► σ ──► maximize
```

DPO is popular because it needs no reward model, no critic, and no sampling loop — just a forward pass on chosen and rejected under both policy and reference. It is essentially supervised training on preference data: stable, cheap, and two models in memory instead of four. The trade-off is that it is **offline** (it never samples from the current policy), so it can overfit the preference set, does not explore, and is sensitive to the reference model and to distribution shift.

## 9.2 ORPO (Odds Ratio Preference Optimization)

ORPO's pitch: **merge SFT and preference training into one stage** and drop the reference model too. It adds a preference term directly onto the SFT loss, using the **odds ratio** rather than the probability ratio.

First, **odds** are just a rescaling of probability: $\text{odds}_\theta(y \mid x) = \frac{\pi_\theta(y \mid x)}{1 - \pi_\theta(y \mid x)}$. If a response has probability 0.8, its odds are $0.8 / 0.2 = 4$ ("four times as likely to happen as not"). ORPO prefers odds over raw probability because the odds ratio grows gently and does not explode when a probability approaches 1, which keeps the preference term stable while the SFT term does the heavy lifting.

$$
\mathcal{L}_{\text{ORPO}} = \mathcal{L}_{\text{SFT}}(y_w) \;-\; \lambda \cdot \log \sigma\!\left( \log \frac{\text{odds}_\theta(y_w \mid x)}{\text{odds}_\theta(y_l \mid x)} \right)
$$

| Piece                                                     | Plain meaning                                                            |
| --------------------------------------------------------- | ------------------------------------------------------------------------ |
| $\mathcal{L}_{\text{SFT}}(y_w)$                           | ordinary supervised loss teaching the model to produce the chosen answer |
| $\frac{\text{odds}_\theta(y_w)}{\text{odds}_\theta(y_l)}$ | how many times better the odds of the good answer are than the bad one   |
| $-\lambda \log \sigma(\cdot)$                             | the preference penalty: pushes those odds apart, weighted by $\lambda$   |
| $\lambda$                                                 | balances "imitate the good answer" against "avoid the bad answer"        |

**No reference model** is needed (the two odds are both under the same policy, so the reference cancels out of the ratio), which makes ORPO even lighter than DPO and collapses SFT and preference training into a single stage.

## 9.3 KTO (Kahneman-Tversky Optimization)

KTO removes the need for **pairs** altogether. Drawing on prospect theory (how humans weigh gains and losses), it trains on individual responses each labeled simply **desirable** or **undesirable**, with no matching counterpart. This is valuable in production, where you often have thumbs-up / thumbs-down signals on single answers rather than clean chosen-vs-rejected pairs. It pushes up the likelihood of desirable outputs and down for undesirable ones, weighted by a reference point, so it is the go-to when your feedback is unpaired binary labels.

## 9.4 RLOO (REINFORCE Leave-One-Out)

RLOO keeps **online** RL (it samples fresh completions from the current policy) but throws away the value network. For each prompt it samples $K$ completions and uses the **average reward of the other **$K-1$** samples as the baseline** for each one:

$$
\hat{A}_i = r_i - \frac{1}{K - 1} \sum_{j \neq i} r_j
$$

| Symbol                             | Plain meaning                                                               |
| ---------------------------------- | --------------------------------------------------------------------------- |
| $r_i$                              | the reward of completion $i$ for this prompt                                |
| $\frac{1}{K-1}\sum_{j \neq i} r_j$ | the average reward of the **other** $K-1$ completions, used as the baseline |
| $\hat{A}_i$                        | how much better completion $i$ did than its siblings                        |

The leave-one-out average of the other samples is a stand-in for "what a typical answer to this prompt earns," so subtracting it tells you whether this particular answer beat the pack. It plays exactly the role the critic $V_\psi$ played in PPO, but it is computed for free from the samples you already generated. That reduces variance without training a separate value network, so RLOO is cheaper than PPO while staying on-policy.

## 9.5 GRPO (Group Relative Policy Optimization)

GRPO, introduced with DeepSeek's reasoning models, is the workhorse of modern RL-for-reasoning. Like RLOO it samples a **group** of $G$ completions per prompt and drops the critic, but it normalizes rewards within the group into a **z-score** advantage:

$$
\hat{A}_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G)}
$$

| Symbol                         | Plain meaning                                                                           |
| ------------------------------ | --------------------------------------------------------------------------------------- |
| $G$                            | the group size: how many answers you sample for one prompt (often 8 to 64)              |
| $r_i$                          | the reward of the $i$-th answer in the group                                            |
| $\text{mean}(r_1, \dots, r_G)$ | the group's average reward, the baseline (like RLOO, but including all samples)         |
| $\text{std}(r_1, \dots, r_G)$  | the spread of rewards in the group, used to normalize                                   |
| $\hat{A}_i$                    | the **z-score**: how many standard deviations above or below average this answer scored |

For one question, generate a group of answers, grade them, and score each answer by how far above or below the group average it landed, measured in standard deviations. Answers that beat the group get a positive advantage (reinforced); answers below get a negative one (suppressed). Dividing by the standard deviation means GRPO needs **no absolute reward scale**, only relative ranking within the group, which is exactly why it pairs so naturally with verifiable 0/1 rewards (Section 11): if 3 of 8 sampled solutions pass the tests, those 3 are pushed up and the other 5 down, automatically. Each token in completion $i$ inherits that group advantage and is updated with the same clipped PPO surrogate plus a KL penalty to the reference, but with **no critic network to train**, which is the whole saving over PPO.

```
prompt x ──► sample G completions ──► grade each ──► r_1..r_G
                                              │
                     A_i = (r_i − mean)/std  ◄┘   (group-relative)
                                              │
              clipped PPO update + KL to ref ◄┘   (no critic network)
```

## 9.6 Family comparison

| Method | Online? | Reward model      | Critic | Reference | Data          | One-liner                                       |
| ------ | ------- | ----------------- | ------ | --------- | ------------- | ----------------------------------------------- |
| PPO    | Yes     | Learned RM        | Yes    | Yes       | prompts       | The original, powerful but heavy                |
| DPO    | No      | Implicit (policy) | No     | Yes       | pairs         | Preference pairs as classification              |
| ORPO   | No      | Implicit          | No     | **No**    | pairs         | SFT + preference in one stage                   |
| KTO    | No      | Implicit          | No     | Yes       | single labels | Unpaired desirable / undesirable                |
| RLOO   | Yes     | RM or verifier    | No     | Yes       | prompts       | REINFORCE with leave-one-out baseline           |
| GRPO   | Yes     | RM or verifier    | No     | Yes       | prompts       | Group-normalized advantage, great for reasoning |

---

# 10. Constitutional AI (CAI)

Human labeling is slow, expensive, and exposes raters to harmful content. **Constitutional AI** (Anthropic) replaces most human feedback with **AI feedback guided by a written set of principles** (the "constitution"). It has two stages.

### 10.1 Stage 1: supervised (critique and revise)

The model generates a response, then **critiques its own response against a constitutional principle**, then **revises** it. The revised answers become SFT data. No human is in this loop.

```
prompt ──► initial response
             │
             ├─ critique: "Which principle does this violate?"
             │
             └─ revise:  "Rewrite to follow that principle"
                          │
                          ▼
                revised response ──► SFT dataset
```

### 10.2 Stage 2: RLAIF (RL from AI Feedback)

Instead of humans labeling which of two responses is better, a **feedback model picks the preferred one using the constitution**. Those AI-generated preferences train a reward model (or feed DPO), and the rest is standard RLHF.

```
two responses ──► feedback model + constitution ──► preference label
                                                          │
                                                          ▼
                                     preference dataset ──► RM / DPO ──► RL
```

The model's values become **explicit and editable** (you change behaviour by editing the written principles, not by relabeling data), labeling scales cheaply, and human raters are shielded from harmful content. The main risk is that biases in the base model can propagate, since the model is judging itself.

---

# 11. RL with Verifiable Rewards (RLVR)

For tasks with a **checkable ground truth** (math, code, format compliance), you do not need a learned reward model at all. You replace $r_\phi$ with a **deterministic grader** that returns a reward you can trust. This removes the biggest source of reward hacking, because you cannot fool a unit test the way you can fool a neural reward model. This is the engine behind modern reasoning models, usually paired with GRPO.

### 11.1 The deterministic verification loop

```
prompt (math/code) ──► policy samples answer(s) ──► VERIFIER
                                                        │
                              ┌─────────────────────────┤
                              ▼                         ▼
                     run unit tests /            check final answer
                     execute code               against ground truth
                              │                         │
                              └────────► reward ∈ {0,1} ◄┘
                                              │
                                   GRPO / RLOO update
```

Because the reward is exact and cheap, you can sample many completions per prompt and let GRPO's group normalization do the rest. The reward signal is often sparse (1 if the final answer or all tests pass, else 0), which is exactly what group-relative advantage handles well.

### 11.2 Chain-of-Thought (CoT) optimization

The striking result of RLVR is that when you only reward the **final answer**, the model learns on its own to produce longer, more careful **chains of thought** as an instrumental strategy, because reasoning step by step raises the odds of a correct final answer. Nobody labels the reasoning; the model discovers that "think before answering" earns reward.

- Reward is on the **outcome**, not the steps, yet reasoning quality improves.
- Models learn to **backtrack, self-check, and verify** intermediate steps.
- The visible failure mode is **length hacking**: the model pads reasoning to seem thorough. A length penalty or a cap in the grader keeps this honest.

### 11.3 Structured grader rubrics

For open-ended tasks that still need judgment (essays, helpfulness, safety), the grader is an **LLM-as-judge driven by a rubric** rather than a single scalar model. The rubric decomposes quality into checkable criteria, each scored and weighted, which is far more reliable and auditable than asking for one holistic number.

```
Rubric for a support answer:
  ┌────────────────────────────┬────────┬────────┐
  │ Criterion                  │ Weight │ Score  │
  ├────────────────────────────┼────────┼────────┤
  │ Correctly solves the issue │  0.40  │  0/1   │
  │ Cites the right policy     │  0.20  │  0/1   │
  │ Polite, professional tone  │  0.15  │  0-3   │
  │ No hallucinated facts      │  0.25  │  0/1   │
  └────────────────────────────┴────────┴────────┘
  reward = Σ (weight × normalized score)
```

Good rubric practice: make each criterion **binary or on a small integer scale**, keep them independent, and calibrate the judge against a small set of human-graded examples so its scores track reality.

### 11.4 System and environment architecture

RLVR at scale is a systems problem as much as an ML problem. A generation step must sample from the policy fast, a grading step must run possibly untrusted code safely, and the training step must consume the results. The pieces:

- **Environment:** the interface that presents a prompt, accepts the model's action or answer, and returns a reward. For code it wraps a sandboxed executor; for math a symbolic checker; for tools a simulated API. It is the RL analogue of an OpenAI-Gym environment.
- **Sandboxing:** model-generated code runs inside an isolated container or microVM with no network and strict CPU/time limits, so a malicious or buggy completion cannot harm the training host.
- **Async rollout / training split:** a fast inference engine (for example vLLM) generates completions in parallel while trainer GPUs run updates, so the expensive accelerators are not idle waiting on generation.
- **Reward shaping and caching:** cache verifier results for identical completions, batch grading, and combine multiple reward terms (correctness, format, length penalty) into the final scalar.

```
┌────────────┐   prompts   ┌──────────────┐  completions  ┌─────────────┐
│  Trainer   │ ──────────► │  Inference   │ ────────────► │ Environment │
│ (GRPO/PPO) │             │ engine (vLLM)│               │  + grader   │
│  updates   │ ◄────────── │              │ ◄──────────── │ (sandboxed) │
└────────────┘   rewards   └──────────────┘   rewards      └─────────────┘
```

---

# 12. Putting It All Together: How to Choose

A practical decision guide for aligning a model today.

| Situation                                           | Recommended path                                      |
| --------------------------------------------------- | ----------------------------------------------------- |
| Need instruction following and format only          | **SFT** (with LoRA if memory-bound)                   |
| Have clean preference pairs, want simplicity        | **DPO**                                               |
| Want to skip a separate SFT stage                   | **ORPO**                                              |
| Only have thumbs-up / thumbs-down on single answers | **KTO**                                               |
| Have a verifiable reward (math, code, tests)        | **GRPO / RLOO + verifier (RLVR)**                     |
| Want maximum control, have compute and a good RM    | **PPO** with token-level KL                           |
| Cannot afford human labelers                        | **Constitutional AI / RLAIF** for the preference data |

### 12.1 The whole pipeline on one page

```
              ┌──────────────┐
  raw text ─► │ PRE-TRAINING │  next-token prediction, world knowledge
              └──────┬───────┘
                     ▼
              ┌──────────────┐
 demos     ─► │     SFT      │  imitate ideal answers (chat template, loss mask, PEFT)
              └──────┬───────┘
                     ▼
        ┌────────────┴─────────────┐
        ▼                          ▼
 ┌─────────────┐          ┌──────────────────┐
 │ RLHF path   │          │ Preference-direct │
 │ RM + PPO    │          │ DPO / ORPO / KTO  │
 │ + KL anchor │          │ (no RM, no RL)    │
 └──────┬──────┘          └────────┬─────────┘
        │                          │
        └───────────┬──────────────┘
                    ▼
          ┌───────────────────┐
          │  RLVR (optional)  │  verifiable rewards + GRPO
          │  reasoning, code  │  → chain-of-thought emerges
          └───────────────────┘
                    ▼
            aligned, reasoning model
```

---

# 13. Glossary

| Term                    | Meaning                                                                                            |
| ----------------------- | -------------------------------------------------------------------------------------------------- |
| **SFT**                 | Supervised Fine-Tuning: imitate ideal `(prompt, answer)` demonstrations                            |
| **PEFT / LoRA / QLoRA** | Train tiny low-rank adapters instead of all weights; QLoRA quantizes the frozen base to 4-bit      |
| **RM**                  | Reward Model: scores a response; trained on preference pairs with the Bradley-Terry loss           |
| **RLHF**                | RL from Human Feedback: RM + PPO with a KL penalty                                                 |
| **PPO**                 | Proximal Policy Optimization: clipped-ratio RL, four models in the loop                            |
| **KL penalty**          | Per-token term keeping the policy close to the frozen reference model                              |
| **DPO**                 | Direct Preference Optimization: preference training as classification, no RM or RL                 |
| **ORPO**                | Odds Ratio Preference Optimization: merges SFT and preference training, no reference model         |
| **KTO**                 | Kahneman-Tversky Optimization: trains on unpaired desirable / undesirable labels                   |
| **RLOO**                | REINFORCE with a leave-one-out baseline; online, no critic                                         |
| **GRPO**                | Group Relative Policy Optimization: group-normalized advantage, no critic; powers reasoning models |
| **CAI / RLAIF**         | Constitutional AI: AI feedback guided by written principles replaces human labels                  |
| **RLVR**                | RL with Verifiable Rewards: deterministic grader (tests, math checker) instead of a learned RM     |
| **CoT**                 | Chain of Thought: step-by-step reasoning that emerges when only the final answer is rewarded       |
