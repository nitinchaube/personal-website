---
title: "Voice AI Fundamentals"
date: 2026-04-15
summary: "How a real-time voice agent works — from phone audio to speech recognition, reasoning, and synthesis and what makes latency, turn-taking, and barge-in hard in production."
tags: [Multimodal, VoiceAI]
---

A working note on building voice agents, not chat with a microphone bolted on but a pipeline where audio, timing and human conversation norms all matter at once.

If you have used an LLM over HTTP, you already know half the stack. The other half is everything that happens *before* text reaches the model and *after* text leaves it and the hard part is making those halves feel instant.

---

## What makes voice different from chat

In text chat, the user signals "I'm done" by pressing Enter. On a phone call there is no Enter key. The system has to *infer* when someone finished speaking, respond while they might still be talking and keep the gap between turns short enough to feel natural.

A few properties of voice that text never forced you to think about:

- **No rewind, no scrollback:** Speech is consumed once. If the user mishears the bot, the only fix is to repeat. Prosody, pacing and clarity are part of correctness.
- **Continuous system:** Audio frames arrive every 20 ms regardless of who is "talking". Your code is always processing something.
- **Full-duplex reality: **Humans feel like only one person should talk at a time, but the network carries both directions simultaneously. Your software has to enforce the social rule.
- **Lossy channel.** Phone audio is 8 kHz μ-law (~64 kbps before compression) on the PSTN. Sibilants, soft consonants and rare words degrade more than on a Zoom call.
- **Ambient noise.** STT and VAD both suffer because of Background TV, car cabins, hospital corridors. Domain data beats general-purpose data here.
- **Prosody carries meaning.** "Right?" vs. "Right." vs. "Right!" so the pitch contour changes intent. Most pipelines throw this away when they collapse audio to text.

Humans expect **200–400 ms** between the end of one person's sentence and the start of the other's reply. Past about a second, the agent feels robotic. That constraint shapes almost every design choice below.

---

## The pipeline

Audio flows in one direction and synthesized speech flows back. In the middle, text acts as a bridge to connect both side.

```
┌────────────┐    ┌──────────┐   ┌────────────┐   ┌─────────┐   ┌──────────┐    ┌────────────┐
│  Caller    │───▶│ Telephony│──▶│   Audio    │──▶│   STT   │──▶│   LLM    │───▶│    TTS     │
│ (PSTN/SIP) │    │ (Twilio) │   │  Streamer  │   │         │   │          │    │            │
│            │◀───│          │◀──│ (WS/WebRTC)│◀──│ partials│◀──│  stream  │◀───│  stream    │
└────────────┘    └──────────┘   └────────────┘   └─────────┘   └────┬─────┘    └────────────┘
                                                                     │
                                                                     ▼ tool / function calls
                                                          ┌──────────────────────┐
                                                          │ Your APIs (CRM,      │
                                                          │ scheduling, etc.)    │
                                                          └──────────────────────┘

Cross-cutting: VAD · endpointing · barge-in · logging · evals · metrics
```


| Stage               | Role                                       | Typical vendors                         | Order-of-magnitude latency  |
| ------------------- | ------------------------------------------ | --------------------------------------- | --------------------------- |
| **Telephony**       | Bridges the phone network to your servers  | Twilio, Telnyx, Plivo                   | <50 ms (network)            |
| **Audio transport** | Streams raw frames to your app             | Twilio Media Streams, LiveKit           | <50 ms                      |
| **VAD**             | Detects speech vs. silence per tiny window | Silero, WebRTC VAD                      | <50 ms                      |
| **STT**             | Speech → text, continuously                | Deepgram, AssemblyAI, Whisper           | 180–500 ms to first partial |
| **Endpointing**     | Decides "user finished this turn"          | Often bundled with STT or a small model | 200–700 ms                  |
| **LLM**             | Reasons, replies, may call tools           | GPT-4o, Claude, fine-tuned models       | 300–800 ms to first token   |
| **TTS**             | Text → audio, continuously                 | ElevenLabs, Cartesia, OpenAI TTS        | 150–400 ms to first chunk   |
| **Tools**           | Real lookups against your systems          | Tools (can be db, web, etc)             | 100 ms – several seconds    |


---

## How the streams are wired

A voice agent is not a request/response server. It is a **long-lived session** with several concurrent streams that have to be coordinated. Picture each call as a small swarm of tasks sharing a session state:

```
Session state (per call)
├── inbound audio task       ── caller → VAD → STT
├── endpointing task         ── watches STT partials + VAD silence
├── reasoning task           ── triggers when a turn closes; talks to LLM
├── tool runner              ── executes function calls invoked by the LLM
├── outbound TTS task        ── pulls LLM tokens, produces audio chunks
├── playback task            ── writes audio frames back to the caller
└── barge-in monitor         ── watches inbound VAD during playback
```

Three properties make this work in practice:

1. **Backpressure-Friendly Queues (Decoupling Stages):** Instead of relying on direct, synchronous function calls, the system uses queues to pass data between stages. This prevents bottlenecks and ensures smooth operation; for example, if the Text-to-Speech (TTS) engine experiences a brief delay, the Large Language Model (LLM) can continue generating tokens seamlessly, buffering them in the queue until the TTS is ready.
2. **Single Source of Truth for State:** All critical system data is centralized, typically within a single in-memory Session object. This guarantees that every component references the exact same up-to-date context, managing the conversation history, pending and completed tool calls, slot values, and the current audio playback offset.
3. **Strict Cancellation Discipline:** The architecture must gracefully handle real-time interruptions, such as a user "barge-in." When an interruption occurs, the system immediately and cleanly aborts the current playback task and drops any pending TTS requests. This flushes the pipeline and ensures the next conversational turn starts from a clean, stable state.

In Python this typically means `asyncio` + `asyncio.Queue` between stages, with each vendor's WebSocket SDK running as its own coroutine. 

---

## Latency:

A naive implementation runs stages **in series**: wait for silence → finalize transcript → wait for full LLM reply → synthesize all audio → play. Add the numbers and you land around **750 ms – 2 s** before the caller hears anything which is often too slow.

```
Endpointing           200–500 ms
STT finalize           50–150 ms
LLM (first token)     300–800 ms
TTS (first chunk)     150–400 ms
Network overhead       50–150 ms
─────────────────────────────────
Sequential total      ~750 ms – 2 s   ← feels very laggy (robotic)
```

Production systems **overlap** work instead:

1. **STT streams partials** while the user is still talking.
2. **Endpointing** fires as soon as the turn is plausibly complete.
3. The **LLM streams tokens** so you do not wait for the full answer.
4. **TTS starts on the first phrase** (or first sentence), not the full reply.
5. The caller hears audio **while the LLM is still generating**.

The mental model is a pipeline of **streams**, not a pipeline of **requests**. Chat APIs trained you to wait for `response.choices[0].message.content`. Voice agents break if you keep that habit.

### What to actually measure

"End-to-end latency" is ambiguous. In Voice AI you should track these distinctly so regressions land in the right column.


| Metric           | Definition                                                    | Healthy target                 |
| ---------------- | ------------------------------------------------------------- | ------------------------------ |
| **Mouth-to-ear** | Last user audio frame → first bot audio frame heard by caller | < 800 ms P50, <1.5 s P95       |
| **STT TTFP**     | First audio frame in → first partial out                      | < 300 ms                       |
| **STT finalize** | Endpoint event → final transcript                             | < 200 ms                       |
| **LLM TTFT**     | Prompt sent → first token                                     | < 500 ms                       |
| **TTS TTFB**     | First text chunk in → first audio byte out                    | < 250 ms                       |
| **Tool RTT**     | Call out → result back                                        | < 500 ms ideal, <2 s tolerable |
| **Jitter**       | Variance in inter-frame delivery time                         | < 30 ms                        |


Watch **P95 and P99**, not P50. A single 4-second turn on a 100-call dataset will not show up in averages but will show up in every angry user.

**Endpointing** is its own dial: ~300 ms of silence after speech feels snappy but cuts off people who pause mid-thought and ~700 ms feels patient but adds perceived lag. There is no universal optimum so you need to tune for your users and domain.

---

## Turn detection:

People say "VAD" and "endpointing" interchangeably. They solve different problems at different time scales.


| Layer                              | Granularity             | Question it answers                             |
| ---------------------------------- | ----------------------- | ----------------------------------------------- |
| **VAD** (voice activity detection) | ~10–30 ms frames        | Is there speech in *this* slice of audio?       |
| **Endpointing**                    | Whole utterance         | Has the caller **finished their turn**?         |
| **Semantic endpointing**           | Meaning of partial text | Does this *sound* complete even if they paused? |


- **Voice Activity Detection (Is someone speaking right now?):** This is a fast, lightweight tool that runs directly on your device without needing a lot of computing power. It constantly checks tiny slices of audio to answer a simple "yes" or "no." It typically uses either small, smart AI models (which are great at ignoring background noise) or basic volume checks (which are even faster but might get confused by a TV playing in the background).
- **Endpointing (Is the speaker completely finished?):** This tool uses the information from the VAD to figure out when you have finished your entire thought. It does this by looking for a specific amount of silence (like a third of a second) and applying rules to make sure you didn't just take a breath or say a filler word like "um." Once it decides you are officially done talking, it signals the system to send your request to the AI.

```python
class Endpointer:
    def __init__(self, required_silence_ms=500, required_speech_ms=200):
        # The gap of silence needed to assume the person is done (e.g., 500ms)
        self.required_silence_ms = required_silence_ms
        
        # The minimum length of a sound to count as actual speech.
        # This stops a random cough or table bump from triggering it (e.g., 200ms)
        self.required_speech_ms = required_speech_ms
        
        # Variables to keep track of the timestamps
        self.time_of_last_voice = None
        self.time_speech_started = None

    def feed(self, current_time_ms, is_voice: bool) -> bool:
        # If the mic picks up a voice right now...
        if is_voice:
            self.time_of_last_voice = current_time_ms
            
            # Log the start time if this is the very first sound we're hearing
            if self.time_speech_started is None:
                self.time_speech_started = current_time_ms
                
            # They are currently talking, so the turn definitely isn't over
            return False

        # If the mic picks up silence, but no one has spoken yet, just ignore it
        if self.time_speech_started is None:
            return False
            
        # At this point, we know they WERE talking, but are now quiet.
        # So we do some quick math to check our two rules:
        
        # 1. Was it a real sentence and not just a stray noise?
        time_spent_speaking = self.time_of_last_voice - self.time_speech_started
        spoke_long_enough = time_spent_speaking >= self.required_speech_ms
        
        # 2. Have they been quiet long enough to assume they finished?
        time_spent_silent = current_time_ms - self.time_of_last_voice
        silent_long_enough = time_spent_silent >= self.required_silence_ms
        
        # If both of those are true, the turn is officially over
        return spoke_long_enough and silent_long_enough
```

**Semantic endpointing** uses a small model on the partial transcript. *"I'd like to book an appointment for, um…"* is not done speaking even if VAD sees silence. This matters anywhere callers think out loud like healthcare, support, anything emotional. 

Implementation options:

- A lightweight classifier (distil-BERT-class) fine-tuned on "complete vs. incomplete utterance" examples.
- A few-shot LLM call on the partial, with a hard latency cap (e.g. 80 ms).
- Heuristics on syntactic completeness (does the partial end on a function word, a determiner, an unfinished clause?).


| Silence timeout | Feel     | Risk               |
| --------------- | -------- | ------------------ |
| 300–500 ms      | Snappy   | False cut-offs     |
| 500–700 ms      | Balanced | Reasonable default |
| 700–1000 ms     | Patient  | Sluggish           |


---

## Barge-in (when the user talks over the bot)

While the bot plays TTS, the caller might interrupt with a real question or with **backchannels** ("uh-huh", "yeah") that are not interruptions at all.

**Naive approach is to **gnore user audio until TTS finishes and this is bullshit.

**Better approach:**

1. Keep **VAD running** on the inbound stream during playback.
2. On speech detected, **stop or pause TTS** and buffer new audio for STT.
3. Start a new turn from that transcript.

But the trap is that pure VAD stops the bot on every "mm-hm." **AI verified barge-in** pauses TTS, runs a few words through a tiny classifier or model, and either **resumes** playback (backchannel) or **hands off** to a full new turn (real interrupt).

### Mechanics worth knowing

- **Audio Ducking (The "Soft Pause")** Instead of slamming on the brakes and abruptly cutting the bot's audio the second you make a noise, the system just lowers the bot's volume by about half (6–12 dB). If the AI decides your noise was just an "uh-huh," the bot's volume swells back up. It feels much smoother than a hard stop.
- **Echo Cancellation (Ignoring Itself)** If you are talking to a bot on a speakerphone, the bot's voice comes out of the speaker and goes right back into the microphone. Without Acoustic Echo Cancellation (AEC), the bot will hear its own voice, think *you* are interrupting it, and constantly cut itself off.
- **Hardware Traffic Jams (Half-Duplex)** Some older phone networks and office phone systems are "half-duplex"—meaning audio can only travel in one direction at a time, like a one-lane bridge. If a user is on a system like this, voice interruptions physically cannot work because the bot's speaking audio blocks the user's microphone audio. In these cases, you have to design the system to use keypad presses (DTMF) for interruptions instead.
- **Remembering Its Place (Playback Offset)** If the bot pauses for an interruption, decides it was a false alarm (like a dog barking), and wants to resume speaking, it needs to know exactly where it left off. Tracking the "playback offset" ensures the bot picks up mid-sentence, rather than frustratingly repeating the entire paragraph from the beginning.

---

## STT: speech to text in real time

### How streaming ASR works under the hood

Most production streaming STT systems use one of two architectures:

- **CTC (Connectionist Temporal Classification).** A frame-level model emits a character/phoneme per audio window with blanks for silence; a CTC decoder collapses repeats and blanks. Easy to stream, every new frame can immediately produce output. Slightly less accurate on long-range context than encoder-decoder models.
- **RNN-T / Transformer-Transducer.** Two networks (audio encoder + label predictor) trained jointly with a special "transducer" loss. Streams naturally and handles context better than CTC. This is what Deepgram, AssemblyAI and most modern providers use under the hood.
- **Encoder-decoder (Whisper-style).** Process audio in chunks, run an autoregressive decoder. Excellent accuracy but **not naturally streaming**, vendors that offer "streaming Whisper" usually chunk audio and stitch outputs, which adds latency.

### Choosing a model (trade-offs, not rankings)


| Option                        | Good at                               | Watch out for                            |
| ----------------------------- | ------------------------------------- | ---------------------------------------- |
| **Deepgram Nova**             | Low streaming latency (~180 ms), cost | Rare domain words without a custom vocab |
| **AssemblyAI Universal**      | Accents, streaming                    | Slightly higher latency                  |
| **Whisper large**             | Accuracy, languages                   | Not natively built for live streaming    |
| **Fine-tuned / custom vocab** | Product names, jargon, medical terms  | Ongoing maintenance                      |


### Custom vocabulary, in three escalating tiers

1. **Keyword boost / hot-words.** Pass a list of phrases at session start with a boost weight. The decoder gives them a prior. Cheapest, instant, good for tens of terms.
2. **Custom vocabulary models.** Provider-side language model adaptation, sometimes including pronunciations. Better recall on domain terms; usually a few hours to deploy.
3. **Full fine-tuning.** Train on your own labeled audio. Highest accuracy and the only good answer for heavy accents, code-switching, or specialist domains; costliest and slowest to iterate.

### Gotchas that bite in production

- **Domain vocabulary**: product codes, drug names, internal terms. Fix with keyword boosts or fine-tuning; do not assume general STT will guess right.
- **Spoken numbers**: "twenty twenty-four" vs. "2024", dates, phone numbers, dosages. Always run a normalization pass after STT, or pin slot values via a structured re-extraction step.
- **Accents**: Report word-error rate **per cohort**, not one global number. Average WER hides catastrophic failures on minority cohorts.
- **Diarization** : Telling who is speaking. Trivial on a 2-leg phone call (each leg is its own track). Hard when multiple people share a mic, which happens more than you think (family in waiting rooms, conference rooms).

---

## The LLM:

On a call the model wears several hats at once:

1. **Dialogue**: stay on script enough to be reliable, flexible enough to handle messy speech.
2. **Tool caller**: invoke APIs when facts must come from your systems, not from weights.
3. **Slot filling**: pull structured fields (name, date, intent) from messy utterances.
4. **Guardrails**: refuse out-of-scope asks; escalate to a human when stuck.

### Prompting for voice

Voice prompts look different from chat prompts. A few rules worth burning into the system message:

- **Speak in short turns.** One or two sentences per response by default. Long monologues feel like a lecture and balloon TTS latency.
- **No markdown, no lists, no code.** Anything you write goes through TTS. `**bold**` becomes "asterisk asterisk bold asterisk asterisk".
- **Spell out structured data.** Phone numbers as digits ("five five five, one two three…"), prices as words and digits ("twelve dollars"), times in the user's likely format.
- **Ask one thing at a time.** Compound questions ("What's your name, date of birth, and reason for the visit?") collapse to whichever fragment the user remembers.
- **Repeat back critical slots**. before acting ("Just to confirm, that's…"). Phone audio plus STT errors mean your model's internal state and reality can drift.
- **Define an escalation path.** "If the caller asks for a human, the price of X, or anything outside [scope], say *I'll transfer you*." Make refusal a first-class behavior.

### Streaming is mandatory

Enable token streaming and **pipe tokens to TTS as they arrive**. Waiting for the full completion before synthesis recreates the sequential latency problem.

Practical chunking strategy:

1. Buffer LLM tokens into a small text buffer.
2. Flush to TTS when you hit a sentence-ending punctuation, an em-dash, or N tokens (e.g. 30) without one.
3. Never split a number, abbreviation, or proper noun across TTS calls if you can avoid it then pronunciation suffers.

### Tool calls mid-conversation

When the model emits a tool call, your runtime typically:

1. Detects the call in the stream.
2. **Pauses or masks TTS**: Silence is worse than a short filler.
3. Runs the API.
4. Injects the result into context.
5. Lets the model continue speaking with grounded facts.

Example tool definitions (shape varies by SDK):

```python
tools = [
    {
        "name": "lookup_customer",
        "description": "Find a customer record by name and account ID",
        "parameters": {"name": "string", "account_id": "string"},
    },
    {
        "name": "find_open_slots",
        "description": "List available appointment slots for a provider and date range",
        "parameters": {"provider_id": "string", "start": "date", "end": "date"},
    },
]
```

A few patterns worth adopting:

- **Parallel tool calls:** If the model needs both `lookup_customer` and `find_open_slots`, fire them concurrently and merge results before continuing.
- **Idempotency keys: **Voice connections drop. If the user has to call back, the agent must not double-book or double-charge. Generate an idempotency key per business action and persist it on the session.
- **Confirmations as separate turns:** Get the slots, *speak them back*, get a yes, *then* call the mutating tool. Never write to a system of record without an explicit confirmation token in context.
- **Strict output grammar:** Use JSON Schema or constrained decoding for slot extraction. Free-text "the date is March 15" creates ambiguity; `{"date": "2026-03-15"}` does not.

**On Managing Latency with Filler Audio**: In voice AI, dead air during a tool call is the fastest way to break the user experience because callers will assume the system dropped and start talking over it. To manage that one-to-two second latency gap, the architecture needs a deterministic filler strategy. The moment a tool call is triggered, we immediately push a pre-recorded or low-latency streamed phrase like 'pulling that up now...' over the WebSocket. It completely masks the backend API execution time, maintains natural turn-taking, and keeps the user engaged without them ever realizing they're waiting on a database.

**On the Strict Grounding Rule for Healthcare** : For healthcare applications, strict data grounding isn't a feature; it's a non-negotiable safety requirement. Hallucinating an appointment time or a billing balance isn't a model quirk, it's a critical failure. The core architectural rule I follow is that the agent can never confirm a fact it hasn't explicitly received from a validated tool result. To enforce this, we don't allow optimistic generation. We use an inline gate that completely blocks the LLM from generating a confirmation utterance until the backend EHR or scheduling API returns a verified success code, guaranteeing the agent only speaks absolute truth."

---

## TTS: text to speech

Optimize for **time to first audio chunk** first and polish second.


| Dimension           | Why it matters                                                   |
| ------------------- | ---------------------------------------------------------------- |
| First-chunk latency | Dominates perceived responsiveness (< 300 ms is a common target) |
| Naturalness         | Trust and comprehension on phone audio                           |
| Voice consistency   | Same persona across calls                                        |
| Streaming protocol  | WebSocket / chunked HTTP vs. wait-for-full-file                  |
| SSML / lexicon      | Pauses, emphasis, forced pronunciations                          |



| Vendor (2026)    | Strength                  | Caveat                           |
| ---------------- | ------------------------- | -------------------------------- |
| ElevenLabs Turbo | Very natural, ~150 ms     | Cost at scale                    |
| Cartesia Sonic   | Very low latency (~90 ms) | Newer ecosystem                  |
| OpenAI `tts-1`   | Simple, cheap             | Less control                     |
| Deepgram Aura    | Pairs with their STT      | Convenient if already integrated |


### How streaming TTS actually works

Two protocols always dominates:

- **Chunked HTTP / SSE : **Open a single POST, audio streams back in chunks of ~50–200 ms. Simple, works through most proxies. Slightly higher overhead per request.
- **Persistent WebSocket: **Open once at session start, push text chunks across the session, receive audio back. Lowest latency, requires careful reconnect logic. (BEST FOR PRODUCTION)

The first audio chunk is the one users feel. After that, your job is to keep the buffer non-empty: TTS should be producing the *N+1* sentence while the caller hears the *N*th.

### SSML and pronunciation control

SSML (Speech Synthesis Markup Language) is widely but inconsistently supported. The most useful tags:

```xml
<speak>
  Your appointment is on
  <say-as interpret-as="date" format="mdy">3/15/2026</say-as>
  at <say-as interpret-as="time" format="hms12">2:30pm</say-as>.
  <break time="300ms"/>
  Is that correct?
</speak>
```

- `<say-as interpret-as="...">` : coerce dates, times, digits, ordinals.
- `<break time="...">` : explicit pause; great after greetings and before confirmations.
- `<phoneme>` : override pronunciation for a stubborn proper noun.
- `<emphasis>` : make a word land. Use sparingly.

If your vendor does not support SSML, normalize text before sending: `"$1,234.50"` → `"one thousand two hundred thirty four dollars and fifty cents"`.

### Voice cloning

Cloning lets you keep a consistent persona, but adds responsibility:

- **Consent.** Always record explicit consent from the original speaker. Many jurisdictions now require it.
- **Drift across model upgrades.** A "Sarah" voice can subtly shift when the vendor updates the underlying model. Pin model versions if persona consistency is part of the product.

---

## Telephony and audio formats

Two common paths are applied in industry:

1. **PSTN(Public Switched Telephone Network) → carrier (e.g. Twilio) → WebSocket frames → your server**
2. **WebRTC (LiveKit, Daily) → SFU → your worker** : lower latency, great for browser/voice-app, more moving parts for plain PSTN.

### A bit of SIP/RTP background

- **SIP** (Session Initiation Protocol) is the signaling layer that sets up, modifies and tears down calls. It negotiates "we're going to send audio, here's the codec, here's the IP and port".
- **RTP** (Real-time Transport Protocol) is the media layer that actually carries audio frames over UDP,  typically 20 ms per packet for telephony codecs.
- **Carriers** (Twilio, Telnyx) abstract both: you usually only see a WebSocket of base64-encoded audio frames plus event JSON for start/stop/DTMF.

### Codecs and audio formats


| Codec           | Sample rate | Bitrate    | Where you see it                       |
| --------------- | ----------- | ---------- | -------------------------------------- |
| **G.711 μ-law** | 8 kHz       | 64 kbps    | North American PSTN, default on Twilio |
| **G.711 A-law** | 8 kHz       | 64 kbps    | European PSTN                          |
| **G.722**       | 16 kHz      | 64 kbps    | HD voice between SIP endpoints         |
| **Opus**        | 8–48 kHz    | 6–510 kbps | WebRTC, modern VoIP                    |
| **PCM 16-bit**  | 16 kHz      | 256 kbps   | What STT APIs usually want internally  |


Two conversions you will see or configure:

- **μ-law decode** (companded 8-bit → linear 16-bit PCM). Single lookup table; effectively free.
- **Resample 8 kHz → 16 kHz.** A polyphase filter; &lt;10 ms latency for typical chunk sizes.

### Jitter buffer and packet loss

UDP packets arrive out of order or not at all. The audio pipeline needs a **jitter buffer** : a small queue (typically 40–100 ms) that reorders and smooths inbound frames. Tradeoffs:

- Larger buffer → more resilience to network jitter, more added latency.
- Smaller buffer → snappier, more glitches.

Most carrier-side integrations handle this for you. If you ingest raw RTP, plan for it.

### DTMF (keypad tones)

Touch-tone digits arrive either as **in-band audio tones** or as **RFC 2833 / SIP INFO** out-of-band events. They are how callers say "press 1 to confirm" and how legacy IVRs hand off to your agent. Capture them as first-class events on the session — do not try to detect them through STT.

Formats you will convert between in your app:


| Format          | Where you see it                 |
| --------------- | -------------------------------- |
| **μ-law 8 kHz** | Classic phone audio (narrowband) |
| **PCM 16 kHz**  | What many STT APIs want          |
| **Opus**        | WebRTC                           |


Resampling 8 kHz → 16 kHz is cheap (<10 ms). Plan for it in your audio path, not as an afterthought.

---

## Conversation state and memory

The session object is where the agent's "mind" lives between turns. A reasonable shape:

```python
@dataclass
class Session:
    call_id: str
    user_id: str | None
    transcript: list[Turn]            # full conversation history
    slots: dict[str, Any]             # extracted fields: name, dob, intent, ...
    tool_results: list[ToolCall]      # what we've looked up + when
    pending_action: Action | None     # awaiting explicit confirmation
    idempotency_keys: dict[str, str]  # per business action
    flags: set[str]                   # e.g. "human_requested", "escalated"
```

A few patterns that prevent classic voice-agent bugs:

- **Slots are sticky.** Once a value is confirmed, do not let later turns silently overwrite it; require the model to explicitly say "the user is changing X from A to B".
- **Re-extract on every turn.** Run a small structured-output prompt over the latest user message to extract slot updates, instead of relying on the dialog LLM to maintain state perfectly.
- **Window the prompt context.** Keep the full transcript for analytics but only send the last N turns + a slot summary to the LLM. This caps token cost and keeps the model focused.
- **Externalize long memory.** Anything that survives across calls (preferences, account context) lives in a database keyed by `user_id`, hydrated at call start, written back at call end.

---

## Privacy, security, compliance

Voice content is unusually sensitive: it carries voice biometrics, often PII or PHI, and frequently spans regulated industries.

### Data flow obligations

- **PII / PHI redaction at the log boundary.** Names, DOBs, account numbers, free-text health info. Redact in transcripts before they hit your log store; keep an encrypted, access-controlled raw copy if legally required.
- **In-transit encryption.** SIP/RTP can be unencrypted; insist on **SRTP** (encrypted RTP) where available. WebRTC is encrypted by default (DTLS-SRTP).
- **At-rest encryption.** Recordings, transcripts, vector indices over them — all encrypted with managed keys.
- **Vendor data use.** STT and TTS vendors have varying defaults for training on your data. Confirm and pin the "do not train" setting for every vendor in writing.

### Regulatory baselines

- **HIPAA** (US healthcare). Requires a Business Associate Agreement (BAA) with every vendor in the data path. Many cheap consumer voice APIs do not offer one.
- **GDPR** (EU). Voice is personal data; recording requires lawful basis (often explicit consent at call start). Honor deletion requests across all vendors.
- **SOC 2 / ISO 27001.** Not regulations but commonly required by enterprise customers. Affects vendor selection more than runtime code.
- **TCPA** (US). Outbound automated calls have strict consent rules. Voice AI does not get a pass.

### Consent prompts

The first few seconds of a call usually need: "This call may be recorded for quality. To continue, say yes." Capture the consent token in your session and refuse to record without it.

---

## Observability and evaluation

Unlike text-based chat systems where a slow response or a delayed database call simply means a spinning loading wheel, voice applications fail in ways that standard request-response logs completely mask. Things like user barge-ins, broken turn-taking loops, premature cut-offs, and silent tool-execution waits are native audio-layer failures. To manage a real-time voice pipeline, per-request metrics aren't enough so you need granular, per-turn telemetry to capture exactly when, where and why the conversational flow broke down.

### Per-turn record (the unit of debugging)

A turn is the one user utterance +  the bot's response is the natural granularity. 

Log roughly:

```json
{
  "call_id": "ca_01HXX...",
  "turn_index": 7,
  "user_audio_ms": 2840,
  "stt": {
    "interim_count": 12,
    "final": "I'd like to refill my prescription",
    "confidence": 0.92,
    "ttfp_ms": 220,
    "finalize_ms": 140
  },
  "endpoint": { "silence_ms": 520, "semantic_score": 0.88 },
  "llm": {
    "model": "gpt-4o",
    "ttft_ms": 410,
    "tokens_in": 1840,
    "tokens_out": 86,
    "tools_called": ["lookup_prescription"]
  },
  "tools": [
    { "name": "lookup_prescription", "latency_ms": 380, "ok": true }
  ],
  "tts": { "ttfb_ms": 190, "duration_ms": 3100 },
  "mouth_to_ear_ms": 740,
  "barge_in": false,
  "flags": []
}
```

This record alone gives you P50/P95 dashboards, latency breakdowns per stage, regression detection, and post-mortem material when a call goes sideways.

### Metrics that are be worth logging every call

- Time-to-first-response after each user turn (P50 / P95 / P99)
- STT confidence and endpoint events
- LLM tokens in/out per turn
- Tool call count and latency
- Barge-in and human-transfer events
- Whether the stated goal completed (booking made, ticket created, etc.)

### Quality in layers (no single score)


| Layer            | What it does                                                                  |
| ---------------- | ----------------------------------------------------------------------------- |
| **Heuristics**   | Required slots filled? Repeated "I don't understand"? Goal API succeeded?     |
| **LLM-as-judge** | Post-call rubric scores (helpful, safe, on-policy) : cheap, needs calibration |
| **Human review** | Sample of calls/week as ground truth                                          |
| **User signal**  | Callback rate, survey, task completion proxies                                |


### Some of the Eval techniques worth knowing are:

- **WER (Word Error Rate)** for STT, but compute it **per cohort** (accent, ambient noise, domain) so cohort-specific regressions are visible.
- **MOS (Mean Opinion Score)** for TTS : a 1–5 perceived-quality rating, usually estimated by a model (e.g. NISQA) rather than humans.
- **Task success rate**: Did the agent complete the user's stated goal? The most important number, hardest to measure automatically.
- **Conversation simulators.** A second LLM plays "the caller" against your agent through real STT/TTS. Lets you A/B test prompt changes against thousands of synthetic conversations before any real customer hears them.
- **Shadow traffic.** Run a new prompt or model on the same audio as the production agent without affecting the user, compare offline.

### Regression harness

Keep a set of **golden transcripts** (and optionally synthetic audio) with expected outcomes. Replay them when you change prompts, models, or endpointing. Voice regressions are subtle; automate what you can before shipping.

---

## Speech-to-speech (the emerging shortcut)

Recent models (e.g. GPT-4o realtime, Gemini live, Moshi) skip the text round-trip and process **audio in, audio out** directly. Implications:

- **Lower latency.** No separate STT and TTS stages; first audio out can land in 200–300 ms.
- **Preserves prosody and non-verbal cues.** The model "hears" tone, hesitation, laughter and can respond in kind.
- **Different tradeoffs.** Tool calling is still text-mediated. Observability is harder (no clean intermediate transcript). Latency wins are real but quality on long-tail domain terms can lag specialist STT.

Most production stacks in 2026 still use the explicit STT → LLM → TTS pipeline for control, evaluability, and vendor flexibility. The hybrid is becoming common: realtime S2S for the front of the call (fast, natural), with structured slot extraction and tool calls running on side prompts.

---

## Full Pipeline

Putting the whole pipeline together, here is what a single turn looks like with timings on a healthy production stack:

```
t=0 ms      Caller starts speaking: "I'd like to refill—"
t=200 ms    First STT partial: "I'd like"
t=900 ms    Caller still speaking: "—my prescription"
t=1200 ms   Caller stops speaking. VAD silent.
t=1500 ms   Endpointer fires (300 ms of silence after speech).
            Final transcript: "I'd like to refill my prescription"
t=1520 ms   Prompt assembled (history + slots) and sent to LLM.
t=1880 ms   First LLM tokens stream back: "Sure, let me look that up..."
t=1900 ms   First sentence flushed to TTS.
t=2080 ms   First TTS audio chunk reaches the caller.
            Mouth-to-ear latency: 880 ms.
t=2100 ms   LLM emits tool call: lookup_prescription(user_id)
t=2120 ms   Tool runner fires API; LLM paused awaiting result.
t=2480 ms   Tool result returned. Re-prompt LLM with result.
t=2700 ms   LLM continues: "I see one ready at Walgreens on Main..."
t=2720 ms   Next TTS chunk queued.
t=5800 ms   Bot finishes speaking; VAD watches inbound.
t=6200 ms   Caller: "Yes, please refill it."
            (next turn begins)
```

Three things make this work:

1. **TTS started at 2080 ms**, while the LLM was still generating and before the tool call had even fired.
2. **The tool call did not produce silence** because the model had already said "let me look that up".
3. **The session state carried** the `user_id` so the tool did not have to re-ask the caller.

---

## Failure modes:


| Symptom                     | Likely cause                         | Direction for a fix                                    |
| --------------------------- | ------------------------------------ | ------------------------------------------------------ |
| Bot confirms wrong facts    | Model spoke without tool grounding   | Verify tool output before TTS; constrain confirmations |
| User cut off mid-sentence   | Endpointing too aggressive           | Longer silence threshold; semantic endpointing         |
| Bot stops on every "uh-huh" | VAD-only barge-in                    | Backchannel classifier                                 |
| Awkward silence             | Slow tool API                        | Fillers + timeout → escalate                           |
| Sensitive data in logs      | Raw transcript logging               | Redact at ingest                                       |
| Misheard jargon             | OOV terms in STT                     | Custom vocabulary / fine-tune                          |
| Same reply in a loop        | Model stuck                          | Loop detector → human                                  |
| Off-topic answers           | Weak system prompt                   | Intent guardrail + scope rules                         |
| Choppy audio                | Packet loss                          | Jitter buffer, reconnect                               |
| Double-booked appointment   | Reconnect after drop, no idempotency | Idempotency keys on mutating tool calls                |
| Wrong persona after upgrade | Vendor model auto-rolled forward     | Pin model versions; canary new versions                |
| Echo loop on speakerphone   | TTS bleeding into mic                | Confirm AEC enabled; check half/full duplex            |


---

## Sources:

### Telephony & audio transport

- [Twilio Media Streams](https://www.twilio.com/docs/voice/media-streams): WebSocket audio frames from PSTN calls
- [Telnyx Voice API](https://developers.telnyx.com/docs/voice): alternative carrier with streaming media
- [LiveKit docs](https://docs.livekit.io/): WebRTC SFU for browser and app voice
- [WebRTC 1.0 specification](https://www.w3.org/TR/webrtc/): browser real-time media standard
- [RFC 3550: RTP](https://www.rfc-editor.org/rfc/rfc3550): real-time transport protocol reference

### VAD, STT & speech recognition

- [Silero VAD](https://github.com/snakers4/silero-vad): lightweight neural VAD used in many pipelines
- [Deepgram streaming STT](https://developers.deepgram.com/docs/getting-started-with-live-streaming-audio): live transcription API
- [AssemblyAI streaming](https://www.assemblyai.com/docs/speech-to-text/streaming): real-time STT with turn detection options
- [OpenAI Speech-to-Text](https://platform.openai.com/docs/guides/speech-to-text): Whisper-based transcription API
- [Whisper paper (arXiv)](https://arxiv.org/abs/2212.04356): original model and evaluation approach

### LLM, tools & realtime voice

- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime): speech-to-speech and low-latency multimodal sessions
- [OpenAI function calling](https://platform.openai.com/docs/guides/function-calling): tool schema and invocation patterns
- [Anthropic tool use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use): Claude tool-calling reference
- [Python `asyncio` docs](https://docs.python.org/3/library/asyncio.html): queues, tasks, and cancellation for streaming pipelines

### TTS & speech synthesis

- [ElevenLabs API docs](https://elevenlabs.io/docs/api-reference/introduction): streaming TTS and voice cloning
- [Cartesia docs](https://docs.cartesia.ai/): low-latency streaming synthesis
- [OpenAI Text-to-Speech](https://platform.openai.com/docs/guides/text-to-speech): `tts-1` / `tts-1-hd` API
- [W3C SSML 1.1](https://www.w3.org/TR/speech-synthesis11/): pronunciation, pauses, and `say-as` markup

### Observability, eval & quality

- [Deepgram: measuring voice agent latency](https://deepgram.com/learn/measuring-latency-in-voice-agents): mouth-to-ear and per-stage breakdowns
- [OpenAI evals guide](https://platform.openai.com/docs/guides/evals): structured evaluation and regression testing patterns

### Privacy, compliance & security

- [HHS HIPAA guidance](https://www.hhs.gov/hipaa/for-professionals/index.html): US healthcare data handling baseline
- [GDPR: European Commission overview](https://commission.europa.eu/law/law-topic/data-protection_en): EU personal data and recording consent
- [FCC TCPA rules](https://www.fcc.gov/consumers/guides/stop-unwanted-robocalls-and-texts): US outbound automated-call restrictions

### Background reading

- [ITU-T G.711](https://www.itu.int/rec/T-REC-G.711): μ-law / A-law telephony codec standard
- [Rasa: voice AI architecture overview](https://rasa.com/blog/voice-ai-architecture/): accessible end-to-end pipeline walkthrough