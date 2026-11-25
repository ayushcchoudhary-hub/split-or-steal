# Architecture & Experiment Decision Log (ADRs)

Every major technical, modeling, game-design, and infrastructure choice in **Split or Steal — Persona Simulation Lab** is logged here with context, alternatives considered, rationale, and observable outcomes. Designed both as an engineering paper trail and as a concise reference for explaining tradeoffs and results later.

---

## Index

| ID | Date | Decision | Status |
| :--- | :--- | :--- | :--- |
| [ADR-001](#adr-001-3-round-escalating-split-or-steal-with-private-thought-monologue) | 2026-11-25 | 3-Round Escalating Split or Steal (\$100 → \$500 → \$1,000) + Private `<thought>` Monologue | Accepted |
| [ADR-002](#adr-002-weight-level-sft--dpo-lora-with-3-way-scientific-control) | 2026-11-25 | Weight-Level SFT + DPO LoRA with 3-Way Scientific Control (`Prompt` vs. `SFT` vs. `SFT+DPO`) | Accepted |
| [ADR-003](#adr-003-hybrid-compute-apple-m4-local-dev--ephemeral-colab-cli-gpus) | 2026-11-25 | Hybrid Compute: Apple M4 (24 GB) Local Dev + Ephemeral Colab CLI GPUs (`T4`/`L4`/`G4`/`A100`/`H100`) | Accepted |
| [ADR-004](#adr-004-0-hosting-3d-blender-apartment-showcase--open-source-synthetic-tts) | 2026-11-25 | \$0-Hosting Showcase: Blender Apartment (`.glb` → Three.js) + Stylized Avatars + Open-Source Synthetic TTS | Accepted |

---

## ADR-001: 3-Round Escalating Split or Steal with Private `<thought>` Monologue

- **Date:** 2026-11-25
- **Status:** Accepted

### Context
A single-shot Prisoner's Dilemma or Split or Steal game produces noisy binary outcomes and cannot measure trust-building, betrayal, retaliation, or endgame defection. Conversely, a 10+ round game inflates token costs and makes replay clips slow to watch.

### Options Considered
1. **1-Round Split or Steal:** Fast and cheap, but cannot observe revenge, forgiveness, or reputation dynamics.
2. **3-Round Escalating Split or Steal (\$100 → \$500 → \$1,000), 2 negotiation turns/round:** Compact enough for 50–100 simulations per matchup and crisp video/3D replays, while creating a classic game-theoretic arc (low-stakes trust test → mid-stakes temptation/retaliation → high-stakes endgame backstab).
3. **5–10 Round Iterated Game:** Richer long-horizon strategy, but 3×–5× token cost and harder to narrate visually.

### Decision
Adopt **3-Round Escalating Split or Steal (\$100 → \$500 → \$1,000)** with **2 back-and-forth negotiation turns per player per round**, a **private `<thought>` reasoning block** invisible to the opponent, and a **strict structured final action** (`SPLIT` or `STEAL`), with round outcomes and running bankrolls carried into subsequent rounds. Keep stakes and turn counts configurable in code so we can ablate them later.

### Consequences & Interview Framing
- **Unlock:** Separates *what a persona promises publicly* from *what it plans privately*, enabling direct measurement of **First-Strike Betrayal Rate**, **Revenge Rate**, **Forgiveness Rate**, **Endgame (Round 3) Backstab Rate**, and **Deception Gap** (promised Split vs. executed Steal).
- **Risk:** Small context windows or poor structured-output adherence could break parsing; mitigated by strict schema/grammar constraints during vLLM inference.

---

## ADR-002: Weight-Level SFT + DPO LoRA with 3-Way Scientific Control

- **Date:** 2026-11-25
- **Status:** Accepted

### Context
Prompting off-the-shelf RLHF instruct models (*"Act like X"*) frequently produces superficial catchphrases layered over identical polite, risk-averse, cooperative assistant behavior. If all prompted personas converge on "cooperate politely," the experiment has no real behavioral signal.

### Options Considered
1. **Prompt-Only Personas:** Fastest to build, zero training cost, but vulnerable to strong RLHF compliance/politeness priors.
2. **SFT LoRA Only:** Teaches authentic cadence, vocabulary, and framing from primary-source text, but SFT alone often preserves underlying RLHF refusal/cooperation preferences under dilemma pressure.
3. **SFT + DPO LoRA (3-Way Control):** Train SFT on primary-source voice/reasoning, then apply DPO preference pairs (`chosen` = authentic in-character decision/reasoning vs. `rejected` = generic polite AI assistant hedging), and benchmark **Prompt-Only vs. SFT vs. SFT+DPO** identically across every matchup.

### Decision
Implement a **3-way controlled experiment** (`Prompt-Only` vs. `SFT LoRA` vs. `SFT+DPO LoRA`) on a shared 8B–14B open-weights base model (exact checkpoint TBD in Phase 0/1 benchmark), serving all character adapters concurrently via **vLLM Multi-LoRA**.

### Consequences & Interview Framing
- **Scientific rigor:** Turns a fun demo into a measurable post-training study answering: *"Does preference optimization (DPO) on character worldviews move downstream game-theoretic decisions beyond what prompting or SFT alone achieves?"*
- **Data requirement:** Requires high-quality, copyright-clean primary-source corpora per character plus carefully constructed DPO dilemma pairs that do not leak the Split or Steal game itself (preventing trivial memorization).

---

## ADR-003: Hybrid Compute (Apple M4 Local Dev + Ephemeral Colab CLI GPUs)

- **Date:** 2026-11-25
- **Status:** Accepted

### Context
Local hardware is an **Apple M4 MacBook (24 GB unified memory)** — great for local prototyping, code iteration, and small sanity checks, but slow for multi-character 8B–14B LoRA SFT+DPO training and 1,000-game batched rollouts. We have a **Google AI Ultra** plan (~1,000 Colab Compute Units/month; verified starting balance: `1,209.41 CUs`) accessible programmatically via `google-colab-cli`.

### Options Considered
1. **Local M4 Only (MLX/PyTorch MPS):** Zero cloud cost, but multi-hour/multi-day training and slow batch inference.
2. **Always-On Cloud VM (RunPod / Lambda / GCP):** Predictable hardware, but incurs out-of-pocket cash spend and risks idle billing.
3. **Hybrid M4 + Ephemeral `colab-cli` Jobs:** Develop and unit-test locally on M4; run heavy training (`A100`/`H100`), vLLM Multi-LoRA tournaments (`A100`/`G4` 96 GB), and GPU TTS (`L4`) as short-lived, auto-torn-down Colab CLI jobs capped well under 20% of a monthly CU allowance (target: ~35–75 CUs for v1).

### Decision
Use the **Hybrid M4 + Ephemeral Colab CLI** workflow (`google-colab-cli 0.8.0` in Python 3.14 `.venv`). Enforce strict cost guardrails: check `colab usage` before/after every phase, log every run in [WORK_MAP.md](./WORK_MAP.md), cap training steps/wall-clock time, and tear down sessions immediately upon job completion.

### Consequences & Interview Framing
- **Concrete evidence:** Running an early M4 vs. Colab GPU micro-benchmark ("Silicon Receipt") quantifies the exact speedup and CU cost per character/simulation rather than guessing.
- **Portability:** Because training and simulation scripts run headlessly via CLI rather than manual notebook clicks, runs are reproducible and version-controlled.

---

## ADR-004: \$0-Hosting 3D Blender Apartment Showcase + Open-Source Synthetic TTS

- **Date:** 2026-11-25
- **Status:** Accepted

### Context
Raw JSON logs and tables alone don't convey the psychological drama of negotiation, bluffing, and private inner thoughts. At the same time, live LLM/TTS inference in a public web app creates ongoing API hosting costs, and cloning real actors' or living people's voices introduces legal/ethical risks.

### Options Considered
1. **Static Dashboard Only (Charts + Transcripts):** Simple, \$0 hosting, but misses the memorable storytelling hook.
2. **Live Interactive Web App with Cloud GPU Backend:** Lets users play live, but requires paid GPU hosting and rate-limiting.
3. **Pre-Rendered Static Replay Theater + Analytics Dashboard (\$0 Hosting):** Export Ayush's existing Blender apartment (`.blend` → compressed `.glb`) into Three.js / React Three Fiber, seat stylized 3D avatars inside it, pre-render highlight game audio on Colab L4 using open-source royalty-free synthetic TTS (e.g., Kokoro / Chatterbox — no voice cloning of real individuals), and host the entire web app statically for \$0.

### Decision
Build the **\$0-hosting static Three.js 3D Apartment Replay Theater + 100-Sim Analytics Dashboard** with toggleable *Spoken Dialogue* vs. *Hidden `<thought>` Monologue* and an embedded *Silicon Receipt* showing exact compute usage.

### Consequences & Interview Framing
- **Zero ongoing burn:** Scales to arbitrary traffic on static hosting (Vercel / GitHub Pages) at \$0/month.
- **Copyright hygiene:** Stylized avatars + synthetic voice archetypes keep the project clean for open-source release.
