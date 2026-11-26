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
| [ADR-005](#adr-005-sprint-1-vertical-slice--qwen3-8b-machiavelli-vs-gandhi-full-corpus-cpt--sft--dpo--modular-character-engine) | 2026-11-25 | Sprint 1 Vertical Slice: `Qwen/Qwen3-8B`, Machiavelli vs. Gandhi, Full-Corpus CPT + SFT + DPO, Modular Character Engine & Replayable Storage | Accepted |
| [ADR-006](#adr-006-iteration-2-post-training-calibration--eos-preservation-unconditional-weight-baking--calibrated-cptdpo-lrs) | 2026-11-25 | Iteration 2 Post-Training Calibration: EOS Preservation, Unconditional Persona Weight Baking & Calibrated CPT/DPO LRs | Accepted |
| [ADR-007](#adr-007-sprint-2a--fair-primary-source-dpo-unprompted-arena-control--qwen3-14b-scale-up-test) | 2026-11-26 | Sprint 2A: 100% Primary-Source DPO (Anti-Overtraining), Unprompted Weights-Only Arena Control & `Qwen/Qwen3-14B` Scale-Up Test | Accepted |
| [ADR-008](#adr-008-empirical-verdict-on-primary-source-dpo-vs-synthetic-dpo--phase-scoped-anti-repetition) | 2026-11-26 | Empirical Verdict on 100% Primary-Source DPO vs. Synthetic Dilemma DPO & Phase-Scoped Anti-Repetition | Accepted |

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

---

## ADR-005: Sprint 1 Vertical Slice — Qwen3-8B, Machiavelli vs. Gandhi, Full-Corpus CPT + SFT + DPO & Modular Character Engine

- **Date:** 2026-11-25
- **Status:** Accepted

### Context
Before curating 6+ character corpora, scaling training, or building the 3D showcase, we want to:
1. Prove empirically that weight-level fine-tuning measurably shifts an open model's weights, held-out perplexity, zero-prompt decision priors, and Split or Steal gameplay compared to a `Prompt-Only` control.
2. Test training on **all primary-source text** (raw books, letters, speeches, dispatches via Continued Pre-Training / CPT) combined with conversational SFT and anti-RLHF DPO, rather than only narrow Q&A pairs.
3. Architect the pipeline so **any** character can be dropped in via a single config file, trained, verified against the base model, and pitted in configurable-turn simulations whose full transcripts are stored cleanly for future replay, analytics, or Supabase import.

### Decision
1. **Base Model:** Start with **`Qwen/Qwen3-8B`** (Apache 2.0, ungated, native `<think>` private monologue, fast Colab A100/L4 LoRA training, first-class vLLM Multi-LoRA support). Re-evaluate `Qwen3-14B` after inspecting Sprint 1 quality.
2. **Vertical-Slice Pair:** **Niccolò Machiavelli** vs. **Mahatma Gandhi** (100% US Public Domain, maximally opposite trust priors, massive clean English corpora). Final 6+ cast expansion (e.g., adding Buffett, Ali, Wilde, Van Gogh, and/or Richard Nixon) happens after Sprint 1 proves the pipeline end-to-end.
3. **3-Stage Modular Training Stack per Character:**
   - **Stage 1 (CPT — Full Raw Corpus Immersion):** LoRA causal-LM training on all cleaned primary-source writings/transcripts (with 10% held-out validation text for cross-corpus perplexity proof).
   - **Stage 2 (SFT — Interactive Voice & Dialogue):** Conversational/epistolary fine-tuning so the model stays sharp in back-and-forth negotiation turns without losing instruction-following.
   - **Stage 3 (DPO — Decision Policy Override):** Preference tuning contrasting authentic in-character dilemma reasoning (`chosen`) against the base model's own generic polite RLHF reply (`rejected`), strictly excluding Split or Steal prompts to avoid leakage.
4. **Configurable Engine & Replayable Storage:** Support varying turn counts per round (e.g. 2 vs. 4 turns) and persist every turn, hidden `<thought>`, decision, payout, and metadata to structured JSONL + SQLite (`results/simulations.db`) with a matching Postgres/Supabase SQL schema (`sim/supabase_schema.sql`) ready for cloud sync.

### Consequences & Interview Framing
- **De-risking first:** ~4–8 CUs validates the hardest ML assumption before larger compute or 3D web work.
- **Extensibility:** Plug-and-play character configs (`characters/<id>.yaml`) turn the repo into a general-purpose persona simulation lab rather than a one-off script.

---

## ADR-006: Iteration 2 Post-Training Calibration — EOS Preservation, Unconditional Weight Baking & Calibrated CPT/DPO LRs

- **Date:** 2026-11-25
- **Status:** Accepted

### Context (Empirical Diagnosis from Iteration 1 Verification)
Iteration 1 (`results/verification_v1/`) proved strong cross-corpus specialization in Proof 2 (unseen Machiavelli perplexity dropped **-33.1%** `18.90 -> 12.65`; unseen Gandhi perplexity dropped **-25.1%** `17.87 -> 13.39`), **but** Proofs 3 & 4 caught three classic post-training failure modes before we ran a single game:
1. **Truncated-EOS Repetition Loops (Proof 4):** Aggressive `CPT` (`lr=1e-4` on raw book blocks without `<|im_end|>`) combined with SFT sequence truncation (`full_ids[:512]` slicing off the trailing `<|im_end|>` token) caused greedy generations to loop instead of stopping cleanly.
2. **System-Prompt Conditioning Leak (Proof 3):** Including the character `system_prompt` during SFT and DPO trained the adapter to wait for the persona prompt, so Zero-Prompt Logit Probes showed almost zero shift (`Δ P(Defect) = -0.029` / `+0.002`).
3. **Under-Scaled DPO Signal:** `DPO` at `lr=5e-6` moved global LoRA Frobenius norm by only `0.016` (`7.911 -> 7.927`, ~0.2% of adapter norm) on top of the CPT+SFT adapter.

### Decision
Apply four targeted fixes for **Iteration 2 (`v2`)** and re-verify on the exact same 4-stage proof suite:
1. **Preserve EOS (`<|im_end|>`) on SFT & DPO:** Truncate assistant bodies *before* `<|im_end|>` so every supervised example trains clean turn termination; explicitly append `tokenizer.eos_token` to DPO `chosen`/`rejected` targets.
2. **Unconditional Persona Baking (`--drop-system-prompt`):** Strip character system prompts during SFT and DPO so the persona worldview and decision policy live *inside the LoRA weights themselves*, even with zero system prompt.
3. **Gentle CPT + Calibrated SFT/DPO LRs:** Reduce CPT LR from `1e-4 -> 2e-5` (30 steps) to absorb primary-source prose without eroding chat stopping; run SFT at `6e-5` (40 steps) and DPO at `3e-5` (45 steps, `beta=0.1`) so preference optimization exerts meaningful weight-level policy shift.
4. **Preserve Iteration 1 Artifacts (`results/verification_v1/`):** Keep both `v1` and `v2` reports in git as a concrete before/after post-training debugging case study.

### Consequences & Interview Framing
- Turns a potential silent failure into a textbook post-training story: *how automated perplexity + zero-prompt logit probes + qualitative checks caught EOS truncation and system-prompt conditioning leakage in <1 hour for ~1.5 CUs, and how v2 fixed them.*

---

## ADR-007: Sprint 2A — Fair Primary-Source DPO, Unprompted Arena Control & `Qwen/Qwen3-14B` Scale-Up Test

- **Date:** 2026-11-26
- **Status:** Accepted

### Context
Two methodological questions emerged directly from Sprint 1 (`Qwen/Qwen3-8B`):
1. **Is synthetic DPO "overtraining" or caricaturing a persona?** In Sprint 1, our 60 DPO `chosen` responses were handcrafted and trained for 5.6 epochs (`loss -> 0.001`), which risks memorizing our stereotype of a figure and contributed to repetitive phrasing in 4-turn negotiation.
2. **Do the weights alone beat prompting, and does a bigger base model (`Qwen/Qwen3-14B`) produce sharper strategy and wit?** Before training our full 7-character cast (keeping **Vincent van Gogh** alongside **Machiavelli, Gandhi, Buffett, Ali, Oscar Wilde, Richard Nixon**), we want to verify both fairness and model scale on our Machiavelli vs. Gandhi benchmark pair.

### Decision
1. **100% Verbatim Primary-Source DPO (`dpo_primary_train.jsonl` + `dpo_primary_val.jsonl`):** Automatically extract 120 train + 30 held-out validation DPO pairs where every `chosen` response is the author's **literal historical prose** from their own books/letters/speeches, paired against the unmodified base model's own generic assistant reply (`rejected`). Cap DPO at **2 epochs** with held-out validation eval to eliminate overtraining AND remove manual DPO authoring for all future characters.
2. **Unprompted Weights-Only Arena Control (`--include-unprompted`):** Run `cpt_sft_unprompted` and `cpt_sft_dpo_unprompted` in live Split or Steal games with **blank persona prompts and anonymous seat labels (`Player A` / `Player B`)** directly against `prompt_only` (Untrained Base + detailed persona prompt).
3. **Anti-Repetition Decoding:** Add `no_repeat_ngram_size=5` + `repetition_penalty=1.12` to eliminate cross-turn echo contagion in 4-turn negotiations.
4. **Upgrade Benchmark to `Qwen/Qwen3-14B`:** Test `Qwen/Qwen3-14B` (4-bit NF4 + bf16 LoRA) head-to-head against Sprint 1's `Qwen3-8B` before locking the base model for the full 7-character Sprint 2B cast.

### Consequences & Interview Framing
- Directly addresses the toughest ML skepticism ("did you just overfit synthetic caricature prompts?") with verbatim primary-source DPO, held-out DPO eval metrics, and anonymous unprompted arena games.

---

## ADR-008: Empirical Verdict on 100% Primary-Source DPO vs. Synthetic Dilemma DPO & Phase-Scoped Anti-Repetition

- **Date:** 2026-11-26
- **Status:** Accepted

### Context
Sprint 2A Experiments A & B (`results/sprint2a_unprompted_arena/`, `results/sprint2a_primary_dpo/`) tested two critical refinements on `Qwen/Qwen3-8B`:
1. **Decoding anti-repetition (`no_repeat_ngram_size`):** Applying `no_repeat_ngram_size=5` during structured XML decisions blocked `</thought>\n<decision>` (which appeared verbatim in the prompt), causing `Qwen3-8B` to emit Chinese `<决策>` tags and `78–81%` parse failures.
2. **DPO Fairness (`dpo_primary` vs. `dpo_synth`):** Comparing 100% verbatim primary-source DPO (`88` author passages vs. base assistant, `2.0` epochs, zero human-written dilemmas) head-to-head against Sprint 1's `60` hand-written dilemma DPO (`5.6` epochs) revealed a clear fidelity-vs-steering tradeoff:
   - `dpo_primary` achieved the **best unseen book perplexity** of all 7 models (`15.42` Machiavelli, `14.54` Gandhi) and shifted Machiavelli's zero-prompt `P(Defect)` organically (`47.09% -> 50.71%`, `+3.63 pp`), with `100%` held-out unseen DPO accuracy.
   - `dpo_synth` pushed zero-prompt `P(Defect)` `4×` harder (`60.69%`, `+13.60 pp`), *but* measurably drifted unseen book perplexity worse than SFT (`15.62 -> 15.85` and `14.62 -> 14.75`).

### Decision
1. **Phase-Scoped Anti-Repetition Only:** Apply `no_repeat_ngram_size=12` + `repetition_penalty=1.08` strictly during `phase == "negotiation"`, and disable both during `phase == "decision"` (verified `0.0% / 2.8%` parse failures and `6/6` echo-free Gandhi SPLIT in 4-turn games).
2. **Adopt 100% Verbatim Primary-Source DPO (`2.0` epochs, editorial/3rd-person filtered) as the Default Scientifically Fair Pipeline** for scaling to the 7-character cast (`Machiavelli, Gandhi, Van Gogh, Oscar Wilde, Warren Buffett, Muhammad Ali, Richard Nixon`), keeping `dpo_synth` strictly as an optional "behavioral amplifier" ablation — and test whether **`Qwen/Qwen3-14B`** extracts a stronger organic dilemma shift purely from `CPT + SFT + 100% Primary-Source DPO` once Colab `L4`/`A100` capacity opens up.

### Consequences & Interview Framing
- Eliminates manual dilemma authoring across 7+ characters while guaranteeing every single training token comes from verified historical primary sources (zero stereotype leakage).
- Demonstrates empirical discipline: catching and quantifying a `+0.23 PPL` overtraining drift in synthetic DPO rather than blindly optimizing a headline metric.



