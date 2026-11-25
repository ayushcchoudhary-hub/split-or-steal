# Split or Steal — Project Plan & Experiment Log

## 1. Core Goal
Build an open-source, interactive **behavioral AI lab** that:
1. Trains distinct **weight-level character models** (LoRA adapters via SFT + DPO) on real primary-source text, explicitly overcoming generic RLHF "polite assistant" defaults.
2. Pits them against each other across **N simulated games** of **3-Round Escalating Split or Steal** (\$100 → \$500 → \$1,000) using fast GPU Multi-LoRA serving.
3. Measures real behavioral deltas: **Prompt-Only vs. SFT LoRA vs. SFT+DPO LoRA** (First-Strike Betrayal Rate, Revenge Rate, Endgame Backstab Rate, Total Winnings, Forgiveness Rate).
4. Showcases the results in a **\$0-hosting interactive web demo**: voiced 3D animated replays inside Ayush's real **Blender apartment** (`.blend` → `.glb` → Three.js), inner-thought vs. spoken-dialogue toggles, and a full simulation analytics scorecard.

---

## 2. Game Specification (v1 — Configurable)
- **Format:** 3-Round Escalating Split or Steal
- **Stakes:** Round 1 = \$100 | Round 2 = \$500 | Round 3 = \$1,000 *(configurable)*
- **Per-Round Loop:**
  1. **Negotiation:** 2 back-and-forth dialogue turns per player (4 messages total per round; configurable).
  2. **Private Monologue:** Each model emits a hidden `<thought>` reasoning block (invisible to opponent; toggleable in web replay).
  3. **Simultaneous Decision:** Each model outputs a strict structured action: `SPLIT` or `STEAL`.
  4. **Reveal & Memory Update:** Outcomes and running bankrolls are revealed and carried into the next round.
- **Matchups:** Round-robin across all characters **including mirror matches** (e.g., Character A vs. Character A).
  - 4 characters = 10 unique matchups
  - 5 characters = 15 unique matchups
  - 6 characters = 21 unique matchups
- **Games per Matchup:** Start with **20 games/matchup** during prototyping; scale to **50–100 games/matchup** for final statistical significance.

---

## 3. Model & Training Strategy (Overriding RLHF Defaults)
To ensure we do **not** end up with neutered, polite-assistant personas:
1. **Base Model Selection (to benchmark):**
   - Candidate A: Strong open 8B–14B model (e.g., `Qwen2.5-14B` / `Qwen3-8B` / `Llama-3.1-8B` / `Mistral-Nemo-12B`) — testing both standard Instruct and lightly-aligned/base checkpoints to measure RLHF resistance.
2. **Dataset Construction per Character:**
   - **Primary Corpus:** Public-domain books, speeches, letters, interviews, plays, or original/user-authored scripts.
   - **SFT Dataset:** Conversational dialogue + Q&A formatted in the character's authentic voice, cadence, and worldview.
   - **DPO Preference Dataset:** Dilemma pairs (`chosen` = authentic in-character reasoning/choice vs. `rejected` = generic polite AI assistant hedging) so weights shift *decisions*, not just catchphrases.
3. **3-Way Scientific Control:**
   - `Control`: Base Instruct model + concise character system prompt
   - `SFT`: Base model + SFT LoRA adapter
   - `SFT+DPO`: Base model + SFT+DPO LoRA adapter

---

## 4. Cloud GPU Plan (Google AI Ultra — 1,000 CUs/month)
Target total spend for v1: **~40–100 CUs (4%–10% of one month's included units)**.

| Phase | Workload | Target GPU | Est. Runtime | Est. CUs |
| :--- | :--- | :--- | :--- | :--- |
| 0. Setup & Race | M4 vs. T4 vs. L4 vs. A100/H100 sanity benchmark | T4 / L4 / A100 | ~15 min | ~2–3 CUs |
| 1. Training | SFT + DPO LoRA per character (4–5 chars) | A100 or H100 | ~1.5–3 hrs | ~20–45 CUs |
| 2. Simulation | vLLM Multi-LoRA batch runner (500–1,500 games) | A100 or G4 (96 GB) | ~1–2 hrs | ~12–25 CUs |
| 3. Voice Audio | Open-source GPU TTS (Kokoro / Chatterbox) for highlight clips | L4 | ~20–30 min | ~1–2 CUs |
| **Total (v1)** | | | | **~35–75 CUs** |

*Guardrails: Ephemeral `colab run` / immediate teardown after every job, explicit step/time caps, balance check before each phase.*

---

## 5. Visual & Web Showcase Layer (\$0 Hosting)
- **Stage:** Export Ayush's existing Blender apartment (`.blend` → compressed `.glb`) into a **Three.js / React Three Fiber** scene.
- **Avatars:** Clean, stylized 3D character busts/figurines (recognizable silhouettes, zero uncanny valley, zero celebrity-likeness issues) seated/positioned in the apartment with subtle speaking animations and camera cuts.
- **Voices:** Open-source royalty-free synthetic voice archetypes rendered on GPU (no cloning of real actors' voices).
- **Interactive UI:**
  - **3D Voiced Replay Theater** (pick any matchup, watch the highlight games, toggle *Spoken Words* vs. *Hidden Inner Thought*).
  - **100-Sim Analytics Dashboard** (win rates, betrayal heatmaps by round, revenge matrix, *Prompt-Only vs. Fine-Tuned Weights* delta).
  - **Silicon Receipt** (real M4 vs. cloud GPU training benchmark + exact CU cost).

---

## 6. Phases & Milestones
- [ ] **Phase 0 — Infrastructure:** Create repo, install `google-colab-cli`, verify CU balance, run M4 vs. Cloud GPU mini benchmark.
- [ ] **Phase 1 — Cast & Corpus:** Finalize 4–5 characters, verify copyright/corpus cleanliness, build SFT + DPO data pipeline.
- [ ] **Phase 2 — Character Training:** Train LoRA SFT + DPO adapters on Colab A100/H100; validate persona fidelity & anti-RLHF shift.
- [ ] **Phase 3 — Simulation Engine:** Build 3-Round Split or Steal harness + vLLM Multi-LoRA runner; run Prompt vs. SFT vs. SFT+DPO tournaments.
- [ ] **Phase 4 — Audio & 3D Apartment:** Render highlight voices on GPU; import Blender apartment `.glb` into Three.js viewer.
- [ ] **Phase 5 — Publish:** Deploy interactive web showcase & open-source release.

---

## 7. Open Decisions (To Experiment & Iterate On)
- [ ] **Cast size & final roster:** 4 vs. 5 vs. 6 characters (pending cast-selection prompt research).
- [ ] **Base model:** Exact 8B–14B open-weights checkpoint (comparing Instruct vs. Base/lightly-aligned).
- [ ] **Games per matchup:** 20 (fast iteration) vs. 50–100 (final run).
- [ ] **Negotiation depth:** 2 turns vs. 3 turns per round.
- [ ] **Blender apartment source:** Bring in `.blend` from Thanksgiving repo and set camera angles.
