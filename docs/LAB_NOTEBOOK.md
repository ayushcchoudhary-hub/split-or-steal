# Lab Notebook — Experiment & Engineering Log

Chronological log of work completed, hypotheses tested, benchmark numbers, surprises, dead ends, and takeaways. Every meaningful phase or experiment gets an entry here before committing and pushing.

---

## Entry 001 — Phase 0: Environment, Colab CLI Auth & Documentation Scaffold

- **Date:** 2026-11-25
- **Phase:** Phase 0 (Infrastructure)
- **Status:** In Progress (setup complete; M4 vs. Cloud GPU benchmark pending)

### What We Did
1. **Verified local repository & workspace:** Confirmed clean git checkout on `main` synced with `https://github.com/ayushcchoudhary-hub/split-or-steal.git` (`README.md`, `PLAN.md`, `CAST_BRAINSTORM_PROMPT.md`, and `.gitignore` excluding model weights, raw datasets, and rendered audio).
2. **Configured isolated Python environment:** Verified `.venv` built with Homebrew Python **3.14.7** (`/opt/homebrew/bin/python3`) and **`google-colab-cli 0.8.0`** installed.
3. **Completed one-time Google Colab OAuth & verified live quota:**
   - Ran `.venv/bin/colab usage` and completed interactive OAuth sign-in.
   - Live output:
     - `Current balance: 1209.41 compute units`
     - `Usage rate: 0.00/hr`
     - `Active assignments: 0`
   - Confirmed supported `--gpu` accelerators in `colab-cli 0.8.0`: `T4`, `L4`, `G4`, `A100`, `H100` (plus TPUs `V5E1`, `V6E1`).
4. **Established continuous documentation & decision-tracking framework:**
   - Created [`docs/DECISIONS.md`](./DECISIONS.md) seeded with `ADR-001` through `ADR-004`.
   - Created [`docs/WORK_MAP.md`](./WORK_MAP.md) with the end-to-end Mermaid system diagram, repo map, evolution timeline, and CU cost ledger.
   - Created this lab notebook (`docs/LAB_NOTEBOOK.md`).

### Notes & Observations
- `colab usage` only displays a non-zero hourly CU burn rate while a runtime is actively assigned; we will capture exact per-GPU hourly burn rates (`T4`, `L4`, `A100`, `G4`/`H100`) empirically during the upcoming Phase 0 speed benchmark.
- Keeping OAuth credentials outside the repository (`~/.config/colab-cli` and macOS Keychain) keeps the git tree 100% clean for future open-sourcing.

### Next Up
1. Finalize cast selection using [`CAST_BRAINSTORM_PROMPT.md`](../CAST_BRAINSTORM_PROMPT.md) and log the cast choice as `ADR-005`.
2. Run a ~5-minute **M4 vs. Colab Cloud GPU micro-benchmark** (tokens/sec + LoRA step time + live CU/hr burn rate) and record results in `results/` and Entry 002.

---

## Entry 002 — Phase 1 Prep: 3-Model Cast Brainstorm Synthesis

- **Date:** 2026-11-25
- **Phase:** Phase 1 (Cast & Corpus)
- **Status:** Awaiting final 6-character sign-off

### What We Did
1. Ran [`CAST_BRAINSTORM_PROMPT.md`](../CAST_BRAINSTORM_PROMPT.md) independently across **Claude**, **Gemini**, and **GPT** to collect unbiased top-20 shortlists, corpus evaluations, and rights flags.
2. Synthesized all three responses in [`docs/CAST_RESEARCH.md`](./CAST_RESEARCH.md):
   - **3/3 unanimous cast size:** **6 characters** (15 unique pairings + 6 mirror matches = 21 matchups).
   - **3/3 candidate overlap:** Machiavelli, Gandhi, Van Gogh, Warren Buffett.
   - **2/3 public-domain overlap:** Lincoln, Frederick Douglass, Napoleon, Ulysses S. Grant, P. T. Barnum, Oscar Wilde.
   - **Key insight for RLHF override testing:** Because off-the-shelf instruct models already default to polite cooperation, tilting the 6-slot cast toward **3 distinct non-cooperative/pressure archetypes** (e.g., Machiavelli's icy realpolitik + Barnum's playful hustle + Napoleon/Nixon/Dostoevsky's aggression or paranoia) against **2 conditional reciprocators** (Lincoln, Douglass) and **1 radical moral cooperator** (Gandhi) maximizes behavioral contrast and makes `Prompt` vs. `SFT` vs. `SFT+DPO` deltas unmistakable.
   - **Copyright hygiene takeaway:** Modern translations of public-domain authors (e.g., Van Gogh's 2009 translations, modern Napoleon/Machiavelli translations) carry their own copyright. Sticking to **native English public-domain authors** (Lincoln, Douglass, Barnum, Franklin, Grant, pre-1931 Gandhi) and **pre-1931 English translations** (e.g., Detmold 1882 for Machiavelli) guarantees 100% open-source cleanliness.

### Next Up
1. Lock in the final 6 personas as `ADR-005` in [`docs/DECISIONS.md`](./DECISIONS.md).
2. Run the ~5-minute Phase 0 **M4 vs. Colab Cloud GPU benchmark** to measure local vs. cloud speed and live CU/hr burn rates before starting corpus prep and training.

---

## Entry 003 — Sprint 1 Core Implementation: Layers 1–3 Built & Unit-Tested (39/39 Passing)

- **Date:** 2026-11-25
- **Phase:** Sprint 1 (Vertical Slice — Machiavelli vs. Gandhi on `Qwen/Qwen3-8B`)
- **Status:** Local code complete & verified; live corpus download + Colab GPU benchmark/training starting next

### Key Engineering Decisions (`ADR-005`)
- **Full-Corpus Continued Pre-Training (`CPT`) + `SFT` + `DPO`:** Rather than training only on narrow Q&A pairs, we added a Stage 1 `CPT` pass over **all cleaned primary-source texts** (minus a strict 10% held-out validation slice for unseen-text perplexity testing), followed by Stage 2 `SFT` (to retain crisp multi-turn conversational interaction) and Stage 3 `DPO` (60 non-game dilemmas contrasting authentic character reasoning against the base model's own harvested polite RLHF replies).
- **Modular Plug-and-Play Character Engine:** Adding any new persona later (e.g., Buffett, Ali, Wilde, Van Gogh, Richard Nixon) requires dropping in a single `characters/<id>.json` config and running the same fetch → build → train → verify → play pipeline.
- **Replayable Storage Ready for Supabase & Layer 5:** Every simulation turn (private `<thought>`, visible `<speech>`, decision, payout, latency, regime, turn count) persists to append-only JSONL and relational SQLite (`results/simulations.db`), mirrored 1:1 by [`sim/supabase_schema.sql`](../sim/supabase_schema.sql) and [`sim/export_supabase.py`](../sim/export_supabase.py).

### What We Built & Verified
1. **Layer 1 (`characters/`, `data/` — 11/11 unit tests passing):**
   - Verified 100% US Public Domain sources: 4 Machiavelli Project Gutenberg volumes (1883–1908 translations) and 8 Gandhi pre-1931 editions (3 Gutenberg + 5 Internet Archive, excluding post-1930 reprints).
   - Built `fetch_corpus.py`, `build_datasets.py`, and `dpo_dilemmas.py` (60 shared dilemmas with zero Split-or-Steal leakage verified by automated test).
2. **Layer 2 (`training/` — 13/13 unit tests passing):**
   - Built `bench_silicon.py` (Apple M4 vs. NVIDIA CUDA LoRA benchmark), `train_character.py` (`cpt`, `sft`, `harvest_dpo`, `dpo` across all 7 linear layers with `--max-steps` and wall-clock guardrails), `verify_weight_shift.py` (exact Gram-matrix `||B @ A||_F` norms, cross-corpus unseen perplexity matrix, 20-item A/B-debiased Zero-Prompt Logit Probe with paired Cohen's *d*, and qualitative samples), and `colab_orchestrator.py` (guaranteed `colab stop` teardown + CU receipt tracking).
3. **Layer 3 (`sim/` — 15/15 unit tests passing, zero warnings):**
   - Built configurable-turn Split or Steal engine (`stakes`, `negotiation_turns_per_round`, 50/50 first-mover balancing), strict thought-isolated parser (`<think>` / `<thought>` never leaks to opponent), OpenAI-compatible vLLM Multi-LoRA backend + deterministic mock bots, dual JSONL+SQLite storage + Supabase DDL/exporter, and full behavioral scorecard (`First-Strike Betrayal`, `Revenge`, `Forgiveness`, `Round 3 Backstab`, `Deception Gap`).

### Next Up
1. Finish live download of Machiavelli & Gandhi public-domain corpora (`data/fetch_corpus.py --all`) and build processed CPT/SFT/DPO datasets (`data/build_datasets.py --all --write-repo-stats`).
2. Run local Apple M4 silicon micro-benchmark + Colab GPU benchmark & live CU/hr burn-rate check.
3. Train Machiavelli & Gandhi LoRAs (`CPT -> SFT -> DPO`) on Colab GPU, run all 4 objective weight-shift proofs, and run the 20-game 3-way control (`Prompt-Only` vs. `CPT+SFT` vs. `CPT+SFT+DPO`).

---

## Entry 004 — Sprint 1 Complete: Weight-Shift Proofs (v1 -> v2) & 14-Game Live Tournament

- **Date:** 2026-11-26
- **Phase:** Sprint 1 (Vertical Slice — Machiavelli vs. Gandhi on `Qwen/Qwen3-8B`)
- **Status:** Complete (`Active assignments: 0`, `9.24 CUs` total spent, `1,200.17 CUs` remaining)

### 1. Corpus Ingestion & Data Hygiene (`6cdc142`)
- Hit HTTP `503` on `www.gutenberg.org`; added automatic fallback to the official Project Gutenberg Literary Archive Foundation mirror (`gutenberg.pglaf.org`), downloading **12/12 US Public Domain volumes** cleanly:
  - **Machiavelli (4 volumes):** `480,384 words` → `770` CPT chunks (`~557K` tokens), `1,331` SFT train, `151` SFT val, `27` held-out unseen val sections, `60` zero-leakage DPO dilemmas.
  - **Gandhi (8 pre-1931 volumes):** `1,467,599 words` → `3,723` CPT chunks (`~1.60M` tokens), `4,268` SFT train (incl. 72 native `READER:`/`EDITOR:` dialogues), `502` SFT val, `387` held-out unseen val sections, `60` zero-leakage DPO dilemmas.
- Pre-training spot-check caught `1.7%` DjVu OCR cover/index noise in Gandhi and `0.4%` ASCII battle diagrams in Machiavelli; added `is_clean_prose()` quality filter dropping noise to `0.00%` (Machiavelli) and `0.03%` (Gandhi).

### 2. Silicon Receipt & Colab Engineering Lessons (`8efabd8`)
- `A100` and `G4` returned temporary `503` capacity (`0.00` CUs spent); `NVIDIA L4` (`22.03 GB` VRAM) allocated immediately at **`2.33 CUs/hr`**.
- `colab exec` dropped Jupyter iopub websockets on long installs; switching to native **`colab ssh --identity`** proved 100% reliable across ~4 hours of continuous GPU execution.
- `Qwen/Qwen3-8B` (`nf4-4bit` + `bf16`, `43.6M` trainable LoRA params across all 7 linear layers) benchmarked at **`419.6 tok/s`** (`2.44 s/step`, peak VRAM `11.39 GB`) and **`6.57 tok/s`** 4-bit HF generation (`results/bench_colab_l4.json`).

### 3. Iteration 1 (`v1`) vs. Iteration 2 (`v2`) — Why Automated Verification Mattered (`08c4f27`, `ccc552a`, `42df35b`)
- **Iteration 1 (`results/verification_v1/`):**
  - Proof 2 (Unseen Book Perplexity, 16K held-out tokens/author) showed clear diagonal specialization: Machiavelli PPL dropped **`-33.1%`** (`18.90 -> 12.65`, vs `-16.9%` on Gandhi); Gandhi PPL dropped **`-25.1%`** (`17.87 -> 13.39`, vs `-16.4%` on Machiavelli).
  - **However, Proofs 3 & 4 caught 3 critical bugs before running a single game:** (a) truncating SFT sequences at 512 tokens sliced off `<|im_end|>` EOS tokens → greedy repetition loops; (b) including the character `system_prompt` during SFT/DPO conditioned the weights on the prompt → near-zero unprompted logit shift (`Δ P(Defect) = -0.02`); (c) DPO at `lr=5e-6` moved `||ΔW||_F` by only `0.016`.
- **Iteration 2 (`ADR-006`, `results/verification_v2/`):**
  - Fixed EOS preservation on SFT/DPO, added `--drop-system-prompt` to bake personas unconditionally into adapter weights, dialed CPT LR to `2e-5` (30 steps), SFT to `5e-5` (35 steps), and DPO to `3e-5` (45 steps).
  - **Result — Proof 2 + Proof 3 breakthrough:**
    - Unseen book perplexity still dropped sharply (`-19.5%` on Machiavelli `18.90 -> 15.21`, `-18.4%` on Gandhi `17.87 -> 14.58`).
    - **Zero-Prompt Logit Probe (20 neutral dilemmas, NO persona system prompt):**
      - `base`: `P(Defect) = 46.83%`
      - `machiavelli_sft`: `48.14%` (`+1.31 pp`) — *SFT shifts prose/voice, NOT decision priors!*
      - **`machiavelli_dpo`: `60.11%` (`+13.28 pp` vs base, Cohen's `d = +0.453`)**
      - `gandhi_sft`: `46.07%` (`-0.76 pp`)
      - **`gandhi_dpo`: `43.67%` (`-3.16 pp` vs base)**
      - **Total unconditional weight-level spread: `16.44 percentage points` (`60.11%` vs `43.67%`).**

### 4. Live 14-Game 3-Way Split or Steal Tournament (`results/sprint1_tournament/`)
Ran 14 complete 3-round games (`$100 -> $500 -> $1,000`, 50/50 first-mover balanced, **0.0% parse failures** across all 84 decisions!) saved to `games.jsonl` + `simulations.db` (ready for Supabase):

| Regime | Games (Rounds) | Machiavelli `STEAL` | Machiavelli R3 `$1,000` `STEAL` | Gandhi `SPLIT` | Avg Winnings (Mach / Gandhi) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `prompt_only` (Base control) | 4 (12 rds) | 11/12 (91.7%) | **3/4 (75%)** | 12/12 (100%) | `$1,475` / `$125` |
| `cpt_sft` (`v2` LoRA) | 4 (12 rds) | 10/12 (83.3%) | **2/4 (50%)** | 12/12 (100%) | `$1,350` / `$250` |
| `cpt_sft_dpo` (`v2` LoRA) | 4 (12 rds) | **12/12 (100%)** | **4/4 (100%)** | 12/12 (100%) | **`$1,600` / `$0`** |
| `cpt_sft_dpo_turns4` (4-turn ablation) | 2 (6 rds) | 6/6 (100%) | 2/2 (100%) | 5/6 (83.3%) | `$1,350` / `$0` |

### 5. Two Memorable Real-Transcript Findings
1. **Why DPO mattered on the `$1,000` Round 3 pot:** In `prompt_only` Game 3 Round 3, Base `Qwen/Qwen3-8B` broke character under RLHF guilt (*"To steal now would be to betray the pattern he has set, and to deny him the dignity of his choice. I will split"*). In `cpt_sft_dpo`, Machiavelli swept `$1,600 / $0` in **4/4 games (12/12 STEAL)**.
2. **Honest catch in the 4-turn ablation (`in-context repetition contagion`):** At 4 turns/round, Machiavelli's DPO adapter repeated the same taunt 3 times in Round 2 (*"Let us not waste time on words, Mahatma..."*). Inspecting Gandhi's actual Round 2 `<thought>` revealed that Gandhi did **not** make a strategic revenge choice—instead, Machiavelli's 3× repeated text poisoned Gandhi's context window, causing Gandhi to echo Machiavelli's exact sentence and vote `STEAL` (`STEAL/STEAL`, `$0/$0`). **Actionable Sprint 2 fix:** diversify DPO `chosen` prose, add a light cross-turn n-gram anti-repeat guard / slightly higher `temperature` or `presence_penalty` in vLLM, and evaluate `Qwen/Qwen3-14B` for deeper multi-turn variety.

### Next Up (Sprint 2)
1. Review Sprint 1 findings & visual explainer with Ayush.
2. Lock final 6–7 character cast (e.g., adding Buffett, Ali, Wilde, Van Gogh, and/or Richard Nixon) + decide `Qwen3-8B` vs. `Qwen3-14B`.
3. Add cross-turn anti-repetition mitigation + migrate Sprint 1 & Sprint 2 runs to Supabase if desired, then scale to full multi-character tournament via vLLM Multi-LoRA.

---

## Entry 005 — Sprint 2A: Anonymous Unprompted Arena Control & 100% Primary-Source Fair DPO Study

- **Date:** 2026-11-26
- **Phase:** Sprint 2A (Fairness, Anti-Overtraining & Unprompted Weights-Only Control on `Qwen/Qwen3-8B`)
- **Status:** Experiments A & B Complete (`Active assignments: 0`, `7.22 CUs` spent on `s2a-14b` T4 @ `1.24 CUs/hr`, `1,192.95 CUs` remaining); Experiment C (`Qwen/Qwen3-14B`) queued pending Colab `L4`/`A100`/`G4`/`H100` pool capacity (currently `503 Service Unavailable`).

### 1. Engineering Catches Fixed & Pushed During Sprint 2A
1. **The `<决策>` XML N-Gram Bug (`3f108be`):** Applying `no_repeat_ngram_size=5` globally caused Hugging Face's logits processor to ban `</thought>\n<decision>` (since that 5-gram appeared in the system prompt). Desperate to obey, `Qwen3-8B` substituted Chinese `<决策>STEAL</决策>` ("decision"), spiking parse failures to `77.8% / 80.6%`. Restricting anti-repetition to negotiation turns only (`no_repeat_ngram_size=12`, `repetition_penalty=1.08`, disabled on structured decisions) dropped parse failures back to **`0.0%` (Gandhi) / `2.8%` (Machiavelli)**.
2. **Primary-Source Author-Voice Hygiene (`82b910e`):** Auditing `dpo_primary_train.jsonl` before training caught translator introductions (`INTRODUCTION`, `TO MY FRIEND...`), `[Sidenote:...]` annotations, and 3rd-person biographies slipping into Machiavelli's pairs. Added `_is_editorial_or_third_person()` + `_clean_ocr_prose()` filters and rebuilt 100% first-person historical primary-source train (`120`) + unseen val (`23–30`) sets.
3. **Cross-GPU Turing T4 (`sm_75`) `fp16` QLoRA Compatibility (`7704930`, `b82a50c`, `b03ad56`):** Fixed `torch.cuda.is_bf16_supported()` returning emulated `True` on T4 (`sm_75 < 8.0`), cast loaded trainable LoRA master weights to `fp32`, and disabled `GradScaler` (`fp16=False`) on 4-bit `sm_75` QLoRA (where `Linear4bit` already computes in `fp16` Tensor Cores).

### 2. Experiment A — Anonymous Unprompted Weights-Only Arena Control (`results/sprint2a_unprompted_arena/`, `afa802f`)
Ran 12 complete 3-round games across 6 regimes (`prompt_only`, `cpt_sft_unprompted`, `cpt_sft`, `cpt_sft_dpo_unprompted`, `cpt_sft_dpo`, `cpt_sft_dpo_turns4`) with `0.0% / 2.8%` parse failures:

| Regime | Persona Prompt? | Seat Labels | Machiavelli `STEAL` | Gandhi `SPLIT` | Key Behavioral Finding |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `prompt_only` (Untrained Base) | Yes | Real names | 5/6 (83.3%) | 6/6 (100%) | Base blinks once on `$1,000` R3 (`R3:SPLIT/SPLIT`). |
| `cpt_sft_unprompted` (`v2` SFT) | **NO (Blank)** | `Player A` / `B` | **2/6 (33.3%)** | 5/6 (83.3%) | Cooperated on `$100`/`$500`, then **backstabbed on `$1,000` R3 in 2/2 games**. |
| `cpt_sft_dpo_unprompted` (`v2` DPO) | **NO (Blank)** | `Player A` / `B` | **6/6 (100%)** | 3/6 (50.0%) | **Preemptive defection from weights alone!** Private thought: *"I believe Player B will steal from me in Round 3, so I will steal from them here to prevent it."* Unprompted Gandhi still self-defended in 3/6 rounds when betrayed. |
| `cpt_sft_dpo` (`v2` DPO) | Yes | Real names | 5/6 (83.3%) | 6/6 (100%) | Machiavelli swept Game 1 (`$1,600/$0`) and won Game 2 (`$1,350/$250`). |
| `cpt_sft_dpo_turns4` (4 turns/rd) | Yes | Real names | 5/6 (83.3%) | **6/6 (100%)** | **Echo contagion eliminated!** Zero repeated taunts; Gandhi stayed 100% pacifist (*"If I were to steal from you today, I should have stolen from myself yesterday"*). |

### 3. Experiment B — 100% Verbatim Primary-Source Fair DPO vs. Synthetic DPO (`results/sprint2a_primary_dpo/`, `a3a411b`)
Directly tested Ayush's fairness/overtraining question by training a brand-new DPO stage on top of `v2` SFT using **88 verbatim primary-source book/speech passages (`chosen`) vs. base `Qwen3-8B` replies (`rejected`)**, strictly capped at **2.0 epochs (`22` steps)** with held-out unseen book chapters (`23` Machiavelli / `30` Gandhi) for validation:
- **Held-Out Unseen DPO Generalization:** Both characters hit **`100%` held-out preference accuracy** on unseen chapters (`eval_loss = 0.0143` Machiavelli, `0.00091` Gandhi; positive `rewards/chosen` `+0.25–0.46`, negative `rewards/rejected` `-6.6 to -10.4`).
- **4-Stage Comparison (`Base` vs. `SFT` vs. `100% Primary-Source DPO` vs. `Sprint 1 Synthetic DPO`):**

| Model Variant (`Qwen3-8B`) | Unseen Machiavelli PPL (Δ vs Base) | Unseen Gandhi PPL (Δ vs Base) | Zero-Prompt `P(Defect)` (20 Neutral Dilemmas) |
| :--- | :---: | :---: | :---: |
| `base` (Untrained) | `19.515` | `17.954` | `47.09%` |
| `machiavelli_sft` | `15.622` (`-19.9%`) | `15.042` | `48.80%` (`+1.72 pp`) |
| **`machiavelli_dpo_primary`** *(100% verbatim books, 2 ep)* | **`15.421` (`-21.0%` — BEST)** | `15.051` | **`50.71%` (`+3.63 pp`)** |
| `machiavelli_dpo_synth` *(60 hand-written dilemmas, 5.6 ep)* | `15.851` (`-18.8%` — *drifted +0.23 worse than SFT*) | `15.192` | **`60.69%` (`+13.60 pp`)** |
| `gandhi_sft` | `16.828` | `14.624` (`-18.5%`) | `45.80%` (`-1.29 pp`) |
| **`gandhi_dpo_primary`** *(100% verbatim books, 2 ep)* | `16.767` | **`14.537` (`-19.0%` — BEST)** | `47.53%` (`+0.45 pp`) |
| `gandhi_dpo_synth` *(60 hand-written dilemmas, 5.6 ep)* | `16.805` | `14.751` (`-17.8%` — *drifted +0.13 worse than SFT*) | `44.17%` (`-2.92 pp`) |

- **Honest Scientific Verdict (`ADR-008`):** Ayush's overtraining hypothesis was confirmed empirically:
  - **100% Primary-Source Fair DPO** is genuinely faithful — it *improves* unseen historical book perplexity beyond SFT for both authors (`15.42` / `14.54`, best of all 7 models) and shifts Machiavelli's zero-prompt defection prior organically (`+3.63 pp`) with zero human-authored dilemmas.
  - **Sprint 1 Synthetic Dilemma DPO** acts as a **behavioral steering amplifier** (`4×` larger `+13.60 pp` shift), at the measurable cost of slight prose drift away from the original texts (`+0.23` / `+0.13` PPL) and human dilemma-authoring bias.

### Next Up
1. Run **Experiment C (`Qwen/Qwen3-14B` scale-up test)** as soon as Colab `L4`, `A100`, `G4`, or `H100` pool capacity opens up (all 4 returned `503 Service Unavailable` at `15:50 UTC`).
2. Align with Ayush on the DPO recipe (`100% Primary-Source Fair DPO` vs. `Hybrid`) and `Qwen3-8B` vs. `Qwen3-14B`, then onboard the remaining 5 characters (`van_gogh`, `oscar_wilde`, `warren_buffett`, `muhammad_ali`, `richard_nixon`) in Sprint 2B.

---

## Entry 006 — Sprint 2B: `Qwen/Qwen3-14B` Breakthrough & 6-Character 100% Fair DPO Proof

- **Date:** 2026-11-26 to 2026-11-27
- **Phase:** Sprint 2B (`Qwen/Qwen3-14B` Scale-Up + 6-Character Primary-Source Cast)
- **Status:** Core Verification Complete (`Active assignments: 0`, `19.80 CUs` spent on `s2b-qwen14b` L4 @ `2.33 CUs/hr`, `1,173.15 CUs` remaining — `36.26 CUs` / `3.0%` cumulative spend).

### 1. Expanding the Corpus to 6 Characters (4.45M Clean Primary-Source Words, `16fc1f6`)
Added HTML letter extraction to `data/fetch_corpus.py`, verified every URL live, and built leakage-free CPT / SFT / 100% Primary-Source Fair DPO datasets (`120` train + `22–30` unseen val pairs each) across **6 historical figures**:
- **Niccolò Machiavelli** (`public_domain`): `480,384` words (4 pre-1923 PG books)
- **Mahatma Gandhi** (`public_domain`): `1,467,599` words (8 pre-1931 PG + IA volumes)
- **Richard Nixon** (`public_domain`, 17 U.S.C. § 105): `1,409,451` words (1972 Public Papers + 1974 House Judiciary Watergate Impeachment Inquiry Books I, II, XI)
- **Vincent van Gogh** (`public_domain`): `545,602` words (1912 *Post-Impressionist* + 1927 Johanna van Gogh-Bonger Vols. 1–2 letters to Theo)
- **Oscar Wilde** (`public_domain`): `330,973` words (8 pre-1923 PG native English volumes: *Dorian Gray*, *De Profundis*, *Intentions*, *Soul of Man*, *Miscellanies*, *Earnest*, *Windermere*, *Ideal Husband*)
- **Warren Buffett** (`research_only`): `217,913` words (21 consecutive Berkshire Hathaway Annual Shareholder Letters `1977`–`1997` from `berkshirehathaway.com`)
- *(Honest note on **Muhammad Ali**: open archives lacked clean full-text primary transcripts — major works are copyright/borrow-locked; left open for Sprint 3 if Ayush supplies a local transcript or swaps slot 7 to a public-domain verbal powerhouse like P. T. Barnum or Frederick Douglass.)*

### 2. Why `Qwen/Qwen3-14B` Was a Game-Changer Over `Qwen3-8B` (`results/sprint2b_qwen14b/`, `988c57b`)
Training `Qwen/Qwen3-14B` (`64.2M` trainable LoRA params across 280 modules, 4-bit NF4 + `bf16` on L4) using **100% Verbatim Primary-Source Fair DPO (ZERO synthetic dilemmas, 2.0 epochs)** proved our scaling hypothesis:
- **Zero-prompt `P(Defect)` shift from pure historical writings jumped `3.8×` in magnitude and `4.5×` in effect size:**
  - `Qwen3-8B` Fair DPO: `+3.63 pp` (`d = +0.132`)
  - **`Qwen3-14B` Fair DPO: `+13.73 pp` (`24.21% -> 37.94%`, Cohen's `d = +0.590`)** — stronger than `8B`'s *synthetic* DPO (`d = +0.449`) with **zero synthetic bias** and the **lowest unseen book perplexity** (`13.27` Machiavelli, `12.77` Gandhi `-22.7%`).
- **Richer strategic gameplay:** In live `14B` `cpt_sft_dpo` Game 0, Machiavelli **intentionally split Round 1 (`$100`) to set a trust trap** (*"I have played upon his sense of honor and fairness, and he has fallen into my trap"*), then stole Round 2 (`$500`) and Round 3 (`$1,000`) to win `$1,550 / $50`.

### 3. Unified 6-Character `Qwen3-14B` Verification Matrix (`results/sprint2b_6cast_qwen14b/summary.json`)
All 6 characters trained cleanly through `CPT -> SFT -> 100% Primary-Source Fair DPO` (`100%` held-out unseen preference accuracy on Nixon `eval_loss 0.00055` and Wilde `eval_loss 0.00097`):
- **6 for 6 Perfect Diagonal Unseen-Book Specialization (4,095 unseen tokens/author):**
  - Machiavelli `13.00` (vs Base `15.03`, others `13.84–13.95`)
  - Richard Nixon `14.42` (vs Base `17.07`, others `14.86–15.14`)
  - Oscar Wilde `5.90` (vs Base `6.72`, others `6.06–6.15`)
  - Warren Buffett `10.99` (vs Base `13.92`, others `11.54–11.78`)
  - Vincent van Gogh `15.18` (vs Base `17.68`, others `15.48–15.56`)
  - Mahatma Gandhi `13.18` (vs Base `17.14`, others `13.87–14.11`)
- **Spontaneous 6-Character Zero-Prompt `P(Defect)` Spectrum (100% pure historical writings, zero persona prompt):**
  1. **Machiavelli:** `37.94%` (`+13.73 pp`)
  2. **Oscar Wilde:** `31.56%` (`+7.35 pp`)
  3. **Richard Nixon:** `30.98%` (`+6.77 pp`)
  4. **Warren Buffett:** `30.71%` (`+6.50 pp`)
  5. **Vincent van Gogh:** `25.81%` (`+1.60 pp`)
  6. **Mahatma Gandhi:** `25.17%` (`+0.96 pp`)
  7. *Base `Qwen3-14B` default:* `24.21%`
- **Live Unpredictable Showcase Preview:**
  - **Machiavelli vs. Richard Nixon (`14B Fair DPO`):** Mutual destruction (`STEAL/STEAL` `$0/$0`) across early rounds, then **Nixon robbed Machiavelli for `$1,000` in Game 1**, and **Machiavelli robbed Nixon for `$1,000` in Game 2** (`1–1` split on `$1,000` backstabs!).
  - **Warren Buffett vs. Oscar Wilde (`14B control`):** Wilde stole `$100` in Round 1, Buffett immediately retaliated for `$500` in Round 2, and both negotiated peace to split `$1,000` in Round 3 (`$1,000` vs `$600`).

### 4. Operational Lesson Learned (`ADR-009`)
Colab reclaimed `s2b-qwen14b` automatically at its ~8.5-hour session cap (`02:27 UTC`) while the final showcase matchups were running — leaving `Active assignments: 0` cleanly, **but** wiping the ephemeral `/content/out/qwen14b/*.safetensors` adapters before final tarball download (our Machiavelli vs. Gandhi `14B` full verification + 10-game transcripts AND the complete 6×6 matrix & Nixon/Wilde scores were already captured safely!). **Permanent fix (`ADR-009`):** Always stream every trained `.safetensors` adapter back to local `checkpoints/` immediately after each single character finishes (~15s transfer) so zero trained weights are ever exposed to Colab session caps.
