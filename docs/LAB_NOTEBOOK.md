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
