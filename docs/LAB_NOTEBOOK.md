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
