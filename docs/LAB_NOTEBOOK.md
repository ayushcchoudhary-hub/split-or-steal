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
