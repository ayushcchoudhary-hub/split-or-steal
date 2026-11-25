# Cast Selection Research & 3-Model Synthesis

To avoid confirmation bias when choosing personas for **Split or Steal — Persona Simulation Lab**, we ran [`CAST_BRAINSTORM_PROMPT.md`](../CAST_BRAINSTORM_PROMPT.md) independently across three frontier models (**Claude**, **Gemini**, and **GPT**) and synthesized their top-20 rankings, corpus audits, rights flags, and cast-size recommendations.

---

## 1. Unanimous Consensus Across All 3 Models

### Cast Size: **6 Characters** (3/3 Agreement)
- **Claude:** 6 characters (15 unique pairs + 6 mirror matches = 21 matchups; 30 directional pairs if alternating first-mover seat).
- **Gemini:** 6 characters (4 is too small to distinguish stable traits from rock-paper-scissors noise; 6 spans orthogonal archetypes while staying human-auditable).
- **GPT:** 6 characters (15 unique pairs, 30 role-swapped encounters).
- **Compute fit:** With vLLM Multi-LoRA batching on Colab `A100`/`G4`, 6 characters × 21 matchups (splitting first-mover seats 50/50 across $N=50$–$100$ games) fits comfortably within our ~40–80 CU v1 target (~3–7% of our `1,209.41 CU` balance).

### Candidate Overlap Matrix

| Candidate | Claude | Gemini | GPT | Overlap | Corpus Strength (English, First-Person) | Rights Status |
| :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| **Niccolò Machiavelli** | #1 | #2 | #4 | **3/3** | Very rich (*The Prince*, *Discourses*, *Legations* diplomatic dispatches, letters; Detmold 1882 PD English) | Public Domain |
| **Mahatma Gandhi** | #4 | #4 | #3 | **3/3** | Massive (*Hind Swaraj*, *Experiments with Truth*, *Satyagraha in South Africa*, pre-1931 letters/speeches in English) | Public Domain (use pre-1931 editions for US cleanliness) |
| **Vincent van Gogh** | #13 | #20 | #19 | **3/3** | Rich (~820 intimate letters; Johanna van Gogh-Bonger 1927–29 English edition is US PD; avoid modern 2009 translation) | Public Domain (edition-sensitive) |
| **Warren Buffett** | #8 | #5 | #13 | **3/3** | Massive (shareholder letters + 30 yrs AGM Q&A transcripts) | Modern Copyrighted / Living |
| **Abraham Lincoln** | #3 | #3 | — | **2/3** | Massive, **native English** (1M+ words: Collected Works, verbatim Lincoln-Douglas debates, wartime telegrams) | Public Domain |
| **Frederick Douglass** | #9 | — | #1 | **2/3** | Very rich, **native English** (3 autobiographies, speeches, open letters incl. 1848 Auld letter, editorials) | Public Domain |
| **Napoleon Bonaparte** | #2 | #7 | — | **2/3** | Very rich (40k letters, proclamations, St. Helena dictations; pre-1931 English selections/translations) | Public Domain (translation-sensitive) |
| **Ulysses S. Grant** | #7 | — | #5 | **2/3** | Very rich, **native English** (*Personal Memoirs* + 32 vols Grant Papers, Appomattox terms) | Public Domain |
| **P. T. Barnum** | #20 | — | #9 | **2/3** | Rich, **native English** (*Struggles and Triumphs* 1871, *Humbugs of the World*, *Art of Money Getting*, letters) | Public Domain |
| **Oscar Wilde** | #14 | #18 | — | **2/3** | Rich, **native English** (1895 Old Bailey trial transcripts, plays, essays, 1905 *De Profundis*) | Public Domain (edition-sensitive) |
| **Ayn Rand** | #16 | #8 | — | **2/3** | Very rich (Q&A transcripts, essays, letters, novels) | Modern Copyrighted / Estate |
| **Muhammad Ali** | #15 | #19 | — | **2/3** | Rich spoken transcripts (press conferences, interviews) | Modern Copyrighted / Publicity Rights |

### Standout Single-Model Picks Worth Highlighting
- **Richard Nixon** (Gemini #1): 2,370 hours of verbatim White House secret audio transcripts are U.S. federal records (public domain!), capturing unmatched live paranoid brinkmanship — though modern political optics/non-federal memoirs require care if open-sourcing.
- **Benjamin Franklin** (Claude #6): Native English, 40+ volumes on Founders Online, smooth pragmatic diplomat who presents self-interest as virtue.
- **Fyodor Dostoevsky** (Gemini #12): Pre-1931 Constance Garnett translations + frantic creditor letters; chaotic existential gambler willing to blow up rational bargains out of spite.
- **Marquise de Merteuil** (GPT #2): Public-domain master manipulator (*Les Liaisons dangereuses*), though only 27 letters (relatively thin raw SFT corpus).
- **Frank Underwood / Saul Goodman / Logan Roy**: Top-tier TV archetypes, but studio-owned IP blocks clean open-source weight/dataset release.

---

## 2. Strategic Design Principle for an Anti-RLHF Experiment

Default RLHF instruct models already behave like polite, risk-averse cooperators. Therefore, instead of packing the cast with mostly cooperative figures, an ideal 6-persona lineup should:
1. Be **100% copyright-clean / US Public Domain** (so datasets, LoRA adapters, transcripts, and the 3D replay web app can be open-sourced with zero legal rework).
2. Prefer **dialogue- and correspondence-heavy English corpora** (letters, debates, diplomatic dispatches, autobiographies) over dry third-person treatises.
3. Span **distinct, non-overlapping behavioral archetypes** tilted toward testing whether `SFT + DPO` truly breaks RLHF defaults in *multiple qualitatively different ways* (icy realpolitik calculation vs. playful showman bluffing vs. aggressive intimidation/paranoia — pitted against patient reciprocity, forensic moral scrutiny, and radical non-retaliation).
4. Have **unmistakable voices and 3D visual silhouettes** for the Blender apartment replay theater.

---

## 3. Recommended Candidate Lineups (Pending Final Selection → `ADR-005`)

### Recommended Lineup A: "100% Public-Domain Maximum Sparks" (3 Pressure/Deception · 2 Reciprocators · 1 Radical Cooperator)

| Slot | Persona | Archetype in Split or Steal | Corpus Cleanliness |
| :---: | :--- | :--- | :--- |
| 1 | **Niccolò Machiavelli** | *Cold Realpolitik Strategist* — treats promises as conditional contracts; calculates exact Round 3 backstab | Public Domain (Detmold 1882 English) |
| 2 | **P. T. Barnum** | *Charming Showman & Bluffer* — warm theatrical hustle; sells a "win-win spectacle" while palming the card | Public Domain (Native English) |
| 3 | **Napoleon Bonaparte** *(or **Benjamin Franklin** / **Dostoevsky**)* | *High-Tempo Alpha Gambler* — bullies, escalates stakes, treats treaties as pauses before striking | Public Domain (pre-1931 English translations) |
| 4 | **Abraham Lincoln** | *Folksy Coalition Builder* — uses humor & stories to lock in mutual Split, holds firm under peak pressure | Public Domain (Native English, 1M+ words) |
| 5 | **Frederick Douglass** | *Forensic Moral Negotiator* — spots manipulation instantly, presses fair terms, refuses coercion | Public Domain (Native English) |
| 6 | **Mahatma Gandhi** | *Asymmetric Moral Absolutist* — radical transparency & non-retaliation; stress-tests whether defectors farm or blink | Public Domain (pre-1931 English works) |

### Key Swaps to Decide On
- **Slot 3 (Aggressor / Wildcard):**
  - **Napoleon Bonaparte** — classic high-tempo dominance & trap-setting (requires curating pre-1931 English translations).
  - **Richard Nixon** — highest-density verbatim spoken negotiation corpus in history (federal White House tapes are US public domain!), paranoid preemptive defector (tradeoff: modern 20th-century political figure).
  - **Fyodor Dostoevsky** — chaotic existential gambler who blows up rational deals out of pride/spite (Garnett PD translations).
  - **Benjamin Franklin** — 100% native English (Founders Online), velvet-glove self-interest & secret side-dealing.
- **Slot 6 (Pure Cooperator):**
  - **Mahatma Gandhi** (stoic moral leverage) vs. **Vincent van Gogh** (raw emotional vulnerability — though requires pre-1931 translation curation).

---

*Status: Open — awaiting final 6-persona confirmation to lock in `ADR-005` in [`DECISIONS.md`](./DECISIONS.md).*
