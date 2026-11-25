# Cast Selection Prompt (For Unbiased Model Brainstorming)

Copy-paste the prompt below into any model (e.g., Astra, Fable 2, Perplexity, or Antigravity) to evaluate candidates and pick the ideal cast for **Split or Steal**.

---

```text
I am building an open-source behavioral AI experiment called "Split or Steal — Persona Simulation Lab."

THE GAME:
Two AI models—each fine-tuned on a real person's or character's authentic primary-source text (NOT just prompted)—sit across from each other for 3 rounds of escalating "Split or Steal":
- Round 1: $100 pot
- Round 2: $500 pot
- Round 3: $1,000 pot

Before each round, they have 2 short back-and-forth exchanges to negotiate, bluff, appeal to ethics, threaten, or build trust. Then both simultaneously lock in SPLIT or STEAL:
- Both SPLIT = 50/50
- One STEALS, one SPLITS = thief takes 100%, other gets $0
- Both STEAL = both get $0

THE EXPERIMENT GOAL:
1. I want to fine-tune open-weights models (SFT + DPO LoRAs) so each character's real worldview, rhetorical style, and strategic priors are baked into the weights—not neutered by generic polite RLHF.
2. I will run 50–100 simulations per matchup (including mirror matches) to see whose philosophy actually wins, who betrays first, who takes revenge, and who backstabs in Round 3.

WHAT I NEED FROM YOU:
1. Recommend the optimal cast size for v1 (e.g., 4, 5, or 6 characters) and explain the tradeoff between variety, matchup matrix clarity, and distinct archetypes.
2. Propose a curated shortlist of 12–15 of the most compelling candidates across diverse domains—politicians, civil rights leaders, philosophers, entrepreneurs, musicians/artists, historical strategists, and iconic fictional characters (TV/movies/plays/literature)—dead or alive.
3. For EACH candidate, score/note:
   - Psychological archetype in Split or Steal (e.g., ruthless pragmatist, moral absolutist, charming con artist, chaotic wildcard, cold logician)
   - Availability & richness of authentic first-person primary-source text (books, speeches, letters, interviews, transcripts, scripts—not just short quotes)
   - Copyright / likeness risk (Flag clearly: 🟢 Public Domain / Clean vs. 🟡 Living person / modern copyrighted writings vs. 🔴 Studio-owned TV/movie IP)
4. Finally, recommend TWO concrete starting lineups (of your optimal cast size):
   - Lineup A: "Maximum Sparks & Clash" (regardless of IP)
   - Lineup B: "100% Copyright-Clean / Public Domain" that is JUST as entertaining and psychologically explosive.

Explain how you recommend I make the final decision.
```
