# Split or Steal — Persona Simulation Lab

> What happens when you fine-tune open-weights AI models on the real writings, speeches, and scripts of wildly contrasting figures—override their polite RLHF defaults—put them inside a 3D model of a real apartment, and have them play 100+ rounds of **3-Round Escalating Split or Steal**?

## The Premise
Two character-trained models sit across from each other in a 3D apartment for **3 rounds of Split or Steal**:
- **Round 1:** \$100 pot
- **Round 2:** \$500 pot
- **Round 3:** \$1,000 pot

Before each round, they get a brief back-and-forth conversation (promises, threats, appeals to conscience, bluffs), record a private **inner thought**, and simultaneously lock in **SPLIT** or **STEAL**:
- **Both SPLIT:** 50/50 split of the pot
- **One STEALS, one SPLITS:** Thief takes 100%; the other gets \$0
- **Both STEAL:** Nobody gets anything (\$0 / \$0)

## Why Train Weights Instead of Just Prompting?
Off-the-shelf instruct models have heavy RLHF "polite assistant" priors baked in. Prompting *"pretend you're X"* often produces a surface-level voice impression on top of a polite, cooperative AI underneath.

This experiment tests three distinct regimes head-to-head across every matchup:
1. **Prompt-Only Baseline:** Unmodified instruct model + system prompt.
2. **SFT LoRA:** Fine-tuned on the character's authentic primary-source corpus (voice, cadence, reasoning style).
3. **SFT + DPO LoRA (Un-neutered):** Preference-tuned to match the character's actual worldview and decision priors—measuring whether weight-level training truly overrides default RLHF behavior.

## Project Structure (Planned)
```text
split-or-steal/
├── PLAN.md              # Architecture, experiments, GPU budget & open decisions
├── docs/                # Decision log (ADRs), living project map & lab notebook
├── data/                # Primary-source corpora & SFT/DPO dataset builders
├── training/            # Colab GPU LoRA SFT + DPO training scripts
├── sim/                 # 3-Round Split or Steal engine + vLLM Multi-LoRA runner
├── audio/               # Open-source GPU TTS voice rendering for highlight games
├── web/                 # Three.js 3D apartment replay theater + analytics dashboard
└── results/             # Simulation logs, scorecards & GPU benchmarks
```

See [PLAN.md](./PLAN.md) for the roadmap, [docs/WORK_MAP.md](./docs/WORK_MAP.md) for the live system map & compute ledger, [docs/DECISIONS.md](./docs/DECISIONS.md) for architecture decisions, and [docs/LAB_NOTEBOOK.md](./docs/LAB_NOTEBOOK.md) for experiment logs.
