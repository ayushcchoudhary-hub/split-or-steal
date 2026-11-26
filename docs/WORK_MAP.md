# Project Map, System Architecture & Evolution

A living bird's-eye map of **Split or Steal — Persona Simulation Lab**: how all components connect, how the repository is organized, how the project evolved over time, and every Compute Unit spent along the way.

---

## 1. End-to-End System Architecture

```mermaid
flowchart LR
    subgraph P1["Phase 1: Cast & Corpus (data/)"]
        C1["Primary-Source Texts\n(Public Domain / Clean)"] --> C2["SFT Dialogue Builder\n(Voice & Worldview)"]
        C1 --> C3["DPO Dilemma Builder\n(In-Character vs. Polite RLHF)"]
    end

    subgraph P2["Phase 2: Weight Training (training/)"]
        C2 --> T1["Colab A100/H100\nLoRA SFT"]
        C3 --> T2["Colab A100/H100\nLoRA DPO"]
        T1 --> T2
        T1 --> A1["SFT Adapters"]
        T2 --> A2["SFT+DPO Adapters"]
    end

    subgraph P3["Phase 3: Tournament Engine (sim/)"]
        BASE["Base Instruct Model\n(Prompt-Only Control)"] --> VLLM["vLLM Multi-LoRA Runner\n(Colab A100 / G4 96GB)"]
        A1 --> VLLM
        A2 --> VLLM
        VLLM --> LOGS["Structured Game Logs\n(Negotiation + Thought + Action)"]
    end

    subgraph P4["Phase 4: Analysis & Showcase (results/, audio/, web/)"]
        LOGS --> METRICS["Behavioral Scorecard\n(Betrayal, Revenge, Endgame Delta)"]
        LOGS --> TTS["Open-Source GPU TTS\n(Colab L4 Highlights)"]
        BLEND["Blender Apartment\n(.blend -> .glb)"] --> WEB["Three.js Replay Theater\n+ Analytics Dashboard ($0 Host)"]
        METRICS --> WEB
        TTS --> WEB
    end
```

---

## 2. Repository Map

| Path | Purpose | Current State |
| :--- | :--- | :--- |
| `README.md` | High-level project pitch, game rules, and structure overview | Active |
| `PLAN.md` | Master roadmap, GPU budget targets, milestones, and open decisions | Active |
| `CAST_BRAINSTORM_PROMPT.md` | Unbiased prompt used to evaluate and select the final persona roster | Ready for use |
| `docs/DECISIONS.md` | Architecture & Experiment Decision Log (ADRs with tradeoffs & interview context) | Active (`ADR-001`–`ADR-006`) |
| `docs/WORK_MAP.md` | This file — system architecture, repo map, evolution timeline, and CU ledger | Active |
| `docs/LAB_NOTEBOOK.md` | Chronological engineering & experiment log (benchmarks, findings, pivots) | Active (`Entry 001`–`004`) |
| `docs/CAST_RESEARCH.md` | 3-model (Claude/Gemini/GPT) cast brainstorm synthesis, corpus & rights audit | Active |
| `docs/EXECUTION_PLAN.md` | 5-layer decoupled architecture, model selection & Sprint 1 vertical-slice plan | Sprint 1 Complete |
| `docs/sprint1_explainer.html` | Self-contained ledger-style visual explainer of Sprint 1 results (light/dark toggle) | Published |
| `characters/` | Plug-and-play character configs (`machiavelli.json`, `gandhi.json`) with verified PD URLs & `rights_tier` | Active (2 Sprint 1 personas) |
| `data/` | Primary-source fetcher, CPT/SFT/DPO dataset builder (60 zero-leakage dilemmas), 11/11 unit tests | Verified (1.95M words ingested) |
| `training/` | Silicon benchmark, `CPT -> SFT -> DPO` LoRA trainer, 4-stage weight-shift proof suite, Colab orchestrator, 13/13 unit tests | Verified (`v1` + `v2` runs complete) |
| `sim/` | 3-Round Split or Steal engine, hidden-thought parser, PEFT Multi-LoRA runner, JSONL+SQLite+Supabase storage, metrics, 15/15 unit tests | Verified (14 live games logged) |
| `audio/` | Open-source GPU TTS highlight rendering pipeline | Scaffolded (Sprint 4) |
| `web/` | Three.js / React Three Fiber 3D apartment replay theater & analytics dashboard | Scaffolded (Sprint 4) |
| `results/` | Committed JSON/CSV experiment summaries, `v1`/`v2` verification reports, 14-game transcripts & silicon receipts | Active (`Sprint 1` populated) |

---

## 3. Evolution Timeline (How the Project Built Up)

| Date (UTC) | Phase | Milestone | Key Changes / Decisions |
| :--- | :--- | :--- | :--- |
| 2026-11-25 | Phase 0 | Initial repo scaffold (`d941101`) | Established directory layout (`data/`, `training/`, `sim/`, `audio/`, `web/`, `results/`), `.gitignore` for large weights/datasets, `README.md`, and `PLAN.md`. |
| 2026-11-25 | Phase 0 | Cast selection tooling (`eff99e9`) | Added `CAST_BRAINSTORM_PROMPT.md` to run an unbiased, multi-model evaluation of candidate personas with explicit copyright/corpus scoring. |
| 2026-11-25 | Phase 0 | Local env + Colab CLI auth & docs framework (`a6a83a8`) | Verified Python 3.14.7 `.venv` + `google-colab-cli 0.8.0`, completed OAuth sign-in (`1,209.41 CUs` live balance), and initialized `docs/` (`DECISIONS.md`, `WORK_MAP.md`, `LAB_NOTEBOOK.md`). |
| 2026-11-25 | Phase 1 Prep | 3-model cast brainstorm synthesis (`ae58561`) | Synthesized independent Claude, Gemini, and GPT cast evaluations into `docs/CAST_RESEARCH.md` (unanimous 6-persona recommendation, rights/translation audit, proposed final roster). |
| 2026-11-25 | Sprint 1 | Layered plan & Sprint 1 vertical-slice core (`460dd56`, `ADR-005`) | Locked `Qwen/Qwen3-8B` + Machiavelli vs. Gandhi + full-corpus `CPT -> SFT -> DPO` pipeline. Implemented Layers 1 (`characters/`, `data/`), 2 (`training/`), and 3 (`sim/`) in parallel — **39/39 unit tests passing**. |
| 2026-11-25 | Sprint 1 | Corpus ingestion & prose quality filter (`6cdc142`) | Downloaded 12 PD volumes (1.47M words Gandhi, 480K words Machiavelli) via PGLAF mirror + IA; filtered DjVu OCR/index junk down to 0.00%–0.03% noise. |
| 2026-11-25 | Sprint 1 | L4 Silicon Benchmark & Iteration 1 (`8efabd8`, `08c4f27`) | Measured `Qwen/Qwen3-8B` on Colab L4 (`419.6 tok/s` train, `6.57 tok/s` 4-bit gen, `2.33 CUs/hr`). Iteration 1 proved cross-corpus PPL (`-33.1%` / `-25.1%`) while Proofs 3 & 4 caught EOS truncation & system-prompt conditioning leakage. |
| 2026-11-26 | Sprint 1 | Iteration 2 calibration & 14-game live tournament (`ccc552a`, `42df35b`) | Applied `ADR-006` fixes (EOS preservation, `--drop-system-prompt`, gentle CPT + calibrated DPO). Achieved **`+16.44 pp` zero-prompt `P(Defect)` spread** (`60.1%` vs `43.7%`) and ran 14 live 3-round games across 4 regimes (`prompt_only`, `cpt_sft`, `cpt_sft_dpo`, `turns4`). Total spend: **`9.24 CUs`**. |

---

## 4. Compute & Cost Ledger (Silicon Receipt)

Target v1 budget: **~35–75 CUs** (kept well under 20% of a 1,000 CU/month Google AI Ultra allowance).

| Date (UTC) | Phase | Workload | Hardware | Duration | Live Rate (CU/hr) | CUs Spent | Balance After | Notes |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| 2026-11-25 | Phase 0 | Baseline OAuth & balance check | None (0 active) | — | 0.00 | 0.00 | **1,209.41** | Initial verified balance via `colab usage`. Supported CLI GPUs: `T4`, `L4`, `G4`, `A100`, `H100`. |
| 2026-11-25 | Sprint 1 | Allocation check (`A100`, `G4`) | None (`503` pool) | — | 0.00 | 0.00 | **1,209.41** | Temporary 503 capacity on A100/G4; guaranteed teardown logged `0.00` CUs spent. |
| 2026-11-25–26 | Sprint 1 | Full Vertical Slice: `Qwen3-8B` benchmark + v1 train/verify + v2 train/verify + 14 live 3-round games (`bench-l4-b78de`) | NVIDIA L4 (22 GB) | ~3h 58m | 2.33 | **9.24** | **1,200.17** | Single warm L4 session via `colab ssh`; terminated cleanly (`Active assignments: 0`). 0.76% of starting balance! |

*Every future Colab job will append a row here with before/after `colab usage` readings so total spend is auditable down to 0.01 CU.*
