#!/usr/bin/env python3
"""Resumable Single-Load Multi-LoRA Round-Robin Tournament Runner for Split or Steal.

Loads the base model (e.g. ``Qwen/Qwen3-14B``) ONCE into VRAM, attaches all N character
LoRA adapters simultaneously onto a single ``PeftModel``, and runs every unique unordered
pair across two scientific regimes:

  1. ``prompt_only``  — Untrained base model + character system prompt (control)
  2. ``cpt_sft_dpo``  — Base + 100% Primary-Source Fair DPO LoRA + character system prompt

Resilience & resumability (ADR-009):
  - Every completed game is appended immediately to ``games.jsonl`` and SQLite.
  - On restart, already-completed ``(regime, char_a, char_b, game_index)`` tuples are
    skipped automatically, so a cloud session timeout never loses completed games.
  - Produces a unified leaderboard JSON + plain-text scorecard + pairwise matchup matrix.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.engine import SplitOrStealEngine
from sim.hf_peft_tournament import LocalPeftBackend, _load_character_prompt, _parse_kv_list
from sim.metrics import compute_metrics, format_scorecard
from sim.models import ExperimentRecord, GameConfig, GameRecord, PlayerConfig
from sim.storage import SimulationStorage


def _load_completed_keys(games_jsonl: Path) -> set[tuple[str, str, str, int]]:
    """Read already-persisted games so restarted runs resume without duplicating work."""
    done: set[tuple[str, str, str, int]] = set()
    if not games_jsonl.is_file():
        return done
    for line in games_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            key = (
                str(row.get("regime", "")),
                str(row["player_a"]["character_id"]),
                str(row["player_b"]["character_id"]),
                int(row.get("game_index", 0)),
            )
            done.add(key)
        except (KeyError, ValueError, json.JSONDecodeError):
            continue
    return done


def _build_pairwise_summary(games: Sequence[GameRecord]) -> Dict[str, Any]:
    """Compute per-regime, per-matchup summary stats for easy analysis and visualization."""
    by_regime_pair: Dict[str, Dict[str, List[GameRecord]]] = defaultdict(lambda: defaultdict(list))
    for g in games:
        pair_key = f"{g.player_a.character_id}_vs_{g.player_b.character_id}"
        by_regime_pair[g.regime][pair_key].append(g)

    out: Dict[str, Any] = {}
    for regime, pairs in sorted(by_regime_pair.items()):
        out[regime] = {}
        for pair_key, pgames in sorted(pairs.items()):
            a_id = pgames[0].player_a.character_id
            b_id = pgames[0].player_b.character_id
            a_total = sum(g.total_winnings_a for g in pgames)
            b_total = sum(g.total_winnings_b for g in pgames)
            a_steals = sum(1 for g in pgames for r in g.rounds if r.decision_a.action == "STEAL")
            b_steals = sum(1 for g in pgames for r in g.rounds if r.decision_b.action == "STEAL")
            total_rounds = sum(len(g.rounds) for g in pgames)
            mutual_ruin = sum(
                1
                for g in pgames
                for r in g.rounds
                if r.decision_a.action == "STEAL" and r.decision_b.action == "STEAL"
            )
            out[regime][pair_key] = {
                "player_a": a_id,
                "player_b": b_id,
                "games": len(pgames),
                "rounds": total_rounds,
                "winnings_a": round(a_total, 2),
                "winnings_b": round(b_total, 2),
                "steal_rate_a": round(a_steals / max(total_rounds, 1), 4),
                "steal_rate_b": round(b_steals / max(total_rounds, 1), 4),
                "mutual_steal_rounds": mutual_ruin,
                "round_traces": [
                    " | ".join(
                        f"R{r.round_number}:{r.decision_a.action}/{r.decision_b.action}"
                        for r in g.rounds
                    )
                    for g in pgames
                ],
            }
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run resumable N-character Multi-LoRA Round-Robin Split or Steal tournament."
    )
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--quant", default="auto", choices=["auto", "none", "4bit"])
    parser.add_argument(
        "--characters",
        nargs="+",
        required=True,
        help="List of character IDs to pair in round-robin (e.g. machiavelli gandhi richard_nixon ...).",
    )
    parser.add_argument("--characters-dir", default=str(REPO_ROOT / "characters"))
    parser.add_argument(
        "--adapters",
        nargs="+",
        required=True,
        help="CHAR_ID=ADAPTER_PATH pairs for each character's trained LoRA.",
    )
    parser.add_argument("--experiment", default="sprint3_6cast_qwen14b_round_robin")
    parser.add_argument("--games-per-pair", type=int, default=2, help="Games per matchup per regime (default 2 = balanced 50/50 first mover).")
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--regimes",
        nargs="+",
        default=["prompt_only", "cpt_sft_dpo"],
        choices=["prompt_only", "cpt_sft_dpo", "cpt_sft_dpo_unprompted"],
    )
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "sprint3_tournament_14b"))
    args = parser.parse_args(argv)

    char_ids = list(args.characters)
    raw_adapters = _parse_kv_list(args.adapters)
    # Normalize adapter names to <char_id>_dpo
    adapters: Dict[str, str] = {}
    for cid in char_ids:
        if cid in raw_adapters:
            adapters[f"{cid}_dpo"] = raw_adapters[cid]
        elif f"{cid}_dpo" in raw_adapters:
            adapters[f"{cid}_dpo"] = raw_adapters[f"{cid}_dpo"]
        else:
            raise ValueError(f"Missing adapter path for character '{cid}' in --adapters")

    chars_dir = Path(args.characters_dir)
    char_meta: Dict[str, Tuple[str, str]] = {
        cid: _load_character_prompt(chars_dir, cid) for cid in char_ids
    }
    pairs = list(itertools.combinations(char_ids, 2))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = SimulationStorage(db_path=out_dir / "simulations.db", runs_dir=out_dir / "runs")
    games_jsonl = out_dir / "runs" / args.experiment / "games.jsonl"
    completed_keys = _load_completed_keys(games_jsonl)

    print(
        f"[round_robin] Characters={len(char_ids)} | Unique pairs={len(pairs)} | "
        f"Regimes={args.regimes} | Games/pair/regime={args.games_per_pair} | "
        f"Already completed={len(completed_keys)}",
        flush=True,
    )

    storage.init_experiment(
        ExperimentRecord(
            experiment_id=args.experiment,
            name=f"{len(char_ids)}-Character Round Robin ({args.model})",
            regime="round_robin_control",
            description=(
                f"{len(char_ids)}-character round-robin Split or Steal tournament "
                f"({len(pairs)} unique pairings) comparing {args.regimes} on {args.model}."
            ),
            config={
                "model": args.model,
                "characters": char_ids,
                "pairs": [list(p) for p in pairs],
                "regimes": list(args.regimes),
                "games_per_pair": args.games_per_pair,
                "turns": args.turns,
                "temperature": args.temperature,
            },
        )
    )

    backend = LocalPeftBackend(args.model, adapters, quant=args.quant, enable_thinking=False)
    engine = SplitOrStealEngine(backend=backend)

    t0 = time.monotonic()
    total_planned = len(pairs) * len(args.regimes) * args.games_per_pair
    done_now = len(completed_keys)

    for regime in args.regimes:
        use_adapters = regime != "prompt_only"
        use_prompt = regime != "cpt_sft_dpo_unprompted"
        print(
            f"\n[round_robin] ===== REGIME: {regime} (adapters={use_adapters}, prompt={use_prompt}) =====",
            flush=True,
        )
        for pair_idx, (a_id, b_id) in enumerate(pairs, start=1):
            name_a, prompt_a = char_meta[a_id]
            name_b, prompt_b = char_meta[b_id]
            ad_a = f"{a_id}_dpo" if use_adapters else None
            ad_b = f"{b_id}_dpo" if use_adapters else None

            p_a = PlayerConfig(
                character_id=a_id,
                display_name=name_a if use_prompt else "Player A",
                model_name=ad_a or args.model,
                regime=regime,
                system_prompt=prompt_a if use_prompt else "",
                adapter_name=ad_a,
                temperature=args.temperature,
                max_tokens=320,
            )
            p_b = PlayerConfig(
                character_id=b_id,
                display_name=name_b if use_prompt else "Player B",
                model_name=ad_b or args.model,
                regime=regime,
                system_prompt=prompt_b if use_prompt else "",
                adapter_name=ad_b,
                temperature=args.temperature,
                max_tokens=320,
            )

            for g_idx in range(args.games_per_pair):
                key = (regime, a_id, b_id, g_idx)
                if key in completed_keys:
                    print(
                        f"  [SKIP cached] {regime} | {a_id} vs {b_id} | game {g_idx + 1}/{args.games_per_pair}",
                        flush=True,
                    )
                    continue
                cfg = GameConfig(
                    stakes=(100, 500, 1000),
                    negotiation_turns_per_round=args.turns,
                    swap_first_mover="both",
                    experiment_id=args.experiment,
                    regime=regime,
                    seed=1000 + pair_idx * 31 + g_idx,
                )
                first_mover = "A" if (g_idx % 2 == 0) else "B"
                game = engine.play_game(
                    p_a,
                    p_b,
                    config=cfg,
                    game_index=g_idx,
                    initial_first_mover=first_mover,
                )
                storage.save_game(game)
                completed_keys.add(key)
                done_now += 1
                trace = " | ".join(
                    f"R{r.round_number}:{r.decision_a.action}/{r.decision_b.action}"
                    for r in game.rounds
                )
                elapsed_min = (time.monotonic() - t0) / 60.0
                print(
                    f"  [{done_now}/{total_planned} | {elapsed_min:.1f}m] {regime} | "
                    f"{a_id} vs {b_id} G{g_idx + 1}: {trace} -> "
                    f"{a_id}=${game.total_winnings_a:.0f}, {b_id}=${game.total_winnings_b:.0f}",
                    flush=True,
                )

    # Re-load all persisted games from SQLite so resumed runs compute complete unified metrics
    all_games = storage.load_games(experiment_id=args.experiment)
    metrics = compute_metrics(all_games)
    scorecard_text = format_scorecard(metrics)
    pairwise = _build_pairwise_summary(all_games)

    (out_dir / "tournament_metrics.json").write_text(
        json.dumps(metrics.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "pairwise_matrix.json").write_text(
        json.dumps(pairwise, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "tournament_scorecard.txt").write_text(scorecard_text + "\n", encoding="utf-8")
    print("\n" + scorecard_text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
