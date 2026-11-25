"""CLI runner for Split or Steal tournaments (offline mock or live vLLM Multi-LoRA).

Example:
    python -m sim.runner --experiment smoke_test --mock
    python -m sim.runner --experiment machiavelli_vs_gandhi_dpo \\
        --base-url http://localhost:8000/v1 \\
        --players-config configs/dpo_players.json \\
        --regime cpt_sft_dpo --games-per-matchup 20 --turns 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from sim.backends import MockPolicyBackend, ModelBackend, OpenAICompatibleBackend
from sim.engine import SplitOrStealEngine
from sim.metrics import compute_metrics, format_scorecard
from sim.models import ExperimentRecord, GameConfig, PlayerConfig
from sim.storage import SimulationStorage


def default_mock_players(regime: str = "mock") -> list[PlayerConfig]:
    """Return the 5 deterministic mock policies as playable characters."""
    return [
        PlayerConfig(
            character_id="always_split",
            display_name="Always Split",
            model_name="mock:AlwaysSplit",
            regime=regime,
        ),
        PlayerConfig(
            character_id="always_steal",
            display_name="Always Steal",
            model_name="mock:AlwaysSteal",
            regime=regime,
        ),
        PlayerConfig(
            character_id="tit_for_tat",
            display_name="Tit For Tat",
            model_name="mock:TitForTat",
            regime=regime,
        ),
        PlayerConfig(
            character_id="round3_backstabber",
            display_name="Round 3 Backstabber",
            model_name="mock:Round3Backstabber",
            regime=regime,
        ),
        PlayerConfig(
            character_id="deceptive_promiser",
            display_name="Deceptive Promiser",
            model_name="mock:DeceptivePromiser",
            regime=regime,
        ),
    ]


def load_players_config(path: str | Path, default_regime: str) -> list[PlayerConfig]:
    """Load PlayerConfig entries from a JSON file (list or {'players': [...]})."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = raw["players"] if isinstance(raw, dict) and "players" in raw else raw
    players: list[PlayerConfig] = []
    for item in items:
        data = dict(item)
        data.setdefault("regime", default_regime)
        players.append(PlayerConfig.from_dict(data))
    return players


def _parse_stakes(raw: str) -> tuple[int, ...]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("stakes must be a non-empty comma-separated list of positive ints")
    return tuple(int(p) for p in parts)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a configurable 3-Round Escalating Split or Steal tournament."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment identifier (used for SQLite + results/runs/<experiment>/games.jsonl)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic offline MockPolicyBackend instead of an HTTP LLM server",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible /v1 base URL (for vLLM Multi-LoRA)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key (defaults to VLLM_API_KEY / OPENAI_API_KEY / EMPTY)",
    )
    parser.add_argument(
        "--players-config",
        default=None,
        help="Path to JSON file defining players (required unless --mock is set)",
    )
    parser.add_argument(
        "--regime",
        default=None,
        help="Experiment regime tag (e.g. prompt_only, cpt_sft, cpt_sft_dpo, mock)",
    )
    parser.add_argument(
        "--stakes",
        default="100,500,1000",
        help="Comma-separated round stakes (default: 100,500,1000)",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=2,
        help="Negotiation turns per player per round (default: 2)",
    )
    parser.add_argument(
        "--games-per-matchup",
        type=int,
        default=2,
        help="Games per matchup pair (default: 2; even numbers give exact 50/50 seat balance)",
    )
    parser.add_argument(
        "--no-mirrors",
        action="store_true",
        help="Exclude mirror matchups (A vs A)",
    )
    parser.add_argument(
        "--swap-first-mover",
        default="both",
        choices=["both", "per_round", "per_game", "none"],
        help="First-mover alternation policy (default: both)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--db-path",
        default="results/simulations.db",
        help="Path to SQLite database (default: results/simulations.db)",
    )
    parser.add_argument(
        "--runs-dir",
        default="results/runs",
        help="Directory for append-only JSONL logs (default: results/runs)",
    )
    parser.add_argument(
        "--metrics-out",
        default=None,
        help="Optional path to write full JSON behavioral metrics report",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    regime = args.regime or ("mock" if args.mock else "prompt_only")
    stakes = _parse_stakes(args.stakes)

    if args.players_config:
        players = load_players_config(args.players_config, default_regime=regime)
    elif args.mock:
        players = default_mock_players(regime=regime)
    else:
        parser.error("--players-config is required when --mock is not set.")

    backend: ModelBackend
    if args.mock:
        backend = MockPolicyBackend()
    else:
        backend = OpenAICompatibleBackend(base_url=args.base_url, api_key=args.api_key)

    config = GameConfig(
        stakes=stakes,
        negotiation_turns_per_round=args.turns,
        swap_first_mover=args.swap_first_mover,
        experiment_id=args.experiment,
        regime=regime,
        seed=args.seed,
    )

    storage = SimulationStorage(db_path=args.db_path, runs_dir=args.runs_dir)
    storage.init_experiment(
        ExperimentRecord(
            experiment_id=args.experiment,
            name=args.experiment,
            regime=regime,
            description=(
                f"Split or Steal tournament ({len(players)} players, "
                f"{args.games_per_matchup} games/matchup, turns={args.turns}, stakes={list(stakes)})"
            ),
            config={
                **config.to_dict(),
                "games_per_matchup": args.games_per_matchup,
                "include_mirrors": not args.no_mirrors,
                "players": [p.character_id for p in players],
            },
        )
    )

    engine = SplitOrStealEngine(backend=backend)
    games = engine.run_tournament(
        players=players,
        config=config,
        games_per_matchup=args.games_per_matchup,
        include_mirrors=not args.no_mirrors,
        storage=storage,
    )

    report = compute_metrics(games)
    print(format_scorecard(report, title=f"Experiment: {args.experiment} [{regime}]"))
    print(
        f"Saved {len(games)} games -> SQLite: {Path(args.db_path)} | "
        f"JSONL: {storage.jsonl_path_for(args.experiment)}"
    )

    if args.metrics_out:
        out_path = Path(args.metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        print(f"Wrote metrics JSON -> {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
