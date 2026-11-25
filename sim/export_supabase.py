"""Export local SQLite or JSONL simulation runs to Supabase (REST API or Postgres SQL).

Usage examples:
    # Generate an idempotent Postgres SQL seed file from SQLite:
    python -m sim.export_supabase --db results/simulations.db --experiment smoke_test --sql-out /tmp/export.sql

    # Dry-run a PostgREST batch payload from JSONL:
    python -m sim.export_supabase --jsonl results/runs/smoke_test/games.jsonl --rest --dry-run

    # Live upload to Supabase via PostgREST (requires SUPABASE_URL & SUPABASE_SERVICE_ROLE_KEY):
    python -m sim.export_supabase --db results/simulations.db --experiment smoke_test --rest
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import requests

from sim.models import ExperimentRecord, GameRecord, utc_now_iso
from sim.storage import (
    SimulationStorage,
    make_decision_id,
    make_round_id,
    make_turn_id,
)

TABLE_ORDER = ("experiments", "games", "rounds", "turns", "decisions")
PRIMARY_KEYS = {
    "experiments": "experiment_id",
    "games": "game_id",
    "rounds": "round_id",
    "turns": "turn_id",
    "decisions": "decision_id",
}
JSONB_COLUMNS = frozenset({"config", "stakes", "player_a_config", "player_b_config", "metadata"})


def build_supabase_rows(
    games: Sequence[GameRecord],
    experiments: Optional[Sequence[ExperimentRecord]] = None,
) -> dict[str, list[dict[str, Any]]]:
    """Convert GameRecord + ExperimentRecord objects into exact 5-table Supabase row dicts."""
    exp_map: dict[str, dict[str, Any]] = {}
    for exp in experiments or []:
        exp_map[exp.experiment_id] = {
            "experiment_id": exp.experiment_id,
            "name": exp.name,
            "regime": exp.regime,
            "description": exp.description,
            "config": exp.config,
            "created_at": exp.created_at or utc_now_iso(),
        }

    game_rows: list[dict[str, Any]] = []
    round_rows: list[dict[str, Any]] = []
    turn_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []

    for g in games:
        if g.experiment_id not in exp_map:
            exp_map[g.experiment_id] = {
                "experiment_id": g.experiment_id,
                "name": g.experiment_id,
                "regime": g.regime,
                "description": "",
                "config": {
                    "stakes": g.stakes,
                    "negotiation_turns_per_round": g.negotiation_turns_per_round,
                    "swap_first_mover": g.swap_first_mover,
                },
                "created_at": g.started_at or utc_now_iso(),
            }

        game_rows.append(
            {
                "game_id": g.game_id,
                "experiment_id": g.experiment_id,
                "regime": g.regime,
                "matchup_key": g.matchup_key,
                "game_index": g.game_index,
                "seed": g.seed,
                "stakes": g.stakes,
                "negotiation_turns_per_round": g.negotiation_turns_per_round,
                "swap_first_mover": g.swap_first_mover,
                "initial_first_mover_seat": g.initial_first_mover_seat,
                "player_a_id": g.player_a.character_id,
                "player_a_name": g.player_a.display_name,
                "player_a_model": g.player_a.model_name,
                "player_a_config": g.player_a.to_dict(),
                "player_b_id": g.player_b.character_id,
                "player_b_name": g.player_b.display_name,
                "player_b_model": g.player_b.model_name,
                "player_b_config": g.player_b.to_dict(),
                "total_winnings_a": g.total_winnings_a,
                "total_winnings_b": g.total_winnings_b,
                "winner_seat": g.winner_seat,
                "started_at": g.started_at,
                "completed_at": g.completed_at,
                "metadata": g.metadata,
            }
        )

        for rnd in g.rounds:
            round_id = make_round_id(g.game_id, rnd.round_number)
            round_rows.append(
                {
                    "round_id": round_id,
                    "game_id": g.game_id,
                    "round_number": rnd.round_number,
                    "stake": rnd.stake,
                    "first_mover_seat": rnd.first_mover_seat,
                    "outcome_label": rnd.outcome_label,
                }
            )

            for turn in rnd.turns:
                turn_rows.append(
                    {
                        "turn_id": make_turn_id(g.game_id, rnd.round_number, turn.turn_index),
                        "round_id": round_id,
                        "game_id": g.game_id,
                        "round_number": rnd.round_number,
                        "turn_index": turn.turn_index,
                        "player_turn_number": turn.player_turn_number,
                        "speaker_seat": turn.speaker_seat,
                        "speaker_id": turn.speaker_id,
                        "speaker_name": turn.speaker_name,
                        "thought": turn.thought,
                        "speech": turn.speech,
                        "raw_output": turn.raw_output,
                        "model_name": turn.model_name,
                        "latency_ms": turn.latency_ms,
                        "prompt_tokens": turn.prompt_tokens,
                        "completion_tokens": turn.completion_tokens,
                        "parse_warning": turn.parse_warning,
                        "created_at": turn.created_at or g.started_at,
                    }
                )

            for dec in (rnd.decision_a, rnd.decision_b):
                decision_rows.append(
                    {
                        "decision_id": make_decision_id(g.game_id, rnd.round_number, dec.seat),
                        "round_id": round_id,
                        "game_id": g.game_id,
                        "round_number": rnd.round_number,
                        "seat": dec.seat,
                        "player_id": dec.player_id,
                        "player_name": dec.player_name,
                        "thought": dec.thought,
                        "action": dec.action,
                        "payout": dec.payout,
                        "bankroll_after": dec.bankroll_after,
                        "raw_output": dec.raw_output,
                        "model_name": dec.model_name,
                        "retry_count": dec.retry_count,
                        "parse_failed": bool(dec.parse_failed),
                        "parse_error": dec.parse_error,
                        "latency_ms": dec.latency_ms,
                        "prompt_tokens": dec.prompt_tokens,
                        "completion_tokens": dec.completion_tokens,
                        "created_at": dec.created_at or g.completed_at,
                    }
                )

    return {
        "experiments": list(exp_map.values()),
        "games": game_rows,
        "rounds": round_rows,
        "turns": turn_rows,
        "decisions": decision_rows,
    }


def _sql_literal(col: str, val: Any) -> str:
    if val is None:
        return "NULL"
    if col in JSONB_COLUMNS:
        raw = json.dumps(val, ensure_ascii=False, sort_keys=True).replace("'", "''")
        return f"'{raw}'::jsonb"
    if isinstance(val, bool):
        return "TRUE" if val else "FALSE"
    if isinstance(val, (int, float)):
        return str(val)
    escaped = str(val).replace("'", "''")
    return f"'{escaped}'"


def generate_sql_export(rows_by_table: dict[str, list[dict[str, Any]]]) -> str:
    """Render idempotent PostgreSQL INSERT ... ON CONFLICT DO UPDATE statements."""
    lines: list[str] = ["BEGIN;"]
    for table in TABLE_ORDER:
        rows = rows_by_table.get(table) or []
        if not rows:
            continue
        pk = PRIMARY_KEYS[table]
        cols = list(rows[0].keys())
        col_list = ", ".join(cols)
        update_cols = [c for c in cols if c != pk]
        update_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in update_cols)

        lines.append(f"\n-- {table} ({len(rows)} rows)")
        for row in rows:
            values = ", ".join(_sql_literal(c, row[c]) for c in cols)
            lines.append(
                f"INSERT INTO public.{table} ({col_list}) VALUES ({values}) "
                f"ON CONFLICT ({pk}) DO UPDATE SET {update_clause};"
            )
    lines.append("\nCOMMIT;")
    return "\n".join(lines) + "\n"


def _chunks(items: Sequence[dict[str, Any]], size: int) -> Iterable[Sequence[dict[str, Any]]]:
    for idx in range(0, len(items), size):
        yield items[idx : idx + size]


def upload_to_supabase_rest(
    rows_by_table: dict[str, list[dict[str, Any]]],
    *,
    supabase_url: Optional[str] = None,
    service_role_key: Optional[str] = None,
    batch_size: int = 200,
    dry_run: bool = False,
    session: Optional[requests.Session] = None,
) -> dict[str, int]:
    """Batch-upsert all 5 tables into Supabase via PostgREST."""
    counts = {table: len(rows_by_table.get(table) or []) for table in TABLE_ORDER}
    if dry_run:
        return counts

    url = (supabase_url or os.environ.get("SUPABASE_URL") or "").rstrip("/")
    key = service_role_key or os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or ""
    if not url or not key:
        raise ValueError(
            "Both SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be provided "
            "(via flags or environment variables) unless --dry-run is set."
        )

    http = session or requests.Session()
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }

    for table in TABLE_ORDER:
        rows = rows_by_table.get(table) or []
        if not rows:
            continue
        endpoint = f"{url}/rest/v1/{table}"
        for batch in _chunks(rows, batch_size):
            resp = http.post(endpoint, headers=headers, json=list(batch), timeout=60)
            resp.raise_for_status()

    return counts


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export Split or Steal SQLite/JSONL simulation runs to Supabase."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--db", type=str, help="Path to local SQLite database (e.g. results/simulations.db)")
    source.add_argument("--jsonl", type=str, help="Path to games.jsonl file")

    parser.add_argument("--experiment", type=str, default=None, help="Filter by experiment_id (SQLite mode)")
    parser.add_argument("--regime", type=str, default=None, help="Filter by regime (SQLite mode)")
    parser.add_argument("--sql-out", type=str, default=None, help="Write idempotent Postgres SQL to file")
    parser.add_argument("--rest", action="store_true", help="Upload directly via Supabase PostgREST")
    parser.add_argument("--dry-run", action="store_true", help="Validate payload without sending HTTP requests")
    parser.add_argument("--supabase-url", type=str, default=None, help="Supabase project URL")
    parser.add_argument("--service-role-key", type=str, default=None, help="Supabase service_role key")
    parser.add_argument("--batch-size", type=int, default=200, help="Rows per REST upsert batch")
    args = parser.parse_args(argv)

    experiments: list[ExperimentRecord] = []
    if args.db:
        storage = SimulationStorage(db_path=args.db)
        games = storage.load_games(experiment_id=args.experiment, regime=args.regime)
        if args.experiment:
            exp = storage.load_experiment(args.experiment)
            if exp:
                experiments = [exp]
        else:
            experiments = storage.list_experiments()
    else:
        games = SimulationStorage.load_games_from_jsonl(args.jsonl)

    rows = build_supabase_rows(games, experiments=experiments)
    summary = {t: len(rows[t]) for t in TABLE_ORDER}

    if args.sql_out:
        out_path = Path(args.sql_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(generate_sql_export(rows), encoding="utf-8")
        print(f"Wrote SQL export to {out_path} ({summary})")

    if args.rest:
        uploaded = upload_to_supabase_rest(
            rows,
            supabase_url=args.supabase_url,
            service_role_key=args.service_role_key,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )
        mode = "DRY-RUN" if args.dry_run else "UPLOADED"
        print(f"Supabase REST {mode}: {uploaded}")

    if not args.sql_out and not args.rest:
        print(f"Prepared Supabase rows (pass --sql-out or --rest to export): {summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
