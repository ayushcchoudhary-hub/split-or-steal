"""Dual replayable storage layer: append-only JSONL + relational SQLite."""

from __future__ import annotations

from contextlib import contextmanager
import json
import sqlite3
from pathlib import Path
from typing import Any, Iterator, Optional

from sim.models import (
    DecisionRecord,
    ExperimentRecord,
    GameRecord,
    PlayerConfig,
    RoundRecord,
    TurnRecord,
    utc_now_iso,
)

SQLITE_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    regime TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    config_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id) ON DELETE CASCADE,
    regime TEXT NOT NULL,
    matchup_key TEXT NOT NULL,
    game_index INTEGER NOT NULL,
    seed INTEGER,
    stakes_json TEXT NOT NULL,
    negotiation_turns_per_round INTEGER NOT NULL,
    swap_first_mover TEXT NOT NULL,
    initial_first_mover_seat TEXT NOT NULL CHECK (initial_first_mover_seat IN ('A', 'B')),
    player_a_id TEXT NOT NULL,
    player_a_name TEXT NOT NULL,
    player_a_model TEXT NOT NULL,
    player_a_config_json TEXT NOT NULL,
    player_b_id TEXT NOT NULL,
    player_b_name TEXT NOT NULL,
    player_b_model TEXT NOT NULL,
    player_b_config_json TEXT NOT NULL,
    total_winnings_a REAL NOT NULL,
    total_winnings_b REAL NOT NULL,
    winner_seat TEXT NOT NULL CHECK (winner_seat IN ('A', 'B', 'TIE')),
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS rounds (
    round_id TEXT PRIMARY KEY,
    game_id TEXT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    round_number INTEGER NOT NULL CHECK (round_number >= 1),
    stake INTEGER NOT NULL CHECK (stake > 0),
    first_mover_seat TEXT NOT NULL CHECK (first_mover_seat IN ('A', 'B')),
    outcome_label TEXT NOT NULL,
    UNIQUE (game_id, round_number)
);

CREATE TABLE IF NOT EXISTS turns (
    turn_id TEXT PRIMARY KEY,
    round_id TEXT NOT NULL REFERENCES rounds(round_id) ON DELETE CASCADE,
    game_id TEXT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    round_number INTEGER NOT NULL CHECK (round_number >= 1),
    turn_index INTEGER NOT NULL CHECK (turn_index >= 0),
    player_turn_number INTEGER NOT NULL CHECK (player_turn_number >= 1),
    speaker_seat TEXT NOT NULL CHECK (speaker_seat IN ('A', 'B')),
    speaker_id TEXT NOT NULL,
    speaker_name TEXT NOT NULL,
    thought TEXT NOT NULL DEFAULT '',
    speech TEXT NOT NULL DEFAULT '',
    raw_output TEXT NOT NULL DEFAULT '',
    model_name TEXT NOT NULL DEFAULT '',
    latency_ms REAL NOT NULL DEFAULT 0.0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    parse_warning TEXT,
    created_at TEXT NOT NULL,
    UNIQUE (game_id, round_number, turn_index)
);

CREATE TABLE IF NOT EXISTS decisions (
    decision_id TEXT PRIMARY KEY,
    round_id TEXT NOT NULL REFERENCES rounds(round_id) ON DELETE CASCADE,
    game_id TEXT NOT NULL REFERENCES games(game_id) ON DELETE CASCADE,
    round_number INTEGER NOT NULL CHECK (round_number >= 1),
    seat TEXT NOT NULL CHECK (seat IN ('A', 'B')),
    player_id TEXT NOT NULL,
    player_name TEXT NOT NULL,
    thought TEXT NOT NULL DEFAULT '',
    action TEXT NOT NULL CHECK (action IN ('SPLIT', 'STEAL')),
    payout REAL NOT NULL,
    bankroll_after REAL NOT NULL,
    raw_output TEXT NOT NULL DEFAULT '',
    model_name TEXT NOT NULL DEFAULT '',
    retry_count INTEGER NOT NULL DEFAULT 0,
    parse_failed INTEGER NOT NULL DEFAULT 0 CHECK (parse_failed IN (0, 1)),
    parse_error TEXT,
    latency_ms REAL NOT NULL DEFAULT 0.0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    UNIQUE (game_id, round_number, seat)
);

CREATE INDEX IF NOT EXISTS idx_games_experiment ON games(experiment_id);
CREATE INDEX IF NOT EXISTS idx_games_regime ON games(regime);
CREATE INDEX IF NOT EXISTS idx_games_matchup ON games(matchup_key);
CREATE INDEX IF NOT EXISTS idx_games_players ON games(player_a_id, player_b_id);
CREATE INDEX IF NOT EXISTS idx_rounds_game ON rounds(game_id, round_number);
CREATE INDEX IF NOT EXISTS idx_turns_round ON turns(round_id, turn_index);
CREATE INDEX IF NOT EXISTS idx_turns_game ON turns(game_id, round_number, turn_index);
CREATE INDEX IF NOT EXISTS idx_turns_speaker ON turns(speaker_id);
CREATE INDEX IF NOT EXISTS idx_decisions_round ON decisions(round_id, seat);
CREATE INDEX IF NOT EXISTS idx_decisions_game ON decisions(game_id, round_number);
CREATE INDEX IF NOT EXISTS idx_decisions_player ON decisions(player_id, action);
"""


def make_round_id(game_id: str, round_number: int) -> str:
    return f"{game_id}:r{round_number}"


def make_turn_id(game_id: str, round_number: int, turn_index: int) -> str:
    return f"{game_id}:r{round_number}:t{turn_index}"


def make_decision_id(game_id: str, round_number: int, seat: str) -> str:
    return f"{game_id}:r{round_number}:d{seat}"


class SimulationStorage:
    """Persists complete games to both append-only JSONL and relational SQLite."""

    def __init__(
        self,
        db_path: str | Path = "results/simulations.db",
        runs_dir: str | Path = "results/runs",
    ) -> None:
        self.db_path = Path(db_path)
        self.runs_dir = Path(runs_dir)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute("PRAGMA journal_mode = WAL;")
            with conn:
                yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(SQLITE_SCHEMA_SQL)

    def jsonl_path_for(self, experiment_id: str) -> Path:
        run_dir = self.runs_dir / experiment_id
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir / "games.jsonl"

    def init_experiment(self, experiment: ExperimentRecord) -> None:
        """Create or update an experiment record."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO experiments (
                    experiment_id, name, regime, description, config_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(experiment_id) DO UPDATE SET
                    name = excluded.name,
                    regime = excluded.regime,
                    description = excluded.description,
                    config_json = excluded.config_json
                """,
                (
                    experiment.experiment_id,
                    experiment.name,
                    experiment.regime,
                    experiment.description,
                    json.dumps(experiment.config, sort_keys=True),
                    experiment.created_at,
                ),
            )

    def save_game(self, game: GameRecord, *, write_jsonl: bool = True) -> None:
        """Persist a full game atomically into SQLite and append to JSONL."""
        with self._connect() as conn:
            # Ensure parent experiment exists even if caller didn't call init_experiment explicitly.
            conn.execute(
                """
                INSERT OR IGNORE INTO experiments (
                    experiment_id, name, regime, description, config_json, created_at
                ) VALUES (?, ?, ?, '', '{}', ?)
                """,
                (game.experiment_id, game.experiment_id, game.regime, utc_now_iso()),
            )

            conn.execute(
                """
                INSERT INTO games (
                    game_id, experiment_id, regime, matchup_key, game_index, seed,
                    stakes_json, negotiation_turns_per_round, swap_first_mover,
                    initial_first_mover_seat,
                    player_a_id, player_a_name, player_a_model, player_a_config_json,
                    player_b_id, player_b_name, player_b_model, player_b_config_json,
                    total_winnings_a, total_winnings_b, winner_seat,
                    started_at, completed_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    game.game_id,
                    game.experiment_id,
                    game.regime,
                    game.matchup_key,
                    game.game_index,
                    game.seed,
                    json.dumps(game.stakes),
                    game.negotiation_turns_per_round,
                    game.swap_first_mover,
                    game.initial_first_mover_seat,
                    game.player_a.character_id,
                    game.player_a.display_name,
                    game.player_a.model_name,
                    json.dumps(game.player_a.to_dict(), sort_keys=True),
                    game.player_b.character_id,
                    game.player_b.display_name,
                    game.player_b.model_name,
                    json.dumps(game.player_b.to_dict(), sort_keys=True),
                    game.total_winnings_a,
                    game.total_winnings_b,
                    game.winner_seat,
                    game.started_at,
                    game.completed_at,
                    json.dumps(game.metadata, sort_keys=True),
                ),
            )

            for rnd in game.rounds:
                round_id = make_round_id(game.game_id, rnd.round_number)
                conn.execute(
                    """
                    INSERT INTO rounds (
                        round_id, game_id, round_number, stake, first_mover_seat, outcome_label
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        round_id,
                        game.game_id,
                        rnd.round_number,
                        rnd.stake,
                        rnd.first_mover_seat,
                        rnd.outcome_label,
                    ),
                )

                for turn in rnd.turns:
                    conn.execute(
                        """
                        INSERT INTO turns (
                            turn_id, round_id, game_id, round_number, turn_index,
                            player_turn_number, speaker_seat, speaker_id, speaker_name,
                            thought, speech, raw_output, model_name, latency_ms,
                            prompt_tokens, completion_tokens, parse_warning, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            make_turn_id(game.game_id, rnd.round_number, turn.turn_index),
                            round_id,
                            game.game_id,
                            rnd.round_number,
                            turn.turn_index,
                            turn.player_turn_number,
                            turn.speaker_seat,
                            turn.speaker_id,
                            turn.speaker_name,
                            turn.thought,
                            turn.speech,
                            turn.raw_output,
                            turn.model_name,
                            turn.latency_ms,
                            turn.prompt_tokens,
                            turn.completion_tokens,
                            turn.parse_warning,
                            turn.created_at,
                        ),
                    )

                for dec in (rnd.decision_a, rnd.decision_b):
                    conn.execute(
                        """
                        INSERT INTO decisions (
                            decision_id, round_id, game_id, round_number, seat,
                            player_id, player_name, thought, action, payout,
                            bankroll_after, raw_output, model_name, retry_count,
                            parse_failed, parse_error, latency_ms,
                            prompt_tokens, completion_tokens, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            make_decision_id(game.game_id, rnd.round_number, dec.seat),
                            round_id,
                            game.game_id,
                            rnd.round_number,
                            dec.seat,
                            dec.player_id,
                            dec.player_name,
                            dec.thought,
                            dec.action,
                            dec.payout,
                            dec.bankroll_after,
                            dec.raw_output,
                            dec.model_name,
                            dec.retry_count,
                            1 if dec.parse_failed else 0,
                            dec.parse_error,
                            dec.latency_ms,
                            dec.prompt_tokens,
                            dec.completion_tokens,
                            dec.created_at,
                        ),
                    )

        if write_jsonl:
            jsonl_path = self.jsonl_path_for(game.experiment_id)
            with jsonl_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(game.to_dict(), ensure_ascii=False) + "\n")

    def _hydrate_game(self, conn: sqlite3.Connection, row: sqlite3.Row) -> GameRecord:
        game_id = str(row["game_id"])

        round_rows = conn.execute(
            "SELECT * FROM rounds WHERE game_id = ? ORDER BY round_number ASC",
            (game_id,),
        ).fetchall()

        turn_rows = conn.execute(
            "SELECT * FROM turns WHERE game_id = ? ORDER BY round_number ASC, turn_index ASC",
            (game_id,),
        ).fetchall()
        turns_by_round: dict[int, list[TurnRecord]] = {}
        for t in turn_rows:
            rnum = int(t["round_number"])
            turns_by_round.setdefault(rnum, []).append(
                TurnRecord(
                    round_number=rnum,
                    turn_index=int(t["turn_index"]),
                    player_turn_number=int(t["player_turn_number"]),
                    speaker_seat=str(t["speaker_seat"]),
                    speaker_id=str(t["speaker_id"]),
                    speaker_name=str(t["speaker_name"]),
                    thought=str(t["thought"]),
                    speech=str(t["speech"]),
                    raw_output=str(t["raw_output"]),
                    model_name=str(t["model_name"]),
                    latency_ms=float(t["latency_ms"]),
                    prompt_tokens=int(t["prompt_tokens"]),
                    completion_tokens=int(t["completion_tokens"]),
                    parse_warning=t["parse_warning"],
                    created_at=str(t["created_at"]),
                )
            )

        dec_rows = conn.execute(
            "SELECT * FROM decisions WHERE game_id = ? ORDER BY round_number ASC, seat ASC",
            (game_id,),
        ).fetchall()
        decisions_by_round: dict[int, dict[str, DecisionRecord]] = {}
        for d in dec_rows:
            rnum = int(d["round_number"])
            seat = str(d["seat"])
            decisions_by_round.setdefault(rnum, {})[seat] = DecisionRecord(
                round_number=rnum,
                seat=seat,
                player_id=str(d["player_id"]),
                player_name=str(d["player_name"]),
                thought=str(d["thought"]),
                action=str(d["action"]),
                payout=float(d["payout"]),
                bankroll_after=float(d["bankroll_after"]),
                raw_output=str(d["raw_output"]),
                model_name=str(d["model_name"]),
                retry_count=int(d["retry_count"]),
                parse_failed=bool(d["parse_failed"]),
                parse_error=d["parse_error"],
                latency_ms=float(d["latency_ms"]),
                prompt_tokens=int(d["prompt_tokens"]),
                completion_tokens=int(d["completion_tokens"]),
                created_at=str(d["created_at"]),
            )

        rounds: list[RoundRecord] = []
        for r in round_rows:
            rnum = int(r["round_number"])
            dec_map = decisions_by_round[rnum]
            rounds.append(
                RoundRecord(
                    round_number=rnum,
                    stake=int(r["stake"]),
                    first_mover_seat=str(r["first_mover_seat"]),
                    turns=turns_by_round.get(rnum, []),
                    decision_a=dec_map["A"],
                    decision_b=dec_map["B"],
                    outcome_label=str(r["outcome_label"]),
                )
            )

        return GameRecord(
            game_id=game_id,
            experiment_id=str(row["experiment_id"]),
            regime=str(row["regime"]),
            matchup_key=str(row["matchup_key"]),
            game_index=int(row["game_index"]),
            seed=row["seed"],
            stakes=[int(s) for s in json.loads(row["stakes_json"])],
            negotiation_turns_per_round=int(row["negotiation_turns_per_round"]),
            swap_first_mover=str(row["swap_first_mover"]),
            initial_first_mover_seat=str(row["initial_first_mover_seat"]),
            player_a=PlayerConfig.from_dict(json.loads(row["player_a_config_json"])),
            player_b=PlayerConfig.from_dict(json.loads(row["player_b_config_json"])),
            rounds=rounds,
            total_winnings_a=float(row["total_winnings_a"]),
            total_winnings_b=float(row["total_winnings_b"]),
            winner_seat=str(row["winner_seat"]),
            started_at=str(row["started_at"]),
            completed_at=str(row["completed_at"]),
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def load_game(self, game_id: str) -> Optional[GameRecord]:
        """Reconstruct a single full GameRecord from SQLite."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM games WHERE game_id = ?", (game_id,)).fetchone()
            if row is None:
                return None
            return self._hydrate_game(conn, row)

    def load_games(
        self,
        *,
        experiment_id: Optional[str] = None,
        regime: Optional[str] = None,
        matchup_key: Optional[str] = None,
    ) -> list[GameRecord]:
        """Reconstruct all matching GameRecords from SQLite."""
        clauses: list[str] = []
        params: list[Any] = []
        if experiment_id is not None:
            clauses.append("experiment_id = ?")
            params.append(experiment_id)
        if regime is not None:
            clauses.append("regime = ?")
            params.append(regime)
        if matchup_key is not None:
            clauses.append("matchup_key = ?")
            params.append(matchup_key)

        sql = "SELECT * FROM games"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY started_at ASC, game_index ASC"

        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
            return [self._hydrate_game(conn, row) for row in rows]

    def load_experiment(self, experiment_id: str) -> Optional[ExperimentRecord]:
        """Load experiment metadata from SQLite."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
            if row is None:
                return None
            return ExperimentRecord(
                experiment_id=str(row["experiment_id"]),
                name=str(row["name"]),
                regime=str(row["regime"]),
                description=str(row["description"]),
                config=json.loads(row["config_json"] or "{}"),
                created_at=str(row["created_at"]),
            )

    def list_experiments(self) -> list[ExperimentRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM experiments ORDER BY created_at ASC"
            ).fetchall()
            return [
                ExperimentRecord(
                    experiment_id=str(row["experiment_id"]),
                    name=str(row["name"]),
                    regime=str(row["regime"]),
                    description=str(row["description"]),
                    config=json.loads(row["config_json"] or "{}"),
                    created_at=str(row["created_at"]),
                )
                for row in rows
            ]

    def get_table_counts(self) -> dict[str, int]:
        tables = ("experiments", "games", "rounds", "turns", "decisions")
        counts: dict[str, int] = {}
        with self._connect() as conn:
            for table in tables:
                row = conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()
                counts[table] = int(row["c"]) if row else 0
        return counts

    @staticmethod
    def load_games_from_jsonl(jsonl_path: str | Path) -> list[GameRecord]:
        """Load full GameRecords from an append-only JSONL file."""
        path = Path(jsonl_path)
        games: list[GameRecord] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                games.append(GameRecord.from_dict(json.loads(stripped)))
        return games
