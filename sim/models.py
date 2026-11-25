"""Core data models for the Split or Steal simulation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

SPLIT = "SPLIT"
STEAL = "STEAL"
VALID_ACTIONS = frozenset({SPLIT, STEAL})


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp string with millisecond precision."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def normalize_swap_mode(value: bool | str) -> str:
    """Normalize swap_first_mover into one of: 'none', 'per_round', 'per_game', 'both'."""
    if isinstance(value, bool):
        return "both" if value else "none"
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "both"}:
        return "both"
    if normalized in {"false", "0", "no", "none", "off"}:
        return "none"
    if normalized in {"per_round", "round"}:
        return "per_round"
    if normalized in {"per_game", "game"}:
        return "per_game"
    raise ValueError(
        f"Invalid swap_first_mover mode: {value!r}. "
        "Expected bool or one of 'none', 'per_round', 'per_game', 'both'."
    )


def canonical_matchup_key(player_a_id: str, player_b_id: str) -> str:
    """Build a canonical order-independent matchup identifier."""
    left, right = sorted([player_a_id, player_b_id])
    return f"{left}__vs__{right}"


@dataclass(slots=True)
class PlayerConfig:
    """Configuration for a single character seat in a simulation."""

    character_id: str
    display_name: str
    model_name: str
    regime: str = "prompt_only"
    system_prompt: str = ""
    adapter_name: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 512
    extra_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "character_id": self.character_id,
            "display_name": self.display_name,
            "model_name": self.model_name,
            "regime": self.regime,
            "system_prompt": self.system_prompt,
            "adapter_name": self.adapter_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "extra_metadata": dict(self.extra_metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PlayerConfig":
        return cls(
            character_id=str(data["character_id"]),
            display_name=str(data.get("display_name", data["character_id"])),
            model_name=str(data["model_name"]),
            regime=str(data.get("regime", "prompt_only")),
            system_prompt=str(data.get("system_prompt", "")),
            adapter_name=data.get("adapter_name"),
            temperature=float(data.get("temperature", 0.7)),
            max_tokens=int(data.get("max_tokens", 512)),
            extra_metadata=dict(data.get("extra_metadata") or {}),
        )


@dataclass(slots=True)
class GameConfig:
    """Configurable rules and metadata for a Split or Steal game or tournament."""

    stakes: tuple[int, ...] = (100, 500, 1000)
    negotiation_turns_per_round: int = 2
    swap_first_mover: bool | str = True
    max_decision_retries: int = 1
    fallback_action_on_parse_failure: str = SPLIT
    experiment_id: str = "default"
    regime: str = "prompt_only"
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        stakes_tuple = tuple(int(s) for s in self.stakes)
        if not stakes_tuple:
            raise ValueError("stakes must contain at least one round stake.")
        if any(s <= 0 for s in stakes_tuple):
            raise ValueError(f"All stakes must be positive integers, got {stakes_tuple}.")
        self.stakes = stakes_tuple

        if self.negotiation_turns_per_round < 0:
            raise ValueError("negotiation_turns_per_round must be >= 0.")
        if self.max_decision_retries < 0:
            raise ValueError("max_decision_retries must be >= 0.")

        fallback = self.fallback_action_on_parse_failure.strip().upper()
        if fallback not in VALID_ACTIONS:
            raise ValueError(
                f"fallback_action_on_parse_failure must be SPLIT or STEAL, got {fallback!r}."
            )
        self.fallback_action_on_parse_failure = fallback
        self.swap_first_mover = normalize_swap_mode(self.swap_first_mover)

    @property
    def swap_mode(self) -> str:
        return str(self.swap_first_mover)

    @property
    def swaps_within_game(self) -> bool:
        return self.swap_mode in {"per_round", "both"}

    @property
    def swaps_across_games(self) -> bool:
        return self.swap_mode in {"per_game", "both"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "stakes": list(self.stakes),
            "negotiation_turns_per_round": self.negotiation_turns_per_round,
            "swap_first_mover": self.swap_mode,
            "max_decision_retries": self.max_decision_retries,
            "fallback_action_on_parse_failure": self.fallback_action_on_parse_failure,
            "experiment_id": self.experiment_id,
            "regime": self.regime,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameConfig":
        return cls(
            stakes=tuple(int(s) for s in data.get("stakes", (100, 500, 1000))),
            negotiation_turns_per_round=int(data.get("negotiation_turns_per_round", 2)),
            swap_first_mover=data.get("swap_first_mover", True),
            max_decision_retries=int(data.get("max_decision_retries", 1)),
            fallback_action_on_parse_failure=str(
                data.get("fallback_action_on_parse_failure", SPLIT)
            ),
            experiment_id=str(data.get("experiment_id", "default")),
            regime=str(data.get("regime", "prompt_only")),
            seed=data.get("seed"),
        )


@dataclass(slots=True)
class LLMResponse:
    """Normalized response returned by any ModelBackend."""

    raw_text: str
    model: str
    latency_ms: float
    thought: str = ""
    content: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    finish_reason: str = "stop"


@dataclass(slots=True)
class TurnRecord:
    """One negotiation message (with private thought and visible speech)."""

    round_number: int
    turn_index: int
    player_turn_number: int
    speaker_seat: str
    speaker_id: str
    speaker_name: str
    thought: str
    speech: str
    raw_output: str
    model_name: str
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    parse_warning: Optional[str] = None
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_number": self.round_number,
            "turn_index": self.turn_index,
            "player_turn_number": self.player_turn_number,
            "speaker_seat": self.speaker_seat,
            "speaker_id": self.speaker_id,
            "speaker_name": self.speaker_name,
            "thought": self.thought,
            "speech": self.speech,
            "raw_output": self.raw_output,
            "model_name": self.model_name,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "parse_warning": self.parse_warning,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TurnRecord":
        return cls(
            round_number=int(data["round_number"]),
            turn_index=int(data["turn_index"]),
            player_turn_number=int(data["player_turn_number"]),
            speaker_seat=str(data["speaker_seat"]),
            speaker_id=str(data["speaker_id"]),
            speaker_name=str(data["speaker_name"]),
            thought=str(data.get("thought", "")),
            speech=str(data.get("speech", "")),
            raw_output=str(data.get("raw_output", "")),
            model_name=str(data.get("model_name", "")),
            latency_ms=float(data.get("latency_ms", 0.0)),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            parse_warning=data.get("parse_warning"),
            created_at=str(data.get("created_at", "")),
        )


@dataclass(slots=True)
class DecisionRecord:
    """One player's locked-in simultaneous decision for a round."""

    round_number: int
    seat: str
    player_id: str
    player_name: str
    thought: str
    action: str
    payout: float
    bankroll_after: float
    raw_output: str
    model_name: str
    retry_count: int = 0
    parse_failed: bool = False
    parse_error: Optional[str] = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_number": self.round_number,
            "seat": self.seat,
            "player_id": self.player_id,
            "player_name": self.player_name,
            "thought": self.thought,
            "action": self.action,
            "payout": self.payout,
            "bankroll_after": self.bankroll_after,
            "raw_output": self.raw_output,
            "model_name": self.model_name,
            "retry_count": self.retry_count,
            "parse_failed": self.parse_failed,
            "parse_error": self.parse_error,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionRecord":
        return cls(
            round_number=int(data["round_number"]),
            seat=str(data["seat"]),
            player_id=str(data["player_id"]),
            player_name=str(data["player_name"]),
            thought=str(data.get("thought", "")),
            action=str(data["action"]),
            payout=float(data["payout"]),
            bankroll_after=float(data["bankroll_after"]),
            raw_output=str(data.get("raw_output", "")),
            model_name=str(data.get("model_name", "")),
            retry_count=int(data.get("retry_count", 0)),
            parse_failed=bool(data.get("parse_failed", False)),
            parse_error=data.get("parse_error"),
            latency_ms=float(data.get("latency_ms", 0.0)),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            created_at=str(data.get("created_at", "")),
        )


@dataclass(slots=True)
class RoundRecord:
    """Complete record of a single Split or Steal round."""

    round_number: int
    stake: int
    first_mover_seat: str
    turns: list[TurnRecord]
    decision_a: DecisionRecord
    decision_b: DecisionRecord
    outcome_label: str

    def decision_for_seat(self, seat: str) -> DecisionRecord:
        if seat == "A":
            return self.decision_a
        if seat == "B":
            return self.decision_b
        raise ValueError(f"Unknown seat: {seat!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "round_number": self.round_number,
            "stake": self.stake,
            "first_mover_seat": self.first_mover_seat,
            "turns": [t.to_dict() for t in self.turns],
            "decision_a": self.decision_a.to_dict(),
            "decision_b": self.decision_b.to_dict(),
            "outcome_label": self.outcome_label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RoundRecord":
        return cls(
            round_number=int(data["round_number"]),
            stake=int(data["stake"]),
            first_mover_seat=str(data["first_mover_seat"]),
            turns=[TurnRecord.from_dict(t) for t in data.get("turns", [])],
            decision_a=DecisionRecord.from_dict(data["decision_a"]),
            decision_b=DecisionRecord.from_dict(data["decision_b"]),
            outcome_label=str(data["outcome_label"]),
        )


@dataclass(slots=True)
class GameRecord:
    """Complete replayable record of a multi-round Split or Steal game."""

    game_id: str
    experiment_id: str
    regime: str
    matchup_key: str
    game_index: int
    seed: Optional[int]
    stakes: list[int]
    negotiation_turns_per_round: int
    swap_first_mover: str
    initial_first_mover_seat: str
    player_a: PlayerConfig
    player_b: PlayerConfig
    rounds: list[RoundRecord]
    total_winnings_a: float
    total_winnings_b: float
    winner_seat: str
    started_at: str
    completed_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "experiment_id": self.experiment_id,
            "regime": self.regime,
            "matchup_key": self.matchup_key,
            "game_index": self.game_index,
            "seed": self.seed,
            "stakes": list(self.stakes),
            "negotiation_turns_per_round": self.negotiation_turns_per_round,
            "swap_first_mover": self.swap_first_mover,
            "initial_first_mover_seat": self.initial_first_mover_seat,
            "player_a": self.player_a.to_dict(),
            "player_b": self.player_b.to_dict(),
            "rounds": [r.to_dict() for r in self.rounds],
            "total_winnings_a": self.total_winnings_a,
            "total_winnings_b": self.total_winnings_b,
            "winner_seat": self.winner_seat,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameRecord":
        return cls(
            game_id=str(data["game_id"]),
            experiment_id=str(data["experiment_id"]),
            regime=str(data["regime"]),
            matchup_key=str(data["matchup_key"]),
            game_index=int(data.get("game_index", 0)),
            seed=data.get("seed"),
            stakes=[int(s) for s in data["stakes"]],
            negotiation_turns_per_round=int(data["negotiation_turns_per_round"]),
            swap_first_mover=str(data["swap_first_mover"]),
            initial_first_mover_seat=str(data["initial_first_mover_seat"]),
            player_a=PlayerConfig.from_dict(data["player_a"]),
            player_b=PlayerConfig.from_dict(data["player_b"]),
            rounds=[RoundRecord.from_dict(r) for r in data.get("rounds", [])],
            total_winnings_a=float(data["total_winnings_a"]),
            total_winnings_b=float(data["total_winnings_b"]),
            winner_seat=str(data["winner_seat"]),
            started_at=str(data["started_at"]),
            completed_at=str(data["completed_at"]),
            metadata=dict(data.get("metadata") or {}),
        )


@dataclass(slots=True)
class ExperimentRecord:
    """Top-level metadata for a simulation experiment / tournament run."""

    experiment_id: str
    name: str
    regime: str
    description: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "regime": self.regime,
            "description": self.description,
            "config": dict(self.config),
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentRecord":
        return cls(
            experiment_id=str(data["experiment_id"]),
            name=str(data["name"]),
            regime=str(data["regime"]),
            description=str(data.get("description", "")),
            config=dict(data.get("config") or {}),
            created_at=str(data.get("created_at", "")),
        )
