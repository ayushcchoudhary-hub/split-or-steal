"""Core 3-Round Escalating Split or Steal simulation engine."""

from __future__ import annotations

import uuid
from typing import Any, Optional, Sequence

from sim.backends import ModelBackend
from sim.models import (
    DecisionRecord,
    GameConfig,
    GameRecord,
    PlayerConfig,
    RoundRecord,
    SPLIT,
    STEAL,
    TurnRecord,
    VALID_ACTIONS,
    canonical_matchup_key,
    utc_now_iso,
)
from sim.parser import parse_decision_output, parse_negotiation_output
from sim.prompts import (
    build_decision_messages,
    build_decision_retry_messages,
    build_negotiation_messages,
)


def resolve_payout(action_a: str, action_b: str, stake: int) -> tuple[float, float, str]:
    """Resolve payouts for a single Split or Steal round.

    Rules for pot $S:
    - SPLIT / SPLIT -> ($S/2, $S/2, "MUTUAL_SPLIT")
    - STEAL / SPLIT -> ($S, $0, "A_STOLE")
    - SPLIT / STEAL -> ($0, $S, "B_STOLE")
    - STEAL / STEAL -> ($0, $0, "MUTUAL_STEAL")
    """
    a = action_a.strip().upper()
    b = action_b.strip().upper()
    if a not in VALID_ACTIONS or b not in VALID_ACTIONS:
        raise ValueError(f"Invalid actions for payout resolution: ({action_a!r}, {action_b!r})")
    if stake <= 0:
        raise ValueError(f"Stake must be positive, got {stake}")

    if a == SPLIT and b == SPLIT:
        half = stake / 2.0
        return half, half, "MUTUAL_SPLIT"
    if a == STEAL and b == SPLIT:
        return float(stake), 0.0, "A_STOLE"
    if a == SPLIT and b == STEAL:
        return 0.0, float(stake), "B_STOLE"
    return 0.0, 0.0, "MUTUAL_STEAL"


class SplitOrStealEngine:
    """Configurable multi-round Split or Steal state machine & tournament runner."""

    def __init__(
        self,
        backend: ModelBackend,
        *,
        seat_backends: Optional[dict[str, ModelBackend]] = None,
    ) -> None:
        self.backend = backend
        self.seat_backends = dict(seat_backends or {})

    def _backend_for(self, seat: str, player: PlayerConfig) -> ModelBackend:
        if player.character_id in self.seat_backends:
            return self.seat_backends[player.character_id]
        if seat in self.seat_backends:
            return self.seat_backends[seat]
        return self.backend

    @staticmethod
    def _round_first_mover(initial_first_mover: str, round_index_0: int, swaps_within_game: bool) -> str:
        other = "B" if initial_first_mover == "A" else "A"
        if not swaps_within_game:
            return initial_first_mover
        return initial_first_mover if (round_index_0 % 2 == 0) else other

    @staticmethod
    def _public_context(
        *,
        round_number: int,
        total_rounds: int,
        stake: int,
        seat: str,
        prior_rounds: Sequence[RoundRecord],
        current_round_turns: Sequence[TurnRecord],
        player_bankroll: float,
        opponent_bankroll: float,
        seed: Optional[int],
        player_turn_number: Optional[int] = None,
    ) -> dict[str, Any]:
        """Build strictly public context for backends (NEVER contains opponent thoughts)."""
        opp_seat = "B" if seat == "A" else "A"
        return {
            "round_number": round_number,
            "total_rounds": total_rounds,
            "stake": stake,
            "seat": seat,
            "player_turn_number": player_turn_number,
            "player_bankroll": player_bankroll,
            "opponent_bankroll": opponent_bankroll,
            "self_prior_actions": [r.decision_for_seat(seat).action for r in prior_rounds],
            "opponent_prior_actions": [r.decision_for_seat(opp_seat).action for r in prior_rounds],
            "visible_speeches": [
                {"speaker_seat": t.speaker_seat, "speaker_name": t.speaker_name, "speech": t.speech}
                for t in current_round_turns
            ],
            "seed": seed,
        }

    def _run_decision_for_seat(
        self,
        *,
        seat: str,
        player: PlayerConfig,
        opponent: PlayerConfig,
        config: GameConfig,
        round_number: int,
        stake: int,
        prior_rounds: Sequence[RoundRecord],
        current_round_turns: Sequence[TurnRecord],
        player_bankroll: float,
        opponent_bankroll: float,
        game_seed: Optional[int],
    ) -> tuple[str, str, str, int, bool, Optional[str], float, int, int, str]:
        """Prompt one player for a locked-in SPLIT/STEAL decision with automatic retries."""
        messages = build_decision_messages(
            player=player,
            opponent_name=opponent.display_name,
            config=config,
            round_number=round_number,
            stake=stake,
            prior_rounds=prior_rounds,
            current_round_turns=current_round_turns,
            player_seat=seat,
            player_bankroll=player_bankroll,
            opponent_bankroll=opponent_bankroll,
        )
        ctx = self._public_context(
            round_number=round_number,
            total_rounds=len(config.stakes),
            stake=stake,
            seat=seat,
            prior_rounds=prior_rounds,
            current_round_turns=current_round_turns,
            player_bankroll=player_bankroll,
            opponent_bankroll=opponent_bankroll,
            seed=game_seed,
        )
        backend = self._backend_for(seat, player)

        total_latency_ms = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        retry_count = 0
        current_messages = messages
        last_raw = ""
        last_thought = ""
        last_error: Optional[str] = None
        used_model = player.model_name

        for attempt in range(config.max_decision_retries + 1):
            resp = backend.generate(
                messages=current_messages,
                player=player,
                phase="decision" if attempt == 0 else "decision_retry",
                context=ctx,
            )
            used_model = resp.model or player.model_name
            total_latency_ms += resp.latency_ms
            total_prompt_tokens += resp.prompt_tokens
            total_completion_tokens += resp.completion_tokens
            last_raw = resp.raw_text

            parsed = parse_decision_output(resp.raw_text, reasoning_content=resp.thought or None)
            last_thought = parsed.thought
            if parsed.is_valid and parsed.action is not None:
                return (
                    parsed.action,
                    last_thought,
                    last_raw,
                    retry_count,
                    False,
                    None,
                    round(total_latency_ms, 3),
                    total_prompt_tokens,
                    total_completion_tokens,
                    used_model,
                )

            last_error = parsed.parse_error or "Unknown decision parse failure"
            if attempt < config.max_decision_retries:
                retry_count += 1
                current_messages = build_decision_retry_messages(
                    messages,
                    previous_raw_output=resp.raw_text,
                    parse_error=last_error,
                )

        return (
            config.fallback_action_on_parse_failure,
            last_thought,
            last_raw,
            retry_count,
            True,
            last_error,
            round(total_latency_ms, 3),
            total_prompt_tokens,
            total_completion_tokens,
            used_model,
        )

    def play_game(
        self,
        player_a: PlayerConfig,
        player_b: PlayerConfig,
        config: GameConfig,
        *,
        game_index: int = 0,
        initial_first_mover: str = "A",
        metadata: Optional[dict[str, Any]] = None,
    ) -> GameRecord:
        """Play one complete multi-round Split or Steal game."""
        if initial_first_mover not in {"A", "B"}:
            raise ValueError(f"initial_first_mover must be 'A' or 'B', got {initial_first_mover!r}")

        game_id = str(uuid.uuid4())
        started_at = utc_now_iso()
        game_seed = (config.seed + game_index) if config.seed is not None else None

        bankroll_a = 0.0
        bankroll_b = 0.0
        rounds: list[RoundRecord] = []
        total_rounds = len(config.stakes)

        players_by_seat = {"A": player_a, "B": player_b}
        other_seat = {"A": "B", "B": "A"}

        for r_idx, stake in enumerate(config.stakes):
            round_number = r_idx + 1
            first_mover = self._round_first_mover(
                initial_first_mover, r_idx, config.swaps_within_game
            )
            second_mover = other_seat[first_mover]
            speaking_order = (first_mover, second_mover)

            round_turns: list[TurnRecord] = []
            turn_index = 0

            # Phase 1: Negotiation turns (configurable turns per player)
            for p_turn in range(1, config.negotiation_turns_per_round + 1):
                for seat in speaking_order:
                    speaker = players_by_seat[seat]
                    opp = players_by_seat[other_seat[seat]]
                    my_bankroll = bankroll_a if seat == "A" else bankroll_b
                    opp_bankroll = bankroll_b if seat == "A" else bankroll_a

                    messages = build_negotiation_messages(
                        player=speaker,
                        opponent_name=opp.display_name,
                        config=config,
                        round_number=round_number,
                        stake=stake,
                        prior_rounds=rounds,
                        current_round_turns=round_turns,
                        player_seat=seat,
                        player_bankroll=my_bankroll,
                        opponent_bankroll=opp_bankroll,
                        player_turn_number=p_turn,
                    )
                    ctx = self._public_context(
                        round_number=round_number,
                        total_rounds=total_rounds,
                        stake=stake,
                        seat=seat,
                        prior_rounds=rounds,
                        current_round_turns=round_turns,
                        player_bankroll=my_bankroll,
                        opponent_bankroll=opp_bankroll,
                        seed=game_seed,
                        player_turn_number=p_turn,
                    )
                    resp = self._backend_for(seat, speaker).generate(
                        messages=messages,
                        player=speaker,
                        phase="negotiation",
                        context=ctx,
                    )
                    parsed = parse_negotiation_output(
                        resp.raw_text, reasoning_content=resp.thought or None
                    )
                    round_turns.append(
                        TurnRecord(
                            round_number=round_number,
                            turn_index=turn_index,
                            player_turn_number=p_turn,
                            speaker_seat=seat,
                            speaker_id=speaker.character_id,
                            speaker_name=speaker.display_name,
                            thought=parsed.thought,
                            speech=parsed.speech,
                            raw_output=resp.raw_text,
                            model_name=resp.model or speaker.model_name,
                            latency_ms=resp.latency_ms,
                            prompt_tokens=resp.prompt_tokens,
                            completion_tokens=resp.completion_tokens,
                            parse_warning=parsed.parse_warning,
                        )
                    )
                    turn_index += 1

            # Phase 2: Simultaneous Decisions (neither sees the other's choice or thought)
            dec_a_tuple = self._run_decision_for_seat(
                seat="A",
                player=player_a,
                opponent=player_b,
                config=config,
                round_number=round_number,
                stake=stake,
                prior_rounds=rounds,
                current_round_turns=round_turns,
                player_bankroll=bankroll_a,
                opponent_bankroll=bankroll_b,
                game_seed=game_seed,
            )
            dec_b_tuple = self._run_decision_for_seat(
                seat="B",
                player=player_b,
                opponent=player_a,
                config=config,
                round_number=round_number,
                stake=stake,
                prior_rounds=rounds,
                current_round_turns=round_turns,
                player_bankroll=bankroll_b,
                opponent_bankroll=bankroll_a,
                game_seed=game_seed,
            )

            # Phase 3: Payout resolution & bankroll update
            payout_a, payout_b, outcome_label = resolve_payout(
                dec_a_tuple[0], dec_b_tuple[0], stake
            )
            bankroll_a += payout_a
            bankroll_b += payout_b

            decision_a = DecisionRecord(
                round_number=round_number,
                seat="A",
                player_id=player_a.character_id,
                player_name=player_a.display_name,
                thought=dec_a_tuple[1],
                action=dec_a_tuple[0],
                payout=payout_a,
                bankroll_after=bankroll_a,
                raw_output=dec_a_tuple[2],
                retry_count=dec_a_tuple[3],
                parse_failed=dec_a_tuple[4],
                parse_error=dec_a_tuple[5],
                latency_ms=dec_a_tuple[6],
                prompt_tokens=dec_a_tuple[7],
                completion_tokens=dec_a_tuple[8],
                model_name=dec_a_tuple[9],
            )
            decision_b = DecisionRecord(
                round_number=round_number,
                seat="B",
                player_id=player_b.character_id,
                player_name=player_b.display_name,
                thought=dec_b_tuple[1],
                action=dec_b_tuple[0],
                payout=payout_b,
                bankroll_after=bankroll_b,
                raw_output=dec_b_tuple[2],
                retry_count=dec_b_tuple[3],
                parse_failed=dec_b_tuple[4],
                parse_error=dec_b_tuple[5],
                latency_ms=dec_b_tuple[6],
                prompt_tokens=dec_b_tuple[7],
                completion_tokens=dec_b_tuple[8],
                model_name=dec_b_tuple[9],
            )

            rounds.append(
                RoundRecord(
                    round_number=round_number,
                    stake=stake,
                    first_mover_seat=first_mover,
                    turns=round_turns,
                    decision_a=decision_a,
                    decision_b=decision_b,
                    outcome_label=outcome_label,
                )
            )

        if bankroll_a > bankroll_b:
            winner_seat = "A"
        elif bankroll_b > bankroll_a:
            winner_seat = "B"
        else:
            winner_seat = "TIE"

        return GameRecord(
            game_id=game_id,
            experiment_id=config.experiment_id,
            regime=config.regime,
            matchup_key=canonical_matchup_key(player_a.character_id, player_b.character_id),
            game_index=game_index,
            seed=game_seed,
            stakes=list(config.stakes),
            negotiation_turns_per_round=config.negotiation_turns_per_round,
            swap_first_mover=config.swap_mode,
            initial_first_mover_seat=initial_first_mover,
            player_a=player_a,
            player_b=player_b,
            rounds=rounds,
            total_winnings_a=bankroll_a,
            total_winnings_b=bankroll_b,
            winner_seat=winner_seat,
            started_at=started_at,
            completed_at=utc_now_iso(),
            metadata=dict(metadata or {}),
        )

    def run_matchup(
        self,
        player_a: PlayerConfig,
        player_b: PlayerConfig,
        config: GameConfig,
        num_games: int,
        *,
        storage: Optional[Any] = None,
    ) -> list[GameRecord]:
        """Run N games for a matchup with automatic 50/50 first-mover balancing."""
        if num_games <= 0:
            raise ValueError(f"num_games must be positive, got {num_games}")

        games: list[GameRecord] = []
        for idx in range(num_games):
            if config.swaps_across_games:
                initial_first = "A" if (idx % 2 == 0) else "B"
            else:
                initial_first = "A"

            game = self.play_game(
                player_a=player_a,
                player_b=player_b,
                config=config,
                game_index=idx,
                initial_first_mover=initial_first,
            )
            if storage is not None:
                storage.save_game(game)
            games.append(game)

        return games

    def run_tournament(
        self,
        players: Sequence[PlayerConfig],
        config: GameConfig,
        games_per_matchup: int = 2,
        *,
        include_mirrors: bool = True,
        storage: Optional[Any] = None,
    ) -> list[GameRecord]:
        """Run a round-robin tournament across all players (optionally with mirror matches)."""
        if not players:
            raise ValueError("players must contain at least one PlayerConfig")

        all_games: list[GameRecord] = []
        n = len(players)
        for i in range(n):
            start_j = i if include_mirrors else i + 1
            for j in range(start_j, n):
                matchup_games = self.run_matchup(
                    player_a=players[i],
                    player_b=players[j],
                    config=config,
                    num_games=games_per_matchup,
                    storage=storage,
                )
                all_games.extend(matchup_games)
        return all_games
