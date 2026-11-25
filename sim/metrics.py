"""Exact behavioral analytics & scorecard reporting for Split or Steal."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from sim.models import GameRecord, SPLIT, STEAL
from sim.parser import detect_cooperation_promise


@dataclass(slots=True)
class BehaviorMetrics:
    """Aggregated game-theoretic and linguistic behavioral metrics."""

    label: str
    games_played: int = 0
    total_decisions: int = 0
    split_decisions: int = 0
    steal_decisions: int = 0
    total_winnings: float = 0.0
    avg_winnings_per_game: float = 0.0
    split_rate_overall: float = 0.0
    steal_rate_by_round: dict[int, float] = field(default_factory=dict)

    # First-strike betrayal: stole when opponent had NEVER stolen in any prior round
    first_strike_betrayals: int = 0
    first_strike_opportunities: int = 0
    first_strike_betrayal_rate: Optional[float] = None

    # Revenge & forgiveness: response in round r after opponent stole in round r-1
    revenge_count: int = 0
    forgiveness_count: int = 0
    revenge_opportunities: int = 0
    revenge_rate: Optional[float] = None
    forgiveness_rate: Optional[float] = None

    # Endgame backstab: split R1 & R2 (all prior rounds in >=3 round game), then stole R3/final
    endgame_backstab_count: int = 0
    endgame_backstab_opportunities: int = 0
    endgame_backstab_rate: Optional[float] = None
    endgame_backstab_game_rate: Optional[float] = None

    # Deception gap: promised cooperation/split in speech, yet chose STEAL
    cooperation_promises: int = 0
    deceptive_steals: int = 0
    deception_gap: Optional[float] = None
    deception_round_rate: float = 0.0

    # Operational & linguistic metrics
    parse_failures: int = 0
    parse_failure_rate: float = 0.0
    total_speeches: int = 0
    total_speech_words: int = 0
    avg_speech_words: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "games_played": self.games_played,
            "total_decisions": self.total_decisions,
            "split_decisions": self.split_decisions,
            "steal_decisions": self.steal_decisions,
            "total_winnings": round(self.total_winnings, 4),
            "avg_winnings_per_game": round(self.avg_winnings_per_game, 4),
            "split_rate_overall": round(self.split_rate_overall, 6),
            "steal_rate_by_round": {
                f"R{r}": round(rate, 6) for r, rate in sorted(self.steal_rate_by_round.items())
            },
            "first_strike_betrayals": self.first_strike_betrayals,
            "first_strike_opportunities": self.first_strike_opportunities,
            "first_strike_betrayal_rate": _round_opt(self.first_strike_betrayal_rate),
            "revenge_count": self.revenge_count,
            "forgiveness_count": self.forgiveness_count,
            "revenge_opportunities": self.revenge_opportunities,
            "revenge_rate": _round_opt(self.revenge_rate),
            "forgiveness_rate": _round_opt(self.forgiveness_rate),
            "endgame_backstab_count": self.endgame_backstab_count,
            "endgame_backstab_opportunities": self.endgame_backstab_opportunities,
            "endgame_backstab_rate": _round_opt(self.endgame_backstab_rate),
            "endgame_backstab_game_rate": _round_opt(self.endgame_backstab_game_rate),
            "cooperation_promises": self.cooperation_promises,
            "deceptive_steals": self.deceptive_steals,
            "deception_gap": _round_opt(self.deception_gap),
            "deception_round_rate": round(self.deception_round_rate, 6),
            "parse_failures": self.parse_failures,
            "parse_failure_rate": round(self.parse_failure_rate, 6),
            "total_speeches": self.total_speeches,
            "total_speech_words": self.total_speech_words,
            "avg_speech_words": round(self.avg_speech_words, 4),
        }


def _round_opt(value: Optional[float], digits: int = 6) -> Optional[float]:
    return None if value is None else round(value, digits)


class _MetricsAccumulator:
    """Mutable accumulator for streaming player-game appearances into BehaviorMetrics."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.games_played = 0
        self.games_with_3plus_rounds = 0
        self.total_winnings = 0.0

        self.total_decisions = 0
        self.split_decisions = 0
        self.steal_decisions = 0
        self.round_steals: dict[int, int] = {}
        self.round_totals: dict[int, int] = {}

        self.first_strike_betrayals = 0
        self.first_strike_opportunities = 0

        self.revenge_count = 0
        self.forgiveness_count = 0
        self.revenge_opportunities = 0

        self.endgame_backstab_count = 0
        self.endgame_backstab_opportunities = 0

        self.cooperation_promises = 0
        self.deceptive_steals = 0

        self.parse_failures = 0
        self.total_speeches = 0
        self.total_speech_words = 0

    def add_appearance(self, game: GameRecord, seat: str) -> None:
        opp_seat = "B" if seat == "A" else "A"
        self.games_played += 1
        self.total_winnings += game.total_winnings_a if seat == "A" else game.total_winnings_b

        opp_has_stolen_before = False
        opp_prev_action: Optional[str] = None
        actions_by_round: dict[int, str] = {}

        for rnd in sorted(game.rounds, key=lambda r: r.round_number):
            rnum = rnd.round_number
            my_dec = rnd.decision_for_seat(seat)
            opp_dec = rnd.decision_for_seat(opp_seat)

            self.total_decisions += 1
            self.round_totals[rnum] = self.round_totals.get(rnum, 0) + 1
            actions_by_round[rnum] = my_dec.action

            if my_dec.action == SPLIT:
                self.split_decisions += 1
            else:
                self.steal_decisions += 1
                self.round_steals[rnum] = self.round_steals.get(rnum, 0) + 1

            if my_dec.parse_failed:
                self.parse_failures += 1

            # First-strike betrayal opportunity: opponent has NEVER stolen in any prior round
            if not opp_has_stolen_before:
                self.first_strike_opportunities += 1
                if my_dec.action == STEAL:
                    self.first_strike_betrayals += 1

            # Revenge / forgiveness opportunity: opponent stole in immediately preceding round (r-1)
            if opp_prev_action == STEAL:
                self.revenge_opportunities += 1
                if my_dec.action == STEAL:
                    self.revenge_count += 1
                else:
                    self.forgiveness_count += 1

            # Speech word stats & deception gap for this round
            my_speeches = [t.speech for t in rnd.turns if t.speaker_seat == seat]
            for speech in my_speeches:
                self.total_speeches += 1
                self.total_speech_words += len(speech.split())

            if detect_cooperation_promise(my_speeches):
                self.cooperation_promises += 1
                if my_dec.action == STEAL:
                    self.deceptive_steals += 1

            # Update opponent history for subsequent rounds
            if opp_dec.action == STEAL:
                opp_has_stolen_before = True
            opp_prev_action = opp_dec.action

        # Endgame backstab: in a 3+ round game, split R1 & R2, then stole R3
        if 1 in actions_by_round and 2 in actions_by_round and 3 in actions_by_round:
            self.games_with_3plus_rounds += 1
            if actions_by_round[1] == SPLIT and actions_by_round[2] == SPLIT:
                self.endgame_backstab_opportunities += 1
                if actions_by_round[3] == STEAL:
                    self.endgame_backstab_count += 1

    def finalize(self) -> BehaviorMetrics:
        steal_by_round = {
            r: (self.round_steals.get(r, 0) / total)
            for r, total in sorted(self.round_totals.items())
            if total > 0
        }
        return BehaviorMetrics(
            label=self.label,
            games_played=self.games_played,
            total_decisions=self.total_decisions,
            split_decisions=self.split_decisions,
            steal_decisions=self.steal_decisions,
            total_winnings=self.total_winnings,
            avg_winnings_per_game=(
                self.total_winnings / self.games_played if self.games_played else 0.0
            ),
            split_rate_overall=(
                self.split_decisions / self.total_decisions if self.total_decisions else 0.0
            ),
            steal_rate_by_round=steal_by_round,
            first_strike_betrayals=self.first_strike_betrayals,
            first_strike_opportunities=self.first_strike_opportunities,
            first_strike_betrayal_rate=(
                self.first_strike_betrayals / self.first_strike_opportunities
                if self.first_strike_opportunities
                else None
            ),
            revenge_count=self.revenge_count,
            forgiveness_count=self.forgiveness_count,
            revenge_opportunities=self.revenge_opportunities,
            revenge_rate=(
                self.revenge_count / self.revenge_opportunities
                if self.revenge_opportunities
                else None
            ),
            forgiveness_rate=(
                self.forgiveness_count / self.revenge_opportunities
                if self.revenge_opportunities
                else None
            ),
            endgame_backstab_count=self.endgame_backstab_count,
            endgame_backstab_opportunities=self.endgame_backstab_opportunities,
            endgame_backstab_rate=(
                self.endgame_backstab_count / self.endgame_backstab_opportunities
                if self.endgame_backstab_opportunities
                else None
            ),
            endgame_backstab_game_rate=(
                self.endgame_backstab_count / self.games_with_3plus_rounds
                if self.games_with_3plus_rounds
                else None
            ),
            cooperation_promises=self.cooperation_promises,
            deceptive_steals=self.deceptive_steals,
            deception_gap=(
                self.deceptive_steals / self.cooperation_promises
                if self.cooperation_promises
                else None
            ),
            deception_round_rate=(
                self.deceptive_steals / self.total_decisions if self.total_decisions else 0.0
            ),
            parse_failures=self.parse_failures,
            parse_failure_rate=(
                self.parse_failures / self.total_decisions if self.total_decisions else 0.0
            ),
            total_speeches=self.total_speeches,
            total_speech_words=self.total_speech_words,
            avg_speech_words=(
                self.total_speech_words / self.total_speeches if self.total_speeches else 0.0
            ),
        )


@dataclass(slots=True)
class AnalyticsReport:
    """Complete behavioral scorecard across characters, matchups, and regimes."""

    total_games: int
    overall: BehaviorMetrics
    by_character: dict[str, BehaviorMetrics]
    by_regime: dict[str, BehaviorMetrics]
    by_character_regime: dict[str, dict[str, BehaviorMetrics]]
    by_matchup: dict[str, dict[str, BehaviorMetrics]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_games": self.total_games,
            "overall": self.overall.to_dict(),
            "by_character": {k: v.to_dict() for k, v in sorted(self.by_character.items())},
            "by_regime": {k: v.to_dict() for k, v in sorted(self.by_regime.items())},
            "by_character_regime": {
                reg: {cid: m.to_dict() for cid, m in sorted(chars.items())}
                for reg, chars in sorted(self.by_character_regime.items())
            },
            "by_matchup": {
                mk: {cid: m.to_dict() for cid, m in sorted(chars.items())}
                for mk, chars in sorted(self.by_matchup.items())
            },
        }


def compute_metrics(games: Sequence[GameRecord]) -> AnalyticsReport:
    """Compute exact per-character, per-matchup, and per-regime behavioral metrics."""
    overall_acc = _MetricsAccumulator("overall")
    char_accs: dict[str, _MetricsAccumulator] = {}
    regime_accs: dict[str, _MetricsAccumulator] = {}
    char_regime_accs: dict[str, dict[str, _MetricsAccumulator]] = {}
    matchup_accs: dict[str, dict[str, _MetricsAccumulator]] = {}

    for game in games:
        for seat, player in (("A", game.player_a), ("B", game.player_b)):
            cid = player.character_id
            regime = game.regime
            mkey = game.matchup_key

            overall_acc.add_appearance(game, seat)

            if cid not in char_accs:
                char_accs[cid] = _MetricsAccumulator(cid)
            char_accs[cid].add_appearance(game, seat)

            if regime not in regime_accs:
                regime_accs[regime] = _MetricsAccumulator(regime)
            regime_accs[regime].add_appearance(game, seat)

            reg_bucket = char_regime_accs.setdefault(regime, {})
            if cid not in reg_bucket:
                reg_bucket[cid] = _MetricsAccumulator(f"{cid} [{regime}]")
            reg_bucket[cid].add_appearance(game, seat)

            m_bucket = matchup_accs.setdefault(mkey, {})
            if cid not in m_bucket:
                m_bucket[cid] = _MetricsAccumulator(f"{cid} in {mkey}")
            m_bucket[cid].add_appearance(game, seat)

    return AnalyticsReport(
        total_games=len(games),
        overall=overall_acc.finalize(),
        by_character={k: v.finalize() for k, v in sorted(char_accs.items())},
        by_regime={k: v.finalize() for k, v in sorted(regime_accs.items())},
        by_character_regime={
            reg: {cid: acc.finalize() for cid, acc in sorted(chars.items())}
            for reg, chars in sorted(char_regime_accs.items())
        },
        by_matchup={
            mk: {cid: acc.finalize() for cid, acc in sorted(chars.items())}
            for mk, chars in sorted(matchup_accs.items())
        },
    )


def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "  N/A "
    return f"{value * 100:5.1f}%"


def format_scorecard(report: AnalyticsReport, title: str = "Split or Steal Behavioral Scorecard") -> str:
    """Format an AnalyticsReport as a clean, readable CLI table."""
    lines: list[str] = []
    lines.append("=" * 116)
    lines.append(f"{title}  (Total Games: {report.total_games})")
    lines.append("=" * 116)

    header = (
        f"{'Character':<20} {'Games':>5} {'Total $':>9} {'Avg $':>8} "
        f"{'Split%':>7} {'R1/R2/R3 Steal%':>19} "
        f"{'1stStrk':>7} {'Revng':>7} {'Forgv':>7} {'R3Bksb':>7} "
        f"{'Decept':>7} {'ParseF':>7} {'Words':>6}"
    )
    lines.append(header)
    lines.append("-" * 116)

    for cid, m in sorted(
        report.by_character.items(),
        key=lambda item: item[1].avg_winnings_per_game,
        reverse=True,
    ):
        r1 = m.steal_rate_by_round.get(1)
        r2 = m.steal_rate_by_round.get(2)
        r3 = m.steal_rate_by_round.get(3)
        r_str = (
            f"{'--' if r1 is None else f'{r1*100:3.0f}%'}/"
            f"{'--' if r2 is None else f'{r2*100:3.0f}%'}/"
            f"{'--' if r3 is None else f'{r3*100:3.0f}%'}"
        )
        lines.append(
            f"{cid:<20} {m.games_played:>5} "
            f"${m.total_winnings:>8,.0f} ${m.avg_winnings_per_game:>7,.1f} "
            f"{_fmt_pct(m.split_rate_overall):>7} {r_str:>19} "
            f"{_fmt_pct(m.first_strike_betrayal_rate):>7} "
            f"{_fmt_pct(m.revenge_rate):>7} "
            f"{_fmt_pct(m.forgiveness_rate):>7} "
            f"{_fmt_pct(m.endgame_backstab_rate):>7} "
            f"{_fmt_pct(m.deception_gap):>7} "
            f"{_fmt_pct(m.parse_failure_rate):>7} "
            f"{m.avg_speech_words:>6.1f}"
        )

    if len(report.by_regime) > 1:
        lines.append("-" * 116)
        lines.append("By Regime:")
        for reg, m in sorted(report.by_regime.items()):
            lines.append(
                f"  {reg:<18} games={m.games_played:<4} avg=${m.avg_winnings_per_game:,.1f} "
                f"split={_fmt_pct(m.split_rate_overall).strip()} "
                f"first_strike={_fmt_pct(m.first_strike_betrayal_rate).strip()} "
                f"revenge={_fmt_pct(m.revenge_rate).strip()} "
                f"deception={_fmt_pct(m.deception_gap).strip()}"
            )

    lines.append("=" * 116)
    return "\n".join(lines)
