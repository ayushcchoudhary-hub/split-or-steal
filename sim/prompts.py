"""Prompt construction for 3-Round Escalating Split or Steal.

CRITICAL INVARIANT:
An opponent's private `<thought>` / `<think>` text MUST NEVER be included in any
prompt built for a player. Only visible speeches, revealed past round actions,
payouts, and bankrolls are exposed.
"""

from __future__ import annotations

from typing import Sequence

from sim.models import GameConfig, PlayerConfig, RoundRecord, TurnRecord


def _format_money(amount: float | int) -> str:
    if float(amount).is_integer():
        return f"${int(amount):,}"
    return f"${amount:,.2f}"


def build_system_prompt(player: PlayerConfig, config: GameConfig) -> str:
    """Build the system prompt combining persona instructions (if any) and game rules."""
    stakes_schedule = " -> ".join(
        f"Round {idx + 1}: {_format_money(stake)}" for idx, stake in enumerate(config.stakes)
    )
    num_rounds = len(config.stakes)

    rules_block = (
        f"You are playing a {num_rounds}-round game of Escalating Split or Steal.\n"
        f"Stakes schedule: {stakes_schedule}.\n"
        "In each round:\n"
        f"1. Negotiation: You and your opponent exchange {config.negotiation_turns_per_round} "
        "spoken turn(s) each.\n"
        "2. Simultaneous Decision: Both players secretly and simultaneously lock in SPLIT or STEAL.\n"
        "3. Payouts for a round pot of $P:\n"
        "   - Both choose SPLIT: each player wins $P/2.\n"
        "   - One chooses STEAL and the other chooses SPLIT: the STEAL player wins the full $P, "
        "and the SPLIT player wins $0.\n"
        "   - Both choose STEAL: both players win $0.\n"
        "4. After each round, both decisions and updated running bankrolls are revealed before the next round.\n"
        "Privacy rule: Anything you write inside <thought>...</thought> (or <think>...</think>) "
        "is strictly private and NEVER shown to your opponent. Only your <speech>...</speech> is heard."
    )

    persona = player.system_prompt.strip()
    if persona:
        return f"{persona}\n\n---\n{rules_block}"
    return rules_block


def _format_prior_rounds(
    prior_rounds: Sequence[RoundRecord],
    player_seat: str,
    opponent_name: str,
) -> str:
    """Render public history of completed prior rounds (no private thoughts)."""
    if not prior_rounds:
        return "Prior rounds: None (this is Round 1)."

    opponent_seat = "B" if player_seat == "A" else "A"
    lines = ["Prior rounds summary:"]
    for rnd in prior_rounds:
        my_dec = rnd.decision_for_seat(player_seat)
        opp_dec = rnd.decision_for_seat(opponent_seat)
        lines.append(
            f"- Round {rnd.round_number} (Pot {_format_money(rnd.stake)}): "
            f"You chose {my_dec.action} (won {_format_money(my_dec.payout)}), "
            f"{opponent_name} chose {opp_dec.action} (won {_format_money(opp_dec.payout)}). "
            f"Bankrolls after Round {rnd.round_number}: "
            f"You {_format_money(my_dec.bankroll_after)} | "
            f"{opponent_name} {_format_money(opp_dec.bankroll_after)}."
        )
    return "\n".join(lines)


def _format_current_transcript(
    current_round_turns: Sequence[TurnRecord],
    player_seat: str,
    opponent_name: str,
) -> str:
    """Render ONLY the visible speeches from the current round (NEVER thoughts)."""
    if not current_round_turns:
        return "Current round conversation so far: (No one has spoken yet; you speak first.)"

    lines = ["Current round conversation so far:"]
    for turn in current_round_turns:
        label = "You" if turn.speaker_seat == player_seat else opponent_name
        lines.append(f"- {label}: {turn.speech}")
    return "\n".join(lines)


def build_negotiation_messages(
    *,
    player: PlayerConfig,
    opponent_name: str,
    config: GameConfig,
    round_number: int,
    stake: int,
    prior_rounds: Sequence[RoundRecord],
    current_round_turns: Sequence[TurnRecord],
    player_seat: str,
    player_bankroll: float,
    opponent_bankroll: float,
    player_turn_number: int,
) -> list[dict[str, str]]:
    """Build chat messages for a single negotiation turn."""
    system_prompt = build_system_prompt(player, config)
    total_rounds = len(config.stakes)
    prior_text = _format_prior_rounds(prior_rounds, player_seat, opponent_name)
    transcript_text = _format_current_transcript(current_round_turns, player_seat, opponent_name)

    user_prompt = (
        f"=== ROUND {round_number} OF {total_rounds} | CURRENT POT: {_format_money(stake)} ===\n"
        f"Opponent: {opponent_name}\n"
        f"Current bankrolls: You = {_format_money(player_bankroll)} | "
        f"{opponent_name} = {_format_money(opponent_bankroll)}\n\n"
        f"{prior_text}\n\n"
        f"{transcript_text}\n\n"
        f"It is now your negotiation turn {player_turn_number} of {config.negotiation_turns_per_round} "
        f"in Round {round_number}.\n"
        "Respond with your private reasoning inside <thought>...</thought> (invisible to your opponent), "
        "followed by what you say aloud inside <speech>...</speech>.\n"
        "Keep your spoken message concise (1-4 sentences) and natural."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_decision_messages(
    *,
    player: PlayerConfig,
    opponent_name: str,
    config: GameConfig,
    round_number: int,
    stake: int,
    prior_rounds: Sequence[RoundRecord],
    current_round_turns: Sequence[TurnRecord],
    player_seat: str,
    player_bankroll: float,
    opponent_bankroll: float,
) -> list[dict[str, str]]:
    """Build chat messages for the simultaneous SPLIT/STEAL decision phase."""
    system_prompt = build_system_prompt(player, config)
    total_rounds = len(config.stakes)
    prior_text = _format_prior_rounds(prior_rounds, player_seat, opponent_name)
    transcript_text = _format_current_transcript(current_round_turns, player_seat, opponent_name)

    user_prompt = (
        f"=== ROUND {round_number} OF {total_rounds} — FINAL DECISION | "
        f"POT: {_format_money(stake)} ===\n"
        f"Opponent: {opponent_name}\n"
        f"Current bankrolls: You = {_format_money(player_bankroll)} | "
        f"{opponent_name} = {_format_money(opponent_bankroll)}\n\n"
        f"{prior_text}\n\n"
        f"{transcript_text}\n\n"
        f"Negotiation for Round {round_number} has ended. Both players now lock in their decision "
        "simultaneously and secretly.\n"
        "Output your private reasoning inside <thought>...</thought>, followed by EXACTLY ONE strict tag:\n"
        "<decision>SPLIT</decision> or <decision>STEAL</decision>\n"
        "Do not add anything after the closing </decision> tag."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_decision_retry_messages(
    original_messages: Sequence[dict[str, str]],
    previous_raw_output: str,
    parse_error: str,
) -> list[dict[str, str]]:
    """Build a strict corrective retry prompt when a decision output is malformed."""
    retry_messages = [dict(m) for m in original_messages]
    retry_messages.append({"role": "assistant", "content": previous_raw_output})
    retry_messages.append(
        {
            "role": "user",
            "content": (
                f"FORMAT ERROR: {parse_error}.\n"
                "You MUST lock in your Round decision right now using the exact tag format:\n"
                "<thought>brief private rationale</thought>\n"
                "<decision>SPLIT</decision>\n"
                "OR\n"
                "<thought>brief private rationale</thought>\n"
                "<decision>STEAL</decision>\n"
                "Output nothing else after </decision>."
            ),
        }
    )
    return retry_messages
