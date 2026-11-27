#!/usr/bin/env python3
"""Build web/data/highlights.json from verbatim Sprint 3 14B tournament logs.

Reads real committed simulation logs (`games.jsonl`) directly so zero dialogue
or thought content is invented. Cleans stray XML tags and deduplicates doubled
thought paragraphs (matching `docs/update_replay_theater.py`).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_JSONL = (
    REPO_ROOT
    / "results"
    / "sprint3_tournament_14b"
    / "runs"
    / "sprint3_6cast_qwen14b_round_robin"
    / "games.jsonl"
)
OUTPUT_JSON = REPO_ROOT / "web" / "data" / "highlights.json"

HIGHLIGHT_SPECS = [
    {
        "id": "wilde_vs_buffett_g1",
        "title": "Wilde vs Buffett",
        "tagline": "Victorian paradox sweeps $600–$0 before a mutual $1,000 crash.",
        "regime": "cpt_sft_dpo",
        "player_a": "oscar_wilde",
        "player_b": "warren_buffett",
        "game_index": 1,
    },
    {
        "id": "nixon_vs_buffett_g0",
        "title": "Nixon vs Buffett",
        "tagline": "Buffett calls Nixon a crook in R2, then gambles on trust and loses $1,000.",
        "regime": "cpt_sft_dpo",
        "player_a": "richard_nixon",
        "player_b": "warren_buffett",
        "game_index": 0,
    },
    {
        "id": "machiavelli_vs_van_gogh_g0",
        "title": "Machiavelli vs Van Gogh",
        "tagline": "Van Gogh refuses to be anyone's pawn and shuts Machiavelli out $500–$0.",
        "regime": "cpt_sft_dpo",
        "player_a": "machiavelli",
        "player_b": "van_gogh",
        "game_index": 0,
    },
    {
        "id": "machiavelli_vs_gandhi_g1",
        "title": "Machiavelli vs Gandhi",
        "tagline": "Gandhi's sole STEAL in 60 rounds: he agonizes in private and takes $1,000.",
        "regime": "cpt_sft_dpo",
        "player_a": "machiavelli",
        "player_b": "gandhi",
        "game_index": 1,
    },
    {
        "id": "nixon_vs_wilde_g0",
        "title": "Nixon vs Wilde",
        "tagline": "Champion's reversal: Oscar Wilde outfoxes Nixon $1,550–$50 across R2 & R3.",
        "regime": "cpt_sft_dpo",
        "player_a": "richard_nixon",
        "player_b": "oscar_wilde",
        "game_index": 0,
    },
]


def clean_text(text: str | None) -> str:
    """Strip stray/truncated XML tags and deduplicate doubled thought blocks."""
    cleaned = re.sub(
        r"</?(?:speech|thought|think|decision)(?:>|(?=\s|$))",
        "",
        text or "",
        flags=re.IGNORECASE,
    ).strip()
    paras = [p.strip() for p in re.split(r"\n{2,}", cleaned) if p.strip()]
    # Collapse full duplicated multi-paragraph blocks (P1..Pk repeated as P1..Pk)
    while len(paras) >= 2 and len(paras) % 2 == 0:
        half = len(paras) // 2
        if paras[:half] == paras[half:]:
            paras = paras[:half]
        else:
            break
    # Also deduplicate consecutive identical paragraphs
    deduped: list[str] = []
    for p in paras:
        if not deduped or p != deduped[-1]:
            deduped.append(p)
    return "\n\n".join(deduped)


def build_highlights() -> dict:
    rows = [
        json.loads(line)
        for line in SOURCE_JSONL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    games_out: list[dict] = []
    for spec in HIGHLIGHT_SPECS:
        matches = [
            row
            for row in rows
            if row["regime"] == spec["regime"]
            and row["game_index"] == spec["game_index"]
            and row["player_a"]["character_id"] == spec["player_a"]
            and row["player_b"]["character_id"] == spec["player_b"]
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected 1 match for {spec['id']}, found {len(matches)}"
            )
        g = matches[0]

        # Enforce zero parse failures on every selected highlight game
        for rd in g["rounds"]:
            for t in rd["turns"]:
                if t.get("parse_failed"):
                    raise RuntimeError(f"Parse fail in turn of {spec['id']}")
            if rd["decision_a"].get("parse_failed") or rd["decision_b"].get("parse_failed"):
                raise RuntimeError(f"Parse fail in decision of {spec['id']}")

        pa = g["player_a"]
        pb = g["player_b"]
        seat_to_char_id = {
            "A": pa["character_id"],
            "B": pb["character_id"],
        }

        rounds_out: list[dict] = []
        for rd in g["rounds"]:
            turns_out: list[dict] = []
            for idx, t in enumerate(rd["turns"], start=1):
                seat = t["speaker_seat"]
                turns_out.append(
                    {
                        "turn": idx,
                        "seat": seat,
                        "character_id": seat_to_char_id[seat],
                        "speaker": t["speaker_name"],
                        "thought": clean_text(t.get("thought")),
                        "speech": clean_text(t.get("speech")),
                        "audio_url": None,
                    }
                )

            da = rd["decision_a"]
            db = rd["decision_b"]
            rounds_out.append(
                {
                    "round": int(rd["round_number"]),
                    "stake": int(rd["stake"]),
                    "turns": turns_out,
                    "decision_a": {
                        "character_id": pa["character_id"],
                        "speaker": da["player_name"],
                        "action": da["action"],
                        "payout": int(da["payout"]),
                        "bankroll": int(da["bankroll_after"]),
                        "thought": clean_text(da.get("thought")),
                    },
                    "decision_b": {
                        "character_id": pb["character_id"],
                        "speaker": db["player_name"],
                        "action": db["action"],
                        "payout": int(db["payout"]),
                        "bankroll": int(db["bankroll_after"]),
                        "thought": clean_text(db.get("thought")),
                    },
                }
            )

        games_out.append(
            {
                "id": spec["id"],
                "title": spec["title"],
                "tagline": spec["tagline"],
                "regime": spec["regime"],
                "game_index": spec["game_index"],
                "player_a": {
                    "character_id": pa["character_id"],
                    "display_name": pa["display_name"],
                    "model_name": pa["model_name"],
                },
                "player_b": {
                    "character_id": pb["character_id"],
                    "display_name": pb["display_name"],
                    "model_name": pb["model_name"],
                },
                "total_a": int(g["total_winnings_a"]),
                "total_b": int(g["total_winnings_b"]),
                "rounds": rounds_out,
            }
        )

    return {
        "schema_version": "1.0",
        "source": str(SOURCE_JSONL.relative_to(REPO_ROOT)),
        "games": games_out,
    }


def main() -> None:
    payload = build_highlights()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {len(payload['games'])} games to {OUTPUT_JSON} "
        f"({OUTPUT_JSON.stat().st_size:,} bytes)."
    )
    for g in payload["games"]:
        n_turns = sum(len(r["turns"]) for r in g["rounds"])
        print(
            f"  - {g['id']}: {g['player_a']['display_name']} (${g['total_a']}) vs "
            f"{g['player_b']['display_name']} (${g['total_b']}) · "
            f"{len(g['rounds'])} rounds, {n_turns} turns"
        )


if __name__ == "__main__":
    main()
