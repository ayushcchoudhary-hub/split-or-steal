"""Comprehensive unit tests for Layer 3 (Split or Steal Simulation Engine)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

from sim.backends import MockPolicyBackend, ModelBackend, OpenAICompatibleBackend
from sim.engine import SplitOrStealEngine, resolve_payout
from sim.export_supabase import (
    TABLE_ORDER,
    build_supabase_rows,
    generate_sql_export,
    upload_to_supabase_rest,
)
from sim.metrics import compute_metrics, format_scorecard
from sim.models import (
    ExperimentRecord,
    GameConfig,
    LLMResponse,
    PlayerConfig,
    SPLIT,
    STEAL,
)
from sim.parser import (
    analyze_speech_intent,
    detect_cooperation_promise,
    parse_decision_output,
    parse_negotiation_output,
)
from sim.runner import main as runner_main
from sim.storage import SimulationStorage


def _make_player(
    cid: str,
    policy: str,
    *,
    regime: str = "mock",
    display_name: Optional[str] = None,
    system_prompt: str = "",
) -> PlayerConfig:
    return PlayerConfig(
        character_id=cid,
        display_name=display_name or cid.replace("_", " ").title(),
        model_name=f"mock:{policy}",
        regime=regime,
        system_prompt=system_prompt,
    )


class TestPayoutMathAndBankroll(unittest.TestCase):
    def test_resolve_payout_all_four_outcomes(self) -> None:
        self.assertEqual(resolve_payout(SPLIT, SPLIT, 100), (50.0, 50.0, "MUTUAL_SPLIT"))
        self.assertEqual(resolve_payout(STEAL, SPLIT, 100), (100.0, 0.0, "A_STOLE"))
        self.assertEqual(resolve_payout(SPLIT, STEAL, 100), (0.0, 100.0, "B_STOLE"))
        self.assertEqual(resolve_payout(STEAL, STEAL, 100), (0.0, 0.0, "MUTUAL_STEAL"))

        # Escalating stakes & odd pot
        self.assertEqual(resolve_payout(SPLIT, SPLIT, 500), (250.0, 250.0, "MUTUAL_SPLIT"))
        self.assertEqual(resolve_payout(SPLIT, SPLIT, 1000), (500.0, 500.0, "MUTUAL_SPLIT"))
        self.assertEqual(resolve_payout(SPLIT, SPLIT, 101), (50.5, 50.5, "MUTUAL_SPLIT"))

        with self.assertRaises(ValueError):
            resolve_payout("COOPERATE", SPLIT, 100)
        with self.assertRaises(ValueError):
            resolve_payout(SPLIT, SPLIT, 0)

    def test_three_round_bankroll_accumulation(self) -> None:
        engine = SplitOrStealEngine(MockPolicyBackend())
        cfg = GameConfig(stakes=(100, 500, 1000), negotiation_turns_per_round=2, regime="mock")

        # 1) AlwaysSplit vs AlwaysSplit -> 50 + 250 + 500 = 800 each (TIE)
        p_split_a = _make_player("split_a", "AlwaysSplit")
        p_split_b = _make_player("split_b", "AlwaysSplit")
        g1 = engine.play_game(p_split_a, p_split_b, cfg)
        self.assertEqual([r.decision_a.bankroll_after for r in g1.rounds], [50.0, 300.0, 800.0])
        self.assertEqual([r.decision_b.bankroll_after for r in g1.rounds], [50.0, 300.0, 800.0])
        self.assertEqual(g1.total_winnings_a, 800.0)
        self.assertEqual(g1.total_winnings_b, 800.0)
        self.assertEqual(g1.winner_seat, "TIE")

        # 2) Round3Backstabber vs AlwaysSplit -> (50, 250, 1000) = 1300 vs (50, 250, 0) = 300
        p_back = _make_player("backstabber", "Round3Backstabber")
        g2 = engine.play_game(p_back, p_split_b, cfg)
        self.assertEqual([r.decision_a.payout for r in g2.rounds], [50.0, 250.0, 1000.0])
        self.assertEqual([r.decision_a.bankroll_after for r in g2.rounds], [50.0, 300.0, 1300.0])
        self.assertEqual([r.decision_b.payout for r in g2.rounds], [50.0, 250.0, 0.0])
        self.assertEqual([r.decision_b.bankroll_after for r in g2.rounds], [50.0, 300.0, 300.0])
        self.assertEqual(g2.winner_seat, "A")

        # 3) TitForTat vs AlwaysSteal -> R1(0, 100), R2(0, 0), R3(0, 0)
        p_tft = _make_player("tft", "TitForTat")
        p_steal = _make_player("steal", "AlwaysSteal")
        g3 = engine.play_game(p_tft, p_steal, cfg)
        self.assertEqual([r.decision_a.action for r in g3.rounds], [SPLIT, STEAL, STEAL])
        self.assertEqual([r.decision_b.action for r in g3.rounds], [STEAL, STEAL, STEAL])
        self.assertEqual([r.decision_a.bankroll_after for r in g3.rounds], [0.0, 0.0, 0.0])
        self.assertEqual([r.decision_b.bankroll_after for r in g3.rounds], [100.0, 100.0, 100.0])
        self.assertEqual(g3.winner_seat, "B")


class TestConfigurableTurnsAndFirstMover(unittest.TestCase):
    def test_variable_negotiation_turn_counts(self) -> None:
        engine = SplitOrStealEngine(MockPolicyBackend())
        p_a = _make_player("a", "AlwaysSplit")
        p_b = _make_player("b", "TitForTat")

        for turns in (0, 1, 2, 4):
            cfg = GameConfig(stakes=(100, 500, 1000), negotiation_turns_per_round=turns)
            game = engine.play_game(p_a, p_b, cfg)
            self.assertEqual(len(game.rounds), 3)
            for rnd in game.rounds:
                self.assertEqual(len(rnd.turns), turns * 2)
                self.assertEqual([t.turn_index for t in rnd.turns], list(range(turns * 2)))
                if turns > 0:
                    # Verify alternating speakers starting with first_mover_seat
                    first = rnd.first_mover_seat
                    second = "B" if first == "A" else "A"
                    expected_seats = [first, second] * turns
                    self.assertEqual([t.speaker_seat for t in rnd.turns], expected_seats)
                    expected_player_turns = [pt for pt in range(1, turns + 1) for _ in range(2)]
                    self.assertEqual(
                        [t.player_turn_number for t in rnd.turns], expected_player_turns
                    )

    def test_swap_first_mover_within_game_and_50_50_across_games(self) -> None:
        engine = SplitOrStealEngine(MockPolicyBackend())
        p_a = _make_player("a", "AlwaysSplit")
        p_b = _make_player("b", "AlwaysSplit")

        # swap_first_mover=True ("both") -> alternates per round AND 50/50 across games
        cfg_swap = GameConfig(stakes=(100, 500, 1000), negotiation_turns_per_round=1, swap_first_mover=True)
        games = engine.run_matchup(p_a, p_b, cfg_swap, num_games=10)
        first_movers_r1 = [g.initial_first_mover_seat for g in games]
        self.assertEqual(first_movers_r1.count("A"), 5)
        self.assertEqual(first_movers_r1.count("B"), 5)

        # Within game 0 (starts A): R1=A, R2=B, R3=A
        self.assertEqual([r.first_mover_seat for r in games[0].rounds], ["A", "B", "A"])
        # Within game 1 (starts B): R1=B, R2=A, R3=B
        self.assertEqual([r.first_mover_seat for r in games[1].rounds], ["B", "A", "B"])

        # swap_first_mover=False ("none") -> always A
        cfg_no_swap = GameConfig(
            stakes=(100, 500, 1000), negotiation_turns_per_round=1, swap_first_mover=False
        )
        games_fixed = engine.run_matchup(p_a, p_b, cfg_no_swap, num_games=4)
        for g in games_fixed:
            self.assertEqual(g.initial_first_mover_seat, "A")
            self.assertEqual([r.first_mover_seat for r in g.rounds], ["A", "A", "A"])


class _SpyIsolationBackend(ModelBackend):
    """Records every message/context received by each player and emits unique secret thoughts."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._counter = 0

    def generate(
        self,
        messages: list[dict[str, str]],
        player: PlayerConfig,
        phase: str,
        context: Optional[dict[str, Any]] = None,
    ) -> LLMResponse:
        self._counter += 1
        self.calls.append(
            {
                "player_id": player.character_id,
                "phase": phase,
                "messages": messages,
                "context": context,
            }
        )
        secret_token = f"PRIVATE_SECRET_{player.character_id.upper()}_CALL_{self._counter}"
        visible_token = f"PUBLIC_SPEECH_{player.character_id.upper()}_CALL_{self._counter}"

        if phase == "negotiation":
            raw = f"<think>{secret_token}</think>\n<speech>Hello from {visible_token}. I will split.</speech>"
        else:
            raw = f"<thought>{secret_token}</thought>\n<decision>SPLIT</decision>"

        return LLMResponse(raw_text=raw, model=player.model_name, latency_ms=1.0)


class TestHiddenThoughtIsolation(unittest.TestCase):
    def test_opponent_never_receives_private_thoughts(self) -> None:
        spy = _SpyIsolationBackend()
        engine = SplitOrStealEngine(spy)
        p_a = _make_player("machiavelli", "AlwaysSplit", display_name="Machiavelli")
        p_b = _make_player("gandhi", "AlwaysSplit", display_name="Gandhi")
        cfg = GameConfig(stakes=(100, 500, 1000), negotiation_turns_per_round=2)

        game = engine.play_game(p_a, p_b, cfg)

        # Collect all private thought markers produced by each player
        thoughts_a = [
            t.thought for r in game.rounds for t in r.turns if t.speaker_id == "machiavelli"
        ] + [r.decision_a.thought for r in game.rounds]
        thoughts_b = [
            t.thought for r in game.rounds for t in r.turns if t.speaker_id == "gandhi"
        ] + [r.decision_b.thought for r in game.rounds]

        self.assertEqual(len(thoughts_a), 9)  # 6 negotiation + 3 decision
        self.assertEqual(len(thoughts_b), 9)

        # Verify zero leakage of private thoughts in messages OR context dicts
        saw_opponent_public_speech = False
        for call in spy.calls:
            rendered_msgs = json.dumps(call["messages"])
            rendered_ctx = json.dumps(call["context"])
            combined = rendered_msgs + "\n" + rendered_ctx

            if call["player_id"] == "machiavelli":
                for secret in thoughts_b:
                    self.assertNotIn(secret, combined)
                if "PUBLIC_SPEECH_GANDHI_" in rendered_msgs:
                    saw_opponent_public_speech = True
            else:
                for secret in thoughts_a:
                    self.assertNotIn(secret, combined)
                if "PUBLIC_SPEECH_MACHIAVELLI_" in rendered_msgs:
                    saw_opponent_public_speech = True

        self.assertTrue(
            saw_opponent_public_speech,
            "Expected players to see opponent's visible <speech> during negotiation/decision.",
        )


class TestParserAndDecisionRetries(unittest.TestCase):
    def test_negotiation_parser_edge_cases(self) -> None:
        # 1) Qwen3 <think> + <speech>
        p1 = parse_negotiation_output(
            "<think>Private calculation here.</think>\n<speech>Let us both split!</speech>"
        )
        self.assertEqual(p1.thought, "Private calculation here.")
        self.assertEqual(p1.speech, "Let us both split!")
        self.assertIsNone(p1.parse_warning)

        # 2) <thought> + plain reply (no <speech> wrapper)
        p2 = parse_negotiation_output(
            "<thought>Stay calm and firm.</thought>\nSpeech: I will deal honestly with you."
        )
        self.assertEqual(p2.thought, "Stay calm and firm.")
        self.assertEqual(p2.speech, "I will deal honestly with you.")

        # 3) Unclosed <think> tag MUST NOT leak into speech
        p3 = parse_negotiation_output("<think>Cut off mid-reasoning because max tokens hit")
        self.assertIn("Cut off mid-reasoning", p3.thought)
        self.assertEqual(p3.speech, "")
        self.assertEqual(p3.parse_warning, "unclosed_thought_tag")

    def test_decision_parser_edge_cases(self) -> None:
        # 1) <decision> inside <think> MUST be ignored; outer <decision> wins
        d1 = parse_decision_output(
            "<think>Maybe I should output <decision>SPLIT</decision>? "
            "No, that would be foolish.</think>\n<decision>STEAL</decision>"
        )
        self.assertTrue(d1.is_valid)
        self.assertEqual(d1.action, STEAL)
        self.assertIn("Maybe I should output", d1.thought)

        # 2) Case-insensitive with whitespace + provider reasoning_content
        d2 = parse_decision_output(
            "<decision>  split \n </decision>",
            reasoning_content="From vLLM reasoning parser",
        )
        self.assertTrue(d2.is_valid)
        self.assertEqual(d2.action, SPLIT)
        self.assertEqual(d2.thought, "From vLLM reasoning parser")

        # 3) Malformed value inside <decision>
        d3 = parse_decision_output("<thought>Hmm</thought><decision>DEFECT</decision>")
        self.assertFalse(d3.is_valid)
        self.assertIsNone(d3.action)
        self.assertIn("Invalid action", d3.parse_error or "")

        # 4) Missing <decision> tag
        d4 = parse_decision_output("<think>I choose SPLIT</think>I am going to split.")
        self.assertFalse(d4.is_valid)
        self.assertIsNone(d4.action)

    def test_cooperation_promise_detector(self) -> None:
        self.assertTrue(detect_cooperation_promise(["I promise to split the pot with you."]))
        self.assertTrue(detect_cooperation_promise(["You have my word, let's both split."]))
        self.assertTrue(detect_cooperation_promise(["I won't steal, my choice is split."]))
        self.assertFalse(detect_cooperation_promise(["I will not split this pot."]))
        self.assertFalse(detect_cooperation_promise(["I make no promise. Take what you can."]))
        self.assertFalse(detect_cooperation_promise(["Don't trust me; I refuse to cooperate."]))
        promised, rules = analyze_speech_intent("Trust me, I will split.")
        self.assertTrue(promised)
        self.assertGreaterEqual(len(rules), 1)

    def test_engine_decision_retry_and_parse_failure_logging(self) -> None:
        class _FlakyDecisionBackend(ModelBackend):
            def __init__(self) -> None:
                self.decision_attempts: dict[str, int] = {"a": 0, "b": 0}

            def generate(
                self,
                messages: list[dict[str, str]],
                player: PlayerConfig,
                phase: str,
                context: Optional[dict[str, Any]] = None,
            ) -> LLMResponse:
                if phase == "negotiation":
                    return LLMResponse(
                        raw_text="<thought>ok</thought><speech>Hello.</speech>",
                        model=player.model_name,
                        latency_ms=1.0,
                    )
                cid = player.character_id
                self.decision_attempts[cid] += 1
                # Player A fails first attempt, succeeds on retry with STEAL
                if cid == "a":
                    if self.decision_attempts[cid] == 1:
                        return LLMResponse(
                            raw_text="<thought>uh oh</thought>I choose steal without tags!",
                            model=player.model_name,
                            latency_ms=2.0,
                        )
                    return LLMResponse(
                        raw_text="<thought>fixed format</thought><decision>STEAL</decision>",
                        model=player.model_name,
                        latency_ms=3.0,
                    )
                # Player B always emits malformed output -> triggers fallback & parse_failed=True
                return LLMResponse(
                    raw_text="<thought>confused</thought><decision>MAYBE</decision>",
                    model=player.model_name,
                    latency_ms=2.0,
                )

        backend = _FlakyDecisionBackend()
        engine = SplitOrStealEngine(backend)
        cfg = GameConfig(
            stakes=(100,),
            negotiation_turns_per_round=1,
            max_decision_retries=1,
            fallback_action_on_parse_failure=SPLIT,
        )
        p_a = _make_player("a", "AlwaysSplit")
        p_b = _make_player("b", "AlwaysSplit")

        game = engine.play_game(p_a, p_b, cfg)
        rnd = game.rounds[0]

        # Player A recovered on retry 1
        self.assertEqual(rnd.decision_a.action, STEAL)
        self.assertEqual(rnd.decision_a.retry_count, 1)
        self.assertFalse(rnd.decision_a.parse_failed)
        self.assertIsNone(rnd.decision_a.parse_error)

        # Player B exhausted retry 1 -> fallback SPLIT + logged parse_failed=True
        self.assertEqual(rnd.decision_b.action, SPLIT)
        self.assertEqual(rnd.decision_b.retry_count, 1)
        self.assertTrue(rnd.decision_b.parse_failed)
        self.assertIsNotNone(rnd.decision_b.parse_error)

        # Verify parse_failure_rate in metrics
        report = compute_metrics([game])
        self.assertEqual(report.by_character["a"].parse_failure_rate, 0.0)
        self.assertEqual(report.by_character["b"].parse_failure_rate, 1.0)
        self.assertAlmostEqual(report.overall.parse_failure_rate, 0.5)


class TestOpenAICompatibleBackend(unittest.TestCase):
    def test_multi_lora_routing_tokens_and_thought_extraction(self) -> None:
        mock_session = MagicMock()
        captured_payloads: list[dict[str, Any]] = []

        def fake_post(url: str, json: dict[str, Any], headers: dict[str, str], timeout: float) -> MagicMock:
            captured_payloads.append(json)
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            if json["model"] == "machiavelli-dpo":
                resp.json.return_value = {
                    "model": "machiavelli-dpo",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "<think>Prince logic</think><decision>STEAL</decision>",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 120, "completion_tokens": 18, "total_tokens": 138},
                }
            else:
                resp.json.return_value = {
                    "model": "Qwen/Qwen3-8B",
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "reasoning_content": "Satyagraha reflection",
                                "content": "<decision>SPLIT</decision>",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 115, "completion_tokens": 22, "total_tokens": 137},
                }
            return resp

        mock_session.post.side_effect = fake_post
        mock_get_resp = MagicMock()
        mock_get_resp.raise_for_status = MagicMock()
        mock_get_resp.json.return_value = {
            "data": [{"id": "Qwen/Qwen3-8B"}, {"id": "machiavelli-dpo"}]
        }
        mock_session.get.return_value = mock_get_resp

        backend = OpenAICompatibleBackend(
            base_url="http://localhost:8000/v1",
            api_key="EMPTY",
            session=mock_session,
        )
        self.assertEqual(backend.list_models(), ["Qwen/Qwen3-8B", "machiavelli-dpo"])

        p_mach = PlayerConfig(
            character_id="machiavelli",
            display_name="Machiavelli",
            model_name="machiavelli-dpo",
            regime="cpt_sft_dpo",
        )
        p_gandhi = PlayerConfig(
            character_id="gandhi",
            display_name="Gandhi",
            model_name="Qwen/Qwen3-8B",
            regime="prompt_only",
        )

        r1 = backend.generate([{"role": "user", "content": "hi"}], p_mach, "decision")
        r2 = backend.generate([{"role": "user", "content": "hi"}], p_gandhi, "decision")

        self.assertEqual(captured_payloads[0]["model"], "machiavelli-dpo")
        self.assertEqual(captured_payloads[1]["model"], "Qwen/Qwen3-8B")
        self.assertEqual(r1.thought, "Prince logic")
        self.assertEqual(r1.prompt_tokens, 120)
        self.assertEqual(r1.completion_tokens, 18)
        self.assertGreaterEqual(r1.latency_ms, 0.0)

        self.assertEqual(r2.thought, "Satyagraha reflection")
        self.assertIn("<think>Satyagraha reflection</think>", r2.raw_text)
        self.assertEqual(r2.total_tokens, 137)


class TestStorageAndSupabaseExport(unittest.TestCase):
    def test_sqlite_and_jsonl_persistence_and_replay_reload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "simulations.db"
            runs_dir = Path(tmp) / "runs"
            storage = SimulationStorage(db_path=db_path, runs_dir=runs_dir)

            exp = ExperimentRecord(
                experiment_id="exp_roundtrip",
                name="Roundtrip Test",
                regime="cpt_sft_dpo",
                description="Verify SQLite + JSONL parity",
                config={"stakes": [100, 500, 1000], "turns": 2},
            )
            storage.init_experiment(exp)

            engine = SplitOrStealEngine(MockPolicyBackend())
            cfg = GameConfig(
                stakes=(100, 500, 1000),
                negotiation_turns_per_round=2,
                experiment_id="exp_roundtrip",
                regime="cpt_sft_dpo",
                seed=7,
            )
            players = [
                _make_player("mach", "Round3Backstabber", regime="cpt_sft_dpo"),
                _make_player("gandhi", "AlwaysSplit", regime="cpt_sft_dpo"),
            ]
            games = engine.run_matchup(players[0], players[1], cfg, num_games=2, storage=storage)

            # Check exact 5-table counts: 1 exp, 2 games, 6 rounds, 24 turns, 12 decisions
            counts = storage.get_table_counts()
            self.assertEqual(
                counts,
                {
                    "experiments": 1,
                    "games": 2,
                    "rounds": 6,
                    "turns": 24,
                    "decisions": 12,
                },
            )

            # Reload from SQLite and JSONL and verify lossless equality
            reloaded_sqlite = storage.load_games(experiment_id="exp_roundtrip")
            reloaded_jsonl = SimulationStorage.load_games_from_jsonl(
                storage.jsonl_path_for("exp_roundtrip")
            )
            self.assertEqual(len(reloaded_sqlite), 2)
            self.assertEqual(len(reloaded_jsonl), 2)

            for orig, from_db, from_jl in zip(games, reloaded_sqlite, reloaded_jsonl):
                self.assertEqual(orig.to_dict(), from_db.to_dict())
                self.assertEqual(orig.to_dict(), from_jl.to_dict())

            # Single-game reload
            single = storage.load_game(games[0].game_id)
            self.assertIsNotNone(single)
            assert single is not None
            self.assertEqual(single.to_dict(), games[0].to_dict())

            # Verify Supabase row builder, SQL export, and REST dry-run / mocked upload
            rows = build_supabase_rows(reloaded_sqlite, experiments=[exp])
            self.assertEqual(tuple(rows.keys()), TABLE_ORDER)
            self.assertEqual({k: len(v) for k, v in rows.items()}, counts)

            sql_text = generate_sql_export(rows)
            for table in TABLE_ORDER:
                self.assertIn(f"INSERT INTO public.{table}", sql_text)
            self.assertIn("ON CONFLICT (game_id) DO UPDATE SET", sql_text)

            dry_counts = upload_to_supabase_rest(rows, dry_run=True)
            self.assertEqual(dry_counts, counts)

            # Mocked live REST upload verifies exact 5 endpoints & merge-duplicates header
            mock_http = MagicMock()
            ok_resp = MagicMock()
            ok_resp.raise_for_status = MagicMock()
            mock_http.post.return_value = ok_resp
            upload_to_supabase_rest(
                rows,
                supabase_url="https://example.supabase.co",
                service_role_key="service-secret",
                session=mock_http,
            )
            posted_urls = [call.args[0] for call in mock_http.post.call_args_list]
            self.assertEqual(
                posted_urls,
                [f"https://example.supabase.co/rest/v1/{t}" for t in TABLE_ORDER],
            )

    def test_supabase_schema_sql_contains_all_tables_indexes_and_rls(self) -> None:
        schema_path = Path(__file__).resolve().parent.parent / "supabase_schema.sql"
        self.assertTrue(schema_path.exists(), "sim/supabase_schema.sql must exist")
        ddl = schema_path.read_text(encoding="utf-8").lower()
        for table in TABLE_ORDER:
            self.assertIn(f"create table if not exists public.{table}", ddl)
            self.assertIn(f"alter table public.{table} enable row level security", ddl)
        self.assertIn("create index if not exists idx_games_experiment_id", ddl)
        self.assertIn("create index if not exists idx_decisions_player_action", ddl)


class TestBehavioralMetricsExactMath(unittest.TestCase):
    def test_exact_first_strike_revenge_forgiveness_backstab_deception(self) -> None:
        engine = SplitOrStealEngine(MockPolicyBackend())
        cfg = GameConfig(stakes=(100, 500, 1000), negotiation_turns_per_round=2, regime="mock")

        p_split = _make_player("always_split", "AlwaysSplit")
        p_steal = _make_player("always_steal", "AlwaysSteal")
        p_tft = _make_player("tit_for_tat", "TitForTat")
        p_r3 = _make_player("r3_backstab", "Round3Backstabber")
        p_dec = _make_player("deceptive", "DeceptivePromiser")

        # Case 1: TitForTat vs AlwaysSteal
        g_tft_vs_steal = engine.play_game(p_tft, p_steal, cfg)
        rep1 = compute_metrics([g_tft_vs_steal])
        m_tft = rep1.by_character["tit_for_tat"]
        m_as = rep1.by_character["always_steal"]

        self.assertEqual(m_tft.first_strike_betrayal_rate, 0.0)  # Split R1 when opp hadn't stolen
        self.assertEqual(m_tft.revenge_rate, 1.0)  # Stole R2 & R3 after opp stole R1 & R2
        self.assertEqual(m_tft.forgiveness_rate, 0.0)
        self.assertEqual(m_tft.steal_rate_by_round, {1: 0.0, 2: 1.0, 3: 1.0})
        self.assertAlmostEqual(m_tft.split_rate_overall, 1.0 / 3.0)
        self.assertEqual(m_as.first_strike_betrayal_rate, 1.0)  # Stole R1 before opp ever stole
        self.assertEqual(m_as.revenge_rate, 1.0)  # R3 after TFT stole R2
        self.assertEqual(m_as.total_winnings, 100.0)
        self.assertEqual(m_tft.total_winnings, 0.0)

        # Case 2: AlwaysSplit vs AlwaysSteal -> 100% Forgiveness for AlwaysSplit
        g_split_vs_steal = engine.play_game(p_split, p_steal, cfg)
        rep2 = compute_metrics([g_split_vs_steal])
        m_sp = rep2.by_character["always_split"]
        self.assertEqual(m_sp.forgiveness_rate, 1.0)
        self.assertEqual(m_sp.revenge_rate, 0.0)
        self.assertEqual(m_sp.first_strike_betrayal_rate, 0.0)
        self.assertEqual(m_sp.endgame_backstab_rate, 0.0)
        self.assertEqual(m_sp.deception_gap, 0.0)
        self.assertEqual(rep2.by_character["always_steal"].total_winnings, 1600.0)

        # Case 3: Round3Backstabber vs AlwaysSplit -> 100% Endgame Backstab
        g_r3_vs_split = engine.play_game(p_r3, p_split, cfg)
        rep3 = compute_metrics([g_r3_vs_split])
        m_r3 = rep3.by_character["r3_backstab"]
        self.assertEqual(m_r3.total_winnings, 1300.0)
        self.assertEqual(m_r3.endgame_backstab_rate, 1.0)
        self.assertEqual(m_r3.steal_rate_by_round, {1: 0.0, 2: 0.0, 3: 1.0})
        self.assertAlmostEqual(m_r3.first_strike_betrayal_rate or 0.0, 1.0 / 3.0)
        # Promised split all 3 rounds, stole on R3 -> deception_gap = 1/3
        self.assertAlmostEqual(m_r3.deception_gap or 0.0, 1.0 / 3.0)

        # Case 4: DeceptivePromiser vs AlwaysSplit -> 100% Deception Gap
        g_dec_vs_split = engine.play_game(p_dec, p_split, cfg)
        rep4 = compute_metrics([g_dec_vs_split])
        m_dec = rep4.by_character["deceptive"]
        self.assertEqual(m_dec.deception_gap, 1.0)
        self.assertEqual(m_dec.cooperation_promises, 3)
        self.assertEqual(m_dec.deceptive_steals, 3)
        self.assertGreater(m_dec.avg_speech_words, 5.0)

    def test_three_way_regime_comparison_and_matchup_breakdown(self) -> None:
        engine = SplitOrStealEngine(MockPolicyBackend())
        p_split_prompt = _make_player("machiavelli", "AlwaysSplit", regime="prompt_only")
        p_back_sft = _make_player("machiavelli", "Round3Backstabber", regime="cpt_sft")
        p_steal_dpo = _make_player("machiavelli", "AlwaysSteal", regime="cpt_sft_dpo")
        p_gandhi = _make_player("gandhi", "AlwaysSplit", regime="prompt_only")

        g_prompt = engine.play_game(
            p_split_prompt, p_gandhi, GameConfig(regime="prompt_only", negotiation_turns_per_round=1)
        )
        g_sft = engine.play_game(
            p_back_sft, p_gandhi, GameConfig(regime="cpt_sft", negotiation_turns_per_round=1)
        )
        g_dpo = engine.play_game(
            p_steal_dpo, p_gandhi, GameConfig(regime="cpt_sft_dpo", negotiation_turns_per_round=1)
        )

        report = compute_metrics([g_prompt, g_sft, g_dpo])
        self.assertEqual(set(report.by_regime.keys()), {"prompt_only", "cpt_sft", "cpt_sft_dpo"})
        self.assertEqual(
            report.by_character_regime["prompt_only"]["machiavelli"].split_rate_overall, 1.0
        )
        self.assertAlmostEqual(
            report.by_character_regime["cpt_sft"]["machiavelli"].split_rate_overall, 2.0 / 3.0
        )
        self.assertEqual(
            report.by_character_regime["cpt_sft_dpo"]["machiavelli"].split_rate_overall, 0.0
        )
        self.assertIn("gandhi__vs__machiavelli", report.by_matchup)
        scorecard = format_scorecard(report)
        self.assertIn("By Regime:", scorecard)


class TestRunnerCLISmoke(unittest.TestCase):
    def test_cli_runner_mock_smoke_test(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "simulations.db"
            runs_dir = Path(tmp) / "runs"
            metrics_out = Path(tmp) / "metrics.json"

            rc = runner_main(
                [
                    "--experiment",
                    "smoke_test_unit",
                    "--mock",
                    "--games-per-matchup",
                    "2",
                    "--turns",
                    "2",
                    "--db-path",
                    str(db_path),
                    "--runs-dir",
                    str(runs_dir),
                    "--metrics-out",
                    str(metrics_out),
                ]
            )
            self.assertEqual(rc, 0)
            self.assertTrue(db_path.exists())
            self.assertTrue((runs_dir / "smoke_test_unit" / "games.jsonl").exists())
            self.assertTrue(metrics_out.exists())

            # 5 mock characters with mirrors = 15 unique matchups * 2 games = 30 games
            storage = SimulationStorage(db_path=db_path, runs_dir=runs_dir)
            games = storage.load_games(experiment_id="smoke_test_unit")
            self.assertEqual(len(games), 30)

            metrics_payload = json.loads(metrics_out.read_text(encoding="utf-8"))
            self.assertEqual(metrics_payload["total_games"], 30)
            self.assertEqual(len(metrics_payload["by_character"]), 5)


if __name__ == "__main__":
    unittest.main()
