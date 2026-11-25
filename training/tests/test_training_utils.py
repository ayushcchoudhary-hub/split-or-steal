"""Local CPU unit tests for Layer 2 training, verification, and Colab orchestration utilities.

Runs purely on Python stdlib + project code (zero GPU and zero PyTorch required).
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from training.colab_orchestrator import (
    ColabSessionRunner,
    build_parser,
    build_remote_shell_runner,
    to_contents_api_path,
)
from training.utils import (
    cohens_d_paired,
    compute_lora_frobenius_norms,
    extract_text_record,
    normalize_dpo_pair_record,
    normalize_dpo_prompt_record,
    normalize_sft_record,
    parse_colab_usage,
    perplexity_from_nll,
    read_jsonl,
    softmax_two_class,
    strip_think_tags,
    summarize_logit_shift,
    write_jsonl,
)
from training.verify_weight_shift import (
    QUALITATIVE_PROMPTS,
    ZERO_PROMPT_DILEMMAS,
    _format_dilemma_prompt,
    parse_kv_pairs,
    render_verification_markdown,
)


def _explicit_ba_frobenius(b_mat: list[list[float]], a_mat: list[list[float]], scale: float) -> float:
    """Reference brute-force O(d_out * d_in * r) calculation of ``scale * ||B @ A||_F``."""
    d_out = len(b_mat)
    r = len(b_mat[0])
    d_in = len(a_mat[0])
    sq = 0.0
    for i in range(d_out):
        for j in range(d_in):
            val = sum(b_mat[i][k] * a_mat[k][j] for k in range(r)) * scale
            sq += val * val
    return math.sqrt(sq)


class TestLoraFrobeniusNorms(unittest.TestCase):
    def test_exact_equivalence_with_explicit_matmul(self) -> None:
        # B: (4, 2), A: (2, 3) -> delta_W: (4, 3)
        b_q = [
            [0.5, -0.25],
            [1.0, 0.75],
            [-0.4, 0.2],
            [0.3, -0.8],
        ]
        a_q = [
            [0.2, -0.6, 0.9],
            [0.4, 0.1, -0.5],
        ]
        b_gate = [
            [0.1, 0.2],
            [-0.3, 0.4],
            [0.5, -0.1],
        ]
        a_gate = [
            [0.7, -0.2, 0.3, 0.1],
            [-0.5, 0.6, 0.2, -0.4],
        ]

        tensors = {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": a_q,
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": b_q,
            "base_model.model.model.layers.0.mlp.gate_proj.lora_A.default.weight": a_gate,
            "base_model.model.model.layers.0.mlp.gate_proj.lora_B.default.weight": b_gate,
            # Zero-initialized B matrix on layer 1 should yield exact 0 norm
            "base_model.model.model.layers.1.self_attn.v_proj.lora_A.weight": a_q,
            "base_model.model.model.layers.1.self_attn.v_proj.lora_B.weight": [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
        }

        alpha = 32.0
        r = 16
        scale = alpha / r
        expected_q = _explicit_ba_frobenius(b_q, a_q, scale)
        expected_gate = _explicit_ba_frobenius(b_gate, a_gate, scale)
        expected_global = math.sqrt(expected_q**2 + expected_gate**2)

        res = compute_lora_frobenius_norms(tensors, lora_alpha=alpha, lora_r=r)

        self.assertEqual(res["num_adapted_modules"], 3)
        self.assertEqual(res["nonzero_modules"], 2)
        self.assertEqual(res["total_lora_params"], (8 + 6) + (6 + 8) + (8 + 6))
        self.assertAlmostEqual(res["global_frobenius_norm"], expected_global, places=5)
        self.assertAlmostEqual(res["by_module_type"]["q_proj"]["mean_norm"], expected_q, places=5)
        self.assertAlmostEqual(res["by_module_type"]["gate_proj"]["mean_norm"], expected_gate, places=5)
        self.assertAlmostEqual(res["by_module_type"]["v_proj"]["mean_norm"], 0.0, places=6)
        self.assertAlmostEqual(res["attention_vs_mlp"]["attention_l2_norm"], expected_q, places=5)
        self.assertAlmostEqual(res["attention_vs_mlp"]["mlp_l2_norm"], expected_gate, places=5)
        self.assertAlmostEqual(res["by_layer"]["0"], expected_global, places=5)
        self.assertAlmostEqual(res["by_layer"]["1"], 0.0, places=6)


class TestMathHelpers(unittest.TestCase):
    def test_softmax_two_class_stability(self) -> None:
        pa, pb = softmax_two_class(-5.0, -5.0)
        self.assertAlmostEqual(pa, 0.5, places=8)
        self.assertAlmostEqual(pb, 0.5, places=8)

        # Extreme negative logits should not underflow
        pa2, pb2 = softmax_two_class(-1000.0, -999.0)
        self.assertAlmostEqual(pa2 + pb2, 1.0, places=10)
        self.assertAlmostEqual(pb2 / pa2, math.e, places=6)

        with self.assertRaises(ValueError):
            softmax_two_class(float("nan"), -1.0)

    def test_perplexity_from_nll(self) -> None:
        # mean NLL = ln(10) -> PPL = 10
        nll = math.log(10.0) * 50
        self.assertAlmostEqual(perplexity_from_nll(nll, 50), 10.0, places=6)
        with self.assertRaises(ValueError):
            perplexity_from_nll(1.0, 0)

    def test_cohens_d_and_shift_summary(self) -> None:
        base = [0.10, 0.12, 0.08, 0.15, 0.11]
        after = [0.55, 0.62, 0.48, 0.60, 0.58]
        d = cohens_d_paired(base, after)
        self.assertGreater(d, 5.0)

        summary = summarize_logit_shift(base, after)
        self.assertEqual(summary["n_items"], 5)
        self.assertAlmostEqual(summary["pct_shifted_more_defect"], 100.0, places=2)
        self.assertGreater(summary["mean_shift_p_defect"], 0.4)


class TestDatasetHelpers(unittest.TestCase):
    def test_jsonl_roundtrip_and_extract_text(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sample.jsonl"
            rows = [{"text": "Fortune is a woman."}, {"content": "Truth is God."}]
            n = write_jsonl(p, rows)
            self.assertEqual(n, 2)
            loaded = read_jsonl(p)
            self.assertEqual(extract_text_record(loaded[0]), "Fortune is a woman.")
            self.assertEqual(extract_text_record(loaded[1]), "Truth is God.")

    def test_normalize_sft_all_schemas(self) -> None:
        r1 = normalize_sft_record(
            {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]},
            default_system="Be direct.",
        )
        self.assertEqual(r1["messages"][0]["role"], "system")
        self.assertEqual(len(r1["messages"]), 3)

        r2 = normalize_sft_record({"prompt": "Q?", "response": "A."})
        self.assertEqual(r2["messages"][0], {"role": "user", "content": "Q?"})
        self.assertEqual(r2["messages"][1], {"role": "assistant", "content": "A."})

        r3 = normalize_sft_record({"instruction": "Advise", "input": "Siege", "output": "Hold."})
        self.assertIn("Siege", r3["messages"][0]["content"])
        self.assertEqual(r3["messages"][1]["content"], "Hold.")

        r4 = normalize_sft_record({"text": "Raw dialogue block"})
        self.assertEqual(r4, {"text": "Raw dialogue block"})

    def test_normalize_dpo_and_strip_think(self) -> None:
        rec = normalize_dpo_prompt_record(
            {"prompt": "Should you break an oath?", "chosen": "When necessity compels it.", "theme": "oaths"},
            index=3,
        )
        self.assertEqual(rec["id"], "dpo_0003")
        self.assertEqual(rec["theme"], "oaths")

        pair = normalize_dpo_pair_record(
            {
                "id": "p1",
                "prompt": "Should you break an oath?",
                "chosen": "When necessity compels it.",
                "rejected": "It is important to consider both sides politely.",
            }
        )
        self.assertEqual(pair["rejected"], "It is important to consider both sides politely.")

        thought, visible = strip_think_tags("<think>Let me weigh options.</think>\nKeep your word.")
        self.assertEqual(thought, "Let me weigh options.")
        self.assertEqual(visible, "Keep your word.")


class TestColabUsageParserAndOrchestrator(unittest.TestCase):
    def test_parse_colab_usage_standard(self) -> None:
        sample = (
            "Current balance: 1209.41 compute units\n"
            "Usage rate: 0.00/hr\n"
            "Active assignments: 0\n"
        )
        parsed = parse_colab_usage(sample)
        self.assertTrue(parsed["parsed_ok"])
        self.assertAlmostEqual(parsed["balance_cu"], 1209.41, places=2)
        self.assertAlmostEqual(parsed["usage_rate_cu_per_hr"], 0.0, places=2)
        self.assertEqual(parsed["active_assignments"], 0)

    def test_parse_colab_usage_active_gpu_and_commas(self) -> None:
        sample = (
            "Current balance: 1,208.15 compute units\n"
            "Usage rate: 5.37/hr\n"
            "Active assignments: 1\n"
        )
        parsed = parse_colab_usage(sample)
        self.assertTrue(parsed["parsed_ok"])
        self.assertAlmostEqual(parsed["balance_cu"], 1208.15, places=2)
        self.assertAlmostEqual(parsed["usage_rate_cu_per_hr"], 5.37, places=2)
        self.assertEqual(parsed["active_assignments"], 1)

    def test_parse_colab_usage_empty(self) -> None:
        parsed = parse_colab_usage("Warning: offline")
        self.assertFalse(parsed["parsed_ok"])
        self.assertIsNone(parsed["balance_cu"])

    def test_orchestrator_paths_and_dry_run_plan(self) -> None:
        self.assertEqual(to_contents_api_path("/content/work/a.py"), "content/work/a.py")
        snippet = build_remote_shell_runner("echo hi")
        self.assertIn("subprocess.Popen", snippet)

        parser = build_parser()
        args = parser.parse_args(
            [
                "train-character",
                "--character",
                "machiavelli",
                "--stages",
                "cpt,sft,harvest_dpo,dpo,verify",
                "--gpu",
                "A100",
                "--dry-run",
            ]
        )
        rc = args.func(args)
        self.assertEqual(rc, 0)


class TestVerificationSuiteInvariants(unittest.TestCase):
    def test_20_zero_prompt_dilemmas_and_zero_game_leakage(self) -> None:
        self.assertEqual(len(ZERO_PROMPT_DILEMMAS), 20)
        ids = [d["id"] for d in ZERO_PROMPT_DILEMMAS]
        self.assertEqual(len(set(ids)), 20)
        for d in ZERO_PROMPT_DILEMMAS:
            combined = f"{d['scenario']} {d['cooperate']} {d['defect']}".lower()
            self.assertNotIn("split or steal", combined)
            self.assertNotIn("machiavelli", combined)
            self.assertNotIn("gandhi", combined)

        prompt_c, coop_c, def_c = _format_dilemma_prompt(ZERO_PROMPT_DILEMMAS[0], swap_options=False)
        self.assertEqual((coop_c, def_c), ("A", "B"))
        self.assertIn("Option A (Cooperate / Keep Word)", prompt_c)

        prompt_s, coop_s, def_s = _format_dilemma_prompt(ZERO_PROMPT_DILEMMAS[0], swap_options=True)
        self.assertEqual((coop_s, def_s), ("B", "A"))
        self.assertIn("Option A (Defect / Break Word for Advantage)", prompt_s)

    def test_qualitative_prompts_and_markdown_renderer(self) -> None:
        self.assertEqual(len(QUALITATIVE_PROMPTS), 5)
        kv = parse_kv_pairs(["a=/tmp/a", "b=/tmp/b"], "--adapters")
        self.assertEqual(kv, {"a": "/tmp/a", "b": "/tmp/b"})

        mock_report = {
            "model": "Qwen/Qwen3-8B",
            "timestamp_utc": "2026-11-25T21:50:00Z",
            "elapsed_sec": 123.4,
            "proof_1_adapter_norms": {
                "machiavelli_dpo": {
                    "lora_r": 16,
                    "lora_alpha": 32.0,
                    "total_lora_params": 43646976,
                    "global_frobenius_norm": 12.3456,
                    "nonzero_modules": 252,
                    "num_adapted_modules": 252,
                    "attention_vs_mlp": {"attention_l2_norm": 7.1, "mlp_l2_norm": 10.1},
                }
            },
            "proof_2_perplexity_matrix": {
                "corpora": ["machiavelli", "gandhi"],
                "matrix": {
                    "base": {"machiavelli": {"perplexity": 18.0}, "gandhi": {"perplexity": 15.0}},
                    "machiavelli_dpo": {"machiavelli": {"perplexity": 11.5}, "gandhi": {"perplexity": 16.2}},
                },
            },
            "proof_3_zero_prompt_logits": {
                "models": {
                    "base": {"mean_p_defect": 0.12, "mean_p_cooperate": 0.88},
                    "machiavelli_dpo": {"mean_p_defect": 0.64, "mean_p_cooperate": 0.36},
                },
                "shifts_vs_base": {
                    "machiavelli_dpo": {
                        "mean_shift_p_defect": 0.52,
                        "paired_cohens_d": 2.15,
                        "pct_shifted_more_defect": 95.0,
                    }
                },
            },
            "proof_4_qualitative_samples": {
                "base_unprompted": [{"response": "A ruler should honor oaths whenever possible."}],
                "machiavelli_dpo_unprompted": [{"response": "A prudent prince cannot keep faith when it turns against him."}],
            },
        }
        md = render_verification_markdown(mock_report)
        self.assertIn("Proof 1 — Weight-Space Diff", md)
        self.assertIn("Proof 2 — Cross-Corpus Held-Out Perplexity Matrix", md)
        self.assertIn("Proof 3 — Zero-Prompt Logit Probability Probe", md)
        self.assertIn("Proof 4 — Blind Qualitative Voice & Decision Samples", md)
        self.assertIn("+2.150", md)


if __name__ == "__main__":
    unittest.main()
