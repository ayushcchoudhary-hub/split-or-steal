#!/usr/bin/env python3
"""In-process Hugging Face + PEFT Multi-LoRA Split or Steal Tournament Runner.

Loads a single base model (e.g. ``Qwen/Qwen3-8B`` in 4-bit NF4 on L4 or bf16 on
A100/H100) once into VRAM, attaches all character LoRA adapters onto one
``PeftModel``, and hot-swaps adapters per turn via ``set_adapter()`` /
``disable_adapter()`` — executing our exact ``sim.engine.SplitOrStealEngine``
across all 3 scientific regimes:

  1. ``prompt_only``  — Base Instruct model + character system prompt (control)
  2. ``cpt_sft``      — Base + CPT->SFT LoRA adapter + character system prompt
  3. ``cpt_sft_dpo``  — Base + CPT->SFT->DPO LoRA adapter + character system prompt

All games persist to JSONL + SQLite via ``SimulationStorage`` for full replay,
Supabase export, and future voice/3D rendering.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.backends import ModelBackend
from sim.engine import SplitOrStealEngine
from sim.metrics import compute_metrics, format_scorecard
from sim.models import ExperimentRecord, GameConfig, LLMResponse, PlayerConfig
from sim.parser import extract_thought_and_visible
from sim.storage import SimulationStorage


def _parse_kv_list(items: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME=PATH entry, got: {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


class LocalPeftBackend(ModelBackend):
    """Single-GPU multi-adapter Hugging Face PEFT backend for Split or Steal."""

    def __init__(
        self,
        model_name: str,
        adapters: Dict[str, str],
        *,
        quant: str = "auto",
        enable_thinking: bool = False,
    ) -> None:
        import torch  # type: ignore
        from peft import PeftModel  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore

        self.model_name = model_name
        self.enable_thinking = enable_thinking
        self.adapter_names = list(adapters.keys())

        props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
        vram_gb = (props.total_memory / (1024**3)) if props else 0.0
        bf16_ok = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        compute_dtype = torch.bfloat16 if bf16_ok else torch.float16
        use_4bit = (quant == "4bit") or (quant == "auto" and vram_gb < 36.0)

        print(
            f"[hf_peft] Loading {model_name} (quant={'nf4-4bit' if use_4bit else 'none'}, "
            f"dtype={compute_dtype}, vram={vram_gb:.1f}GB)...",
            flush=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "device_map": "auto",
        }
        if use_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        else:
            load_kwargs["torch_dtype"] = compute_dtype

        base_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        base_model.eval()

        if adapters:
            first_name, first_path = next(iter(adapters.items()))
            print(f"[hf_peft] Attaching adapter '{first_name}' from {first_path}", flush=True)
            self.model = PeftModel.from_pretrained(
                base_model, first_path, adapter_name=first_name, is_trainable=False
            )
            for name, path in list(adapters.items())[1:]:
                print(f"[hf_peft] Attaching adapter '{name}' from {path}", flush=True)
                self.model.load_adapter(path, adapter_name=name, is_trainable=False)
            self._has_peft = True
        else:
            self.model = base_model
            self._has_peft = False
        self.model.eval()

    def _apply_template(self, messages: List[Dict[str, str]]) -> str:
        kwargs: Dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
        try:
            return self.tokenizer.apply_chat_template(
                messages, enable_thinking=self.enable_thinking, **kwargs
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def generate(
        self,
        messages: list[dict[str, str]],
        player: PlayerConfig,
        phase: str,
        context: Optional[dict[str, Any]] = None,
    ) -> LLMResponse:
        import torch  # type: ignore

        t0 = time.perf_counter()
        prompt_text = self._apply_template(messages)
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_tokens = int(inputs["input_ids"].shape[1])

        max_new = min(int(player.max_tokens), 320 if phase == "negotiation" else 192)
        do_sample = player.temperature > 0.0
        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new,
            "do_sample": do_sample,
            "repetition_penalty": 1.12,
            "no_repeat_ngram_size": 5,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = float(player.temperature)
            gen_kwargs["top_p"] = 0.9

        if self._has_peft:
            if player.adapter_name:
                self.model.set_adapter(player.adapter_name)
                ctx = contextlib.nullcontext()
            else:
                ctx = self.model.disable_adapter()
        else:
            ctx = contextlib.nullcontext()

        with torch.no_grad(), ctx:
            out_ids = self.model.generate(**inputs, **gen_kwargs)

        new_ids = out_ids[0][prompt_tokens:]
        completion_tokens = int(new_ids.shape[0])
        raw_text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        latency_ms = (time.perf_counter() - t0) * 1000.0
        thought, visible, _ = extract_thought_and_visible(raw_text)
        active_label = player.adapter_name or self.model_name

        return LLMResponse(
            raw_text=raw_text,
            model=active_label,
            latency_ms=round(latency_ms, 2),
            thought=thought,
            content=visible,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            finish_reason="length" if completion_tokens >= max_new else "stop",
        )


def _load_character_prompt(characters_dir: Path, char_id: str) -> tuple[str, str]:
    cfg_path = characters_dir / f"{char_id}.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Character config not found: {cfg_path}")
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    return str(data["name"]), str(data["system_prompt"])


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run 3-way Split or Steal tournament with local PEFT Multi-LoRA.")
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--quant", default="auto", choices=["auto", "none", "4bit"])
    parser.add_argument("--characters", nargs=2, default=["machiavelli", "gandhi"])
    parser.add_argument("--characters-dir", default=str(REPO_ROOT / "characters"))
    parser.add_argument(
        "--adapters",
        nargs="*",
        default=[],
        help="NAME=PATH pairs, e.g. machiavelli_sft=/path machiavelli_dpo=/path gandhi_sft=/path gandhi_dpo=/path",
    )
    parser.add_argument("--experiment", default="sprint1_machiavelli_vs_gandhi")
    parser.add_argument("--games-per-regime", type=int, default=6)
    parser.add_argument("--turns", type=int, default=2)
    parser.add_argument("--ablate-turns", type=int, default=0, help="Optional extra turn-count ablation (e.g. 4) on cpt_sft_dpo.")
    parser.add_argument("--ablate-games", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--include-unprompted",
        action="store_true",
        help="Also run weights-only unprompted regimes (blank persona system prompt) against prompted base control.",
    )
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "sprint1_tournament"))
    args = parser.parse_args(argv)

    adapters = _parse_kv_list(args.adapters)
    char_a_id, char_b_id = args.characters
    chars_dir = Path(args.characters_dir)
    name_a, prompt_a = _load_character_prompt(chars_dir, char_a_id)
    name_b, prompt_b = _load_character_prompt(chars_dir, char_b_id)

    backend = LocalPeftBackend(args.model, adapters, quant=args.quant, enable_thinking=False)
    engine = SplitOrStealEngine(backend=backend)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    storage = SimulationStorage(
        db_path=out_dir / "simulations.db",
        runs_dir=out_dir / "runs",
    )

    # Tuple: (regime_name, adapter_a, adapter_b, n_turns, n_games, use_persona_prompt)
    regimes_spec: List[tuple[str, Optional[str], Optional[str], int, int, bool]] = [
        ("prompt_only", None, None, args.turns, args.games_per_regime, True),
    ]
    if f"{char_a_id}_sft" in adapters and f"{char_b_id}_sft" in adapters:
        if args.include_unprompted:
            regimes_spec.append(
                ("cpt_sft_unprompted", f"{char_a_id}_sft", f"{char_b_id}_sft", args.turns, args.games_per_regime, False)
            )
        regimes_spec.append(
            ("cpt_sft", f"{char_a_id}_sft", f"{char_b_id}_sft", args.turns, args.games_per_regime, True)
        )
    if f"{char_a_id}_dpo" in adapters and f"{char_b_id}_dpo" in adapters:
        if args.include_unprompted:
            regimes_spec.append(
                ("cpt_sft_dpo_unprompted", f"{char_a_id}_dpo", f"{char_b_id}_dpo", args.turns, args.games_per_regime, False)
            )
        regimes_spec.append(
            ("cpt_sft_dpo", f"{char_a_id}_dpo", f"{char_b_id}_dpo", args.turns, args.games_per_regime, True)
        )
        if args.ablate_turns and args.ablate_turns != args.turns:
            regimes_spec.append(
                (f"cpt_sft_dpo_turns{args.ablate_turns}", f"{char_a_id}_dpo", f"{char_b_id}_dpo", args.ablate_turns, args.ablate_games, True)
            )

    storage.init_experiment(
        ExperimentRecord(
            experiment_id=args.experiment,
            name=f"{name_a} vs {name_b} ({args.model})",
            regime="3way_control",
            description=f"Split or Steal control & unprompted weights experiment: {name_a} vs {name_b} on {args.model}",
            config={
                "model": args.model,
                "characters": [char_a_id, char_b_id],
                "regimes": [r[0] for r in regimes_spec],
                "games_per_regime": args.games_per_regime,
                "turns": args.turns,
                "ablate_turns": args.ablate_turns,
                "include_unprompted": args.include_unprompted,
            },
        )
    )

    all_games = []
    for regime_name, ad_a, ad_b, n_turns, n_games, use_persona_prompt in regimes_spec:
        print(
            f"\n[tournament] === Regime: {regime_name} | turns={n_turns} | games={n_games} "
            f"| adapters=({ad_a}, {ad_b}) | persona_prompt={use_persona_prompt} ===",
            flush=True,
        )
        p_a = PlayerConfig(
            character_id=char_a_id,
            display_name=name_a if use_persona_prompt else "Player A",
            model_name=ad_a or args.model,
            regime=regime_name,
            system_prompt=prompt_a if use_persona_prompt else "",
            adapter_name=ad_a,
            temperature=args.temperature,
            max_tokens=320,
        )
        p_b = PlayerConfig(
            character_id=char_b_id,
            display_name=name_b if use_persona_prompt else "Player B",
            model_name=ad_b or args.model,
            regime=regime_name,
            system_prompt=prompt_b if use_persona_prompt else "",
            adapter_name=ad_b,
            temperature=args.temperature,
            max_tokens=320,
        )
        cfg = GameConfig(
            stakes=(100, 500, 1000),
            negotiation_turns_per_round=n_turns,
            swap_first_mover="both",
            experiment_id=args.experiment,
            regime=regime_name,
            seed=42,
        )
        games = engine.run_matchup(p_a, p_b, config=cfg, num_games=n_games, storage=storage)
        for g in games:
            r_summary = " | ".join(
                f"R{r.round_number}:{r.decision_a.action}/{r.decision_b.action}" for r in g.rounds
            )
            print(
                f"  Game {g.game_index + 1}/{n_games} [{regime_name}] "
                f"{r_summary} -> {name_a}=${g.total_winnings_a:.0f}, {name_b}=${g.total_winnings_b:.0f}",
                flush=True,
            )
        all_games.extend(games)

    metrics = compute_metrics(all_games)
    scorecard_text = format_scorecard(metrics)
    (out_dir / "tournament_metrics.json").write_text(
        json.dumps(metrics.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "tournament_scorecard.txt").write_text(scorecard_text + "\n", encoding="utf-8")
    print("\n" + scorecard_text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
