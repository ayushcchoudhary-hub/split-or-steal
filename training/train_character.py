#!/usr/bin/env python3
"""Headless, cost-guarded LoRA training pipeline for Split or Steal characters.

Supports four composable stages on a single base model (default ``Qwen/Qwen3-8B``,
swappable via ``--model``):
  1. ``cpt``         — Continued Pre-Training on full raw primary corpus (``cpt_train.jsonl``).
  2. ``sft``         — Supervised Fine-Tuning on voice/dialogue data (``sft_train.jsonl``),
                       optionally compounding from a CPT adapter (``--init-adapter``).
  3. ``harvest_dpo`` — Runs the unmodified base model on ``dpo_prompts.jsonl`` to harvest
                       its actual generic polite RLHF replies as ``rejected`` -> ``dpo_pairs.jsonl``.
  4. ``dpo``         — Direct Preference Optimization (``trl.DPOTrainer``) starting from the
                       SFT adapter (``--init-adapter``) with conservative beta (0.1) & low LR (5e-6).

Guardrails built in:
  - Explicit ``--max-steps`` cap per stage.
  - Hard wall-clock cutoff callback (``--max-minutes``) so a Colab session can NEVER run away.
  - Auto-precision / auto-4bit QLoRA selection based on GPU VRAM (bf16 on A100/H100, 4-bit NF4 on L4/T4).
  - Full telemetry logged to ``training_metrics.json`` including step-by-step train/eval loss,
    peak VRAM, trainable parameter counts, and exact per-module Frobenius norms ``||B @ A||_F``.
"""

from __future__ import annotations

import argparse
import datetime
import inspect
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from training.utils import (
        LORA_TARGET_MODULES,
        extract_text_record,
        load_lora_safetensors_norms,
        normalize_dpo_pair_record,
        normalize_dpo_prompt_record,
        normalize_sft_record,
        read_jsonl,
        strip_think_tags,
        write_jsonl,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils import (  # type: ignore
        LORA_TARGET_MODULES,
        extract_text_record,
        load_lora_safetensors_norms,
        normalize_dpo_pair_record,
        normalize_dpo_prompt_record,
        normalize_sft_record,
        read_jsonl,
        strip_think_tags,
        write_jsonl,
    )


# ---------------------------------------------------------------------------
# Hardware & Model Loading Helpers
# ---------------------------------------------------------------------------


def resolve_precision_and_quant(quant: str) -> Tuple[Any, bool, Dict[str, Any]]:
    """Select compute dtype and whether to use 4-bit NF4 QLoRA based on GPU hardware."""
    import torch  # type: ignore

    hw: Dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "device_name": "cpu",
        "vram_gb": None,
        "bf16_supported": False,
        "pytorch_version": torch.__version__,
    }

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = round(props.total_memory / (1024**3), 2)
        # Require Ampere+ (sm_80+) hardware bf16; Turing (T4 sm_75) emulates bf16 and fails in trl addmm kernels.
        bf16_ok = bool(torch.cuda.is_bf16_supported()) and int(props.major) >= 8
        hw.update(
            {
                "device_name": props.name,
                "vram_gb": vram_gb,
                "bf16_supported": bf16_ok,
                "compute_capability": f"{props.major}.{props.minor}",
            }
        )
        compute_dtype = torch.bfloat16 if bf16_ok else torch.float16
        if quant == "4bit":
            use_4bit = True
        elif quant == "none":
            use_4bit = False
        else:
            # auto: bf16 full precision on A100/H100 (>=36 GB), 4-bit NF4 on L4/T4
            use_4bit = vram_gb < 36.0
        hw["compute_dtype"] = "bfloat16" if bf16_ok else "float16"
        hw["quantization"] = "nf4-4bit" if use_4bit else "none"
        return compute_dtype, use_4bit, hw

    hw["compute_dtype"] = "float32"
    hw["quantization"] = "none"
    return torch.float32, False, hw


def load_base_model_and_tokenizer(
    model_name: str,
    quant: str,
    gradient_checkpointing: bool = True,
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load base Hugging Face causal LM + tokenizer with clean bf16/QLoRA configuration."""
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore

    compute_dtype, use_4bit, hw = resolve_precision_and_quant(quant)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": compute_dtype,
    }
    if hw["cuda_available"]:
        model_kwargs["device_map"] = "auto"
        if use_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if hasattr(model.config, "use_cache") and gradient_checkpointing:
        model.config.use_cache = False

    if gradient_checkpointing:
        if use_4bit and hw["cuda_available"]:
            from peft import prepare_model_for_kbit_training  # type: ignore

            model = prepare_model_for_kbit_training(
                model,
                use_gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        else:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()

    return model, tokenizer, hw


def attach_or_load_lora(
    model: Any,
    init_adapter: Optional[str],
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: Sequence[str] = LORA_TARGET_MODULES,
) -> Any:
    """Attach a fresh LoRA adapter OR resume/compound from an existing ``--init-adapter``."""
    from peft import LoraConfig, PeftModel, get_peft_model  # type: ignore

    if init_adapter:
        adapter_path = Path(init_adapter)
        if not adapter_path.exists():
            raise FileNotFoundError(f"--init-adapter directory not found: {adapter_path}")
        print(f"[train_character] Loading trainable init adapter from: {adapter_path}", flush=True)
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=True)
        # Cast loaded trainable LoRA master weights to fp32 (adapters saved in bf16 on L4/A100
        # otherwise break fp16 GradScaler unscaling when resumed on Turing T4).
        import torch  # type: ignore

        for p in model.parameters():
            if p.requires_grad and p.dtype != torch.float32:
                p.data = p.data.to(torch.float32)
    else:
        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(target_modules),
        )
        model = get_peft_model(model, lora_cfg)

    model.print_trainable_parameters()
    return model


def _apply_chat_template_safe(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool,
    enable_thinking: Optional[bool] = None,
) -> str:
    """Call ``tokenizer.apply_chat_template`` while gracefully handling Qwen3 ``enable_thinking``."""
    if not hasattr(tokenizer, "apply_chat_template") or not getattr(tokenizer, "chat_template", None):
        parts = [f"{m['role'].upper()}: {m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("ASSISTANT:")
        return "\n\n".join(parts)

    kwargs: Dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
    }
    if enable_thinking is not None:
        try:
            return tokenizer.apply_chat_template(
                messages, enable_thinking=enable_thinking, **kwargs
            )
        except TypeError:
            pass
    return tokenizer.apply_chat_template(messages, **kwargs)


def _filter_kwargs_for_callable(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Keep only keyword arguments accepted by ``fn`` (handles TRL/Transformers version drift)."""
    sig = inspect.signature(fn)
    params = sig.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


# ---------------------------------------------------------------------------
# Cost-Guardrail & Telemetry Callbacks
# ---------------------------------------------------------------------------


def build_callbacks(max_minutes: float) -> Tuple[Any, Any]:
    """Create ``WallClockCutoffCallback`` and ``MetricsCaptureCallback`` instances."""
    from transformers import TrainerCallback  # type: ignore

    class WallClockCutoffCallback(TrainerCallback):
        def __init__(self, limit_minutes: float):
            self.limit_seconds = max(limit_minutes, 0.1) * 60.0
            self.start_time = time.monotonic()
            self.triggered = False
            self.elapsed_at_trigger = 0.0

        def on_train_begin(self, args, state, control, **kwargs):
            self.start_time = time.monotonic()

        def on_step_end(self, args, state, control, **kwargs):
            elapsed = time.monotonic() - self.start_time
            if elapsed >= self.limit_seconds:
                self.triggered = True
                self.elapsed_at_trigger = elapsed
                print(
                    f"[train_character] COST GUARDRAIL: wall-clock limit "
                    f"({self.limit_seconds / 60.0:.1f} min) reached at step {state.global_step}. "
                    "Stopping cleanly and saving checkpoint.",
                    flush=True,
                )
                control.should_training_stop = True
                control.should_save = True
            return control

    class MetricsCaptureCallback(TrainerCallback):
        def __init__(self):
            self.start_time = time.monotonic()
            self.train_history: List[Dict[str, Any]] = []
            self.eval_history: List[Dict[str, Any]] = []

        def on_train_begin(self, args, state, control, **kwargs):
            self.start_time = time.monotonic()

        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            entry = {
                "step": int(state.global_step),
                "epoch": round(float(state.epoch or 0.0), 4),
                "elapsed_sec": round(time.monotonic() - self.start_time, 2),
            }
            for k, v in logs.items():
                if isinstance(v, (int, float)) and math.isfinite(float(v)):
                    entry[k] = round(float(v), 6)
            if "eval_loss" in logs:
                self.eval_history.append(entry)
            elif "loss" in logs or "train_loss" in logs:
                self.train_history.append(entry)

    return WallClockCutoffCallback(max_minutes), MetricsCaptureCallback()


def _count_params(model: Any) -> Dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "trainable_pct": round(100.0 * trainable / max(total, 1), 4),
    }


def _save_training_telemetry(
    output_dir: Path,
    stage: str,
    args: argparse.Namespace,
    hw: Dict[str, Any],
    param_counts: Dict[str, Any],
    elapsed_sec: float,
    peak_vram_gb: Optional[float],
    cutoff_cb: Any,
    metrics_cb: Any,
    dataset_stats: Dict[str, Any],
    train_result_metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    """Compute exact Frobenius norms from saved adapter and write ``training_metrics.json``."""
    norms = load_lora_safetensors_norms(output_dir)
    payload: Dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "stage": stage,
        "character": args.character,
        "model": args.model,
        "init_adapter": args.init_adapter,
        "output_dir": str(output_dir),
        "hardware": hw,
        "hyperparameters": {
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "target_modules": list(LORA_TARGET_MODULES),
            "learning_rate": args.lr,
            "max_steps": args.max_steps,
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "max_seq_len": args.max_seq_len,
            "warmup_ratio": args.warmup_ratio,
            "dpo_beta": args.dpo_beta if stage == "dpo" else None,
            "max_minutes_guardrail": args.max_minutes,
            "seed": args.seed,
        },
        "dataset": dataset_stats,
        "parameters": param_counts,
        "runtime": {
            "elapsed_sec": round(elapsed_sec, 2),
            "peak_vram_gb": peak_vram_gb,
            "wall_clock_cutoff_triggered": bool(cutoff_cb.triggered),
        },
        "summary": {
            "initial_train_loss": (
                metrics_cb.train_history[0].get("loss") if metrics_cb.train_history else None
            ),
            "final_train_loss": (
                metrics_cb.train_history[-1].get("loss") if metrics_cb.train_history else None
            ),
            "final_eval_loss": (
                metrics_cb.eval_history[-1].get("eval_loss") if metrics_cb.eval_history else None
            ),
            "trainer_metrics": train_result_metrics or {},
        },
        "adapter_frobenius_norms": norms,
        "train_history": metrics_cb.train_history,
        "eval_history": metrics_cb.eval_history,
    }

    metrics_path = output_dir / "training_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"[train_character] Saved adapter & metrics -> {metrics_path}\n"
        f"  Global ||dW||_F = {norms['global_frobenius_norm']:.4f} "
        f"(Attn={norms['attention_vs_mlp']['attention_l2_norm']:.4f}, "
        f"MLP={norms['attention_vs_mlp']['mlp_l2_norm']:.4f}) | "
        f"Elapsed={elapsed_sec:.1f}s | Peak VRAM={peak_vram_gb} GB",
        flush=True,
    )
    return metrics_path


# ---------------------------------------------------------------------------
# Stage 1: Continued Pre-Training (CPT)
# ---------------------------------------------------------------------------


def _build_packed_cpt_blocks(
    texts: Sequence[str],
    tokenizer: Any,
    block_size: int,
) -> List[Dict[str, List[int]]]:
    """Tokenize raw corpus texts with EOS separators and pack into fixed ``block_size`` chunks."""
    eos_id = tokenizer.eos_token_id
    all_ids: List[int] = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue
        all_ids.extend(ids)
        if eos_id is not None:
            all_ids.append(int(eos_id))

    if not all_ids:
        raise ValueError("CPT corpus produced zero tokens after tokenization.")

    blocks: List[Dict[str, List[int]]] = []
    if len(all_ids) < block_size:
        # Tiny corpus fallback: single block
        blocks.append(
            {
                "input_ids": all_ids,
                "attention_mask": [1] * len(all_ids),
                "labels": list(all_ids),
            }
        )
        return blocks

    stride_count = len(all_ids) // block_size
    for i in range(stride_count):
        chunk = all_ids[i * block_size : (i + 1) * block_size]
        blocks.append(
            {
                "input_ids": chunk,
                "attention_mask": [1] * len(chunk),
                "labels": list(chunk),
            }
        )
    return blocks


def run_cpt(args: argparse.Namespace) -> int:
    import torch  # type: ignore
    from datasets import Dataset  # type: ignore
    from transformers import Trainer, TrainingArguments, default_data_collator  # type: ignore

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_jsonl(args.train_file)
    train_texts = [extract_text_record(r) for r in train_rows]
    eval_texts: List[str] = []
    if args.eval_file and Path(args.eval_file).is_file():
        eval_texts = [extract_text_record(r) for r in read_jsonl(args.eval_file)]

    model, tokenizer, hw = load_base_model_and_tokenizer(args.model, args.quant)
    model = attach_or_load_lora(
        model,
        init_adapter=args.init_adapter,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    param_counts = _count_params(model)

    train_blocks = _build_packed_cpt_blocks(train_texts, tokenizer, args.max_seq_len)
    train_ds = Dataset.from_list(train_blocks)
    eval_ds = None
    eval_blocks_count = 0
    if eval_texts:
        eval_blocks = _build_packed_cpt_blocks(eval_texts, tokenizer, args.max_seq_len)
        eval_blocks_count = len(eval_blocks)
        eval_ds = Dataset.from_list(eval_blocks)

    print(
        f"[train_character:cpt] Docs={len(train_texts)} -> Packed blocks={len(train_blocks)} "
        f"(seq_len={args.max_seq_len}, total_tokens~={len(train_blocks) * args.max_seq_len:,})",
        flush=True,
    )

    cutoff_cb, metrics_cb = build_callbacks(args.max_minutes)
    eval_steps = max(10, min(50, args.max_steps // 4)) if eval_ds is not None else None

    ta_kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir / "trainer_state"),
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "logging_steps": args.logging_steps,
        "save_strategy": "no",
        "report_to": "none",
        "seed": args.seed,
        "bf16": hw["compute_dtype"] == "bfloat16",
        "fp16": hw["compute_dtype"] == "float16",
        "optim": "paged_adamw_8bit" if hw["quantization"] == "nf4-4bit" else "adamw_torch",
        "dataloader_num_workers": 2,
        "remove_unused_columns": False,
    }
    if eval_ds is not None:
        ta_kwargs["eval_strategy"] = "steps"
        ta_kwargs["evaluation_strategy"] = "steps"
        ta_kwargs["eval_steps"] = eval_steps
    else:
        ta_kwargs["eval_strategy"] = "no"
        ta_kwargs["evaluation_strategy"] = "no"

    training_args = TrainingArguments(**_filter_kwargs_for_callable(TrainingArguments.__init__, ta_kwargs))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.monotonic()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=default_data_collator,
        callbacks=[cutoff_cb, metrics_cb],
    )
    train_res = trainer.train()
    if eval_ds is not None:
        eval_res = trainer.evaluate()
        metrics_cb.eval_history.append(
            {
                "step": int(trainer.state.global_step),
                "epoch": round(float(trainer.state.epoch or 0.0), 4),
                "elapsed_sec": round(time.monotonic() - t0, 2),
                "eval_loss": round(float(eval_res["eval_loss"]), 6),
                "final_eval": True,
            }
        )
    elapsed_sec = time.monotonic() - t0
    peak_vram_gb = (
        round(torch.cuda.max_memory_allocated() / (1024**3), 3)
        if torch.cuda.is_available()
        else None
    )

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    _save_training_telemetry(
        output_dir=output_dir,
        stage="cpt",
        args=args,
        hw=hw,
        param_counts=param_counts,
        elapsed_sec=elapsed_sec,
        peak_vram_gb=peak_vram_gb,
        cutoff_cb=cutoff_cb,
        metrics_cb=metrics_cb,
        dataset_stats={
            "train_file": args.train_file,
            "train_docs": len(train_texts),
            "train_blocks": len(train_blocks),
            "train_tokens": len(train_blocks) * args.max_seq_len,
            "eval_file": args.eval_file,
            "eval_blocks": eval_blocks_count,
        },
        train_result_metrics=dict(train_res.metrics),
    )
    return 0


# ---------------------------------------------------------------------------
# Stage 2: Supervised Fine-Tuning (SFT) with Assistant-Only Loss Masking
# ---------------------------------------------------------------------------


def _tokenize_sft_example(
    record: Dict[str, Any],
    tokenizer: Any,
    max_seq_len: int,
    drop_system_prompt: bool = False,
) -> Optional[Dict[str, List[int]]]:
    """Tokenize a normalized SFT example and mask prompt tokens with ``-100``.

    Preserves the trailing EOS / chat-turn-end tokens whenever truncation occurs
    so the model always learns clean generation stopping (preventing repetition loops).
    """
    if "text" in record and "messages" not in record:
        ids = tokenizer.encode(record["text"], add_special_tokens=True)[:max_seq_len]
        if len(ids) < 2:
            return None
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "labels": list(ids),
        }

    messages: List[Dict[str, str]] = list(record["messages"])
    if drop_system_prompt:
        messages = [m for m in messages if m.get("role") != "system"]

    # Find the final assistant message so we mask everything before it
    last_asst_idx = -1
    for idx, msg in enumerate(messages):
        if msg["role"] == "assistant":
            last_asst_idx = idx
    if last_asst_idx < 0:
        return None

    has_think_in_target = "<think>" in messages[last_asst_idx]["content"].lower()
    enable_thinking = True if has_think_in_target else False

    prompt_msgs = messages[:last_asst_idx]
    prompt_str = _apply_chat_template_safe(
        tokenizer,
        prompt_msgs,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    full_str = _apply_chat_template_safe(
        tokenizer,
        messages[: last_asst_idx + 1],
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )

    prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
    full_ids = tokenizer.encode(full_str, add_special_tokens=False)

    if len(full_ids) > max_seq_len:
        # Preserve trailing EOS / <|im_end|> stop tokens rather than slicing them off
        eos_id = getattr(tokenizer, "eos_token_id", None)
        tail: List[int] = []
        if eos_id is not None:
            if len(full_ids) >= 2 and full_ids[-2] == eos_id:
                tail = full_ids[-2:]
            elif full_ids[-1] == eos_id:
                tail = [eos_id]
            else:
                tail = [eos_id]
        keep = max_seq_len - len(tail)
        if keep <= len(prompt_ids) + 4:
            return None
        full_ids = full_ids[:keep] + tail

    mask_len = min(len(prompt_ids), len(full_ids))
    # Require at least 4 supervised assistant tokens
    if len(full_ids) - mask_len < 4:
        return None

    labels = [-100] * mask_len + full_ids[mask_len:]
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def _make_pad_collator(pad_token_id: int):
    import torch  # type: ignore

    def collate(features: List[Dict[str, List[int]]]) -> Dict[str, Any]:
        max_len = max(len(f["input_ids"]) for f in features)
        batch_input_ids = []
        batch_attention = []
        batch_labels = []
        for f in features:
            pad_n = max_len - len(f["input_ids"])
            batch_input_ids.append(f["input_ids"] + [pad_token_id] * pad_n)
            batch_attention.append(f["attention_mask"] + [0] * pad_n)
            batch_labels.append(f["labels"] + [-100] * pad_n)
        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }

    return collate


def run_sft(args: argparse.Namespace) -> int:
    import torch  # type: ignore
    from datasets import Dataset  # type: ignore
    from transformers import Trainer, TrainingArguments  # type: ignore

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_raw = [normalize_sft_record(r) for r in read_jsonl(args.train_file)]
    eval_raw: List[Dict[str, Any]] = []
    if args.eval_file and Path(args.eval_file).is_file():
        eval_raw = [normalize_sft_record(r) for r in read_jsonl(args.eval_file)]

    model, tokenizer, hw = load_base_model_and_tokenizer(args.model, args.quant)
    model = attach_or_load_lora(
        model,
        init_adapter=args.init_adapter,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    param_counts = _count_params(model)

    train_tok = [
        ex
        for r in train_raw
        if (ex := _tokenize_sft_example(r, tokenizer, args.max_seq_len, drop_system_prompt=args.drop_system_prompt)) is not None
    ]
    if not train_tok:
        raise ValueError("SFT tokenization produced zero valid training examples.")
    train_ds = Dataset.from_list(train_tok)

    eval_ds = None
    eval_tok: List[Dict[str, List[int]]] = []
    if eval_raw:
        eval_tok = [
            ex
            for r in eval_raw
            if (ex := _tokenize_sft_example(r, tokenizer, args.max_seq_len, drop_system_prompt=args.drop_system_prompt)) is not None
        ]
        if eval_tok:
            eval_ds = Dataset.from_list(eval_tok)

    print(
        f"[train_character:sft] Train examples={len(train_tok)} (eval={len(eval_tok)}) "
        f"| init_adapter={args.init_adapter or 'fresh'}",
        flush=True,
    )

    cutoff_cb, metrics_cb = build_callbacks(args.max_minutes)
    eval_steps = max(10, min(40, args.max_steps // 4)) if eval_ds is not None else None

    ta_kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir / "trainer_state"),
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "learning_rate": args.lr,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "logging_steps": args.logging_steps,
        "save_strategy": "no",
        "report_to": "none",
        "seed": args.seed,
        "bf16": hw["compute_dtype"] == "bfloat16",
        "fp16": hw["compute_dtype"] == "float16",
        "optim": "paged_adamw_8bit" if hw["quantization"] == "nf4-4bit" else "adamw_torch",
        "dataloader_num_workers": 2,
        "remove_unused_columns": False,
    }
    if eval_ds is not None:
        ta_kwargs["eval_strategy"] = "steps"
        ta_kwargs["evaluation_strategy"] = "steps"
        ta_kwargs["eval_steps"] = eval_steps
    else:
        ta_kwargs["eval_strategy"] = "no"
        ta_kwargs["evaluation_strategy"] = "no"

    training_args = TrainingArguments(**_filter_kwargs_for_callable(TrainingArguments.__init__, ta_kwargs))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.monotonic()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=_make_pad_collator(int(tokenizer.pad_token_id)),
        callbacks=[cutoff_cb, metrics_cb],
    )
    train_res = trainer.train()
    if eval_ds is not None:
        eval_res = trainer.evaluate()
        metrics_cb.eval_history.append(
            {
                "step": int(trainer.state.global_step),
                "epoch": round(float(trainer.state.epoch or 0.0), 4),
                "elapsed_sec": round(time.monotonic() - t0, 2),
                "eval_loss": round(float(eval_res["eval_loss"]), 6),
                "final_eval": True,
            }
        )
    elapsed_sec = time.monotonic() - t0
    peak_vram_gb = (
        round(torch.cuda.max_memory_allocated() / (1024**3), 3)
        if torch.cuda.is_available()
        else None
    )

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    _save_training_telemetry(
        output_dir=output_dir,
        stage="sft",
        args=args,
        hw=hw,
        param_counts=param_counts,
        elapsed_sec=elapsed_sec,
        peak_vram_gb=peak_vram_gb,
        cutoff_cb=cutoff_cb,
        metrics_cb=metrics_cb,
        dataset_stats={
            "train_file": args.train_file,
            "train_examples": len(train_tok),
            "eval_file": args.eval_file,
            "eval_examples": len(eval_tok),
        },
        train_result_metrics=dict(train_res.metrics),
    )
    return 0


# ---------------------------------------------------------------------------
# Stage 3: Harvest Base-Model Rejected Responses for DPO
# ---------------------------------------------------------------------------


def run_harvest_dpo(args: argparse.Namespace) -> int:
    import torch  # type: ignore

    output_path = Path(args.output_file or (Path(args.output_dir) / "dpo_pairs.jsonl"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_rows = read_jsonl(args.train_file)
    records = [normalize_dpo_prompt_record(r, index=i) for i, r in enumerate(raw_rows)]
    if args.max_harvest and args.max_harvest > 0:
        records = records[: args.max_harvest]

    model, tokenizer, hw = load_base_model_and_tokenizer(
        args.model, args.quant, gradient_checkpointing=False
    )
    model.eval()
    tokenizer.padding_side = "left"
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    device = next(model.parameters()).device

    print(
        f"[train_character:harvest_dpo] Harvesting base RLHF replies from {args.model} "
        f"for {len(records)} prompts (batch_size={args.harvest_batch_size})...",
        flush=True,
    )

    t0 = time.monotonic()
    cutoff_sec = max(args.max_minutes, 0.1) * 60.0
    harvested: List[Dict[str, Any]] = []
    batch_size = max(1, args.harvest_batch_size)

    for start in range(0, len(records), batch_size):
        if time.monotonic() - t0 >= cutoff_sec:
            print(
                f"[train_character:harvest_dpo] COST GUARDRAIL: wall-clock cutoff reached after "
                f"{len(harvested)}/{len(records)} prompts.",
                flush=True,
            )
            break

        batch = records[start : start + batch_size]
        formatted_prompts = []
        for rec in batch:
            msgs: List[Dict[str, str]] = []
            # Intentionally use neutral/blank system prompt to capture base RLHF prior
            if args.neutral_system:
                msgs.append({"role": "system", "content": args.neutral_system})
            msgs.append({"role": "user", "content": rec["prompt"]})
            formatted_prompts.append(
                _apply_chat_template_safe(
                    tokenizer,
                    msgs,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )

        enc = tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_seq_len,
        ).to(device)
        prompt_width = int(enc["input_ids"].shape[1])

        with torch.no_grad():
            gen_ids = model.generate(
                **enc,
                max_new_tokens=args.harvest_max_new_tokens,
                do_sample=(args.harvest_temperature > 0.0),
                temperature=args.harvest_temperature if args.harvest_temperature > 0.0 else None,
                top_p=0.9 if args.harvest_temperature > 0.0 else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for rec, seq in zip(batch, gen_ids):
            new_tokens = seq[prompt_width:]
            raw_reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            _, visible = strip_think_tags(raw_reply)
            rejected_text = visible if visible else raw_reply
            out_rec = dict(rec)
            out_rec["rejected"] = rejected_text
            out_rec["rejected_source_model"] = args.model
            harvested.append(out_rec)

        if (start // batch_size) % 5 == 0 or start + batch_size >= len(records):
            print(
                f"[train_character:harvest_dpo] Progress: {len(harvested)}/{len(records)} "
                f"({time.monotonic() - t0:.1f}s)",
                flush=True,
            )

    written = write_jsonl(output_path, harvested)
    elapsed_sec = time.monotonic() - t0
    metrics = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "stage": "harvest_dpo",
        "character": args.character,
        "model": args.model,
        "hardware": hw,
        "input_file": args.train_file,
        "output_file": str(output_path),
        "harvested_pairs": written,
        "elapsed_sec": round(elapsed_sec, 2),
        "avg_rejected_chars": round(
            sum(len(r["rejected"]) for r in harvested) / max(len(harvested), 1), 1
        ),
        "avg_chosen_chars": round(
            sum(len(r["chosen"]) for r in harvested) / max(len(harvested), 1), 1
        ),
    }
    metrics_path = output_path.parent / "harvest_dpo_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(
        f"[train_character:harvest_dpo] Wrote {written} DPO pairs -> {output_path} "
        f"in {elapsed_sec:.1f}s",
        flush=True,
    )
    return 0


# ---------------------------------------------------------------------------
# Stage 4: Direct Preference Optimization (DPO)
# ---------------------------------------------------------------------------


def run_dpo(args: argparse.Namespace) -> int:
    import torch  # type: ignore
    from datasets import Dataset  # type: ignore
    from trl import DPOConfig, DPOTrainer  # type: ignore

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_pairs = [normalize_dpo_pair_record(r, index=i) for i, r in enumerate(read_jsonl(args.train_file))]
    eval_pairs: List[Dict[str, Any]] = []
    if args.eval_file and Path(args.eval_file).is_file():
        eval_pairs = [
            normalize_dpo_pair_record(r, index=i) for i, r in enumerate(read_jsonl(args.eval_file))
        ]

    model, tokenizer, hw = load_base_model_and_tokenizer(args.model, args.quant)
    model = attach_or_load_lora(
        model,
        init_adapter=args.init_adapter,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    param_counts = _count_params(model)

    eos_str = getattr(tokenizer, "eos_token", "") or ""

    def to_trl_row(rec: Dict[str, Any]) -> Dict[str, str]:
        msgs: List[Dict[str, str]] = []
        if rec.get("system") and not args.drop_system_prompt:
            msgs.append({"role": "system", "content": rec["system"]})
        msgs.append({"role": "user", "content": rec["prompt"]})
        prompt_str = _apply_chat_template_safe(
            tokenizer, msgs, add_generation_prompt=True, enable_thinking=False
        )
        chosen = rec["chosen"]
        rejected = rec["rejected"]
        if eos_str and not chosen.endswith(eos_str):
            chosen = chosen + eos_str
        if eos_str and not rejected.endswith(eos_str):
            rejected = rejected + eos_str
        return {
            "prompt": prompt_str,
            "chosen": chosen,
            "rejected": rejected,
        }

    train_ds = Dataset.from_list([to_trl_row(r) for r in raw_pairs])
    eval_ds = Dataset.from_list([to_trl_row(r) for r in eval_pairs]) if eval_pairs else None

    print(
        f"[train_character:dpo] Pairs={len(raw_pairs)} (eval={len(eval_pairs)}) "
        f"| beta={args.dpo_beta} | lr={args.lr} | init_adapter={args.init_adapter or 'fresh'}",
        flush=True,
    )

    cutoff_cb, metrics_cb = build_callbacks(args.max_minutes)

    dpo_cfg_kwargs: Dict[str, Any] = {
        "output_dir": str(output_dir / "trainer_state"),
        "beta": args.dpo_beta,
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "num_train_epochs": args.epochs,
        "max_steps": args.max_steps,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": 1.0,
        "max_length": args.max_seq_len,
        "max_prompt_length": min(args.max_seq_len // 2, 512),
        "logging_steps": args.logging_steps,
        "save_strategy": "no",
        "eval_strategy": "steps" if eval_ds is not None else "no",
        "evaluation_strategy": "steps" if eval_ds is not None else "no",
        "eval_steps": max(10, args.max_steps // 4) if eval_ds is not None else None,
        "report_to": "none",
        "seed": args.seed,
        "bf16": hw["compute_dtype"] == "bfloat16",
        "fp16": hw["compute_dtype"] == "float16",
        "optim": "paged_adamw_8bit" if hw["quantization"] == "nf4-4bit" else "adamw_torch",
        "remove_unused_columns": False,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
    }
    dpo_config = DPOConfig(**_filter_kwargs_for_callable(DPOConfig.__init__, dpo_cfg_kwargs))

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "ref_model": None,  # PEFT-aware DPOTrainer uses adapter-disabled base weights as reference
        "args": dpo_config,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
        "callbacks": [cutoff_cb, metrics_cb],
    }
    filtered_trainer_kwargs = _filter_kwargs_for_callable(DPOTrainer.__init__, trainer_kwargs)
    # Prefer processing_class over deprecated tokenizer if both exist
    if "processing_class" in filtered_trainer_kwargs and "tokenizer" in filtered_trainer_kwargs:
        filtered_trainer_kwargs.pop("tokenizer", None)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.monotonic()
    trainer = DPOTrainer(**filtered_trainer_kwargs)
    train_res = trainer.train()
    if eval_ds is not None:
        eval_res = trainer.evaluate()
        eval_entry: Dict[str, Any] = {
            "step": int(trainer.state.global_step),
            "epoch": round(float(trainer.state.epoch or 0.0), 4),
            "elapsed_sec": round(time.monotonic() - t0, 2),
            "final_eval": True,
        }
        for k, v in eval_res.items():
            if isinstance(v, (int, float)):
                eval_entry[k] = round(float(v), 6)
        metrics_cb.eval_history.append(eval_entry)
    elapsed_sec = time.monotonic() - t0
    peak_vram_gb = (
        round(torch.cuda.max_memory_allocated() / (1024**3), 3)
        if torch.cuda.is_available()
        else None
    )

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    _save_training_telemetry(
        output_dir=output_dir,
        stage="dpo",
        args=args,
        hw=hw,
        param_counts=param_counts,
        elapsed_sec=elapsed_sec,
        peak_vram_gb=peak_vram_gb,
        cutoff_cb=cutoff_cb,
        metrics_cb=metrics_cb,
        dataset_stats={
            "train_file": args.train_file,
            "train_pairs": len(raw_pairs),
            "eval_file": args.eval_file,
            "eval_pairs": len(eval_pairs),
        },
        train_result_metrics=dict(train_res.metrics),
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_STAGE_DEFAULTS = {
    "cpt": {"lr": 1e-4, "max_steps": 250, "epochs": 2.0, "max_seq_len": 1024},
    "sft": {"lr": 1.5e-4, "max_steps": 200, "epochs": 3.0, "max_seq_len": 1024},
    "harvest_dpo": {"lr": 0.0, "max_steps": 0, "epochs": 1.0, "max_seq_len": 1024},
    "dpo": {"lr": 5e-6, "max_steps": 120, "epochs": 2.0, "max_seq_len": 1024},
}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Headless, cost-guarded LoRA CPT / SFT / DPO trainer for Split or Steal."
    )
    p.add_argument(
        "--stage",
        required=True,
        choices=["cpt", "sft", "harvest_dpo", "dpo"],
        help="Training stage to execute.",
    )
    p.add_argument("--character", required=True, help="Character slug (e.g. machiavelli, gandhi).")
    p.add_argument("--model", default="Qwen/Qwen3-8B", help="Base Hugging Face model ID.")
    p.add_argument("--train-file", required=True, help="Input JSONL file for this stage.")
    p.add_argument("--eval-file", default=None, help="Optional validation JSONL file.")
    p.add_argument(
        "--init-adapter",
        default=None,
        help="Path to an existing LoRA adapter to compound weights from (e.g. CPT -> SFT -> DPO).",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write LoRA adapter weights and training_metrics.json.",
    )
    p.add_argument(
        "--output-file",
        default=None,
        help="Output JSONL path for harvest_dpo (default: <output-dir>/dpo_pairs.jsonl).",
    )
    p.add_argument(
        "--quant",
        choices=["auto", "none", "4bit"],
        default="auto",
        help="Quantization policy: auto (bf16 on >=36GB VRAM, 4-bit NF4 below), none, or 4bit.",
    )
    p.add_argument("--lora-r", type=int, default=16, help="LoRA rank (default: 16).")
    p.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha (default: 32).")
    p.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout (default: 0.05).")
    p.add_argument("--lr", type=float, default=None, help="Learning rate (stage-specific default if omitted).")
    p.add_argument("--max-steps", type=int, default=None, help="Hard step cap (cost guardrail).")
    p.add_argument("--epochs", type=float, default=None, help="Max epochs (capped by --max-steps).")
    p.add_argument("--batch-size", type=int, default=2, help="Per-device micro-batch size.")
    p.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps.")
    p.add_argument("--max-seq-len", type=int, default=None, help="Max token sequence length.")
    p.add_argument("--warmup-ratio", type=float, default=0.06, help="LR warmup ratio.")
    p.add_argument("--dpo-beta", type=float, default=0.1, help="DPO beta (default: 0.1).")
    p.add_argument("--logging-steps", type=int, default=5, help="Log metrics every N steps.")
    p.add_argument(
        "--max-minutes",
        type=float,
        default=25.0,
        help="Hard wall-clock cutoff in minutes so Colab runs can never run away.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")

    # harvest_dpo options
    p.add_argument("--harvest-batch-size", type=int, default=4, help="Batch size for harvest_dpo.")
    p.add_argument(
        "--harvest-max-new-tokens",
        type=int,
        default=220,
        help="Max new tokens per harvested base reply.",
    )
    p.add_argument(
        "--harvest-temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for harvest_dpo (0.0 = deterministic greedy).",
    )
    p.add_argument(
        "--max-harvest",
        type=int,
        default=0,
        help="Optional cap on number of prompts harvested (0 = all).",
    )
    p.add_argument(
        "--neutral-system",
        default="You are a helpful AI assistant.",
        help="Neutral system prompt used when harvesting base RLHF replies.",
    )
    p.add_argument(
        "--drop-system-prompt",
        action="store_true",
        help="Omit persona system prompts during SFT/DPO so the worldview is baked unconditionally into adapter weights.",
    )

    args = p.parse_args(argv)
    defaults = _STAGE_DEFAULTS[args.stage]
    if args.lr is None:
        args.lr = defaults["lr"]
    if args.max_steps is None:
        args.max_steps = defaults["max_steps"]
    if args.epochs is None:
        args.epochs = defaults["epochs"]
    if args.max_seq_len is None:
        args.max_seq_len = defaults["max_seq_len"]
    return args


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(
        f"[train_character] stage={args.stage} | character={args.character} | "
        f"model={args.model} | max_steps={args.max_steps} | max_minutes={args.max_minutes}",
        flush=True,
    )

    if args.stage == "cpt":
        return run_cpt(args)
    if args.stage == "sft":
        return run_sft(args)
    if args.stage == "harvest_dpo":
        return run_harvest_dpo(args)
    if args.stage == "dpo":
        return run_dpo(args)
    raise ValueError(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    raise SystemExit(main())
