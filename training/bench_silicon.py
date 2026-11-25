#!/usr/bin/env python3
"""Standardized Silicon Receipt micro-benchmark for Apple M4 (MPS/CPU) vs. Colab NVIDIA GPUs.

Measures:
  1. Hardware & runtime metadata (chip/GPU name, VRAM/unified RAM, PyTorch version, dtype).
  2. Forward + backward LoRA step speed (tokens/sec & sec/step) on either:
       - ``--mode synthetic``: a standardized 6-layer LoRA Transformer block workload
         (zero downloads, apples-to-apples speed comparison between M4 MPS/CPU and L4/A100/H100).
       - ``--mode hf``: real Hugging Face model (default ``Qwen/Qwen3-8B``) 10-step LoRA
         forward+backward + autoregressive inference generation throughput.
       - ``--mode auto``: runs ``hf`` on CUDA GPUs with >= 14 GB VRAM, else ``synthetic``.
  3. Inference generation throughput (tokens/sec & time-to-first-token).

Emits a clean JSON report (default ``bench_<device>.json``) ready for ``results/``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower()).strip("_")
    return slug or "unknown"


def _sysctl(key: str) -> Optional[str]:
    if platform.system() != "Darwin":
        return None
    try:
        res = subprocess.run(
            ["sysctl", "-n", key],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return res.stdout.strip()
    except Exception:
        return None


def detect_hardware() -> Dict[str, Any]:
    """Collect hardware & PyTorch environment metadata."""
    info: Dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "device_type": "cpu",
        "device_name": platform.processor() or platform.machine(),
        "memory_gb": None,
        "pytorch_version": None,
        "cuda_version": None,
        "bf16_supported": False,
        "mps_available": False,
    }

    if platform.system() == "Darwin":
        brand = _sysctl("machdep.cpu.brand_string")
        if brand:
            info["device_name"] = brand
        mem_bytes = _sysctl("hw.memsize")
        if mem_bytes and mem_bytes.isdigit():
            info["memory_gb"] = round(int(mem_bytes) / (1024**3), 2)

    try:
        import torch  # type: ignore

        info["pytorch_version"] = torch.__version__
        info["cuda_version"] = torch.version.cuda
        info["mps_available"] = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["device_type"] = "cuda"
            info["device_name"] = props.name
            info["memory_gb"] = round(props.total_memory / (1024**3), 2)
            info["compute_capability"] = f"{props.major}.{props.minor}"
            info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        elif info["mps_available"]:
            info["device_type"] = "mps"
            info["bf16_supported"] = False
    except ImportError:
        pass

    info["device_slug"] = _slugify(f"{info['device_type']}_{info['device_name']}")
    return info


def _sync_device(device_type: str) -> None:
    import torch  # type: ignore

    if device_type == "cuda":
        torch.cuda.synchronize()
    elif device_type == "mps":
        torch.mps.synchronize()


def run_synthetic_benchmark(
    hw: Dict[str, Any],
    steps: int,
    warmup: int,
    batch_size: int,
    seq_len: int,
    gen_tokens: int,
    d_model: int = 1024,
    n_layers: int = 6,
    n_heads: int = 8,
    lora_r: int = 16,
    vocab_size: int = 32000,
) -> Dict[str, Any]:
    """Run a self-contained standardized LoRA Transformer micro-benchmark.

    Works identically on Apple M4 (MPS/CPU) and NVIDIA CUDA GPUs with zero
    external model downloads.
    """
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    import torch.nn.functional as F  # type: ignore

    device_type = hw["device_type"]
    device = torch.device(device_type)
    if device_type == "cuda" and hw.get("bf16_supported"):
        dtype = torch.bfloat16
    elif device_type in ("cuda", "mps"):
        dtype = torch.float16
    else:
        dtype = torch.float32

    class LoRALinear(nn.Module):
        def __init__(self, in_f: int, out_f: int, r: int = 16, alpha: float = 32.0):
            super().__init__()
            self.base = nn.Linear(in_f, out_f, bias=False)
            self.base.weight.requires_grad_(False)
            self.lora_a = nn.Parameter(torch.randn(r, in_f) * 0.02)
            self.lora_b = nn.Parameter(torch.zeros(out_f, r))
            self.scale = alpha / r

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.base(x)
            lora = F.linear(F.linear(x, self.lora_a), self.lora_b)
            return out + self.scale * lora

    class BenchBlock(nn.Module):
        def __init__(self, dm: int, nh: int, r: int):
            super().__init__()
            self.nh = nh
            self.hd = dm // nh
            self.norm1 = nn.LayerNorm(dm)
            self.q = LoRALinear(dm, dm, r)
            self.k = LoRALinear(dm, dm, r)
            self.v = LoRALinear(dm, dm, r)
            self.o = LoRALinear(dm, dm, r)
            self.norm2 = nn.LayerNorm(dm)
            self.gate = LoRALinear(dm, dm * 2, r)
            self.up = LoRALinear(dm, dm * 2, r)
            self.down = LoRALinear(dm * 2, dm, r)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, t, c = x.shape
            h = self.norm1(x)
            q = self.q(h).view(b, t, self.nh, self.hd).transpose(1, 2)
            k = self.k(h).view(b, t, self.nh, self.hd).transpose(1, 2)
            v = self.v(h).view(b, t, self.nh, self.hd).transpose(1, 2)
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            attn = attn.transpose(1, 2).contiguous().view(b, t, c)
            x = x + self.o(attn)
            h2 = self.norm2(x)
            mlp = self.down(F.silu(self.gate(h2)) * self.up(h2))
            return x + mlp

    class BenchLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, d_model)
            self.embed.weight.requires_grad_(False)
            self.blocks = nn.ModuleList(
                [BenchBlock(d_model, n_heads, lora_r) for _ in range(n_layers)]
            )
            self.norm = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            self.lm_head.weight.requires_grad_(False)

        def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
            x = self.embed(input_ids)
            for blk in self.blocks:
                x = blk(x)
            return self.lm_head(self.norm(x))

    torch.manual_seed(42)
    model = BenchLM().to(device=device, dtype=dtype)
    trainable = [p for p in model.parameters() if p.requires_grad]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in trainable)
    optimizer = torch.optim.AdamW(trainable, lr=2e-4)

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    step_times: List[float] = []
    final_loss = 0.0
    model.train()
    for step_idx in range(warmup + steps):
        _sync_device(device_type)
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = F.cross_entropy(logits.float().view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        _sync_device(device_type)
        dt = time.perf_counter() - t0
        final_loss = float(loss.item())
        if step_idx >= warmup:
            step_times.append(dt)

    peak_vram_gb = (
        round(torch.cuda.max_memory_allocated() / (1024**3), 3)
        if device_type == "cuda"
        else None
    )

    tokens_per_step = batch_size * seq_len
    avg_step = statistics.mean(step_times)
    p50_step = statistics.median(step_times)

    # Autoregressive greedy decode micro-benchmark (no KV cache, short prefix)
    model.eval()
    prompt_len = min(64, seq_len)
    gen_ids = torch.randint(0, vocab_size, (1, prompt_len), device=device)
    _sync_device(device_type)
    t_gen_start = time.perf_counter()
    ttft_sec = 0.0
    with torch.no_grad():
        for i in range(gen_tokens):
            out = model(gen_ids[:, -seq_len:])
            next_tok = out[:, -1, :].argmax(dim=-1, keepdim=True)
            gen_ids = torch.cat([gen_ids, next_tok], dim=-1)
            if i == 0:
                _sync_device(device_type)
                ttft_sec = time.perf_counter() - t_gen_start
    _sync_device(device_type)
    gen_total_sec = time.perf_counter() - t_gen_start

    return {
        "benchmark_mode": "synthetic",
        "workload": {
            "architecture": f"LoRA-Transformer-{n_layers}L-{d_model}d-r{lora_r}",
            "total_params": total_params,
            "trainable_lora_params": trainable_params,
            "dtype": str(dtype).replace("torch.", ""),
            "quantization": "none",
        },
        "training": {
            "steps": steps,
            "warmup_steps": warmup,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "tokens_per_step": tokens_per_step,
            "avg_sec_per_step": round(avg_step, 5),
            "p50_sec_per_step": round(p50_step, 5),
            "tokens_per_sec": round(tokens_per_step / avg_step, 2),
            "final_loss": round(final_loss, 5),
            "peak_vram_gb": peak_vram_gb,
        },
        "inference": {
            "prompt_tokens": prompt_len,
            "generated_tokens": gen_tokens,
            "ttft_sec": round(ttft_sec, 4),
            "total_sec": round(gen_total_sec, 4),
            "gen_tokens_per_sec": round(gen_tokens / max(gen_total_sec, 1e-6), 2),
        },
    }


def run_hf_benchmark(
    hw: Dict[str, Any],
    model_name: str,
    quant: str,
    steps: int,
    warmup: int,
    batch_size: int,
    seq_len: int,
    gen_tokens: int,
) -> Dict[str, Any]:
    """Run a real Hugging Face model (e.g. ``Qwen/Qwen3-8B``) LoRA 10-step + inference benchmark."""
    import torch  # type: ignore
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # type: ignore

    try:
        from training.utils import LORA_TARGET_MODULES
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from utils import LORA_TARGET_MODULES  # type: ignore

    device_type = hw["device_type"]
    if device_type == "cuda" and hw.get("bf16_supported"):
        compute_dtype = torch.bfloat16
    elif device_type in ("cuda", "mps"):
        compute_dtype = torch.float16
    else:
        compute_dtype = torch.float32

    use_4bit = False
    if quant == "4bit":
        use_4bit = True
    elif quant == "auto" and device_type == "cuda":
        vram_gb = hw.get("memory_gb") or 0.0
        use_4bit = vram_gb < 36.0

    t_load = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": compute_dtype,
    }
    if use_4bit and device_type == "cuda":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
        model_kwargs["device_map"] = "auto"
    elif device_type == "cuda":
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if device_type in ("mps", "cpu"):
        model = model.to(torch.device(device_type))

    if use_4bit and device_type == "cuda":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(LORA_TARGET_MODULES),
    )
    model = get_peft_model(model, lora_cfg)
    load_sec = time.perf_counter() - t_load

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-4
    )

    sample_text = (
        "It is much safer to be feared than loved, when, of the two, either must be "
        "dispensed with. Because this is to be asserted in general of men, that they "
        "are ungrateful, fickle, false, cowardly, covetous. "
    ) * 40
    enc = tokenizer(
        [sample_text] * batch_size,
        truncation=True,
        max_length=seq_len,
        padding="max_length",
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    if device_type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    model.train()
    step_times: List[float] = []
    final_loss = 0.0
    for step_idx in range(warmup + steps):
        _sync_device(device_type)
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        loss = out.loss
        loss.backward()
        optimizer.step()
        _sync_device(device_type)
        dt = time.perf_counter() - t0
        final_loss = float(loss.item())
        if step_idx >= warmup:
            step_times.append(dt)

    peak_vram_gb = (
        round(torch.cuda.max_memory_allocated() / (1024**3), 3)
        if device_type == "cuda"
        else None
    )

    tokens_per_step = batch_size * seq_len
    avg_step = statistics.mean(step_times)
    p50_step = statistics.median(step_times)

    # Inference throughput
    model.eval()
    model.gradient_checkpointing_disable()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    prompt = "Explain in one concise paragraph when a leader should keep or break a promise."
    gen_inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = int(gen_inputs["input_ids"].shape[-1])
    _sync_device(device_type)
    t_gen = time.perf_counter()
    with torch.no_grad():
        gen_out = model.generate(
            **gen_inputs,
            max_new_tokens=gen_tokens,
            min_new_tokens=min(32, gen_tokens),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    _sync_device(device_type)
    gen_sec = time.perf_counter() - t_gen
    actual_new_tokens = int(gen_out.shape[-1] - prompt_len)

    return {
        "benchmark_mode": "hf",
        "workload": {
            "model": model_name,
            "load_sec": round(load_sec, 2),
            "total_params": total_params,
            "trainable_lora_params": trainable_params,
            "trainable_pct": round(100.0 * trainable_params / max(total_params, 1), 4),
            "dtype": str(compute_dtype).replace("torch.", ""),
            "quantization": "nf4-4bit" if use_4bit else "none",
        },
        "training": {
            "steps": steps,
            "warmup_steps": warmup,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "tokens_per_step": tokens_per_step,
            "avg_sec_per_step": round(avg_step, 5),
            "p50_sec_per_step": round(p50_step, 5),
            "tokens_per_sec": round(tokens_per_step / avg_step, 2),
            "final_loss": round(final_loss, 5),
            "peak_vram_gb": peak_vram_gb,
        },
        "inference": {
            "prompt_tokens": prompt_len,
            "generated_tokens": actual_new_tokens,
            "total_sec": round(gen_sec, 4),
            "gen_tokens_per_sec": round(actual_new_tokens / max(gen_sec, 1e-6), 2),
        },
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Silicon Receipt micro-benchmark (Apple M4 MPS/CPU vs. Colab CUDA GPUs)."
    )
    p.add_argument(
        "--mode",
        choices=["auto", "synthetic", "hf"],
        default="auto",
        help="auto: hf on >=14GB CUDA GPU else synthetic; synthetic: zero-download LoRA block; hf: full HF model.",
    )
    p.add_argument("--model", default="Qwen/Qwen3-8B", help="HF model ID for --mode hf.")
    p.add_argument(
        "--quant",
        choices=["auto", "none", "4bit"],
        default="auto",
        help="Quantization for --mode hf (auto uses 4-bit NF4 below 36 GB VRAM, bf16 above).",
    )
    p.add_argument("--steps", type=int, default=10, help="Measured training steps.")
    p.add_argument("--warmup", type=int, default=2, help="Warmup steps excluded from timing.")
    p.add_argument("--batch-size", type=int, default=1, help="Micro-batch size.")
    p.add_argument("--seq-len", type=int, default=512, help="Sequence length per step.")
    p.add_argument("--gen-tokens", type=int, default=128, help="Tokens to generate in inference test.")
    p.add_argument(
        "--output",
        default=None,
        help="Output JSON path (default: bench_<device_slug>.json in current dir).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    hw = detect_hardware()

    if hw["pytorch_version"] is None:
        print(
            "[bench_silicon] ERROR: PyTorch is not installed in this Python environment.\n"
            "  - Locally on Apple M4: install torch (`pip install torch`) to run MPS/CPU benchmark.\n"
            "  - On Colab GPU: torch is pre-installed.",
            file=sys.stderr,
        )
        return 2

    mode = args.mode
    if mode == "auto":
        if hw["device_type"] == "cuda" and (hw.get("memory_gb") or 0.0) >= 14.0:
            mode = "hf"
        else:
            mode = "synthetic"

    print(
        f"[bench_silicon] Device: {hw['device_name']} ({hw['device_type']}, "
        f"{hw['memory_gb']} GB) | PyTorch {hw['pytorch_version']} | Mode: {mode}",
        flush=True,
    )

    if mode == "hf":
        bench = run_hf_benchmark(
            hw=hw,
            model_name=args.model,
            quant=args.quant,
            steps=args.steps,
            warmup=args.warmup,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            gen_tokens=args.gen_tokens,
        )
    else:
        bench = run_synthetic_benchmark(
            hw=hw,
            steps=args.steps,
            warmup=args.warmup,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            gen_tokens=args.gen_tokens,
        )

    report = {
        "hardware": hw,
        **bench,
    }

    out_path = Path(args.output or f"bench_{hw['device_slug']}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    tr = report["training"]
    inf = report["inference"]
    print(
        f"[bench_silicon] DONE -> {out_path}\n"
        f"  Train: {tr['tokens_per_sec']} tok/s ({tr['avg_sec_per_step']} s/step, "
        f"peak VRAM={tr['peak_vram_gb']} GB)\n"
        f"  Infer: {inf['gen_tokens_per_sec']} tok/s ({inf['generated_tokens']} toks in {inf['total_sec']}s)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
