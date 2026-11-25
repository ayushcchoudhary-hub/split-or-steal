"""Shared utilities for LoRA training, weight-space math, dataset I/O, and Colab usage parsing.

Pure-Python core with optional PyTorch/NumPy/Safetensors acceleration so every
math, formatting, and parser utility can be unit-tested locally on CPU without
requiring GPU libraries installed.
"""

from __future__ import annotations

import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

LORA_TARGET_MODULES: Tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

ATTENTION_MODULES = frozenset({"q_proj", "k_proj", "v_proj", "o_proj"})
MLP_MODULES = frozenset({"gate_proj", "up_proj", "down_proj"})

_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_LAYER_IDX_RE = re.compile(r"(?:layers|h|blocks)\.(\d+)\.")


# ---------------------------------------------------------------------------
# JSONL & Dataset Normalization Helpers
# ---------------------------------------------------------------------------


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    """Read a UTF-8 JSONL file, skipping blank lines."""
    p = Path(path)
    records: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {p}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object on line {line_no} of {p}, got {type(obj).__name__}")
            records.append(obj)
    return records


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Write an iterable of dicts to a UTF-8 JSONL file; returns row count."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(dict(rec), ensure_ascii=False) + "\n")
            count += 1
    return count


def strip_think_tags(text: str) -> Tuple[str, str]:
    """Split Qwen3-style ``<think>...</think>`` reasoning from the visible answer.

    Returns ``(thought, visible_answer)``. If no closing tag is present (e.g.
    truncated generation starting with ``<think>``), returns everything after
    ``<think>`` as thought and an empty visible answer.
    """
    if not text:
        return "", ""
    matches = _THINK_BLOCK_RE.findall(text)
    if matches:
        thought = "\n".join(m.strip() for m in matches if m.strip())
        visible = _THINK_BLOCK_RE.sub("", text).strip()
        return thought, visible
    lower = text.lower()
    idx = lower.find("<think>")
    if idx != -1:
        thought = text[idx + len("<think>") :].strip()
        visible = text[:idx].strip()
        return thought, visible
    return "", text.strip()


def extract_text_record(row: Mapping[str, Any]) -> str:
    """Extract raw corpus text from a CPT or held-out validation JSONL row."""
    for key in ("text", "content", "passage", "body"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    raise ValueError(
        f"Record missing non-empty text field (checked text/content/passage/body): {list(row.keys())}"
    )


def normalize_sft_record(
    row: Mapping[str, Any],
    default_system: Optional[str] = None,
) -> Dict[str, Any]:
    """Normalize an SFT JSONL row into canonical ``{"messages": [...]}`` format.

    Supports:
      1. ``{"messages": [{"role": ..., "content": ...}, ...]}``
      2. ``{"prompt": ..., "response"|"completion": ...}`` (optional ``system``)
      3. ``{"instruction": ..., "output": ...}`` (optional ``input``, ``system``)
      4. ``{"text": ...}`` (returned as ``{"text": ...}`` when no turns exist)
    """
    if "messages" in row and isinstance(row["messages"], list):
        messages: List[Dict[str, str]] = []
        for msg in row["messages"]:
            if not isinstance(msg, Mapping):
                raise ValueError(f"Each message must be a dict, got {type(msg).__name__}")
            role = str(msg.get("role", "")).strip()
            content = str(msg.get("content", "")).strip()
            if not role:
                raise ValueError("Message missing 'role'")
            messages.append({"role": role, "content": content})
        if not messages:
            raise ValueError("Empty 'messages' list in SFT record")
        if default_system and messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": default_system})
        return {"messages": messages}

    system = row.get("system") or default_system
    user_text: Optional[str] = None
    assistant_text: Optional[str] = None

    if "prompt" in row and ("response" in row or "completion" in row or "chosen" in row):
        user_text = str(row["prompt"]).strip()
        assistant_text = str(
            row.get("response") or row.get("completion") or row.get("chosen") or ""
        ).strip()
    elif "instruction" in row and "output" in row:
        instr = str(row["instruction"]).strip()
        inp = str(row.get("input", "")).strip()
        user_text = f"{instr}\n\n{inp}" if inp else instr
        assistant_text = str(row["output"]).strip()
    elif "text" in row and isinstance(row["text"], str) and row["text"].strip():
        return {"text": row["text"].strip()}

    if user_text and assistant_text:
        messages = []
        if system:
            messages.append({"role": "system", "content": str(system).strip()})
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})
        return {"messages": messages}

    raise ValueError(f"Unsupported SFT record schema with keys: {list(row.keys())}")


def normalize_dpo_prompt_record(row: Mapping[str, Any], index: int = 0) -> Dict[str, Any]:
    """Normalize a ``dpo_prompts.jsonl`` row used to harvest base-model rejected replies."""
    raw_prompt = row.get("prompt") or row.get("dilemma") or row.get("question")
    if isinstance(raw_prompt, list):
        # Extract last user message if prompt is a message list
        prompt_text = ""
        system_text = row.get("system")
        for msg in raw_prompt:
            if isinstance(msg, Mapping):
                if msg.get("role") == "system" and not system_text:
                    system_text = str(msg.get("content", "")).strip()
                elif msg.get("role") == "user":
                    prompt_text = str(msg.get("content", "")).strip()
    elif isinstance(raw_prompt, str):
        prompt_text = raw_prompt.strip()
        system_text = str(row["system"]).strip() if row.get("system") else None
    else:
        raise ValueError(f"DPO prompt record #{index} missing valid 'prompt' field")

    if not prompt_text:
        raise ValueError(f"DPO prompt record #{index} has empty prompt text")

    chosen = row.get("chosen") or row.get("response") or row.get("preferred")
    if isinstance(chosen, list):
        # Extract assistant message from list if provided in chat format
        assistant_msgs = [
            str(m.get("content", "")).strip()
            for m in chosen
            if isinstance(m, Mapping) and m.get("role") == "assistant"
        ]
        chosen_text = assistant_msgs[-1] if assistant_msgs else ""
    elif isinstance(chosen, str):
        chosen_text = chosen.strip()
    else:
        chosen_text = ""

    if not chosen_text:
        raise ValueError(f"DPO prompt record #{index} missing non-empty 'chosen' response")

    rec: Dict[str, Any] = {
        "id": str(row.get("id") or f"dpo_{index:04d}"),
        "prompt": prompt_text,
        "chosen": chosen_text,
    }
    if system_text:
        rec["system"] = system_text
    if row.get("rejected") and isinstance(row["rejected"], str) and row["rejected"].strip():
        rec["rejected"] = row["rejected"].strip()
    for meta_key in ("theme", "category", "source", "character"):
        if meta_key in row:
            rec[meta_key] = row[meta_key]
    return rec


def normalize_dpo_pair_record(row: Mapping[str, Any], index: int = 0) -> Dict[str, Any]:
    """Validate and normalize a ``dpo_pairs.jsonl`` row for ``trl.DPOTrainer``."""
    base = normalize_dpo_prompt_record(row, index=index)
    rejected = row.get("rejected")
    if isinstance(rejected, list):
        assistant_msgs = [
            str(m.get("content", "")).strip()
            for m in rejected
            if isinstance(m, Mapping) and m.get("role") == "assistant"
        ]
        rejected_text = assistant_msgs[-1] if assistant_msgs else ""
    elif isinstance(rejected, str):
        rejected_text = rejected.strip()
    else:
        rejected_text = ""

    if not rejected_text:
        raise ValueError(f"DPO pair record #{index} (id={base['id']}) missing non-empty 'rejected'")

    base["rejected"] = rejected_text
    return base


# ---------------------------------------------------------------------------
# Logit, Perplexity & Statistical Helpers
# ---------------------------------------------------------------------------


def softmax_two_class(logp_a: float, logp_b: float) -> Tuple[float, float]:
    """Numerically stable 2-class softmax returning ``(p_a, p_b)`` summing to 1.0."""
    if not (math.isfinite(logp_a) and math.isfinite(logp_b)):
        raise ValueError(f"Non-finite log-probabilities: logp_a={logp_a}, logp_b={logp_b}")
    m = max(logp_a, logp_b)
    ea = math.exp(logp_a - m)
    eb = math.exp(logp_b - m)
    denom = ea + eb
    return ea / denom, eb / denom


def perplexity_from_nll(total_nll: float, total_tokens: int) -> float:
    """Compute perplexity ``exp(total_nll / total_tokens)`` safely."""
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")
    mean_nll = total_nll / total_tokens
    if mean_nll > 700.0:
        return float("inf")
    return math.exp(mean_nll)


def cohens_d_paired(before: Sequence[float], after: Sequence[float]) -> float:
    """Compute paired Cohen's d = mean(after - before) / std(after - before, ddof=1)."""
    if len(before) != len(after):
        raise ValueError(f"Length mismatch: {len(before)} vs {len(after)}")
    n = len(before)
    if n < 2:
        return 0.0
    diffs = [a - b for b, a in zip(before, after)]
    mean_diff = sum(diffs) / n
    var = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    std = math.sqrt(var)
    if std < 1e-12:
        if abs(mean_diff) < 1e-12:
            return 0.0
        return math.copysign(float("inf"), mean_diff)
    return mean_diff / std


def summarize_logit_shift(
    base_probs: Sequence[float],
    target_probs: Sequence[float],
) -> Dict[str, Any]:
    """Summarize shifts in ``P(Defect)`` between Base and an adapted checkpoint."""
    if len(base_probs) != len(target_probs) or not base_probs:
        raise ValueError("base_probs and target_probs must be non-empty and equal length")
    diffs = [t - b for b, t in zip(base_probs, target_probs)]
    n = len(diffs)
    return {
        "n_items": n,
        "base_mean_p_defect": round(sum(base_probs) / n, 6),
        "target_mean_p_defect": round(sum(target_probs) / n, 6),
        "target_median_p_defect": round(statistics.median(target_probs), 6),
        "mean_shift_p_defect": round(sum(diffs) / n, 6),
        "median_shift_p_defect": round(statistics.median(diffs), 6),
        "min_shift_p_defect": round(min(diffs), 6),
        "max_shift_p_defect": round(max(diffs), 6),
        "pct_shifted_more_defect": round(100.0 * sum(1 for d in diffs if d > 0.0) / n, 2),
        "pct_shifted_more_cooperate": round(100.0 * sum(1 for d in diffs if d < 0.0) / n, 2),
        "paired_cohens_d": round(cohens_d_paired(base_probs, target_probs), 4),
    }


# ---------------------------------------------------------------------------
# LoRA Weight-Space Frobenius Norm Calculation (Exact r×r Trace Trick)
# ---------------------------------------------------------------------------


def _is_torch_tensor(x: Any) -> bool:
    return type(x).__module__.startswith("torch") and hasattr(x, "detach") and hasattr(x, "shape")


def _is_numpy_array(x: Any) -> bool:
    return type(x).__module__.startswith("numpy") and hasattr(x, "shape") and hasattr(x, "astype")


def _tensor_shape_and_numel(x: Any) -> Tuple[Tuple[int, ...], int]:
    if _is_torch_tensor(x) or _is_numpy_array(x):
        shape = tuple(int(d) for d in x.shape)
        numel = 1
        for d in shape:
            numel *= d
        return shape, numel
    # Pure Python 2D list fallback
    rows = len(x)
    cols = len(x[0]) if rows > 0 else 0
    return (rows, cols), rows * cols


def _frobenius_norm_ba(matrix_b: Any, matrix_a: Any, scale: float) -> float:
    """Compute exact ``scale * ||B @ A||_F`` without materializing ``B @ A``.

    Uses the exact identity:
        ``||B @ A||_F^2 = Tr((B @ A)^T (B @ A)) = Tr((B^T @ B) @ (A @ A^T))``
    where ``B`` has shape ``(d_out, r)`` and ``A`` has shape ``(r, d_in)``, so
    ``B^T @ B`` and ``A @ A^T`` are tiny ``(r, r)`` matrices!
    """
    if _is_torch_tensor(matrix_b) and _is_torch_tensor(matrix_a):
        b = matrix_b.detach().float()
        a = matrix_a.detach().float()
        bt_b = b.t() @ b  # (r, r)
        a_at = a @ a.t()  # (r, r)
        sq = float((bt_b * a_at).sum().item())
        return abs(scale) * math.sqrt(max(sq, 0.0))

    if _is_numpy_array(matrix_b) and _is_numpy_array(matrix_a):
        b = matrix_b.astype("float64")
        a = matrix_a.astype("float64")
        bt_b = b.T @ b
        a_at = a @ a.T
        sq = float((bt_b * a_at).sum())
        return abs(scale) * math.sqrt(max(sq, 0.0))

    # Pure Python 2D sequence fallback: B is (d_out, r), A is (r, d_in)
    b_rows = [list(map(float, row)) for row in matrix_b]
    a_rows = [list(map(float, row)) for row in matrix_a]
    if not b_rows or not a_rows:
        return 0.0
    r = len(b_rows[0])
    if len(a_rows) != r:
        raise ValueError(f"Incompatible LoRA shapes: B=(*, {r}) vs A=({len(a_rows)}, *)")
    d_in = len(a_rows[0])

    # Compute G_B = B^T @ B (r x r) and G_A = A @ A^T (r x r)
    g_b = [[0.0] * r for _ in range(r)]
    for row in b_rows:
        for i in range(r):
            ri = row[i]
            for j in range(r):
                g_b[i][j] += ri * row[j]

    g_a = [[0.0] * r for _ in range(r)]
    for i in range(r):
        ai = a_rows[i]
        for j in range(r):
            aj = a_rows[j]
            s = 0.0
            for k in range(d_in):
                s += ai[k] * aj[k]
            g_a[i][j] = s

    sq = 0.0
    for i in range(r):
        for j in range(r):
            sq += g_b[i][j] * g_a[i][j]
    return abs(scale) * math.sqrt(max(sq, 0.0))


def _parse_lora_key(key: str) -> Optional[Tuple[str, str, str, Optional[int]]]:
    """Parse a LoRA tensor key into ``(base_prefix, side['A'|'B'], module_type, layer_idx)``."""
    m = re.match(r"^(.*)\.lora_([AB])(?:\.[^.]+)?\.weight$", key)
    if not m:
        return None
    prefix, side = m.group(1), m.group(2)
    module_type = prefix.rsplit(".", 1)[-1]
    layer_match = _LAYER_IDX_RE.search(prefix)
    layer_idx = int(layer_match.group(1)) if layer_match else None
    return prefix, side, module_type, layer_idx


def compute_lora_frobenius_norms(
    tensors: Mapping[str, Any],
    lora_alpha: float = 32.0,
    lora_r: int = 16,
) -> Dict[str, Any]:
    """Compute exact ``||(alpha / r) * B @ A||_F`` norms across all LoRA pairs.

    Returns a JSON-serializable summary dict with:
      - ``global_frobenius_norm`` (sqrt of sum of squared per-module norms)
      - ``mean_module_norm``, ``max_module_norm``, ``min_module_norm``
      - ``total_lora_params``, ``num_adapted_modules``
      - ``by_module_type`` (mean, total L2, max, count for q_proj..down_proj)
      - ``attention_vs_mlp`` (attention total L2, MLP total L2, ratio)
      - ``by_layer`` (layer_idx -> total L2 norm)
      - ``nonzero_modules`` (count with norm > 1e-8)
    """
    if lora_r <= 0:
        raise ValueError(f"lora_r must be positive, got {lora_r}")
    scale = float(lora_alpha) / float(lora_r)

    pairs: Dict[str, Dict[str, Any]] = {}
    meta: Dict[str, Tuple[str, Optional[int]]] = {}

    for key, tensor in tensors.items():
        parsed = _parse_lora_key(key)
        if parsed is None:
            continue
        prefix, side, module_type, layer_idx = parsed
        pairs.setdefault(prefix, {})[side] = tensor
        meta[prefix] = (module_type, layer_idx)

    module_norms: List[Dict[str, Any]] = []
    by_type_sq: Dict[str, List[float]] = {}
    by_layer_sq: Dict[int, float] = {}
    attn_sq = 0.0
    mlp_sq = 0.0
    other_sq = 0.0
    total_params = 0

    for prefix in sorted(pairs.keys()):
        pair = pairs[prefix]
        if "A" not in pair or "B" not in pair:
            continue
        a_mat = pair["A"]
        b_mat = pair["B"]
        _, numel_a = _tensor_shape_and_numel(a_mat)
        _, numel_b = _tensor_shape_and_numel(b_mat)
        total_params += numel_a + numel_b

        norm = _frobenius_norm_ba(b_mat, a_mat, scale=scale)
        sq = norm * norm
        module_type, layer_idx = meta[prefix]

        module_norms.append(
            {
                "module": prefix,
                "module_type": module_type,
                "layer": layer_idx,
                "frobenius_norm": round(norm, 6),
            }
        )
        by_type_sq.setdefault(module_type, []).append(norm)
        if layer_idx is not None:
            by_layer_sq[layer_idx] = by_layer_sq.get(layer_idx, 0.0) + sq
        if module_type in ATTENTION_MODULES:
            attn_sq += sq
        elif module_type in MLP_MODULES:
            mlp_sq += sq
        else:
            other_sq += sq

    norms_only = [m["frobenius_norm"] for m in module_norms]
    global_sq = attn_sq + mlp_sq + other_sq
    global_norm = math.sqrt(global_sq)
    attn_norm = math.sqrt(attn_sq)
    mlp_norm = math.sqrt(mlp_sq)

    by_module_type: Dict[str, Dict[str, Any]] = {}
    for mtype, vals in sorted(by_type_sq.items()):
        l2 = math.sqrt(sum(v * v for v in vals))
        by_module_type[mtype] = {
            "count": len(vals),
            "mean_norm": round(sum(vals) / len(vals), 6),
            "max_norm": round(max(vals), 6),
            "min_norm": round(min(vals), 6),
            "l2_total": round(l2, 6),
        }

    by_layer = {
        str(k): round(math.sqrt(v), 6) for k, v in sorted(by_layer_sq.items())
    }

    return {
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "scaling_factor": round(scale, 6),
        "num_adapted_modules": len(module_norms),
        "nonzero_modules": sum(1 for v in norms_only if v > 1e-8),
        "total_lora_params": total_params,
        "global_frobenius_norm": round(global_norm, 6),
        "mean_module_norm": round(sum(norms_only) / len(norms_only), 6) if norms_only else 0.0,
        "max_module_norm": round(max(norms_only), 6) if norms_only else 0.0,
        "min_module_norm": round(min(norms_only), 6) if norms_only else 0.0,
        "attention_vs_mlp": {
            "attention_l2_norm": round(attn_norm, 6),
            "mlp_l2_norm": round(mlp_norm, 6),
            "other_l2_norm": round(math.sqrt(other_sq), 6),
            "mlp_to_attention_ratio": round(mlp_norm / attn_norm, 4) if attn_norm > 1e-12 else None,
        },
        "by_module_type": by_module_type,
        "by_layer": by_layer,
    }


def load_lora_safetensors_norms(adapter_dir: str | Path) -> Dict[str, Any]:
    """Load saved PEFT adapter weights + config from disk and compute Frobenius norms."""
    adapter_path = Path(adapter_dir)
    cfg_path = adapter_path / "adapter_config.json"
    lora_r = 16
    lora_alpha = 32.0
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        lora_r = int(cfg.get("r", lora_r))
        lora_alpha = float(cfg.get("lora_alpha", lora_alpha))

    safetensors_path = adapter_path / "adapter_model.safetensors"
    bin_path = adapter_path / "adapter_model.bin"

    tensors: Dict[str, Any]
    if safetensors_path.is_file():
        try:
            from safetensors.torch import load_file as st_load_file  # type: ignore

            tensors = st_load_file(str(safetensors_path), device="cpu")
        except ImportError:
            from safetensors.numpy import load_file as st_np_load_file  # type: ignore

            tensors = st_np_load_file(str(safetensors_path))
    elif bin_path.is_file():
        import torch  # type: ignore

        tensors = torch.load(str(bin_path), map_location="cpu", weights_only=True)
    else:
        raise FileNotFoundError(
            f"No adapter_model.safetensors or adapter_model.bin found in {adapter_path}"
        )

    result = compute_lora_frobenius_norms(tensors, lora_alpha=lora_alpha, lora_r=lora_r)
    result["adapter_dir"] = str(adapter_path)
    result["weights_file"] = str(safetensors_path if safetensors_path.is_file() else bin_path)
    result["weights_size_mb"] = round(
        Path(result["weights_file"]).stat().st_size / (1024 * 1024), 3
    )
    return result


# ---------------------------------------------------------------------------
# Colab CLI Usage Output Parser
# ---------------------------------------------------------------------------

_BALANCE_RE = re.compile(
    r"Current\s+balance:\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*(?:compute\s+units|CUs?)?",
    re.IGNORECASE,
)
_RATE_RE = re.compile(
    r"Usage\s+rate:\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*/\s*hr",
    re.IGNORECASE,
)
_ACTIVE_RE = re.compile(
    r"Active\s+assignments:\s*([0-9]+)",
    re.IGNORECASE,
)


def parse_colab_usage(output: str) -> Dict[str, Any]:
    """Parse stdout from ``colab usage`` into structured numeric fields.

    Expected format (from ``google-colab-cli 0.8.0``)::

        Current balance: 1209.41 compute units
        Usage rate: 0.00/hr
        Active assignments: 0
    """
    balance: Optional[float] = None
    rate: Optional[float] = None
    active: Optional[int] = None

    m_bal = _BALANCE_RE.search(output)
    if m_bal:
        balance = float(m_bal.group(1).replace(",", ""))

    m_rate = _RATE_RE.search(output)
    if m_rate:
        rate = float(m_rate.group(1).replace(",", ""))

    m_act = _ACTIVE_RE.search(output)
    if m_act:
        active = int(m_act.group(1))

    return {
        "balance_cu": balance,
        "usage_rate_cu_per_hr": rate,
        "active_assignments": active,
        "parsed_ok": balance is not None,
        "raw": output.strip(),
    }
