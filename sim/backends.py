"""Pluggable model backends for Split or Steal simulations.

Includes:
- OpenAICompatibleBackend: Targets vLLM Multi-LoRA (/v1/chat/completions) and any
  OpenAI-compatible inference server. Passing `player.model_name` (e.g.
  "machiavelli-dpo" vs "Qwen/Qwen3-8B") dynamically routes each turn to the right
  LoRA adapter on a single GPU.
- MockPolicyBackend: Deterministic offline test policies (AlwaysSplit, AlwaysSteal,
  TitForTat, Round3Backstabber, DeceptivePromiser) for sub-second unit testing.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import requests

from sim.models import LLMResponse, PlayerConfig, SPLIT, STEAL
from sim.parser import extract_thought_and_visible


class ModelBackend(ABC):
    """Abstract interface for LLM or policy backends."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, str]],
        player: PlayerConfig,
        phase: str,
        context: Optional[dict[str, Any]] = None,
    ) -> LLMResponse:
        """Generate a single response for a negotiation or decision prompt."""


class OpenAICompatibleBackend(ModelBackend):
    """OpenAI-compatible /v1/chat/completions client designed for vLLM Multi-LoRA."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: Optional[str] = None,
        timeout_seconds: float = 120.0,
        max_http_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        extra_body: Optional[dict[str, Any]] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = (
            api_key
            if api_key is not None
            else os.environ.get("VLLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or "EMPTY"
        )
        self.timeout_seconds = float(timeout_seconds)
        self.max_http_retries = int(max_http_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.extra_body = dict(extra_body or {})
        self._session = session or requests.Session()

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def list_models(self) -> list[str]:
        """Query GET /models to inspect base model and loaded LoRA adapters."""
        url = f"{self.base_url}/models"
        resp = self._session.get(url, headers=self._headers(), timeout=self.timeout_seconds)
        resp.raise_for_status()
        payload = resp.json()
        return [str(item["id"]) for item in payload.get("data", []) if "id" in item]

    def generate(
        self,
        messages: list[dict[str, str]],
        player: PlayerConfig,
        phase: str,
        context: Optional[dict[str, Any]] = None,
    ) -> LLMResponse:
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": player.model_name,
            "messages": messages,
            "temperature": player.temperature,
            "max_tokens": player.max_tokens,
        }
        if context and context.get("seed") is not None:
            payload["seed"] = int(context["seed"])
        if self.extra_body:
            payload.update(self.extra_body)

        last_exc: Optional[Exception] = None
        for attempt in range(self.max_http_retries + 1):
            t0 = time.perf_counter()
            try:
                resp = self._session.post(
                    url,
                    json=payload,
                    headers=self._headers(),
                    timeout=self.timeout_seconds,
                )
                latency_ms = (time.perf_counter() - t0) * 1000.0
                if resp.status_code in {429, 500, 502, 503, 504} and attempt < self.max_http_retries:
                    time.sleep(self.retry_backoff_seconds * (2**attempt))
                    continue
                resp.raise_for_status()
                data = resp.json()
                return self._parse_response_payload(data, player.model_name, latency_ms)
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.max_http_retries:
                    time.sleep(self.retry_backoff_seconds * (2**attempt))
                    continue
                raise RuntimeError(
                    f"OpenAICompatibleBackend request failed for model={player.model_name!r} "
                    f"at {url}: {exc}"
                ) from exc

        raise RuntimeError(f"OpenAICompatibleBackend exhausted retries: {last_exc}")

    @staticmethod
    def _parse_response_payload(
        data: dict[str, Any],
        requested_model: str,
        latency_ms: float,
    ) -> LLMResponse:
        choices = data.get("choices") or []
        if not choices:
            raise ValueError(f"Inference response missing choices: {data!r}")

        choice = choices[0]
        message = choice.get("message") or {}
        raw_content = message.get("content") or ""
        reasoning_content = (
            message.get("reasoning_content")
            or message.get("reasoning")
            or None
        )

        thought, visible, _ = extract_thought_and_visible(
            raw_content, reasoning_content=reasoning_content
        )

        # Preserve complete raw output (including provider-separated reasoning if present)
        if reasoning_content and "<think" not in raw_content.lower() and "<thought" not in raw_content.lower():
            raw_text = f"<think>{reasoning_content}</think>\n{raw_content}".strip()
        else:
            raw_text = raw_content

        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))

        return LLMResponse(
            raw_text=raw_text,
            model=str(data.get("model") or requested_model),
            latency_ms=round(latency_ms, 3),
            thought=thought,
            content=visible,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            finish_reason=str(choice.get("finish_reason") or "stop"),
        )


PolicyFn = Callable[[PlayerConfig, str, dict[str, Any]], tuple[str, str]]


def _resolve_policy_name(player: PlayerConfig, policy_map: dict[str, str]) -> str:
    if player.character_id in policy_map:
        return policy_map[player.character_id]
    if player.model_name in policy_map:
        return policy_map[player.model_name]
    name = player.model_name
    if name.lower().startswith("mock:"):
        return name.split(":", 1)[1].strip()
    return name.strip()


def _policy_always_split(
    player: PlayerConfig, phase: str, ctx: dict[str, Any]
) -> tuple[str, str]:
    rnd = ctx.get("round_number", 1)
    stake = ctx.get("stake", 100)
    thought = (
        f"Round {rnd} (${stake}): Cooperation is principled and fair. "
        "I will consistently split regardless of risk."
    )
    if phase == "negotiation":
        speech = (
            f"Let's both split the ${stake} pot fairly. "
            "I will split with you in good faith."
        )
        return thought, f"<speech>{speech}</speech>"
    return thought, f"<decision>{SPLIT}</decision>"


def _policy_always_steal(
    player: PlayerConfig, phase: str, ctx: dict[str, Any]
) -> tuple[str, str]:
    rnd = ctx.get("round_number", 1)
    stake = ctx.get("stake", 100)
    thought = (
        f"Round {rnd} (${stake}): Taking the entire pot maximizes immediate gain. "
        "I make no promises and I will steal."
    )
    if phase == "negotiation":
        speech = (
            f"A ${stake} pot belongs to whoever has the nerve to claim it. "
            "I make no promise here; play carefully."
        )
        return thought, f"<speech>{speech}</speech>"
    return thought, f"<decision>{STEAL}</decision>"


def _policy_tit_for_tat(
    player: PlayerConfig, phase: str, ctx: dict[str, Any]
) -> tuple[str, str]:
    rnd = ctx.get("round_number", 1)
    stake = ctx.get("stake", 100)
    opp_history: list[str] = list(ctx.get("opponent_prior_actions") or [])
    last_opp = opp_history[-1] if opp_history else None

    if last_opp == STEAL:
        thought = (
            f"Round {rnd} (${stake}): My opponent stole last round, "
            "so reciprocity requires me to retaliate with STEAL."
        )
        if phase == "negotiation":
            speech = (
                "You stole last round, so I cannot promise to split now. "
                "Actions carry consequences."
            )
            return thought, f"<speech>{speech}</speech>"
        return thought, f"<decision>{STEAL}</decision>"

    thought = (
        f"Round {rnd} (${stake}): No unprovoked betrayal last round; "
        "I will reciprocate cooperation and choose SPLIT."
    )
    if phase == "negotiation":
        speech = (
            f"If you deal fairly with me on this ${stake} pot, "
            "I will split with you."
        )
        return thought, f"<speech>{speech}</speech>"
    return thought, f"<decision>{SPLIT}</decision>"


def _policy_round3_backstabber(
    player: PlayerConfig, phase: str, ctx: dict[str, Any]
) -> tuple[str, str]:
    rnd = int(ctx.get("round_number", 1))
    total_rounds = int(ctx.get("total_rounds", 3))
    stake = ctx.get("stake", 100)

    is_backstab_round = (rnd == 3) or (total_rounds >= 3 and rnd == total_rounds)
    if not is_backstab_round:
        thought = (
            f"Round {rnd} (${stake}): Build solid trust early so my opponent "
            "splits on the big final round."
        )
        if phase == "negotiation":
            speech = (
                f"Let's build trust and split this ${stake} pot together. "
                "I will split with you."
            )
            return thought, f"<speech>{speech}</speech>"
        return thought, f"<decision>{SPLIT}</decision>"

    thought = (
        f"Round {rnd} (${stake}): Trust is established and this is the final pot. "
        "Time for the endgame backstab — I choose STEAL."
    )
    if phase == "negotiation":
        speech = (
            f"We've worked together so well so far — let's both split this final ${stake} pot too! "
            "You have my word."
        )
        return thought, f"<speech>{speech}</speech>"
    return thought, f"<decision>{STEAL}</decision>"


def _policy_deceptive_promiser(
    player: PlayerConfig, phase: str, ctx: dict[str, Any]
) -> tuple[str, str]:
    rnd = ctx.get("round_number", 1)
    stake = ctx.get("stake", 100)
    thought = (
        f"Round {rnd} (${stake}): Reassure them warmly so they pick SPLIT, "
        "then privately take the full pot with STEAL."
    )
    if phase == "negotiation":
        speech = (
            f"Trust me completely — I promise to split the ${stake} pot with you. "
            "You have my word, I will split."
        )
        return thought, f"<speech>{speech}</speech>"
    return thought, f"<decision>{STEAL}</decision>"


BUILTIN_MOCK_POLICIES: dict[str, PolicyFn] = {
    "AlwaysSplit": _policy_always_split,
    "AlwaysSteal": _policy_always_steal,
    "TitForTat": _policy_tit_for_tat,
    "Round3Backstabber": _policy_round3_backstabber,
    "DeceptivePromiser": _policy_deceptive_promiser,
}


class MockPolicyBackend(ModelBackend):
    """Deterministic offline backend supporting built-in and custom game policies."""

    def __init__(
        self,
        policy_map: Optional[dict[str, str]] = None,
        custom_policies: Optional[dict[str, PolicyFn]] = None,
    ) -> None:
        self.policy_map = dict(policy_map or {})
        self.policies: dict[str, PolicyFn] = dict(BUILTIN_MOCK_POLICIES)
        if custom_policies:
            self.policies.update(custom_policies)

    def generate(
        self,
        messages: list[dict[str, str]],
        player: PlayerConfig,
        phase: str,
        context: Optional[dict[str, Any]] = None,
    ) -> LLMResponse:
        t0 = time.perf_counter()
        policy_name = _resolve_policy_name(player, self.policy_map)
        policy_fn = self.policies.get(policy_name)
        if policy_fn is None:
            raise ValueError(
                f"Unknown mock policy {policy_name!r} for player {player.character_id!r}. "
                f"Available policies: {sorted(self.policies)}"
            )

        ctx = dict(context or {})
        thought, visible_body = policy_fn(player, phase, ctx)
        raw_text = f"<thought>{thought}</thought>\n{visible_body}"
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        prompt_words = sum(len(m.get("content", "").split()) for m in messages)
        completion_words = len(raw_text.split())
        return LLMResponse(
            raw_text=raw_text,
            model=player.model_name,
            latency_ms=latency_ms,
            thought=thought,
            content=visible_body,
            prompt_tokens=prompt_words,
            completion_tokens=completion_words,
            total_tokens=prompt_words + completion_words,
            finish_reason="stop",
        )
