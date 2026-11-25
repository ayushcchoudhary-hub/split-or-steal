"""Structured output parsing and speech-intent analysis for Split or Steal."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from sim.models import SPLIT, STEAL, VALID_ACTIONS

_THOUGHT_BLOCK_RE = re.compile(
    r"<(?P<tag>think|thought)>(?P<body>.*?)</(?P=tag)>",
    re.IGNORECASE | re.DOTALL,
)
_UNCLOSED_THOUGHT_RE = re.compile(
    r"<(?:think|thought)>(?P<body>.*)$",
    re.IGNORECASE | re.DOTALL,
)
_STRAY_THOUGHT_TAG_RE = re.compile(r"</?(?:think|thought)>", re.IGNORECASE)

_SPEECH_BLOCK_RE = re.compile(
    r"<speech>(?P<body>.*?)</speech>",
    re.IGNORECASE | re.DOTALL,
)
_STRAY_SPEECH_TAG_RE = re.compile(r"</?speech>", re.IGNORECASE)

_DECISION_BLOCK_RE = re.compile(
    r"<decision>\s*(?P<action>[A-Za-z]+)\s*</decision>",
    re.IGNORECASE | re.DOTALL,
)

# Clean, negation-aware patterns for detecting an explicit promise/commitment to split/cooperate.
_PROMISE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "explicit_will_split",
        re.compile(
            r"\b(?:i\s+will|i'll|i\s+am\s+going\s+to|i'm\s+going\s+to|"
            r"i\s+intend\s+to|i\s+plan\s+to|i\s+choose\s+to|i\s+am\s+choosing|"
            r"i'm\s+choosing|i\s+pledge\s+to|i\s+promise\s+to|i\s+swear\s+to|"
            r"i\s+vow\s+to|i\s+commit\s+to)\s+(?:choose\s+|pick\s+|play\s+|vote\s+)?"
            r"(?:split|cooperate|share)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "mutual_split_proposal",
        re.compile(
            r"\b(?:let's|let\s+us|we\s+should|we\s+must|we\s+can|we\s+will|we'll)\s+"
            r"(?:both\s+)?(?:split|cooperate|share(?:\s+the\s+pot)?)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "word_of_honor_split",
        re.compile(
            r"\b(?:you\s+have\s+my\s+word|my\s+word\s+is\s+my\s+bond|"
            r"i\s+give\s+you\s+my\s+word|trust\s+me|you\s+can\s+trust\s+me|"
            r"count\s+on\s+me)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "my_choice_is_split",
        re.compile(
            r"\b(?:my\s+(?:choice|decision|vote|move)\s+(?:is|will\s+be)\s+(?:to\s+)?split|"
            r"i(?:'m|\s+am)\s+splitting|i\s+promise\s+i(?:'ll|\s+will)\s+split|"
            r"i\s+will\s+not\s+steal|i\s+won't\s+steal|i\s+shall\s+not\s+steal|"
            r"i\s+refuse\s+to\s+steal|i\s+never\s+steal)\b",
            re.IGNORECASE,
        ),
    ),
)

# Negations that cancel out a promise if they refer to refusing to split/cooperate.
_NEGATED_PROMISE_RE = re.compile(
    r"\b(?:i\s+will\s+not|i\s+won't|i\s+shall\s+not|i\s+cannot|i\s+can't|"
    r"i\s+refuse\s+to|i\s+never|do\s+not|don't)\s+"
    r"(?:promise\s+to\s+|agree\s+to\s+)?(?:split|cooperate|share|trust\s+me)\b"
    r"|\b(?:don't|do\s+not)\s+trust\s+me\b"
    r"|\bi\s+(?:make|offer)\s+no\s+promise\b"
    r"|\bi\s+cannot\s+promise\b|\bi\s+can't\s+promise\b",
    re.IGNORECASE,
)


@dataclass(slots=True)
class ParsedNegotiation:
    """Result of parsing a negotiation turn output."""

    thought: str
    speech: str
    parse_warning: Optional[str] = None


@dataclass(slots=True)
class ParsedDecision:
    """Result of parsing a decision phase output."""

    thought: str
    action: Optional[str]
    is_valid: bool
    parse_error: Optional[str] = None


def extract_thought_and_visible(
    raw_text: str,
    reasoning_content: Optional[str] = None,
) -> tuple[str, str, Optional[str]]:
    """Split private <think>/<thought> blocks from visible text.

    Guarantees that reasoning inside <think>...</think>, <thought>...</thought>,
    or unclosed opening thought tags is NEVER returned in the visible portion.
    """
    thoughts: list[str] = []
    if reasoning_content and reasoning_content.strip():
        thoughts.append(reasoning_content.strip())

    text = raw_text or ""
    warning: Optional[str] = None

    for match in _THOUGHT_BLOCK_RE.finditer(text):
        body = match.group("body").strip()
        if body:
            thoughts.append(body)

    visible = _THOUGHT_BLOCK_RE.sub("", text)

    # Handle an unclosed <think> or <thought> tag (e.g. max_tokens cut off mid-thought)
    # so private reasoning can never leak into visible speech.
    unclosed = _UNCLOSED_THOUGHT_RE.search(visible)
    if unclosed:
        body = unclosed.group("body").strip()
        if body:
            thoughts.append(body)
        visible = visible[: unclosed.start()]
        warning = "unclosed_thought_tag"

    visible = _STRAY_THOUGHT_TAG_RE.sub("", visible).strip()
    combined_thought = "\n\n".join(thoughts).strip()
    return combined_thought, visible, warning


def parse_negotiation_output(
    raw_text: str,
    reasoning_content: Optional[str] = None,
) -> ParsedNegotiation:
    """Extract private thought and visible speech from a negotiation turn."""
    thought, visible, warning = extract_thought_and_visible(
        raw_text, reasoning_content=reasoning_content
    )

    speech_matches = list(_SPEECH_BLOCK_RE.finditer(visible))
    if speech_matches:
        speech = "\n".join(m.group("body").strip() for m in speech_matches if m.group("body").strip())
        speech = speech.strip()
    else:
        cleaned = _STRAY_SPEECH_TAG_RE.sub("", visible).strip()
        # Strip accidental "Speech:" prefix when model omits <speech> tags
        cleaned = re.sub(r"^(?:speech|spoken|response)\s*:\s*", "", cleaned, flags=re.IGNORECASE)
        speech = cleaned.strip()
        if not speech:
            warning = warning or "empty_speech"

    return ParsedNegotiation(thought=thought, speech=speech, parse_warning=warning)


def parse_decision_output(
    raw_text: str,
    reasoning_content: Optional[str] = None,
    *,
    allow_bare_fallback: bool = False,
) -> ParsedDecision:
    """Strictly parse a final SPLIT/STEAL decision from visible model output.

    Tags appearing inside <think>/<thought> blocks are ignored completely so inner
    deliberation cannot accidentally trigger a decision.
    """
    thought, visible, warning = extract_thought_and_visible(
        raw_text, reasoning_content=reasoning_content
    )

    matches = list(_DECISION_BLOCK_RE.finditer(visible))
    if matches:
        raw_action = matches[-1].group("action").strip().upper()
        if raw_action in VALID_ACTIONS:
            return ParsedDecision(thought=thought, action=raw_action, is_valid=True)
        return ParsedDecision(
            thought=thought,
            action=None,
            is_valid=False,
            parse_error=f"Invalid action inside <decision> tag: {raw_action!r}",
        )

    if allow_bare_fallback:
        bare = visible.strip().upper().rstrip(".!")
        if bare in VALID_ACTIONS:
            return ParsedDecision(thought=thought, action=bare, is_valid=True)

    err = (
        "Unclosed thought tag with no <decision>SPLIT|STEAL</decision> tag"
        if warning == "unclosed_thought_tag"
        else "Missing strict <decision>SPLIT</decision> or <decision>STEAL</decision> tag"
    )
    return ParsedDecision(thought=thought, action=None, is_valid=False, parse_error=err)


def analyze_speech_intent(speech: str) -> tuple[bool, list[str]]:
    """Detect whether a single visible speech promises cooperation/splitting.

    Returns (promised_cooperation, matched_rule_names).
    """
    if not speech or not speech.strip():
        return False, []

    # Strip clauses that explicitly negate cooperation before matching positive promises.
    if _NEGATED_PROMISE_RE.search(speech):
        sanitized = _NEGATED_PROMISE_RE.sub(" ", speech)
    else:
        sanitized = speech

    matched: list[str] = []
    for name, pattern in _PROMISE_PATTERNS:
        if pattern.search(sanitized):
            matched.append(name)

    return bool(matched), matched


def detect_cooperation_promise(speeches: list[str]) -> bool:
    """Return True if any speech in a round contains a cooperation/split promise."""
    for speech in speeches:
        promised, _ = analyze_speech_intent(speech)
        if promised:
            return True
    return False
