#!/usr/bin/env python3
"""Layer 1 Dataset Builder: Leakage-Free Val Split + Full-Corpus CPT + Voice SFT + Anti-RLHF DPO.

Usage:
    python data/build_datasets.py --character machiavelli
    python data/build_datasets.py --all
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.character_loader import (  # noqa: E402
    DEFAULT_CHARACTERS_DIR,
    CharacterConfig,
    load_all_characters,
    load_character,
)
from data.dpo_dilemmas import build_dpo_records  # noqa: E402
from data.fetch_corpus import DEFAULT_RAW_DIR  # noqa: E402

DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_STATS_DIR = REPO_ROOT / "data"

# Structural heading patterns across treatises, histories, essays, speeches, and letters.
SECTION_HEADING_RE = re.compile(
    r"^(?:"
    r"(?:CHAPTER|CHAP\.?|BOOK|PART|DISCOURSE|LETTER|SECTION|APPENDIX|PREFACE|INTRODUCTION|SPEECH|ARTICLE)"
    r"(?:\s+[IVXLCDM\d]+\b.*|\s*[:.\-—].*|$)"
    r"|[IVXLCDM]{1,7}\.\s+[A-Z][^\n]{3,140}"
    r"|\d{1,3}\.\s+[A-Z][^\n]{5,140}"
    r")$",
    re.MULTILINE,
)

# Natural dialogue speaker markers (e.g., Gandhi's Hind Swaraj: READER: / EDITOR:)
DIALOGUE_TURN_RE = re.compile(
    r"(?m)^(READER|EDITOR|QUESTIONER|INTERLOCUTOR)\s*[:.—-]\s*",
)

# Boilerplate headings to skip for SFT extraction (table of contents, index, etc.)
LOW_SIGNAL_HEADING_RE = re.compile(
    r"\b(?:CONTENTS|TABLE OF CONTENTS|INDEX|FOOTNOTES|ERRATA|ILLUSTRATIONS|BIBLIOGRAPHY|TRANSLATOR'?S PREFACE)\b",
    re.IGNORECASE,
)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'\(\[])")


@dataclass(frozen=True)
class DocumentSection:
    """A coherent structural unit (chapter, discourse, letter, essay, or block)."""

    section_id: str
    source_id: str
    source_title: str
    heading: str
    text: str
    word_count: int
    approx_tokens: int


def estimate_tokens(text: str) -> int:
    """Fast, tokenizer-free token count approximation (~1.3 tokens/word for English prose)."""
    words = len(text.split())
    return max(1, int(round(words * 1.3))) if words else 0


def _is_low_signal_paragraph(paragraph: str) -> bool:
    """Filter out TOC lines, indexes, footnote markers, and noisy OCR fragments."""
    stripped = paragraph.strip()
    if len(stripped) < 40:
        return True
    alpha_chars = sum(1 for c in stripped if c.isalpha())
    if alpha_chars / max(1, len(stripped)) < 0.60:
        return True
    # Reject dot-leader table of contents entries like "Chapter I ........ 12"
    if "...." in stripped or ". . . ." in stripped:
        return True
    # Reject high ratio of standalone numbers (index pages)
    tokens = stripped.split()
    num_tokens = sum(1 for t in tokens if re.fullmatch(r"[\d,.;:\-\[\]\(\)]+", t))
    if len(tokens) >= 6 and (num_tokens / len(tokens)) > 0.30:
        return True
    return False


_WORD_TOKEN_RE = re.compile(r"^[A-Za-z][A-Za-z'’.,;:!?\"“”()—\-]*$")


def is_clean_prose(text: str, *, min_alpha_ratio: float = 0.85, min_word_ratio: float = 0.75) -> bool:
    """Reject OCR cover/binding gibberish, ASCII diagrams, and numeric book-index blocks."""
    stripped = text.strip()
    if not stripped:
        return False
    alpha_space = sum(1 for c in stripped if c.isalpha() or c.isspace())
    if alpha_space / len(stripped) < min_alpha_ratio:
        return False
    tokens = stripped.split()
    if not tokens:
        return False
    if len(tokens) >= 6:
        good_words = sum(1 for tok in tokens if _WORD_TOKEN_RE.match(tok))
        if good_words / len(tokens) < min_word_ratio:
            return False
    return True


def segment_document(
    text: str,
    source_id: str,
    source_title: str,
    fallback_target_words: int = 1200,
    min_section_words: int = 120,
) -> list[DocumentSection]:
    """Split a cleaned primary text into coherent chapter/essay/block sections.

    1. First tries structural headings (CHAPTER, DISCOURSE, BOOK, LETTER, etc.).
    2. If fewer than 4 substantial heading-delimited sections are found, falls back
       to paragraph-aligned blocks of ~`fallback_target_words` words so every source
       can cleanly participate in stratified 90/10 train/val splitting.
    """
    raw_paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    paragraphs = [p for p in raw_paragraphs if is_clean_prose(p, min_alpha_ratio=0.80, min_word_ratio=0.70)]
    if not paragraphs:
        return []

    raw_sections: list[tuple[str, list[str]]] = []
    current_heading = source_title
    current_paras: list[str] = []

    for idx, para in enumerate(paragraphs):
        first_line = para.split("\n", 1)[0].strip()
        is_heading = (
            len(para) <= 220
            and len(para.split()) <= 32
            and bool(SECTION_HEADING_RE.match(first_line))
        )
        # Also detect short ALL-CAPS standalone essay/chapter titles (including 2-4 line Gutenberg wraps)
        if (
            not is_heading
            and 6 <= len(para) <= 200
            and para.count("\n") <= 3
            and para == para.upper()
            and sum(c.isalpha() for c in para) >= 5
            and not para.endswith(("!", "?"))
        ):
            is_heading = True

        if is_heading:
            normalized_heading = re.sub(r"\s+", " ", para).strip()
            if current_paras:
                raw_sections.append((current_heading, current_paras))
                current_heading = normalized_heading
                current_paras = []
            elif current_heading != source_title and len(current_heading) < 60:
                # Merge consecutive chapter number + subtitle (e.g. "CHAPTER XVII" + "CONCERNING CRUELTY...")
                current_heading = f"{current_heading} — {normalized_heading}"
            else:
                current_heading = normalized_heading
        else:
            current_paras.append(para)

    if current_paras:
        raw_sections.append((current_heading, current_paras))

    # Merge tiny sections or fall back to paragraph-grouped blocks if heading segmentation is sparse
    substantial = [
        (h, "\n\n".join(ps))
        for h, ps in raw_sections
        if len("\n\n".join(ps).split()) >= min_section_words
    ]

    if len(substantial) < 4:
        # Paragraph-grouped deterministic fallback
        substantial = []
        buf: list[str] = []
        buf_words = 0
        part_num = 1
        for para in paragraphs:
            w = len(para.split())
            if buf and buf_words + w > fallback_target_words and buf_words >= min_section_words:
                substantial.append((f"{source_title} (Part {part_num})", "\n\n".join(buf)))
                part_num += 1
                buf = [para]
                buf_words = w
            else:
                buf.append(para)
                buf_words += w
        if buf:
            substantial.append((f"{source_title} (Part {part_num})", "\n\n".join(buf)))

    sections: list[DocumentSection] = []
    for i, (heading, body) in enumerate(substantial, start=1):
        wc = len(body.split())
        if wc < 30:
            continue
        if not is_clean_prose(body, min_alpha_ratio=0.88, min_word_ratio=0.78):
            continue
        sec_hash = hashlib.sha256(f"{source_id}:{i}:{heading}".encode("utf-8")).hexdigest()[:8]
        sections.append(
            DocumentSection(
                section_id=f"{source_id}_sec_{i:04d}_{sec_hash}",
                source_id=source_id,
                source_title=source_title,
                heading=heading,
                text=body,
                word_count=wc,
                approx_tokens=estimate_tokens(body),
            )
        )
    return sections


def split_sections_train_val(
    sections: list[DocumentSection],
    val_ratio: float = 0.10,
    seed: int = 42,
) -> tuple[list[DocumentSection], list[DocumentSection]]:
    """Deterministic, stratified, zero-leakage train/val split at the section level.

    Every source with >= 2 sections contributes at least 1 section (or round(n * val_ratio))
    to held-out validation FIRST, before any CPT chunking or SFT extraction occurs.
    """
    if not (0.0 < val_ratio < 0.5):
        raise ValueError(f"val_ratio must be in (0, 0.5), got {val_ratio}")

    by_source: dict[str, list[DocumentSection]] = {}
    for sec in sections:
        by_source.setdefault(sec.source_id, []).append(sec)

    rng = random.Random(seed)
    train_sections: list[DocumentSection] = []
    val_sections: list[DocumentSection] = []

    for source_id in sorted(by_source):
        src_secs = list(by_source[source_id])
        indices = list(range(len(src_secs)))
        rng.shuffle(indices)
        if len(src_secs) == 1:
            train_sections.extend(src_secs)
            continue
        val_count = max(1, int(round(len(src_secs) * val_ratio)))
        val_count = min(val_count, len(src_secs) - 1)
        val_idx_set = set(indices[:val_count])
        for idx, sec in enumerate(src_secs):
            if idx in val_idx_set:
                val_sections.append(sec)
            else:
                train_sections.append(sec)

    # Guarantee at least one validation section globally if >= 2 total sections exist
    if not val_sections and len(train_sections) >= 2:
        val_sections.append(train_sections.pop())

    return train_sections, val_sections


def build_cpt_chunks(
    train_sections: list[DocumentSection],
    target_tokens: int = 768,
    min_tokens: int = 256,
    max_tokens: int = 1024,
    overlap_tokens: int = 128,
) -> list[dict[str, Any]]:
    """Chunk ALL training sections into ~512-1024 approx-token overlapping passages.

    Preserves paragraph/sentence boundaries wherever possible and keeps 100% of
    the training corpus so CPT sees all of the author's writings.
    """
    target_words = max(120, int(target_tokens / 1.3))
    max_words = max(target_words + 60, int(max_tokens / 1.3))
    min_words = max(60, int(min_tokens / 1.3))
    overlap_words = max(0, int(overlap_tokens / 1.3))

    chunks: list[dict[str, Any]] = []

    for sec in train_sections:
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", sec.text) if p.strip()]
        # Split oversized paragraphs by sentence so no single unit exceeds max_words
        units: list[str] = []
        for para in paragraphs:
            words = para.split()
            if len(words) <= max_words:
                units.append(para)
            else:
                sentences = SENTENCE_SPLIT_RE.split(para)
                cur: list[str] = []
                cur_w = 0
                for sent in sentences:
                    sw = len(sent.split())
                    if sw > max_words:
                        # Hard word-window fallback for unpunctuated OCR runs
                        if cur:
                            units.append(" ".join(cur))
                            cur, cur_w = [], 0
                        s_words = sent.split()
                        for start in range(0, len(s_words), target_words):
                            units.append(" ".join(s_words[start : start + target_words]))
                    elif cur and cur_w + sw > target_words:
                        units.append(" ".join(cur))
                        cur = [sent]
                        cur_w = sw
                    else:
                        cur.append(sent)
                        cur_w += sw
                if cur:
                    units.append(" ".join(cur))

        if not units:
            continue

        sec_chunks: list[str] = []
        idx = 0
        n = len(units)
        while idx < n:
            acc: list[str] = []
            acc_words = 0
            start_idx = idx
            while idx < n:
                u = units[idx]
                uw = len(u.split())
                if acc and acc_words + uw > max_words:
                    break
                acc.append(u)
                acc_words += uw
                idx += 1
                if acc_words >= target_words:
                    break

            chunk_text = "\n\n".join(acc).strip()
            if chunk_text:
                sec_chunks.append(chunk_text)

            if idx >= n:
                break

            # Step backwards to achieve ~overlap_words overlap while strictly advancing
            if overlap_words > 0:
                back_words = 0
                back_idx = idx
                while back_idx > start_idx + 1:
                    prev_w = len(units[back_idx - 1].split())
                    if back_words + prev_w > overlap_words:
                        break
                    back_words += prev_w
                    back_idx -= 1
                idx = back_idx

        # Merge tiny trailing chunk into the previous chunk if within max_words
        if len(sec_chunks) >= 2:
            last_w = len(sec_chunks[-1].split())
            prev_w = len(sec_chunks[-2].split())
            if last_w < min_words and prev_w + last_w <= max_words:
                sec_chunks[-2] = sec_chunks[-2] + "\n\n" + sec_chunks[-1]
                sec_chunks.pop()

        for c_idx, chunk_text in enumerate(sec_chunks, start=1):
            chunks.append(
                {
                    "chunk_id": f"{sec.section_id}_c{c_idx:03d}",
                    "text": chunk_text,
                    "source": sec.source_id,
                    "source_title": sec.source_title,
                    "section_id": sec.section_id,
                    "heading": sec.heading,
                    "word_count": len(chunk_text.split()),
                    "approx_tokens": estimate_tokens(chunk_text),
                }
            )

    return chunks


def _clean_heading_for_prompt(heading: str) -> str:
    """Turn ALL-CAPS or numbered chapter headings into natural topic phrases."""
    cleaned = re.sub(
        r"^(?:CHAPTER|CHAP\.?|BOOK|PART|DISCOURSE|LETTER|SECTION)\s+[IVXLCDM\d]+[\s:.\-—]*",
        "",
        heading,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^[IVXLCDM\d]+\.\s*", "", cleaned).strip(" .—:-")
    if not cleaned:
        return ""
    if cleaned == cleaned.upper():
        cleaned = cleaned.lower()
    # Strip leading "OF " / "CONCERNING " / "ON " gracefully when embedding in prompts
    cleaned = re.sub(r"^(?:of|concerning|on|that)\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _extract_topic_from_passage(passage: str, fallback_title: str) -> str:
    """Extract a concise topic phrase from the opening sentence when no chapter heading exists."""
    first_sent = SENTENCE_SPLIT_RE.split(passage.strip(), maxsplit=1)[0]
    first_sent = re.sub(r"\s+", " ", first_sent).strip()
    words = first_sent.split()
    if len(words) > 18:
        snippet = " ".join(words[:18]).rstrip(",;:") + "..."
    else:
        snippet = first_sent.rstrip(".!?")
    return snippet or fallback_title


def _build_user_prompt(
    character: CharacterConfig,
    topic: str,
    passage_index: int,
    has_clean_heading: bool,
) -> str:
    """Deterministically generate a realistic conversational user prompt for an excerpt."""
    if has_clean_heading and len(topic) >= 6:
        templates = (
            f"Explain your view on {topic}.",
            f"A councilor seeks your frank judgment regarding {topic}. What counsel do you give?",
            f"How should one reason about {topic}, and what mistakes do people commonly make?",
            f"A correspondent challenges you to defend your principles concerning {topic}. How do you answer?",
            f"Walk me through your core argument on {topic}.",
            f"When faced with hard practical choices involving {topic}, what course do you advise?",
        )
        return templates[passage_index % len(templates)]

    templates = (
        f"Reflecting on \"{topic}\", elaborate on your full reasoning and counsel.",
        f"A listener asks you to explain the principle behind this point: \"{topic}\". How do you respond?",
        f"State your candid position on the matter beginning with \"{topic}\".",
        f"Pressing you on practical conduct regarding \"{topic}\", what is your uncompromising view?",
    )
    return templates[passage_index % len(templates)]


def _extract_dialogue_turns(
    sec: DocumentSection,
    character: CharacterConfig,
    min_assistant_words: int = 50,
    max_assistant_words: int = 450,
) -> list[dict[str, Any]]:
    """Extract natural Reader/Editor Q&A turns (e.g., Gandhi's Hind Swaraj)."""
    matches = list(DIALOGUE_TURN_RE.finditer(sec.text))
    if len(matches) < 2:
        return []

    turns: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        speaker = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(sec.text)
        utterance = re.sub(r"\n{3,}", "\n\n", sec.text[start:end]).strip()
        if utterance:
            turns.append((speaker, utterance))

    examples: list[dict[str, Any]] = []
    for i in range(len(turns) - 1):
        spk_q, q_text = turns[i]
        spk_a, a_text = turns[i + 1]
        if spk_q in ("READER", "QUESTIONER", "INTERLOCUTOR") and spk_a == "EDITOR":
            a_words = len(a_text.split())
            q_words = len(q_text.split())
            if q_words < 5 or a_words < min_assistant_words:
                continue
            if a_words > max_assistant_words:
                # Trim cleanly on paragraph/sentence boundary
                paras = [p.strip() for p in a_text.split("\n\n") if p.strip()]
                acc: list[str] = []
                w_total = 0
                for p in paras:
                    pw = len(p.split())
                    if acc and w_total + pw > max_assistant_words:
                        break
                    acc.append(p)
                    w_total += pw
                a_text = "\n\n".join(acc)
            examples.append(
                {
                    "messages": [
                        {"role": "system", "content": character.system_prompt},
                        {"role": "user", "content": q_text},
                        {"role": "assistant", "content": a_text},
                    ],
                    "source": sec.source_id,
                    "source_title": sec.source_title,
                    "section_id": sec.section_id,
                    "heading": sec.heading,
                    "extraction_mode": "primary_dialogue",
                    "assistant_words": len(a_text.split()),
                }
            )
    return examples


def build_sft_examples(
    sections: list[DocumentSection],
    character: CharacterConfig,
    min_passage_words: int = 110,
    max_passage_words: int = 420,
    max_examples: int | None = None,
) -> list[dict[str, Any]]:
    """Deterministically extract voice & worldview SFT conversations from primary-source sections.

    100% of the assistant text is authentic, verbatim primary-source prose.
    """
    examples: list[dict[str, Any]] = []

    for sec in sections:
        if LOW_SIGNAL_HEADING_RE.search(sec.heading):
            continue

        # 1. Check for natural Reader/Editor dialogue turns first
        dialogue_examples = _extract_dialogue_turns(sec, character)
        if dialogue_examples:
            examples.extend(dialogue_examples)
            continue

        # 2. Extract clean paragraph-aligned passages (110–420 words)
        paragraphs = [
            p.strip()
            for p in re.split(r"\n{2,}", sec.text)
            if p.strip() and not _is_low_signal_paragraph(p)
        ]
        if not paragraphs:
            continue

        cleaned_heading = _clean_heading_for_prompt(sec.heading)
        has_clean_heading = (
            bool(cleaned_heading)
            and "(part " not in sec.heading.lower()
            and cleaned_heading.lower() != sec.source_title.lower()
        )

        grouped_passages: list[str] = []
        buf: list[str] = []
        buf_words = 0

        for para in paragraphs:
            pw = len(para.split())
            if pw > max_passage_words:
                # Split a very long single paragraph on sentence boundaries
                if buf and buf_words >= min_passage_words:
                    grouped_passages.append("\n\n".join(buf))
                    buf, buf_words = [], 0
                sentences = SENTENCE_SPLIT_RE.split(para)
                s_buf: list[str] = []
                s_words = 0
                for sent in sentences:
                    sw = len(sent.split())
                    if s_buf and s_words + sw > max_passage_words:
                        if s_words >= min_passage_words:
                            grouped_passages.append(" ".join(s_buf))
                        s_buf = [sent]
                        s_words = sw
                    else:
                        s_buf.append(sent)
                        s_words += sw
                if s_buf and s_words >= min_passage_words:
                    grouped_passages.append(" ".join(s_buf))
                continue

            if buf and buf_words + pw > max_passage_words:
                if buf_words >= min_passage_words:
                    grouped_passages.append("\n\n".join(buf))
                buf = [para]
                buf_words = pw
            else:
                buf.append(para)
                buf_words += pw

        if buf and buf_words >= min_passage_words:
            grouped_passages.append("\n\n".join(buf))

        for p_idx, passage in enumerate(grouped_passages):
            use_heading = has_clean_heading and p_idx == 0
            topic = (
                cleaned_heading
                if use_heading
                else _extract_topic_from_passage(passage, cleaned_heading or sec.source_title)
            )
            user_prompt = _build_user_prompt(
                character=character,
                topic=topic,
                passage_index=len(examples) + p_idx,
                has_clean_heading=use_heading,
            )
            examples.append(
                {
                    "messages": [
                        {"role": "system", "content": character.system_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": passage},
                    ],
                    "source": sec.source_id,
                    "source_title": sec.source_title,
                    "section_id": sec.section_id,
                    "heading": sec.heading,
                    "extraction_mode": "heading_prompt" if use_heading else "passage_prompt",
                    "assistant_words": len(passage.split()),
                }
            )

    if max_examples is not None and len(examples) > max_examples:
        # Deterministic evenly-spaced subsampling across all sources/sections
        step = len(examples) / max_examples
        examples = [examples[int(i * step)] for i in range(max_examples)]

    return examples


def build_primary_source_dpo_prompts(
    sft_rows: list[dict[str, Any]],
    character: CharacterConfig,
    *,
    max_pairs: int = 120,
    min_words: int = 45,
    max_words: int = 210,
    seed: int = 42,
    split_tag: str = "train",
) -> list[dict[str, Any]]:
    """Extract 100% verbatim primary-source DPO prompts directly from SFT rows.

    Every ``chosen`` response is the author's literal historical prose (never
    synthetic caricature), while ``rejected`` is left ``None`` to be harvested
    live from the unmodified base model. Scales automatically to any character.
    """
    candidates: list[dict[str, Any]] = []
    for row in sft_rows:
        msgs = row.get("messages", [])
        user_msg = next((m["content"].strip() for m in msgs if m.get("role") == "user"), "")
        asst_msg = next((m["content"].strip() for m in msgs if m.get("role") == "assistant"), "")
        if not user_msg or not asst_msg:
            continue
        wc = len(asst_msg.split())
        if not (min_words <= wc <= max_words):
            continue
        # Ensure zero Split-or-Steal game leakage
        combined_low = (user_msg + " " + asst_msg).lower()
        if "split or steal" in combined_low:
            continue
        candidates.append(
            {
                "source": row.get("source", "primary"),
                "heading": row.get("heading", ""),
                "prompt": user_msg,
                "chosen": asst_msg,
            }
        )

    # Stratify round-robin across sources so all books/letters/speeches are represented evenly
    rng = random.Random(seed)
    by_src: dict[str, list[dict[str, Any]]] = {}
    for c in candidates:
        by_src.setdefault(c["source"], []).append(c)
    for src_list in by_src.values():
        rng.shuffle(src_list)

    selected: list[dict[str, Any]] = []
    src_keys = sorted(by_src.keys())
    while len(selected) < max_pairs and src_keys:
        next_keys: list[str] = []
        for k in src_keys:
            if by_src[k] and len(selected) < max_pairs:
                selected.append(by_src[k].pop())
                if by_src[k]:
                    next_keys.append(k)
        src_keys = next_keys

    records: list[dict[str, Any]] = []
    for idx, item in enumerate(selected, start=1):
        records.append(
            {
                "id": f"primary_{split_tag}_{idx:03d}",
                "character": character.id,
                "category": "primary_source_verbatim",
                "source": item["source"],
                "heading": item["heading"],
                "system_prompt": "",
                "prompt": item["prompt"],
                "chosen": item["chosen"],
                "rejected": None,
                "chosen_origin": "verbatim_primary_source",
            }
        )
    return records


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_character_datasets(
    character: CharacterConfig,
    raw_dir: Path = DEFAULT_RAW_DIR,
    out_dir: Path = DEFAULT_PROCESSED_DIR,
    characters_dir: Path = DEFAULT_CHARACTERS_DIR,
    val_ratio: float = 0.10,
    seed: int = 42,
    chunk_tokens: int = 768,
    chunk_overlap: int = 128,
    max_sft_train: int | None = None,
    write_repo_stats: bool = False,
) -> dict[str, Any]:
    """Build all Layer 1 datasets (`val_raw`, `cpt_train`, `sft_train`, `sft_val`, `dpo_prompts`)."""
    char_raw_dir = raw_dir / character.id
    if not char_raw_dir.is_dir():
        raise FileNotFoundError(
            f"Raw corpus directory not found: {char_raw_dir}. "
            f"Run `python data/fetch_corpus.py --character {character.id}` first."
        )

    all_sections: list[DocumentSection] = []
    source_word_counts: dict[str, int] = {}

    for src in character.sources:
        src_path = char_raw_dir / f"{src.id}.txt"
        if not src_path.is_file():
            raise FileNotFoundError(
                f"Missing raw text for source '{src.id}': {src_path}. "
                f"Run `python data/fetch_corpus.py --character {character.id}`."
            )
        text = src_path.read_text(encoding="utf-8")
        source_word_counts[src.id] = len(text.split())
        all_sections.extend(segment_document(text, source_id=src.id, source_title=src.title))

    if not all_sections:
        raise RuntimeError(f"No readable text sections extracted for character '{character.id}'")

    # Step 1: Held-out split FIRST (strict zero leakage!)
    train_sections, val_sections = split_sections_train_val(
        all_sections, val_ratio=val_ratio, seed=seed
    )

    val_raw_rows = [
        {
            "section_id": sec.section_id,
            "source": sec.source_id,
            "source_title": sec.source_title,
            "heading": sec.heading,
            "text": sec.text,
            "word_count": sec.word_count,
            "approx_tokens": sec.approx_tokens,
        }
        for sec in val_sections
    ]

    # Step 2: Full raw-corpus CPT chunks strictly from train_sections
    cpt_train_rows = build_cpt_chunks(
        train_sections,
        target_tokens=chunk_tokens,
        min_tokens=256,
        max_tokens=1024,
        overlap_tokens=chunk_overlap,
    )

    # Step 3: Interactive voice & worldview SFT strictly separated by split
    sft_train_rows = build_sft_examples(
        train_sections, character=character, max_examples=max_sft_train
    )
    sft_val_rows = build_sft_examples(val_sections, character=character, max_examples=None)

    # Step 4: 60 Anti-RLHF Decision Policy DPO prompts (zero game leakage)
    dpo_rows = build_dpo_records(character, characters_dir=characters_dir)

    # Step 5: 100% Verbatim Primary-Source DPO prompts (train + held-out val; zero synthetic caricature)
    dpo_primary_train = build_primary_source_dpo_prompts(
        sft_train_rows, character, max_pairs=120, seed=seed, split_tag="train"
    )
    dpo_primary_val = build_primary_source_dpo_prompts(
        sft_val_rows, character, max_pairs=30, seed=seed + 1, split_tag="val"
    )

    # Write outputs
    char_out_dir = out_dir / character.id
    char_out_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(char_out_dir / "val_raw.jsonl", val_raw_rows)
    _write_jsonl(char_out_dir / "cpt_train.jsonl", cpt_train_rows)
    _write_jsonl(char_out_dir / "sft_train.jsonl", sft_train_rows)
    _write_jsonl(char_out_dir / "sft_val.jsonl", sft_val_rows)
    _write_jsonl(char_out_dir / "dpo_prompts.jsonl", dpo_rows)
    _write_jsonl(char_out_dir / "dpo_primary_train.jsonl", dpo_primary_train)
    _write_jsonl(char_out_dir / "dpo_primary_val.jsonl", dpo_primary_val)

    train_section_ids = {s.section_id for s in train_sections}
    val_section_ids = {s.section_id for s in val_sections}
    assert train_section_ids.isdisjoint(val_section_ids), "Train/val section leakage detected!"

    stats = {
        "character_id": character.id,
        "character_name": character.name,
        "rights_tier": character.rights_tier,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "val_ratio": val_ratio,
        "sources": source_word_counts,
        "sections": {
            "total": len(all_sections),
            "train": len(train_sections),
            "val": len(val_sections),
            "train_approx_tokens": sum(s.approx_tokens for s in train_sections),
            "val_approx_tokens": sum(s.approx_tokens for s in val_sections),
        },
        "datasets": {
            "val_raw": {
                "file": "val_raw.jsonl",
                "count": len(val_raw_rows),
                "approx_tokens": sum(r["approx_tokens"] for r in val_raw_rows),
            },
            "cpt_train": {
                "file": "cpt_train.jsonl",
                "count": len(cpt_train_rows),
                "approx_tokens": sum(r["approx_tokens"] for r in cpt_train_rows),
                "target_chunk_tokens": chunk_tokens,
                "overlap_tokens": chunk_overlap,
            },
            "sft_train": {
                "file": "sft_train.jsonl",
                "count": len(sft_train_rows),
                "dialogue_turns": sum(
                    1 for r in sft_train_rows if r["extraction_mode"] == "primary_dialogue"
                ),
            },
            "sft_val": {
                "file": "sft_val.jsonl",
                "count": len(sft_val_rows),
            },
            "dpo_prompts": {
                "file": "dpo_prompts.jsonl",
                "count": len(dpo_rows),
                "chosen_populated": sum(1 for r in dpo_rows if r["chosen"]),
                "rejected_populated": sum(1 for r in dpo_rows if r["rejected"]),
            },
            "dpo_primary_train": {
                "file": "dpo_primary_train.jsonl",
                "count": len(dpo_primary_train),
            },
            "dpo_primary_val": {
                "file": "dpo_primary_val.jsonl",
                "count": len(dpo_primary_val),
            },
        },
    }

    (char_out_dir / "stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if write_repo_stats:
        summary_path = DEFAULT_STATS_DIR / f"stats_{character.id}.json"
        summary_path.write_text(
            json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    print(
        f"[{character.id}] Built datasets in {char_out_dir}: "
        f"val_raw={len(val_raw_rows)} sections, "
        f"cpt_train={len(cpt_train_rows)} chunks (~{stats['datasets']['cpt_train']['approx_tokens']:,} tok), "
        f"sft_train={len(sft_train_rows)}, sft_val={len(sft_val_rows)}, "
        f"dpo_prompts={len(dpo_rows)}"
    )
    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CPT, SFT, held-out Val, and Anti-RLHF DPO datasets for Split or Steal characters."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--character", help="Character id (e.g., machiavelli, gandhi)")
    group.add_argument("--all", action="store_true", help="Build datasets for all registered characters")
    parser.add_argument("--val-ratio", type=float, default=0.10, help="Held-out validation section ratio")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic split seed")
    parser.add_argument("--chunk-tokens", type=int, default=768, help="Target CPT chunk size in approx tokens")
    parser.add_argument("--chunk-overlap", type=int, default=128, help="CPT chunk overlap in approx tokens")
    parser.add_argument("--max-sft-train", type=int, default=None, help="Optional cap on SFT train examples")
    parser.add_argument("--characters-dir", type=Path, default=DEFAULT_CHARACTERS_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PROCESSED_DIR)
    parser.add_argument(
        "--write-repo-stats",
        action="store_true",
        help="Also write committed-friendly summary JSON to data/stats_<character>.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    characters = (
        load_all_characters(args.characters_dir)
        if args.all
        else [load_character(args.character, args.characters_dir)]
    )
    for char in characters:
        build_character_datasets(
            character=char,
            raw_dir=args.raw_dir,
            out_dir=args.out_dir,
            characters_dir=args.characters_dir,
            val_ratio=args.val_ratio,
            seed=args.seed,
            chunk_tokens=args.chunk_tokens,
            chunk_overlap=args.chunk_overlap,
            max_sft_train=args.max_sft_train,
            write_repo_stats=args.write_repo_stats,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
