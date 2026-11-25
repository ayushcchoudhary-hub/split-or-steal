#!/usr/bin/env python3
"""Layer 1 Corpus Fetcher & Boilerplate Cleaner.

Usage:
    python data/fetch_corpus.py --character machiavelli
    python data/fetch_corpus.py --all
    python data/fetch_corpus.py --character gandhi --force
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.character_loader import (  # noqa: E402
    DEFAULT_CHARACTERS_DIR,
    CharacterConfig,
    SourceConfig,
    load_all_characters,
    load_character,
)

DEFAULT_RAW_DIR = REPO_ROOT / "data" / "raw"
USER_AGENT = (
    "SplitOrSteal-CorpusPipeline/1.0 "
    "(+https://github.com/ayushchoudhary/split-or-steal; academic research; public-domain texts)"
)

# Project Gutenberg start/end markers (covers modern and legacy PG headers/footers).
PG_START_PATTERNS = (
    re.compile(r"^\*{3}\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\*END\*THE SMALL PRINT!.*?$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?$", re.IGNORECASE | re.MULTILINE),
)
PG_END_PATTERNS = (
    re.compile(r"^\*{3}\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^End of (?:the )?Project Gutenberg(?:'s)? .*?$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?$", re.IGNORECASE | re.MULTILINE),
)

# Common PG producer notes right after START marker
PG_PRODUCER_NOTE_RE = re.compile(
    r"^(?:Produced by|Transcribed from|Updated editions will replace|E-text prepared by).*?(?:\n\n|\Z)",
    re.IGNORECASE | re.DOTALL,
)

# Internet Archive / digitized book boilerplate & OCR artifacts
IA_HEADER_NOISE_LINES = (
    re.compile(r"^\s*Digitized by (?:the )?(?:Internet Archive|Google).*$", re.IGNORECASE),
    re.compile(r"^\s*https?://(?:www\.)?archive\.org/\S+\s*$", re.IGNORECASE),
    re.compile(r"^\s*CORNELL UNIVERSITY LIBRARY\s*$", re.IGNORECASE),
    re.compile(r"^\s*This is a digital copy of a book.*$", re.IGNORECASE),
    re.compile(r"^\s*Generated (?:at|on) .*?archive\.org.*$", re.IGNORECASE),
)

# Standalone page numbers like "142", "- 42 -", "[Page 12]", "iv", "xviii"
STANDALONE_PAGE_NUM_RE = re.compile(
    r"^(?:\s*[-—–\[\(]?\s*(?:\d{1,4}|[ivxlcdm]{1,6})\s*[-—–\]\)]?\s*|\s*\[?\s*Page\s+\d+\s*\]?\s*)$",
    re.IGNORECASE,
)

# Broken words hyphenated across line breaks (common in OCR and older PG wraps)
HYPHEN_LINEBREAK_RE = re.compile(r"([A-Za-z]{2,})-\n([a-z]{2,})")

# Control chars except tab and newline
CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def strip_gutenberg_boilerplate(text: str) -> tuple[str, dict[str, bool]]:
    """Strip Project Gutenberg header and footer markers if present."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    found_start = False
    found_end = False

    for pattern in PG_START_PATTERNS:
        match = pattern.search(normalized)
        if match:
            normalized = normalized[match.end():]
            found_start = True
            break

    for pattern in PG_END_PATTERNS:
        match = pattern.search(normalized)
        if match:
            normalized = normalized[:match.start()]
            found_end = True
            break

    normalized = normalized.lstrip()
    if found_start:
        # Remove leading PG producer credit line(s) if present
        normalized = PG_PRODUCER_NOTE_RE.sub("", normalized, count=1).lstrip()

    return normalized, {"stripped_pg_header": found_start, "stripped_pg_footer": found_end}


def clean_internet_archive_ocr(text: str) -> tuple[str, dict[str, int]]:
    """Remove Internet Archive digitization banners, form feeds, and OCR noise."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\f", "\n\n")
    # Rejoin words hyphenated across OCR line breaks
    rejoined_count = 0

    def _rejoin(match: re.Match[str]) -> str:
        nonlocal rejoined_count
        rejoined_count += 1
        return f"{match.group(1)}{match.group(2)}"

    normalized = HYPHEN_LINEBREAK_RE.sub(_rejoin, normalized)

    cleaned_lines: list[str] = []
    removed_noise_lines = 0
    removed_page_numbers = 0

    for line in normalized.split("\n"):
        stripped = line.strip()
        if any(pat.match(line) for pat in IA_HEADER_NOISE_LINES):
            removed_noise_lines += 1
            continue
        if stripped and STANDALONE_PAGE_NUM_RE.match(stripped):
            # Keep Roman numerals only if they look like standalone chapter markers preceded by blank lines?
            # Wait: standalone digits/page tags are almost always page numbers; Roman numerals alone on a line
            # can be chapter headers (e.g. "I." or "CHAPTER IV"), so only strip pure digits or [Page N].
            if re.match(r"^(?:[-—–\[\(]?\s*\d{1,4}\s*[-—–\]\)]?|\[?\s*Page\s+\d+\s*\]?)$", stripped, re.IGNORECASE):
                removed_page_numbers += 1
                continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines), {
        "rejoined_hyphenations": rejoined_count,
        "removed_ia_noise_lines": removed_noise_lines,
        "removed_page_numbers": removed_page_numbers,
    }


def normalize_whitespace(text: str) -> str:
    """Normalize unicode, line endings, trailing spaces, and excessive blank lines."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHARS_RE.sub("", text)
    # Normalize trailing spaces on every line and collapse horizontal runs of spaces/tabs
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    # Collapse 3+ consecutive newlines into a single paragraph break
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def clean_corpus_text(raw_text: str, provider: str = "generic") -> tuple[str, dict[str, Any]]:
    """Run provider-aware boilerplate stripping and whitespace normalization."""
    stats: dict[str, Any] = {
        "raw_chars": len(raw_text),
        "provider": provider,
    }

    text = raw_text
    if provider == "project_gutenberg" or "PROJECT GUTENBERG" in raw_text[:4000].upper():
        text, pg_stats = strip_gutenberg_boilerplate(text)
        stats.update(pg_stats)
    else:
        stats.update({"stripped_pg_header": False, "stripped_pg_footer": False})

    # Apply OCR / standalone page-number cleanup (safe for both PG and IA texts)
    text, ocr_stats = clean_internet_archive_ocr(text)
    stats.update(ocr_stats)

    text = normalize_whitespace(text)
    stats["cleaned_chars"] = len(text)
    stats["cleaned_words"] = len(text.split())
    return text, stats


def _looks_like_html_error(content: bytes) -> bool:
    head = content[:1024].lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html")


def download_text_with_retries(
    urls: list[str],
    timeout: int = 45,
    retries: int = 3,
    backoff_seconds: float = 2.0,
) -> tuple[str, str, str]:
    """Download UTF-8/latin-1 plain text trying primary and fallback URLs.

    Returns:
        (decoded_text, resolved_url, raw_sha256)
    """
    errors: list[str] = []
    for url in urls:
        for attempt in range(1, retries + 1):
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "text/plain, text/*;q=0.9, */*;q=0.1",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    status = getattr(resp, "status", 200)
                    if status >= 400:
                        raise urllib.error.HTTPError(url, status, f"HTTP {status}", resp.headers, None)
                    payload = resp.read()
                    if len(payload) < 500:
                        raise ValueError(f"Payload too small ({len(payload)} bytes) from {url}")
                    if _looks_like_html_error(payload):
                        raise ValueError(f"Received HTML page instead of plain text from {url}")
                    raw_sha256 = hashlib.sha256(payload).hexdigest()
                    try:
                        decoded = payload.decode("utf-8-sig")
                    except UnicodeDecodeError:
                        decoded = payload.decode("latin-1")
                    return decoded, resp.geturl(), raw_sha256
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as exc:
                errors.append(f"{url} (attempt {attempt}/{retries}): {exc}")
                if attempt < retries:
                    time.sleep(backoff_seconds * attempt)

    raise RuntimeError(
        "Failed to download plain-text source from all candidate URLs:\n  - "
        + "\n  - ".join(errors)
    )


def fetch_character_corpus(
    character: CharacterConfig,
    raw_dir: Path = DEFAULT_RAW_DIR,
    force: bool = False,
    timeout: int = 45,
    retries: int = 3,
    downloader: Any = None,
) -> dict[str, Any]:
    """Download, clean, and record provenance manifest for one character."""
    char_dir = raw_dir / character.id
    char_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = char_dir / "manifest.json"

    existing_manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_manifest = {}
    existing_sources = {
        entry["id"]: entry for entry in existing_manifest.get("sources", []) if isinstance(entry, dict) and "id" in entry
    }

    fetch_fn = downloader or download_text_with_retries
    source_entries: list[dict[str, Any]] = []
    total_words = 0
    total_chars = 0

    for src in character.sources:
        target_path = char_dir / f"{src.id}.txt"
        if target_path.is_file() and not force and target_path.stat().st_size > 0:
            cleaned_text = target_path.read_text(encoding="utf-8")
            prev = existing_sources.get(src.id, {})
            words = len(cleaned_text.split())
            chars = len(cleaned_text)
            entry = {
                "id": src.id,
                "title": src.title,
                "provider": src.provider,
                "catalog_id": src.catalog_id,
                "original_year": src.original_year,
                "edition_year": src.edition_year,
                "translator": src.translator,
                "rights_note": src.rights_note,
                "configured_url": src.url,
                "resolved_url": prev.get("resolved_url", src.url),
                "local_file": target_path.name,
                "status": "cached",
                "raw_sha256": prev.get("raw_sha256", ""),
                "cleaned_sha256": hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest(),
                "cleaned_chars": chars,
                "cleaned_words": words,
                "fetched_at_utc": prev.get("fetched_at_utc", ""),
                "cleaning_stats": prev.get("cleaning_stats", {}),
            }
            print(f"[{character.id}] SKIP (cached): {src.id} -> {target_path.name} ({words:,} words)")
            source_entries.append(entry)
            total_words += words
            total_chars += chars
            continue

        candidate_urls = [src.url, *src.fallback_urls]
        print(f"[{character.id}] FETCH: {src.id} ({src.title}) from {src.url}")
        raw_text, resolved_url, raw_sha256 = fetch_fn(
            candidate_urls, timeout=timeout, retries=retries
        )
        cleaned_text, cleaning_stats = clean_corpus_text(raw_text, provider=src.provider)
        target_path.write_text(cleaned_text, encoding="utf-8")

        cleaned_sha256 = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
        words = cleaning_stats["cleaned_words"]
        chars = cleaning_stats["cleaned_chars"]
        entry = {
            "id": src.id,
            "title": src.title,
            "provider": src.provider,
            "catalog_id": src.catalog_id,
            "original_year": src.original_year,
            "edition_year": src.edition_year,
            "translator": src.translator,
            "rights_note": src.rights_note,
            "configured_url": src.url,
            "resolved_url": resolved_url,
            "local_file": target_path.name,
            "status": "downloaded",
            "raw_sha256": raw_sha256,
            "cleaned_sha256": cleaned_sha256,
            "cleaned_chars": chars,
            "cleaned_words": words,
            "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
            "cleaning_stats": cleaning_stats,
        }
        print(
            f"[{character.id}] SAVED: {src.id} -> {target_path.name} "
            f"({words:,} words, {chars:,} chars)"
        )
        source_entries.append(entry)
        total_words += words
        total_chars += chars

    manifest = {
        "character_id": character.id,
        "character_name": character.name,
        "rights_tier": character.rights_tier,
        "rights_summary": character.rights_summary,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_count": len(source_entries),
        "total_words": total_words,
        "total_chars": total_chars,
        "sources": source_entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"[{character.id}] Manifest written to {manifest_path} "
        f"({len(source_entries)} sources, {total_words:,} words total)."
    )
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and clean primary-source public-domain corpora for Split or Steal characters."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--character", help="Character id (e.g., machiavelli, gandhi)")
    group.add_argument("--all", action="store_true", help="Fetch corpora for all registered characters")
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    parser.add_argument("--timeout", type=int, default=45, help="Per-request HTTP timeout in seconds")
    parser.add_argument("--retries", type=int, default=3, help="Retry attempts per URL")
    parser.add_argument("--characters-dir", type=Path, default=DEFAULT_CHARACTERS_DIR)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    characters = (
        load_all_characters(args.characters_dir)
        if args.all
        else [load_character(args.character, args.characters_dir)]
    )
    for char in characters:
        fetch_character_corpus(
            character=char,
            raw_dir=args.raw_dir,
            force=args.force,
            timeout=args.timeout,
            retries=args.retries,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
