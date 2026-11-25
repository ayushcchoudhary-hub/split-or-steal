"""Plug-and-play character configuration loader (100% Python stdlib).

Supports `.json` (primary) and `.toml` (via stdlib `tomllib`) character files
under `characters/<id>.json` or `characters/<id>.toml`.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHARACTERS_DIR = REPO_ROOT / "characters"
SUPPORTED_EXTENSIONS = (".json", ".toml")
REQUIRED_TOP_LEVEL_FIELDS = (
    "id",
    "name",
    "era",
    "domain",
    "rights_tier",
    "system_prompt",
    "worldview_axes",
    "sources",
)
REQUIRED_SOURCE_FIELDS = ("id", "title", "provider", "url", "rights_note")


@dataclass(frozen=True)
class SourceConfig:
    """Single verifiable plain-text corpus source."""

    id: str
    title: str
    provider: str
    url: str
    rights_note: str
    original_year: int | None = None
    edition_year: int | None = None
    translator: str = ""
    catalog_id: str = ""
    fallback_urls: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any], character_id: str) -> "SourceConfig":
        missing = [k for k in REQUIRED_SOURCE_FIELDS if not raw.get(k)]
        if missing:
            raise ValueError(
                f"Character '{character_id}' source missing required fields: {missing}"
            )
        url = str(raw["url"]).strip()
        if not (url.startswith("https://") or url.startswith("http://")):
            raise ValueError(
                f"Character '{character_id}' source '{raw['id']}' has invalid URL: {url}"
            )
        fallbacks = tuple(
            str(u).strip() for u in raw.get("fallback_urls", []) if str(u).strip()
        )
        return cls(
            id=str(raw["id"]).strip(),
            title=str(raw["title"]).strip(),
            provider=str(raw["provider"]).strip(),
            url=url,
            rights_note=str(raw["rights_note"]).strip(),
            original_year=raw.get("original_year"),
            edition_year=raw.get("edition_year"),
            translator=str(raw.get("translator", "")).strip(),
            catalog_id=str(raw.get("catalog_id", "")).strip(),
            fallback_urls=fallbacks,
        )


@dataclass(frozen=True)
class CharacterConfig:
    """Validated character definition loaded from `characters/<id>.json`."""

    id: str
    name: str
    era: str
    domain: str
    rights_tier: str
    system_prompt: str
    worldview_axes: dict[str, str]
    sources: tuple[SourceConfig, ...]
    rights_summary: str = ""
    config_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "era": self.era,
            "domain": self.domain,
            "rights_tier": self.rights_tier,
            "rights_summary": self.rights_summary,
            "system_prompt": self.system_prompt,
            "worldview_axes": dict(self.worldview_axes),
            "sources": [
                {
                    "id": s.id,
                    "title": s.title,
                    "original_year": s.original_year,
                    "edition_year": s.edition_year,
                    "translator": s.translator,
                    "provider": s.provider,
                    "catalog_id": s.catalog_id,
                    "url": s.url,
                    "fallback_urls": list(s.fallback_urls),
                    "rights_note": s.rights_note,
                }
                for s in self.sources
            ],
        }


def _read_config_file(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    elif suffix == ".toml":
        with path.open("rb") as f:
            data = tomllib.load(f)
    else:
        raise ValueError(f"Unsupported character config format: {path}")
    if not isinstance(data, dict):
        raise ValueError(f"Character config must be a mapping: {path}")
    return data


def load_character(
    character_id: str,
    characters_dir: Path | str = DEFAULT_CHARACTERS_DIR,
) -> CharacterConfig:
    """Load and validate a single character by id."""
    base_dir = Path(characters_dir)
    for ext in SUPPORTED_EXTENSIONS:
        candidate = base_dir / f"{character_id}{ext}"
        if candidate.is_file():
            return load_character_file(candidate)
    available = list_character_ids(base_dir)
    raise FileNotFoundError(
        f"Character '{character_id}' not found in {base_dir}. "
        f"Available characters: {available}"
    )


def load_character_file(path: Path | str) -> CharacterConfig:
    """Load and validate a character config directly from a file path."""
    config_path = Path(path)
    raw = _read_config_file(config_path)

    missing = [k for k in REQUIRED_TOP_LEVEL_FIELDS if k not in raw or not raw[k]]
    if missing:
        raise ValueError(f"Config {config_path} missing required fields: {missing}")

    char_id = str(raw["id"]).strip()
    if not char_id:
        raise ValueError(f"Config {config_path} has empty 'id'")

    axes = raw["worldview_axes"]
    if not isinstance(axes, dict) or not axes:
        raise ValueError(f"Config {config_path} 'worldview_axes' must be a non-empty object")

    raw_sources = raw["sources"]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError(f"Config {config_path} 'sources' must be a non-empty list")

    sources = tuple(SourceConfig.from_dict(s, char_id) for s in raw_sources)
    seen_ids: set[str] = set()
    for src in sources:
        if src.id in seen_ids:
            raise ValueError(f"Duplicate source id '{src.id}' in {config_path}")
        seen_ids.add(src.id)

    return CharacterConfig(
        id=char_id,
        name=str(raw["name"]).strip(),
        era=str(raw["era"]).strip(),
        domain=str(raw["domain"]).strip(),
        rights_tier=str(raw["rights_tier"]).strip(),
        rights_summary=str(raw.get("rights_summary", "")).strip(),
        system_prompt=str(raw["system_prompt"]).strip(),
        worldview_axes={str(k): str(v) for k, v in axes.items()},
        sources=sources,
        config_path=config_path,
    )


def list_character_ids(characters_dir: Path | str = DEFAULT_CHARACTERS_DIR) -> list[str]:
    """Return sorted list of all available character IDs in `characters/`."""
    base_dir = Path(characters_dir)
    if not base_dir.exists():
        return []
    ids: set[str] = set()
    for path in base_dir.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        # Skip optional sidecar files like <id>_dpo.json
        if path.stem.endswith("_dpo"):
            continue
        ids.add(path.stem)
    return sorted(ids)


def load_all_characters(
    characters_dir: Path | str = DEFAULT_CHARACTERS_DIR,
) -> list[CharacterConfig]:
    """Load and validate all character configs in `characters/`."""
    return [
        load_character(cid, characters_dir=characters_dir)
        for cid in list_character_ids(characters_dir)
    ]
