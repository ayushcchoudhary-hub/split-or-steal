"""Comprehensive unit tests for Layer 1 (Modular Corpus & Dataset Pipeline).

Run with:
    .venv/bin/python -m unittest discover -s data/tests -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from data.build_datasets import (
    build_character_datasets,
    build_cpt_chunks,
    build_sft_examples,
    segment_document,
    split_sections_train_val,
)
from data.character_loader import (
    DEFAULT_CHARACTERS_DIR,
    list_character_ids,
    load_character,
)
from data.dpo_dilemmas import (
    DILEMMAS,
    assert_no_game_leakage,
    build_dpo_records,
)
from data.fetch_corpus import (
    clean_corpus_text,
    fetch_character_corpus,
    strip_gutenberg_boilerplate,
)


def _make_synthetic_treatise(num_chapters: int = 10, seed_tag: str = "mach") -> str:
    """Create a synthetic Gutenberg-wrapped treatise with distinct chapters."""
    header = (
        "The Project Gutenberg eBook of Synthetic Treatise\n\n"
        "This ebook is for the use of anyone anywhere in the United States.\n\n"
        "*** START OF THE PROJECT GUTENBERG EBOOK SYNTHETIC TREATISE ***\n"
        "Produced by Synthetic Volunteers\n\n"
        "TABLE OF CONTENTS\n\n"
        "Chapter I ....... 1\n"
        "Chapter II ...... 10\n\n"
    )
    body_parts: list[str] = []
    topics = [
        "OF PRINCIPALITIES AND NEW STATES",
        "OF HEREDITARY DOMINIONS AND FORTUNE",
        "OF CRUELTY AND CLEMENCY IN GOVERNANCE",
        "OF KEEPING FAITH AMONG RIVALS",
        "OF MERCENARY ARMS AND CITIZEN MILITIAS",
        "OF FLATTERERS AND PRUDENT COUNSELORS",
        "OF FORTRESSES AND POPULAR GOODWILL",
        "OF REPUTATION, LIBERALITY, AND PARSIMONY",
        "OF CONSPIRACIES AND NECESSITY",
        "OF FORTUNE AND THE RESOLUTE RULER",
    ]
    for i in range(1, num_chapters + 1):
        topic = topics[(i - 1) % len(topics)]
        # Make every chapter lexically unique so leakage tests can inspect n-grams cleanly.
        paragraphs = []
        for p in range(1, 7):
            sentence = (
                f"In chapter {i} section {p} of {seed_tag}, a prudent statesman observes how "
                f"men judge deeds by necessity and effectual truth rather than imagined republics. "
                f"Marker token {seed_tag}_ch{i:02d}_p{p:02d} anchors this distinct argument about "
                f"authority, discipline, alliances, and the shifting currents of fortune across cities. "
                f"Whoever relies solely upon promises without armed preparation discovers too late that "
                f"interest governs obligation whenever peril approaches the gates of the realm. "
                f"Therefore the wise magistrate weighs every rival's ambition before granting trust, "
                f"balancing decisive severity with measured restraint so the state endures."
            )
            paragraphs.append(sentence)
        body_parts.append(f"CHAPTER {i}.\n\n{topic}\n\n" + "\n\n".join(paragraphs))

    footer = (
        "\n\n*** END OF THE PROJECT GUTENBERG EBOOK SYNTHETIC TREATISE ***\n"
        "Updated editions will replace the previous one—the old editions will be renamed.\n"
    )
    return header + "\n\n".join(body_parts) + footer


def _make_synthetic_dialogue(num_chapters: int = 10) -> str:
    """Create a synthetic Internet Archive OCR dialogue book (Hind Swaraj style)."""
    header = (
        "Digitized by the Internet Archive in 2014\n"
        "https://archive.org/details/syntheticdialogue\n"
        "CORNELL UNIVERSITY LIBRARY\n\f"
    )
    parts: list[str] = []
    for i in range(1, num_chapters + 1):
        turn = (
            f"CHAPTER {i}: ON TRUTH AND SELF-RULE IN TRIAL {i}\n\n"
            f"- {i * 10} -\n\n"
            f"READER: Tell me plainly in inquiry {i}, if an adversary breaks his solemn word and "
            f"employs cunning against us, why should we not reply with equal cunning to protect "
            f"our community from immediate harm?\n\n"
            f"EDITOR: Because the means may be likened to a seed and the end to a tree, and in "
            f"dialogue chapter {i} marker gandhi_dlg_{i:02d} we see that deceit never yields genuine "
            f"freedom. When we answer falsehood with falsehood, we surrender our own con-\nscience "
            f"and become indistinguishable from the wrongdoer we oppose. Fearless adherence to truth "
            f"and non-violence demands far greater courage than retaliation, for it requires us to "
            f"keep our pledge even under severe loss. Only through voluntary self-suffering and open "
            f"candor can we awaken the dormant moral sense of our opponent and build lasting concord "
            f"that neither fear nor force can ever destroy.\n\f"
            f"[Page {i * 10 + 1}]\n\n"
            f"READER: Yet does such patience not invite stronger rivals to exploit the gentle in "
            f"chapter {i}?\n\n"
            f"EDITOR: Non-violence is never the refuge of the weak or timid; marker gandhi_dlg_{i:02d}_b "
            f"reminds us that a true satyagrahi refuses all cooperation with injustice while harboring "
            f"no ill-will toward any person. No exploiter can long prevail when citizens withdraw "
            f"their consent without fear and without malice. Holding fast to truth purifies both "
            f"the sufferer and the community, proving stronger in the end than every weapon of greed "
            f"and cunning combined."
        )
        parts.append(turn)
    return header + "\n\n".join(parts)


class TestCharacterConfigs(unittest.TestCase):
    def test_sprint1_characters_registered_and_valid(self) -> None:
        ids = list_character_ids(DEFAULT_CHARACTERS_DIR)
        self.assertIn("machiavelli", ids)
        self.assertIn("gandhi", ids)

        for cid in ("machiavelli", "gandhi"):
            cfg = load_character(cid, DEFAULT_CHARACTERS_DIR)
            self.assertEqual(cfg.id, cid)
            self.assertEqual(cfg.rights_tier, "public_domain")
            self.assertGreaterEqual(len(cfg.system_prompt), 80)
            self.assertGreaterEqual(len(cfg.worldview_axes), 4)
            self.assertGreaterEqual(len(cfg.sources), 3)
            for src in cfg.sources:
                self.assertTrue(src.url.startswith("https://"), f"Non-HTTPS URL: {src.url}")
                self.assertTrue(src.rights_note)
                if cid == "gandhi" and src.edition_year is not None:
                    self.assertLess(
                        src.edition_year,
                        1931,
                        f"Gandhi source {src.id} edition_year must be pre-1931 US PD",
                    )

    def test_plug_and_play_toml_character_loading(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            toml_path = Path(tmp) / "seneca.toml"
            toml_path.write_text(
                'id = "seneca"\n'
                'name = "Lucius Annaeus Seneca"\n'
                'era = "4 BCE–65 CE"\n'
                'domain = "Stoic philosophy"\n'
                'rights_tier = "public_domain"\n'
                'system_prompt = "You are Seneca, Roman Stoic statesman and philosopher."\n'
                '[worldview_axes]\n'
                'virtue = "Virtue alone is the highest good."\n'
                '[[sources]]\n'
                'id = "moral_letters"\n'
                'title = "Moral Letters to Lucilius"\n'
                'provider = "project_gutenberg"\n'
                'url = "https://www.gutenberg.org/cache/epub/56075/pg56075.txt"\n'
                'rights_note = "Public domain"\n',
                encoding="utf-8",
            )
            self.assertEqual(list_character_ids(tmp), ["seneca"])
            cfg = load_character("seneca", tmp)
            self.assertEqual(cfg.name, "Lucius Annaeus Seneca")
            self.assertEqual(len(cfg.sources), 1)


class TestCorpusCleaningAndFetching(unittest.TestCase):
    def test_gutenberg_header_footer_stripping(self) -> None:
        raw = _make_synthetic_treatise(num_chapters=2)
        stripped, flags = strip_gutenberg_boilerplate(raw)
        self.assertTrue(flags["stripped_pg_header"])
        self.assertTrue(flags["stripped_pg_footer"])
        self.assertNotIn("START OF THE PROJECT GUTENBERG", stripped)
        self.assertNotIn("END OF THE PROJECT GUTENBERG", stripped)
        self.assertNotIn("Produced by Synthetic Volunteers", stripped)
        self.assertNotIn("Updated editions will replace", stripped)
        self.assertIn("CHAPTER 1.", stripped)

    def test_internet_archive_ocr_cleaning(self) -> None:
        raw = _make_synthetic_dialogue(num_chapters=2)
        cleaned, stats = clean_corpus_text(raw, provider="internet_archive")
        self.assertNotIn("Digitized by the Internet Archive", cleaned)
        self.assertNotIn("CORNELL UNIVERSITY LIBRARY", cleaned)
        self.assertNotIn("\f", cleaned)
        self.assertNotIn("[Page 11]", cleaned)
        self.assertNotIn("- 10 -", cleaned)
        # Verify hyphenated word "con-\nscience" was cleanly rejoined
        self.assertIn("conscience", cleaned)
        self.assertNotIn("con-\nscience", cleaned)
        self.assertGreaterEqual(stats["rejoined_hyphenations"], 2)
        self.assertGreaterEqual(stats["removed_page_numbers"], 4)

    def test_fetch_corpus_caching_and_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            cfg = load_character("machiavelli", DEFAULT_CHARACTERS_DIR)
            call_count = 0

            def fake_downloader(urls: list[str], timeout: int = 45, retries: int = 3) -> tuple[str, str, str]:
                nonlocal call_count
                call_count += 1
                return _make_synthetic_treatise(num_chapters=3, seed_tag=f"src{call_count}"), urls[0], f"sha_{call_count}"

            manifest1 = fetch_character_corpus(cfg, raw_dir=tmp_path, downloader=fake_downloader)
            self.assertEqual(call_count, len(cfg.sources))
            self.assertEqual(manifest1["source_count"], len(cfg.sources))
            self.assertTrue((tmp_path / "machiavelli" / "manifest.json").is_file())
            self.assertTrue(all(s["status"] == "downloaded" for s in manifest1["sources"]))

            # Second call without --force must skip all downloads
            manifest2 = fetch_character_corpus(cfg, raw_dir=tmp_path, force=False, downloader=fake_downloader)
            self.assertEqual(call_count, len(cfg.sources))
            self.assertTrue(all(s["status"] == "cached" for s in manifest2["sources"]))

            # Third call with --force must re-download
            manifest3 = fetch_character_corpus(cfg, raw_dir=tmp_path, force=True, downloader=fake_downloader)
            self.assertEqual(call_count, len(cfg.sources) * 2)
            self.assertTrue(all(s["status"] == "downloaded" for s in manifest3["sources"]))


class TestDatasetPipeline(unittest.TestCase):
    def test_deterministic_90_10_split_and_zero_leakage(self) -> None:
        cleaned, _ = clean_corpus_text(_make_synthetic_treatise(num_chapters=10), provider="project_gutenberg")
        sections = segment_document(cleaned, source_id="the_prince", source_title="The Prince")
        self.assertEqual(len(sections), 10)
        self.assertIn("OF PRINCIPALITIES AND NEW STATES", sections[0].heading)

        train1, val1 = split_sections_train_val(sections, val_ratio=0.10, seed=42)
        train2, val2 = split_sections_train_val(sections, val_ratio=0.10, seed=42)

        # Deterministic
        self.assertEqual([s.section_id for s in train1], [s.section_id for s in train2])
        self.assertEqual([s.section_id for s in val1], [s.section_id for s in val2])

        # Exact 90/10 on 10 sections
        self.assertEqual(len(train1), 9)
        self.assertEqual(len(val1), 1)
        self.assertTrue({s.section_id for s in train1}.isdisjoint({s.section_id for s in val1}))

    def test_cpt_chunking_bounds_and_coverage(self) -> None:
        cleaned, _ = clean_corpus_text(_make_synthetic_treatise(num_chapters=10), provider="project_gutenberg")
        sections = segment_document(cleaned, source_id="the_prince", source_title="The Prince")
        train_secs, _ = split_sections_train_val(sections, val_ratio=0.10, seed=42)

        chunks = build_cpt_chunks(
            train_secs, target_tokens=300, min_tokens=100, max_tokens=450, overlap_tokens=60
        )
        self.assertGreater(len(chunks), len(train_secs))
        for ch in chunks:
            self.assertIn("text", ch)
            self.assertIn("source", ch)
            self.assertLessEqual(ch["approx_tokens"], 460)
            self.assertGreater(ch["word_count"], 20)

        # Every training section marker must appear in CPT chunks (full raw-corpus coverage)
        combined_cpt = "\n".join(c["text"] for c in chunks)
        for sec in train_secs:
            self.assertIn(sec.text[:80], combined_cpt)

    def test_end_to_end_pipeline_treatise_and_dialogue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw_dir = tmp_path / "raw"
            out_dir = tmp_path / "processed"

            # Prepare synthetic cleaned raw files for Machiavelli and Gandhi
            mach = load_character("machiavelli", DEFAULT_CHARACTERS_DIR)
            (raw_dir / "machiavelli").mkdir(parents=True)
            for idx, src in enumerate(mach.sources, start=1):
                cleaned, _ = clean_corpus_text(
                    _make_synthetic_treatise(num_chapters=10, seed_tag=f"mach_s{idx}"),
                    provider=src.provider,
                )
                (raw_dir / "machiavelli" / f"{src.id}.txt").write_text(cleaned, encoding="utf-8")

            gandhi = load_character("gandhi", DEFAULT_CHARACTERS_DIR)
            (raw_dir / "gandhi").mkdir(parents=True)
            for idx, src in enumerate(gandhi.sources, start=1):
                raw_sample = (
                    _make_synthetic_dialogue(num_chapters=10)
                    if src.id == "indian_home_rule"
                    else _make_synthetic_treatise(num_chapters=10, seed_tag=f"gandhi_s{idx}")
                )
                cleaned, _ = clean_corpus_text(raw_sample, provider=src.provider)
                (raw_dir / "gandhi" / f"{src.id}.txt").write_text(cleaned, encoding="utf-8")

            # Build datasets for both characters
            mach_stats = build_character_datasets(
                mach, raw_dir=raw_dir, out_dir=out_dir, val_ratio=0.10, seed=42
            )
            gandhi_stats = build_character_datasets(
                gandhi, raw_dir=raw_dir, out_dir=out_dir, val_ratio=0.10, seed=42
            )

            # Verify all expected output files exist
            for cid in ("machiavelli", "gandhi"):
                for fname in (
                    "val_raw.jsonl",
                    "cpt_train.jsonl",
                    "sft_train.jsonl",
                    "sft_val.jsonl",
                    "dpo_prompts.jsonl",
                    "stats.json",
                ):
                    self.assertTrue((out_dir / cid / fname).is_file(), f"Missing {cid}/{fname}")

            # Verify ZERO leakage between val_raw and cpt_train / sft_train
            for cid in ("machiavelli", "gandhi"):
                val_raw = [
                    json.loads(line)
                    for line in (out_dir / cid / "val_raw.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                cpt_train = [
                    json.loads(line)
                    for line in (out_dir / cid / "cpt_train.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                sft_train = [
                    json.loads(line)
                    for line in (out_dir / cid / "sft_train.jsonl").read_text(encoding="utf-8").splitlines()
                ]
                sft_val = [
                    json.loads(line)
                    for line in (out_dir / cid / "sft_val.jsonl").read_text(encoding="utf-8").splitlines()
                ]

                val_ids = {r["section_id"] for r in val_raw}
                cpt_ids = {r["section_id"] for r in cpt_train}
                sft_train_ids = {r["section_id"] for r in sft_train}
                sft_val_ids = {r["section_id"] for r in sft_val}

                self.assertTrue(val_ids.isdisjoint(cpt_ids), f"CPT leakage for {cid}")
                self.assertTrue(val_ids.isdisjoint(sft_train_ids), f"SFT train leakage for {cid}")
                self.assertTrue(sft_val_ids.issubset(val_ids), f"SFT val not subset of val_raw for {cid}")
                self.assertTrue(sft_train_ids.isdisjoint(sft_val_ids), f"SFT train/val overlap for {cid}")

                # Check SFT message schema & authentic primary-source prose
                self.assertGreater(len(sft_train), 0)
                self.assertGreater(len(sft_val), 0)
                for row in sft_train + sft_val:
                    msgs = row["messages"]
                    self.assertEqual([m["role"] for m in msgs], ["system", "user", "assistant"])
                    self.assertTrue(all(isinstance(m["content"], str) and m["content"].strip() for m in msgs))

            # Verify Gandhi's Hind Swaraj dialogue extraction produced primary_dialogue turns
            self.assertGreater(gandhi_stats["datasets"]["sft_train"]["dialogue_turns"], 0)
            self.assertEqual(mach_stats["datasets"]["dpo_prompts"]["count"], 60)
            self.assertEqual(gandhi_stats["datasets"]["dpo_prompts"]["count"], 60)


class TestAntiRLHFDPO(unittest.TestCase):
    def test_60_dilemmas_zero_game_leakage_and_authentic_chosen(self) -> None:
        self.assertEqual(len(DILEMMAS), 60)
        ids = [d["id"] for d in DILEMMAS]
        self.assertEqual(len(set(ids)), 60, "Dilemma IDs must be unique")

        categories = {d["category"] for d in DILEMMAS}
        self.assertGreaterEqual(len(categories), 10)

        # Must pass strict zero-leakage validator
        assert_no_game_leakage(DILEMMAS)

        mach = load_character("machiavelli", DEFAULT_CHARACTERS_DIR)
        gandhi = load_character("gandhi", DEFAULT_CHARACTERS_DIR)
        mach_dpo = build_dpo_records(mach, DEFAULT_CHARACTERS_DIR)
        gandhi_dpo = build_dpo_records(gandhi, DEFAULT_CHARACTERS_DIR)

        self.assertEqual(len(mach_dpo), 60)
        self.assertEqual(len(gandhi_dpo), 60)

        for m_row, g_row in zip(mach_dpo, gandhi_dpo):
            self.assertEqual(m_row["id"], g_row["id"])
            self.assertEqual(m_row["prompt"], g_row["prompt"])
            self.assertIsNotNone(m_row["chosen"])
            self.assertIsNotNone(g_row["chosen"])
            self.assertGreaterEqual(len(m_row["chosen"].split()), 25)
            self.assertGreaterEqual(len(g_row["chosen"].split()), 25)
            self.assertNotEqual(m_row["chosen"], g_row["chosen"])
            # Rejected must be None until harvested live from base Qwen3-8B on Colab
            self.assertIsNone(m_row["rejected"])
            self.assertIsNone(g_row["rejected"])

    def test_leakage_validator_catches_split_or_steal(self) -> None:
        bad_dilemma = (
            {
                "id": "bad_01",
                "prompt": "You are playing Split or Steal for $1,000 in Round 3. What do you do?",
                "chosen": {"machiavelli": "I choose steal."},
            },
        )
        with self.assertRaises(ValueError):
            assert_no_game_leakage(bad_dilemma)

    def test_custom_character_dpo_override_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "custom.json").write_text(
                json.dumps(
                    {
                        "id": "custom",
                        "name": "Custom Thinker",
                        "era": "Modern",
                        "domain": "Philosophy",
                        "rights_tier": "user_authored",
                        "system_prompt": "You are Custom Thinker.",
                        "worldview_axes": {"core": "Radical candor."},
                        "sources": [
                            {
                                "id": "essays",
                                "title": "Essays",
                                "provider": "generic",
                                "url": "https://example.com/essays.txt",
                                "rights_note": "Original",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (tmp_path / "custom_dpo.json").write_text(
                json.dumps({"dilemma_01": "Custom character bespoke response for dilemma 01."}),
                encoding="utf-8",
            )
            cfg = load_character("custom", tmp_path)
            records = build_dpo_records(cfg, tmp_path)
            self.assertEqual(len(records), 60)
            self.assertEqual(records[0]["chosen"], "Custom character bespoke response for dilemma 01.")
            self.assertIsNone(records[1]["chosen"])


if __name__ == "__main__":
    unittest.main()
