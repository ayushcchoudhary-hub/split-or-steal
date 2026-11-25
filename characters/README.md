# Character Registry (`characters/`)

Every character in **Split or Steal** is defined by a single plug-and-play JSON (or TOML) config file at `characters/<id>.json`.

## Adding a New Character

1. Drop a new file `characters/<id>.json` following the schema below.
2. (Optional) Add bespoke `chosen` responses for `<id>` in `data/dpo_dilemmas.py` (or `characters/<id>_dpo.json`) so DPO training uses authentic character-grounded preferences.
3. Run the Layer 1 pipeline:
   ```bash
   python data/fetch_corpus.py --character <id>
   python data/build_datasets.py --character <id>
   ```

## Required Schema

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `string` | Lowercase slug matching the filename (`<id>.json`) |
| `name` | `string` | Display name of the character |
| `era` | `string` | Historical era / dates |
| `domain` | `string` | Intellectual or strategic domain |
| `rights_tier` | `string` | Provenance classification (e.g., `public_domain`, `cc_by`, `user_authored`) |
| `system_prompt` | `string` | Concise baseline prompt used for the Prompt-Only control and SFT/DPO system turn |
| `worldview_axes` | `object` | Key strategic/moral priors (`axis_name -> description`) |
| `sources` | `array` | List of verifiable plain-text corpus sources (`id`, `title`, `original_year`, `edition_year`, `translator`, `provider`, `catalog_id`, `url`, `fallback_urls`, `rights_note`) |
