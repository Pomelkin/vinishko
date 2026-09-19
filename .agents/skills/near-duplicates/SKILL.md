---
name: near-duplicates
description: Maintain and audit this repository's verified wine near-duplicate CSV and image gallery. Use for near-duplicates, visually close catalog items, hard negatives, alias/redesign classification, or files and scripts under data/near_duplicates.
---

# Near-duplicates

Apply the object-level definition: a near-duplicate is a pair of **different wines or
vintages** whose packaging is identical or visually close. Similar files are only candidates.

Do not count any of these as near-duplicates:

- two renders or two slugs of the same wine/object;
- the same wine before and after a packaging redesign;
- two catalog rows incorrectly sharing an image;
- visually distinct products grouped only by text.

## Authoritative project state

- Allowed slugs come only from `data/strapi/catalog_dataset.csv`, column `Slug`, with exact
  string equality. The file currently contains 2103 unique slugs.
- `data/near_duplicates/all_candidates.csv` is the aggregate registry despite its legacy
  filename. It contains only 378 reviewed rows with
  `review_verdict=confirmed_near_duplicate` and `is_confirmed_near_duplicate=true`.
- `data/near_duplicates/C-visually-close/` contains exactly the same 378 pairs, one folder per
  CSV row. `candidate_number` matches the three-digit folder prefix. Gaps are intentional.
- Each confirmed folder contains `info.md`, two numbered bottle PNGs, `_compare.png`, and
  `_zoom.png`.
- `data/near_duplicates/index.md` must not exist. Keep the aggregate table in CSV, not Markdown.
- Candidate numbers `1, 8, 72, 82, 239, 251, 315, 347, 387` are reviewed exclusions and must
  not appear in the CSV or C-gallery.
- `A-one-image-many-slugs/` and `B-same-series/` are legacy diagnostic galleries, not the
  authoritative near-duplicate registry. Never include their directory counts in the 378.

The broader research in `docs/NEAR_DUPLICATES.md` explains semantic families and aliases.
When prose or legacy script output conflicts with the persisted contract above, preserve the
contract and investigate the mismatch before changing data.

## Required workflow

For reports and downstream datasets, read `all_candidates.csv` with a real CSV parser. Do not
reconstruct the registry from folder names or Markdown.

For a rebuild, do not run `scripts/collect_near_duplicates.py --force` against
`data/near_duplicates/`. The legacy generator reads the older catalog path, defaults to only
150 C-pairs, recreates `index.md`, and includes nine reviewed exclusions. Instead:

1. Generate all 387 visual candidates into a new temporary directory with
   `--max-visual-groups 0`.
2. Re-evaluate/filter pairs at object level and require both exact slugs to exist in
   `catalog_dataset.csv`.
3. Synchronize only the 378 confirmed candidate numbers into the working CSV and C-gallery.
4. Remove the temporary build after synchronizing the confirmed artifacts.

If the confirmed set intentionally changes, update the CSV and image folders atomically. Do
not leave a CSV row without its folder or a folder without its CSV row. Preserve the original
candidate numbers unless the user explicitly requests renumbering.

Pixel metrics (`diff_percent`, RMSE, `max_diff`) describe rendered files, not product identity.
Use label text, producer, vintage, grape, category/color, sugar, and packaging identity for the
semantic verdict.
