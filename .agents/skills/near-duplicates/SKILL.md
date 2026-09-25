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
  filename. It contains 2097 reviewed rows with
  `review_verdict=confirmed_near_duplicate` and `is_confirmed_near_duplicate=true`.
- `data/near_duplicates/C-visually-close/` contains exactly the same 2097 pairs, one folder per
  CSV row. `candidate_number` matches the folder's numeric prefix (three or more digits).
  Gaps in the legacy number range are intentional.
- Each confirmed folder contains `info.md`, two numbered bottle PNGs, `_compare.png`, and
  `_zoom.png`.
- `data/near_duplicates/index.md` must not exist. Keep the aggregate table in CSV, not Markdown.
- Candidate numbers `1, 8, 72, 82, 239, 251, 315, 347, 387` are earlier reviewed exclusions.
  Numbers `71, 204, 209, 214` were excluded after comparing old gallery images with current
  references. Numbers `148, 211, 233, 258, 287, 291, 296, 317` were visually audited but
  still lack a correct reference for one product. None may appear in the confirmed CSV or
  C-gallery; the latter eight remain in `review_required.csv` and `review_required_gallery/`.
- The old 378-pair registry and the A/B diagnostic galleries are preserved under
  `legacy/data/near_duplicates/`. They are not the current confirmed registry.
- `review_queue.csv` is the original 1570-pair candidate input; those pairs and the wider
  0.75-threshold, visual-sweep, cross-winery, and component-closure inputs have all been
  reviewed. `reviewed_candidates.csv` contains all 3651 post-review verdicts with evidence.
  There are 43 `needs_review` identity ambiguities and 11 `needs_correct_reference` cases;
  neither class is included in the confirmed registry.
- `data/strapi/image_mismatch_reviews.csv` records two accepted-by-name but wrong-product
  images now excluded by source SHA-256. The Catalog has 2055 accepted reference positions.

The broader research in `docs/NEAR_DUPLICATES.md` explains semantic families and aliases.
When prose or legacy script output conflicts with the persisted contract above, preserve the
contract and investigate the mismatch before changing data.

## Required workflow

For reports and downstream datasets, read `all_candidates.csv` with a real CSV parser. Do not
reconstruct the registry from folder names or Markdown.

For a rebuild, do not run `scripts/collect_near_duplicates.py --force` against
`data/near_duplicates/`. It defaults to only 150 C-pairs, recreates `index.md`, and
includes unreviewed pairs. Instead:

1. Run `semantic_near_duplicates.py --output` and
   `export_near_duplicate_review_queue.py` to collect new candidates. The semantic audit
   uses only collision-reviewed references from `catalog_dataset.csv`.
2. If visual galleries are needed, generate candidates into a new temporary directory with
   `--max-visual-groups 0`; do not assume the old 387-candidate count applies to the full dump.
3. Re-evaluate pairs at object level and require both exact slugs and accepted image statuses
   in `catalog_dataset.csv`.
4. Save a verdict and image-based evidence for every candidate. Run
   `scripts/validate_near_duplicate_review_decisions.py` and
   `scripts/generate_near_duplicate_closure_queue.py --check` before publishing.
5. Publish only manually confirmed pairs with
   `python scripts/publish_near_duplicate_reviews.py --apply` after inspecting its plan.
   It stages the CSV and C-gallery together and retains the former version as a backup.
6. Run `.agents/skills/near-duplicates/scripts/validate.py` and
   `scripts/audit_near_duplicate_gallery_images.py`, then
   `scripts/report_near_duplicate_reviews.py --json`; remove any temporary build only
   after validation.

If the confirmed set intentionally changes, update the CSV and image folders atomically. Do
not leave a CSV row without its folder or a folder without its CSV row. Preserve the original
candidate numbers unless the user explicitly requests renumbering.

Pixel metrics (`diff_percent`, RMSE, `max_diff`) describe rendered files, not product identity.
Use label text, producer, vintage, grape, category/color, sugar, and packaging identity for the
semantic verdict.
