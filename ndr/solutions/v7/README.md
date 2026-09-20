# NDR solution v7: image-only one-shot selection

`v7` is an immutable experiment derived from `v5`. It keeps the same one-call, complete-group
selection pipeline, model configuration, candidate order, sampling settings, image detail, and
strict structured-output validation.

## Primary hypothesis

Catalog attributes and descriptive slugs can condition the model's OCR and final choice. Removing
that text may make the decision depend on evidence actually visible in the query and reference
images.

## Experimental change

The model receives:

1. the QUERY image;
2. each catalog reference image, labeled only `element_1`, `element_2`, and so on;
3. the strict response schema with those opaque element identifiers and `not_found`.

The model does not receive `CARD_JSON`, catalog attributes, catalog vintage policies, or slugs.
After schema validation, Python maps the selected opaque identifier back to the corresponding slug.
The full mapping remains in the local trace for auditability, but is not included in the model
request.

`v7` has one prompt file, `prompts/select_nearest.txt`, registered as `select_nearest` in
`config.py`. It does not carry the legacy three-prompt layout because this pipeline makes one
joint selection call.

## Baseline for comparison

The complete `v5` run at `ndr/results/v5/20260919T214443.536610Z` scored 49/51 end-to-end
(96.08%), with zero contract or predictor errors. Its two wrong answers were semantic decisions:

- `q-000005`: false negative (`not_found`) after reading the dry Semillon query as semi-sweet;
- `q-000047`: false positive for the dry Millstream candidate instead of the semi-dry candidate.

Compare `v7` with that run only when model, dataset, seed, provider route, sampling, and concurrency
are kept compatible.
