---
name: ndr-experiments
description: Prepare reproducible improvement experiments for the NDR solutions under ndr/solutions, analyze user-produced lossless run artifacts, and compare versions without retrying or hiding model failures. Use for NDR/NRP baselines, new solution versions, DeepSeek prompt or pipeline changes, and reports from ndr/results.
---

# NDR experiments

Treat each solution directory as an immutable experiment version. The authoritative inputs are
`ndr/dataset/`; the runner is `ndr/run/run.py`; source versions live directly under
`ndr/solutions/`; timestamped, lossless outputs belong under `ndr/results/`.

## User-owned execution

Only the user may start an NDR experiment run. Never execute `ndr/run/run.py` with a solution,
call the model/provider directly, or start, resume, retry, or rerun any model-backed execution.
This prohibition applies even when credentials are available and even when the task asks to conduct
an experiment. The agent may prepare source versions and run offline validation, unit tests, and
fully mocked/no-network tests that cannot reach a model. If it is uncertain whether a command can
make a model request, it must not run the command.

After preflight, give the user the exact runner command and stop before execution. Analyze and
report only after the user has produced the timestamped artifacts under `ndr/results/`.

## Iteration contract

1. Inspect the previous version's `metrics.json`, `run.json`, and relevant `by_case/*.json`.
   Classify failures as predictor/HTTP, response-contract, semantic false negative, semantic
   false positive, resolver error, or dataset/gold inconsistency. Do not infer a prompt defect
   from the aggregate metric alone.
2. State one primary hypothesis. Unless the user requests a combined change, copy the complete
   previous solution to a new direct child of `ndr/solutions/` and change only what tests that
   hypothesis. Never edit an already-run version to represent a later experiment.
3. Keep the model selected by `OPENROUTER_MODEL` in the root `.env` unless the user explicitly
   asks for another model. Never print or copy the API key into code, prompts, traces, or reports.
4. Before handing the experiment to the user, run
   `python .agents/skills/near-duplicates/scripts/validate.py`,
   `python ndr/run/build_dataset.py --check`, the runner tests, and relevant solution unit tests.
   All tests must be offline and incapable of reaching a model; do not invoke the runner for them.
5. Instruct the user to run the new solution exactly once; never run it on the user's behalf. A
   failed call or malformed generation is an experiment result: do not ask the user to retry the
   request, failed case, or experiment, and do not add provider fallbacks. A preflight failure
   before the user starts any model request may be corrected, after which provide the command again.
6. Preserve the runner's timestamped result, solution fingerprint, and per-file hashes. The runner
   executes the source solution in place, so never edit that version after its run. Compare versions
   only when dataset, model, seed, sampling, provider route, and concurrency are compatible; call
   out every difference that remains.

## Structured output

Use the Pydantic model in the solution as the single schema source. Send its exact JSON Schema in
both `response_format` and the system message. The system message must make the complete object
shape, required fields, enums, `additionalProperties` rule, and nested checklist structure visible
to the model. For the resolver, include the dynamically generated slug enum for that call. Validate
the returned JSON with the same Pydantic model. Do not repair malformed JSON, extract a JSON-looking
substring, coerce values, or make a second call.

## Reporting

Errors are visible outcomes, not removable observations. Always report:

- attempted cases, correct cases, and end-to-end accuracy `correct / attempted` with errors
  counted as misses;
- the runner's quality accuracy and its excluded-error count, clearly labeled as conditional;
- contract and predictor errors, `not_found`, answered accuracy, latency p50/p95/max, request
  count, token usage, and cost when present;
- when the solution has a tie-breaker/resolver, the number of queries sent to it and its share of
  all attempted queries; count an attempted tie-breaker even when that call fails;
- per-case evidence for every error and wrong answer, including stage, `finish_reason`, validation
  error, and the decisive model observation.

If QUERY, reference image, catalog card, and folder-derived gold disagree, record a dataset/gold
inconsistency instead of weakening the matcher to reproduce the label. Do not modify the confirmed
near-duplicate registry as part of an NDR experiment.
