---
name: ai-somelier-eval
description: Evaluate AI sommelier run artifacts against per-response atomic must-include criteria, extracted wrong claims, and global binary tone-of-voice criteria. Use for grading or comparing runs under ai_somelier/results and for producing evaluation.json and metrics.json.
---

# AI sommelier evaluation

Codex is the semantic evaluator. Do not call a provider model, add keyword matching, or compare
the generated answer to the reference wording. The deterministic package only scaffolds the
complete checklist, locks provenance, validates decisions, and calculates metrics.

## Workflow

1. Create a draft from the exact lossless run:

   ```powershell
   python -m ai_somelier.eval scaffold `
     --run ai_somelier/results/<run-id> `
     --output ai_somelier/results/<run-id>/evaluation.json
   ```

2. Evaluate every task in `evaluation.json`. Read its actual response together with the matching
   case in `ai_somelier/golden_dataset.json`, the conversation through that turn, the found-wine
   card, candidate cards when the answer discusses alternatives, and injected expert blocks from
   the run trace when relevant. Never infer facts from a filename or another case.
3. Replace every checklist `passed: null` with `true` or `false` and write non-empty `evidence`.
   Change the root `status` to `complete` only after all tasks are evaluated.
4. Validate and calculate metrics:

   ```powershell
   python -m ai_somelier.eval metrics `
     ai_somelier/results/<run-id>/evaluation.json `
     --output ai_somelier/results/<run-id>/metrics.json
   ```

Do not modify the run, golden dataset, or criteria while grading. If provenance validation fails,
stop and report the mismatch rather than weakening the check.

## Must-include decisions

Judge meaning, not wording. Mark one criterion `true` only when the actual response clearly conveys
the whole criterion in the required response field; paraphrases and different sentence order are
valid. Do not award partial credit, infer an omitted idea from general context, or use the reference
answer as a substring template. A criterion contradicted elsewhere in the same response is `false`.

For a failed or missing generation, mark every must-include and tone criterion `false`, cite the
generation status/error as evidence, and leave `wrong_claims` empty because there is no claim to
extract.

## Wrong claims

Extract every distinct factual assertion in the actual answer that is contradicted by the available
case evidence. Also count a value that the answer presents as known when the relevant source says it
is unknown or empty. Use one object per atomic claim:

```json
{
  "id": "wc_01",
  "claim": "Краткая нормализованная формулировка ошибочного утверждения.",
  "evidence": "Короткая точная цитата из фактического ответа.",
  "reason": "Почему утверждение неверно или недоказанно.",
  "source": "Точный источник проверки: поле карточки, история, кандидат или expert block."
}
```

Use the Catalog-backed cards and explicit conversation state as primary evidence. General expert
knowledge injected for the found wine may support general wine guidance, but it does not establish
an unlisted property of the particular bottle or a candidate. Preserve unknowns: an empty flag is
not `нет`, and `Игристое = не определено` is not a still wine.

Do not record omissions, failed must-include criteria, style defects, debatable taste preferences,
reasonable recommendations, or merely unverified claims as wrong claims. If a claim cannot be
shown false from the available evidence, do not add it. Do not duplicate one error just because it
appears in more than one sentence.

## Tone-of-voice decisions

Apply every global criterion from `ai_somelier/eval/criteria.json` independently to every task.
Use its `pass_when` and `fail_when` anchors. Tone decisions remain separate from completeness and
factuality: a factually wrong answer can still sound natural, and a complete answer can still fail
readability. Evidence should quote the decisive wording or briefly identify the observable absence.

The strict task pass calculated by the package means all must-includes passed, zero wrong claims,
and all tone checks passed. Never replace these dimensions with an overall subjective score.
