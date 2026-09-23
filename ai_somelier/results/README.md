# AI sommelier run results

Каждый generation-only прогон создаёт здесь папку `<solution>_<UTC timestamp>` с общим
`run.json` и полными JSON-логами отдельных диалогов в `cases/`.

Codex-оценка того же прогона сохраняется рядом как `evaluation.json`, а рассчитанные
`python -m ai_somelier.eval metrics ...` показатели — как `metrics.json`. Процедура и
контракт описаны в `ai_somelier/eval/README.md` и project skill `ai-somelier-eval`.
