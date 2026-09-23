# AI-сомелье

## Быстрый запуск и оценка

Все команды выполняются из корня репозитория.

Запустить прогон golden dataset:

```powershell
python -m ai_somelier.run --experiment v1-baseline
```

Runner напечатает путь вида `ai_somelier/results/<run-id>`. Создать для него чеклист оценки:

```powershell
python -m ai_somelier.eval scaffold `
  --run ai_somelier/results/<run-id> `
  --output ai_somelier/results/<run-id>/evaluation.json
```

Попросить Codex оценить этот прогон с помощью `$ai-somelier-eval`. После заполнения
`evaluation.json` посчитать итоговые метрики:

```powershell
python -m ai_somelier.eval metrics `
  ai_somelier/results/<run-id>/evaluation.json `
  --output ai_somelier/results/<run-id>/metrics.json
```

Для модельного прогона в корневом `.env` должен быть задан `OPENROUTER_API_KEY`; шаблон —
`.env.example`. Eval-пакет работает локально и сам модель не вызывает.

## Что находится в пакете

- `solution/v1/` — текущая реализация цифрового сомелье и промпты;
- `golden_dataset.json` — 10 сценариев, эталонные ответы и атомарные `must_include`;
- `run/` — lossless runner, сохраняющий полные ответы и трассировку;
- `eval/` — JSON-чеклист, проверка provenance и расчёт метрик;
- `kb/EXPERT_KNOWLEDGE.json` — экспертная база;
- `results/` — неизменяемые артефакты прогонов, оценки и метрики;
- `docs/` — требования и контракты экспертных знаний.

Подробные параметры runner описаны в `ai_somelier/run/README.md`, формат оценки — в
`ai_somelier/eval/README.md`.
