# AI-сомелье

## Запуск и оценка

Все команды выполняются из корня репозитория.

### 1. Запустить прогон сомелье

```powershell
python -m ai_somelier.run --experiment v1-baseline
```

Runner сохранит результат в новой папке `ai_somelier/results/<run-id>`.

### 2. Запустить Codex и вставить готовый промпт

```powershell
codex
```

Скопировать в открывшийся Codex весь следующий текст:

```text
Используй $ai-somelier-eval. Найди самый свежий прогон сомелье с run.json в ai_somelier/results и полностью оцени его без дополнительных вопросов ко мне. Сам создай рядом evaluation.json через eval scaffold, затем оцени каждый фактический ответ по всем must_include и глобальным tone-of-voice критериям, извлеки все подтверждённые wrong_claims, заполни evidence и установи status=complete. После этого сам запусти eval metrics и создай рядом metrics.json. Не запускай сомелье или provider повторно и не изменяй run.json, case-файлы, golden_dataset.json либо criteria.json. В конце проверь оба JSON и сообщи их пути и краткую итоговую сводку метрик.
```

Codex сам выполняет создание чеклиста, смысловую оценку и подсчёт метрик. Вручную запускать
команды из `ai_somelier/eval/` или редактировать `evaluation.json` не требуется.

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
