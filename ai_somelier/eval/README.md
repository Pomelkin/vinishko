# Structured evaluation

Пакет не вызывает модель и не оценивает ответы автоматически. Он создаёт полный JSON-чеклист
для lossless-прогона, а Codex по project skill проставляет бинарные решения и извлекает
фактические ошибочные утверждения.

Создать черновик:

```powershell
python -m ai_somelier.eval scaffold `
  --run ai_somelier/results/<run-id> `
  --output ai_somelier/results/<run-id>/evaluation.json
```

После ручной оценки изменить корневой `status` на `complete` и посчитать метрики:

```powershell
python -m ai_somelier.eval metrics `
  ai_somelier/results/<run-id>/evaluation.json `
  --output ai_somelier/results/<run-id>/metrics.json
```

Скрипт сверяет SHA-256 golden dataset и глобальных критериев, fingerprint `run.json` и всех
case-файлов, полный набор кейсов/ходов, фактические ответы и каждый criterion ID. Черновые
`null`, пропущенные задачи, пустые evidence и изменённые исходные ответы не принимаются.

Отчёт содержит долю пройденных `must_include`, число `wrong_claims`, долю задач без них,
tone-of-voice compliance и строгий pass задачи (`all must_include` + `0 wrong_claims` +
`all tone_of_voice`). Ошибка генерации остаётся задачей: все бинарные проверки получают
`false`, а `wrong_claims` остаётся пустым, поскольку извлекать утверждения не из чего.
