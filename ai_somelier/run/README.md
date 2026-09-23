# Golden dataset runner

Runner генерирует диалоги для всех кейсов `ai_somelier/golden_dataset.json`. Сам runner оценок
не выставляет: эталонные поля сохраняются только внутри снимка исходного кейса. После прогона
структурированную оценку создаёт пакет `ai_somelier/eval/` по project skill
`ai-somelier-eval`.

Из корня репозитория:

```powershell
python -m ai_somelier.run --experiment v1-baseline
```

Можно вызвать и сам файл: `python ai_somelier/run/run.py --experiment v1-baseline`.

Явный эквивалент с параметрами по умолчанию:

```powershell
python -m ai_somelier.run `
  --experiment v1-baseline `
  --solution v1 `
  --dataset ai_somelier/golden_dataset.json `
  --results-dir ai_somelier/results `
  --concurrency 10 `
  --env-file .env
```

Независимые кейсы выполняются параллельно. Первое сообщение и пользовательские вопросы
внутри одного кейса всегда выполняются последовательно в одной истории. Повторных запросов
после ошибки нет.

Каждый запуск создаёт `ai_somelier/results/<experiment>_<UTC timestamp>/`:

```text
run.json
cases/
  0000_<case-id>.json
  0001_<case-id>.json
  ...
```

`run.json` хранит конфигурацию, хеш датасета и файлов solution/runner, окружение без секретов,
тайминги и статусы. JSON кейса хранит исходный кейс целиком, найденное вино, кандидатов,
вопросы по порядку, каждый запрос и полный ответ solution, provider payload/response,
`reasoning`, `reasoning_details`, ошибки и итоговую историю сессии.
