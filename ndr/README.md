# NDR: цикл экспериментов

Каждая версия решения — отдельная самодостаточная папка в `ndr/solutions/`. Текущее решение
называется `baseline`. Для нового эксперимента скопируйте всю папку и меняйте копию:

```powershell
Copy-Item -Recurse ndr/solutions/baseline ndr/solutions/deepseek-v2
python ndr/run/run.py --list-solutions
```

Запуск выбранной версии из корня репозитория:

```powershell
Copy-Item .env.example .env
# Заполните OPENROUTER_API_KEY и при необходимости OPENROUTER_MODEL.
python ndr/run/build_dataset.py --check
python ndr/run/run.py `
  --solution deepseek-v2 `
  --concurrency 4
```

`--solution` — имя непосредственной дочерней папки `ndr/solutions/`; по умолчанию запускается
`baseline`. Настройки конкретной версии находятся в её `config.py`, а разовые CLI overrides
не изменяют solution.

## Результаты без перезаписи

Каждый вызов создаёт новый каталог по имени solution и точному UTC timestamp:

```text
ndr/results/
└── deepseek-v2/
    └── 20260919T172530.123456Z/
        ├── run.json
        ├── metrics.json
        ├── by_case/
        │   ├── q-000001.json
        │   └── ...
        └── solution/
            ├── config.py
            ├── models.py
            ├── predictor.py
            └── prompts/
```

Старые результаты никогда не перезаписываются, поэтому `--force` больше не нужен. Runner
перед запуском копирует выбранную папку в `solution/` внутри эксперимента и исполняет именно
этот snapshot. В `run.json` сохраняются имя solution, timestamp, SHA-256 каждого файла и
общий fingerprint версии. В `metrics.json` имя и fingerprint продублированы для агрегации.

Для короткой проверки инфраструктуры без API-вызовов:

```powershell
python ndr/run/run.py `
  --solution baseline `
  --predictor ndr/run/example_predictor.py `
  --limit 3 `
  --concurrency 3
```

`--results-dir PATH` меняет только корень дерева результатов; структура
`<solution>/<timestamp>/` сохраняется.

## Что сохраняется

`run.json` атомарно обновляется после каждого завершённого кейса, поэтому уже полученные
ответы остаются при позднем сбое. `metrics.json` создаётся после завершения прогона.
`by_case/` содержит полную отдельную запись каждого query, включая:

- вход, порядок и карточки кандидатов;
- фактически использованные prompts и runtime;
- gold, выбранный slug, статус, ошибку и latency;
- сырой JSON каждого model call и все choices/generations;
- возвращённые `content`, `reasoning`, `reasoning_details`, refusal, annotations и usage.

Ошибки predictor/контракта не превращаются в `not_found` и исключаются из знаменателя
accuracy. `not_found` является полноценным ответом и входит в знаменатель accuracy.

## Основные флаги

```text
--solution NAME             имя папки в ndr/solutions (default: baseline)
--list-solutions            показать доступные solutions
--model ID                  модель; альтернатива — OPENROUTER_MODEL
--concurrency N             число параллельных кейсов
--generations N             значение n в Chat Completions
--reasoning-effort VALUE    none|minimal|low|medium|high|xhigh|max или 1..100
--max-completion-tokens N   лимит output/reasoning tokens
--timeout SECONDS           timeout одного HTTP-запроса
--limit N                   прогнать первые N кейсов
--seed N                    детерминированный порядок кандидатов
--results-dir PATH          корень versioned results, default ndr/results
--env-file PATH             dotenv-файл, default корневой .env
```

Подробности dataset и контракта predictor-а — в `run/README.md`; описание текущего pipeline —
в `solutions/baseline/README.md`.
