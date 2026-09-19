# NDR: запуск

Из корня репозитория в PowerShell:

```powershell
Copy-Item .env.example .env
# Заполните OPENROUTER_API_KEY и OPENROUTER_MODEL в .env
python ndr/run/build_dataset.py --check
python ndr/run/run.py `
  --concurrency 4 `
  --force
```

Результаты появятся здесь:

```text
ndr/results/run.json
ndr/results/metrics.json
ndr/results/by_case/q-000001.json
ndr/results/by_case/q-000002.json
...
```

Для короткой проверки без API-вызовов:

```powershell
python ndr/run/run.py `
  --predictor ndr/run/example_predictor.py `
  --limit 3 `
  --concurrency 3 `
  --force
```

## Что запускается

По умолчанию runner использует:

- решение `ndr/solution/predictor.py`;
- отдельные промты сравнения с обязательным/необязательным годом;
- resolver-промт для случая, когда первый этап принял несколько кандидатов;
- 51 query из 30 многопозиционных near-duplicate-групп;
- OpenRouter-compatible endpoint `https://openrouter.ai/api/v1`.

`--concurrency N` задаёт максимальное число одновременно обрабатываемых кейсов. Порядок
кандидатов детерминированно перемешивается для каждого query, а порядок кейсов в итоговом
`run.json` всегда совпадает с manifest независимо от порядка завершения потоков.

## Артефакты

`run.json` содержит параметры прогона, краткие метрики и полную запись каждого кейса.
Итоговые агрегаты качества, latency, model usage и разбивка по размеру группы записываются
в `metrics.json` только после завершения прогона.
В `by_case/` лежит та же полная запись, но по одному JSON на query. Запись включает:

- входное query, порядок и полные карточки кандидатов;
- все три промта и все три structured-output schema;
- gold и выбранный slug;
- статус, ошибку и latency;
- полный сырой JSON-ответ каждого candidate-comparison и resolver-вызова;
- каждую generation/choice каждого этапа отдельно;
- все возвращённые провайдером `content`, `reasoning`, `reasoning_details`, refusal и annotations;
- usage и остальные provider-specific поля внутри `raw_response`.

Runner не может сохранить скрытое chain-of-thought, если модель или провайдер его не
возвращает. Все фактически возвращённые reasoning-поля сохраняются без фильтрации.

Файлы перезаписываются только с `--force`. Во время прогона `run.json` атомарно обновляется
после каждого завершённого кейса, поэтому уже полученные ответы не теряются при позднем сбое.

## Основные флаги

```text
--model ID                 модель OpenRouter; альтернатива — OPENROUTER_MODEL
--concurrency N            число параллельных кейсов, по умолчанию 1
--generations N            значение n в Chat Completions, по умолчанию 1
--reasoning-effort LEVEL   none|minimal|low|medium|high
--max-tokens N             лимит output/reasoning tokens
--timeout SECONDS          timeout одного HTTP-запроса
--limit N                  прогнать первые N кейсов
--seed N                   детерминированный порядок кандидатов
--results-dir PATH         каталог результатов, по умолчанию ndr/results
--env-file PATH            dotenv-файл, по умолчанию корневой .env
--force                    заменить run.json, metrics.json и by_case/*.json
```

Для другого OpenAI-compatible сервера задайте `--api-base`. Имя переменной с ключом можно
изменить через `--api-key-env`; её значение никогда не записывается в результаты.

Подробности датасета — в `run/README.md`; описание pipeline — в `solution/README.md`.
