# NDR runner: запуск

Из корня репозитория в PowerShell:

```powershell
Copy-Item .env.example .env
# Заполните OPENROUTER_API_KEY; настройки baseline — в ndr/solutions/baseline/config.py
python ndr/run/build_dataset.py --check
python ndr/run/run.py --solution baseline
```

Runner выбирает самодостаточную папку по `--solution`. Все versioned solutions лежат в
`ndr/solutions/<name>/`; список готовых к запуску папок выводит `--list-solutions`.
CLI остаётся механизмом разовых overrides; например:

```powershell
python ndr/run/run.py `
  --solution baseline `
  --model '<openrouter/model-id>' `
  --concurrency 4 `
  --results-dir ndr/results
```

Каждый вызов создаёт новый каталог `ndr/results/<solution>/<UTC timestamp>/` с `run.json`,
`metrics.json` и `by_case/`. Папка solution не копируется; в `run.json` записываются её путь,
по-файловые SHA-256 и общий fingerprint на момент запуска. Старые эксперименты не
перезаписываются.

## Dataset

Источники истины:

- запросы — архивный размеченный набор из `data/legacy/test/`: каталожные slug с подтверждёнными near-duplicates
  и сопоставленные `not_found_*`;
- каталожные пары — только `data/near_duplicates/all_candidates.csv`;
- пары для отсутствующих в Каталоге вин — только
  `data/near_duplicates/not_found_candidates.csv`;
- карточки — технический нормализованный источник `data/technical/strapi/catalog_dataset.csv`;
- группа — полная связная компонента подтверждённых пар.

Сгенерированный набор лежит в `ndr/dataset/`. Одноэлементные каталожные группы и
несопоставленные `not_found_*` не тестируются. Их файлы перечислены в
`ndr/dataset/excluded.jsonl`, но не попадают в `manifest.jsonl` и не передаются решению.
Для сопоставленного `not_found_*` в gold записывается `expected_slug="not_found"`, а в запрос
передаётся полная каталожная компонента указанного anchor-slug.

Состав snapshot:

- `manifest.jsonl` — 51 тестовый query и gold slug;
- `groups.jsonl` — 30 групп и рёбра реестра;
- `catalog.jsonl` — 132 карточки кандидатов и пути к Эталонам;
- `excluded.jsonl` — 25 исключённых случаев: 22 каталожных singleton и 3 несопоставленных
  `not_found`;
- `metadata.json` — правила, хеши источников и контрольные количества.

Пересборка и проверка:

```powershell
python ndr/run/build_dataset.py
python ndr/run/build_dataset.py --check
python -m unittest ndr/run/test_toolkit.py -v
```

## Контракт predictor-а

Файл predictor-а экспортирует потокобезопасную функцию:

```python
def predict(request: dict) -> dict:
    return {"slug": "candidate-slug"}
```

Допустимы точный slug из переданной группы или `not_found`. Встроенное решение добавляет к
внутреннему ответу `_trace`, `_status` и `_error`; модель по-прежнему генерирует строгий
публичный объект только с полем `slug`.

При `--concurrency N` функция `predict()` может быть одновременно вызвана из N потоков.
Встроенный HTTP predictor не хранит изменяемого общего состояния и поддерживает этот режим.

## Что сохраняется

Общий `run.json` и каждый JSON в `by_case/` содержат не только метрику и slug, но и полный
`predictor_response`. Для встроенного решения это:

- сырой ответ OpenRouter-compatible API без удаления provider-specific полей;
- все choices/generations;
- `content`, `reasoning`, `reasoning_details`, refusal и annotations каждой choice;
- результат JSON parsing и schema validation каждой generation;
- выбранный generation index;
- HTTP status/headers и текст тела при ошибке;
- request trace без повторения base64-данных изображений.

Base64 не дублируется в результатах: вместо него сохраняются путь, MIME по сигнатуре, размер
и SHA-256 каждого изображения. API-ключ не входит ни в request trace, ни в run metadata.

Встроенный predictor сначала отдельно сравнивает QUERY с каждым ELEMENT. Для карточки с
непустым `vintage` используется `compare_year_matters`, для остальных —
`compare_year_not_matter`. Ноль совпадений даёт `not_found`, одно — его slug, два и более
передаются в `resolve_multiple_same`.

`run.json` обновляется атомарно после каждого ответа. Параллельное завершение не влияет на
итоговый порядок: кейсы записываются в порядке `manifest.jsonl`.
`metrics.json` атомарно создаётся в конце завершённого прогона и содержит accuracy, latency,
throughput, число запросов/generations, usage и разбивку по размеру группы. Если predictor
содержит tie-breaker, блок `tie_breaker` показывает число переданных в него кейсов и их долю
от всех attempted queries; неуспешный вызов tie-breaker тоже учитывается. Ошибки исключены из
quality denominator, а `not_found` считается полноценным ответом. Поле `not_found` сохраняет
число таких ответов; соседний блок `not_found_metrics` считает `not_found` положительным
классом и содержит support, predicted, TP/FP/FN/TN, precision, recall, F1 и accuracy.
