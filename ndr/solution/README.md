# NDR solution: запуск

Из корня репозитория в PowerShell:

```powershell
Copy-Item .env.example .env
# Заполните OPENROUTER_API_KEY; остальные настройки — в ndr/solution/config.py
python ndr/run/run.py
```

Результаты: `ndr/results/run.json`, `ndr/results/metrics.json` и
`ndr/results/by_case/<query_id>.json`.

Все постоянные настройки находятся в одном файле: `ndr/solution/config.py`, блок `SETTINGS`.
Там задаются модель и endpoint, параметры генерации, reasoning, routing провайдеров, timeout,
concurrency и seed порядка кандидатов. API-ключ остаётся в переменной окружения.

## Pipeline

Для каждого query выполняются следующие шаги:

1. QUERY отдельно сравнивается с каждым ELEMENT группы.
2. Если у карточки ELEMENT непустой `vintage`, используется
   `prompts/compare_year_matters.txt` и Pydantic-модель `YearMattersOutput`.
3. Если `vintage` пустой, используется `prompts/compare_year_not_matter.txt` и
   Pydantic-модель `YearNotMatterOutput`.
4. Ноль `verdict=same` превращается в `not_found`.
5. Один `verdict=same` сразу превращается в slug этого ELEMENT.
6. Два и более `same` передаются в `prompts/resolve_multiple_same.txt`; resolver обязан
   выбрать один slug по динамической Pydantic-модели с `Literal` только из прошедших первый
   этап slug.

Авторитетные контракты находятся в `models.py`. Pydantic запрещает дополнительные поля и
приведение типов, ограничивает verdict значениями `same/different`, проверяет согласованность
verdict с checklist и ограничивает observation по длине. Predictor отправляет провайдеру
JSON Schema, полученную из той же модели, которой затем валидирует декодированный ответ.

## Multimodal-вход

На каждом сравнении передаются QUERY, один ELEMENT и компактная карточка ELEMENT. Resolver
получает QUERY и все ELEMENT с `verdict=same`. Изображения передаются отдельными image content
blocks; MIME определяется по сигнатуре, а не расширению. `generation.image_detail` явно
передаётся в каждом image block; по умолчанию используется `original`. Для модели или
провайдера без поддержки `original` его можно заменить на `high` в `config.py`.

Карточка содержит `slug`, `name`, `winery`, `vintage`, `grapes`, `category`, `sugar`,
`sparkling`, `abv` и `aging_or_reserve`. Длинное дегустационное описание не передаётся.

## Reasoning и generations

Для каждого HTTP-вызова без фильтрации сохраняются сырой ответ и все choices, включая
`content`, `reasoning`, `reasoning_details`, refusal, annotations, usage и provider-specific
поля. `generation.generations` (или разовый `--generations N`) задаёт `n`, но при значении 1
параметр не отправляется; решением этапа становится первая generation, прошедшая JSON parsing
и прикладную проверку контракта.

Скрытое chain-of-thought, которое провайдер не возвращает, сохранить невозможно. Параметр
`generation.reasoning_exclude` управляет передачей доступных reasoning-данных; по умолчанию
он равен `False`. Значение `reasoning_effort="none"` отправляется явно и отключает reasoning;
`None` означает, что весь блок `reasoning` нужно опустить.

## Единая конфигурация

Главный редактируемый блок:

```python
SETTINGS = NdrSettings(
    openrouter=OpenRouterSettings(
        model="provider/model",
        routing=ProviderRoutingSettings(
            routing_mode="latency",  # None, price, throughput или latency
            only=(),
            ignore=(),
            allow_fallbacks=True,
            require_parameters=True,
        ),
    ),
    generation=GenerationSettings(
        temperature=0.0,
        max_completion_tokens=16000,
        generations=1,
        image_detail="original",
        reasoning_effort="max",
    ),
    execution=ExecutionSettings(
        timeout_seconds=120.0,
        concurrency=4,
        candidate_order_seed=0,
    ),
)
```

`routing_mode` преобразуется в `provider.sort`; по умолчанию используется `latency`.
`require_parameters=True` не позволяет маршрутизатору выбрать провайдера без поддержки
запрошенных параметров. Весь непустой provider-блок попадает в каждый comparison и resolver
request и сохраняется в request trace. Доступны также `order`, `only`, `ignore`,
`data_collection`, `zdr`, `quantizations` и `max_price`.

Приоритет connection/run overrides: CLI → `.env`/process environment → `config.py`. Параметры,
которых нет среди CLI-флагов (`top_p`, `top_k`, `min_p`, penalties, generation seed, stop и
provider routing), меняются только в `config.py`.

Runner читает эти переменные из корневого `.env`; явные переменные процесса и CLI-флаги
имеют приоритет. Другой файл можно указать через `--env-file`.

```text
OPENROUTER_API_KEY       ключ API
OPENROUTER_MODEL         model id вместо --model
OPENROUTER_API_BASE      endpoint вместо --api-base
OPENROUTER_SITE_URL      необязательный HTTP-Referer
OPENROUTER_APP_TITLE     необязательный X-Title
```

CLI-флаги из `ndr/run/run.py --help` предназначены для разового override. Predictor не хранит
изменяемого общего состояния и безопасен для одновременных вызовов.
