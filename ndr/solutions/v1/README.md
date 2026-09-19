# NDR solution v1: схема structured output в system prompt

## Гипотеза эксперимента

Baseline передавал JSON Schema только в `response_format`. DeepSeek в шести вызовах не видел
форму ответа, начинал рассуждение с `Need infer schema?`, исчерпывал 4096 токенов и возвращал
`finish_reason=length`. В v1 точная схема из той же Pydantic-модели дополнительно включается в
system prompt каждого вызова. Для resolver туда попадает динамический enum допустимых slug.

Других изменений относительно baseline нет: pipeline, prompts, модели данных, параметры
генерации и routing сохранены. Ретраев и исправления невалидного JSON нет; любая ошибка
сохраняется runner-ом как результат эксперимента.

Из корня репозитория в PowerShell:

```powershell
Copy-Item .env.example .env
# Заполните OPENROUTER_API_KEY; остальные настройки — в этой папке, config.py
python ndr/run/run.py --solution v1
```

Результаты: новый каталог `ndr/results/v1/<UTC timestamp>/` на каждый прогон.

Все постоянные настройки находятся в одном файле: `ndr/solutions/v1/config.py`, блок `SETTINGS`.
Там задаются модель и endpoint, параметры генерации, reasoning, routing провайдеров, timeout,
concurrency и seed порядка кандидатов. API-ключ остаётся в переменной окружения.

Чтобы создать независимую версию, скопируйте всю папку предыдущего solution под новым именем и
запускайте её через `--solution <новое-имя>`. Runner не копирует solution в results, но сохраняет
его путь, по-файловые SHA-256 и общий fingerprint на момент запуска. Уже прогнанную версию
после этого не изменяйте.

## Анализ baseline

Baseline прогнан на `deepseek/deepseek-v4.1-flash`, 51 кейсе и concurrency 16. Headline accuracy
runner-а — 43/45 = 95,56%, но она условная: шесть contract errors исключены из знаменателя.
Сквозной результат без сокрытия ошибок — **43/51 = 84,31%**.

| Показатель | Baseline |
|---|---:|
| Верных / всех кейсов | 43 / 51 |
| Сквозная accuracy | 84,31% |
| Условная accuracy runner-а | 95,56% |
| Contract errors | 6 |
| Predictor/HTTP errors | 0 |
| Неверный slug | 1 |
| `not_found` вместо gold | 1 |
| Latency p50 / p95 / max | 29,21 / 72,45 / 121,08 с |
| Model requests | 265 |
| Prompt / completion tokens | 436386 / 234420 |
| Стоимость | $0,291835 |

Шесть ошибок `q-000033`, `q-000034`, `q-000039`, `q-000042`, `q-000044` и `q-000046`
одинаковы: HTTP 200, затем текст `Need infer schema?`, ровно 4096 completion tokens,
`finish_reason=length` и невалидный JSON. Это непосредственно проверяемая причина изменения v1.

Оставшиеся два промаха не относятся к отсутствующей схеме:

- `q-000045`: QUERY явно подписан `BLANC DE BLANCS BRUT 2019`, а Эталон и карточка gold-slug —
  `EXTRA BRUT 2019`. Модель последовательно отметила конфликт сахара. Это несогласованность
  QUERY/Эталона/карточки с folder-derived gold, а не подтверждённая ошибка распознавания.
- `q-000050`: сравнение ошибочно приняло за `same` и сухой, и полусухой Millstream AV;
  resolver выбрал полусухой. В observation модель прямо приписала QUERY слово «полусухое».
  Это семантическая/OCR-ошибка на минимальном различии этикеток.

Pipeline также дорог для product SLA: каждый QUERY сравнивается с кандидатами отдельными
вызовами, поэтому baseline сделал 265 запросов, а p50 составил 29,21 с. Это отдельная гипотеза
для следующей версии; v1 её намеренно не смешивает с исправлением контракта.

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
            routing_mode="throughput",  # Nitro
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

`routing_mode` преобразуется в `provider.sort`; по умолчанию используется `throughput`,
то есть режим OpenRouter Nitro. Это поле добавляется в каждый comparison и resolver request.
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
