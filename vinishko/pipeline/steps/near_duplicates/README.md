# Второй уровень: выбор позиции внутри группы near-duplicates

Шаг пайплайна после визуального поиска. Вход — только бутылки с кандидатами, `list[BottleCandidates]`; выход — по одному ответу на каждую,
в том же порядке: `MatchedBottle` с ровно одной позицией каталога либо `UnmatchedBottle` с отказом `near_duplicate_not_found`
(`vinishko/pipeline/structs.py`); отказ — `Rejection` с `NearDuplicateReason`, `stage = resolve`. Списки кандидатов шаг не переставляет, он принимает решение.

```python
from vinishko.pipeline.pipeline import Pipeline

result = Pipeline()(
    photo
)  # второй уровень поднимается по умолчанию, resolve=False — без него
result.resolution  # MatchedBottle | UnmatchedBottle на каждый ответ поиска; отказы поиска переходят сюда как есть
result.matched  # только MatchedBottle: .candidate — выбранная позиция, .source — vector | ndr_v5, .checklist — наблюдения модели
```

`Pipeline` сам отдаёт второму уровню только `BottleCandidates` и ставит его ответы на места ответов поиска; `result.items` и `result.crops`
учитывают отказы обоих шагов. Отдельно: `NearDuplicateResolver()(found)`. В `config.yaml` поиска по умолчанию `search.mode: groups`, как нужно этому шагу.

## Как выбирает

Код `configs.py`, `models.py`, `predictor.py` и три prompt-файла адаптированы из `ndr/solutions/v5` исходного репозитория, `resolve.py` —
адаптер к структурам пайплайна и картинкам в памяти.

Берётся группа top-1 кандидата. Группа нужна целиком, поэтому **поиск должен идти в режиме `search.mode: groups`**: в `top_n` члены группы
за пределами `top_k` в кандидаты не попадают, и шаг падает с `NearDuplicateError`. Для группы из одной позиции ответ — она, без модели,
`source = vector`. Для группы из нескольких — один запрос к OpenRouter (`source = ndr_v5`): QUERY — `BottleCrop.box_crop` запроса, не крупнее
1600 px; ELEMENT — картинка позиции из коллекции (`Candidate.image`, вся бутылка каталожного фото) и карточка: название, винодельня,
винтаж, сорт, категория, сахар, игристость, крепость, выдержка. Карточка берётся из метаданных точки; `cards` в конструкторе подменяет её полями каталога (`cards_from_rows` по строкам CSV,
`Pipeline` передаёт их из своего каталога сам) — это нужно для коллекций, собранных до появления поля `aging_or_reserve` в метаданных. Ответ модели
строго ограничен схемой: slug из группы либо `not_found`.

`not_found` даёт `UnmatchedBottle` с наблюдениями модели в `detail`. Ошибка API, таймаут или нарушение схемы — `NearDuplicateError`,
top-1 вместо ответа не подставляется; полный ответ модели без base64 и без ключа пишется в `trace_dir/<uuid>.json`, если директория задана.
Вызовы модели идут параллельно, не больше 32 разом.

## Настройки

Модель, параметры генерации, маршрутизация провайдера и таймаут — в `config.yaml` рядом с модулем, загрузчик `configs.py`;
свой файл — `NearDuplicateResolver(settings=load_config(Path(...)))`. Ключ — `OPENROUTER_API_KEY` из `.env` в корне проекта или из окружения,
без него вызов завершается ошибкой с подсказкой; `OPENROUTER_MODEL` перекрывает модель из конфига.

Дополнительные поля JSON-запроса задаются в `openrouter.extra_body`:

```yaml
openrouter:
  extra_body:
    reasoning:
      enabled: false
```

Поля добавляются на верхний уровень рядом с `messages`, без обёртки `extra_body`, и перекрывают одноимённые поля из `generation` целиком.
`model`, `messages`, `provider` и `response_format` подменять нельзя. `extra_body: {}` оставляет параметры генерации без изменений.
Отключение reasoning уже задано в конфиге; поддержка зависит от модели. См. [документацию OpenRouter](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).

Для HTTP-прокси запросов NDR добавьте в `.env`:

```dotenv
openrouter_http_proxy=http://127.0.0.1:8080
```

Укажите адрес и порт своего прокси. Настройка действует на вызовы NDR к OpenRouter; без неё используются стандартные настройки `urllib`.
Окружение процесса имеет приоритет над `.env`. После изменения перезапустите evaluate или API-сервис.

## Прогон и проверки

```bash
python -m vinishko.pipeline.evaluate_pipeline --image photo.jpg -o runs/         # выходы всех шагов, trace модели в runs/photo/resolve/<uuid>.json
python -m vinishko.pipeline.evaluate_pipeline --test-dir datasets/hack-vine/test  # метрики поиска и классификации итога, reports/pipeline/<время>
python -m unittest vinishko.pipeline.steps.near_duplicates.test_resolve -v   # офлайн, без qdrant, S3 и модели
```
