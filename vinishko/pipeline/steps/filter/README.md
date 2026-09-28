# Фильтр кандидатов top_n

Копия `near_duplicates`, которая сравнивает всех кандидатов поиска без отбора по группам.
Вход — `list[BottleCandidates]`, выход — по одному `MatchedBottle` либо `UnmatchedBottle` на бутылку, в том же порядке.
Даже единственный кандидат проверяется моделью. Промпты скопированы дословно.

## Подключение

В `../vis_searcher/config.yaml`:

```yaml
top_k: 5
search:
  mode: top_n
  cosine_threshold: 0.7
```

`Pipeline`, API и `evaluate_pipeline.py` автоматически выбирают `FilterResolver` для `top_n`;
для `groups` используется прежний `NearDuplicateResolver`. `resolve=False` отключает второй уровень.
Если лучший скор ниже порога, поиск отказывает. Иначе фильтр получает до пяти кандидатов
по итоговому скору, включая позиции ниже порога; группы не расширяются.

```python
from vinishko.pipeline.pipeline import Pipeline

result = Pipeline()(photo)
result.matched  # выбранные позиции: .candidate.slug, .source, .checklist
```

Один вызов модели на бутылку: QUERY — `BottleCrop.box_crop` до 1600 px,
ELEMENT — картинка позиции `Candidate.image` и её карточка из каталога.
Ответ — переданный slug (`source = filter_v1`) либо `not_found`
(`reason = filter_not_found`, `stage = resolve`). Группы и `group_slugs` не нужны фильтру;
поле группы остаётся метаданными общих структур и API.
Ошибка API, таймаут или посторонний slug — `FilterError`; top-1 вместо ответа не подставляется.
Trace сохраняется в `resolve/<uuid>.json` при оценке с дампами.

## Настройки и оценка

Модель, маршрутизация, генерация и таймаут — в `config.yaml` этой папки.
Сохранены `openrouter.extra_body.reasoning.enabled: false` и HTTP-прокси
из `openrouter_http_proxy` в окружении либо `.env` корня проекта.
Ключ — `OPENROUTER_API_KEY`; `OPENROUTER_MODEL` перекрывает модель конфига.
Параллелизм и размер кропа настраиваются переменными в начале `resolve.py`.

Итоговая оценка сравнивает выбранный slug с `slug` в разметке `test.csv` точно;
совпадение группы не засчитывает другой slug. Пустой ожидаемый slug означает верный отказ.
`recall@1/3/5` оценивает ретривал отдельно от итогового решения фильтра.
Ожидаемые позиции вне коллекции исключаются из `accuracy`, но считаются ошибками в `accuracy_all_images`.

Офлайн-проверки:

```bash
python -m unittest vinishko.pipeline.steps.filter.test_resolve
```
