# Пайплайн

Оркестратор `vinishko/pipeline/pipeline.py` связывает шаги, которые друг о друге не знают: нормализация (`steps/normalization`) →
визуальный поиск (`steps/vis_searcher`) → выбор позиции либо отказ. Для `search.mode: top_n` автоматически подключается `steps/filter`,
для `groups` — прежний `steps/near_duplicates`. По умолчанию `top_k: 5`, `top_n`: фильтр сравнивает всех кандидатов, даже единственного.
Промпты прежние; настройки фильтра — в `steps/filter/config.yaml`. Общие структуры — `structs.py`: `BottleCrop` и `RejectedBottle` от
нормализации, `Candidate`, `BottleCandidates` и `UnmatchedBottle` от поиска, `MatchedBottle` от второго уровня, `Rejection` и `BottleOutcome` общие; причины отказов у каждого шага свои, наследник `Reason` со своим `stage`. HTTP-приложение — `vinishko/app`.

```python
from vinishko.pipeline.pipeline import Pipeline

result = Pipeline()(
    photo
)  # Normalizer(), VisSearcher(), второй уровень по режиму поиска; свой шаг или Catalog — аргументом, search=False / resolve=False — без шага
result.normalization  # разметка нормализации как есть
result.search  # BottleCandidates | UnmatchedBottle на каждую годную бутылку, в том же порядке
result.resolution  # MatchedBottle | UnmatchedBottle на каждый ответ поиска; пусто без resolver
result.matched  # бутылки с выбранной позицией: .candidate, .source, .checklist
result.bottles  # BottleOutcome на каждую бутылку: маска, скор и ровно один исход — match, candidates, rejection (stage, reason) либо normalized
result.items  # итоговая разметка: бутылка без ответа поиска или второго уровня — RejectedBottle с причиной
result.timings  # секунды на шаг
```

Устройства — `segmentation.device` в normalize.toml и `device` в config.yaml (`auto`, `cpu` либо `cuda:<индекс>`); переменные окружения `NORMALIZER_DEV` и `VIS_SEARCHER_DEV` их перекрывают.

С поиском `Pipeline` при создании сверяет нормализатор с каталогом: часть normalize.toml, определяющая пиксели кропов (`Normalizer.crop_config`,
без устройства, порогов отбора и записи файлов), должна совпадать с `manifest.json`, который `build_catalog` кладёт рядом с картинками коллекции.
Иначе ошибка перечисляет расхождения: пересобрать коллекцию либо вернуть конфиг. Там же каждая позиция коллекции (`searcher.slugs()`)
сверяется с каталогом: если какой-то нет, ошибка с примерами — строка каталога нужна в ответе на любого кандидата.

## Конфиг оркестратора и каталог

`config.yaml` рядом с модулем: откуда брать CSV каталога — `catalog.kind: local` с `path` либо `s3` с `endpoint`, `bucket`, `key`;
`slug_column` — колонка с идентификатором позиции. По умолчанию `../app/catalog.csv`: копия каталога хакатона лежит рядом с приложением
и едет вместе с кодом. `catalog.py` читает его в `Catalog` (строки по slug) один раз при создании `Pipeline`, оркестратор отдаёт строку
позиции в ответ API и карточки позиций второму уровню. Конфиги шагов лежат рядом с шагами.

## Итог по бутылке

`result.bottles` — `BottleOutcome` на каждую бутылку фото: маска и скор отбора, кроп (если нормализация пропустила) и ровно один исход:
`match` (`MatchedBottle`), `candidates` (второй уровень выключен), `rejection` (`Rejection`: `stage`, `reason`, `detail`, `message`) либо
ничего у годной бутылки без поиска; `status` — `matched | candidates | rejected | normalized`. Отказы всех шагов устроены одинаково:
`Rejection` с причиной из перечисления шага (`NormalizationReason`, `SearchReason`, `NearDuplicateReason`), шаг виден по `rejection.stage`.
`RejectedBottle` — бутылка, отвергнутая нормализацией, либо годная, которую отверг поздний шаг (`RejectedBottle.of(crop, rejection)`).

## Оценка с дампами шагов: evaluate_pipeline.py

```bash
python -m vinishko.pipeline.evaluate_pipeline --test-dir datasets/hack-vine/test   # метрики + выходы шагов, reports/pipeline/<время>
python -m vinishko.pipeline.evaluate_pipeline --image photo.jpg -o runs/           # одно фото без разметки: выходы шагов в runs/photo/
```

`--no-resolve` — без второго уровня, итоговый ответ — top-1 поиска; `--no-search` — только нормализация; `--dump misses|none` — папки только
для промахов либо без папок; `--config`, `-c/--set` — конфиги поиска и нормализации. На каждое фото папка (`<out>/dumps/<фото>/`, для `--image` —
`<out>/<фото>/`):

| Путь                         | Что внутри                                                                                                   |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `source.<ext>`               | копия исходника                                                                                              |
| `normalization/`             | на каждую годную бутылку `photo_bN.jpg` (кроп поиска), `photo_bN_box.jpg` (вся бутылка для второго уровня), маски и json, как у CLI нормализации; `markup.json` — все бутылки, отказы с шагом и причиной |
| `search/`                    | разбор поиска как при `debug_path`: `q<N>_query_<uuid>.jpg` и `_box.jpg`, `q<N>_<ранг>_<группа>_<slug>_<скор>[_bygroup].jpg`, `results.json` с косинусами по входам |
| `resolve/`                   | trace модели по uuid бутылки; filter пишет даже для одного кандидата, near_duplicates пропускает singleton группы |
| `result.json`                | итог по каждой бутылке: выбранная позиция с источником и наблюдениями модели, кандидаты либо отказ с шагом и причиной, время шагов |

По тестовому набору ещё `report.json` и `per_image.csv`. В `report.json` два блока метрик. `search` — как у `vis_searcher.evaluate`: recall@1/3/5,
`group_recall`, отказ поиска. `e2e` — классификация итогового ответа, ведь наружу уходит один slug либо отказ: `accuracy` — верный slug либо верный
отказ на фото без ответа, по всем фото, кроме тех, чей ответ не попал в коллекцию; `accuracy_all_images` — то же, но такие фото считаются
ошибкой, это честный счёт на тесте; `precision` — доля верных среди выданных ответов, `recall` — среди фото с ответом, `f1`; `confusion` — исходы
в штуках; дальше `group_of_answer`, `wrong_slug`, ложные и верные отказы по шагам, `resolved_by_model` — сколько раз звали модель, время по шагам.

## Сквозная проверка через API: vinishko/e2e.py

```bash
python -m vinishko.app --port 8000                       # в одном терминале
python -m vinishko.e2e --url http://127.0.0.1:8000       # в другом: /health, потом каждое фото теста в /recognize
```

Сервис как чёрный ящик, без промежуточных картинок: те же метрики классификации по ответам API (`accuracy`, `precision`, `recall`, `f1`,
`confusion`, отказы по шагам, ошибки сервиса, время ответа p50/p90), отчёт в `reports/e2e/<время>`. Описание ответа API — `vinishko/app/README.md`.
