# Пайплайн

## Быстрый запуск

Команды выполняются из корня репозитория. Для полного прогона скопируйте `.env.example` в `.env` и впишите ключ OpenRouter (ключ Qdrant — если сервер его требует):

На CPU перед поиском установите OpenVINO (для одной нормализации он не нужен):

```powershell
uv sync --group cpu-inference
```

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

```dotenv
OPENROUTER_API_KEY=ваш_ключ
# QDRANT_API_KEY=ключ_если_сервер_его_требует
```

Актуальная выборка из `s3://vino/test-data-norm/norm-images.tar.gz` лежит в `datasets/local/normalized/test`: `images_crop` содержит 176 кропов этикеток для DINO, `images_crop_box` — 176 кропов бутылок для NDR. Исходные фото и `test.csv` остаются в `datasets/local/test`. Эти команды **не загружают SAM3** и после прогона записывают `metrics.json` и `per_image.csv` в каталог `-o`:

```powershell
# Только векторный поиск по готовым кропам DINO
python -m vinishko.pipeline.debug datasets\local\test -o datasets\local\runs_s3 --stage search --skip-normalization --normalized-dir datasets\local\normalized\test --limit 2

# Векторный поиск + выбор внутри группы NDR v5; для вызова модели нужен OPENROUTER_API_KEY
python -m vinishko.pipeline.debug datasets\local\test -o datasets\local\runs_s3 --stage full --skip-normalization --normalized-dir datasets\local\normalized\test --limit 2

# Вся тестовая выборка: 176 фото идут в поиск
python -m vinishko.pipeline.debug datasets\local\test -o datasets\local\runs_s3 --stage search --skip-normalization --normalized-dir datasets\local\normalized\test

# Только NDR по уже сохранённому поиску в том же каталоге -o; Qdrant и DINO не запускаются
python -m vinishko.pipeline.debug datasets\local\test -o datasets\local\runs_s3 --stage ndr --normalized-dir datasets\local\normalized\test
```

Для `--stage ndr` нужны прежние `runs_s3/<имя фото>/search/results.json`, JPEG в `search/` и `result.json`. Каталог `search/` сохраняется, `rerank/`, `result.json` и итоговые метрики обновляются. Ключ `OPENROUTER_API_KEY` нужен для групп с несколькими SKU; если в группе один SKU, NDR выбирает его без вызова модели. `--normalized-dir` здесь выбирает фото с готовыми кропами бутылок для NDR.

Новые локальные прогоны нормализации и другие режимы:

```powershell
# Только нормализация
python -m vinishko.pipeline.debug datasets\local\test -o datasets\local\runs --stage normalization --limit 2

# Быстрая проба на небольшом фото с меньшим входом SAM3
python -m vinishko.pipeline.debug datasets\local\test\images\test_000002.jpg -o datasets\local\runs --stage normalization --set segmentation.imgsz=672

# Нормализация и векторный поиск — нужен доступ к Qdrant
python -m vinishko.pipeline.debug datasets\local\test -o datasets\local\runs --stage search --limit 2 --batch-size 2

# Повторный поиск по уже готовой нормализации в том же каталоге runs — SAM3 не загружается
python -m vinishko.pipeline.debug datasets\local\test -o datasets\local\runs --stage search --limit 2 --skip-normalization

# Полный pipeline, включая NDR — OpenRouter вызывается для групп из нескольких SKU
python -m vinishko.pipeline.debug datasets\local\test -o datasets\local\runs --stage full --limit 2 --batch-size 2

# Повторить поиск и NDR без нормализации
python -m vinishko.pipeline.debug datasets\local\test -o datasets\local\runs --stage full --limit 2 --skip-normalization

```

`debug.py` автоматически загружает `.env` из корня репозитория; файл исключён из Git. Переменные окружения PowerShell имеют приоритет. Обычный запуск перезаписывает папки выбранных фото в `datasets/local/runs`. `--stage search` повторно выполняет нормализацию, если не указан режим обхода; без `--limit` выбираются все фото.

Есть два способа использовать готовую нормализацию без SAM3:

| Флаг | Что передаётся в поиск | Нужен предыдущий запуск нормализации |
| --- | --- | --- |
| `--skip-normalization` | Сохранённые кропы и разметка из `<output-dir>/<имя фото>/normalization/`; с `--normalized-dir` — готовые кропы S3 | Для сохранённого запуска — да; для S3 — нет |
| `--normalized-dir <папка>` | `images_crop` для DINO и `images_crop_box` для NDR; старая плоская папка JPEG использует исходное фото для NDR | Нет; кропы должны лежать в указанной папке |

Режимы работают только с `--stage search` или `full` и не допускают `--set`; `--skip-normalization` можно сочетать с `--normalized-dir`. Старые `search/`, `rerank/` и `result.json` заменяются, а сохранённая папка `normalization/` остаётся на месте. Новые результаты нормализации содержат два точных несжатых кропа: этикетка для DINO и вся бутылка для просмотра и каталога. В режиме готовых парных JPEG search получает `images_crop`, а NDR — `images_crop_box` без уменьшения. Старая плоская папка JPEG использует исходное фото для NDR.

При наличии `test.csv` `metrics.json` разделяет оценку поиска (`search`) и итогового выбора (`ndr`, только для `--stage full`, `ndr` и `metrics`). Поиск оценивается по сохранённому `search/results.json` до NDR: `recall@1/3/5` по первому найденному кропу, `recall@k_any_bottle` по всем кропам, `group_recall` по группе и распределение косинуса top-1. NDR оставляет один SKU или отказ, поэтому у него `accuracy` по всем размеченным фото, включая slug без вектора в индексе; `accuracy_indexed_or_empty` исключает такие фото, а `correct_slug_rate_for_indexed_answers` показывает долю верно выбранных slug среди фото, чей правильный slug есть в индексе. Метрик `@k` для NDR нет. Оба этапа показывают `false_reject`, `correct_reject`, `false_accept`, `not_indexed_accepted` и `no_bottle`. В знаменатель recall входят только фото, чей slug реально проиндексирован в Qdrant; пустые slug проверяют отказ, а заданные, но отсутствующие в индексе, учитываются отдельно как `answer_not_indexed`. При поиске сохраняется `indexed_slugs.json`, который затем использует отдельный запуск NDR или пересчёт. `per_image.csv` содержит исходный slug, наличие в индексе, позиции ответа и решения обоих этапов. С `--limit 2` метрики относятся только к первым двум фото из `test.csv`. `--labels путь\к\test.csv` задаёт разметку явно.

Пересчитать метрики уже завершённого прогона без моделей и повторных вызовов NDR:

```powershell
.venv/Scripts/python.exe -m vinishko.pipeline.debug datasets/local/test -o datasets/local/runs_s3 --stage metrics
```

Текущий удалённый Qdrant требует `QDRANT_API_KEY`: без него сервер отвечает 401. Замените пример `ключ_если_сервер_его_требует` в `.env` настоящим ключом от коллеги и уберите `#` перед строкой. Не добавляйте `.env` в Git.

`--limit` ограничивает число фото всего прогона, `--batch-size` — число фото в одной пачке поиска (по умолчанию 2). SAM3 нормализует их последовательно: установленный predictor поддерживает один кадр за вызов, но промпты бутылки, этикетки и крышки выполняются общим батчем для кадра. Кропы из пачки фото собираются для энкодера и Qdrant. На этой машине с 16 ГБ ОЗУ `cpu_batch_size: 1` в конфиге поиска ограничивает размер одного вызова OpenVINO, но Qdrant получает векторы всей пачки одним запросом. Когда будет достаточно свободной памяти, можно проверить `cpu_batch_size: 2`; если начнётся подкачка памяти, задайте также `--batch-size 1`.

Первый запуск нормализации скачивает `sam3.pt` (около 3,5 ГБ) из [ModelScope](https://modelscope.cn/models/facebook/sam3/resolve/master/sam3.pt) в `%LOCALAPPDATA%\vino\Cache\sam3.pt`. После успешного скачивания файл используется при следующих запусках без сети; если прервать скачивание, текущий загрузчик начинает файл заново. Первый поиск дополнительно загружает ONNX-модель из Hugging Face в кэш `huggingface_hub`; она тоже переиспользуется. На Ryzen 7 5825U в текущем прогоне первая фотография заняла 372 с вместе со скачиванием SAM3, следующая — 34 с на тёплой модели. Отдельный прогон небольшого фото с `imgsz=672` занял 25 с, но это меняет настройки относительно калибровки отбора и может ухудшить распознавание на крупных фото. Для времени поиска и полного пайплайна замера пока нет.

## Как устроено

Оркестратор `vinishko/pipeline/pipeline.py` связывает шаги: нормализация (`steps/normalization`) →
визуальный поиск (`steps/vis_searcher`) → выбор точного SKU внутри группы (`steps/near_duplicates`, NDR v5). Общие структуры — `structs.py`: `BottleCrop` и `RejectedBottle` от
нормализации, `Candidate`, `BottleCandidates` и `UnmatchedBottle` от поиска; причины отказов у каждого шага свои, наследник `Reason`.

```python
from vinishko.pipeline.pipeline import Pipeline
from vinishko.pipeline.steps.vis_searcher import VisSearcher

result = Pipeline(searcher=VisSearcher())(photo)   # Normalizer и NDR-реранкер подключаются автоматически
result.normalization    # разметка нормализации как есть
result.search           # BottleCandidates | UnmatchedBottle на каждую годную бутылку, в том же порядке
result.items            # итоговая разметка: бутылка без ответа поиска — RejectedBottle с причиной
result.timings          # секунды на шаг
results = Pipeline(searcher=VisSearcher()).run_many([photo1, photo2])  # общий поиск кропов; ограничивайте размер списка
```

Поиск берёт один ближайший вектор из удалённого Qdrant и по его `group_slugs` загружает всю группу. Если в группе одна позиция, она становится ответом без вызова OpenRouter. Для группы из нескольких позиций NDR v5 получает `images_crop_box` в режиме парной S3-выборки или исходное фото запроса при обычном запуске, изображения кандидатов из локальной копии S3-каталога и карточки из `catalog.csv`. Для старых точек Qdrant без признака `image_crop=bottle_box` используются оригинальные эталоны. Он возвращает один slug либо `near_duplicate_not_found`; ошибки вызова и нарушения контракта поднимаются как ошибки пайплайна. `result.search[*].selection` показывает источник решения и наблюдения NDR.

## Исходные фото и кропы

`datasets/local/catalog/images` и `datasets/local/test/images` содержат исходные фото. Из S3 скачаны готовые кропы: `datasets/local/normalized/test/images_crop` (176 JPEG для DINO, 76,1 МБ), `datasets/local/normalized/test/images_crop_box` (176 JPEG для NDR, 33,1 МБ) и `datasets/local/normalized/catalog/images` (1981 JPEG для кандидатов VLM, 165,8 МБ). Из каталоговых файлов 1977 загружены после обновления с двумя кропами, четыре остались от прежней загрузки; принадлежность этих четырёх новой коллекции Qdrant без ключа пока не проверить. У исходников бывает `.webp`, а нормализованный файл называется `<stem>.jpg`. При обычном запуске нормализатор создаёт кроп поиска и кроп всей бутылки; отладочный CLI сохраняет оба в `runs/<фото>/normalization/`.

Векторы каталога рассчитываются при отдельной сборке коллекции: `build_catalog` читает исходные фото, нормализует каждое принятое фото, считает вектор по кропу этикетки и сохраняет кроп всей бутылки для VLM в хранилище `images`. Запрос использует уже заполненную коллегой коллекцию.

Текущий `steps/vis_searcher/config.yaml` настроен на модель `vitl16-1024`, коллекцию `catalog_vitl16_1024` и локальную копию S3-картинок кандидатов. На CPU OpenVINO считает модель батчами по одному изображению (`cpu_batch_size: 1`), а векторы пачки передаются Qdrant совместным запросом. Для запросов S3 не нужен: данные уже скачаны локально. Удалённый Qdrant требует ключ, которого пока нет; без `QDRANT_API_KEY` живой поиск и итоговые метрики проверить нельзя.

Модель NDR по умолчанию `deepseek/deepseek-v4.1-flash`; `OPENROUTER_MODEL` её перекрывает. Настройки вызова и prompt-файлы скопированы из `ndr/solutions/v5`.

Пробный запрос по одному фото:

```powershell
.venv/Scripts/python.exe -m vinishko.pipeline.debug datasets/local/test/images/<имя_из_test.csv> -o datasets/local/runs
```

Устройства — `segmentation.device` в normalize.toml и `device` в config.yaml (`auto`, `cpu` либо `cuda:<индекс>`); переменные окружения `NORMALIZER_DEV` и `VIS_SEARCHER_DEV` их перекрывают.

## Выходные файлы

```bash
python -m vinishko.pipeline.debug photo.jpg -o runs/            # --no-search: синоним --stage normalization
```

В `runs/photo/` (старая директория с тем же именем удаляется):

| Путь                         | Что внутри                                                                                                   |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `normalization/`             | кропы `photo_bN.jpg` для DINO и `photo_bN_box.jpg` для просмотра, их точные `.npy`, маски и json; `markup.json` — все бутылки и причины отказов |
| `search/`                    | оба кропа запроса, картинки кандидатов с косинусом и `results.json` |
| `rerank/`                    | полный trace NDR v5 по UUID бутылки для тех групп, где потребовался вызов модели |
| `result.json`                | итог по каждой бутылке: кандидаты либо отказ с шагом и причиной, время шагов                                  |

В корне `-o` лежат `metrics.json` и `per_image.csv` по выбранным изображениям.

Конфиг поиска — `--search-config`, по умолчанию `steps/vis_searcher/config.yaml`; его `debug_path` здесь не используется.
