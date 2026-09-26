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

Первые две фотографии из `datasets/local/test`:

```powershell
# Только нормализация
python -m vinishko.pipeline.debug datasets\local\test -o datasets\local\runs --stage normalization --limit 2

# Быстрая проба на небольшом фото с меньшим входом SAM3
python -m vinishko.pipeline.debug datasets\local\test\images\test_000002.jpg -o datasets\local\runs --stage normalization --set segmentation.imgsz=672

# Нормализация и векторный поиск — нужен доступ к Qdrant
python -m vinishko.pipeline.debug datasets\local\test -o datasets\local\runs --stage search --limit 2 --batch-size 2

# Повторный поиск по уже готовой нормализации в том же каталоге runs — SAM3 не загружается
python -m vinishko.pipeline.debug datasets\local\test -o datasets\local\runs --stage search --limit 2 --skip-normalization

# Вообще без нормализации: каждое исходное фото целиком становится запросом поиска
python -m vinishko.pipeline.debug datasets\local\test -o datasets\local\runs_raw --stage search --limit 2 --no-normalization

# Полный pipeline, включая NDR — OpenRouter вызывается для групп из нескольких SKU
python -m vinishko.pipeline.debug datasets\local\test -o datasets\local\runs --stage full --limit 2 --batch-size 2

# Повторить поиск и NDR без нормализации
python -m vinishko.pipeline.debug datasets\local\test -o datasets\local\runs --stage full --limit 2 --skip-normalization

# Полный pipeline по исходным фото целиком, без SAM3
python -m vinishko.pipeline.debug datasets\local\test -o datasets\local\runs_raw --stage full --limit 2 --no-normalization
```

`debug.py` автоматически загружает `.env` из корня репозитория; файл исключён из Git. Переменные окружения PowerShell имеют приоритет. Обычный запуск перезаписывает папки выбранных фото в `datasets/local/runs`. `--stage search` повторно выполняет нормализацию, если не указан `--skip-normalization`; без `--limit` выбираются все фото.

Есть два независимых режима без SAM3:

| Флаг | Что передаётся в поиск | Нужен предыдущий запуск нормализации |
| --- | --- | --- |
| `--skip-normalization` | Сохранённые кропы и разметка из `<output-dir>/<имя фото>/normalization/` | Да; используйте тот же `-o` и те же исходные фото |
| `--no-normalization` | Исходное фото целиком как один запрос, без выделения бутылки или этикетки | Нет |

Оба флага работают только с `--stage search` или `full` и не допускают `--set`; вместе их указывать нельзя. Старые `search/`, `rerank/` и `result.json` заменяются, а сохранённая папка `normalization/` остаётся на месте. Новые результаты нормализации содержат точный несжатый кроп для повторного поиска; старые результаты тоже поддерживаются, но кроп в них берётся из JPEG и может дать немного другой эмбеддинг. В режиме `--no-normalization` фото с несколькими бутылками также даёт один общий запрос, поэтому точность поиска и NDR может снизиться.

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

Поиск берёт один ближайший вектор из удалённого Qdrant и по его `group_slugs` загружает всю группу. Если в группе одна позиция, она становится ответом без вызова OpenRouter. Для группы из нескольких позиций NDR v5 получает нормализованный кроп запроса, оригинальные эталоны и карточки из локального `catalog.csv`. Он возвращает один slug либо `near_duplicate_not_found`; ошибки вызова и нарушения контракта поднимаются как ошибки пайплайна. `result.search[*].selection` показывает источник решения и наблюдения NDR.

## Исходные фото и кропы

`datasets/local/catalog/images` и `datasets/local/test/images` содержат исходные фото, без заранее подготовленных кропов. При обычном запуске нормализатор создаёт кроп запроса, поиск считает по нему вектор и обращается к коллекции Qdrant из конфига. Отладочный CLI дополнительно сохраняет кроп в `runs/<фото>/normalization/`; при `--skip-normalization` повторный поиск читает сохранённую нормализацию.

Векторы каталога рассчитываются при отдельной сборке коллекции: `build_catalog` читает исходные фото, нормализует каждое принятое фото, сохраняет кроп в хранилище `images` из **конфига сборки** и записывает вектор в Qdrant. Кропы каталога не обязаны лежать рядом с исходными фото или тестом. Текущее местонахождение кропов, использованных для удалённой коллекции, конфиг не устанавливает.

Текущий `steps/vis_searcher/config.yaml` предназначен для **запросов к уже собранной коллекции**: `images: null` отключает чтение её кропов, а `reference_images` указывает на локальные исходные фото `datasets/local/catalog/images`. Поиск берёт их имена из поля `source_image` в Qdrant. NDR получает те же оригиналы и карточки из `datasets/local/catalog/catalog.csv`. S3 при таком запуске не используется; копии `datasets/local/catalog` и `datasets/local/test` исключены из Git. Для новой сборки коллекции нужен отдельный конфиг с заполненным `images` — с текущим `images: null` `build_catalog` завершится ошибкой. Пример S3-префикса в комментарии конфига не подтверждает, что там хранятся кропы существующей коллекции.

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
| `normalization/`             | кропы `photo_bN.jpg`, точные кропы `photo_bN_pipeline.npy`, маски и json на каждую годную бутылку; `markup.json` — все бутылки и причины отказов |
| `search/`                    | разбор поиска как при `debug_path`: `q<N>_query_<uuid>.jpg`, `q<N>_<ранг>_<группа>_<slug>_<cos>[_bygroup].jpg`, `results.json` |
| `rerank/`                    | полный trace NDR v5 по UUID бутылки для тех групп, где потребовался вызов модели |
| `result.json`                | итог по каждой бутылке: кандидаты либо отказ с шагом и причиной, время шагов                                  |

Конфиг поиска — `--search-config`, по умолчанию `steps/vis_searcher/config.yaml`; его `debug_path` здесь не используется.
