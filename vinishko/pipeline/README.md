# Пайплайн

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
```

Поиск берёт один ближайший вектор из удалённого Qdrant и по его `group_slugs` загружает всю группу. Если в группе одна позиция, она становится ответом без вызова OpenRouter. Для группы из нескольких позиций NDR v5 получает нормализованный кроп запроса, оригинальные эталоны и карточки из локального `catalog.csv`. Он возвращает один slug либо `near_duplicate_not_found`; ошибки вызова и нарушения контракта поднимаются как ошибки пайплайна. `result.search[*].selection` показывает источник решения и наблюдения NDR.

По текущему `steps/vis_searcher/config.yaml` оригинальные эталоны читаются из `datasets/local/catalog/images` по полю `source_image` в Qdrant, а полные карточки — из `datasets/local/catalog/catalog.csv`. S3 для запроса не нужен; `images: null` означает чтение оригиналов вместо кропов коллекции. Копия данных лежит в `datasets/local/catalog` и `datasets/local/test`, эти папки исключены из Git. Модель NDR по умолчанию `deepseek/deepseek-v4.1-flash`; `OPENROUTER_MODEL` её перекрывает. Настройки вызова и prompt-файлы скопированы из `ndr/solutions/v5`.

## Ключи и запуск

Скопируйте `.env.example` в `.env` в корне репозитория и впишите ключ:

```powershell
Copy-Item .env.example .env
```

```dotenv
OPENROUTER_API_KEY=ваш_ключ
QDRANT_API_KEY=ключ_если_сервер_его_требует
```

`debug.py` автоматически загружает `.env` из корня репозитория; `.env` исключён из Git. Для `--stage normalization` ключи не нужны. Для `--stage search` нужен только доступ к Qdrant; для `--stage full` OpenRouter вызывается, если в найденной группе больше одного SKU. Можно задать ключи переменными окружения PowerShell; они имеют приоритет над `.env`.

Первые две фотографии из `test.csv`, только нормализация:

```powershell
.venv/Scripts/python.exe -m vinishko.pipeline.debug datasets/local/test -o datasets/local/runs --stage normalization --limit 2
```

Те же два фото до векторного поиска или до финального ответа:

```powershell
.venv/Scripts/python.exe -m vinishko.pipeline.debug datasets/local/test -o datasets/local/runs --stage search --limit 2
.venv/Scripts/python.exe -m vinishko.pipeline.debug datasets/local/test -o datasets/local/runs --stage full --limit 2
```

Каждый запуск перезаписывает папки выбранных фото в `datasets/local/runs`. Нормализатор и поиск загружаются один раз на весь выбранный набор. Можно передать один файл или директорию `images` вместо папки датасета. Без `--limit` выбираются все фото.

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
| `normalization/`             | кропы `photo_bN.jpg`, маски и json на каждую годную бутылку, как у CLI нормализации; `markup.json` — все бутылки, отказы с причиной |
| `search/`                    | разбор поиска как при `debug_path`: `q<N>_query_<uuid>.jpg`, `q<N>_<ранг>_<группа>_<slug>_<cos>[_bygroup].jpg`, `results.json` |
| `rerank/`                    | полный trace NDR v5 по UUID бутылки для тех групп, где потребовался вызов модели |
| `result.json`                | итог по каждой бутылке: кандидаты либо отказ с шагом и причиной, время шагов                                  |

Конфиг поиска — `--search-config`, по умолчанию `steps/vis_searcher/config.yaml`; его `debug_path` здесь не используется.
