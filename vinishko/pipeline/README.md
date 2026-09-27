# Пайплайн

Оркестратор `vinishko/pipeline/pipeline.py` связывает шаги, которые друг о друге не знают: нормализация (`steps/normalization`) →
визуальный поиск (`steps/vis_searcher`) → реранкер (пока нет). Общие структуры — `structs.py`: `BottleCrop` и `RejectedBottle` от
нормализации, `Candidate`, `BottleCandidates` и `UnmatchedBottle` от поиска; причины отказов у каждого шага свои, наследник `Reason`.

```python
from vinishko.pipeline.pipeline import Pipeline
from vinishko.pipeline.steps.vis_searcher import VisSearcher

result = Pipeline(searcher=VisSearcher())(
    photo
)  # нормализатор поднимается сам: Normalizer()
result.normalization  # разметка нормализации как есть
result.search  # BottleCandidates | UnmatchedBottle на каждую годную бутылку, в том же порядке
result.items  # итоговая разметка: бутылка без ответа поиска — RejectedBottle с причиной
result.timings  # секунды на шаг
```

Устройства — `segmentation.device` в normalize.toml и `device` в config.yaml (`auto`, `cpu` либо `cuda:<индекс>`); переменные окружения `NORMALIZER_DEV` и `VIS_SEARCHER_DEV` их перекрывают.

С поиском `Pipeline` при создании сверяет нормализатор с каталогом: часть normalize.toml, определяющая пиксели кропов (`Normalizer.crop_config`,
без устройства, порогов отбора и записи файлов), должна совпадать с `manifest.json`, который `build_catalog` кладёт рядом с картинками коллекции.
Иначе ошибка перечисляет расхождения: пересобрать коллекцию либо вернуть конфиг.

## Прогон одной картинки

```bash
python -m vinishko.pipeline.debug photo.jpg -o runs/            # --no-search: только нормализация
```

В `runs/photo/` (старая директория с тем же именем удаляется):

| Путь                         | Что внутри                                                                                                   |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `normalization/`             | на каждую годную бутылку `photo_bN.jpg` (кроп поиска), `photo_bN_box.jpg` (вся бутылка для второго уровня), маски и json, как у CLI нормализации; `markup.json` — все бутылки, отказы с причиной |
| `search/`                    | разбор поиска как при `debug_path`: `q<N>_query_<uuid>.jpg` и `_box.jpg` (кроп поиска и вся бутылка), `q<N>_<ранг>_<группа>_<slug>_<cos>[_bygroup].jpg`, `results.json` |
| `result.json`                | итог по каждой бутылке: кандидаты либо отказ с шагом и причиной, время шагов                                  |

Конфиг поиска — `--search-config`, по умолчанию `steps/vis_searcher/config.yaml`; его `debug_path` здесь не используется.
