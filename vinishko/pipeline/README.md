# Пайплайн

Оркестратор `vinishko/pipeline/pipeline.py` связывает шаги, которые друг о друге не знают: нормализация (`steps/normalization`) →
визуальный поиск (`steps/vis_searcher`) → реранкер (пока нет). Общие структуры — `structs.py`: `BottleCrop` и `RejectedBottle` от
нормализации, `Candidate`, `BottleCandidates` и `UnmatchedBottle` от поиска; причины отказов у каждого шага свои, наследник `Reason`.

```python
from vinishko.pipeline.pipeline import Pipeline
from vinishko.pipeline.steps.vis_searcher import VisSearcher

result = Pipeline(searcher=VisSearcher())(photo)   # нормализатор поднимается сам: Normalizer()
result.normalization    # разметка нормализации как есть
result.search           # BottleCandidates | UnmatchedBottle на каждую годную бутылку, в том же порядке
result.items            # итоговая разметка: бутылка без ответа поиска — RejectedBottle с причиной
result.timings          # секунды на шаг
```

Устройства — переменные окружения `NORMALIZER_DEV` и `VIS_SEARCHER_DEV` (`cpu` либо `cuda:<индекс>`, без них `cuda:0` при доступной CUDA).

## Прогон одной картинки

```bash
python -m vinishko.pipeline.run photo.jpg -o runs/            # --no-search: только нормализация
```

В `runs/photo/` (старая директория с тем же именем удаляется):

| Путь                         | Что внутри                                                                                                   |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `normalization/`             | кропы `photo_bN.jpg`, маски и json на каждую годную бутылку, как у CLI нормализации; `markup.json` — все бутылки, отказы с причиной |
| `search/`                    | разбор поиска как при `debug_path`: `q<N>_query_<uuid>.jpg`, `q<N>_<ранг>_<группа>_<slug>_<cos>[_bygroup].jpg`, `results.json` |
| `result.json`                | итог по каждой бутылке: кандидаты либо отказ с шагом и причиной, время шагов                                  |

Конфиг поиска — `--search-config`, по умолчанию `steps/vis_searcher/config.yaml`; его `debug_path` здесь не используется.
