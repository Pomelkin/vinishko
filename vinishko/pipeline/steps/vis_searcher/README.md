# Визуальный поиск

Шаг пайплайна после нормализации: кроп бутылки → вектор DinoV3ForWine → ближайший вектор каталога в qdrant → вся его группа → ответ на каждый кроп
(`vinishko/pipeline/structs.py`): `BottleCandidates` с кандидатами `Candidate`, у каждого картинка позиции для второго уровня, либо `UnmatchedBottle`
с отказом `RejectedBottle` и причиной `SearchReason`, когда ничего похожего нет. Ответов ровно столько, сколько кропов, и в том же порядке. Всё запускается из корня репозитория.

```python
from vinishko.pipeline.steps.vis_searcher import VisSearcher

searcher = VisSearcher()  # config.yaml рядом с модулем, устройство из конфига или VIS_SEARCHER_DEV; свой конфиг: VisSearcher(load_config(Path(...)))
results = searcher(
    crops
)  # crops: list[BottleCrop] от нормализации → list[BottleCandidates | UnmatchedBottle], у каждого .candidates либо .rejected
```

`Pipeline(searcher=VisSearcher())` из `vinishko/pipeline/pipeline.py` связывает шаги сам: нормализация, поиск top-1 группы и NDR v5 внутри неё. Бутылка с отказом поиска или NDR попадает в разметку как `RejectedBottle` с тем же `uuid`.

## Модель и устройство

Модель — репозиторий Hugging Face с экспортом обучения: `config.json`, `preprocess.json`, `model.onnx` (float32) и `model.bf16.onnx`.
Предобработка целиком по `preprocess.json`: кроп вписывается в `input_size` с сохранением пропорций, поля цвета `pad_color` по центру,
float32 в 0…255, нормировка ImageNet внутри графа.

Устройство — поле `device` конфига: `auto`, `cpu` либо `cuda:<индекс>`; переменная окружения `VIS_SEARCHER_DEV` его перекрывает. `auto` — `cuda:0`, если CUDA доступна, иначе `cpu`.
Заданное проверяется: нет такой карты или нужного пакета — ошибка, а не тихий откат. На CUDA граф исполняет TensorRT
(пакет `tensorrt`, группа `flash-inference`), в bfloat16 на картах с его аппаратной поддержкой, Ampere и новее, иначе во float32;
engine собирается при первом запуске несколько минут и кэшируется в `cache_dir/engines/<репозиторий>/<ревизия>/`. На CPU —
OpenVINO во float32 (пакет `openvino`, группа `cpu-inference`), около секунды на картинку.

## Готовая коллекция и текущий поиск

Текущий `config.yaml` настроен на коллекцию `catalog_vitl16_512` на `pomelka.pro`. При обычном запуске нормализатор создаёт кроп запроса из исходного фото; при повторном поиске с `--skip-normalization` берётся сохранённый кроп. Энкодер считает его вектор, Qdrant находит ближайшие векторы каталога и возвращает оценки и метаданные. `images: null` означает, что поиск **не читает кропы каталога**: картинка кандидата берётся из локального `reference_images` по полю `source_image` в метаданных точки. Сейчас это исходные фото `datasets/local/catalog/images`; NDR получает их же и карточки из `datasets/local/catalog/catalog.csv`. Исходные фото теста лежат отдельно в `datasets/local/test/images`. В этих папках кропы не требуются, и S3 при запросе не используется.

Адрес S3 `vino/catalog/vitl16_512` в комментарии `config.yaml` — пример возможного хранилища кропов. Он не доказывает, что кропы существующей удалённой коллекции лежат там. Qdrant хранит векторы, метаданные `image` и `source_image`, но не сами JPEG-кропы.

## Сборка новой коллекции

Сборка — отдельный запуск. `--images` ниже указывает **входную папку с исходными фото** из CSV; поле `images` в YAML указывает **выходное хранилище кропов** (локальную директорию или S3). С текущим `images: null` сборщик останавливается до обработки каталога. Скопируйте `config.yaml` рядом как `config.build.yaml`, задайте в копии `images` и нужную коллекцию Qdrant. Относительные пути в YAML считаются от расположения файла. Для поиска по новой коллекции модель, её ревизия и имя коллекции должны совпасть с настройками сборки.

Например, для локальной сборки замените в копии секции `qdrant` и `images`:

```yaml
qdrant:
  collection: catalog_vitl16_512_local
  path: ../../../../.qdrant
images:
  kind: local
  dir: ../../../../datasets/local/catalog_crops
```

```bash
python -m vinishko.pipeline.steps.vis_searcher.build_catalog \
  --csv datasets/local/catalog/catalog.csv --images datasets/local/catalog/images \
  --photo-column image_filename --group-column near_duplicate_group_slug \
  --config vinishko/pipeline/steps/vis_searcher/config.build.yaml --on-failure skip
```

Модель и ревизия, Qdrant, хранилище кропов, `batch_size` и `cache_dir` берутся из YAML, переданного через `--config`. Для поиска по новой коллекции можно использовать другой YAML с `images: null` и `reference_images`, если модель, ревизия и коллекция совпадают. При такой настройке сохранённые кропы не нужны во время запроса, но они были нужны сборщику для записи каталога.
Устройство энкодера — `VIS_SEARCHER_DEV`, нормализации — `NORMALIZER_DEV`.

Каждое фото каталога проходит нормализацию, на нём должна найтись ровно одна годная бутылка (`--on-failure skip` пропускает остальные
и перечисляет их в конце, `--report` пишет JSON). Кроп уходит в хранилище картинок под именем `<slug>.jpg`, вектор — в qdrant.
Метаданные точки: `slug`, `group`, `group_slugs` — все позиции группы, попавшие в коллекцию, `image`, `source_image`, поля каталога
(`name`, `winery`, `vintage`, `abv`, …), `model`, `model_revision`, `input_size`, `precision`. Группа берётся из колонки `--group-column`;
пустое значение или отсутствие колонки — позиция сама себе группа. Дешёвые проверки идут до загрузки моделей: колонки CSV и дубли slug,
наличие всех фото, qdrant отвечает и коллекции ещё нет, хранилище доступно (директория с правами на запись либо `head_bucket`).
Существующая коллекция пересоздаётся только после подтверждения в терминале; без терминала это ошибка. `qdrant.path` в конфиге
поднимает встроенный qdrant в директории вместо сервера.

## Поиск

`top_k` ближайших векторов, дальше по `search.mode`:

- `top_n` — кандидаты как есть; если косинус лучшего ниже `cosine_threshold`, отказ `no_match`.
- `groups` — позиции из выдачи собираются в группы по `group`, скор группы — лучший косинус её позиций. Берутся `top_groups` лучших групп
  с косинусом не ниже `group_threshold`; кандидатами идут все их позиции: пришедшие в выдачу со своим косинусом и `retrieved=True`,
  остальные по `group_slugs` с косинусом группы. Внутри группы сначала пришедшие по убыванию косинуса. Ни одна группа не прошла — отказ `no_match`.

При создании `VisSearcher` проверяет: Qdrant отвечает, коллекция есть и не пуста, построена той же моделью и ревизией, размерность
и размер входа совпадают, картинка первой точки читается из выбранного источника (`images` либо `reference_images`). При чтении картинок из S3 они оседают в `cache_dir/images/`.

`debug_path` — директория для разбора глазами: на каждый вызов поддиректория с меткой времени, в ней `q<N>_query_<uuid>.jpg` — кроп
запроса, `q<N>_<ранг>_<slug>_<cos>.jpg` — кандидаты, в режиме групп `q<N>_<ранг>_<группа>_<slug>_<cos>[_bygroup].jpg`, и `results.json`.
