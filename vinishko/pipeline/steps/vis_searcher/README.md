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

Модель — репозиторий Hugging Face или локальная директория экспорта: `config.json`, `preprocess.json`, `model.onnx` (float32) и `model.bf16.onnx`.
Предобработка целиком по `preprocess.json`: `resize: pad` вписывает кроп с полями цвета `pad_color`, `resize: squash` растягивает его до `input_size`; без поля используется `pad`. Вход float32 в 0…255, нормировка внутри графа. Внешние данные ONNX (`.onnx.data`) также скачиваются из Hugging Face.

Устройство — поле `device` конфига: `auto`, `cpu` либо `cuda:<индекс>`; переменная окружения `VIS_SEARCHER_DEV` его перекрывает. `auto` — `cuda:0`, если CUDA доступна, иначе `cpu`.
Заданное проверяется: нет такой карты или нужного пакета — ошибка, а не тихий откат. На CUDA граф исполняет TensorRT
(пакет `tensorrt`, группа `flash-inference`), в bfloat16 на картах с его аппаратной поддержкой, Ampere и новее, иначе во float32;
engine собирается при первом запуске несколько минут и кэшируется в `cache_dir/engines/<репозиторий>/<ревизия>/`. На CPU —
OpenVINO во float32 (пакет `openvino`, группа `cpu-inference`). На Ryzen 7 5825U новая модель дала эмбеддинг 1024 длины за 2,38 с на одном готовом кропе; первый запуск дополнительно скачал ONNX-граф размером 1,22 ГБ и скомпилировал его. `cpu_batch_size` ограничивает число изображений в одном вызове OpenVINO (в текущем конфиге — 2).

## Другая модель

Модель поиска меняется строкой `model` в `config.yaml`: репозиторий Hugging Face либо локальная директория с тем же набором файлов
(`config.json` с `embed_dim`, `preprocess.json`, `model.onnx`, `model.bf16.onnx`). Контракт один: вход float32 RGB NCHW 0…255, нормировка
внутри графа, выход L2-нормированные эмбеддинги; как готовить вход, говорит `preprocess.json`: `input_size`, `resize` (`pad` — вписать
с полями цвета `pad_color`, `squash` — растянуть без полей), `interpolation` (`area_cubic`, `bilinear`, `bicubic`), `mean` и `std` только для справки. Коллекция помнит модель и ревизию,
поэтому после смены модели её надо пересобрать.

Экспорт визуальной башни TULIP из форка open_clip в этот формат: `python -m scripts.export_tulip -o weights/tulip_so400m_14_384` (чекпоинт
`weights/tulip-so400m-14-384.ckpt`, вход 384×384, `resize: squash`, как в замерах `scripts/bench_tulip`). SigLIP2 с Hugging Face: `python -m scripts.export_siglip2 -o weights/siglip2_so400m_14_384` (`google/siglip2-so400m-patch14-384`,
эмбеддинг — выход MAP-головы, вход 384×384, `resize: squash`). Общий код обоих экспортов — `scripts/export_common.py`.
Локальные экспорты лежат в `weights/`: `dinov3_vitl16_512`, `dinov3_vitl16_1024`, `tulip_so400m_14_384`, `siglip2_so400m_14_384`; в `model` можно указать репозиторий HF либо путь к такой директории, относительный
путь считается от файла конфига, например `../../../../weights/tulip_so400m_14_384`.

## Готовая коллекция и текущий поиск

Текущий `config.yaml` настроен на модель `vitl16-1024` и коллекцию `catalog_vitl16_1024` на `pomelka.pro`. При обычном запуске нормализатор создаёт `crop` этикетки для энкодера и `box_crop` всей бутылки для сохранения и NDR. При `--skip-normalization --normalized-dir datasets/local/normalized/test` энкодер получает JPEG из `images_crop`, а NDR — JPEG из `images_crop_box`. `images` указывает на локальную копию `vino/catalog/` в `datasets/local/normalized/catalog/images`; по ней поиск читает кандидатов. Для старых точек без `image_crop=bottle_box` NDR использует исходные эталоны из `reference_images`. Карточки берутся из `datasets/local/catalog/catalog.csv`.

Из S3 скачаны 176 пар JPEG теста из `test-data-norm/norm-images.tar.gz` и 1981 JPEG каталога (префикс `catalog/`). Поиск требует `QDRANT_API_KEY`; без ключа удалённый сервер отвечает 401, поэтому наличие и содержимое новой коллекции локально пока не проверены.

## Сборка новой коллекции

Сборка — отдельный запуск. `--images` ниже указывает **входную папку с исходными фото** из CSV; поле `images` в YAML указывает **выходное хранилище кропов всей бутылки** (локальную директорию или S3). Скопируйте `config.yaml` рядом как `config.build.yaml`, задайте в копии отдельные директорию и имя коллекции, чтобы не затронуть рабочий каталог. Относительные пути в YAML считаются от расположения файла. Для поиска по новой коллекции модель, её ревизия и имя коллекции должны совпасть с настройками сборки.

Например, для локальной сборки замените в копии секции `qdrant` и `images`:

```yaml
qdrant:
  collection: catalog_vitl16_1024_local
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

Модель и ревизия, Qdrant, хранилище кропов, `batch_size` и `cache_dir` берутся из YAML, переданного через `--config`. Во время запроса используются картинки из `images`; `reference_images` остаётся резервом для старых точек Qdrant.
Устройство энкодера — `VIS_SEARCHER_DEV`, нормализации — `NORMALIZER_DEV`.

Каждое фото каталога проходит нормализацию, на нём должна найтись ровно одна годная бутылка (`--on-failure skip` пропускает остальные
и перечисляет их в конце, `--report` пишет JSON). Вектор считается с кропа, заданного `encoder_input` конфига: `crop` — окно этикетки с залитым фоном, вход обучения DINO, либо `box_crop` —
вся бутылка как на фото, для TULIP и подобных; коллекция помнит выбор и с другим значением не примется. В хранилище картинок под именем
`<slug>.jpg` всегда уходит вся бутылка каталожного фото (`BottleCrop.box_crop`, bbox маски с запасом, фон не тронут): её получает второй уровень как `Candidate.image`,
она того же вида, что `box_crop` запроса, который получает NDR.
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
запроса для поиска, `q<N>_query_<uuid>_box.jpg` — кроп бутылки для просмотра (VLM получает исходное фото), `q<N>_<ранг>_<slug>_<cos>.jpg` — кандидаты, в режиме групп `q<N>_<ранг>_<группа>_<slug>_<cos>[_bygroup].jpg`, и `results.json`.

## Оценка на тесте

```bash
python -m vinishko.pipeline.steps.vis_searcher.evaluate --test-dir datasets/hack-vine/test        # test.csv: image_filename, slug
```

Каждое фото проходит нормализацию и поиск по `config.yaml`; ответ — кандидаты бутылки с лучшим скором отбора. По фото, чей slug есть
в коллекции, считаются recall@1/3/5 и `group_recall` (slug состоит в группе какого-то кандидата); отдельно `recall@k_any_bottle` по всем
бутылкам фото, для полок. Пустой slug значит, что ответа нет: `correct_reject` — поиск отказал, `false_accept` — выдал кандидатов; slug задан, но в коллекции
его нет (фото каталога не прошло нормализацию) — `answer_not_indexed`, такие фото ни в recall, ни в отказ не идут;
`false_reject` — ответ был, а поиск отказал; `no_bottle` — нормализация не нашла годной бутылки. В отчёт идут квантили косинуса top-1
для попаданий, промахов и ложных принятий — по ним калибруются пороги. Результат в `reports/vis_searcher/<время>/`: `report.json`,
`per_image.csv` и `dumps/<фото>/` на каждое фото (`--dump misses` — только промахи, `none` — без дампов): копия исходника, `normalization.json`
со всеми бутылками и причинами отказов, а если поиск был — оба кропа запроса, картинки кандидатов и `results.json`.
