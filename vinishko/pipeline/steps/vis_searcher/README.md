# Визуальный поиск

Шаг пайплайна после нормализации: кроп бутылки → вектор DinoV3ForWine → ближайшие векторы каталога в qdrant → ответ на каждый кроп
(`vinishko/pipeline/structs.py`): `BottleCandidates` с кандидатами `Candidate`, у каждого картинка позиции из коллекции для второго уровня, либо `UnmatchedBottle`
с отказом `RejectedBottle` и причиной `SearchReason`, когда ничего похожего нет. Ответов ровно столько, сколько кропов, и в том же порядке. Всё запускается из корня репозитория.

```python
from vinishko.pipeline.steps.vis_searcher import VisSearcher

searcher = VisSearcher()  # config.yaml рядом с модулем, устройство из конфига или VIS_SEARCHER_DEV; свой конфиг: VisSearcher(load_config(Path(...)))
results = searcher(
    crops
)  # crops: list[BottleCrop] от нормализации → list[BottleCandidates | UnmatchedBottle], у каждого .candidates либо .rejected
```

`Pipeline(searcher=VisSearcher())` из `vinishko/pipeline/pipeline.py` связывает шаги сам; нормализатор он поднимает как `Normalizer()`,
а бутылку с отказом поиска подставляет в разметку `RejectedBottle` с тем же `uuid`.

## Модель и устройство

Модель — репозиторий Hugging Face с экспортом обучения: `config.json`, `preprocess.json`, `model.onnx` (float32) и `model.bf16.onnx`.
Предобработка целиком по `preprocess.json`: кроп вписывается в `input_size` с сохранением пропорций, поля цвета `pad_color` по центру,
float32 в 0…255, нормировка ImageNet внутри графа.

Устройство — поле `device` конфига: `auto`, `cpu` либо `cuda:<индекс>`; переменная окружения `VIS_SEARCHER_DEV` его перекрывает. `auto` — `cuda:0`, если CUDA доступна, иначе `cpu`.
Заданное проверяется: нет такой карты или нужного пакета — ошибка, а не тихий откат. На CUDA граф исполняет TensorRT
(пакет `tensorrt`, группа `flash-inference`), в bfloat16 на картах с его аппаратной поддержкой, Ampere и новее, иначе во float32;
engine собирается при первом запуске несколько минут и кэшируется в `cache_dir/engines/<репозиторий>/<ревизия>/`. На CPU —
OpenVINO во float32 (пакет `openvino`, группа `cpu-inference`), около секунды на картинку.

## Другая модель

Модель поиска меняется строкой `model` в `config.yaml`: репозиторий Hugging Face либо локальная директория с тем же набором файлов
(`config.json` с `embed_dim`, `preprocess.json`, `model.onnx`, `model.bf16.onnx`). Контракт один: вход float32 RGB NCHW 0…255, нормировка
внутри графа, выход L2-нормированные эмбеддинги; как готовить вход, говорит `preprocess.json`: `input_size`, `resize` (`pad` — вписать
с полями цвета `pad_color`, `squash` — растянуть без полей), `mean` и `std` только для справки. Коллекция помнит модель и ревизию,
поэтому после смены модели её надо пересобрать.

Экспорт визуальной башни TULIP из форка open_clip в этот формат: `python -m scripts.export_tulip -o weights/tulip_so400m_14_384` (чекпоинт
`weights/tulip-so400m-14-384.ckpt`, вход 384×384, `resize: squash`, как в замерах `scripts/bench_tulip`). Локальные экспорты лежат в `weights/`:
`dinov3_vitl16_512`, `dinov3_vitl16_1024`, `tulip_so400m_14_384`; в `model` можно указать репозиторий HF либо путь к такой директории, относительный
путь считается от файла конфига, например `../../../../weights/tulip_so400m_14_384`.

## Коллекция

```bash
python -m vinishko.pipeline.steps.vis_searcher.build_catalog \
  --csv datasets/hack-vine/catalog/catalog.csv --images datasets/hack-vine/catalog/images \
  --photo-column image_filename --group-column near_duplicate_group_slug --on-failure skip
```

Модель и ревизия, qdrant с именем коллекции, хранилище кропов, `batch_size` и `cache_dir` берутся из того же `config.yaml`
(`--config` — другой файл), с которым потом ищет `VisSearcher`: собрать одной моделью, а искать другой нельзя по построению.
Устройство энкодера — `VIS_SEARCHER_DEV`, нормализации — `NORMALIZER_DEV`.

Каждое фото каталога проходит нормализацию, на нём должна найтись ровно одна годная бутылка (`--on-failure skip` пропускает остальные
и перечисляет их в конце, `--report` пишет JSON). Вектор считается с кропа поиска (`BottleCrop.crop`, окно этикетки с залитым фоном), а в хранилище картинок под именем `<slug>.jpg` уходит вся
бутылка каталожного фото (`BottleCrop.box_crop`, bbox маски с запасом, фон не тронут): её получает второй уровень как `Candidate.image`,
она того же вида, что `box_crop` запроса.
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

При создании `VisSearcher` проверяет: qdrant отвечает, коллекция есть и не пуста, построена той же моделью и ревизией, размерность
и размер входа совпадают, картинка первой точки читается из хранилища. Картинки из S3 оседают в `cache_dir/images/`.

`debug_path` — директория для разбора глазами: на каждый вызов поддиректория с меткой времени, в ней `q<N>_query_<uuid>.jpg` — кроп
запроса для поиска, `q<N>_query_<uuid>_box.jpg` — вся бутылка для второго уровня, `q<N>_<ранг>_<slug>_<cos>.jpg` — кандидаты, в режиме групп `q<N>_<ранг>_<группа>_<slug>_<cos>[_bygroup].jpg`, и `results.json`.

## Оценка на тесте

```bash
python -m vinishko.pipeline.steps.vis_searcher.evaluate --test-dir datasets/hack-vine/test        # test.csv: image_filename, slug
```

Каждое фото проходит нормализацию и поиск по `config.yaml`; ответ — кандидаты бутылки с лучшим скором отбора. По фото, чей slug есть
в коллекции, считаются recall@1/3/5 и `group_recall` (slug состоит в группе какого-то кандидата); отдельно `recall@k_any_bottle` по всем
бутылкам фото, для полок. Пустой slug значит, что ответа нет: `correct_reject` — поиск отказал, `false_accept` — выдал кандидатов;
`false_reject` — ответ был, а поиск отказал; `no_bottle` — нормализация не нашла годной бутылки. В отчёт идут квантили косинуса top-1
для попаданий, промахов и ложных принятий — по ним калибруются пороги. Результат в `reports/vis_searcher/<время>/`: `report.json`,
`per_image.csv` и дампы поиска по промахам (`--dump all|none`).
