# Визуальный поиск

Шаг пайплайна после нормализации: кроп бутылки → вектор DinoV3ForWine → ближайшие векторы каталога в qdrant → ответ на каждый кроп
(`vinishko/pipeline/structs.py`): `BottleCandidates` с кандидатами `Candidate`, у каждого картинка позиции из коллекции для второго уровня, либо `UnmatchedBottle`
с отказом `RejectedBottle` и причиной `SearchReason`, когда ничего похожего нет. Ответов ровно столько, сколько кропов, и в том же порядке. Всё запускается из корня репозитория.

```python
from vinishko.pipeline.steps.vis_searcher import VisSearcher

searcher = VisSearcher()  # config.yaml рядом с модулем, устройство по VIS_SEARCHER_DEV; свой конфиг: VisSearcher(load_config(Path(...)))
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

Устройство задаёт переменная окружения `VIS_SEARCHER_DEV`: `cpu` либо `cuda:<индекс>`; без неё `cuda:0`, если CUDA доступна, иначе `cpu`.
Заданное проверяется: нет такой карты или нужного пакета — ошибка, а не тихий откат. На CUDA граф исполняет TensorRT
(пакет `tensorrt`, группа `flash-inference`), в bfloat16 на картах с его аппаратной поддержкой, Ampere и новее, иначе во float32;
engine собирается при первом запуске несколько минут и кэшируется в `cache_dir/engines/<репозиторий>/<ревизия>/`. На CPU —
OpenVINO во float32 (пакет `openvino`, группа `cpu-inference`), около секунды на картинку.

## Коллекция

```bash
python -m vinishko.pipeline.steps.vis_searcher.build_catalog \
  --csv datasets/hack/strapi/catalog_dataset.csv --images datasets/hack/strapi/img \
  --collection catalog_vitl16_512 --store-dir datasets/hack/catalog_crops        # либо --s3-endpoint … --s3-bucket … [--s3-prefix …], но не то и другое сразу
```

Каждое фото каталога проходит нормализацию, на нём должна найтись ровно одна годная бутылка (`--on-failure skip` пропускает остальные
и перечисляет их в конце, `--report` пишет JSON). Кроп уходит в хранилище картинок под именем `<slug>.jpg`, вектор — в qdrant.
Метаданные точки: `slug`, `group`, `group_slugs` — все позиции группы, попавшие в коллекцию, `image`, `source_image`, поля каталога
(`name`, `winery`, `vintage`, `abv`, …), `model`, `model_revision`, `input_size`, `precision`. Группа берётся из колонки `group` CSV;
без колонки каждая позиция — своя группа. Существующая коллекция пересоздаётся только после подтверждения в терминале; без терминала это ошибка.
`--qdrant-path` поднимает встроенный qdrant в директории вместо сервера.

## Поиск

`top_k` ближайших векторов, дальше по `search.mode`:

- `top_n` — кандидаты как есть; если косинус лучшего ниже `cosine_threshold`, отказ `no_match`.
- `groups` — позиции из выдачи собираются в группы по `group`, скор группы — лучший косинус её позиций. Берутся `top_groups` лучших групп
  с косинусом не ниже `group_threshold`; кандидатами идут все их позиции: пришедшие в выдачу со своим косинусом и `retrieved=True`,
  остальные по `group_slugs` с косинусом группы. Внутри группы сначала пришедшие по убыванию косинуса. Ни одна группа не прошла — отказ `no_match`.

При создании `VisSearcher` проверяет: qdrant отвечает, коллекция есть и не пуста, построена той же моделью и ревизией, размерность
и размер входа совпадают, картинка первой точки читается из хранилища. Картинки из S3 оседают в `cache_dir/images/`.

`debug_path` — директория для разбора глазами: на каждый вызов поддиректория с меткой времени, в ней `q<N>_query_<uuid>.jpg` — кроп
запроса, `q<N>_<ранг>_<slug>_<cos>.jpg` — кандидаты, в режиме групп `q<N>_<ранг>_<группа>_<slug>_<cos>[_bygroup].jpg`, и `results.json`.
