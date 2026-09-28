# Pipeline SigLIP2

Фото или сохранённый box-кроп → [SigLIP2 ONNX](https://huggingface.co/pomelk1n/siglipus-twinturbo-gguf-umer/tree/main) → Qdrant → кандидаты.
Нормализация/SAM3, near_duplicates и vanilla_vlm_rerank не импортируются и не выполняются.

Из корня проекта, в окружении с зависимостями проекта:

```powershell
python -m vinishko.pipeline-siglip.debug
```

В начале `debug.py` задаются `INPUT_PATH` (фото, папка или датасет с CSV),
`INPUT_MODE`, `BOX_RUNS_DIR`, `OUTPUT_DIR`, `LIMIT`, `BATCH_SIZE`, `DROP_NA`.
`images_crop_box` читает сохранённые SAM3 `*_b1_box.jpg` из
`BOX_RUNS_DIR/<имя фото>/normalization/` и подаёт их в SigLIP. Фото без
box-кропа получает пустой `result.json`; `raw` подаёт целое фото.
Аргументов CLI нет.
Результаты каждого запуска — в новой подпапке OUTPUT_DIR: `result.json` и
`search/` с запросом, изображениями кандидатов, косинусами и `results.json`.

Настройки модели, устройства, top-k, порога, Qdrant и картинок — в
`steps/vis_searcher/configs.py`. По умолчанию: `127.0.0.1:6333`,
`catalog_siglip2_dense_new`, top-10, порог −1 (всегда вернуть ближайших).
Картинки читаются из `datasets/local/catalog/images` по `payload.photo`.
Серверный proxy из окружения для Qdrant отключён; ключ OpenRouter не нужен.

На CUDA используется существующий TensorRT, на CPU — OpenVINO
(`uv sync --group cpu-inference`). Первый запуск скачивает ONNX через
Hugging Face и собирает engine; следующие используют кэш. `VIS_SEARCHER_DEV`
перекрывает DEVICE. Кэш engine — CACHE_DIR, кэш скачивания — стандартный
Hugging Face (можно задать HF_HOME до запуска).

Из Python (дефис в имени каталога требует importlib):

```python
from importlib import import_module

siglip = import_module("vinishko.pipeline-siglip")
searcher = siglip.VisSearcher()
try:
    pipeline = siglip.Pipeline(searcher)
    result = pipeline("photo.jpg")
    results = pipeline.run_many(["photo.jpg", "other.jpg"])
    print([(c.slug, c.score) for c in result.search[0].candidates])
finally:
    searcher.close()
```

Ответы используют исходные `BottleCrop`, `Candidate`, `BottleCandidates`,
`UnmatchedBottle`. Сохранены порядок батча, callbacks `on_result`, `items`,
`crops`, UUID отказов, пороги и дампы поиска. Поле `result.normalization`
оставлено для совместимости: там одно целое фото, этапа нормализации нет.
Время загрузки фото — `raw_input`, поиска — `search`. Если порог не пройден,
вместо `.candidates` возвращается `UnmatchedBottle` с `.rejected`.

В режиме raw вход повторяет DenseRetriever: RGB без EXIF-поворота, масок и выделения бутылок.
Затем bilinear resize 384×384 без полей. Нормировка пикселей и L2 выхода
встроены в экспорт; отдельно используется страховочная L2-нормировка энкодера.
В коллекции нужны безымянные dense-векторы 1152 с Cosine и payload `slug`, `photo`.
У старой dense-коллекции нет метаданных модели: совпадение размерности не
доказывает происхождение векторов. Если метаданные есть, несовпадение — ошибка.
Коллекция не пересобирается и не изменяется.

Проверка без скачивания весов и сервера, на Qdrant в памяти:

```powershell
python -m vinishko.pipeline-siglip.test_pipeline
```
