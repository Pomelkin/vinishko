# API

FastAPI поверх пайплайна. Запуск из корня:

```bash
python -m vinishko.app --host 0.0.0.0 --port 8000        # по умолчанию 127.0.0.1:8000; --search-config, -c/--norm-config, --pipeline-config, --no-resolve
uvicorn vinishko.app.main:app --port 8000                 # то же с конфигами по умолчанию
```

Пайплайн поднимается при старте: нормализация, поиск, второй уровень и каталог по своим конфигам (`steps/normalization/normalize.toml`,
`steps/vis_searcher/config.yaml`, `steps/near_duplicates/config.yaml`, `pipeline/config.yaml`). Устройства — `NORMALIZER_DEV` и `VIS_SEARCHER_DEV`,
ключ модели второго уровня — `OPENROUTER_API_KEY`, ключ qdrant — `QDRANT_API_KEY`. Запросы к пайплайну идут по одному под замком:
SAM3 и engine TensorRT не рассчитаны на параллельные вызовы из одного процесса; параллелизм — несколько процессов.

Каталог по умолчанию — `vinishko/app/catalog.csv` (копия `datasets/hack-vine/catalog/catalog.csv`, путь в `pipeline/config.yaml`): читается
при старте целиком в память, строка позиции уходит в ответ у выбранной позиции и у каждого кандидата. При старте каждая позиция коллекции
поиска сверяется с каталогом; если хоть одной нет — сервис не поднимается (коллекция и каталог из разных выгрузок).

## `POST /recognize`

Multipart-поле `image`: jpeg, png, webp, heic. Ответ — бутылки, дошедшие до поиска, по убыванию скора отбора: с выбранной позицией либо
с отказом поиска или второго уровня, у всех маска. Объекты, которые нормализация не сочла целевой бутылкой с читаемой этикеткой, в список
не входят, их число — `ignored`. Координаты в пикселях фото после EXIF-поворота (`image.width`, `image.height`):

```json
{
  "image": {"width": 3024, "height": 4032},
  "bottles": [
    {
      "uuid": "…", "index": 1, "score": 0.98,
      "polygons": [[[x, y], …]], "label_polygons": [[[x, y], …]], "bbox": [x1, y1, x2, y2],
      "status": "matched",
      "match": {"slug": "…", "score": 0.83, "group": "…",
                "image_url": "/catalog/images/<slug>.jpg", "catalog": {"Название вина": "…", "Винодельня": "…", …},
                "source": "ndr_v5", "checklist": {"maker": {"observation": "…"}, "profile": {…}, "year": {…}}},
      "candidates": [], "rejection": null
    },
    {
      "uuid": "…", "index": 2, "score": 0.91, "polygons": [[[x, y], …]], "label_polygons": [[[x, y], …]], "bbox": [x1, y1, x2, y2],
      "status": "rejected", "match": null, "candidates": [],
      "rejection": {"stage": "search", "reason": "no_match", "label": "Нет в каталоге", "description": "…",
                    "detail": "средний косинус лучшей группы 0.612, порог 0.7", "message": "Нет в каталоге: средний косинус лучшей группы 0.612, порог 0.7"}
    }
  ],
  "ignored": 1,
  "timings_s": {"normalization": 0.41, "search": 0.05, "resolve": 2.3}
}
```

`status`: `matched` — выбрана одна позиция, `match.catalog` — строка каталога целиком, `match.source` — `vector` (в группе одна позиция)
либо `ndr_v5` (выбрала модель); `rejected` — отказ, `rejection.stage` говорит, какой шаг: `search` (нет похожих в каталоге) либо `resolve`
(модель не нашла точного совпадения в группе); `candidates` — сервис поднят с `--no-resolve`, кандидаты поиска как есть, у каждого те же
`slug`, `score`, `group`, `image_url` и `catalog`, что у `match`, без `source` и `checklist`. Фото, где ни одна
бутылка не дошла до поиска, — пустой список и `ignored` со счётом. Файл, который не читается как изображение, — 400.

## `GET /catalog/images/{name}`, `GET /health`

Картинка позиции из хранилища коллекции, JPEG; путь — `image_url` из ответа, имя сейчас `<slug>.<формат сборки>`. `/health` — коллекция, модель поиска и её входы, режим поиска,
модель второго уровня, источник и размер каталога.

## Проверка

`python -m vinishko.e2e --url http://127.0.0.1:8000` прогоняет тестовый набор через поднятый сервис и считает метрики классификации
итогового ответа; разбор шагов с дампами — `python -m vinishko.pipeline.evaluate_pipeline`.
