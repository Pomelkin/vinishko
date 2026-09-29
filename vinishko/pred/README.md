# Распознавание

Ручки пайплайна в [router.py](router.py), схемы ответа — [schemas.py](schemas.py), сам пайплайн — [pipeline/](pipeline/README.md).
Подключает их приложение [vinishko/app.py](../app.py), запуск и окружение — [README приложения](../README.md).

Каталог по умолчанию — `vinishko/pred/catalog.csv` (копия `datasets/hack-vine/catalog/catalog.csv`, путь в `pipeline/config.yaml`): читается
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
                "source": "final", "checklist": {"maker": {"observation": "…"}, "profile": {…}, "year": {…}}},
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

`status`: `matched` — выбрана одна позиция, `match.catalog` — строка каталога целиком, `match.source` — `group` (модель выбрала позицию
внутри её группы, других финалистов не было) либо `final` (модель выбрала её среди лучших позиций групп и одиночек либо подтвердила
единственную позицию); `rejected` — отказ, `rejection.stage` говорит, какой шаг: `search` (нет похожих в каталоге) либо `resolve`
(модель не нашла точного совпадения среди кандидатов); `candidates` — сервис поднят с `--no-resolve`, кандидаты поиска как есть, у каждого те же
`slug`, `score`, `group`, `image_url` и `catalog`, что у `match`, без `source` и `checklist`. Фото, где ни одна
бутылка не дошла до поиска, — пустой список и `ignored` со счётом. Файл, который не читается как изображение, — 400.

## `GET /catalog/images/{name}`, `GET /health`

Картинка позиции из хранилища коллекции, JPEG; путь — `image_url` из ответа, имя сейчас `<slug>.<формат сборки>`. `/health` — коллекция, модель поиска и её входы, режим поиска,
модель второго уровня, источник и размер каталога.

## Проверка

`python -m vinishko.e2e --url http://127.0.0.1:8000` прогоняет тестовый набор через поднятый сервис и считает метрики классификации
итогового ответа; разбор шагов с дампами — `python -m vinishko.pred.pipeline.evaluate_pipeline`.

## Признаки вина вне каталога в ответе `/recognize`

`AUTO_WHATIS=false` в окружении (по умолчанию) отключает автоматические вызовы whatis. При `AUTO_WHATIS=true` (после перезапуска)
`POST /recognize` отправляет в whatis JPEG-кроп **всей бутылки** (`box_crop`) по каждой бутылке с отказом поиска или второго уровня,
по очереди. Для найденных позиций, кандидатов без второго уровня и объектов, отброшенных нормализацией, whatis не вызывается.
Каждый вызов расходует токены OpenRouter и увеличивает время ответа. Исходные `status`, `rejection` и `match` сохраняются, у бутылки
добавляются поля:

```json
{
  "unknown_wine": {"category": "Красное", "brand": "Фанагория"},
  "unknown_wine_error": null
}
```

Если whatis ответил ошибкой, распознавание всё равно возвращает HTTP 200 с исходным результатом, `unknown_wine` — `null`,
`unknown_wine_error` — код и `detail`, например `{"status_code": 502, "detail": "Recognition is temporarily unavailable"}`.
У остальных бутылок оба поля `null`. Общее время вызовов whatis — `timings_s.whatis`.

## Контракт интерфейса

`/api/recognize` использует тот же пайплайн, нормализует маски и добавляет до пяти похожих кандидатов. `/api/wines` и `/api/wines/{slug}` предоставляют каталог; `/api/catalog/images/{name}` — изображения. Полный контракт — [frontend/API.md](../../frontend/API.md). Исходный `/recognize` сохранён. Лимит фото — 20 МиБ/80 МП, число бутылок задаётся `MAX_BOTTLES` (по умолчанию 10) или `--max-bottles`.
