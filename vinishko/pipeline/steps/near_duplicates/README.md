# Выбор SKU внутри near-duplicates

Код `config.py`, `models.py`, `predictor.py` и три prompt-файла перенесены из `ndr/solutions/v5` исходного репозитория. Адаптация принимает изображения в памяти и подключает решение к `Pipeline` через `NearDuplicateReranker`.

Векторный поиск возвращает группу позиции с максимальным косинусом. Реранкер проверяет, что получены все `group_slugs`. Для группы из одного SKU он сразу возвращает этот SKU. Для группы из нескольких SKU он делает один запрос к OpenRouter: QUERY — `BottleCrop.box_crop` (`images_crop_box` из парной S3-выборки или кроп всей бутылки после SAM3), ELEMENT — кропы кандидатов из `images`, если точка Qdrant помечена `image_crop=bottle_box`; для старых точек используются оригинальные эталоны из `reference_images`. Поля карточек берутся из локального `catalog_csv`. Ответ строго ограничен slug этой группы или `not_found`.

При `not_found` бутылка становится `UnmatchedBottle` с причиной `near_duplicate_not_found`. Ошибка API или контракта прерывает запрос с `NearDuplicateError`; решение не подменяется top-1. Краткое обоснование лежит в `BottleCandidates.selection` или `UnmatchedBottle.selection`. При запуске `python -m vinishko.pipeline.debug` полный trace модели записывается в `runs/<фото>/rerank/<uuid>.json` без base64 изображений и без API-ключа.

Модель, параметры генерации, маршрутизация провайдера и таймаут находятся в `config.py`. Ключ берётся из `OPENROUTER_API_KEY`, модель можно перекрыть `OPENROUTER_MODEL`. Путь к готовым изображениям кандидатов и CSV — в конфиге визуального поиска. В режиме `--skip-normalization --normalized-dir datasets/local/normalized/test` QUERY для NDR берётся из `images_crop_box` как при полном прогоне, так и при отдельном `--stage ndr` после поиска. Вызовы модели выполняются параллельно с общим семафором на 32 запроса.

Офлайн-проверка, которая не обращается к Qdrant, S3 или модели:

```powershell
.venv/Scripts/python.exe -B -m unittest vinishko.pipeline.steps.near_duplicates.test_rerank -v
```
