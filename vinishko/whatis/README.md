# Признаки неизвестного вина

Код из ветки `whatis-service` (коммит `be9dcc1`) перенесён в этот пакет; импорты адаптированы к `vinishko.whatis`.
Отдельным процессом не запускается: ручку `POST /whatis` подключает приложение [vinishko/app.py](../app.py),
настройки (ключ, proxy) — в [README приложения](../README.md).

## `POST /whatis`

Multipart-поле `image`: фото одной бутылки, JPEG, PNG, WebP или GIF до 10 МиБ.

```bash
curl --noproxy '*' -F 'image=@bottle.jpg' http://127.0.0.1:8000/whatis
```

Ответ HTTP 200: `{"category": "Красное", "brand": "Фанагория"}`. `Не удалось определить` — штатный отказ модели по признаку.
Неверный формат — 415, слишком большой файл — 413, ошибка провайдера или невалидный ответ — 502, нет ключа — 503.
Вызов из `POST /recognize` для бутылок с отказом — флаг `AUTO_WHATIS`, см. [README распознавания](../pred/README.md).

`router.py` — ручка; `models.py` — контракты; `service.py` — обработка результата.
`solution/` содержит модель, настройки и исходный промпт; `kb/catalog_values.json` — справочник категорий и виноделен.

Офлайн-проверки из корня:

```bash
python -m unittest vinishko.whatis.tests.test_api vinishko.whatis.tests.test_solution_bundle
```
