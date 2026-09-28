# Признаки неизвестного вина

Код из ветки `whatis-service` (коммит `be9dcc1`) перенесён в этот пакет; импорты адаптированы к `vinishko.app.whatis`.

Из корня проекта, в активированном Python-окружении. Сначала настройте ключ и proxy при необходимости в корневом `.env` по [инструкции запуска](../README.md#запуск-всего-приложения).

```powershell
python -m uvicorn vinishko.app.whatis.main:app --env-file .env --host 127.0.0.1 --port 8810 --workers 1
```

Сервис работает отдельным процессом на `http://127.0.0.1:8810`; `--env-file .env` загружает настройки. Остановка — Ctrl+C.

`POST /v1/predict` принимает multipart-поле `image` (JPEG, PNG, WebP, GIF до 10 МиБ).
Ответ: `{"category":"Красное","brand":"Фанагория"}`. Неустановленный признак — `Не удалось определить`.
Ошибки: 413 — размер, 415 — формат, 502 — ошибка провайдера или ответа, 503 — не задан ключ.
`GET /health` проверяет процесс; документация — `/docs`, снимок — [docs/openapi.json](docs/openapi.json).

`main.py` — API; `models.py` — контракты; `service.py` — обработка результата.
`solution/` содержит модель, настройки и исходный промпт; `kb/catalog_values.json` — справочник категорий и виноделен.

Офлайн-проверки из корня:

```powershell
python -m unittest vinishko.app.whatis.tests.test_api vinishko.app.whatis.tests.test_solution_bundle
```
