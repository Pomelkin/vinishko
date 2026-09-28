# AI-сомелье

Код сервиса из ветки `ai-somelier-service` (коммит `04bf9af`) перенесён в этот пакет; импорты адаптированы к `vinishko.app.sommelier`.

Из корня проекта, в активированном Python-окружении. Сначала настройте корневой `.env` по [инструкции запуска](../README.md#запуск-всего-приложения): ключ, proxy при необходимости и `SOMMELIER_DATA_DIR=vinishko/app/sommelier/data/sessions`.

```powershell
python -m uvicorn vinishko.app.sommelier.main:app --env-file .env --host 127.0.0.1 --port 8805 --workers 1
```

Сервис работает отдельным процессом на `http://127.0.0.1:8805`; `--env-file .env` загружает настройки.
Сессии сохраняются по `SOMMELIER_DATA_DIR`; без настройки — в `data/sessions/`. Используйте один worker. Остановка — Ctrl+C.

`POST /v1/sessions/{UUIDv7}` принимает `wine` и `candidates` (0–5 карточек), возвращает первое сообщение.
`POST /v1/sessions/{UUIDv7}/messages` принимает `content` (1–4000 символов) и возвращает ответ.
`GET /v1/sessions/{UUIDv7}` читает историю, `DELETE` удаляет диалог. Документация — `/docs`, снимок — [docs/openapi.json](docs/openapi.json).

`main.py` — API; `models.py` — контракты; `service.py` — диалоги; `storage.py` — атомарное хранение JSON.
`solution/` содержит модель, конфигурацию и промпты; `kb/` — экспертную базу; `docs/` — её шаблон и OpenAPI.

Офлайн-проверки из корня:

```powershell
python -m unittest vinishko.app.sommelier.tests.test_api vinishko.app.sommelier.tests.test_solution_bundle
```
