# AI-сомелье

Код сервиса из ветки `ai-somelier-service` (коммит `04bf9af`) перенесён в этот пакет; импорты адаптированы к `vinishko.sommelier`.
Отдельным процессом не запускается: ручки `/sommelier/sessions/...` подключает приложение [vinishko/app.py](../app.py),
настройки (ключ, proxy, `SOMMELIER_DATA_DIR`) — в [README приложения](../README.md).

## `/sommelier/sessions/{session_id}` — чат с сомелье

Клиент создаёт новый UUIDv7 для каждого диалога. Передайте `match.catalog` из `/recognize` как `wine`;
`candidates` — обязательный список из 0–5 альтернативных карточек (можно взять `candidate.catalog`
из ответа поиска без второго уровня). Карточки передаются без переименования колонок.

```http
POST /sommelier/sessions/01993959-0000-7000-8000-000000000001
Content-Type: application/json

{"wine": {"Название вина": "Кокур", "Категория": "Белое"}, "candidates": []}
```

Ответ HTTP 201:

```json
{"session_id": "01993959-0000-7000-8000-000000000001", "message": {"role": "assistant", "content": "Описание и совет по подаче", "suggestions": ["Как подавать?"]}}
```

Далее `POST /sommelier/sessions/{session_id}/messages` с `{"content": "С чем подавать?"}`
(1–4000 символов после удаления пробелов по краям). Ответ HTTP 201 того же вида, `suggestions` — `null`.
`GET /sommelier/sessions/{session_id}` — HTTP 200 с `{"session_id": "…", "messages": [...]}`.
`DELETE /sommelier/sessions/{session_id}` — HTTP 204 без тела. Диалог хранится JSON-файлом в `SOMMELIER_DATA_DIR`, запросы
одной сессии выполняются по очереди. Ошибки: 404 — нет сессии, 409 — UUID уже занят, 422 — невалидные данные,
502 — сбой провайдера, 503 — нет ключа. Автоматических повторов нет.

`router.py` — ручки; `models.py` — контракты; `service.py` — диалоги; `storage.py` — атомарное хранение JSON.
`solution/` содержит модель, конфигурацию и промпты; `kb/` — экспертную базу; `docs/` — её шаблон.

Офлайн-проверки из корня:

```bash
python -m unittest vinishko.sommelier.tests.test_api vinishko.sommelier.tests.test_solution_bundle
```
