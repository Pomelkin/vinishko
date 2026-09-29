# vinishko: приложение

Один процесс FastAPI, все ручки в [app.py](app.py). Каждый сервис — свой модуль со своим `router.py`:

```text
vinishko/
  app.py         # приложение: подключает роутеры, при старте поднимает пайплайн и сервисы; CLI запуска
  pred/          # распознавание: router.py — /recognize, /catalog/images/{name}, /health; schemas.py; pipeline/ — шаги пайплайна
  sommelier/     # router.py — /sommelier/sessions/...; диалог с сомелье по карточке вина
  whatis/        # router.py — /whatis; категория и винодельня бутылки вне каталога
  e2e.py         # сквозная проверка через поднятый сервис
```

Ручки: распознавание — [pred/README.md](pred/README.md), сомелье — [sommelier/README.md](sommelier/README.md),
whatis — [whatis/README.md](whatis/README.md). Схема целиком — `/docs` и `/openapi.json` поднятого приложения.

Запуск из корня:

```bash
python -m vinishko.app --host 0.0.0.0 --port 8000        # по умолчанию 127.0.0.1:8000; --search-config, -c/--norm-config, --pipeline-config, --no-resolve
uvicorn vinishko.app:app --port 8000                      # то же с конфигами по умолчанию
```

В Docker, из корня; ключи — в `.env` (образец `.env.example`: `OPENROUTER_API_KEY`, `QDRANT_API_KEY`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`):

```bash
docker compose up -d --build                                                          # CPU: qdrant, приложение на :8000, фронт на :8080
docker compose -f docker-compose.yml -f docker-compose.gpu.override.yml up -d --build  # приложение на GPU
```

Образы — `docker/`: `app-cpu.Dockerfile` (python 3.13 slim, группа `cpu-inference`, поиск на OpenVINO), `app-gpu.Dockerfile`
(`nvidia/cuda:13.0.3-base-ubuntu24.04`, группа `flash-inference`, поиск на TensorRT; драйвер ≥ 580 и NVIDIA Container Toolkit),
`frontend.Dockerfile` (сборка Vite, nginx раздаёт её и проксирует `/api/` в приложение). Колёса из `wheels/` копируются в сборку обоих
образов приложения: uv сверяет с `uv.lock` все path-источники. torch из `uv.lock` — сборка PyPI под CUDA 12.8, её библиотеки CUDA едут и в
CPU-образ. Веса SAM3, модель поиска, engine TensorRT и картинки из S3 качаются при первом старте в том `cache`, диалоги сомелье — в том
`sessions`; коллекцию qdrant приложение восстанавливает из снапшота в S3. Фронт пока собирается в mock-режиме: на контракт API он не переведён.

Пайплайн поднимается при старте: нормализация, поиск, второй уровень и каталог по своим конфигам (`pred/pipeline/steps/normalization/normalize.toml`,
`pred/pipeline/steps/vis_searcher/config.yaml`, `pred/pipeline/steps/near_duplicates/config.yaml`, `pred/pipeline/config.yaml`). Устройства — `NORMALIZER_DEV` и `VIS_SEARCHER_DEV`,
ключ qdrant — `QDRANT_API_KEY`. Запросы к пайплайну идут по одному под замком: SAM3 и engine TensorRT не рассчитаны на параллельные
вызовы из одного процесса; параллелизм — несколько процессов. Сомелье и whatis замка не ждут: их вызовы OpenRouter идут в потоках параллельно
с распознаванием.

При старте читается корневой `.env`; уже заданные переменные окружения имеют приоритет. Второй уровень, сомелье и whatis берут ключ
из `OPENROUTER_API_KEY` (сомелье и whatis — ещё из файла по `OPENROUTER_API_KEY_FILE`). `openrouter_http_proxy` — proxy только для запросов
сомелье и whatis к OpenRouter; пусто или нет — напрямую, `HTTP_PROXY`/`HTTPS_PROXY` они не читают. `SOMMELIER_DATA_DIR` — каталог диалогов
сомелье, без неё — `data/sessions/` от текущего каталога.

Офлайн-проверки сомелье, whatis и второго уровня, без GPU, пайплайна и вызовов OpenRouter:

```bash
python -m unittest vinishko.sommelier.tests.test_api vinishko.sommelier.tests.test_solution_bundle vinishko.whatis.tests.test_api vinishko.whatis.tests.test_solution_bundle
python -m unittest vinishko.pred.pipeline.steps.near_duplicates.test_resolve
```
