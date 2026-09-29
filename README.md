# Своё Вино

Интерфейс в `frontend/` подключён к единому FastAPI-приложению `vinishko/app.py`:
распознавание нескольких бутылок, каталог, похожие кандидаты, признаки неизвестного вина,
диалоги с AI-сомелье. По умолчанию используется настоящий API.

## Запуск в Docker

Скопируйте `.env.example` в `.env`. Для полного сервиса нужны `OPENROUTER_API_KEY`,
доступ к S3 через `AWS_ACCESS_KEY_ID` и `AWS_SECRET_ACCESS_KEY`; при необходимости
`QDRANT_API_KEY` и `HF_TOKEN`. Без ключей можно собрать образы и открыть интерфейс,
но распознавание требует модели, коллекции и изображений каталога, а AI — OpenRouter.
Адреса S3 и модель: `vinishko/pred/pipeline/steps/vis_searcher/config.yaml`.

В `wheels/` должны находиться:

- `causal_conv1d-1.6.2.post1-cp313-cp313-linux_x86_64.whl`
- `flash_attn-2.8.3-cp313-cp313-linux_x86_64.whl`

Оба Dockerfile приложения копируют эту папку до `uv sync`: uv проверяет локальные
источники даже тогда, когда группа `flash-inference` не устанавливается.

CPU, Python 3.13 и OpenVINO:

```powershell
docker compose up -d --build
```

GPU, NVIDIA Container Toolkit или GPU-поддержка Docker Desktop/WSL2:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.override.yml up -d --build
```

Откройте [интерфейс](http://localhost:8080). API: [документация](http://localhost:8000/docs).
Все Dockerfile находятся в `docker`: `app-cpu.Dockerfile`, `app-gpu.Dockerfile`,
`frontend.Dockerfile`. CPU ставит только группу `cpu-inference`; GPU — `flash-inference`.
В `uv.lock` TensorRT 11.3 зависит от `tensorrt-cu13`, поэтому GPU-образ использует
CUDA 13 (`13.0.3-base-ubuntu24.04`), с драйвером NVIDIA 580 или новее.
PyTorch 2.10 устанавливает собственные CUDA 12.8-библиотеки из lock-файла — также в CPU-образе.
Зависимости фиксированы через `uv sync --frozen`.

Compose поднимает Qdrant, единое приложение и nginx с frontend. Проверка готовности
Qdrant завершается до запуска приложения. Nginx сохраняет префикс `/api` и обслуживает
прямые ссылки `/scan/:id`, `/wine/:slug`. Для телефона используйте HTTPS перед nginx.
Proxy OpenRouter на Windows-хосте указывайте через `host.docker.internal`, а не `127.0.0.1`.

Первый старт скачивает SAM3 и энкодер, восстанавливает Qdrant из готового снапшота,
на GPU ещё собирает TensorRT engine. Это может занять несколько минут.
CPU-распознавание значительно медленнее GPU. Логи: `docker compose logs -f app`.
Данные хранятся в volumes `qdrant_storage`, `cache`, `sessions`.
`docker compose down` останавливает приложение, сохраняя данные.
CPU/GPU-конфигурации используют одни volumes; запускайте один вариант за раз.

## Разработка

Для Linux amd64 с Python 3.13:

```bash
uv sync --locked --no-default-groups --group cpu-inference
# Для GPU вместо cpu-inference используйте flash-inference.
```

При локальном запуске приложения с Qdrant в Docker задайте в `.env`
`QDRANT_HOST=127.0.0.1`, `QDRANT_PORT=6333`, `QDRANT_HTTPS=false`.
Сомелье и whatis запускаются внутри приложения, отдельные процессы не нужны:

```powershell
python -m vinishko.app
```

В другом терминале:

```powershell
cd frontend
npm ci
npm run dev
```

Откройте [Vite](http://localhost:5173): `/api` проксируется на порт 8000.
`npm run build` создаёт `frontend/dist`; при локальном запуске FastAPI также может
раздавать эту сборку на порту 8000. В Docker frontend раздаёт nginx на 8080.

`MAX_BOTTLES=10` задаёт лимит бутылок веб-сервиса; CLI принимает `--max-bottles`.
`AUTO_WHATIS=true` включает автоматическое определение признаков отвергнутых бутылок.
По умолчанию whatis вызывается пользователем отдельно. Демонстрация без API включается
явно через `VITE_API_MODE=mock`. Секреты нельзя помещать в `VITE_*`.

Контракт: [frontend/API.md](frontend/API.md). Архитектура: [frontend/ARCHITECTURE.md](frontend/ARCHITECTURE.md).
Сервер: [vinishko/README.md](vinishko/README.md). Проверки: [frontend/qa/VERIFICATION.md](frontend/qa/VERIFICATION.md).
