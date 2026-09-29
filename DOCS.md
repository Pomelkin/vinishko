# Документация проекта «Своё Вино»

(!) Если возникли проблемы с запуском, пишите в Telegram: [@pomelk1n](https://t.me/pomelk1n) или [@zeromikhai](https://t.me/zeromikhai).

Сводное руководство по всему репозиторию: возможности, запуск, API, данные, обучение и проверки. Состояние описано на 29 сентября 2026 года по всем 15 исходным Markdown-файлам, структуре проекта и действующим конфигурациям. Устройство компонентов описано в [ARCHITECTURE.md](ARCHITECTURE.md).

## 1. Назначение и возможности

Проект создан для кейса РСХБ «Сканер российских вин с описанием на платформе “Своё вино”». По фотографии бутылки или полки приложение находит позиции российского винного каталога, показывает карточки и похожие варианты, определяет признаки неизвестного вина и предоставляет чат с AI-сомелье.

Основные сценарии:

- Съёмка камерой или загрузка изображения из галереи.
- Распознавание нескольких бутылок с выбором по контурам на фотографии.
- Точное совпадение с каталогом либо отказ с причиной; просмотр до пяти похожих кандидатов.
- Поиск и фильтрация каталога, переход к подробной карточке вина.
- Определение категории и винодельни для неизвестной бутылки через WhatIs.
- Диалог с сомелье по исходной карточке и альтернативам, восстановление и удаление истории.
- Возврат к сохранённому сканированию после обновления страницы без повторного распознавания.
- Отдельный плоский ответ со slug для скрипта оценки организаторов.

Целевая точность 90–100% и задержка до 3 секунд из раннего описания кейса — требования, а не подтверждённые гарантии сервиса. Косинусное сходство не является вероятностью совпадения.

## 2. Карта репозитория

```text
.
├── README.md                       краткое руководство по запуску
├── DOCS.md                         общая документация проекта
├── ARCHITECTURE.md                 архитектура всей системы
├── CONTEXT.md                      решения, история экспериментов и ранние планы
├── pyproject.toml / uv.lock        Python-зависимости и фиксированное окружение
├── .env.example                    образец настроек окружения
├── docker-compose*.yml             CPU, GPU и TLS-конфигурации
├── docker/                        образы приложения/frontend и nginx
├── vinishko/
│   ├── app.py                     единое FastAPI-приложение и CLI
│   ├── openrouter_proxy.py        общий HTTP-прокси OpenRouter
│   ├── e2e.py                     сквозная оценка через HTTP
│   ├── pred/                      распознавание, каталог и адаптер UI
│   │   ├── catalog.csv            каталог, поставляемый с приложением
│   │   └── pipeline/              оркестратор, структуры и три шага
│   ├── sommelier/                 диалоги, JSON-хранилище, промпты и знания
│   └── whatis/                    признаки неизвестного вина и справочники
├── frontend/                      React/TypeScript-приложение
│   ├── src/                       страницы, функции, контракты и хранилище
│   ├── public/assets/             изображения, шрифт и SVG
│   ├── e2e/                       браузерные сценарии
│   ├── qa/                        отчёты и визуальные сверки
│   └── tools/                     подготовка ресурсов и разбор референсов
├── vis_seacher_training/           обучение DINOv3; имя каталога сохранено как в коде
│   └── experiments/               конфиги vitl16_512 и vitl16_1024_cont
├── scripts/                       датасеты, экспорт моделей и бенчмарки
├── datasets/                      указатели DVC и локальные данные при восстановлении
├── weights.dvc                    указатель DVC на веса и экспорты
├── reports/                       сохранённые отчёты экспериментов
├── notebooks/playground.ipynb      исследовательский ноутбук
└── wheels/                        Linux-колёса flash-attn и causal-conv1d
```

MHTML, фотографии и PDF в корне — исходные материалы кейса и визуальные референсы. `.venv`, кэши, служебные логи и настройки IDE относятся к локальному окружению. Содержимое больших датасетов и весов не следует считать присутствующим только по наличию указателей DVC.

## 3. Требования и зависимости

Сервер использует Python `>=3.13,<3.14`, FastAPI, PyTorch, Ultralytics/SAM3, Qdrant, ONNX и HTTP-клиенты. Зависимости описаны в `pyproject.toml`, версии разрешены в `uv.lock`.

| Группа Python-зависимостей | Назначение |
| --- | --- |
| `cpu-inference` | OpenVINO для визуального поиска на CPU |
| `flash-inference` | TensorRT и дополнительные GPU-библиотеки |
| `vis-searcher-training` | Lightning, аугментации, TensorBoard и обучение |
| `dev` | Ноутбуки, DVC, загрузка и анализ данных |

Frontend использует React 19, TypeScript, Vite, React Router, Zustand и Zod. Для разработки указан Node.js 22.12+. Vitest и Playwright входят в инструменты проверок.

Поддерживаемый путь сборки приложения — Linux amd64, в том числе контейнеры Docker на Windows. В обоих образах приложения обязательны локальные файлы:

- `wheels/causal_conv1d-1.6.2.post1-cp313-cp313-linux_x86_64.whl`.
- `wheels/flash_attn-2.8.3-cp313-cp313-linux_x86_64.whl`.

`uv` проверяет эти path-источники даже при установке только CPU-группы. GPU-образ использует CUDA 13.0.3 и требует совместимый NVIDIA-драйвер; в проектной документации указан драйвер 580 или новее. PyTorch из lock-файла поставляет собственные библиотеки CUDA 12.8, в том числе в CPU-образ.

## 4. Запуск в Docker

Из корня проекта скопируйте образец настроек, если `.env` ещё не создан:

```powershell
Copy-Item .env.example .env
```

Заполните ключ OpenRouter и реквизиты доступа к S3. По умолчанию приложение использует приватные изображения каталога и снапшот Qdrant из Selectel. При необходимости задайте ключ Qdrant и токен Hugging Face.

CPU:

```powershell
docker compose up -d --build
```

GPU:

```powershell
docker compose -f docker-compose.yml -f docker-compose.gpu.override.yml up -d --build
```

Для GPU требуется доступная NVIDIA GPU и её поддержка в Docker/WSL2 либо NVIDIA Container Toolkit. Запускайте один вариант за раз: CPU и GPU используют общие тома.

| Адрес | Назначение |
| --- | --- |
| `http://localhost:8080` | Веб-интерфейс через nginx |
| `http://localhost:8000/docs` | Swagger/OpenAPI приложения |
| `http://localhost:8000/health` | Готовность и сведения о моделях/каталоге |
| `http://localhost:6333` | Qdrant |

Compose запускает Qdrant, проверку его готовности `qdrant-ready`, приложение и frontend. Первый старт скачивает веса SAM3 и энкодер, восстанавливает коллекцию из снапшота; на GPU также собирает TensorRT engine. Приложение прогревает модели до готовности. Это может занять несколько минут.

```powershell
docker compose logs -f app
docker compose down
```

Остановка сохраняет тома `qdrant_storage`, `cache`, `sessions`. Без реквизитов внешних сервисов можно собирать frontend и запускать его mock-режим; полноценный запуск API требует согласованного каталога, моделей и коллекции.

### HTTPS и телефон

Для камеры в мобильном браузере нужен HTTPS с доверенным сертификатом. TLS-конфигурация использует уже полученный сертификат Let's Encrypt на хосте: задайте `LETSENCRYPT_DIR` и `TLS_DOMAIN`, затем добавьте `docker-compose.tls.override.yml`:

```powershell
docker compose -f docker-compose.yml -f docker-compose.tls.override.yml up -d --build
```

Для GPU добавьте GPU override перед TLS override. Frontend слушает 443; HTTP на 8080 перенаправляет на HTTPS, кроме `/v1/eval/`. После продления сертификата выполните `docker compose exec frontend nginx -s reload`. Каталог Let's Encrypt монтируется целиком из-за ссылок `live/` на `archive/`.

## 5. Локальная разработка

В Linux amd64 установите серверное окружение:

```bash
uv sync --locked --no-default-groups --group cpu-inference
```

Для GPU замените группу на `flash-inference`. Если Qdrant работает в Docker, а приложение запускается на хосте, задайте `QDRANT_HOST=127.0.0.1`, `QDRANT_PORT=6333`, `QDRANT_HTTPS=false`.

С активированным Python-окружением:

```bash
python -m vinishko.app --host 127.0.0.1 --port 8000
```

Доступны `--pipeline-config`, `--search-config`, `-c/--norm-config`, `--max-bottles` и `--no-resolve`. Без второго уровня исходный API возвращает кандидатов поиска. Альтернатива запуска: `uvicorn vinishko.app:app --port 8000`.

В отдельном терминале из `frontend/`:

```powershell
npm ci
npm run dev
```

Vite доступен на 5173 и проксирует `/api` на 8000. Настройки `frontend/.env.local`:

```dotenv
VITE_API_MODE=http
VITE_API_BASE_URL=/api
API_PROXY_TARGET=http://127.0.0.1:8000
```

`npm run build` создаёт `frontend/dist`. FastAPI при локальном запуске может раздавать эту сборку; Docker использует отдельный nginx. Для независимой демонстрации задайте `VITE_API_MODE=mock` и перезапустите Vite: распознавание работает только на заранее подготовленных демофото, AI-функции недоступны.

## 6. Конфигурация

Корневой `.env` читается при старте; уже заданные переменные окружения имеют приоритет. Секреты не должны попадать в `VITE_*`, поскольку эти значения доступны браузеру.

| Переменная | Назначение |
| --- | --- |
| `OPENROUTER_API_KEY` | Второй уровень распознавания, сомелье и WhatIs |
| `OPENROUTER_API_KEY_FILE` | Файл ключа для сомелье и WhatIs; при заданном пути заменяет прямое значение |
| `OPENROUTER_MODEL` | Переопределение модели второго уровня |
| `OR_PROXY` | Общий HTTP-прокси OpenRouter с проверкой при старте |
| `openrouter_http_proxy` | Прокси сомелье/WhatIs, если общий не задан |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION` | Доступ к S3; регион Compose по умолчанию `ru-6` |
| `QDRANT_API_KEY` | Ключ приложения и контейнера Qdrant |
| `QDRANT_HOST`, `QDRANT_PORT`, `QDRANT_HTTPS` | Переопределение подключения к Qdrant при локальном запуске |
| `HF_TOKEN` | Доступ к моделям Hugging Face при необходимости |
| `NORMALIZER_DEV`, `VIS_SEARCHER_DEV` | `auto`, `cpu` или `cuda:<индекс>`; перекрывают конфиги устройств |
| `MAX_BOTTLES` | Лимит веб-сервиса: по умолчанию 10, допустимо 1–100 |
| `AUTO_WHATIS` | Автоматический WhatIs для отказов на сервере; по умолчанию `false` |
| `SOMMELIER_DATA_DIR` | Каталог диалогов; в Docker `/data/sessions` |
| `OPENROUTER_SITE_URL`, `OPENROUTER_APP_TITLE` | Необязательные заголовки OpenRouter |
| `LETSENCRYPT_DIR`, `TLS_DOMAIN` | Каталог сертификатов и домен TLS override |

В Docker в контейнер передаются только настройки, перечисленные в Compose; каталог диалогов и устройства закреплены также в Dockerfile. Добавление значения в `.env` само по себе не означает его передачу контейнеру.

`OR_PROXY` поддерживает `http://[логин:пароль@]хост:порт`; SOCKS не поддерживается. Для прокси на Windows-хосте из контейнера используйте `host.docker.internal`. Если общий прокси не задан, второй уровень использует прокси окружения, а сомелье/WhatIs — `openrouter_http_proxy` либо прямое соединение.

| Файл | Что настраивает |
| --- | --- |
| `vinishko/pred/pipeline/config.yaml` | Источник CSV и колонка slug |
| `vinishko/pred/pipeline/steps/normalization/normalize.toml` | SAM3, отбор, ориентация, кропы и фон |
| `vinishko/pred/pipeline/steps/normalization/calibration.json` | Калибровка отбора бутылок |
| `vinishko/pred/pipeline/steps/vis_searcher/config.yaml` | Энкодер, Qdrant, поиск, S3, снапшоты и кэш |
| `vinishko/pred/pipeline/steps/near_duplicates/config.yaml` | Модель OpenRouter, генерация, таймаут, повторы и параллелизм |
| `vinishko/sommelier/solution/config.py` | Настройки сомелье |
| `vinishko/whatis/solution/config.json` | Настройки WhatIs |

Текущий поиск: `pomelk1n/siglipus-twinturbo-gguf-umer`, коллекция `catalog_siglip2_hybrid`, входы `[crop, box_crop]`, `top_k=10`, до пяти групп, порог `0.7`, отставание группы не более `0.07`. Текущий второй уровень: `deepseek/deepseek-v4.1-flash`, провайдер Together, рассуждения `none` в обоих кругах, параллелизм 8, таймаут 60 секунд, настройка повторов 3. Эти значения описывают файлы репозитория и могут быть переопределены окружением.

## 7. HTTP API

Полная актуальная схема работающего приложения доступна в `/docs` и `/openapi.json`. Контракт браузера подробнее описан в [frontend/API.md](frontend/API.md).

| Метод и путь | Назначение |
| --- | --- |
| `POST /api/recognize` | Multipart `image`, адаптированный результат для UI |
| `GET /api/wines?q=...` | Каталог и поиск по словам |
| `GET /api/wines/{slug}` | Подробная карточка; 404 для неизвестного slug |
| `GET /api/catalog/images/{name}` | Изображение из хранилища коллекции |
| `GET /api/health` | Готовность и информация о системе |
| `POST /api/sommelier/sessions/{uuid7}` | Создать диалог: `{wine, candidates}` |
| `POST /api/sommelier/sessions/{uuid7}/messages` | Добавить вопрос: `{content}` |
| `GET /api/sommelier/sessions/{uuid7}` | Загрузить историю |
| `DELETE /api/sommelier/sessions/{uuid7}` | Удалить диалог; 204 |
| `POST /api/whatis` | Multipart `image`, предполагаемые `{category, brand}` |
| `POST /recognize` | Исходный результат пайплайна |
| `POST /v1/eval/predict` | Ответ организаторам `{ "slug": "..." }` или `{ "slug": null }` |

Исходные `/health`, `/catalog/images/{name}`, `/sommelier/...` и `/whatis` также сохранены.

### Распознавание и изображения

Исходный `/recognize` возвращает размеры фото, `bottles`, `ignored`, `timings_s`. У бутылки есть UUID, маски, bbox, скор отбора, статус, совпадение/кандидаты либо причина отказа. Координаты — пиксели изображения после EXIF-поворота. Статус без второго уровня может быть `candidates`; выбранное совпадение имеет источник `group` или `final` и наблюдения модели `checklist`.

Адаптер `/api/recognize` возвращает `schemaVersion: '1.0'`, `image`, `detections`, метрики и время обработки. Координаты контуров нормализованы в `[0,1]`; `polygons` содержит все компоненты, `polygon` — основной контур для совместимости. UI различает `matched` и `unmatched`. В `similar` — до пяти кандидатов без выбранной позиции. Отказы нормализации учитываются в `ignored`, но не выдаются как отдельные detection.

`matchConfidence`, F1 и идентификатор размеченной выборки остаются `null`, если соответствующих данных нет; score выдаётся как `similarityScore`. `processingTimeMs` отражает сумму времени шагов, а не полную HTTP-задержку. Пустой список detection — штатный успешный ответ без годных бутылок.

Сервер принимает изображения до 20 МиБ и 80 Мп; ошибки декодирования дают 400, превышение лимита — 413. Исходный API поддерживает JPEG, PNG, WebP и HEIC; браузер предлагает преобразовать HEIC в JPEG/PNG, применяет EXIF и отправляет JPEG, уменьшенный до 2400 px. Отмена запроса клиентом не гарантирует остановку уже начавшегося вычисления на сервере.

### Каталог, WhatIs и сомелье

Каталог читается из `vinishko/pred/catalog.csv`; полные исходные строки доступны в `catalog`, альтернативы группы — в `candidateCatalogs`. Картинки берутся по реальному имени из payload Qdrant. Неизвестные рейтинги, крепость, температура и сочетания остаются пустыми/null. Похожие позиции в карточке относятся к группе каталога, кандидаты фотографии — к результату визуального поиска.

WhatIs принимает JPEG/PNG/WebP/GIF до 10 МиБ. Ошибки: 415 — формат, 413 — размер, 502 — провайдер/ответ, 503 — нет ключа. Значение `Не удалось определить` — штатный отказ по признаку. UI автоматически запрашивает признаки при открытии неизвестной бутылки по её кропу, а без масок — по уменьшенному фото; сохранённый результат не запрашивается повторно. Это предположение о признаках, а не подтверждение SKU.

Для сомелье браузер создаёт UUIDv7; `wine` — исходная карточка, `candidates` — обязательный список из 0–5 карточек. Сообщение содержит 1–4000 символов после удаления пробелов по краям. Создание диалога и сообщения возвращают 201; история — 200. Ошибки: 404 — нет сессии, 409 — UUID занят, 422 — данные, 502 — провайдер, 503 — нет ключа. После неоднозначной ошибки POST сначала загрузите историю: автоматического повтора нет.

### Режим оценки организаторов

`/v1/eval/predict` выбирает годную бутылку с наибольшей площадью маски, запускает поиск только для неё и выделяет второму уровню остаток бюджета 8 секунд с начала обработки. При ошибке или таймауте второго уровня возвращается top-1 поиска; явный отказ модели/поиска возвращает `null`. Эта политика отличается от пользовательского API, где ошибка второго уровня не заменяется подтверждённым совпадением. Нормализация, ожидание замка и поиск сами по себе не ограничены этим бюджетом, поэтому 8 секунд не являются жёстким пределом всей HTTP-обработки. Для выбора среди соседей оставьте `MAX_BOTTLES` больше 1.

## 8. Данные, индекс и модели

Большие артефакты отслеживаются DVC: `weights.dvc` и `datasets/visual-encoder.dvc`. Remote в `.dvc/config` — `s3://vino/dvc` в Selectel. При настроенном доступе и установленной dev-группе:

```bash
dvc pull weights.dvc datasets/visual-encoder.dvc
```

Восстановление данных обучения через DVC и восстановление поисковой коллекции из снапшота — разные операции. Для запуска сервиса весь обучающий датасет не нужен.

Внешние датасеты подготавливаются отдельно:

```bash
python -m scripts.prepare_datasets status datasets/visual-encoder
python -m scripts.prepare_datasets sync datasets/visual-encoder
python -m scripts.normalize_dataset run datasets/visual-encoder/winesensed
```

Основной обучающий источник — WineSensed; Open Food Facts используется как каталог для проверок, Products-10K — как негативы. Разметка нормализации хранится в `normalization.jsonl`; кропы при обучении восстанавливаются с аугментациями. Условия лицензий источников и моделей перечислены в исходной документации; сведения из `CONTEXT.md` о них относятся к истории проекта и требуют отдельной проверки при изменении способа использования.

### Сборка и перенос коллекции

После подготовки CSV и фотографий:

```bash
python -m vinishko.pred.pipeline.steps.vis_searcher.build_catalog --csv datasets/hack-vine/catalog/catalog.csv --images datasets/hack-vine/catalog/images --photo-column image_filename --group-column near_duplicate_group_slug --on-failure skip
python -m vinishko.pred.pipeline.steps.vis_searcher.dump_collection
```

Сборка нормализует фото, создаёт векторы и изображения, сохраняет `manifest.json`. Непрошедшие нормализацию позиции могут быть пропущены с отчётом. Существующие коллекции/снапшоты не перезаписываются без подтверждения CLI. `dump_collection` сохраняет снапшот и JSON-паспорт с sha256 и метаданными; `build_catalog` снапшот автоматически не обновляет.

Если коллекции нет, приложение восстанавливает её из `snapshots` в конфиге поиска. Картинки и манифест остаются в отдельном хранилище `images`. Смена модели, входов энкодера или параметров, определяющих пиксели кропа, требует согласованной пересборки коллекции и артефактов. Старые коллекции без манифеста не принимаются.

## 9. Обучение и экспорт

Модуль называется `vis_seacher_training` именно с таким написанием. Он обучает DINOv3 с CLS ⊕ GeM, проекционной головой и sub-center ArcFace. Поиск в текущем приложении использует SigLIP2; наличие обучения DINOv3 не означает его использование по умолчанию.

В репозитории присутствуют конфиги `experiments/dinov3_vitl16_512/config.yaml` и `experiments/dinov3_vitl16_1024_cont/config.yaml`. Примеры с ViT-B из раннего README относятся к прошлому состоянию и не указывают на существующий конфиг.

```bash
uv sync --all-groups
hf auth login
python -m vis_seacher_training.preview_augs -c vis_seacher_training/experiments/dinov3_vitl16_512/config.yaml -o augs.png
python -m vis_seacher_training.init_model -c vis_seacher_training/experiments/dinov3_vitl16_512/config.yaml -o weights/init/dinov3_vitl16_512
python -m vis_seacher_training.run -d vis_seacher_training/experiments/dinov3_vitl16_512
tensorboard --logdir vis_seacher_training/experiments/dinov3_vitl16_512
```

Полная установка групп предназначена для подходящего Linux/GPU-окружения. До запуска проверьте пути данных и начальной модели в конфиге. PCA-whitening и центры классов подготавливает `init_model`; `widen_head` расширяет голову для продолжения обучения; `augs_fixture` сохраняет и сверяет эталоны аугментаций.

Эксперимент сохраняет конфигурацию, `label2id.json`, логи, TensorBoard, чекпоинты и экспорт лучшей модели. Экспорт содержит веса, `config.json`, `preprocess.json`, `model.onnx` и `model.bf16.onnx`. Вход ONNX — float32 RGB NCHW в диапазоне 0–255; нормировка внутри графа, выход L2-нормированный.

```bash
python -m vis_seacher_training.export -c <checkpoint> -o <export-directory>
python -m scripts.export_siglip2 -o weights/siglip2_so400m_14_384
python -m scripts.export_tulip -o weights/tulip_so400m_14_384
```

Экспорты SigLIP2 и TULIP используют общий код `scripts/export_common.py`. Бенчмарки `scripts/bench_dino`, `scripts/bench_tulip` используют утилиты `scripts/bench_common`; сохранённые HTML в `reports/` относятся к конкретным экспериментам, а не к текущей гарантированной точности API.

## 10. Проверки и диагностика

Офлайн-проверки без OpenRouter, S3, Qdrant и модельного инференса:

```bash
python -m unittest vinishko.pred.tests.test_ui vinishko.pred.tests.test_app
python -m unittest vinishko.sommelier.tests.test_api vinishko.sommelier.tests.test_solution_bundle vinishko.whatis.tests.test_api vinishko.whatis.tests.test_solution_bundle
python -m unittest vinishko.pred.pipeline.steps.near_duplicates.test_resolve vinishko.test_openrouter_proxy
python -m vinishko.pred.pipeline.steps.normalization.normalize --selftest
```

Frontend из его каталога:

```bash
npm run build
npm test
npm run test:e2e
```

Оценка с моделями и доступом к данным:

```bash
python -m vinishko.pred.pipeline.steps.vis_searcher.evaluate --test-dir datasets/hack-vine/test
python -m vinishko.pred.pipeline.evaluate_pipeline --test-dir datasets/hack-vine/test
python -m vinishko.pred.pipeline.evaluate_pipeline --image photo.jpg -o runs/
python -m vinishko.e2e --url http://127.0.0.1:8000
```

Поиск оценивается через recall@1/3/5, `group_recall`, ложные/верные отказы. Сквозной итог — accuracy, precision, recall, F1, confusion и задержки. Если правильный slug не попал в индекс, `accuracy_all_images` учитывает это как ошибку; отдельная accuracy может исключать такие фото. Отчёты помещаются в `reports/vis_searcher`, `reports/pipeline`, `reports/e2e`; дампы включают исходник, маски, кропы, кандидатов и trace второго уровня.

При ошибке запуска сначала проверьте `/health` и логи приложения, доступ к Qdrant/S3, наличие снапшота и совпадение модели/манифеста. При отсутствии AI проверьте ключ и прокси; при медленном первом старте — загрузку весов и сборку engine. При несовместимой нормализации верните параметры сборки или пересоберите индекс.

## 11. Подтверждённое состояние и ограничения

Согласно [frontend/qa/VERIFICATION.md](frontend/qa/VERIFICATION.md), при интеграции 29 сентября прошли frontend-сборка, 21 серверная проверка, валидация 2046 карточек и ряд сценариев браузера; Compose и nginx проверены. Полная сборка CPU/GPU-образов приложения прервалась из-за Docker Desktop и не подтверждена. В доступном WSL-окружении NVIDIA GPU не обнаружена.

Это ранее зафиксированные результаты, а не повторный запуск проверок при подготовке данного документа. Исторические 18 Vitest и 15 Playwright-проверок от 19 сентября относятся к исходному mock-интерфейсу. Физические мобильные устройства, настоящее распознавание, S3 и OpenRouter в интеграционной проверке не подтверждены.

Основные ограничения: ошибки и дубли исходного каталога, визуально одинаковые позиции, нечитаемые этикетки, отсутствие отдельного классификатора «вино/не вино», зависимость второго уровня от внешней модели и сети, медленный CPU-инференс. Замеры в `CONTEXT.md` относятся к конкретным версиям и выборкам; переносить их на новые фото без оценки нельзя.

## 12. Исходные документы

| Документ | Роль |
| --- | --- |
| [README.md](README.md) | Быстрый запуск проекта |
| [CONTEXT.md](CONTEXT.md) | История решений, данные, эксперименты и планы |
| [vinishko/README.md](vinishko/README.md) | Приложение, Docker, TLS и окружение |
| [vinishko/pred/README.md](vinishko/pred/README.md) | Исходный API распознавания и eval |
| [vinishko/pred/pipeline/README.md](vinishko/pred/pipeline/README.md) | Оркестратор и оценка |
| [normalization/README.md](vinishko/pred/pipeline/steps/normalization/README.md) | Сегментация, кропы и калибровка |
| [vis_searcher/README.md](vinishko/pred/pipeline/steps/vis_searcher/README.md) | Энкодер, поиск, индекс и снапшоты |
| [near_duplicates/README.md](vinishko/pred/pipeline/steps/near_duplicates/README.md) | Проверка кандидатов моделью |
| [vinishko/sommelier/README.md](vinishko/sommelier/README.md) | Диалоги сомелье |
| [vinishko/whatis/README.md](vinishko/whatis/README.md) | Признаки неизвестного вина |
| [vis_seacher_training/README.md](vis_seacher_training/README.md) | Обучение и экспорт DINOv3 |
| [frontend/README.md](frontend/README.md) | Разработка интерфейса |
| [frontend/API.md](frontend/API.md) | Контракт UI |
| [frontend/ARCHITECTURE.md](frontend/ARCHITECTURE.md) | Устройство frontend |
| [frontend/qa/VERIFICATION.md](frontend/qa/VERIFICATION.md) | Проверки и их ограничения |

При расхождении исторического описания с текущей реализацией для этой сводки использованы код и конфиги: CSV находится в `pred/catalog.csv`, текущий энкодер — SigLIP2, фон кропа белый, рассуждения второго уровня отключены в обоих кругах.
