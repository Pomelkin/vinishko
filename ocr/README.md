# OCR-бенчмарк: `baidu/Unlimited-OCR`

В папке лежат ручная разметка видимых признаков на изображениях `data/test`, клиент для
удалённого vLLM и оценка OCR. Клиент реализует именно официальный single-image recipe
Unlimited-OCR, а не просит OCR-модель генерировать JSON.

## Что важно для этой модели

Unlimited-OCR — document-parsing VLM с контекстом 32K. Для одного изображения модель
использует режим `gundam`: `base_size=1024`, `image_size=640`, crop mode. В реализации vLLM
этот режим выбирается автоматически для single-image запроса.

Обязательный протокол из vLLM recipe:

- prompt начинается буквально с `<image>`; здесь используется `<image>document parsing.`;
- `temperature=0`, `max_tokens=8192`;
- в запросе передаётся `skip_special_tokens=false`;
- в запросе передаётся `vllm_xargs={"ngram_size":35,"window_size":128}`;
- сервер запускается с `NGramPerReqLogitsProcessor`;
- prefix cache и multimodal processor cache отключены.

Без `<image>` или при `skip_special_tokens=true` модель обычно возвращает пустой ответ.
Без logits processor длинный вывод может зациклиться на координатах.

Ссылки:

- [карточка `baidu/Unlimited-OCR`](https://huggingface.co/baidu/Unlimited-OCR);
- [официальный vLLM recipe](https://recipes.vllm.ai/baidu/Unlimited-OCR).

## Датасет

`dataset.jsonl` содержит 76 запросов — по одному на каждый файл `data/test` — и 287 вручную
проверенных признаков. Четыре изображения `not_found` просмотрены вручную: у них
`expected_slug=null`, но видимые OCR-признаки размечены так же, как у каталоговых позиций.

Gold содержит только текст, который действительно виден на конкретном изображении:

| Роль | Примеры | Вес |
|---|---|---:|
| `identity` | бренд, производитель, название, линейка | 3 |
| `disambiguation` | сорт, год, сахар, тип/цвет вина, Reserve/Brut | 3 |
| `secondary` | регион, аппелласьон, крепость, метод | 1 |

Поля Каталога, которых не видно на этикетке, в gold не подставляются. Это особенно важно для
визуально близких позиций: год, сорт и сахар оцениваются по реальному тексту этикетки.

Файлы:

- `dataset.jsonl` — готовый датасет;
- `build_dataset.py` — воспроизводит JSONL из ручной разметки;
- `validate_dataset.py` — проверяет покрытие `data/test`, SHA-256, Каталог и схему;
- `run.py` — единый запуск OCR, разбор ответа и fuzzy-оценка видимых признаков;
- `test_run.py` — contract-тест тела запроса, MIME и постобработки.

Проверка датасета:

```powershell
uv run python ocr/validate_dataset.py
```

## Запуск vLLM на GPU-сервере

Команда из официального recipe (Linux, Docker с NVIDIA Container Toolkit):

```bash
docker run --rm --gpus all --network host --ipc host \
  vllm/vllm-openai:unlimited-ocr \
  baidu/Unlimited-OCR \
  --trust-remote-code \
  --logits_processors vllm.model_executor.models.unlimited_ocr:NGramPerReqLogitsProcessor \
  --no-enable-prefix-caching \
  --mm-processor-cache-gb 0
```

Основной образ использует CUDA 13.0. Для Hopper/CUDA 12.9 карточка модели предлагает образ
`vllm/vllm-openai:unlimited-ocr-cu129`. По умолчанию API доступен на порту `8000`, модель
обслуживается под именем `baidu/Unlimited-OCR`.

## Настройка клиента

`OCR_BASE_URL` должен включать `/v1`. Остальные значения уже имеют правильные дефолты:

```powershell
$env:OCR_BASE_URL = "http://gpu-server:8000/v1"
$env:OCR_API_KEY = "EMPTY" # либо ключ reverse proxy
$env:OCR_MODEL = "baidu/Unlimited-OCR"
$env:OCR_TIMEOUT = "120"
$env:OCR_MAX_TOKENS = "8192"
$env:OCR_CONCURRENCY = "1"
$env:OCR_PREPROCESS = "original"
```

Если сервер запущен с другим `--served-model-name`, переопределите `OCR_MODEL`. При запуске
`run.py` автоматически загружает `.env` из корня репозитория. Уже заданные переменные
PowerShell имеют приоритет над значениями файла. `ocr/.env.example` можно использовать как
шаблон для корневого `.env`.

Проверить доступность сервера:

```powershell
Invoke-RestMethod "$env:OCR_BASE_URL/models" `
  -Headers @{ Authorization = "Bearer $env:OCR_API_KEY" }
```

## Прогон

Сначала один запрос:

```powershell
uv run python ocr/run.py --limit 1 --only-predicted
```

Полный последовательный прогон:

```powershell
uv run python ocr/run.py
```

Параллельный прогон, например четыре одновременных запроса:

```powershell
uv run python ocr/run.py --concurrency 4
```

### Если модель возвращает только `image`

Unlimited-OCR — parser документов, а не специализированный scene-text OCR. На фото
бутылки он может классифицировать весь кадр как иллюстрацию и вернуть только
`<|det|>image ...<|/det|>`, сознательно не читая текст внутри неё. Это не пустой ответ
API и не исправляется увеличением `max_tokens`.

Для проверки этой гипотезы есть два режима препроцессинга:

```powershell
# Один тесный центральный кроп этикетки, один API-запрос на изображение.
uv run python ocr/run.py `
  --query-id ocr-000016 `
  --only-predicted `
  --preprocess center-label

# Оригинал и три перекрывающихся кропа; четыре отдельных single-image запроса.
uv run python ocr/run.py --preprocess label-scan --concurrency 2
```

`label-scan` намеренно не отправляет четыре картинки одним chat-запросом: multi-image
перевёл бы vLLM из single-image `gundam` в другой режим. Тексты непустых вариантов
объединяются в `raw_text` в parsed-файле, а каждый исходный ответ сохраняется в `attempts`
raw-файла. Это диагностический upper-bound без детектора этикетки; latency в строке — сумма всех
вариантов. `--concurrency` по-прежнему ограничивает число одновременно обрабатываемых
query, а не число скрытых ретраев — автоматических ретраев нет.

Начинайте с `1` и повышайте concurrency только если vLLM не уходит в OOM и latency остаётся
приемлемой. Один OpenAI client переиспользуется всеми worker-потоками; каждый запрос содержит
ровно одно изображение. Результаты записываются сразу по мере завершения, поэтому при
`--concurrency > 1` порядок строк JSONL может отличаться от порядка датасета.

Одна команда выполняет OCR, парсинг и оценку. Она создаёт три отдельных артефакта:

```text
ocr/results/unlimited-ocr-20260920-143012.raw.jsonl
ocr/results/unlimited-ocr-20260920-143012.parsed.jsonl
ocr/results/unlimited-ocr-20260920-143012.report.json
```

Raw JSONL записывается потоково и после закрытия считается неизменяемым. Парсер только читает
его и пишет соседний parsed JSONL; `--overwrite` разрешает заменить parsed/report, но никогда
не raw-файл.

Свой путь можно указать явно:

```powershell
uv run python ocr/run.py `
  --concurrency 4 `
  --output ocr/results/unlimited-ocr-c4.raw.jsonl
```

Также доступны `--query-id ocr-000001` (можно повторять), `--timeout`, `--max-tokens`,
`--model`, `--base-url`, `--api-key`, `--parsed-output` и `--report`. Автоматических ретраев
нет. Существующий raw-файл не перезаписывается даже с `--overwrite`.

Каждая строка raw JSONL содержит:

- `raw_output` — исходный вывод модели с `<|ref|>` / `<|det|>` grounding-разметкой;
- `latency_ms`, `finish_reason`, token usage и ошибку запроса;
- `preprocess` и `attempts` — использованный вид изображения и сырые ответы по каждому виду.

В parsed JSONL поля `raw_output` отсутствуют: там лежат `raw_text` после разворачивания `ref`,
удаления координат `det` и end-токена, а также `grounding_only` для ответа вида `image`.

Клиент определяет MIME по сигнатуре через Pillow, а не по расширению файла. Это нужно для
наборов, где WebP может называться `.jpg`.

## Повторный парсинг и оценка

Чтобы повторить парсинг и оценку существующего raw-файла без запроса к GPU:

```powershell
uv run python ocr/run.py `
  --raw-input ocr/results/unlimited-ocr-20260920-143012.raw.jsonl `
  --overwrite
```

Для частичного raw-файла добавьте `--only-predicted`:

```powershell
uv run python ocr/run.py `
  --raw-input ocr/results/smoke.raw.jsonl `
  --only-predicted
```

Без `--only-predicted` отсутствующие запросы считаются промахами. Основные метрики:

- `text_recall` — доля gold-признаков, найденных в очищенном OCR;
- `text_weighted_recall` — recall с весами ролей 3/3/1;
- `query_exact_core_recall` — доля запросов, где найдены все identity/disambiguation признаки;
- `text_output_rate` — доля запросов, где parser вообще вернул текст;
- `grounding_only_queries` — число ответов, состоящих из image-grounding без OCR-текста;
- `text_recall_given_output` и `query_exact_core_recall_given_output` — качество только
  на запросах с непустым текстом; они отделяют распознавание символов от layout-gate модели;
- latency `mean/p50/p95/max`, ошибки и число ответов с `finish_reason=length`.

Сопоставление допускает перечисленные aliases и небольшую OCR-ошибку через fuzzy token-window.
Полной посимвольной транскрипции этикеток в gold нет, поэтому benchmark измеряет feature
recall, но не CER/WER и не precision лишних слов.

## Локальные contract-тесты

Они не обращаются к GPU-серверу:

```powershell
uv run python -m unittest ocr.test_run
```
