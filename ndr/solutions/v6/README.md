# NDR solution v6: атомарное pairwise-evidence

`v6` — экспериментальная версия NDR, которая отделяет извлечение визуального evidence
моделью от принятия решения в Python. Для каждого кандидата VLM заполняет строгий checklist
из восьми различителей, а код детерминированно вычисляет `same` или `different`. Если точных
совпадений несколько, отдельный resolver выбирает один допустимый slug; если совпадений нет,
код возвращает `not_found`.

Основная гипотеза версии: явное состояние `unknown` и декомпозиция общего «профиля вина» на
атомарные признаки должны уменьшить ложные совпадения, не превращая отсутствующий или
нечитаемый текст в конфликт.

> `v6` уже была прогнана один раз. Этот README добавлен после прогона; Python-код и prompts
> не менялись. Runner включает **все** файлы solution в fingerprint, поэтому новый запуск из
> текущего каталога получил бы другой fingerprint только из-за документации. Для следующего
> эксперимента нужно скопировать каталог в новую версию, а не перезапускать `v6`.

## Pipeline

Для каждого QUERY версия выполняет такие шаги:

1. Последовательно сравнивает QUERY с каждым ELEMENT из candidate group отдельным model
   request.
2. Передаёт в request изображение QUERY, Эталон ELEMENT и компактную карточку товара.
3. Выбирает prompt по полю `vintage` карточки:
   - непустой `vintage` → `compare_year_matters.txt`;
   - пустой `vintage` → `compare_year_not_matter.txt`.
4. Получает и валидирует восемь evidence-полей:
   `maker`, `product_name`, `line_variant`, `grape`, `color`, `sweetness`,
   `carbonation`, `vintage`.
5. Для каждого поля модель обязана вернуть:
   - `query_value` — наблюдение по QUERY;
   - `element_value` — наблюдение по Эталону и/или карточке;
   - `state` — строго `match`, `conflict` или `unknown`.
6. `comparison_verdict()` в `models.py` вычисляет `same/different` без model-generated
   verdict.
7. Результаты всех сравнений агрегируются:
   - ни одного `same` → программный `not_found`;
   - ровно один `same` → slug этого кандидата;
   - два и более `same` → дополнительный resolver-call.
8. Resolver может вернуть только один из прошедших первый этап slug. `not_found` в его
   динамической JSON Schema отсутствует.

Таким образом, `not_found` никогда не генерируется моделью напрямую. Это решение кода,
означающее, что ни один кандидат не прошёл положительные правила `same`.

## Детерминированный verdict

Активными считаются все восемь различителей, если в карточке есть винтаж, и первые семь,
если винтаж пустой.

Кандидат получает `different`, если:

- хотя бы один активный признак имеет состояние `conflict`;
- либо карточка содержит винтаж, но `vintage.state` не равен `match` — то есть даже
  `unknown` недостаточно для подтверждения такого SKU.

При отсутствии конфликтов кандидат получает `same`, только если есть положительная опора
для идентичности:

- совпадает `product_name` или `line_variant`, и всего есть не менее двух активных
  `match`; либо
- одновременно совпадают `maker`, `grape` и хотя бы один стилевой признак из `color`,
  `sweetness`, `carbonation`.

Остальные случаи дают `different`. Само по себе `unknown` не считается конфликтом, но и не
создаёт положительного evidence.

В ветке без catalog vintage наблюдение `vintage` всё равно сохраняется для аудита, однако
не влияет на verdict даже при `conflict`. В ветке с catalog vintage требуется точное
совпадение явно прочитанного четырёхзначного года.

## Structured output и обработка ошибок

`models.py` — единственный источник контрактов:

- Pydantic работает в strict-режиме и запрещает дополнительные поля;
- comparison schema требует все восемь различителей и все три поля каждого evidence item;
- resolver schema строится динамически через `Literal` из slug, получивших `same`;
- одна и та же JSON Schema передаётся в `response_format`, полностью включается в system
  message и затем используется для валидации ответа.

Невалидный JSON не исправляется и не извлекается из произвольного текста. Если generation
не проходит контракт, этап возвращает `contract_error`. HTTP-, provider- и timeout-ошибки
возвращаются как `predictor_error`, после чего весь QUERY завершается ошибкой. Повторных
model calls нет, включая HTTP 429: `MAX_429_RETRIES = 0`.

Trace сохраняет request без base64 изображения, заголовки и сырой ответ provider-а, все
choices, `finish_reason`, reasoning-поля, результат parsing/validation, программный verdict,
список `same_slugs` и resolver-call. Это позволяет восстановить причину решения по каждому
кандидату.

## Multimodal-вход

Comparison получает два изображения отдельными content blocks: QUERY и Эталон одного
ELEMENT. MIME определяется по сигнатуре файла, а не по расширению. Поддерживаются JPEG,
PNG, GIF и WebP; для `v6` используется `image_detail="original"`.

В компактную карточку ELEMENT входят:

```text
slug, name, winery, vintage, grapes, category, sugar,
sparkling, abv, aging_or_reserve
```

Resolver получает исходный QUERY и изображения с карточками только тех ELEMENT, которые
прошли pairwise-правила. Его ответ содержит один `slug` без объясняющего evidence.

## Конфигурация прогона

Постоянные настройки находятся в `config.py`, в блоке `SETTINGS`. Зафиксированные defaults
`v6`:

| Настройка | Значение |
|---|---|
| Model | `deepseek/deepseek-v4.1-flash` |
| Provider | только `together` |
| Provider fallback | выключен |
| Temperature | `0.0` |
| Max completion tokens | `4096` |
| Reasoning effort | `none` |
| Generations | `1` |
| Image detail | `original` |
| Model-call timeout | `60` с |
| Config concurrency | `1` |
| Candidate-order seed | `0` |
| Generation seed | не задан |

API-ключ читается из `OPENROUTER_API_KEY` и не сохраняется в исходниках или trace. CLI и
переменные окружения могут переопределять connection/run-настройки; provider routing
задаётся в `config.py`.

Исторический полный прогон использовал CLI override `concurrency=16`; остальные указанные
параметры совпадают с defaults.

## Состав каталога

| Файл | Назначение |
|---|---|
| `config.py` | строгая конфигурация OpenRouter, генерации и локального исполнения |
| `models.py` | Pydantic-схемы и детерминированный `comparison_verdict()` |
| `predictor.py` | сбор multimodal requests, HTTP-вызовы, trace и pipeline |
| `prompts/compare_year_matters.txt` | evidence extraction с обязательным точным винтажом |
| `prompts/compare_year_not_matter.txt` | evidence extraction без влияния винтажа на verdict |
| `prompts/resolve_multiple_same.txt` | выбор одного ELEMENT среди нескольких `same` |
| `test_predictor.py` | offline-тесты схем, правил verdict и pipeline без сети |

## Результат существующего прогона

Авторитетный артефакт:
`ndr/results/v6/20260919T213900.872958Z/`.

Run ID: `ndr-v6-20260919T213900.872958Z-692eafa1671f`.
Исторический fingerprint исходников до добавления README:
`692eafa1671f41a9580b758e927431b0eb1aeb791cf227911183480355e1d45a`.

| Метрика | v6 |
|---|---:|
| Attempted / correct | 51 / 49 |
| End-to-end accuracy | **96,08%** |
| Conditional runner quality | 49/51 = **96,08%** |
| Excluded errors | 0 |
| Contract / predictor errors | 0 / 0 |
| `not_found` | 1, правильный |
| Pairwise TP / FP / TN / FN | 50 / 13 / 202 / 0 |
| Pairwise precision / recall / F1 для `same` | 79,37% / 100% / 88,50% |
| Resolver queries | 9/51 = 17,65% |
| Resolver correct / wrong | 7 / 2 |
| Model requests | 274: 265 comparison + 9 resolver |
| Latency p50 / p95 / max | 12,083 / 28,026 / 30,527 с |
| Total tokens | 876 341 |
| Cost | $0,184976748 |

Все 274 generation завершились с `finish_reason=stop`, прошли schema validation и были
обслужены Together. Ошибок и retry не было.

По сравнению с pairwise-версией `v2` атомарное evidence повысило итог с 48/51 до 49/51,
снизило число pairwise false positive с 18 до 13, устранило единственный false negative и
уменьшило число resolver queries с 13 до 9. Цена — рост total tokens на 40,77% и стоимости
на 46,43%.

Подробный разбор находится в:

- `ndr/reports/V6_VS_V2_ATOMIC_EVIDENCE.md`;
- `ndr/reports/V6_VS_V5_NOT_FOUND.md`.

## Известные ограничения

Главный дефект `v6` — candidate-conditioned OCR. QUERY извлекается не один раз: модель заново
читает его рядом с каждым ELEMENT. Поэтому одно и то же изображение может получить
взаимоисключающие `query_value`.

Оба итоговых промаха имеют именно такую природу:

- `q-000002`: один comparison прочитал `ПОЛУСЛАДКОЕ`, другой — `ПОЛУСУХОЕ`; оба кандидата
  получили `same`, resolver выбрал полусухое вместо gold-полусладкого;
- `q-000047`: QUERY был прочитан и как `СУХОЕ`, и как `ПОЛУСУХОЕ`; resolver выбрал сухое
  вместо gold-полусухого.

Resolver возвращает только slug и не объясняет финальный выбор, поэтому после pairwise-этапа
его решение менее аудируемо.

Есть и семантическое ограничение `not_found`: ноль `same` может означать как явные конфликты
со всеми кандидатами, так и недостаток положительного evidence. Эти причины не разделены.
Кроме того, в текущем NDR gold содержит лишь один `not_found`, поэтому результат 1/1 нельзя
считать надёжной оценкой OOD-качества.

Наконец, версия не соответствует продуктовому SLA: p50 равен 12,08 с, а 30 из 51 QUERY
заняли больше 10 секунд. `v6` полезна прежде всего как аудируемый диагностический
эксперимент, а не как оптимальная версия для production.

## Offline-проверки

Из корня репозитория:

```powershell
python -B .agents/skills/near-duplicates/scripts/validate.py
python -B ndr/run/build_dataset.py --check
python -B -m unittest ndr.run.test_toolkit ndr.solutions.v6.test_predictor -v
```

Эти команды не обращаются к модели. `ndr/run/run.py` здесь намеренно не запускается:
model-backed NDR-прогоны выполняет только пользователь, а уже прогнанная версия должна
оставаться исторической.

Следующая архитектурная гипотеза — один раз извлечь неизменяемый QUERY profile без
кандидатов, а затем сравнивать все ELEMENT с одним набором evidence. Это должно убрать
возможность прочитать один QUERY одновременно как сухое и полусухое в разных calls.
