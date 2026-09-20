# NDR solution v5: one-shot выбор с `not_found`

`v5` — экспериментальная версия NDR, которая сравнивает QUERY сразу со всей candidate group
за один multimodal model request. Модель должна совместно рассмотреть все Эталоны и карточки,
а затем вернуть ровно один slug либо `not_found`, если каждый кандидат исключён надёжным
продуктовым конфликтом.

Версия развивает one-shot pipeline `v4`. Основная гипотеза: добавление явного `not_found` в
динамический structured-output enum и строгих правил отказа позволит распознавать hard
negative, сохранив один request на QUERY и низкую latency независимо от числа кандидатов.

> Для `v5` сохранены два завершённых артефакта с одинаковым fingerprint: частичный прогон
> одного QUERY и полный прогон 51 QUERY. Этот README добавлен после них; Python-код и prompts
> не менялись. Runner включает **все** файлы solution в fingerprint, поэтому новый запуск из
> текущего каталога получил бы другой fingerprint только из-за документации. Для следующего
> эксперимента нужно скопировать каталог в новую версию, а не перезапускать `v5`.

## Pipeline

Для каждого QUERY версия выполняет один этап `select_nearest`:

1. Получает полную группу кандидатов от NDR runner-а.
2. Строит динамическую Pydantic-модель, где `slug` ограничен enum из всех slug группы и
   специального значения `not_found`.
3. Для каждого ELEMENT формирует индивидуальную vintage policy:
   - при непустом catalog `vintage` требуется точное совпадение года;
   - при пустом `vintage` год не должен влиять на выбор.
4. Собирает единый system prompt из основного selection prompt, нужных vintage-фрагментов и
   списка правил года для каждого ELEMENT.
5. Передаёт модели изображение QUERY, а затем изображения и компактные карточки всех
   ELEMENT в исходном порядке.
6. Делает ровно один model request и валидирует ответ строгой Pydantic-моделью.
7. Возвращает выбранный slug или `not_found` без дополнительных comparison- или
   resolver-calls.

Модель сама выполняет визуальное сравнение, интерпретирует признаки, решает, исключены ли
все кандидаты, и принимает финальное решение. Отдельного программного `same/different`, как
в `v6`, здесь нет.

## Контракт ответа

Ответ содержит:

```json
{
  "checklist": {
    "maker": {"observation": "..."},
    "profile": {"observation": "..."},
    "year": {"observation": "..."}
  },
  "slug": "<один slug группы или not_found>"
}
```

Каждое observation обязательно, не может быть пустым и ограничено 240 символами:

- `maker` фиксирует совпадение, конфликт или неизвестность производителя/бренда;
- `profile` описывает название, тип, сорт, сахар, игристость и другие SKU-модификаторы;
- `year` объясняет сравнение винтажа или отсутствие требования точного года.

`models.py` — единственный источник JSON Schema. Pydantic работает в strict-режиме,
запрещает дополнительные поля и не допускает slug вне текущей группы. `not_found`
зарезервирован и не может одновременно быть candidate slug. Одна и та же schema:

- передаётся провайдеру в `response_format`;
- полностью включается в system message;
- используется для локальной валидации декодированного JSON.

Контракт проверяет форму ответа и допустимый slug, но не может доказать истинность
observations или логическую обоснованность выбора. В частности, код не проверяет, что перед
`not_found` действительно найден hard conflict для каждого ELEMENT.

## Политика `not_found`

Prompt разрешает `not_found` только тогда, когда **каждый** кандидат исключён хотя бы одним
уверенно наблюдаемым конфликтом идентичности товара. Нечитаемый, скрытый, обрезанный или
отсутствующий признак считается `unknown`, а не конфликтом. Неизвестный maker или vintage
сами по себе не должны приводить к отказу.

К hard conflicts отнесены, среди прочего:

- другой обязательный винтаж;
- другое название, номер линейки, сорт или именованный вариант;
- другой цвет или уровень сахара;
- `DRY` против `SEMI-DRY`, `BRUT` против `EXTRA BRUT`;
- тихое против игристого;
- наличие или отсутствие SKU-значимого `reserve`.

Если несколько кандидатов остаются правдоподобными, модель должна выбрать самый близкий
полный визуальный match. Первый ELEMENT используется только при полной неразличимости.

Эти правила существуют на уровне prompt, а не детерминированного кода. Поэтому семантически
ошибочный `not_found` с валидным checklist проходит response contract.

## Vintage policy

Файлы `compare_year_matters.txt` и `compare_year_not_matter.txt` в `v5` являются короткими
фрагментами основного prompt:

- непустой catalog vintage делает точный год SKU-значимым; соседний винтаж той же линейки —
  другой SKU;
- пустой catalog vintage исключает год из решения: присутствие, отсутствие или отличие
  напечатанного года само по себе не должно штрафовать ELEMENT.

`selection_system_prompt()` добавляет явную строку для каждого кандидата с номером ELEMENT,
slug, catalog vintage и флагом `exact_year_required`. Смешанная группа поэтому может
содержать обе политики в одном model request.

## Multimodal-вход

Модель получает QUERY и всю группу ELEMENT в одном user message. Каждый ELEMENT представлен
изображением Эталона и компактной карточкой со следующими полями:

```text
slug, name, winery, vintage, grapes, category, sugar,
sparkling, abv, aging_or_reserve
```

MIME изображений определяется по сигнатуре, а не по расширению. Поддерживаются JPEG, PNG,
GIF и WebP; для `v5` используется `image_detail="original"`. В trace base64 не сохраняется:
вместо него записываются путь, MIME, размер, SHA-256 и detail.

Основной selection prompt исторически лежит в
`prompts/resolve_multiple_same.txt`, хотя отдельного resolver-этапа у `v5` нет. Имя файла
унаследовано от предыдущего pairwise pipeline.

## Обработка ошибок и trace

Невалидный JSON не исправляется, не извлекается из произвольного текста и не отправляется
повторно. Если ни одна generation не проходит schema validation, QUERY завершается
`contract_error`. HTTP-, provider- и timeout-ошибки возвращаются как `predictor_error`.
Повторных model calls нет, включая HTTP 429: `MAX_429_RETRIES = 0`.

Trace сохраняет:

- endpoint и request без base64 изображений;
- candidate slugs и per-ELEMENT vintage policies;
- заголовки и сырой ответ provider-а;
- все choices, `finish_reason`, reasoning-поля и validation result;
- один `comparison_calls`-элемент со stage `select_nearest`.

Название `comparison_calls` здесь техническое: внутри находится единственный joint call на
всю группу, а не отдельные pairwise-сравнения.

## Конфигурация

Постоянные настройки находятся в `config.py`, в блоке `SETTINGS`. Defaults `v5`:

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

Оба исторических прогона использовали CLI override `concurrency=16`; остальные указанные
параметры совпадают с defaults.

## Состав каталога

| Файл | Назначение |
|---|---|
| `config.py` | строгая конфигурация OpenRouter, генерации и локального исполнения |
| `models.py` | Pydantic-контракт joint selection и динамический slug enum |
| `predictor.py` | сбор полного multimodal request, HTTP-вызов и trace |
| `prompts/resolve_multiple_same.txt` | основной prompt выбора кандидата или `not_found` |
| `prompts/compare_year_matters.txt` | fragment для ELEMENT с обязательным винтажом |
| `prompts/compare_year_not_matter.txt` | fragment для ELEMENT без обязательного винтажа |
| `test_predictor.py` | offline-тесты контракта, vintage policies и one-shot pipeline |

## Результаты прогонов

### Полный прогон

Авторитетный полный артефакт:
`ndr/results/v5/20260919T214443.536610Z/`.

Run ID: `ndr-v5-20260919T214443.536610Z-a634c9d506e8`.
Исторический fingerprint исходников до добавления README:
`a634c9d506e8f1fd351d2308e58a64f55a7968209f3dbab9ed8c2d65095285d3`.

| Метрика | v5 |
|---|---:|
| Attempted / correct | 51 / 49 |
| End-to-end accuracy | **96,08%** |
| Conditional runner quality | 49/51 = **96,08%** |
| Excluded errors | 0 |
| Answered accuracy | 49/51 = **96,08%** |
| Contract / predictor errors | 0 / 0 |
| `not_found` responses | 2 |
| `not_found` TP / FP / FN / TN | 1 / 1 / 0 / 49 |
| `not_found` precision / recall / F1 | 50% / 100% / 66,67% |
| Model requests | 51 — ровно один на QUERY |
| Latency mean / p50 / p95 / max | 3,869 / 3,811 / 5,637 / 6,404 с |
| Wall time при concurrency 16 | 15,905 с |
| Throughput | 3,207 QUERY/с |
| Prompt / completion / total tokens | 285 954 / 8 219 / 294 173 |
| Cost | $0,0853002 |

Все 51 generation завершились с `finish_reason=stop`, прошли schema validation и были
обслужены Together. Ошибок и retry не было.

### Частичный прогон

До полного запуска был выполнен отдельный прогон только `q-000001`:
`ndr/results/v5/20260919T214421.960697Z/`.

Он использовал тот же fingerprint, дал правильный ответ 1/1 за 2,159 с, сделал один request,
потратил 2 763 tokens и $0,0009864. В итоговую таблицу он не включён. Этот вызов мог прогреть
provider cache перед полным прогоном, поэтому стоимость полного запуска нельзя считать
чистым cold-cache измерением.

## Ошибки и правильный `not_found`

В полном прогоне было два неправильных ответа. Ошибок HTTP, predictor, JSON parsing или
response contract не было; оба ответа имели `finish_reason=stop` и
`validation_error=null`.

### `q-000005`: ложный `not_found`

Ожидался `alma-valley-semilon-beloe-suhoe-135`, но модель вернула `not_found`. QUERY и
Эталон согласованы: на QUERY читаются Alma Valley, Sémillon, 2023 и «белое сухое».

Решающее observation ошибочно утверждает, что QUERY — semi-sweet, а ближайший Sémillon —
dry. Из-за вымышленного конфликта сахара правильный кандидат был отвергнут. Это semantic
false negative и ложный отказ от catalog SKU, а не inconsistency датасета.

### `q-000047`: неверный slug

Ожидалось полусухое `millstream-av-igristoe-molodoe-polusuhoe-beloe`, но выбрано сухое
`millstream-av-igristoe-molodoe-suhoe-beloe`. Model observation утверждает, что QUERY
читается как «МОЛОДОЕ СУХОЕ БЕЛОЕ», хотя на QUERY и правильном Эталоне указано полусухое.
Это semantic/OCR wrong-SKU selection.

### `q-000050`: правильный hard-negative

QUERY показывает Primum Alveus IV Blanc de Blancs **BRUT** 2019. Ближайший кандидат
Каталога совпадает по производителю, линейке, IV, Blanc de Blancs и году, но является
**EXTRA BRUT**. `v5` правильно вернула `not_found` по этому SKU-значимому конфликту.

При этом evidence сжато в три observations: поле `year` достигло лимита 240 символов и
оборвалось при перечислении кандидатов. Trace подтверждает финальное решение, но не даёт
полного отдельного доказательства исключения всех 12 ELEMENT.

## Сравнение с соседними версиями

Относительно `v4` версия добавила `not_found` в динамический enum и переписала model-facing
инструкции на английский. На совпадающих 51 QUERY итоговая accuracy не изменилась:
49/51 у обеих версий. `v5` исправила hard-negative `q-000050`, который `v4` вынужденно
сопоставляла с ближайшим EXTRA BRUT, но одновременно создала ложный `not_found` на
`q-000005`. Ошибка `q-000047` осталась общей.

По сравнению с `v6` итог также одинаков — 49/51, однако `v5` существенно компактнее:

| Метрика | v5 | v6 |
|---|---:|---:|
| Model requests | 51 | 274 |
| Mean latency | 3,869 с | 13,977 с |
| Total tokens | 294 173 | 876 341 |
| Cost | $0,0853002 | $0,184976748 |

`v6` лучше аудирует каждого кандидата и на единственном gold-`not_found` получила precision
и recall 100%, но текущая выборка слишком мала для общего вывода об OOD. По инженерному
балансу уже прогнанных версий `v5` быстрее и дешевле при той же end-to-end accuracy.

Подробное сопоставление находится в
`ndr/reports/V6_VS_V5_NOT_FOUND.md`.

## Известные ограничения

- Финальный выбор, abstention и evidence генерирует одна модель; код проверяет форму, но не
  семантическую согласованность ответа.
- Три коротких observations не дают per-candidate verdict и могут обрезаться на больших
  группах.
- `not_found` смешивает доказанное отсутствие точного SKU и ошибочное чтение мелкого текста.
- В NDR gold содержит лишь один `not_found`, поэтому precision/recall нельзя переносить на
  неизвестные бренды, слабые фото и другие OOD-режимы.
- Provider generation seed не задан; temperature 0 не гарантирует полной
  воспроизводимости удалённой модели.
- Product SLA <3 с не достигнут: p50 равен 3,811 с, хотя все 51 ответа уложились в 10 секунд.

Сильная сторона `v5` — один joint call, в котором модель видит взаимоисключающие варианты
одновременно. Слабая — решение трудно проверить и невозможно программно отделить уверенный
hard conflict от ошибочного model observation.

## Offline-проверки

Из корня репозитория:

```powershell
python -B .agents/skills/near-duplicates/scripts/validate.py
python -B ndr/run/build_dataset.py --check
python -B -m unittest ndr.run.test_toolkit ndr.solutions.v5.test_predictor -v
```

Эти команды не обращаются к модели. `ndr/run/run.py` здесь намеренно не запускается:
model-backed NDR-прогоны выполняет только пользователь, а уже прогнанная версия должна
оставаться исторической.
