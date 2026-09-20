# NDR: v6 против v5 — качество и `not_found`

Дата анализа: 2026-09-20.

## Краткий вывод

На полном парном наборе из 51 QUERY версии показали одинаковый итог: **49/51 = 96,08%**.
Ошибок HTTP, predictor или response contract не было, поэтому conditional quality accuracy
runner в обоих случаях совпадает с честной end-to-end accuracy.

При этом результат устроен по-разному:

- v6 устранила единственный ложный `not_found` v5 на q-000005 и сохранила правильный
  `not_found` на q-000050;
- одновременно v6 сломала q-000002, который v5 решала правильно;
- общий catalog-only результат поэтому не изменился: **48/50 = 96%** у обеих версий;
- на наблюдаемом `not_found` v5 получила precision 50% и recall 100%, v6 — 100% и 100%,
  но gold содержит всего **один** такой QUERY, поэтому это доказательство исправления двух
  конкретных кейсов, а не надёжная оценка OOD-качества;
- v6 делает решение `not_found` намного лучше аудируемым, но стоит **274 model requests
  вместо 51**, почти втрое больше токенов, в 2,17 раза больше денег и в 3,61 раза больше
  mean latency;
- при одинаковой точности v5 существенно практичнее по SLA. У v6 30/51 QUERY заняли больше
  10 секунд, тогда как у v5 таких не было.

Итог: **v6 лучше именно на двух проверенных аспектах `not_found` — не отвергает Semillon и
правильно отвергает ближайший Extra Brut, — но не лучше v5 как решение целиком**. Главный
оставшийся дефект v6 — QUERY читается заново рядом с каждым кандидатом и получает
взаимоисключающие значения. Это уже привело к обеим финальным ошибкам v6 и делает будущие
ложные `not_found` возможными.

## Какие прогоны сравниваются

| Версия | Полный result | Run ID | Fingerprint |
|---|---|---|---|
| v5 | `ndr/results/v5/20260919T214443.536610Z` | `ndr-v5-20260919T214443.536610Z-a634c9d506e8` | `a634c9d506e8f1fd351d2308e58a64f55a7968209f3dbab9ed8c2d65095285d3` |
| v6 | `ndr/results/v6/20260919T213900.872958Z` | `ndr-v6-20260919T213900.872958Z-692eafa1671f` | `692eafa1671f41a9580b758e927431b0eb1aeb791cf227911183480355e1d45a` |

Во всех 51 парах совпадают SHA-256 QUERY, `expected_slug`, `group_id` и полный
`candidate_order`. Также совпадают хеши трёх источников датасета:

- `data/near_duplicates/all_candidates.csv`;
- `data/near_duplicates/not_found_candidates.csv`;
- `data/strapi/catalog_dataset.csv`.

Совпадают модель `deepseek/deepseek-v4.1-flash`, provider Together,
`allow_fallbacks=false`, temperature 0, reasoning `none`, `max_completion_tokens=4096`,
`image_detail=original`, timeout 60 секунд, concurrency 16 и runner seed 0. Все 325 calls
двух полных прогонов получили HTTP 200 от Together; все generations завершились
`finish_reason=stop`, все `validation_error=null`, retry attempts отсутствуют.
Текущие восемь source-файлов каждой версии совпадают с записанными в её `run.json` хешами:
после прогонов v5 и v6 не изменялись.

Ограничения причинного сравнения:

- provider generation seed не задан, а temperature 0 не гарантирует полной
  воспроизводимости удалённой модели;
- прогоны сделаны в разное время, причём v6 раньше v5;
- одновременно изменены архитектура вызовов, prompt, schema и deterministic aggregation;
  результат нельзя приписать одному отдельному текстовому правилу.

### Дополнительный частичный прогон v5

До полного прогона существует отдельный завершённый артефакт
`ndr/results/v5/20260919T214421.960697Z` с тем же fingerprint, но только с q-000001.
Он дал правильный ответ: 1/1, один request, 2 763 tokens, $0,0009864 и 2,159 секунды.
Из парного сравнения он исключён, потому что его область — один QUERY, а не 51.

Это означает, что v5 как версия фактически вызывалась больше одного раза, а q-000001 был
повторён. Частичный call также прогрел provider cache: в полном q-000001 было 2 560 cached
prompt tokens. Поэтому стоимость полного v5 не является чистым cold-cache измерением.

## Чем архитектурно отличаются версии

### v5: один joint selection

v5 отправляет QUERY и сразу всю группу кандидатов одним model call. Structured output содержит:

- три кратких observations: `maker`, `profile`, `year`;
- `slug` из динамического enum всех кандидатов плюс `not_found`.

То есть модель сама одновременно сравнивает группу, решает, все ли варианты исключены, и
возвращает финальный slug. На каждый QUERY всегда приходится ровно один request.

### v6: pairwise evidence, кодовый verdict, resolver

v6 делает отдельный call для каждого кандидата. Каждый call обязан вернуть восемь атомарных
полей — maker, product name, line variant, grape, color, sweetness, carbonation и vintage —
с `query_value`, `element_value` и состоянием `match/conflict/unknown`.

После валидации код вычисляет `same/different`:

1. релевантный `conflict` отвергает кандидата;
2. при непустом catalog vintage требуется именно `vintage=match`;
3. без конфликта всё равно нужен положительный identity anchor и дополнительный match;
4. ноль `same` даёт программный `not_found`;
5. один `same` возвращается напрямую;
6. несколько `same` отправляются в отдельный resolver, который может выбрать только один
   переданный slug и не может вернуть `not_found`.

Важная семантическая разница: **в v6 `not_found` вообще не генерируется моделью и не входит
в model schema**. Его возвращает Python-код, если ни один pairwise call не прошёл порог `same`.

Это улучшает трассировку, но создаёт проблему: `not_found` означает не только «у каждого
кандидата есть уверенный hard conflict». Он также возможен, когда у всех кандидатов просто
недостаточно положительных evidence или у vintage-кандидата год остался `unknown`. Тем самым
один внешний ответ смешивает два разных состояния:

- подтверждённое отсутствие точного SKU в группе;
- недостаточно читаемый QUERY, чтобы подтвердить какой-либо SKU.

Для продуктового API это должны быть разные внутренние причины, даже если внешний контракт
в итоге требует свести их к одному ответу.

## Итоговые метрики

| Метрика | v5 | v6 | Изменение v6 к v5 |
|---|---:|---:|---:|
| Attempted | 51 | 51 | 0 |
| Correct, end-to-end | 49 | 49 | 0 |
| End-to-end accuracy, errors = misses | **96,08%** | **96,08%** | 0 п.п. |
| Runner quality accuracy, условная | 49/51 = 96,08% | 49/51 = 96,08% | 0 п.п. |
| Excluded errors | 0 | 0 | 0 |
| Answered (`status=ok`) | 51 | 51 | 0 |
| Answered accuracy | 49/51 | 49/51 | 0 |
| `not_found` responses | 2 | 1 | −1 |
| Contract errors | 0 | 0 | 0 |
| Predictor errors | 0 | 0 | 0 |
| Model requests | 51 | 274 | +223; ×5,37 |
| Resolver QUERY | 0 | 9/51 = 17,65% | +9 |
| Latency mean | 3,869 с | 13,977 с | +261,26%; ×3,61 |
| Latency p50 | 3,811 с | 12,083 с | +217,08%; ×3,17 |
| Latency p95 | 5,637 с | 28,026 с | +397,22%; ×4,97 |
| Latency max | 6,404 с | 30,527 с | +376,72%; ×4,77 |
| QUERY ≤ 3 с | 4/51 | 0/51 | −4 |
| QUERY > 10 с | 0/51 | 30/51 | +30 |
| Wall time при concurrency 16 | 15,905 с | 64,214 с | +303,73%; ×4,04 |
| Throughput | 3,207 q/s | 0,794 q/s | −75,23% |
| Prompt tokens | 285 954 | 789 985 | +176,26% |
| Completion tokens | 8 219 | 86 356 | +950,69% |
| Total tokens | 294 173 | 876 341 | +197,90%; ×2,98 |
| Cost | $0,0853002 | $0,184976748 | +116,85%; ×2,17 |

У v6 265 pairwise comparisons и 9 resolver calls. При этом prompt cost вырос только на 7,84%
за счёт provider caching, а completion cost — в 10,51 раза: восемь обязательных evidence-блоков
делают ответы значительно длиннее.

Рост latency закономерно зависит от размера группы:

| Кандидатов | Mean v5 | Mean v6 | Коэффициент |
|---:|---:|---:|---:|
| 2 | 3,505 с | 6,087 с | ×1,74 |
| 4 | 3,354 с | 11,257 с | ×3,36 |
| 8 | 3,794 с | 20,520 с | ×5,41 |
| 9 | 4,815 с | 24,613 с | ×5,11 |
| 12 | 4,291 с | 28,026 с | ×6,53 |

Обе версии не достигают целевого SLA <3 секунд по p50. v6 дополнительно не укладывается в
10-секундный timeout внешнего eval-контракта на 30 из 51 локально измеренных QUERY ещё до
возможного HTTP overhead сервиса.

## `not_found`: точные результаты

В gold всего один `not_found`: q-000050. Остальные 50 QUERY соответствуют позициям Каталога.

| Матрица `not_found` | v5 | v6 |
|---|---:|---:|
| True positive | 1 | 1 |
| False positive / ложный отказ от catalog SKU | 1 | 0 |
| False negative / выбран catalog SKU для OOD | 0 | 0 |
| True negative | 49 | 50 |
| Precision | 1/2 = 50% | 1/1 = 100% |
| Recall | 1/1 = 100% | 1/1 = 100% |
| F1 | 66,67% | 100% |
| Catalog false-abstention rate | 1/50 = 2% | 0/50 = 0% |

### q-000005: ложный `not_found` v5 исправлен

Gold: `alma-valley-semilon-beloe-suhoe-135`.

QUERY и Эталон согласованы: на QUERY читаются Alma Valley, Sémillon, 2023 и «белое сухое»;
это не dataset/gold inconsistency.

v5, stage `select_nearest`, вернула `not_found`. Решающее model observation:

> Query reads SEMILLON and semi-sweet; Element 2 is Semillon but dry, a hard sweetness conflict.

То есть модель правильно узнала Sémillon, но ошибочно прочитала сахар и отвергла точный SKU.
Call завершён `finish_reason=stop`, `validation_error=null`. Класс ошибки: semantic false
negative, выраженный как ложный `not_found`.

v6 исправила финальный ответ. Для правильного кандидата она записала:

- `product_name`: SÉMILLON = Семильон, `match`;
- `grape`: SÉMILLON = Семильон, `match`;
- `sweetness`: «БЕЛОЕ СУХОЕ ВИНО» = сухое, `match`;
- verdict: `same`.

Три остальных кандидата получили `different` по конфликту сорта/названия, поэтому resolver
не понадобился.

Однако исправление не полностью устойчиво: в независимом call против Colombard та же v6
прочитала QUERY уже как «ПОЛУСУХОЕ» и поставила sweetness `match`. Этот кандидат всё равно
был отвергнут по Sémillon против Colombard, но наблюдение показывает candidate-conditioned
OCR: значение QUERY меняется в зависимости от стоящего рядом ELEMENT.

### q-000050: настоящий `not_found` найден обеими версиями

QUERY: Primum Alveus, Roman numeral IV, Blanc de Blancs, **BRUT**, 2019. Ближайший кандидат
Каталога совпадает по maker, линейке, IV, Blanc de Blancs и 2019, но является **EXTRA BRUT**.
Это реальный hard negative и согласованный gold.

v5 в одном joint call вернула правильный `not_found`. Решающий observation явно фиксирует
BRUT против EXTRA BRUT. Но evidence сжато в три строки; поле `year` достигло максимума в
240 символов и оборвалось на `Element 8/9`, поэтому trace не доказывает по отдельности
исключение всех 12 кандидатов.

v6 выполнила 12 отдельных сравнений и каждому кандидату дала `different`. Для ближайшего
кандидата evidence особенно чистое:

- maker: `match`;
- product name: `match`;
- line variant: Blanc de Blancs, IV — `match`;
- vintage: 2019 — `match`;
- sweetness: BRUT против EXTRA BRUT — `conflict`.

Resolver не вызывался: `same_slugs` пуст, после чего код вернул `not_found`. Все 12 calls:
`finish_reason=stop`, `validation_error=null`.

Цена более полного доказательства:

| q-000050 | v5 | v6 |
|---|---:|---:|
| Requests | 1 | 12 |
| Total tokens | 15 595 | 48 454 |
| Cost | $0,004648008 | $0,009753768 |
| Latency | 4,291 с | 28,026 с |

### Почему метрика `not_found` пока слишком слабая

В `data/test` есть четыре исходных not-found продукта, но в NDR включён только один
с подтверждённым anchor в near-duplicate component. Три других исключены с причиной
`no_confirmed_near_duplicate_mapping`:

- 19 Crimes Red Blend;
- Jacob's Creek Reserve Shiraz Barossa 2013;
- Martini Asti.

Текущая проверка поэтому измеряет один узкий вид OOD: почти точную копию catalog-кандидата с
разницей BRUT/EXTRA BRUT. Она не измеряет незнакомого производителя, отсутствие читаемого
текста, слабый crop, несколько бутылок, неизвестный vintage и случаи без близкого визуального
anchor. Значения 100% precision/recall v6 нельзя экстраполировать на эти режимы.

## Какие финальные ответы изменились

Из 51 top-1 predictions совпали 49. Изменились только два:

| QUERY | v5 | v6 | Эффект |
|---|---|---|---|
| q-000002 | правильное полусладкое | неправильное полусухое | регрессия v6 |
| q-000005 | ложный `not_found` | правильный Sémillon сухое | исправление v6 |

q-000047 остался неправильным в обеих версиях: обе выбрали сухое вместо полусухого.

## Все неправильные ответы

### v5 q-000005

- Ожидалось: Alma Valley Sémillon сухое.
- Получено: `not_found`.
- Stage: `select_nearest`.
- Причина: ошибочное чтение sweetness как semi-sweet и hard conflict с правильным dry SKU.
- `finish_reason=stop`, `validation_error=null`.
- Класс: semantic false negative / ложный `not_found`.

### v5 q-000047

- Ожидалось: `millstream-av-igristoe-molodoe-polusuhoe-beloe`.
- Получено: `millstream-av-igristoe-molodoe-suhoe-beloe`.
- Stage: `select_nearest`.
- Model observation утверждает, что QUERY читается как «МОЛОДОЕ СУХОЕ БЕЛОЕ».
- На QUERY и правильном Эталоне указано полусухое; gold согласован.
- `finish_reason=stop`, `validation_error=null`.
- Класс: semantic/OCR wrong-SKU selection.

### v6 q-000002

- Ожидалось: Абрау-Дюрсо «Русское Игристое» полусладкое.
- Получено: полусухое.
- Pairwise call с gold прочитал QUERY как «ПОЛУСЛАДКОЕ» и дал `same`.
- Независимый call с неправильным кандидатом прочитал тот же QUERY как «ПОЛУСУХОЕ» и также
  дал `same`.
- Resolver получил оба slug, вернул только неправильный slug и не обязан был объяснять выбор.
- Оба comparison calls и resolver: `finish_reason=stop`, `validation_error=null`.
- Класс: два semantic false positive из-за candidate-conditioned OCR, затем resolver error.

### v6 q-000047

- Ожидалось: MILLSTREAM AV полусухое.
- Получено: сухое.
- Call с сухим кандидатом прочитал QUERY как «СУХОЕ»; call с gold — как «ПОЛУСУХОЕ».
- Оба получили `same`, resolver выбрал сухое без evidence-полей.
- Оба comparison calls и resolver: `finish_reason=stop`, `validation_error=null`.
- Класс: два взаимоисключающих чтения QUERY и resolver error.

Ошибок provider, HTTP, JSON parsing, Pydantic contract или gold inconsistency в этих четырёх
случаях нет.

## Диагностика pairwise-слоя v6

Для 265 candidate comparisons:

| Метрика | v6 |
|---|---:|
| Gold candidate `same` | 50 |
| Gold candidate `different` | 0 |
| Wrong candidate `same` | 13 |
| Wrong candidate `different` | 202 |
| Precision класса `same` | 50/63 = 79,37% |
| Recall класса `same` | 50/50 = 100% |
| F1 класса `same` | 88,50% |

Все 13 false-positive `same` распределены по девяти QUERY; именно эти девять вызвали resolver.
Resolver выбрал правильно 7 раз и неправильно 2 раза — на q-000002 и q-000047.

Наиболее показательная системная проблема: v6 не извлекает QUERY evidence один раз. Каждый
candidate-conditioned call заново распознаёт один и тот же QUERY, поэтому `query_value` может
подстраиваться под ELEMENT. А resolver schema содержит только slug и не оставляет evidence,
по которому можно проверить финальный выбор.

## Вывод и следующая гипотеза

Если выбирать между уже прогнанными версиями как целиком готовыми решениями, **v5 лучше по
инженерному балансу**: та же accuracy, в 5,37 раза меньше calls, в 2,17 раза дешевле и в
3,61 раза быстрее по mean latency. v6 ценна как диагностический эксперимент: она устранила
ложный `not_found`, дала полное доказательство настоящего `not_found` и показала точную
причину оставшихся ошибок.

Следующий изолированный эксперимент должен сохранить сильную сторону v6, но убрать повторное
чтение QUERY:

1. один раз извлечь из QUERY независимый immutable profile без изображений и карточек
   кандидатов;
2. сравнивать всех кандидатов с этим одним profile, не позволяя `query_value` меняться между
   calls;
3. возвращать подтверждённый `not_found` только когда каждый кандидат имеет явный hard
   conflict;
4. отсутствие положительного evidence без конфликта учитывать отдельно как `uncertain`, а не
   автоматически приравнивать к отсутствию товара;
5. resolver должен получать и возвращать evidence, особенно по sweetness, vintage и
   line variant;
6. расширить OOD-проверку: одного q-000050 недостаточно для выбора политики `not_found`.

Текущий датасет и реестр при подготовке отчёта не менялись. Offline-проверки проходят:

```text
OK: 378 confirmed pairs; 378 folders; 2103 catalog slugs
Dataset valid: 51 queries, 30 groups, 132 candidates; excluded 25 queries
(22 catalog singletons, 3 unmapped not_found).
```
