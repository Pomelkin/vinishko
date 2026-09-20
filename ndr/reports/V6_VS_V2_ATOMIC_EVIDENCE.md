# NDR: v6 против v2 — атомарное evidence

Дата анализа: 2026-09-20.

## Краткий вывод

v6 создана как отдельная версия на основе v2 и запущена ровно один раз. На строго
совпадающих 51 QUERY она получила **49/51 = 96,08%** против **48/51 = 94,12%** у v2.
Ошибок predictor/HTTP/contract нет; все ошибки учитываются как промахи, поэтому conditional
quality accuracy в обоих прогонах совпадает с end-to-end accuracy.

Основная гипотеза подтверждена частично. Декомпозиция profile на восемь различителей и
состояния match/conflict/unknown:

- устранила единственный false negative v2: TP выросли 49 → 50, FN снизились 1 → 0;
- снизила pairwise FP 18 → 13, precision same выросла с 73,13% до 79,37%;
- снизила resolver-нагрузку с 13/51 до 9/51;
- исправила q-000023 с невидимым maker и настоящий not_found q-000050.

Но заданные цели FP ≤ 7 и resolver queries ≤ 6 не достигнуты. Два итоговых промаха
q-000002 и q-000047 имеют одну причину: независимые candidate-conditioned вызовы
противоречиво читают один QUERY, после чего resolver выбирает неправильное чтение. Атомарная
схема сделала дефект аудируемым, но сама по себе не устранила его.

## Гипотеза и изменения

Проверяемая гипотеза: атомарный checklist с явным unknown уменьшит ложные совпадения
v2, не создавая false negative из-за обрезанного текста или отсутствующей metadata.

Сохранены pairwise pipeline и resolver v2, модель и route. Изменения v6:

1. Все model-facing инструкции переписаны на английском.
2. Каждый comparison возвращает ровно восемь полей:
   maker, product_name, line_variant, grape, color, sweetness,
   carbonation, vintage.
3. Для каждого поля обязательны query_value, element_value и
   state ∈ {match, conflict, unknown}.
4. Model-generated общий verdict удалён. Итог same/different вычисляет код:
   - любой активный conflict отклоняет кандидата;
   - unknown не является конфликтом;
   - положительный ответ требует product identity anchor и дополнительного match-evidence;
   - fallback допускается для maker + grape + один style-признак.
5. Сохранены две vintage-ветки:
   - при непустом catalog vintage точный vintage=match обязателен;
   - при пустом catalog vintage состояние всё равно записывается, но полностью исключается
     из verdict.
6. Pydantic остаётся единственным источником JSON Schema. Одна точная schema передаётся и
   в response_format, и в system message; этот же класс валидирует ответ.
7. В соответствии с experiment contract v6 не делает retry (MAX_429_RETRIES=0).

## Preflight

Первый preflight завершился до model requests и обнаружил:

- устаревший ndr/dataset/excluded.jsonl после замены изображения одного исключённого
  singleton;
- отсутствие новой v6 в уже расширенной для v5 predictor-матрице toolkit.

Generated dataset был пересобран. Изменился только путь и SHA исключённого singleton
esse-prirodno-polusladkoe-krasnoe-merlo-13; manifest.jsonl, groups.jsonl и
catalog.jsonl не изменились. После минимального добавления v6 в матрицу прошли:

    python -B .agents/skills/near-duplicates/scripts/validate.py
    OK: 378 confirmed pairs; 378 folders; 2103 catalog slugs

    python -B ndr/run/build_dataset.py --check
    Dataset valid: 51 queries, 30 groups, 132 candidates; excluded 25 queries
    (22 catalog singletons, 3 unmapped not_found).

    python -B -m unittest ndr.run.test_toolkit ndr.solutions.v6.test_predictor -v
    Ran 42 tests ... OK

No-network тесты отдельно проверяют все восемь полей, enum из трёх состояний, запрет
старого boolean-контракта, hard conflict, unknown maker, обе vintage-ветки,
недостаточное evidence, deterministic verdict и отсутствие retry.

## Прогоны и сопоставимость

| Версия | Result | Run ID | Fingerprint |
|---|---|---|---|
| v2 | ndr/results/v2/20260919T202801.093277Z | ndr-v2-20260919T202801.093277Z-c71b714e760d | c71b714e760d033ad340e87970771f04cb24b0ed38f02ad89d8702cbf5591fd7 |
| v6 | ndr/results/v6/20260919T213900.872958Z | ndr-v6-20260919T213900.872958Z-692eafa1671f | 692eafa1671f41a9580b758e927431b0eb1aeb791cf227911183480355e1d45a |

По всем 51 case-файлам совпадают:

- SHA-256 QUERY;
- expected_slug;
- group_id;
- полный candidate_order.

Совпадают модель deepseek/deepseek-v4.1-flash, provider Together,
allow_fallbacks=false, temperature=0, reasoning none,
max_completion_tokens=4096, image_detail=original, timeout 60 с, concurrency 16,
runner seed 0 и hashes трёх source CSV.

Остающиеся ограничения причинного сравнения:

- provider generation seed не задан;
- прогоны выполнены в разное время, поэтому состояние удалённого сервиса не контролируется;
- source v2 допускает 429 retry, v6 — нет. В обоих сравниваемых прогонах retry не было,
  поэтому ветка не повлияла на наблюдаемый результат;
- prompt, schema и deterministic aggregation изменены вместе, как прямо запросил
  эксперимент.

## Итоговые метрики

| Метрика | v2 | v6 | Изменение |
|---|---:|---:|---:|
| Attempted | 51 | 51 | 0 |
| Correct, end-to-end | 48 | 49 | +1 |
| End-to-end accuracy, errors = misses | 94,12% | **96,08%** | +1,96 п.п. |
| Runner quality, условная | 48/51 = 94,12% | 49/51 = 96,08% | +1,96 п.п. |
| Excluded errors | 0 | 0 | 0 |
| Answered (status=ok) | 51 | 51 | 0 |
| Answered accuracy | 48/51 | 49/51 | +1 |
| not_found responses | 1 | 1 | 0 |
| Contract / predictor errors | 0 / 0 | 0 / 0 | 0 |
| Model requests | 278 | 274 | −4 (−1,44%) |
| Successful / failed requests | 278 / 0 | 274 / 0 | — |
| Resolver QUERY | 13/51 = 25,49% | **9/51 = 17,65%** | −4 |
| Latency mean | 17,000 с | 13,977 с | −17,78% |
| Latency p50 | 12,769 с | 12,083 с | −5,37% |
| Latency p95 | 49,739 с | 28,026 с | −43,65% |
| Latency max | 54,174 с | 30,527 с | −43,65% |
| Wall time | 85,374 с | 64,214 с | −24,79% |
| Throughput | 0,597 q/s | 0,794 q/s | +32,95% |
| Prompt tokens | 584 500 | 789 985 | +35,16% |
| Completion tokens | 38 016 | 86 356 | +127,16% |
| Total tokens | 622 516 | 876 341 | +40,77% |
| Cost | $0,12632472 | $0,184976748 | +46,43% |

Все 274 generation v6 завершились finish_reason=stop, у всех
validation_error=null; все обслужил Together. Retry attempts: 0.

Latency стала лучше при почти неизменном числе calls, но это нельзя уверенно приписать
prompt/schema: существенна неконтролируемая вариативность внешнего сервиса. Рост tokens и
стоимости причинно согласуется с восемью обязательными evidence-блоками.

### not_found

В gold один not_found, q-000050.

- v2: один ответ not_found, но на catalog-кейсе q-000023; precision 0/1, recall 0/1.
- v6: один ответ not_found, на правильном q-000050; precision 1/1, recall 1/1.
- Catalog-only accuracy одинакова: 48/50 = 96%.

Финальное улучшение +1 складывается из двух исправлений и одной регрессии:

| QUERY | v2 | v6 | Эффект |
|---|---|---|---|
| q-000002 | правильное полусладкое | неправильное полусухое | регрессия |
| q-000023 | not_found | правильный Cuvée №2 | исправление |
| q-000050 | неправильный Extra Brut | not_found | исправление |

Остальные 48 top-1 ответов не изменились.

## Pairwise-качество

| Метрика | v2 | v6 | Изменение |
|---|---:|---:|---:|
| TP | 49 | **50** | +1 |
| FN | 1 | **0** | −1 |
| FP | 18 | **13** | −5 |
| TN | 197 | 202 | +5 |
| Comparison errors | 0 | 0 | 0 |
| Precision same | 73,13% | **79,37%** | +6,24 п.п. |
| Recall same | 98,00% | **100%** | +2,00 п.п. |
| F1 same | 83,76% | **88,50%** | +4,74 п.п. |
| Resolver correct / wrong | 12 / 1 | 7 / 2 | — |

Цели гипотезы:

- FP ≤ 7: **не достигнута**, получено 13;
- resolver QUERY ≤ 6: **не достигнута**, получено 9;
- gold TP ≥ 48: **достигнута**, получено 50;
- без новых FN из-за unknown: **достигнута**, FN = 0.

## Исправленные кейсы

### q-000023: unknown maker больше не является конфликтом

Для gold-кандидата model evidence:

- maker=unknown: maker не виден на QUERY;
- product_name=match: Cuvée;
- line_variant=match: No. 2 Cuvée Blanc de Noirs;
- carbonation=match: Méthode Champenoise / sparkling.

Deterministic aggregation дала same; два соседних Cuvée получили
line_variant=conflict. Единственный match выбран без resolver. Это прямое подтверждение
основной части гипотезы.

### q-000050: BRUT / EXTRA BRUT стал hard conflict

У ближайшего Extra Brut 2019 совпали maker, product, line variant и vintage, но:

    sweetness:
      query_value   = BRUT
      element_value = экстра брют / EXTRA BRUT
      state         = conflict

Все 12 кандидатов получили different, итог — правильный not_found. finish_reason=stop,
validation_error=null.

## Все неправильные ответы v6

Ошибок HTTP/predictor/contract нет. Оба промаха — semantic false positive на comparison,
затем resolver error. Во всех model calls finish_reason=stop,
validation_error=null.

### q-000002

Gold: ...polusladkoe...; ответ: ...polusuhoe....

- comparison с gold читает QUERY как ПОЛУСЛАДКОЕ и ставит sweetness=match;
- независимый comparison с неправильным кандидатом читает тот же QUERY как
  ПОЛУСУХОЕ и также ставит sweetness=match;
- оба кандидата проходят как same;
- resolver возвращает только неправильный slug, без evidence.

На QUERY мелкий текст трудночитаем, но gold-Эталон явно содержит ПОЛУСЛАДКОЕ; это не
dataset/gold inconsistency. Класс: semantic FP + resolver error.

### q-000047

Gold: ...polusuhoe...; ответ: ...suhoe....

- comparison с неправильным кандидатом читает QUERY как СУХОЕ;
- comparison с gold читает тот же QUERY как ПОЛУСУХОЕ;
- оба получают sweetness=match и проходят как same;
- resolver выбирает сухое.

На QUERY видна строка ПОЛУСУХОЕ БЕЛОЕ, правильный Эталон и карточка также полусухие.
Это подтверждённая semantic/OCR ошибка, не проблема gold. Класс: semantic FP + resolver
error.

## Почему FP не снизились до цели

13 FP распределены по девяти QUERY. Наиболее показательный новый кластер — q-000042:
один QUERY признан совпадающим ещё с четырьмя Dekanter-кандидатами. В независимых calls
query_value каждый раз меняется вместе с ELEMENT: Cabernet Sauvignon 2018, Saperavi 2017,
Merlot 2018, Cabernet Franc 2019. Это прямое candidate leakage: модель переносит признаки
ELEMENT в якобы независимо прочитанный QUERY, несмотря на явный запрет в prompt.

Та же причина видна в двух финальных промахах по sweetness. Следовательно, prompt-only
усиление pairwise-схемы достигло предела: пока QUERY читается заново рядом с каждым
ELEMENT, взаимоисключающие OCR-наблюдения не сравниваются между calls.

## Вывод и следующая гипотеза

v6 — успешное, но частичное улучшение v2: end-to-end +1, FN устранены, настоящий
not_found работает, FP и resolver load снижены. Цена — +40,77% total tokens и +46,43%
стоимости; SLA <3 с по-прежнему не достигнут (p50 12,08 с).

Следующий изолированный эксперимент не должен снова усиливать этот же pairwise prompt.
Нужно один раз извлечь QUERY evidence без изображений и карточек кандидатов, зафиксировать
его в trace, а затем сравнивать все ELEMENT с одним неизменным набором
maker/product_name/line_variant/grape/color/sweetness/carbonation/vintage.
Альтернатива — один joint family rerank с тем же атомарным контрактом и поддержкой
not_found. Оба варианта устраняют возможность прочитать один QUERY одновременно как
ПОЛУСЛАДКОЕ и ПОЛУСУХОЕ.

Исходники v6 после прогона не изменялись; подтверждённый near-duplicate registry и
существующие result artifacts не модифицировались.
