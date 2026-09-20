# Эксперимент v7: image-only one-shot против v5

## Итог

Основной полный прогон `v7` получил **47/51 = 92,16% end-to-end accuracy**. Это на две
правильные позиции и на **3,92 процентного пункта хуже**, чем совместимый полный прогон
`v5` с 49/51 = 96,08%.

Гипотеза v7 подтверждена только локально. Удаление карточек, slug и catalog vintage policy
исправило q-000005, где v5 выдумала конфликт `semi-sweet` против `dry`. Но взамен v7:

- потеряла два ответа, для которых image-only вход не даёт надёжного способа отделить gold от
  соседа: q-000010 и q-000035;
- перестала находить единственный настоящий `not_found`, q-000050, хотя конфликт `BRUT`
  против `EXTRA BRUT` виден на изображениях;
- не исправила прежнюю OCR-ошибку q-000047.

Таким образом, описательные карточки действительно могут привязывать OCR к кандидату, но их
полное удаление не является улучшением. Они также несут SKU-признаки, которых может не быть на
лицевом Эталоне. Результат v7 хуже по качеству и `not_found`, хотя дешевле по токенам и стоимости.

## Сравниваемые артефакты

Основной артефакт v7:

- каталог: `ndr/results/v7/20260920T053019.676398Z/`;
- run id: `ndr-v7-20260920T053019.676398Z-6f85b99d2096`;
- solution fingerprint:
  `6f85b99d209651c0b3aad4ef71cae654d56694737f76ce12ca17b1e3b5e99159`;
- статус: `completed`, 51 из 51 кейсов завершены.

Baseline v5:

- каталог: `ndr/results/v5/20260919T214443.536610Z/`;
- run id: `ndr-v5-20260919T214443.536610Z-a634c9d506e8`;
- solution fingerprint:
  `a634c9d506e8f1fd351d2308e58a64f55a7968209f3dbab9ed8c2d65095285d3`;
- статус: `completed`, 51 из 51 кейсов завершены.

Прогоны совместимы по всем контролируемым условиям сравнения:

| Условие | v5 | v7 |
|---|---|---|
| Dataset source hashes и counts | совпадают | совпадают |
| Model | `deepseek/deepseek-v4.1-flash` | тот же |
| Provider route | только `together`, fallback выключен | тот же |
| Temperature | 0.0 | 0.0 |
| Max completion tokens | 4096 | 4096 |
| Reasoning | `effort=none`, `exclude=false` | то же |
| Image detail | `original` | `original` |
| Candidate-order seed | 0 | 0 |
| Concurrency | 16 | 16 |

Экспериментальное различие — именно вход модели и prompt: v5 передавала Эталоны вместе с
компактными карточками, slug и правилами винтажа; v7 передавала QUERY и Эталоны только как
изображения с непрозрачными идентификаторами `element_N`.

## Основные метрики

| Метрика | v5 | v7 | Изменение v7 |
|---|---:|---:|---:|
| Attempted / correct | 51 / 49 | **51 / 47** | −2 correct |
| End-to-end accuracy, ошибки считаются промахами | 96,08% | **92,16%** | −3,92 п.п. |
| Conditional runner quality | 49/51 = 96,08% | **47/51 = 92,16%** | −3,92 п.п. |
| Excluded errors | 0 | **0** | 0 |
| Answered / answered accuracy | 51 / 96,08% | **51 / 92,16%** | −3,92 п.п. |
| Contract errors | 0 | **0** | 0 |
| Predictor / HTTP errors | 0 | **0** | 0 |
| `not_found` responses | 2 | **0** | −2 |
| Model requests | 51 | **51** | 0 |
| Resolver/tie-breaker requests | 0 | **0** | 0 |
| Prompt tokens | 285 954 | **238 026** | −16,76% |
| Completion tokens | 8 219 | **8 048** | −2,08% |
| Total tokens | 294 173 | **246 074** | −16,35% |
| Cost | $0,085300200 | **$0,065523384** | −23,18% |

Conditional quality совпадает с end-to-end accuracy только потому, что v7 не имела
исключённых ошибок. Это не отдельная оценка на более лёгком подмножестве.

Все 51 запроса v7 обслужены Together, вернули одну generation с `finish_reason=stop` и прошли
Pydantic-валидацию: `validation_error=null`. Ошибки v7 поэтому семантические или связаны с
идентифицируемостью данных, а не с HTTP, JSON или response contract.

## Изменившиеся ответы

Из 51 финального ответа совпали 47. Изменились четыре:

| QUERY | v5 | v7 | Эффект |
|---|---|---|---|
| q-000005 | ошибочный `not_found` | правильный Sémillon dry | исправление |
| q-000010 | правильный Syrah Reserve | соседний `Бельбек Сира` | регрессия |
| q-000035 | правильный Cabernet Franc rosé | Merlot rosé | регрессия |
| q-000050 | правильный `not_found` | Extra Brut 2019 | регрессия |

q-000047 осталась неправильной в обеих версиях: выбрано сухое вместо полусухого.

По размеру candidate group изменение распределилось ожидаемо:

| Кандидатов | v5 | v7 | Причина изменения |
|---:|---:|---:|---|
| 2 | 13/14 | 12/14 | новая ошибка q-000035; q-000047 осталась |
| 4 | 4/5 | 5/5 | исправлена q-000005 |
| 8 | 9/9 | 8/9 | новая ошибка q-000010 |
| 12 | 1/1 | 0/1 | потерян `not_found` q-000050 |

Группы размера 3, 5, 6 и 9 остались без ошибок.

## Исправленный кейс

### q-000005: удаление карточек помогло

Gold: `alma-valley-semilon-beloe-suhoe-135`.

v5 вернула `not_found`, потому что её decisive observation ошибочно утверждала:

> Query reads SEMILLON and semi-sweet; Element 2 is Semillon but dry, a hard sweetness conflict.

v7 выбрала правильный `element_2` и записала:

> QUERY shows SÉMILLON with a fish illustration and dry white wine text, matching element_2 exactly.

QUERY, Эталон и gold согласованы: видны Alma Valley, Sémillon, dry и 2023. Это прямое
свидетельство в пользу гипотезы v7: карточка кандидата могла подтолкнуть v5 к ошибочному
чтению сахара. Stage обеих версий — `select_nearest`; `finish_reason=stop`,
`validation_error=null`.

## Все неправильные ответы v7

### q-000010: reference/card inconsistency, а не чистая prompt-ошибка

- Gold: `belbek-sira-rezerv-krasnoe-suhoe-132`, то есть «Сира Резерв».
- Ответ: `belbek-belbek-sira-krasnoe-suhoe-132`, карточка «Бельбек Сира» без reserve.
- Stage: `select_nearest`.
- `finish_reason=stop`, `validation_error=null`.

Модель правильно прочитала QUERY как `СИРА РЕЗЕРВ`, но одновременно увидела ту же надпись у
`element_3` и `element_7`, после чего выбрала `element_3` как визуально ближайший:

> QUERY shows 'СИРА РЕЗЕРВ'. Element 3 and element 7 both show the same 'СИРА РЕЗЕРВ' profile.

Визуальный аудит подтверждает наблюдение: оба переданных Эталона действительно печатают
`СИРА РЕЗЕРВ` и показывают практически одинаковую композицию этикетки. При этом карточка
`element_3` называет товар «Бельбек Сира» и не имеет `aging_or_reserve`, тогда как gold-card
явно содержит «Сира Резерв». Значит, reference image для distractor противоречит его карточке
и делает image-only выбор неоднозначным.

Класс результата: **dataset/reference-card inconsistency**, проявившаяся как wrong-SKU top-1.
Ослаблять matcher или учить его выбирать gold вопреки видимому Эталону нельзя. Подтверждённый
реестр near-duplicates этим отчётом не изменяется; отдельно нужно проверить корректность
привязки Эталона к `belbek-belbek-sira-krasnoe-suhoe-132`.

### q-000035: gold не идентифицируется по доступной лицевой стороне

- Gold: `dubinin-winery-roze-kaberne-fran-rozovoe-suhoe-13`.
- Ответ: `dubinin-winery-merlo-roze-rozovoe-suhoe-125`.
- Stage: `select_nearest`.
- `finish_reason=stop`, `validation_error=null`.

Обе карточки обозначают разные вина — Cabernet Franc rosé и Merlot rosé, — но на QUERY и
обоих лицевых Эталонах крупно напечатано только `Розе` и `Вино сухое розовое`. Сорт на
доступной лицевой стороне не читается. Эталонные рендеры визуально различаются главным
образом оттенком жидкости/рендера, а номер бутылки QUERY `1266/3300` не совпадает с номером
рендеров и не связан с каталогом.

Decisive observation v7 поэтому фактически верна:

> Both show the same pink-label Dubinin Winery Rose ... no product-profile conflict.

После этого v7 выбрала первый элемент. Правильный ответ v5 также не был обоснован видимым
SKU-признаком: её observation прямо говорила, что QUERY не называет сорт, хотя карточки
различают Merlot и Cabernet Franc.

Класс результата: **visually underdetermined dataset/gold case**. Runner обязан считать его
промахом, но этот кейс не доказывает дефект prompt v7. Для честной проверки нужен оборотный
контрэтикеточный ракурс, читаемый сорт/крепость либо другой Эталон.

### q-000047: semantic/OCR wrong-SKU selection

- Gold: `millstream-av-igristoe-molodoe-polusuhoe-beloe`.
- Ответ: `millstream-av-igristoe-molodoe-suhoe-beloe`.
- Stage: `select_nearest`.
- `finish_reason=stop`, `validation_error=null`.

На QUERY и правильном Эталоне читается `МОЛОДОЕ ПОЛУСУХОЕ БЕЛОЕ`. v7 вместо этого заявила:

> QUERY and both ELEMENTs show the same product: sparkling wine, white, brut, young, with the
> Cyrillic name 'Цимлянское'.

Это наблюдение выдумывает `brut` и `Цимлянское`, отсутствующие на этикетке, и не замечает
решающий `ПОЛУСУХОЕ`. Ошибка существовала и в v5, которая читала QUERY как `СУХОЕ`.

Класс результата: **semantic false positive / OCR wrong-SKU**. Удаление карточек не устранило
проблему самостоятельного чтения мелкого текста QUERY.

### q-000050: потерян настоящий hard negative

- Gold: `not_found`.
- Ответ:
  `fanagoriya-primum-alveus-blanc-de-blancs-ekstra-bryut-2019-shardone-beloe-11-13`.
- Stage: `select_nearest`.
- `finish_reason=stop`, `validation_error=null`.

QUERY показывает Primum Alveus IV Blanc de Blancs **BRUT** 2019. Выбранный `element_2`
показывает ту же линейку и год, но **EXTRA BRUT**. Gold согласован с изображениями и
folder-derived меткой.

Ответ v7 внутренне противоречив:

> Query is Primum Alveus Blanc de Blancs Brut. Element 2 matches this exact profile; other
> Primus Alveus elements differ by Brut/Extra Brut.

Модель правильно прочитала QUERY как `BRUT`, но назвала `EXTRA BRUT` точным совпадением,
несмотря на явное prompt-правило о hard conflict. v5 с карточкой элемента корректно вернула
`not_found` по этой разнице.

Класс результата: **semantic false positive для catalog SKU / false negative класса
`not_found`**. Resolver отсутствует, поэтому противоречие checklist и selection ничем не
проверяется.

## `not_found`

В датасете только один gold-`not_found`, q-000050.

| `not_found` confusion matrix | v5 | v7 |
|---|---:|---:|
| True positive | 1 | **0** |
| False positive | 1 | **0** |
| False negative | 0 | **1** |
| True negative | 49 | **50** |
| Precision | 50% | **не определена: ответов нет** |
| Recall | 100% | **0%** |
| F1 | 66,67% | **0%** |

v7 устранила ложный отказ q-000005 не за счёт лучшей калибровки abstention, а фактически
перестав использовать `not_found` вообще. Поэтому нулевой false-positive rate не является
улучшением: система также пропустила единственный проверяемый hard negative.

Один положительный OOD-кейс всё равно слишком мал для общей оценки политики `not_found`.
Три других исходных not-found продукта не включены в NDR, потому что не имеют подтверждённого
anchor в near-duplicate component.

## Latency, SLA и цена

| Метрика | v5 | v7 | Изменение v7 |
|---|---:|---:|---:|
| Mean | 3,869 с | **4,242 с** | +9,64% |
| p50 | 3,811 с | **4,108 с** | +7,80% |
| p95 | 5,637 с | **5,785 с** | +2,63% |
| Max | 6,404 с | **6,775 с** | +5,81% |
| Wall time, concurrency 16 | 15,905 с | **16,917 с** | +6,36% |
| Throughput | 3,207 query/с | **3,015 query/с** | −5,98% |

Несмотря на уменьшение prompt tokens, v7 оказалась медленнее на этом прогоне. По одному
запуску это нельзя уверенно приписать изменению prompt: provider latency вариативна.

Целевой SLA `<3 с` не выполнен: только 2 из 51 запросов v7 уложились в 3 секунды, 49 были
медленнее. При этом все локальные model calls завершились быстрее 10 секунд; максимум 6,775 с
ещё не включает HTTP/service overhead будущего eval endpoint.

## Дополнительный одно-кейсовый артефакт

Перед полным прогоном сохранён отдельный завершённый запуск
`ndr/results/v7/20260920T053004.557845Z/` с тем же solution fingerprint:

- q-000001: 1/1 правильно;
- concurrency 1;
- latency 1,713 с;
- 1 request, 2 315 tokens, $0,000846600;
- contract/predictor errors: 0/0.

Он не смешивается с метриками полного запуска. Суммарно два артефакта v7 содержат 52 model
requests, 248 389 tokens и $0,066369984, но один QUERY в них повторяется, поэтому объединённую
accuracy считать нельзя. Такой вызов мог влиять на provider cache; у сравниваемой v5 также был
предшествующий одно-кейсовый артефакт, поэтому сравнение полных прогонов симметрично, но не
является чистым cold-cache benchmark.

## Вывод и следующая гипотеза

v7 не следует продвигать как замену v5. Она дешевле на 23,18%, но теряет качество,
единственный `not_found` и два кейса, где карточки компенсировали слабую или противоречивую
визуальную идентифицируемость.

Следующий изолированный эксперимент должен проверять не «убрать каталог», а **разделить
наблюдение и сопоставление**:

1. один раз извлечь из QUERY immutable profile без карточек и Эталонов кандидатов;
2. отдельно извлечь видимые признаки Эталонов либо использовать проверенные catalog fields;
3. сопоставлять всех кандидатов с одним и тем же QUERY profile, не перечитывая QUERY под
   влиянием конкретного кандидата;
4. программно запрещать выбор, если собственный checklist фиксирует hard conflict вроде
   `BRUT` против `EXTRA BRUT`;
5. помечать визуально неидентифицируемые и reference/card-inconsistent группы как проблемы
   данных, а не подгонять matcher под folder-derived gold.

Это сохраняет подтверждённый плюс v7 на q-000005, но не выбрасывает полезные SKU-признаки и
не требует ослаблять правила для q-000010 или q-000035.

## Проверки источников

При подготовке отчёта выполнены offline-проверки без model calls:

```text
OK: 378 confirmed pairs; 378 folders; 2103 catalog slugs
Dataset valid: 51 queries, 30 groups, 132 candidates; excluded 25 queries
(22 catalog singletons, 3 unmapped not_found).
```

Реестр `data/near_duplicates/all_candidates.csv`, C-галерея, датасет, solutions и исходные
result artifacts не изменялись.
