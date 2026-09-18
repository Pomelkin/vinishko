# Проект API-контракта

Это предложение для будущего backend, а не описание работающего сервиса. Полные TypeScript-типы и схемы валидации Zod находятся в `src/shared/api/contracts.ts`.

| Метод и путь | Запрос | Ответ |
| --- | --- | --- |
| `POST /api/recognize` | `multipart/form-data`, поле `image`, JPEG после нормализации ориентации | `RecognitionResponse` |
| `GET /api/wines/:slug` | URL-encoded slug | `WineDetails` |
| `GET /api/wines?q=...` | строка поиска; пустая строка означает каталог | `WineSummary[]` |

Все три метода принимают `AbortSignal`. Отмена fetch прекращает ожидание клиента; backend сам решает, как останавливать вычисления. HTTP-ошибки не показываются пользователю как JSON. 404 карточки, отсутствие сети, серверная ошибка, повреждённый JSON и неверная схема имеют понятные сообщения.

## Распознавание

```ts
interface RecognitionResponse {
  schemaVersion: '1.0';
  requestId: string;
  image: {
    width: number;
    height: number;
    coordinateSpace: 'normalized';
    orientationApplied: true;
  };
  bestMatch: { slug: string } | null;
  metrics: {
    f1Top1: number | null;
    f1Top5: number | null;
    scope: 'evaluation-dataset';
    datasetId: string | null;
    isMock: boolean;
  };
  detections: BottleDetection[];
  processingTimeMs: number | null;
}
```

Каждая `BottleDetection` содержит стабильный `id`, полигон **всей бутылки** в координатах `[0, 1]` и `detectionConfidence: number | null`. Минимум 3 точки, ненулевая площадь. Максимум 100 объектов, 4096 точек на объект. Полигоны должны описывать простой замкнутый силуэт без отверстий (замыкающий сегмент подразумевается). Чем грубее геометрия backend, тем грубее выделение; frontend не выполняет сегментацию.

- `status: 'matched'`: `match: {slug, wine: WineSummary, matchConfidence}`, отдельно `similar: Candidate[]`.
- `status: 'unmatched'`: `match: null`, `similar: Candidate[]`. Собственного выдуманного slug у неизвестной бутылки нет.
- `Candidate`: `{slug, rank, similarityScore: number | null, wine: WineSummary}`. `slug` равен `wine.slug`; rank — положительное целое число.
- Все confidence/similarity и F1 — числа от 0 до 1 либо `null`.
- `image.width/height` должны совпадать с размерами нормализованного файла, отправленного клиентом. Нельзя незаметно повернуть или обрезать изображение на backend и вернуть координаты относительно другой версии.
- `detections: []` означает, что объектов не найдено; это успешный ответ, а не техническая ошибка.
- `bestMatch` относится ко всему фото; frontend не использует его для автоматического перехода мимо попапа.

**F1 — качество на размеченной выборке, не вероятность правильности отдельной бутылки.** F1 хранится отдельно от confidence и не вычисляется на клиенте. Семантику top-1/top-5, используемую выборку и протокол оценки нужно согласовать с организатором. Mock возвращает `null` для обеих метрик и confidence.

## Данные вина

`WineSummary`: slug, name, producer, imageUrl (локальный путь или http/https, либо null), region, grapeVarieties[], color, category, vintage, shortDescription, ratings.

`ratings.community` и `ratings.roskachestvo` независимы; каждое значение — `{value, scaleMax}` либо `null`. `value` не превышает `scaleMax`. В UI подпись источника и шкала сохраняются. Отсутствующие значения не превращаются в ноль.

`WineDetails` дополнительно содержит description, alcoholPercent (0–100 либо null), servingTemperature, foodPairings[], similarWines[]. Строковые отсутствующие свойства передаются как `null`, массивы — `[]`. `slug` — непустой kebab-case идентификатор.

## Оценочный скрипт хакатона

Плоский ответ `{"slug":"wine-slug"}` из PDF — отдельный формат. Будущий backend или отдельный endpoint оценки должен возвращать его для скрипта организатора. Он не заменяет расширенный контракт UI. Frontend не реализует этот сервер и не заявляет выполнение end-to-end требований хакатона.
