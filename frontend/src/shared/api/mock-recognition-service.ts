import type {
  RecognitionService,
  RecognitionResponse,
  BottleDetection,
  Point,
} from "./contracts";
import { wines, summary, details } from "../../mocks/catalog";
import polygons from "../../mocks/polygons.json";
import { UserError } from "./errors";
import { requestId } from "./request-id";
import { decodeImage, prepareImage } from "../../features/image-input/image";
export const demoScenarios = [
  ["single", "Одна найденная бутылка"],
  ["multiple", "Две найденные бутылки"],
  ["mixed", "Найденная и похожие варианты"],
  ["unmatched", "Бутылка с похожими винами"],
  ["no-similar", "Без похожих вариантов"],
  ["empty", "На фото нет бутылок"],
  ["slow", "Долгое ожидание"],
  ["error", "Техническая ошибка"],
  ["cancel", "Отмена и повторный запуск"],
] as const;
export type DemoScenario = (typeof demoScenarios)[number][0];
export function wait(ms: number, signal?: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    if (signal?.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    const done = () => {
      signal?.removeEventListener("abort", abort);
      resolve();
    };
    const timer = setTimeout(done, ms);
    const abort = () => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", abort);
      reject(new DOMException("Aborted", "AbortError"));
    };
    signal?.addEventListener("abort", abort, { once: true });
  });
}
export class MockRecognitionService implements RecognitionService {
  private prepared = new WeakMap<Blob, DemoScenario>();
  async createDemo(scenario: DemoScenario) {
    const multiple = scenario === "multiple" || scenario === "mixed";
    let source: Blob;
    if (scenario === "empty")
      source = new Blob(
        [
          '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="1000"><rect width="900" height="1000" fill="#e9dfcf"/><path d="M0 710H900V1000H0Z" fill="#d9c8ad"/></svg>',
        ],
        { type: "image/svg+xml" },
      );
    else
      source = await (
        await fetch(`/assets/demo-${multiple ? "multiple" : "single"}.svg`)
      ).blob();
    const prepared = await prepareImage(source, true);
    this.prepared.set(prepared.blob, scenario);
    return prepared;
  }
  async recognize(
    image: Blob,
    options?: { signal?: AbortSignal },
  ): Promise<RecognitionResponse> {
    const scenario = this.prepared.get(image);
    await wait(
      scenario === "slow" ? 14000 : scenario === "cancel" ? 7000 : 650,
      options?.signal,
    );
    if (scenario === "error")
      throw new UserError(
        "Сервис временно недоступен. Фотография сохранена — можно повторить попытку.",
      );
    const decoded = await decodeImage(image);
    options?.signal?.throwIfAborted();
    const multi = scenario === "multiple" || scenario === "mixed";
    const geometry = (multi ? polygons.multiple : polygons.single) as Point[][];
    const candidates = wines
      .slice(1, 5)
      .map((w, i) => ({
        slug: w.slug,
        rank: i + 1,
        similarityScore: null,
        wine: summary(w),
      }));
    const detections: BottleDetection[] =
      !scenario || scenario === "empty"
        ? []
        : geometry.map((polygon, i) => {
            const unmatched =
              scenario === "unmatched" ||
              scenario === "no-similar" ||
              (scenario === "mixed" && i === 1);
            const wine = summary(wines[i === 1 ? 2 : 0]);
            return unmatched
              ? {
                  id: `bottle-${i + 1}`,
                  polygon,
                  detectionConfidence: null,
                  status: "unmatched",
                  match: null,
                  similar: scenario === "no-similar" ? [] : candidates,
                }
              : {
                  id: `bottle-${i + 1}`,
                  polygon,
                  detectionConfidence: null,
                  status: "matched",
                  match: { slug: wine.slug, wine, matchConfidence: null },
                  similar: [],
                };
          });
    return {
      schemaVersion: "1.0",
      requestId: requestId(),
      image: {
        width: decoded.naturalWidth,
        height: decoded.naturalHeight,
        coordinateSpace: "normalized",
        orientationApplied: true,
      },
      bestMatch: detections.find((d) => d.status === "matched")?.match
        ? { slug: detections.find((d) => d.status === "matched")!.match!.slug }
        : null,
      metrics: {
        f1Top1: null,
        f1Top5: null,
        scope: "evaluation-dataset",
        datasetId: null,
        isMock: true,
      },
      detections,
      processingTimeMs: null,
    };
  }
  async getWine(slug: string, options?: { signal?: AbortSignal }) {
    options?.signal?.throwIfAborted();
    const wine = wines.find((w) => w.slug === slug);
    if (!wine)
      throw new UserError(
        "Карточка вина сейчас недоступна. Попробуйте найти другое вино.",
      );
    return details(wine);
  }
  async searchWines(query: string, options?: { signal?: AbortSignal }) {
    options?.signal?.throwIfAborted();
    const q = query.trim().toLocaleLowerCase("ru");
    return wines
      .filter((w) =>
        `${w.name} ${w.producer} ${w.grapeVarieties.join(" ")} ${w.region}`
          .toLocaleLowerCase("ru")
          .includes(q),
      )
      .map(summary);
  }
}
