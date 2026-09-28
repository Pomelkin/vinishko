import { z } from "zod";
import {
  RecognitionSchema,
  WineDetailsSchema,
  WineSummarySchema,
  type RecognitionService,
} from "./contracts";
import { UserError } from "./errors";
export class HttpRecognitionService implements RecognitionService {
  constructor(private baseUrl: string) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }
  private async request(
    path: string,
    init: RequestInit = {},
  ): Promise<unknown> {
    if (!navigator.onLine)
      throw new UserError(
        "Нет подключения к сети. Проверьте интернет и повторите попытку.",
      );
    const response = await fetch(`${this.baseUrl}${path}`, init);
    if (response.status === 404)
      throw new UserError(
        "Карточка вина сейчас недоступна. Попробуйте найти другое вино.",
      );
    if (!response.ok)
      throw new UserError(
        "Сервис временно недоступен. Попробуйте немного позже.",
      );
    try {
      return await response.json();
    } catch {
      throw new UserError(
        "Сервис вернул некорректный ответ. Повторите попытку позже.",
      );
    }
  }
  private parse<T>(schema: z.ZodType<T>, data: unknown): T {
    const result = schema.safeParse(data);
    if (!result.success)
      throw new UserError(
        "Сервис вернул некорректный ответ. Повторите попытку позже.",
      );
    return result.data;
  }
  async recognize(image: Blob, options?: { signal?: AbortSignal }) {
    const body = new FormData();
    body.append("image", image, "photo.jpg");
    return this.parse(
      RecognitionSchema,
      await this.request("/recognize", {
        method: "POST",
        body,
        signal: options?.signal,
      }),
    );
  }
  async getWine(slug: string, options?: { signal?: AbortSignal }) {
    return this.parse(
      WineDetailsSchema,
      await this.request(`/wines/${encodeURIComponent(slug)}`, {
        signal: options?.signal,
      }),
    );
  }
  async searchWines(query: string, options?: { signal?: AbortSignal }) {
    return this.parse(
      z.array(WineSummarySchema),
      await this.request(`/wines?q=${encodeURIComponent(query)}`, {
        signal: options?.signal,
      }),
    );
  }
}
