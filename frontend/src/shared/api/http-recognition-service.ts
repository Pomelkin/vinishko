import { z } from "zod";
import {
  RecognitionSchema,
  WineDetailsSchema,
  WineSummarySchema,
  type RecognitionService,
} from "./contracts";
import { request, parse } from "./client";
export class HttpRecognitionService implements RecognitionService {
  constructor(private baseUrl: string) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }
  private request(path: string, init: RequestInit = {}) {
    return request(path, init, this.baseUrl);
  }
  async recognize(image: Blob, options?: { signal?: AbortSignal }) {
    const body = new FormData();
    body.append("image", image, "photo.jpg");
    return parse(
      RecognitionSchema,
      await this.request("/recognize", {
        method: "POST",
        body,
        signal: options?.signal,
      }),
    );
  }
  async getWine(slug: string, options?: { signal?: AbortSignal }) {
    return parse(
      WineDetailsSchema,
      await this.request(`/wines/${encodeURIComponent(slug)}`, {
        signal: options?.signal,
      }),
    );
  }
  async searchWines(query: string, options?: { signal?: AbortSignal }) {
    return parse(
      z.array(WineSummarySchema),
      await this.request(`/wines?q=${encodeURIComponent(query)}`, {
        signal: options?.signal,
      }),
    );
  }
}
