import { afterEach, it, expect, vi } from "vitest";
import { HttpRecognitionService } from "./http-recognition-service";
import { response } from "../../test/fixture";
afterEach(() => vi.unstubAllGlobals());
it("sends multipart image and passes through abort signal", async () => {
  const fetch = vi
    .fn()
    .mockResolvedValue({ ok: true, json: async () => response });
  vi.stubGlobal("fetch", fetch);
  const signal = new AbortController().signal;
  const result = await new HttpRecognitionService("/api").recognize(
    new Blob(["x"]),
    { signal },
  );
  const [url, init] = fetch.mock.calls[0];
  expect(url).toBe("/api/recognize");
  expect(init.body).toBeInstanceOf(FormData);
  expect(init.body.get("image")).toBeInstanceOf(Blob);
  expect(init.signal).toBe(signal);
  expect(result).toEqual(response);
});
it("turns malformed responses and unavailable wine into understandable errors", async () => {
  vi.stubGlobal(
    "fetch",
    vi
      .fn()
      .mockResolvedValueOnce({ ok: true, json: async () => ({ bad: true }) })
      .mockResolvedValueOnce({ ok: false, status: 404 }),
  );
  const service = new HttpRecognitionService("/api");
  await expect(service.recognize(new Blob(["x"]))).rejects.toThrow(
    "некорректный ответ",
  );
  await expect(service.getWine("missing")).rejects.toThrow("недоступна");
});
it("surfaces offline state without sending a network request", async () => {
  vi.stubGlobal("navigator", { onLine: false });
  const fetch = vi.fn();
  vi.stubGlobal("fetch", fetch);
  await expect(
    new HttpRecognitionService("/api").getWine("wine"),
  ).rejects.toThrow("Нет подключения");
  expect(fetch).not.toHaveBeenCalled();
});
