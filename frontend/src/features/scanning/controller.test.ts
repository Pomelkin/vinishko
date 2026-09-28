import { afterEach, describe, expect, it, vi } from "vitest";
import { ScanController } from "./controller";
import { response, prepared, stub } from "../../test/fixture";
import type { RecognitionResponse } from "../../shared/api/contracts";
afterEach(() => vi.useRealTimers());
describe("scan request ownership", () => {
  it("ignores late cancelled response and preserves the newer session", async () => {
    let finish!: (r: RecognitionResponse) => void;
    let signal: AbortSignal | undefined;
    const recognize = vi
      .fn()
      .mockImplementationOnce((_image, options) => {
        signal = options.signal;
        return new Promise((r) => {
          finish = r;
        });
      })
      .mockResolvedValueOnce({ ...response, requestId: "new" });
    const save = vi.fn().mockResolvedValue(true);
    const c = new ScanController(stub(recognize), save);
    const first = c.start(async () => prepared, "demo");
    await Promise.resolve();
    c.cancel();
    expect(signal?.aborted).toBe(true);
    const second = await c.start(async () => prepared, "demo");
    finish({ ...response, requestId: "old" });
    await first;
    expect(c.store.getState().session?.scanId).toBe(second?.scanId);
    expect(c.store.getState().session?.recognitionResult.requestId).toBe("new");
    expect(save).toHaveBeenCalledTimes(1);
  });
  it("double presses and keep waiting never start a second request", async () => {
    vi.useFakeTimers();
    const recognize = vi.fn(() => new Promise<RecognitionResponse>(() => {}));
    const c = new ScanController(stub(recognize), async () => true, 100);
    void c.start(async () => prepared, "demo");
    await Promise.resolve();
    await c.start(async () => prepared, "demo");
    await vi.advanceTimersByTimeAsync(101);
    expect(c.store.getState().phase).toBe("slow");
    c.keepWaiting();
    expect(c.store.getState().phase).toBe("recognizing");
    expect(recognize).toHaveBeenCalledTimes(1);
    c.cancel();
  });
  it("cancel while preparing never sends an image", async () => {
    let finish!: (p: typeof prepared) => void;
    const recognize = vi.fn();
    const c = new ScanController(stub(recognize));
    const task = c.start(
      () =>
        new Promise((r) => {
          finish = r;
        }),
      "upload",
    );
    c.cancel();
    finish(prepared);
    await task;
    expect(recognize).not.toHaveBeenCalled();
  });
  it("rejects mismatched image geometry and invalid polygons", async () => {
    const c = new ScanController(
      stub(async () => ({
        ...response,
        image: { ...response.image, width: 123 },
      })),
      async () => true,
    );
    await c.start(async () => prepared, "demo");
    expect(c.store.getState().phase).toBe("error");
    expect(c.store.getState().session).toBeNull();
  });
  it("retries the saved image with a new session identifier", async () => {
    const recognize = vi
      .fn()
      .mockRejectedValueOnce(Error("server"))
      .mockResolvedValueOnce(response);
    const c = new ScanController(stub(recognize), async () => true);
    await c.start(async () => prepared, "demo");
    await c.retry();
    expect(recognize.mock.calls[0][0]).toBe(recognize.mock.calls[1][0]);
    expect(c.store.getState().phase).toBe("success");
  });
});
