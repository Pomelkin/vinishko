import { describe, it, expect, vi } from "vitest";
import {
  saveSession,
  loadSession,
  clearMemoryCache,
  cleanupSessions,
} from "./sessions";
import { session } from "../../test/fixture";
describe("durable scan session", () => {
  it("restores Blob, polygon, selection, sheet and scroll after clearing memory", async () => {
    // Node Blob is structured-cloneable in fake-indexeddb, unlike jsdom Blob.
    const { Blob: NodeBlob } = await import("node:buffer");
    vi.stubGlobal("Blob", NodeBlob);
    const value = {
      ...session,
      scanId: "persisted",
      imageBlob: new Blob(["photo"], { type: "image/jpeg" }),
    };
    expect(await saveSession(value)).toBe(true);
    clearMemoryCache();
    const loaded = await loadSession("persisted");
    expect(loaded?.imageBlob.size).toBe(5);
    expect(loaded?.recognitionResult).toEqual(value.recognitionResult);
    expect(loaded?.selectedDetectionId).toBe("bottle");
    expect(loaded?.sheetView).toBe("similar");
    expect(loaded?.relevantScrollPositions.similar).toBe(180);
    vi.unstubAllGlobals();
  });
  it("returns missing for expired sessions", async () => {
    await saveSession({
      ...session,
      scanId: "expired",
      createdAt: Date.now() - 8 * 86400000,
    });
    await cleanupSessions();
    expect(await loadSession("expired")).toBeNull();
  });
  it("keeps a usable in-memory session when IndexedDB is unavailable", async () => {
    vi.stubGlobal("indexedDB", {
      open: () => {
        throw Error("denied");
      },
    });
    expect(await saveSession({ ...session, scanId: "fallback" })).toBe(false);
    expect((await loadSession("fallback"))?.scanId).toBe("fallback");
    vi.unstubAllGlobals();
  });
});
