import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { afterEach, beforeAll, beforeEach, expect, it, vi } from "vitest";
import { ScanProvider } from "../../app/ScanProvider";
import { summary, wines } from "../../mocks/catalog";
import type { BottleDetection } from "../../shared/api/contracts";
import { saveSession } from "../../shared/storage/sessions";
import { response, session } from "../../test/fixture";
import { ScanResultPage } from "./ScanResultPage";

const WHATIS_TITLE = "Что можно узнать по этикетке";

beforeAll(() => {
  const dialog = HTMLDialogElement.prototype as unknown as Record<
    string,
    unknown
  >;
  dialog.showModal ??= function (this: HTMLDialogElement) {
    this.setAttribute("open", "");
  };
  dialog.close ??= function (this: HTMLDialogElement) {
    this.removeAttribute("open");
  };
  vi.stubGlobal("createImageBitmap", async () => ({
    width: 100,
    height: 200,
    close() {},
  }));
  HTMLCanvasElement.prototype.getContext = (() => ({
    drawImage() {},
  })) as never;
  HTMLCanvasElement.prototype.toBlob = function (callback: BlobCallback) {
    callback(new Blob(["x"], { type: "image/jpeg" }));
  };
  URL.createObjectURL = () => "blob:photo";
  URL.revokeObjectURL = () => {};
  window.scrollTo = () => {};
});

afterEach(cleanup);

let fetch: ReturnType<typeof vi.fn>;
beforeEach(() => {
  fetch = vi.fn(async (url: string) => {
    await new Promise((resolve) => setTimeout(resolve, 50));
    return url.endsWith("/whatis")
      ? new Response(
          JSON.stringify({ category: "Красное", brand: "Не удалось определить" }),
        )
      : new Response("{}", { status: 404 });
  });
  vi.stubGlobal("fetch", fetch);
});

async function openUnmatched(detection: Partial<BottleDetection>) {
  const base = response.detections[0];
  await saveSession({
    ...session,
    source: "upload",
    recognitionResult: {
      ...response,
      metrics: { ...response.metrics, isMock: false },
      detections: [{ ...base, ...detection } as BottleDetection],
    },
    selectedDetectionId: base.id,
    sheetView: "similar",
  });
  render(
    <MemoryRouter initialEntries={[`/scan/${session.scanId}`]}>
      <ScanProvider>
        <Routes>
          <Route path="/scan/:scanId" element={<ScanResultPage />} />
        </Routes>
      </ScanProvider>
    </MemoryRouter>,
  );
  await waitFor(() => expect(screen.getByText("Красное")).toBeInTheDocument());
}

it("second-level rejection shows search candidates and the whatis block exactly once", async () => {
  const similar = wines.slice(0, 3).map((wine, i) => ({
    slug: wine.slug,
    rank: i + 1,
    similarityScore: 0.8,
    wine: summary(wine),
  }));
  await openUnmatched({ similar });
  // Duplicate sibling keys used to leak a new block on every re-render while whatis was pending.
  await new Promise((resolve) => setTimeout(resolve, 300));
  expect(screen.getAllByText(WHATIS_TITLE)).toHaveLength(1);
  expect(
    fetch.mock.calls.filter(([url]) => String(url).endsWith("/whatis")),
  ).toHaveLength(1);
  expect(screen.getByRole("heading", { name: "Похожие вина" })).toBeVisible();
  expect(screen.getByText("3 варианта")).toBeVisible();
});

it("search rejection has no similar wines", async () => {
  await openUnmatched({ similar: [] });
  expect(screen.getAllByText(WHATIS_TITLE)).toHaveLength(1);
  expect(screen.queryByRole("heading", { name: "Похожие вина" })).toBeNull();
  expect(screen.getByText(/Похожих вин пока нет/)).toBeVisible();
});
