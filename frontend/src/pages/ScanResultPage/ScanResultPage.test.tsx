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
let suggestions: { title: string | null; wines: unknown[] };
beforeEach(() => {
  suggestions = {
    title: "Белые вина в каталоге",
    wines: wines.slice(0, 3).map(summary),
  };
  fetch = vi.fn(async (url: string) => {
    await new Promise((resolve) => setTimeout(resolve, 50));
    if (url.endsWith("/whatis"))
      return new Response(
        JSON.stringify({ category: "Белое", brand: "Не удалось определить" }),
      );
    if (url.includes("/suggestions?"))
      return new Response(JSON.stringify(suggestions));
    return new Response("{}", { status: 404 });
  });
  vi.stubGlobal("fetch", fetch);
});
const calls = (path: string) =>
  fetch.mock.calls.filter(([url]) => String(url).includes(path)).length;

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
  await waitFor(() => expect(screen.getByText("Белое")).toBeInTheDocument());
  // Раньше одинаковые ключи соседей плодили новый блок whatis на каждой перерисовке, пока шёл запрос.
  await new Promise((resolve) => setTimeout(resolve, 300));
  expect(screen.getAllByText(WHATIS_TITLE)).toHaveLength(1);
  expect(calls("/whatis")).toBe(1);
}

const searchSimilar = wines.slice(3, 5).map((wine, i) => ({
  slug: wine.slug,
  rank: i + 1,
  similarityScore: 0.8,
  wine: summary(wine),
}));

it("second-level rejection keeps search candidates and asks no catalog wines", async () => {
  await openUnmatched({ similar: searchSimilar });
  expect(calls("/suggestions?")).toBe(0);
  expect(screen.getByRole("heading", { name: "Похожие вина" })).toBeVisible();
  expect(screen.getByText("2 варианта")).toBeVisible();
  expect(screen.getByText(/по результатам поиска/)).toBeVisible();
});

it("search rejection shows catalog wines by recognized attributes", async () => {
  await openUnmatched({ similar: [] });
  await waitFor(() =>
    expect(
      screen.getByRole("heading", { name: "Белые вина в каталоге" }),
    ).toBeVisible(),
  );
  expect(
    String(
      fetch.mock.calls.find(([url]) =>
        String(url).includes("/suggestions?"),
      )![0],
    ),
  ).toContain("category=%D0%91%D0%B5%D0%BB%D0%BE%D0%B5");
});

it("search rejection with nothing recognized has no similar wines", async () => {
  suggestions = { title: null, wines: [] };
  await openUnmatched({ similar: [] });
  await waitFor(() => expect(calls("/suggestions?")).toBe(1));
  expect(screen.queryByRole("heading", { name: "Похожие вина" })).toBeNull();
  expect(screen.getByText(/Похожих вин пока нет/)).toBeVisible();
});
