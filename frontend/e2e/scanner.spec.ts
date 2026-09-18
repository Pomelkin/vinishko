import { test, expect, type Page } from "@playwright/test";
import path from "node:path";
async function demo(page: Page, scenario = "mixed") {
  await page.goto("/");
  await page.getByRole("button", { name: /Попробовать на примере/ }).click();
  await page
    .getByRole("combobox", { name: "Сценарий", exact: true })
    .selectOption(scenario);
  await page
    .getByRole("button", { name: "Открыть демофото", exact: true })
    .click();
}
async function ready(page: Page, scenario = "mixed") {
  await demo(page, scenario);
  await expect(page).toHaveURL(/\/scan\//);
  await expect(page.getByTestId("bottle-photo")).toBeVisible();
}
test("matched bottle: details, Back, Forward and persisted reload", async ({
  page,
}) => {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await ready(page);
  const resultUrl = page.url();
  await page.screenshot({
    path: "qa/result-390.png",
    fullPage: true,
    animations: "disabled",
  });
  await page
    .getByRole("button", { name: "Бутылка 1: Симбиоз", exact: true })
    .click();
  await expect(page.getByRole("dialog")).toContainText("Вино найдено");
  await page.screenshot({
    path: "qa/matched-sheet-390.png",
    animations: "disabled",
  });
  await page.getByRole("button", { name: "Подробнее о вине" }).click();
  await expect(page).toHaveURL(/\/wine\/symbiosis$/);
  await expect(
    page.getByRole("heading", { name: "Симбиоз", exact: true }),
  ).toBeVisible();
  await page.screenshot({
    path: "qa/wine-390.png",
    fullPage: true,
    animations: "disabled",
  });
  await page.goBack();
  await expect(page).toHaveURL(resultUrl);
  await expect(page.getByRole("dialog")).toContainText("Вино найдено");
  await expect(
    page.getByRole("button", { name: "Бутылка 1: Симбиоз", exact: true }),
  ).toHaveAttribute("aria-pressed", "true");
  await page.goForward();
  await expect(page).toHaveURL(/\/wine\/symbiosis$/);
  await page.goBack();
  await page.reload();
  await expect(page.getByRole("dialog")).toContainText("Вино найдено");
  await expect(page.locator("svg image").first()).toHaveAttribute(
    "href",
    /^blob:/,
  );
  expect(errors).toEqual([]);
});
test("unmatched bottle, recommendation, list scroll and Back restoration", async ({
  page,
}) => {
  await ready(page);
  await page
    .getByRole("button", {
      name: "Бутылка 2: не найдена в каталоге",
      exact: true,
    })
    .click();
  await expect(page.getByRole("dialog")).toContainText("Похожие вина");
  await page.screenshot({
    path: "qa/similar-sheet-390.png",
    animations: "disabled",
  });
  const scroller = page.locator("[data-sheet-scroll]");
  await scroller.evaluate((el) => (el.scrollTop = 180));
  await expect.poll(() => scroller.evaluate((el) => el.scrollTop)).toBe(180);
  await page
    .getByRole("dialog")
    .getByRole("button", { name: /ИИ ВИНО/ })
    .first()
    .click();
  await expect(page.getByRole("dialog")).toContainText("Похожее вино");
  await page.getByRole("button", { name: "Подробнее о вине" }).click();
  await expect(page).toHaveURL(/\/wine\/ai-wine-red$/);
  await page.goBack();
  await expect(page.getByRole("dialog")).toContainText("Похожее вино");
  await page.reload();
  await expect(page.getByRole("dialog")).toContainText("Похожее вино");
  await page.getByRole("button", { name: "Назад к похожим" }).click();
  await expect(page.getByRole("dialog")).toContainText("Похожие вина");
  expect(await scroller.evaluate((el) => el.scrollTop)).toBeGreaterThan(100);
  await page.keyboard.press("Escape");
  await expect(page.getByRole("dialog")).toHaveCount(0);
});
test("slow request continues, cancellation and restart", async ({ page }) => {
  await demo(page, "slow");
  await expect(page.getByRole("dialog")).toContainText("Еще немного времени");
  await page.screenshot({ path: "qa/loading-390.png", animations: "disabled" });
  await expect(page.getByRole("button", { name: "Подождать еще" })).toBeVisible(
    { timeout: 10000 },
  );
  await page.screenshot({ path: "qa/slow-390.png", animations: "disabled" });
  await page.getByRole("button", { name: "Подождать еще" }).click();
  await expect(page.getByRole("dialog")).toContainText("Еще немного времени");
  await page.getByRole("button", { name: "Отменить", exact: true }).click();
  await expect(page).toHaveURL("/");
  await ready(page, "single");
  await expect(page.getByRole("button", { name: /Бутылка 1:/ })).toBeVisible();
});
test("technical error, retry and working name search", async ({ page }) => {
  await demo(page, "error");
  await expect(page.getByRole("dialog")).toContainText("Не удалось завершить");
  await page.getByRole("button", { name: "Повторить", exact: true }).click();
  await expect(page.getByRole("dialog")).toContainText("Не удалось завершить");
  await page
    .getByRole("button", { name: "Найти по названию", exact: true })
    .click();
  await page.getByRole("searchbox").fill("Симбиоз");
  await expect(
    page.getByRole("button", { name: /Симбиоз WINEPARK/ }),
  ).toHaveCount(1);
});
test("gallery upload preserves actual photo and rejects corrupt and HEIC input", async ({
  page,
}) => {
  await page.goto("/");
  const input = page.getByLabel("Выбрать фото", { exact: true });
  await input.setInputFiles(path.resolve("e2e/fixtures/upload.jpg"));
  await expect(page).toHaveURL(/\/scan\//);
  await expect(
    page.getByText("Фото готово к распознаванию", { exact: true }),
  ).toBeVisible();
  await expect(page.locator("svg polygon")).toHaveCount(0);
  await page.reload();
  await expect(
    page.getByText("Фото готово к распознаванию", { exact: true }),
  ).toBeVisible();
  await input.setInputFiles({
    name: "bad.jpg",
    mimeType: "image/jpeg",
    buffer: Buffer.from("invalid"),
  });
  await expect(page.getByRole("dialog")).toContainText(
    "Возможно, файл повреждён",
  );
  await page.getByRole("button", { name: "Отменить", exact: true }).click();
  await input.setInputFiles({
    name: "photo.heic",
    mimeType: "image/heic",
    buffer: Buffer.from("heic"),
  });
  await expect(page.getByRole("dialog")).toContainText("HEIC/HEIF");
});
test("empty recognition, no recommendations, missing session and direct wine URL", async ({
  page,
}) => {
  await ready(page, "empty");
  await expect(
    page.getByText("Бутылки не обнаружены", { exact: true }),
  ).toBeVisible();
  await ready(page, "no-similar");
  await page.getByRole("button", { name: /Бутылка 1:/ }).click();
  await expect(page.getByRole("dialog")).toContainText("Похожих вин пока нет");
  await page.goto("/scan/missing");
  await expect(page.getByText("Фотография не сохранилась")).toBeVisible();
  await page.goto("/wine/symbiosis");
  await page.getByRole("button", { name: "Свои вина", exact: true }).click();
  await expect(page).toHaveURL("/");
  await page.goto("/wine/unknown");
  await expect(page.getByText("Вино пока недоступно")).toBeVisible();
});
test("camera denial offers gallery and native fallback", async ({ page }) => {
  await page.addInitScript(() => {
    Object.defineProperty(navigator.mediaDevices, "getUserMedia", {
      value: () =>
        Promise.reject(new DOMException("Denied", "NotAllowedError")),
    });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Сканировать", exact: true }).click();
  await expect(
    page.getByRole("dialog", { name: "Камера", exact: true }),
  ).toContainText("Доступ к камере не разрешён");
  await expect(
    page.getByRole("button", { name: "Открыть камеру телефона" }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Отменить", exact: true }).click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
});
test("responsive scanner and result have no horizontal overflow", async ({
  page,
}) => {
  for (const width of [360, 375, 390, 430, 768, 1280]) {
    await page.setViewportSize({ width, height: 844 });
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "Свои вина", exact: true }),
    ).toBeVisible();
    await page.evaluate(() => document.fonts.ready);
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
    await page.screenshot({
      path: `qa/scanner-${width}.png`,
      fullPage: true,
      animations: "disabled",
    });
    for (const box of await page
      .locator('[class*="imageArea"]')
      .evaluateAll((els) =>
        els.map((el) => {
          const img = el.querySelector("img")!;
          return {
            parent: el.getBoundingClientRect().height,
            img: img?.getBoundingClientRect().height ?? 0,
          };
        }),
      ))
      expect(box.img).toBeLessThanOrEqual(box.parent + 1);
  }
  await ready(page, "multiple");
  for (const width of [360, 390, 768, 1280]) {
    await page.setViewportSize({ width, height: 844 });
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth <= innerWidth,
      ),
    ).toBe(true);
  }
});
