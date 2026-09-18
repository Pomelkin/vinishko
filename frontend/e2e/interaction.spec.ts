import { test, expect, type Page } from "@playwright/test";
async function openDemo(page: Page, scenario = "mixed") {
  await page.goto("/?demo=1");
  await page
    .getByRole("combobox", { name: "Сценарий", exact: true })
    .selectOption(scenario);
  await page
    .getByRole("button", { name: "Открыть демофото", exact: true })
    .click();
  await expect(page).toHaveURL(/\/scan\//);
}
async function touch(
  page: Page,
  selector: string,
  startY: number,
  endY: number,
) {
  await page.locator(selector).evaluate(
    (el, { startY, endY }) => {
      const touch = (y: number) =>
        new Touch({ identifier: 1, target: el, clientX: 180, clientY: y });
      el.dispatchEvent(
        new TouchEvent("touchstart", {
          bubbles: true,
          touches: [touch(startY)],
        }),
      );
      el.dispatchEvent(
        new TouchEvent("touchmove", {
          bubbles: true,
          cancelable: true,
          touches: [touch(endY)],
        }),
      );
      el.dispatchEvent(
        new TouchEvent("touchend", {
          bubbles: true,
          changedTouches: [touch(endY)],
        }),
      );
    },
    { startY, endY },
  );
}
test("sheet swipe respects content scroll, closes from handle and restores focus", async ({
  page,
}) => {
  await openDemo(page);
  const bottle = page.getByRole("button", {
    name: "Бутылка 2: не найдена в каталоге",
    exact: true,
  });
  await bottle.click();
  await page
    .locator("[data-sheet-scroll]")
    .evaluate((el) => (el.scrollTop = 100));
  await touch(page, "[data-sheet-scroll]", 300, 440);
  await expect(page.getByRole("dialog")).toBeVisible();
  await touch(page, "[data-handle]", 200, 320);
  await expect(page.getByRole("dialog")).toHaveCount(0);
  await expect(bottle).toBeFocused();
  await bottle.click();
  await page
    .locator("[data-sheet-scroll]")
    .evaluate((el) => (el.scrollTop = 0));
  await touch(page, "[data-sheet-scroll]", 300, 440);
  await expect(page.getByRole("dialog")).toHaveCount(0);
});
test("polygon hit testing stays aligned for portrait and landscape viewports", async ({
  page,
}) => {
  await openDemo(page);
  for (const size of [
    { width: 390, height: 844 },
    { width: 844, height: 390 },
    { width: 1280, height: 844 },
  ]) {
    await page.setViewportSize(size);
    const photo = page.getByTestId("bottle-photo");
    await photo.scrollIntoViewIfNeeded();
    const point = await photo.locator("svg").evaluate((svg) => {
      const matrix = (svg as SVGSVGElement).getScreenCTM()!;
      const p = new DOMPoint(270, 450).matrixTransform(matrix);
      return { x: p.x, y: p.y };
    });
    await page.mouse.click(point.x, point.y);
    await expect(page.getByRole("dialog")).toContainText("Вино найдено");
    await page.keyboard.press("Escape");
  }
});
test("PNG, WebP, repeat selection and cancelled file chooser retain session", async ({
  page,
}) => {
  await page.goto("/");
  const png = await page.evaluate(() => {
    const c = document.createElement("canvas");
    c.width = 60;
    c.height = 100;
    c.getContext("2d")!.fillRect(0, 0, 60, 100);
    return c.toDataURL("image/png").split(",")[1];
  });
  const input = page.getByLabel("Выбрать фото", { exact: true });
  const file = {
    name: "photo.png",
    mimeType: "image/png",
    buffer: Buffer.from(png, "base64"),
  };
  await input.setInputFiles(file);
  await expect(page).toHaveURL(/\/scan\//);
  const first = page.url();
  const chooser = page.waitForEvent("filechooser");
  await page.getByRole("button", { name: "Загрузить другое фото" }).click();
  await (await chooser).setFiles([]);
  await expect(page).toHaveURL(first);
  await input.setInputFiles(file);
  await expect(page).not.toHaveURL(first);
  await expect(page).toHaveURL(/\/scan\//);
  const second = page.url();
  await input.setInputFiles("public/assets/ai-white.webp");
  await expect(page).not.toHaveURL(second);
  await expect(
    page.getByText("Фото готово к распознаванию", { exact: true }),
  ).toBeVisible();
});
test("camera preview, shutter, retake, confirm and track disposal using synthetic stream", async ({
  page,
}) => {
  await page.addInitScript(() => {
    (window as unknown as { stopped: number }).stopped = 0;
    Object.defineProperty(navigator.mediaDevices, "getUserMedia", {
      value: async () => {
        const c = document.createElement("canvas");
        c.width = 320;
        c.height = 480;
        const draw = () => {
          const ctx = c.getContext("2d")!;
          ctx.fillStyle = "#e9dfcf";
          ctx.fillRect(0, 0, 320, 480);
          ctx.fillStyle = "#314a35";
          ctx.fillRect(120, 120, 80, 300);
        };
        draw();
        const stream = c.captureStream(10),
          timer = setInterval(draw, 100);
        stream.getTracks().forEach((track) => {
          const original = track.stop.bind(track);
          track.stop = () => {
            original();
            clearInterval(timer);
            (window as unknown as { stopped: number }).stopped++;
          };
        });
        return stream;
      },
    });
  });
  await page.goto("/");
  await page.getByRole("button", { name: "Сканировать", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Сделать снимок" }),
  ).toBeEnabled();
  await page.getByRole("button", { name: "Сделать снимок" }).click();
  await expect(
    page.getByRole("img", { name: "Сделанный снимок" }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => (window as unknown as { stopped: number }).stopped,
    ),
  ).toBe(1);
  await page.getByRole("button", { name: "Переснять" }).click();
  await expect(
    page.getByRole("button", { name: "Сделать снимок" }),
  ).toBeEnabled();
  await page.getByRole("button", { name: "Сделать снимок" }).click();
  await page.getByRole("button", { name: "Использовать фото" }).click();
  await expect(page).toHaveURL(/\/scan\//);
  expect(
    await page.evaluate(
      () => (window as unknown as { stopped: number }).stopped,
    ),
  ).toBe(2);
  await page.goto("/");
  await page.getByRole("button", { name: "Сканировать", exact: true }).click();
  await expect(
    page.getByRole("button", { name: "Сделать снимок" }),
  ).toBeEnabled();
  await page.getByRole("button", { name: "Отменить", exact: true }).click();
  expect(
    await page.evaluate(
      () => (window as unknown as { stopped: number }).stopped,
    ),
  ).toBe(1);
});
test("native capture input is used when getUserMedia is unavailable", async ({
  page,
}) => {
  await page.addInitScript(() =>
    Object.defineProperty(navigator.mediaDevices, "getUserMedia", {
      value: undefined,
    }),
  );
  await page.goto("/");
  const chooser = page.waitForEvent("filechooser");
  await page.getByRole("button", { name: "Сканировать", exact: true }).click();
  const chosen = await chooser;
  expect(await chosen.element().getAttribute("capture")).toBe("environment");
  await chosen.setFiles([]);
  await expect(page).toHaveURL("/");
});

test("EXIF orientation is normalized and oversized uploads are rejected", async ({
  page,
}) => {
  await page.addInitScript(() =>
    Object.defineProperty(crypto, "randomUUID", { value: undefined }),
  );
  await page.goto("/");
  const jpeg = await page.evaluate(() => {
    const c = document.createElement("canvas");
    c.width = 80;
    c.height = 40;
    c.getContext("2d")!.fillRect(0, 0, 80, 40);
    return c.toDataURL("image/jpeg").split(",")[1];
  });
  const original = Buffer.from(jpeg, "base64");
  // APP1 Exif, TIFF little-endian, orientation=6 (90 degrees clockwise).
  const exif = Buffer.from(
    "ffe1002245786966000049492a0008000000010012010300010000000600000000000000",
    "hex",
  );
  const rotated = Buffer.concat([
    original.subarray(0, 2),
    exif,
    original.subarray(2),
  ]);
  await page
    .getByLabel("Выбрать фото", { exact: true })
    .setInputFiles({
      name: "rotated.jpg",
      mimeType: "image/jpeg",
      buffer: rotated,
    });
  await expect(page).toHaveURL(/\/scan\//);
  await expect(page.getByTestId("bottle-photo").locator("svg")).toHaveAttribute(
    "viewBox",
    "0 0 40 80",
  );
  await page
    .getByLabel("Выбрать фото", { exact: true })
    .setInputFiles({
      name: "large.jpg",
      mimeType: "image/jpeg",
      buffer: Buffer.alloc(20 * 1024 * 1024 + 1),
    });
  await expect(page.getByRole("dialog")).toContainText("до 20 МБ");
});
test("filters, food pairings and missing image fallback are usable", async ({
  page,
}) => {
  await page.route("**/assets/ai-white.webp", (route) => route.abort());
  await page.goto("/");
  await page.getByRole("button", { name: "Фильтр", exact: true }).click();
  await page
    .getByRole("combobox", { name: "Цвет", exact: true })
    .selectOption("Белое");
  await page.getByRole("button", { name: "Показать вина" }).click();
  await expect(
    page.getByRole("button", { name: /ИИ ВИНО. Белое/ }),
  ).toHaveCount(1);
  await expect(page.getByText("Фото появится позже")).toBeVisible();
  await page.getByRole("button", { name: /ИИ ВИНО. Белое/ }).click();
  await page.getByRole("button", { name: "Сыры", exact: true }).click();
  await expect(page.getByRole("dialog")).toContainText("тарелку полутвёрдых");
});
