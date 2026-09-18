import { defineConfig, devices } from "@playwright/test";
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  timeout: 30000,
  expect: { timeout: 7000 },
  reporter: [["list"], ["html", { open: "never" }]],
  use: {
    baseURL: "http://localhost:5173",
    ...devices["iPhone 13"],
    defaultBrowserType: "chromium",
    deviceScaleFactor: 1,
    viewport: { width: 390, height: 844 },
    launchOptions: { channel: process.env.PW_CHANNEL || "msedge" },
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  webServer: {
    command: "npm run dev",
    url: "http://localhost:5173",
    reuseExistingServer: !process.env.CI,
  },
});
