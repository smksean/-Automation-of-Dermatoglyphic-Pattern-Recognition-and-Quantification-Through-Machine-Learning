import { chromium } from "playwright-core";
import path from "node:path";

const edgePath = "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";
const outputDir = path.resolve("assets/source");
const baseUrl = (process.env.BASE_URL ?? "http://127.0.0.1:3000").replace(/\/$/, "");
const routes = [
  ["home", "/"],
  ["analysis", "/analysis"],
  ["methods", "/methods"],
  ["results", "/results"],
  ["quantification", "/quantification"],
  ["models", "/models"],
  ["study", "/study"],
];

const browser = await chromium.launch({
  executablePath: edgePath,
  headless: true,
  args: ["--disable-extensions"],
});

async function inspectViewport(name, width, height) {
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
  const errors = [];
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  page.on("response", (response) => {
    if (response.status() >= 400) errors.push(`${response.status()} ${response.url()}`);
  });

  const pages = [];
  for (const [routeName, routePath] of routes) {
    await page.goto(`${baseUrl}${routePath}`, { waitUntil: "networkidle" });
    const layout = await page.evaluate(() => ({
      pathname: window.location.pathname,
      viewportWidth: window.innerWidth,
      scrollWidth: document.documentElement.scrollWidth,
      heading: document.querySelector("h1, h2")?.textContent ?? "",
    }));

    if (routePath === "/methods" && width >= 760) {
      await page.locator(".method-figure .figure-open").click();
      await page.locator(".figure-dialog").waitFor({ state: "visible" });
      await page.locator(".figure-close").click();
      await page.locator(".figure-dialog").waitFor({ state: "detached" });
    }

    if (routePath === "/models") {
      await page.getByRole("tab", { name: /Arch subtype/ }).click();
      await page.getByRole("heading", { name: /ResNet-18 embeddings/ }).waitFor({ state: "visible" });
      await page.getByRole("tab", { name: /Broad classifier/ }).click();
    }

    await page.screenshot({ path: path.join(outputDir, `web-app-${name}-${routeName}.png`), fullPage: false });
    pages.push(layout);
  }

  await page.close();
  return { name, pages, consoleErrors: errors };
}

const results = [];
results.push(await inspectViewport("desktop", 1440, 1000));
results.push(await inspectViewport("mobile", 390, 844));
console.log(JSON.stringify(results, null, 2));
await browser.close();
