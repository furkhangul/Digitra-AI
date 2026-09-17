// Scroll through the pinned hero and screenshot at several progress points.
const puppeteer = require("puppeteer-core");

(async () => {
  const stops = [2400, 2700, 3000, 3300, 3500];
  const browser = await puppeteer.launch({
    executablePath: "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    headless: true,
    args: ["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist", "--no-sandbox"],
    defaultViewport: { width: 1600, height: 1000 },
  });
  const page = await browser.newPage();
  await page.goto("http://localhost:3000", { waitUntil: "networkidle0", timeout: 45000 });
  await new Promise((r) => setTimeout(r, 2000));
  for (const y of stops) {
    await page.evaluate((top) => window.scrollTo({ top, behavior: "instant" }), y);
    await new Promise((r) => setTimeout(r, 1200));
    await page.screenshot({ path: `C:/Users/Furkan/AppData/Local/Temp/opencode/hero_${y}.png` });
  }
  await browser.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
