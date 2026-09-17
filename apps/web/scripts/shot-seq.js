// Capture the first-open sequence: screenshots at fixed times after
// domcontentloaded, without waiting for network idle, so the model-loading
// window is visible.
const puppeteer = require("puppeteer-core");

(async () => {
  const times = [500, 1500, 3000, 5000, 7000, 10000];
  const browser = await puppeteer.launch({
    executablePath: "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    headless: true,
    args: ["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist", "--no-sandbox"],
    defaultViewport: { width: 1600, height: 1000 },
  });
  const page = await browser.newPage();
  await page.goto("http://localhost:3000", { waitUntil: "domcontentloaded", timeout: 30000 });
  const start = Date.now();
  for (const t of times) {
    const wait = t - (Date.now() - start);
    if (wait > 0) await new Promise((r) => setTimeout(r, wait));
    await page.screenshot({ path: `C:/Users/Furkan/AppData/Local/Temp/opencode/seq_${t}ms.png` });
  }
  await browser.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
