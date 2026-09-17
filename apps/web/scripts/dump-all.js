// Dump all letter poses in one page load → pose_dumps/all.json
const fs = require("fs");
const path = require("path");
const puppeteer = require("puppeteer-core");

(async () => {
  const browser = await puppeteer.launch({
    executablePath: "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    headless: true,
    args: ["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist", "--no-sandbox"],
    defaultViewport: { width: 800, height: 600 },
  });
  const page = await browser.newPage();
  await page.goto("http://localhost:3000/debug-dump-all", { waitUntil: "load", timeout: 90000 });
  await page.waitForFunction("window.__ALL_DONE__ === true", { timeout: 60000 });
  const all = await page.evaluate(() => window.__ALL_LANDMARKS__);
  const out = path.join(__dirname, "pose_dumps", "all.json");
  fs.writeFileSync(out, JSON.stringify(all));
  console.log("dumped", Object.keys(all).length, "letters →", out);
  await browser.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
