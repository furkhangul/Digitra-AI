// Local dev tool: capture a page at several timestamps (and optional scroll
// offsets) in one browser session, for studying how a reference site's intro
// animation unfolds.
//   node scripts/capture-sequence.js <url> <outDir> <ms,ms,...> [scrollPx,...]
const fs = require("fs");
const path = require("path");
const puppeteer = require("puppeteer-core");

(async () => {
  const [, , url, outDir, timesArg, scrollArg] = process.argv;
  const times = (timesArg || "1000,2000,3000,5000").split(",").map(Number);
  const scrolls = scrollArg ? scrollArg.split(",").map(Number) : [];

  fs.mkdirSync(outDir, { recursive: true });

  const browser = await puppeteer.launch({
    executablePath: "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    headless: true,
    args: ["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist", "--no-sandbox"],
    defaultViewport: { width: 1600, height: 1000 },
  });

  const page = await browser.newPage();
  await page.goto(url, { waitUntil: "load", timeout: 45000 });

  let prev = 0;
  for (const t of times) {
    await new Promise((r) => setTimeout(r, Math.max(0, t - prev)));
    prev = t;
    const out = path.join(outDir, `t${t}.png`);
    await page.screenshot({ path: out });
    console.log("saved", out);
  }

  for (const y of scrolls) {
    await page.evaluate((py) => window.scrollTo({ top: py, behavior: "instant" }), y);
    await new Promise((r) => setTimeout(r, 1600));
    const out = path.join(outDir, `scroll${y}.png`);
    await page.screenshot({ path: out });
    console.log("saved", out);
  }

  await browser.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
