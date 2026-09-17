// Local dev tool: headless screenshot of the running dev server, used to
// visually verify the 3D hero without a browser extension.
//   node scripts/shot.js <outPath> [waitMs] [clip: x,y,w,h]
const puppeteer = require("puppeteer-core");

(async () => {
  const [, , outPath = "hero.png", waitArg, clipArg] = process.argv;

  const browser = await puppeteer.launch({
    executablePath: "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    headless: true,
    args: [
      "--use-gl=swiftshader",
      "--enable-webgl",
      "--ignore-gpu-blocklist",
      "--no-sandbox",
      `--window-size=${Number(process.env.SHOT_W) || 1600},${Number(process.env.SHOT_H) || 1000}`,
    ],
    defaultViewport: { width: Number(process.env.SHOT_W) || 1600, height: Number(process.env.SHOT_H) || 1000 },
  });

  const page = await browser.newPage();
  await page.goto(process.env.SHOT_URL || "http://localhost:3000", { waitUntil: process.env.SHOT_WAIT_UNTIL || "networkidle0", timeout: 30000 });
  await new Promise((r) => setTimeout(r, Number(waitArg) || 2500));

  const opts = { path: outPath };
  if (clipArg) {
    const [x, y, width, height] = clipArg.split(",").map(Number);
    opts.clip = { x, y, width, height };
  }
  await page.screenshot(opts);

  await browser.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
