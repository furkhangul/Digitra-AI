// Dump 21 bone world positions per letter from the rigged hand, for
// verification against the trained DIGITRA landmark classifier.
//   node scripts/dump-poses.js
const fs = require("fs");
const path = require("path");
const puppeteer = require("puppeteer-core");

const LETTERS = "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ".split("");
// Turkish letters the ASL model doesn't label are verified against their base
// shape (Ç→C, Ğ→G, İ→I, Ö→O, Ş→S, Ü→U). J/Z are temporal — excluded.
const EXPECTED = {
  Ç: "C", Ğ: "G", İ: "I", Ö: "O", Ş: "S", Ü: "U",
};

(async () => {
  const outDir = path.join(__dirname, "pose_dumps");
  fs.mkdirSync(outDir, { recursive: true });

  const browser = await puppeteer.launch({
    executablePath: "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    headless: true,
    args: ["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist", "--no-sandbox"],
    defaultViewport: { width: 800, height: 600 },
  });
  const page = await browser.newPage();

  for (const ch of LETTERS) {
    if (ch === "J" || ch === "Z") continue;
    await page.goto(`http://localhost:3000/debug-poses?letter=${encodeURIComponent(ch)}`, {
      waitUntil: "load",
      timeout: 60000,
    });
    await page.waitForFunction("window.__LANDMARKS__ != null", { timeout: 30000 });
    await new Promise((r) => setTimeout(r, 400));
    const points = await page.evaluate(() => window.__LANDMARKS__);
    fs.writeFileSync(
      path.join(outDir, `${ch}.json`),
      JSON.stringify({ target: ch, expected: EXPECTED[ch] ?? ch, points }, null, 1)
    );
    console.log("dumped", ch);
  }

  await browser.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
