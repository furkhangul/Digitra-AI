// Screenshot the pinned live-demo section at several scroll progress points.
const puppeteer = require("puppeteer-core");

(async () => {
  const fracs = [0.1, 0.4, 0.65, 0.95];
  const browser = await puppeteer.launch({
    executablePath: "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    headless: true,
    args: ["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist", "--no-sandbox"],
    defaultViewport: { width: 1600, height: 1000 },
  });
  const page = await browser.newPage();
  await page.goto("http://localhost:3000", { waitUntil: "networkidle0", timeout: 45000 });
  await new Promise((r) => setTimeout(r, 2000));
  for (let i = 0; i < fracs.length; i++) {
    const y = await page.evaluate((f) => {
      const el = document.getElementById("demo");
      const total = el.getBoundingClientRect().height - window.innerHeight;
      return el.offsetTop + total * f;
    }, fracs[i]);
    await page.evaluate((top) => window.scrollTo({ top, behavior: "instant" }), y);
    await new Promise((r) => setTimeout(r, 1400));
    await page.screenshot({ path: `C:/Users/Furkan/AppData/Local/Temp/opencode/demo_${i}.png` });
  }
  await browser.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
