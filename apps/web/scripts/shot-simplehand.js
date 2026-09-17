// Screenshot the simplehand debug page for a given pose/axis/sign.
const puppeteer = require("puppeteer-core");

(async () => {
  const [, , pose = "point", axis = "x", sign = "1"] = process.argv;
  const browser = await puppeteer.launch({
    executablePath: "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    headless: true,
    args: ["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist", "--no-sandbox"],
    defaultViewport: { width: 900, height: 700 },
  });
  const page = await browser.newPage();
  await page.goto(`http://localhost:3000/debug-simplehand`, { waitUntil: "networkidle0", timeout: 45000 });
  await page.evaluate((p, a, s) => {
    const btns = [...document.querySelectorAll("button")];
    btns.find((b) => b.textContent === p)?.click();
    const axisBtn = btns.find((b) => b.textContent.startsWith("axis: "));
    if (axisBtn && !axisBtn.textContent.includes(a)) axisBtn.click();
    setTimeout(() => {
      const signBtn = btns.find((b) => b.textContent.startsWith("sign: "));
      if (signBtn && !signBtn.textContent.includes(s)) signBtn.click();
    }, 50);
  }, pose, axis, sign);
  await new Promise((r) => setTimeout(r, 2500));
  await page.screenshot({ path: "C:/Users/Furkan/AppData/Local/Temp/opencode/sh_debug.png" });
  await browser.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
