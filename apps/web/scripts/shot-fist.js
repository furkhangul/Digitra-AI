// Screenshot the A fist after the curl pull-back.
const puppeteer = require("puppeteer-core");

(async () => {
  const browser = await puppeteer.launch({
    executablePath: "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    headless: true,
    args: ["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist", "--no-sandbox"],
    defaultViewport: { width: 1600, height: 1000 },
  });
  const page = await browser.newPage();
  await page.goto("http://localhost:3000", { waitUntil: "load", timeout: 180000 });
  await new Promise((r) => setTimeout(r, 8000));
  await page.evaluate(() => {
    const el = document.getElementById("egitim");
    window.scrollTo({ top: el.offsetTop - 60, behavior: "instant" });
  });
  await new Promise((r) => setTimeout(r, 2500));
  await page.evaluate(() => {
    const btns = [...document.querySelectorAll("#egitim button")];
    btns.find((x) => x.textContent.trim() === "A" && x.closest(".glass"))?.click();
  });
  await new Promise((r) => setTimeout(r, 1300));
  await page.screenshot({ path: "C:/Users/Furkan/AppData/Local/Temp/opencode/fist3_a.png" });
  await browser.close();
  console.log("done");
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
