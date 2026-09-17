// Verify the Ceviri (both directions) and Egitim (both modes) sections.
const puppeteer = require("puppeteer-core");

(async () => {
  const browser = await puppeteer.launch({
    executablePath: "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    headless: true,
    args: ["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist", "--no-sandbox"],
    defaultViewport: { width: 1600, height: 1000 },
  });
  const page = await browser.newPage();
  await page.goto("http://localhost:3000", { waitUntil: "networkidle0", timeout: 45000 });
  await new Promise((r) => setTimeout(r, 2000));

  const scrollTo = (f) =>
    page.evaluate((f) => {
      const el = document.getElementById("ceviri");
      const total = el.getBoundingClientRect().height - window.innerHeight;
      window.scrollTo({ top: el.offsetTop + total * f, behavior: "instant" });
    }, f);

  // Ceviri, Parmak → Ses, mid progress
  await scrollTo(0.65);
  await new Promise((r) => setTimeout(r, 1400));
  await page.screenshot({ path: "C:/Users/Furkan/AppData/Local/Temp/opencode/ceviri_p2s.png" });

  // Toggle Ses → Parmak
  await page.evaluate(() => {
    const btns = [...document.querySelectorAll("#ceviri button")];
    btns.find((b) => b.textContent.includes("Ses → Parmak"))?.click();
  });
  await new Promise((r) => setTimeout(r, 1500));
  await page.screenshot({ path: "C:/Users/Furkan/AppData/Local/Temp/opencode/ceviri_s2p.png" });

  // Egitim — learn mode, pick letter V
  await page.evaluate(() => {
    const el = document.getElementById("egitim");
    window.scrollTo({ top: el.offsetTop - 60, behavior: "instant" });
  });
  await new Promise((r) => setTimeout(r, 1200));
  await page.evaluate(() => {
    const btns = [...document.querySelectorAll("#egitim button")];
    btns.find((b) => b.textContent.trim() === "V" && b.closest(".glass"))?.click();
  });
  await new Promise((r) => setTimeout(r, 600));
  await page.screenshot({ path: "C:/Users/Furkan/AppData/Local/Temp/opencode/egitim_ogren.png" });

  // Egitim — quiz mode, answer one question
  await page.evaluate(() => {
    const btns = [...document.querySelectorAll("#egitim button")];
    btns.find((b) => b.textContent.trim() === "Sınav")?.click();
  });
  await new Promise((r) => setTimeout(r, 800));
  await page.evaluate(() => {
    const btns = [...document.querySelectorAll("#egitim button")].filter(
      (b) => b.closest(".grid-cols-4") && b.getAttribute("disabled") == null
    );
    btns[1]?.click();
  });
  await new Promise((r) => setTimeout(r, 600));
  await page.screenshot({ path: "C:/Users/Furkan/AppData/Local/Temp/opencode/egitim_sinav.png" });

  await browser.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
