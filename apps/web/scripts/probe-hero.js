// Report scroll progress inputs and word opacities at several scroll stops.
const puppeteer = require("puppeteer-core");

(async () => {
  const stops = [2400, 2700, 3000, 3300, 3500, 3800];
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
    await new Promise((r) => setTimeout(r, 800));
    const info = await page.evaluate(() => {
      const section = document.querySelector("section.relative.isolate");
      const rect = section ? section.getBoundingClientRect() : null;
      const words = section
        ? Array.from(section.querySelectorAll("h1 span span")).map((el) => {
            const cs = getComputedStyle(el);
            return {
              text: (el.textContent || "").slice(0, 22),
              opacity: cs.opacity,
              transform: cs.transform.slice(0, 30),
            };
          })
        : [];
      return {
        scrollY: window.scrollY,
        sectionTop: rect ? Math.round(rect.top) : null,
        sectionHeight: rect ? Math.round(rect.height) : null,
        viewportH: window.innerHeight,
        docHeight: document.documentElement.scrollHeight,
        words,
      };
    });
    console.log(`--- scrollY=${y} ---`);
    console.log(JSON.stringify(info, null, 1));
  }
  await browser.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
