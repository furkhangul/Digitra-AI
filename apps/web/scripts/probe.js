// Local dev tool: scroll to a position and report computed styles of selected
// elements, for debugging scroll-linked animation without guessing.
const puppeteer = require("puppeteer-core");

(async () => {
  const [, , url, scrollY] = process.argv;

  const browser = await puppeteer.launch({
    executablePath: "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
    headless: true,
    args: ["--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist", "--no-sandbox"],
    defaultViewport: { width: 1600, height: 1000 },
  });

  const page = await browser.newPage();
  await page.goto(url, { waitUntil: "load", timeout: 45000 });
  await new Promise((r) => setTimeout(r, 2500));

  await page.evaluate((y) => window.scrollTo({ top: y, behavior: "instant" }), Number(scrollY));
  await new Promise((r) => setTimeout(r, 1800));

  const info = await page.evaluate(() => {
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const section = document.querySelector('section[aria-label="Digitra tanıtım"]');
    const rect = section ? section.getBoundingClientRect() : null;

    const chrome = section && section.querySelectorAll("div")[0];
    const results = [];
    if (section) {
      section.querySelectorAll("div, p").forEach((el, i) => {
        const cs = getComputedStyle(el);
        if (i < 12) {
          results.push({
            i,
            cls: (el.className || "").toString().slice(0, 50),
            opacity: cs.opacity,
            transform: cs.transform.slice(0, 40),
          });
        }
      });
    }
    const dbgEl = document.querySelector('[data-dbg]');
    return {
      dbg: dbgEl ? dbgEl.getAttribute('data-dbg') : null,
      dbgMorph: window.__dbgMorph || null,
      dbgCurl: window.__dbgCurl || null,
      reduce,
      scrollY: window.scrollY,
      sectionTop: rect ? rect.top : null,
      sectionHeight: rect ? rect.height : null,
      viewportH: window.innerHeight,
      results,
    };
  });

  console.log(JSON.stringify(info, null, 2));
  await browser.close();
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
