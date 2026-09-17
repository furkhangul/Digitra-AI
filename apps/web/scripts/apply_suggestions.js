// Merge classifier-suggested curl deltas into the ALPHABET poses in
// egitim.tsx. Values clamp to [0, 1.1].
//   node scripts/apply_suggestions.js
const fs = require("fs");
const path = require("path");

const suggestions = JSON.parse(
  fs.readFileSync(path.join(__dirname, "pose_dumps", "suggestions.json"), "utf-8")
);
const file = path.join(__dirname, "..", "src", "components", "home", "egitim.tsx");
let source = fs.readFileSync(file, "utf-8");

const clamp = (v) => Math.max(0, Math.min(1.1, v));
const num = (v) => Number(v.toFixed(2));
let applied = 0;

for (const [ch, deltas] of Object.entries(suggestions)) {
  const marker = `ch: "${ch}", pose: {`;
  const start = source.indexOf(marker);
  if (start === -1) {
    console.error("blok bulunamadı:", ch);
    continue;
  }
  const end = source.indexOf("}", start);
  const block = source.slice(start, end);
  let updated = block;
  for (const [finger, delta] of Object.entries(deltas)) {
    const re = new RegExp(`(${finger}: )(\\d+(?:\\.\\d+)?)`);
    if (!re.test(updated)) {
      console.error("parmak bulunamadı:", ch, finger);
      continue;
    }
    updated = updated.replace(re, (_, p, v) => p + num(clamp(parseFloat(v) + delta)));
    applied++;
  }
  source = source.slice(0, start) + updated + source.slice(end);
}

fs.writeFileSync(file, source);
console.log(`uygulanan ayar: ${applied} → ${file}`);
