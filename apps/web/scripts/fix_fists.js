// Pull the fist-family finger curls back so fingertips rest visibly on the
// palm instead of wrapping into a smooth lump.
const fs = require("fs");
const path = require("path");

const file = path.join(__dirname, "..", "src", "components", "home", "egitim.tsx");
let s = fs.readFileSync(file, "utf8");

const fixes = [
  [/(\{ ch: "A", pose: \{ thumb: 0\.05, index: )1(?:\.\d+)?(, middle: )1(?:\.\d+)?(, ring: )1(?:\.\d+)?(, pinky: )1(?:\.\d+)?/,
    '$10.82$20.82$30.82$40.82'],
  [/(\{ ch: "E", pose: \{ thumb: [\d.]+, index: )[\d.]+(, middle: )[\d.]+(, ring: )[\d.]+(, pinky: )[\d.]+/,
    '$10.62$20.72$30.72$40.72'],
  [/(\{ ch: "M", pose: \{ thumb: [\d.]+, index: )[\d.]+/, '$10.85'],
  [/(\{ ch: "N", pose: \{ thumb: [\d.]+, index: )[\d.]+/, '$10.82'],
  [/(\{ ch: "S", pose: \{ thumb: 1, index: )1(?:\.\d+)?(, middle: )1(?:\.\d+)?(, ring: )1(?:\.\d+)?(, pinky: )1(?:\.\d+)?/,
    '$10.88$20.88$30.88$40.88'],
  [/(\{ ch: "Ş", pose: \{ thumb: [\d.]+, index: )1(?:\.\d+)?(, middle: )1(?:\.\d+)?(, ring: )1(?:\.\d+)?(, pinky: )1(?:\.\d+)?/,
    '$10.88$20.88$30.88$40.88'],
];

for (const [re, rep] of fixes) {
  if (!re.test(s)) {
    console.error("bulunamadı:", re);
    continue;
  }
  s = s.replace(re, rep);
}
fs.writeFileSync(file, s);
console.log("yumruk ailesi ayarlandı");
