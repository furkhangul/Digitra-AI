// Dumps every frame of a time window for one letter, so a fast movement can be
// inspected at full rate instead of through the coarse preview sampling.
// Usage: node scripts/dump-tid-window.cjs Ç 2.2 2.9
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
require.extensions['.ts'] = (module, filename) => module._compile(
  ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020, esModuleInterop: true },
  }).outputText, filename);

const { TID_ALPHABET } = require('../src/components/tid/poses.ts');
const { HandPlayer, serialiseFrame } = require('../src/components/tid/motion.ts');

const [ch, fromS, toS] = [process.argv[2], Number(process.argv[3]), Number(process.argv[4])];
const letter = TID_ALPHABET.find(l => l.ch === ch);
if (!letter) throw new Error('unknown letter ' + ch);
const player = new HandPlayer(letter);
player.select(letter);
const frames = [];
for (let i = 0; i < 60 * 8; i++) {
  const t = i / 60;
  player.tick(1 / 60, true, 1, false);
  if (t >= fromS && t <= toS) frames.push({ t: +t.toFixed(3), phase: player.phase, frame: serialiseFrame(player.current) });
}
const target = path.join(__dirname, '..', '..', '..', 'tmp', 'tid', 'motion-window.json');
fs.writeFileSync(target, JSON.stringify({ letters: [{ ch, from: ch, motion: letter.motion ?? null, frames }] }));
console.log(ch, 'frames', frames.length, 'over', fromS, '-', toS, 's');
