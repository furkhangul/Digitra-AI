// Dumps a time series of frames for the given letters, exactly as the player
// drives them, so the motion can be rendered and inspected frame by frame.
// Usage: node scripts/dump-tid-motion.cjs B G Ğ O Ö
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
require.extensions['.ts'] = (module, filename) => module._compile(
  ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020, esModuleInterop: true },
  }).outputText, filename);

const { TID_ALPHABET } = require('../src/components/tid/poses.ts');
const { HandPlayer, serialiseFrame } = require('../src/components/tid/motion.ts');

const wanted = process.argv.slice(2);
const letters = TID_ALPHABET.filter(l => wanted.includes(l.ch));
const out = { letters: [] };
for (const letter of letters) {
  // start from the previous letter so the approach is part of the recording
  const from = TID_ALPHABET[(TID_ALPHABET.indexOf(letter) + 28) % 29];
  const player = new HandPlayer(from);
  player.select(letter);
  const frames = [];
  for (let i = 0; i < 300; i++) {
    player.tick(1 / 60, true, 1, false);
    if (i % 5 === 0) frames.push({ t: +(i / 60).toFixed(3), phase: player.phase, settled: player.settled, frame: serialiseFrame(player.current) });
  }
  out.letters.push({ ch: letter.ch, from: from.ch, motion: letter.motion ?? null, frames });
  console.log(letter.ch, 'from', from.ch, 'frames', frames.length);
}
const target = path.join(__dirname, '..', '..', '..', 'tmp', 'tid', 'motion-frames.json');
fs.writeFileSync(target, JSON.stringify(out));
console.log('->', target);
