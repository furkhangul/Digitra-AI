// Dumps the authored pose of every TID letter (and its settled motion frame)
// so the Python surface checks and the Blender build read exactly what the site renders.
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
require.extensions['.ts'] = (module, filename) => module._compile(
  ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020, esModuleInterop: true },
  }).outputText, filename);

const { TID_ALPHABET, FINGERS, rigSpec } = require('../src/components/tid/poses.ts');
const { HandPlayer, serialiseFrame } = require('../src/components/tid/motion.ts');

const out = { rigSpec, fingers: FINGERS, letters: [] };
for (const letter of TID_ALPHABET) {
  const player = new HandPlayer(letter);
  // run to the settled hold so we check what the learner actually sees
  for (let i = 0; i < 600 && !player.settled; i++) player.tick(1 / 60, true, 1, false);
  for (let i = 0; i < 30; i++) player.tick(1 / 60, true, 1, false);
  out.letters.push({
    ch: letter.ch,
    motion: letter.motion ?? null,
    sourceVariant: letter.sourceVariant ?? null,
    contact: letter.pose.contact ?? null,
    settled: player.settled,
    frame: serialiseFrame(player.current),
    authored: {
      right: { shape: letter.pose.right.shape, rotation: letter.pose.right.rotation, position: letter.pose.right.position, visible: letter.pose.right.visible },
      left: { shape: letter.pose.left.shape, rotation: letter.pose.left.rotation, position: letter.pose.left.position, visible: letter.pose.left.visible },
    },
  });
}
const target = path.join(__dirname, '..', '..', '..', 'tmp', 'tid', 'letter-frames.json');
fs.mkdirSync(path.dirname(target), { recursive: true });
fs.writeFileSync(target, JSON.stringify(out, null, 1));
console.log('letters', out.letters.length, '->', target);
console.log('unsettled:', out.letters.filter(l => !l.settled).map(l => l.ch).join(',') || 'none');
