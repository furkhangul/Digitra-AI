// Software fixtures from the unchanged teaching animations, never training data
// or a human-camera accuracy evaluation.
const fs = require('fs');
const path = require('path');
const { createRequire } = require('module');
const root = path.resolve(__dirname, '../../..');
const web = path.join(root, 'apps/web');
const req = createRequire(path.join(web, 'package.json'));
const ts = req('typescript');
const { Vector3 } = req('three');
require.extensions['.ts'] = (mod, filename) => {
  const code = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020, esModuleInterop: true },
    fileName: filename,
  }).outputText;
  mod._compile(code, filename);
};
const { TID_ALPHABET, FINGERS, samplePose, anchorOffset } = require(path.join(web, 'src/components/tid/poses.ts'));
const anchors = ['wrist', ...FINGERS.flatMap(f => [`${f}_0`, `${f}_1`, `${f}_2`, `${f}_tip`])];
const sequences = {};
for (const letter of TID_ALPHABET) {
  sequences[letter.ch] = Array.from({length: letter.motion ? 103 : 1}, (_, i) => {
    const time = i / 30;
    const pose = samplePose(letter, time);
    return {time, points: ['right', 'left'].map(side => anchors.map(name => {
      const p = anchorOffset(pose[side], side, name).add(new Vector3(...pose[side].position));
      return [p.x, -p.y];
    }))};
  });
}
const output = path.join(root, 'artifacts/tid-v6/motion/authored_sequences.json');
fs.mkdirSync(path.dirname(output), {recursive: true});
fs.writeFileSync(output, JSON.stringify({scope: 'authored geometry software fixtures only', sequences}));
console.log(output);
