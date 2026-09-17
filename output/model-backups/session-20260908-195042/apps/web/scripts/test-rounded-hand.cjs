const fs = require('node:fs');
const assert = require('node:assert/strict');
const ts = require('typescript');
require.extensions['.ts'] = (module, filename) => module._compile(ts.transpileModule(fs.readFileSync(filename, 'utf8'), {compilerOptions: {module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020, esModuleInterop: true}}).outputText, filename);
const { TID_ALPHABET, samplePose, localAnchor } = require('../src/components/tid/poses.ts');
const { HandPlayer, frameOf, serialiseFrame } = require('../src/components/tid/motion.ts');
const { createHandUniforms, updateHandUniforms } = require('../src/components/tid/rounded-hand.ts');
const snap = TID_ALPHABET.find(l => l.ch === 'Ç');
assert.deepEqual(samplePose(snap, 0), snap.pose, 'Entering the snap must not jump from its authored pose');
assert.deepEqual(samplePose(snap, 3.4), snap.pose, 'The loop must close on the exact pose');
const distance = localAnchor(snap.pose.left.shape, 'middle_tip').distanceTo(localAnchor(snap.pose.left.shape, 'thumb_tip'));
assert(distance > .22 && distance < .255, `Finger pads must meet without penetrating: ${distance}`);
assert(Math.abs(snap.pose.left.shape.middle.spread) <= 22, 'The middle finger must stay within its own lane');
const fired = samplePose(snap, 1.6);
assert(fired.left.shape.middle.bend[0] > snap.pose.left.shape.middle.bend[0] + 10, 'Middle finger must release into the palm');
assert.deepEqual(fired.right, snap.pose.right, 'C must remain steady during the lower-hand snap');
const player = new HandPlayer(snap);
player.tick(.016, true, 1, true);
const still = JSON.stringify(serialiseFrame(player.current));
for (let i = 0; i < 60; i++) assert.equal(player.tick(.016, true, 1, true), false);
assert.equal(JSON.stringify(serialiseFrame(player.current)), still, 'Reduced motion must remain still');
const slow = new HandPlayer(snap), fast = new HandPlayer(snap);
for (let i = 0; i < 60; i++) slow.tick(1/60, true, .5);
for (let i = 0; i < 30; i++) fast.tick(1/60, true, 1);
assert(Math.abs(slow.current.left.angles[10] - fast.current.left.angles[10]) < 1e-6, 'Playback speed must scale the same timeline');
const uniforms = createHandUniforms();
for (const letter of TID_ALPHABET) {
  updateHandUniforms(uniforms, frameOf(letter.pose));
  for (const point of [...uniforms.uStart.value, ...uniforms.uEnd.value]) {
    assert(point.toArray().every(Number.isFinite), `${letter.ch}: invalid surface`);
    assert(point.w > 0, `${letter.ch}: unclosed surface`);
  }
  for (const bounds of uniforms.uBounds.value) assert(bounds.x < bounds.z && bounds.y < bounds.w);
}
console.log(JSON.stringify({passed: true, letters: 29, snapPadDistance: distance, snapMiddleSpread: snap.pose.left.shape.middle.spread, checks: ['loop continuity', 'pad contact', 'stable C', 'reduced motion', 'speed', 'finite closed surfaces']}, null, 2));

