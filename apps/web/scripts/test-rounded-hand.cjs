const fs = require('node:fs');
const assert = require('node:assert/strict');
const ts = require('typescript');
require.extensions['.ts'] = (module, filename) => module._compile(ts.transpileModule(fs.readFileSync(filename, 'utf8'), {compilerOptions: {module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020, esModuleInterop: true}}).outputText, filename);
const { TID_ALPHABET, samplePose, localAnchor, anchorOffset, LOOP, rigSpec } = require('../src/components/tid/poses.ts');
const { Vector3 } = require('three');
const { HandPlayer, frameOf, serialiseFrame } = require('../src/components/tid/motion.ts');
const { createHandUniforms, updateHandUniforms } = require('../src/components/tid/rounded-hand.ts');
const snap = TID_ALPHABET.find(l => l.ch === 'Ç');
assert.deepEqual(samplePose(snap, 0), snap.pose, 'Entering the snap must not jump from its authored pose');
assert.deepEqual(samplePose(snap, 3.4), snap.pose, 'The loop must close on the exact pose');
const distance = localAnchor(snap.pose.left.shape, 'middle_tip').distanceTo(localAnchor(snap.pose.left.shape, 'thumb_tip'));
const snapPadRadii = rigSpec.middle.radius + rigSpec.thumb.radius;
assert(Math.abs(distance - snapPadRadii) < .01,
  `Finger pads must meet without penetrating: ${distance} vs radii ${snapPadRadii}`);
assert(Math.abs(snap.pose.left.shape.middle.spread) <= 22, 'The middle finger must stay within its own lane');
const fired = samplePose(snap, 1.6);
assert(fired.left.shape.middle.bend[0] > snap.pose.left.shape.middle.bend[0] + 10, 'Middle finger must release into the palm');
assert.deepEqual(fired.right, snap.pose.right, 'C must remain steady during the lower-hand snap');
const dottedI = TID_ALPHABET.find(l => l.ch === 'İ');
const plainI = TID_ALPHABET.find(l => l.ch === 'I');
assert.deepEqual(dottedI.pose.right.shape, plainI.pose.right.shape, 'İ lower right hand must point up');
const dottedIndex = anchorOffset(dottedI.pose.right, 'right', 'index_tip')
  .add(new Vector3(...dottedI.pose.right.position));
const dottedSnap = anchorOffset(dottedI.pose.left, 'left', 'middle_tip')
  .add(new Vector3(...dottedI.pose.left.position));
assert(dottedSnap.y > dottedIndex.y + .25 && Math.abs(dottedSnap.x - dottedIndex.x) < .1,
  'İ left snap must sit above the upright right index');
for (const time of [0, .8, 1.12, 1.22, 1.28, 1.6, 2.35, LOOP]) {
  const current = samplePose(dottedI, time), reference = samplePose(snap, time);
  assert.deepEqual(current.right, dottedI.pose.right, 'İ right hand must stay fixed throughout the snap');
  assert.deepEqual(current.left.shape, reference.left.shape, 'İ must use Ç’s same snapping fingers and timing');
  for (const key of ['position', 'rotation']) for (let i = 0; i < 3; i++)
    assert(Math.abs((current.left[key][i] - dottedI.pose.left[key][i])
      - (reference.left[key][i] - snap.pose.left[key][i])) < 1e-9, 'İ must preserve Ç’s snap recoil');
}
assert.deepEqual(samplePose(dottedI, LOOP), dottedI.pose, 'İ snap loop must close');
const dottedO = TID_ALPHABET.find(l => l.ch === 'Ö');
const oPalm = side => anchorOffset(dottedO.pose[side], side, 'palm')
  .add(new Vector3(...dottedO.pose[side].position));
assert(oPalm('left').y > oPalm('right').y + .3, 'Ö snapping hand must be above the steady hand');
assert(oPalm('right').z > oPalm('left').z, 'Ö steady right hand must be in front');
assert.deepEqual(dottedO.pose.left.shape, snap.pose.left.shape, 'Ö must start with the same snapping grip as Ç');
let oStrikes = 0, oPreviouslyFired = false;
const oThreshold = (dottedO.pose.left.shape.middle.bend[0] + 96) / 2;
for (let frame = 0; frame <= LOOP * 120; frame++) {
  const pose = samplePose(dottedO, frame / 120);
  assert.deepEqual(pose.right, dottedO.pose.right, 'Ö right hand must remain fixed throughout both snaps');
  const firing = pose.left.shape.middle.bend[0] > oThreshold;
  if (firing && !oPreviouslyFired) oStrikes++;
  oPreviouslyFired = firing;
}
assert.equal(oStrikes, 2, 'Ö must make exactly two distinct snaps in each loop');
for (const time of [.5, .65, .69, .73, .77, .87, .98, 1.09]) {
  const first = serialiseFrame(frameOf(samplePose(dottedO, time))).left;
  const second = serialiseFrame(frameOf(samplePose(dottedO, time + .65))).left;
  for (const key of ['rotation', 'angles'])
    assert(first[key].every((v, i) => Math.abs(v - second[key][i]) < 1e-9), 'Ö must repeat the same fast snap');
}
for (const time of [1.1, 1.75])
  assert.deepEqual(samplePose(dottedO, time).left.shape, dottedO.pose.left.shape, 'Ö must fully regrip between snaps');
const oFirstDot = samplePose(dottedO, .77).left.position;
const oSecondDot = samplePose(dottedO, 1.42).left.position;
assert(oSecondDot[0] - oFirstDot[0] > .15 && oSecondDot[0] - oFirstDot[0] < .3,
  'Ö second snap must move slightly right to mark the second dot');
for (const time of [0, 2.5, LOOP])
  assert.deepEqual(samplePose(dottedO, time), dottedO.pose, 'Ö must return to the first dot after both snaps');
const j = TID_ALPHABET.find(l => l.ch === 'J');
const l = TID_ALPHABET.find(l => l.ch === 'L');
// L's thumb opening can be tuned independently of J's reviewed tracing path.
for (const name of ['index', 'middle', 'ring', 'pinky'])
  assert.deepEqual(j.pose.right.shape[name], l.pose.right.shape[name], 'J must retain the upright L fingers');
assert.deepEqual(j.pose.right.shape.thumb.bend, [0, 0, 0], 'J thumb must remain extended');
const jTip = pose => anchorOffset(pose.left, 'left', 'index_tip')
  .add(new Vector3(...pose.left.position)).sub(new Vector3(...pose.right.position));
const jStart = jTip(j.pose), jEnd = jTip(samplePose(j, LOOP * .6));
const indexTip = localAnchor(j.pose.right.shape, 'index_tip');
const thumbTip = localAnchor(j.pose.right.shape, 'thumb_tip');
assert(Math.abs(jStart.y - indexTip.y) < 1e-9 && jStart.x < indexTip.x && indexTip.x - jStart.x < .3,
  'J must begin beside the very top of the right index');
assert(Math.abs(jEnd.x - thumbTip.x) < .05 && jEnd.y > thumbTip.y && jEnd.y - thumbTip.y < .32,
  'J must finish immediately above the right thumb tip');
assert(jStart.x - jEnd.x > .4, 'J must follow the opening all the way to the thumb tip');
for (const progress of [0, .2, .4, .6, .8, 1]) {
  const forward = samplePose(j, LOOP * (.08 + .44 * progress));
  const returning = samplePose(j, LOOP * (.68 + .26 * (1 - progress)));
  assert(jTip(forward).distanceTo(jTip(returning)) < 1e-9,
    'J must retrace the same open stroke instead of circling the opening');
  assert.deepEqual(forward.right, j.pose.right, 'J right L must remain fixed');
  assert.deepEqual(forward.left.shape, j.pose.left.shape);
  assert.deepEqual(forward.left.rotation, j.pose.left.rotation);
  const direction = anchorOffset(forward.left, 'left', 'index_tip')
    .sub(anchorOffset(forward.left, 'left', 'index_2')).normalize();
  assert(direction.z > .9999, 'J left index must keep pointing towards the viewer');
}
assert.deepEqual(samplePose(j, 0), j.pose);
assert.deepEqual(samplePose(j, LOOP), j.pose, 'J loop must return to its authored pose');
const player = new HandPlayer(snap);
player.tick(.016, true, 1, true);
const still = JSON.stringify(serialiseFrame(player.current));
for (let i = 0; i < 60; i++) assert.equal(player.tick(.016, true, 1, true), false);
assert.equal(JSON.stringify(serialiseFrame(player.current)), still, 'Reduced motion must remain still');
const slow = new HandPlayer(snap), fast = new HandPlayer(snap);
for (let i = 0; i < 60; i++) slow.tick(1/60, true, .5);
for (let i = 0; i < 30; i++) fast.tick(1/60, true, 1);
assert(Math.abs(slow.current.left.angles[10] - fast.current.left.angles[10]) < 1e-6, 'Playback speed must scale the same timeline');
const g = TID_ALPHABET.find(l => l.ch === 'G');
const softG = TID_ALPHABET.find(l => l.ch === 'Ğ');
assert.equal(g.sourceVariant, 'TDK/MEB 2012 — C el');
assert.deepEqual(softG.pose, g.pose, 'G and Ğ must use the same two-C-hand hold');
// Thumb contact alone also admits two separate C silhouettes. G places the
// right index inside the left thumb/index opening, including its depth.
const gAnchor = (side, name) => anchorOffset(g.pose[side], side, name)
  .add(new Vector3(...g.pose[side].position));
const openingStart = gAnchor('left', 'thumb_tip');
const opening = gAnchor('left', 'index_tip').sub(openingStart);
const inserted = gAnchor('right', 'index_tip').sub(openingStart);
const insertionFraction = inserted.dot(opening) / opening.lengthSq();
const insertionDistance = inserted.clone().sub(opening.clone().multiplyScalar(insertionFraction)).length();
assert(insertionFraction > .3 && insertionFraction < .8 && insertionDistance < .2,
  `G right index must enter the left C opening in 3D: fraction=${insertionFraction}, distance=${insertionDistance}`);
const softGCycle = .85;
const lowered = samplePose(softG, softGCycle / 4);
const raised = samplePose(softG, softGCycle * 3 / 4);
const upperTip = pose => anchorOffset(pose.left, 'left', 'index_tip')
  .add(new Vector3(...pose.left.position));
const baseTip = upperTip(softG.pose), lowTip = upperTip(lowered), highTip = upperTip(raised);
assert(lowTip.y < baseTip.y && highTip.y > baseTip.y, 'Ğ upper index must move down and up');
assert(highTip.y - lowTip.y > .03 && highTip.y - lowTip.y < .1,
  'Ğ index motion must stay subtle');
assert(Math.abs(highTip.x - lowTip.x) < .02, 'Ğ index should move mainly vertically');
for (const animated of [lowered, raised]) {
  assert.deepEqual(animated.right, softG.pose.right, 'Ğ right hand must stay still');
  assert.deepEqual(animated.left.position, softG.pose.left.position);
  assert.deepEqual(animated.left.rotation, softG.pose.left.rotation);
  for (const finger of ['thumb', 'middle', 'ring', 'pinky'])
    assert.deepEqual(animated.left.shape[finger], softG.pose.left.shape[finger]);
}
assert.deepEqual(samplePose(softG, 0), softG.pose);
assert.deepEqual(samplePose(softG, softGCycle), softG.pose, 'Ğ must finish one quick cycle in 0.85 seconds');
assert.deepEqual(samplePose(softG, LOOP), softG.pose, 'Ğ loop must return to the authored pose');
assert.deepEqual(samplePose(g, LOOP / 4), g.pose, 'G must remain static');
const reducedSoftG = new HandPlayer(softG);
for (let i = 0; i < 120; i++) assert.equal(reducedSoftG.tick(1/60, true, 1, true), false);
assert.deepEqual(serialiseFrame(reducedSoftG.current), serialiseFrame(frameOf(softG.pose)));
const pPose = TID_ALPHABET.find(l => l.ch === 'P').pose;
assert(pPose.left.visible && !pPose.right.visible, 'P must use only the left hand');
const pHand = pPose.left;
const pAnchor = name => anchorOffset(pHand, 'left', name);
const pDirection = name => pAnchor(`${name}_tip`).sub(pAnchor(`${name}_2`)).normalize();
assert(pHand.shape.index.bend.every((bend, i) => bend >= [-55, 0, 0][i] && bend <= [0, 110, 80][i]),
  'P extends back only at the knuckle; both outer index joints must curl forwards');
assert(pDirection('index').x < -.999, 'P index tip must return left towards the middle finger');
assert(pDirection('middle').y < -.999, 'P middle finger must point straight down');
const pJoint = pAnchor('middle_1'), pTip = pAnchor('index_tip');
assert(Math.hypot(pTip.x - pJoint.x, pTip.y - pJoint.y) < .01,
  'P index tip must align with the actual middle joint in side view');
assert(Math.abs(pTip.distanceTo(pJoint) - (rigSpec.index.radius * .90 + rigSpec.middle.radius)) < .006,
  'P index pad must meet the side of the middle joint without merging bone centres');
assert(pAnchor('index_1').x > pJoint.x + .25,
  'P must keep an open rounded bowl to the right of the stem');
const rLetter = TID_ALPHABET.find(l => l.ch === 'R');
const rPose = rLetter.pose;
assert(rPose.right.visible && rPose.left.visible, 'R must show both hands');
assert.deepEqual(rPose.left.shape, pPose.left.shape, 'R left hand must retain the reviewed P shape');
assert.deepEqual(rPose.left.rotation, pPose.left.rotation, 'R left hand must retain P orientation');
const rWorld = (side, name) => anchorOffset(rPose[side], side, name)
  .add(new Vector3(...rPose[side].position));
const rTip = rWorld('right', 'index_tip'), rLeftTip = rWorld('left', 'index_tip');
assert.deepEqual(rPose.contact, {right: 'index_tip', left: 'index_tip'});
assert(rTip.distanceTo(rLeftTip) > .20 && rTip.distanceTo(rLeftTip) < .24,
  'R fingertip pads must join with room for their surfaces');
const rLeg = rWorld('right', 'index_0').sub(rTip).normalize();
assert(rLeg.x > .5 && rLeg.y < -.6, 'R leg must extend diagonally down-right from the joint');
assert.deepEqual(rPose.right.shape.index.bend, [0, 0, 0], 'R right index must stay straight');
for (const name of ['thumb', 'middle', 'ring', 'pinky'])
  assert.deepEqual(rPose.right.shape[name], plainI.pose.right.shape[name], 'R other right fingers must stay closed');
const sPose = TID_ALPHABET.find(l => l.ch === 'S').pose;
for (const side of ['right', 'left']) {
  assert.deepEqual(sPose[side].shape, g.pose[side].shape, 'S must retain G hand shapes');
  assert.deepEqual(sPose[side].rotation, g.pose[side].rotation, 'S must retain G hand orientations');
}
assert.deepEqual(sPose.contact, {right: 'index_tip', left: 'thumb_tip'},
  'S must join the right index to the left thumb');
const sWorld = (side, name) => anchorOffset(sPose[side], side, name)
  .add(new Vector3(...sPose[side].position));
const sContact = sWorld('right', 'index_tip').sub(sWorld('left', 'thumb_tip'));
const sPadRadii = rigSpec.index.radius * .90 + rigSpec.thumb.radius * .96;
assert(Math.abs(sContact.length() - sPadRadii) < .003,
  'S fingertip surfaces must meet without merging their centres');
assert(sContact.clone().normalize().dot(sWorld('left', 'thumb_tip').sub(sWorld('left', 'thumb_2')).normalize()) > .8,
  'S right fingertip must meet the end of the left thumb, not its side');
const sh = TID_ALPHABET.find(l => l.ch === 'Ş');
assert.deepEqual(sh.pose, sPose, 'Ş must start in the exact reviewed S pose');
for (const time of [0, .35, .70, 1.12, 1.16, 1.22, 1.28, 1.7, 2.1, 2.35, 2.8, LOOP]) {
  const current = samplePose(sh, time);
  assert.deepEqual(current.left, sPose.left, 'Ş left hand must remain still throughout the snap');
  for (const key of ['position', 'rotation']) assert.deepEqual(current.right[key], sPose.right[key]);
  for (const name of ['index', 'ring', 'pinky'])
    assert.deepEqual(current.right.shape[name], sPose.right.shape[name], 'Ş must keep its upper contact and closed fingers');
}
const shReady = samplePose(sh, .7).right.shape;
assert(Math.abs(localAnchor(shReady, 'middle_tip').distanceTo(localAnchor(shReady, 'thumb_tip')) - snapPadRadii) < .01,
  'Ş lower fingers must prepare a real thumb/middle pad contact');
for (const time of [.8, 1.12, 1.16, 1.22, 1.28, 1.75, 2.35]) for (const name of ['thumb', 'middle'])
  assert.deepEqual(samplePose(sh, time).right.shape[name], samplePose(snap, time).left.shape[name],
    'Ş must use the reviewed single-snap timing and finger strike');
assert(localAnchor(shReady, 'middle_tip').distanceTo(localAnchor(samplePose(sh, 1.28).right.shape, 'middle_tip')) > .3,
  'Ş middle finger must flick visibly into the palm');
assert.deepEqual(samplePose(sh, 2.8), sPose, 'Ş must return to S after one snap');
assert.deepEqual(samplePose(sh, LOOP), sPose, 'Ş must loop without a jump');
const reducedSh = new HandPlayer(sh);
for (let i = 0; i < 120; i++) assert.equal(reducedSh.tick(1/60, true, 1, true), false);
assert.deepEqual(serialiseFrame(reducedSh.current), serialiseFrame(frameOf(sPose)));
const uPose = TID_ALPHABET.find(l => l.ch === 'U').pose;
assert(uPose.right.visible && !uPose.left.visible, 'U must use a single right hand');
for (const name of ['middle', 'ring', 'pinky'])
  assert.deepEqual(uPose.right.shape[name], plainI.pose.right.shape[name], 'U must close the other three fingers');
assert.deepEqual(uPose.right.shape.thumb.bend, [0, 0, 0], 'U must extend its thumb');
assert.deepEqual(uPose.right.shape.index.bend.slice(1), [0, 0], 'U index must stay straight beyond its base knuckle');
const uAnchor = name => anchorOffset(uPose.right, 'right', name);
for (const name of ['index', 'thumb'])
  assert(uAnchor(`${name}_tip`).sub(uAnchor(`${name}_2`)).normalize().y > .95,
    'Both U fingers must point up in side view');
const uOpening = uAnchor('thumb_tip').sub(uAnchor('index_tip'));
assert(uOpening.x > .5 && Math.abs(uOpening.y) < .2,
  'U must have two separated upright tips around an open bowl');
const dottedU = TID_ALPHABET.find(l => l.ch === 'Ü');
assert.deepEqual(dottedU.pose.right.shape, uPose.right.shape, 'Ü must retain the reviewed U fingers');
assert.deepEqual(dottedU.pose.right.rotation, uPose.right.rotation, 'Ü must retain U side view');
assert(!dottedU.pose.contact, 'Ü snaps above U without inter-hand contact');
const dottedUWorld = (side, name) => anchorOffset(dottedU.pose[side], side, name)
  .add(new Vector3(...dottedU.pose[side].position));
assert(dottedUWorld('left', 'palm').y > dottedUWorld('right', 'index_tip').y + .2,
  'Ü snapping palm must sit above the upright U fingers');
assert(dottedUWorld('left', 'palm').z > dottedUWorld('right', 'index_tip').z,
  'Ü snapping hand must appear in front of U, not behind it');
let uStrikes = 0, uPreviouslyFired = false;
for (let frame = 0; frame <= LOOP * 120; frame++) {
  const time = frame / 120, current = samplePose(dottedU, time), reference = samplePose(dottedO, time);
  assert.deepEqual(current.right, dottedU.pose.right, 'Ü right U must stay fixed');
  for (const key of ['shape', 'rotation'])
    assert.deepEqual(current.left[key], reference.left[key], 'Ü must use Ö’s two quick snaps');
  for (let i = 0; i < 3; i++)
    assert(Math.abs((current.left.position[i] - dottedU.pose.left.position[i])
      - (reference.left.position[i] - dottedO.pose.left.position[i])) < 1e-9,
      'Ü must preserve Ö’s shift between the two dots and recoil');
  const firing = current.left.shape.middle.bend[0] > oThreshold;
  if (firing && !uPreviouslyFired) uStrikes++;
  uPreviouslyFired = firing;
}
assert.equal(uStrikes, 2, 'Ü must snap exactly twice per loop');
assert.deepEqual(samplePose(dottedU, 2.5), dottedU.pose, 'Ü must reset after both snaps');
const vPose = TID_ALPHABET.find(l => l.ch === 'V').pose;
const yPose = TID_ALPHABET.find(l => l.ch === 'Y').pose;
assert(yPose.right.visible && yPose.left.visible, 'Y must show both hands');
assert.deepEqual(yPose.right.shape, vPose.right.shape, 'Y right hand must use the reviewed V fingers');
assert.deepEqual(yPose.right.rotation, [0, 180, -9], 'Y right V must show the back of the hand and approach from slightly right');
assert.deepEqual(yPose.left.shape.index.bend, [0, 0, 0], 'Y left index must stay straight');
for (const name of ['thumb', 'middle', 'ring', 'pinky'])
  assert.deepEqual(yPose.left.shape[name], plainI.pose.right.shape[name], 'Y other left fingers must stay closed');
const yTip = anchorOffset(yPose.left, 'left', 'index_tip')
  .add(new Vector3(...yPose.left.position)).sub(new Vector3(...yPose.right.position));
const yIndexRoot = anchorOffset(yPose.right, 'right', 'index_0');
const yMiddleRoot = anchorOffset(yPose.right, 'right', 'middle_0');
assert(yTip.x > Math.min(yIndexRoot.x, yMiddleRoot.x) && yTip.x < Math.max(yIndexRoot.x, yMiddleRoot.x)
  && Math.abs(yTip.y - (yIndexRoot.y + yMiddleRoot.y) / 2) < .05 && yTip.z > .2,
  'Y left fingertip must meet the front of the V roots');
const yStem = anchorOffset(yPose.left, 'left', 'index_0').sub(anchorOffset(yPose.left, 'left', 'index_tip'));
assert(yStem.x < -.04 && yStem.x > -.18 && Math.abs(yStem.z) < 1e-8 && yStem.y < -.8,
  'Y left index must approach the V roots from slightly left while staying nearly vertical');
const uniforms = createHandUniforms();
for (const letter of TID_ALPHABET) {
  updateHandUniforms(uniforms, frameOf(letter.pose), letter.viewElevation, letter.viewAzimuth);
  for (const point of [...uniforms.uStart.value, ...uniforms.uEnd.value]) {
    assert(point.toArray().every(Number.isFinite), `${letter.ch}: invalid surface`);
    assert(point.w > 0, `${letter.ch}: unclosed surface`);
  }
  for (const bounds of uniforms.uBounds.value) assert(bounds.x < bounds.z && bounds.y < bounds.w);
}
console.log(JSON.stringify({passed: true, letters: 29, snapPadDistance: distance, snapMiddleSpread: snap.pose.left.shape.middle.spread, gInsertion: { fraction: insertionFraction, distance: insertionDistance }, softGIndexTravel: highTip.y - lowTip.y, checks: ['loop continuity', 'pad contact', 'stable C', 'İ matches Ç snap', 'Ö two upper-hand snaps with steady right hand', 'J open stroke beside L with fingertip facing viewer', 'reduced motion', 'speed', 'G/Ğ C-hand hold', 'G index inside left C', 'Ğ subtle upper-index motion', 'finite closed surfaces']}, null, 2));

