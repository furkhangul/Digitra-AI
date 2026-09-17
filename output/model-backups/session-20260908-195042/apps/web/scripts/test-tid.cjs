const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const ts = require('typescript');
require.extensions['.ts'] = (module, filename) => module._compile(ts.transpileModule(fs.readFileSync(filename, 'utf8'), {compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020,esModuleInterop:true}}).outputText, filename);
const {TID_ALPHABET, localAnchor, FINGERS, samplePose} = require('../src/components/tid/poses.ts');
const {HandPlayer, frameOf, serialiseFrame} = require('../src/components/tid/motion.ts');
const json = p => JSON.stringify(serialiseFrame(p));
assert.equal(TID_ALPHABET.map(l=>l.ch).join(''), 'ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ');
let frames=0, largestPositionStep=0, largestAngleStep=0;
for (const a of TID_ALPHABET) for (const b of TID_ALPHABET) {
  const p = new HandPlayer(a);
  const before=json(p.current);
  p.select(b);
  assert.equal(json(p.current),before,'Selection must not snap the rendered pose');
  for(let i=0;i<180;i++) {
    const previous=serialiseFrame(p.current);
    p.tick(1/60);
    for(const side of ['right','left']) {
      const h=p.current[side];
      const values=[...h.position.toArray(),...h.rotation.toArray(),...h.angles,h.presence];
      assert(values.every(Number.isFinite),`${a.ch}->${b.ch}: nonfinite frame`);
      assert(Math.abs(h.rotation.length()-1)<1e-6);
      assert(h.presence>=0&&h.presence<=1);
      const step=Math.hypot(...h.position.toArray().map((v,j)=>v-previous[side].position[j]));
      largestPositionStep=Math.max(largestPositionStep,step);
      largestAngleStep=Math.max(largestAngleStep,...h.angles.map((v,j)=>Math.abs(v-previous[side].angles[j])));
      assert(step < .3,`${a.ch}->${b.ch}: positional jump ${step}`);
    }
    frames++;
  }
  assert(p.settled);
  const held=json(p.current); p.tick(1,false); assert.equal(json(p.current),held,'Pause must hold pose');
}
// Rapid mid-flight selection, playback-rate changes, and a background-tab gap.
const p=new HandPlayer(TID_ALPHABET[0]);
for(const letter of TID_ALPHABET) {
  p.tick(.017,true,.5);
  const prior=json(p.current);p.select(letter);assert.equal(json(p.current),prior);
  p.tick(.017,true,1.5);
}
p.tick(60);assert(!p.settled,'Background-tab gap must not teleport to the target');
for(let i=0;i<200;i++)p.tick(.017);
assert(p.settled);
for (const l of TID_ALPHABET.filter(l=>l.motion)) {
  assert.deepEqual(samplePose(l,0),samplePose(l,3.4),`${l.ch}: loop endpoint`);
}
const model = fs.readFileSync(path.join(__dirname,'../public/models/tid-hand.glb'));
assert.equal(model.readUInt32LE(0),0x46546c67);
const gltf=JSON.parse(model.subarray(20,20+model.readUInt32LE(12)).toString());
const names=new Set(gltf.nodes.map(n=>n.name));
for(const f of FINGERS)for(const suffix of ['0','1','2','tip'])assert(names.has(`${f}_${suffix}`));
const result={letters:TID_ALPHABET.length,pairs:TID_ALPHABET.length**2,frames,largestPositionStep,largestAngleStep,modelBytes:model.length,passed:true};
fs.mkdirSync(path.join(__dirname,'../../../output/models'),{recursive:true});
fs.writeFileSync(path.join(__dirname,'../../../output/models/motion-test-report.json'),JSON.stringify(result,null,2));
console.log(result);
if(process.argv.includes('--dump')) fs.writeFileSync(path.join(__dirname,'../../../tmp/tid/poses.json'),JSON.stringify(TID_ALPHABET,null,2));
if(process.argv.includes('--timeline')) {
  const player=new HandPlayer(TID_ALPHABET[0],true);
  const samples=[];
  const append = (letter, count) => { for(let i=0;i<count;i++){ samples.push({letter, ...serialiseFrame(player.current)});player.tick(1/24); } };
  append('A',36);
  for(const letter of [...TID_ALPHABET.slice(1),TID_ALPHABET[0]]){player.select(letter);append(letter.ch,76);}
  fs.writeFileSync(path.join(__dirname,'../../../tmp/tid/timeline.json'),JSON.stringify({fps:24,samples}));
  console.log(`Exported ${samples.length} timeline frames.`);
}
