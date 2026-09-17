const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const ts = require('typescript');

const file = path.resolve(__dirname, '../src/lib/hand-detection.ts');
const compiled = ts.transpileModule(fs.readFileSync(file, 'utf8'), {
  compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 },
}).outputText;
const loaded = new Module(file, module);
loaded._compile(compiled, file);
const { StableHandDetections } = loaded.exports;

function hand(dx = 0) {
  const xy = [[.5,.75],[.45,.69],[.41,.63],[.38,.58],[.35,.54],
    [.44,.60],[.43,.51],[.43,.45],[.43,.40], [.49,.58],[.49,.48],[.49,.41],[.49,.36],
    [.54,.60],[.55,.51],[.55,.45],[.55,.40], [.59,.63],[.61,.56],[.62,.51],[.63,.47]];
  return xy.map(([x,y]) => ({ x:x+dx, y, z:0, visibility:1 }));
}
function result(...hands) {
  const categories = hands.map((_,i) => [{categoryName:i===0?'Left':'Right',score:.51,index:i,displayName:''}]);
  return {landmarks:hands,worldLandmarks:hands,handedness:categories,handednesses:categories};
}
function settle(gate, data, start=0) {
  gate.update(data,start,1);gate.update(data,start+80,1);
  return gate.update(data,start+160,1);
}

test('a transient detection is suppressed; a stable hand is admitted', () => {
  const gate=new StableHandDetections(),data=result(hand());
  assert.equal(gate.update(data,0,1).landmarks.length,0);
  assert.equal(gate.update(data,80,1).landmarks.length,0);
  assert.equal(gate.update(data,160,1).landmarks.length,1);
});
test('brief overlap hides missing landmarks but preserves matching track identity', () => {
  const gate=new StableHandDetections();settle(gate,result(hand()));
  assert.equal(gate.update(result(),180,1).landmarks.length,0);
  assert.equal(gate.update(result(hand()),200,1).landmarks.length,1);
  assert.equal(gate.update(result(),500,1).landmarks.length,0);
  assert.equal(gate.update(result(hand()),520,1).landmarks.length,0);
});
test('overlap recovery never admits a new hand at a different position', () => {
  const gate=new StableHandDetections();settle(gate,result(hand(-.3)));
  gate.update(result(),180,1);
  assert.equal(gate.update(result(hand(.3)),200,1).landmarks.length,0);
});
test('a jump to another region does not inherit the previous hand confirmation', () => {
  const gate=new StableHandDetections();settle(gate,result(hand(-.3)));
  assert.equal(gate.update(result(hand(.3)),180,1).landmarks.length,0);
});
test('collapsed, incomplete and nonfinite landmark sets never pass', () => {
  const malformed=[hand().fill({x:.5,y:.5,z:0}),hand().slice(0,20),hand().map((p,i)=>i===8?{...p,z:NaN}:p)];
  for (const points of malformed) assert.equal(settle(new StableHandDetections(),result(points)).landmarks.length,0);
});
test('folded and side-on fingers remain valid, including hands near the face', () => {
  const folded=hand();
  for(const base of [5,9,13,17]) {
    const p=folded[base];
    folded[base+1]={...p,y:p.y-.05,z:-.04};
    folded[base+2]={...p,y:p.y-.02,z:-.07};
    folded[base+3]={...p,y:p.y+.02,z:-.05};
  }
  const side=hand().map(p=>({...p,x:.5,z:p.x-.5}));
  const raised=hand().map(p=>({...p,y:p.y-.3}));
  for(const points of [folded,side,raised]) assert.equal(settle(new StableHandDetections(),result(points)).landmarks.length,1);
});
test('two-hand ordering and metadata stay aligned when detection order changes', () => {
  const gate=new StableHandDetections(),left=hand(-.2),right=hand(.2);
  settle(gate,result(left,right));
  const swapped=result(right,left);
  const filtered=gate.update(swapped,180,1);
  assert.deepEqual(filtered.landmarks,[right,left]);
  assert.deepEqual(filtered.worldLandmarks,swapped.worldLandmarks);
  assert.deepEqual(filtered.handedness,swapped.handedness);
});
test('filtering one candidate keeps metadata for the surviving hand; reset drops history', () => {
  const gate=new StableHandDetections();
  const input=result(hand().fill({x:.5,y:.5,z:0}),hand());
  const filtered=settle(gate,input);
  assert.equal(filtered.landmarks.length,1);
  assert.deepEqual(filtered.handedness,[input.handedness[1]]);
  gate.reset();
  assert.equal(gate.update(input,180,1).landmarks.length,0);
});
