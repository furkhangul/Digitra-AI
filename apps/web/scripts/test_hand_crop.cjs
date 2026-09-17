const assert=require('node:assert/strict');
const {test}=require('node:test');
const fs=require('node:fs');
const path=require('node:path');
const Module=require('node:module');
const ts=require('typescript');
const file=path.resolve(__dirname,'../src/lib/hand-crop.ts');
const loaded=new Module(file,module);
loaded._compile(ts.transpileModule(fs.readFileSync(file,'utf8'),{
  compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020},
}).outputText,file);
const {HandCropTracker}=loaded.exports;
const box=(x,y,w=.1,h=.2)=>[{x,y},{x:x+w,y:y+h}];

test('a distant hand retains the same relative size in the model input',()=>{
  const near=new HandCropTracker().update([box(.4,.3,.15,.3)],1280,720,0);
  const far=new HandCropTracker().update([box(.46,.42,.06,.12)],1280,720,0);
  assert.ok(Math.abs((.3*720/near.height)-(.12*720/far.height))<1e-9);
  assert.ok(.3*720/near.height>0.7,'hand should fill most of the crop, as in training images');
  assert.ok(far.height<720*.3,'distant hand must not expand to the whole camera height');
});
test('the crop contains both hands and stays inside camera bounds',()=>{
  const hands=[box(.1,.2),box(.7,.5)];
  const crop=new HandCropTracker().update(hands,1280,720,0);
  for(const p of hands.flat()){
    assert.ok(p.x*1280>=crop.x && p.x*1280<=crop.x+crop.width);
    assert.ok(p.y*720>=crop.y && p.y*720<=crop.y+crop.height);
  }
  assert.ok(crop.x>=0 && crop.y>=0 && crop.x+crop.width<=1280 && crop.y+crop.height<=720);
});
test('a briefly missed second hand stays in context, then expires',()=>{
  const tracker=new HandCropTracker(),left=box(.2,.3),right=box(.5,.3);
  const paired=tracker.update([left,right],1280,720,0);
  const brief=tracker.update([left],1280,720,110);
  assert.deepEqual(brief,paired);
  const expired=tracker.update([left],1280,720,701);
  assert.ok(expired.width<paired.width);
});
test('no-hand frames and explicit reset discard old partner context',()=>{
  for(const reset of [tracker=>tracker.reset(),tracker=>tracker.update([],1280,720,100)]){
    const tracker=new HandCropTracker(),left=box(.2,.3),right=box(.5,.3);
    tracker.update([left,right],1280,720,0);reset(tracker);
    assert.deepEqual(tracker.update([left],1280,720,150),new HandCropTracker().update([left],1280,720,150));
  }
});
test('a new hand elsewhere does not borrow an unrelated old pair',()=>{
  const tracker=new HandCropTracker();tracker.update([box(.1,.1),box(.3,.1)],1280,720,0);
  const moved=[box(.8,.7)];
  assert.deepEqual(tracker.update(moved,1280,720,100),new HandCropTracker().update(moved,1280,720,100));
});
test('edge crops remain valid; empty and nonfinite inputs are rejected',()=>{
  for(const dimensions of [[1280,720],[720,1280]]){
    const [w,h]=dimensions;
    for(const points of [box(0,0),box(.9,.8)]){
      const crop=new HandCropTracker().update([points],w,h,0);
      assert.ok(crop.x>=0 && crop.y>=0 && crop.x+crop.width<=w && crop.y+crop.height<=h);
    }
  }
  assert.equal(new HandCropTracker().update([],1280,720,0),null);
  assert.equal(new HandCropTracker().update([[{x:NaN,y:.5}]],1280,720,0),null);
  assert.equal(new HandCropTracker().update([box(.3,.4)],0,720,0),null);
});
