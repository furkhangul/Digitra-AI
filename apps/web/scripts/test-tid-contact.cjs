const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),ts=require('typescript');
require.extensions['.ts']=(m,f)=>m._compile(ts.transpileModule(fs.readFileSync(f,'utf8'),{compilerOptions:{module:ts.ModuleKind.CommonJS,target:ts.ScriptTarget.ES2020,esModuleInterop:true}}).outputText,f);
const {TID_ALPHABET}=require('../src/components/tid/poses.ts');
const {HandPlayer}=require('../src/components/tid/motion.ts');
const {ContactGuard}=require('../src/components/tid/contact-guard.ts');
const letters=TID_ALPHABET.filter(l=>'BGĞOÖ'.includes(l.ch));
const audit=new ContactGuard();let checked=0,maxStep=0,failures=[];
for(const a of letters)for(const b of letters){
 const player=new HandPlayer(a,true);player.select(b);
 for(let i=0;i<120;i++){
  const before=player.current.right.position.clone();player.tick(1/30);
  maxStep=Math.max(maxStep,before.distanceTo(player.current.right.position));
  if(audit.intersects(player.current))failures.push(`${a.ch}->${b.ch}@${i}`);
  checked++;
 }
 player.dispose();
}
audit.dispose();
const report={scope:'Skinned low-resolution inter-hand collision surfaces, B/G/Ğ/O/Ö transitions and motion holds',transitions:25,frames:checked,maxStep,failures};
fs.writeFileSync(path.join(__dirname,'../../../output/models/playback-contact-report.json'),JSON.stringify(report,null,2));
console.log({frames:checked,maxStep,failures:failures.slice(0,10),totalFailures:failures.length});
assert.equal(failures.length,0);
