// Read the existing teaching geometry without changing any web source or asset.
const fs = require('fs');
const path = require('path');
const crypto = require('crypto');
const { createRequire } = require('module');
const root = path.resolve(__dirname, '../../..');
const web = path.join(root, 'apps/web');
const req = createRequire(path.join(web, 'package.json'));
const ts = req('typescript');
require.extensions['.ts'] = (mod, filename) => {
  const code = ts.transpileModule(fs.readFileSync(filename, 'utf8'), {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020, esModuleInterop: true, resolveJsonModule: true },
    fileName: filename,
  }).outputText;
  mod._compile(code, filename);
};
const source = path.join(web, 'src/components/tid');
const { TID_ALPHABET } = require(path.join(source, 'poses.ts'));
const { frameOf } = require(path.join(source, 'motion.ts'));
const { createHandUniforms, updateHandUniforms, contentExtent, handFragmentShader } = require(path.join(source, 'rounded-hand.ts'));
const frames = [];
for (const letter of TID_ALPHABET) {
  // Only the authored pose is used: transition frames must never inherit a label.
  for (const [e, a] of [[0,0],[-6,-8],[6,8],[-4,8],[4,-8]]) {
    const uniforms = createHandUniforms();
    updateHandUniforms(uniforms, frameOf(letter.pose), (letter.viewElevation || 0)+e, (letter.viewAzimuth || 0)+a);
    const extent = contentExtent(uniforms);
    uniforms.uScale.value = Math.max(extent.x, extent.y, 1.15)*1.05;
    const packed = Object.fromEntries(Object.entries(uniforms).map(([k, {value}]) => [k,
      typeof value === 'number' ? value : Array.isArray(value) ? value.flatMap(x=>x.toArray()) : value.toArray()]));
    frames.push({label:letter.ch, dynamic:Boolean(letter.motion), view:[e,a], uniforms:packed});
  }
}
const hashes = Object.fromEntries(fs.readdirSync(source).filter(f=>/\.(ts|tsx|json)$/.test(f)).map(f=>[f,crypto.createHash('sha256').update(fs.readFileSync(path.join(source,f))).digest('hex')]));
const out = path.resolve(process.argv[2] || path.join(root,'artifacts/tid-v6/hands.json'));
fs.mkdirSync(path.dirname(out),{recursive:true});
fs.writeFileSync(out,JSON.stringify({source:'Digitra teaching hands',source_hashes:hashes,frames,fragment:handFragmentShader},null,2));
console.log(JSON.stringify({path:out,letters:TID_ALPHABET.length,frames:frames.length}));
