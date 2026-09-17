// Local dev tool: reads the rigger's own authored poses out of the GLB
// ("Pose_OK" / "Pose_OKHand") and reports, per bone, the rotation delta from
// the bind pose as axis+angle. That is ground truth for which local axis
// flexes each finger and in which direction — no geometric guesswork.
const fs = require("fs");
const path = require("path");
const THREE = require("three");

const glbPath = path.join(__dirname, "..", "public", "models", "hand.glb");
const buf = fs.readFileSync(glbPath);
const jsonLen = buf.readUInt32LE(12);
const gltf = JSON.parse(buf.slice(20, 20 + jsonLen).toString("utf8"));

// GLB layout: 12-byte header, then chunks. Chunk 0 is JSON, chunk 1 is BIN.
const binOffset = 20 + jsonLen + (jsonLen % 4 === 0 ? 0 : 4 - (jsonLen % 4)) + 8;
const bin = buf.slice(binOffset);

const COMPONENT = { 5126: Float32Array, 5123: Uint16Array, 5125: Uint32Array };
const NUM_COMPONENTS = { SCALAR: 1, VEC3: 3, VEC4: 4 };

function readAccessor(idx) {
  const acc = gltf.accessors[idx];
  const view = gltf.bufferViews[acc.bufferView];
  const TypedArray = COMPONENT[acc.componentType];
  const n = NUM_COMPONENTS[acc.type];
  const start = (view.byteOffset || 0) + (acc.byteOffset || 0);
  const out = [];
  for (let i = 0; i < acc.count; i++) {
    const row = [];
    for (let c = 0; c < n; c++) {
      const byteIdx = start + (i * n + c) * TypedArray.BYTES_PER_ELEMENT;
      row.push(new TypedArray(bin.buffer, bin.byteOffset + byteIdx, 1)[0]);
    }
    out.push(n === 1 ? row[0] : row);
  }
  return out;
}

const nodeName = (i) => gltf.nodes[i].name;
const bindRot = (i) =>
  gltf.nodes[i].rotation ? new THREE.Quaternion().fromArray(gltf.nodes[i].rotation) : new THREE.Quaternion();

const wanted = process.argv[2] || "Pose_OK";
const anim = gltf.animations.find((a) => a.name === wanted);
if (!anim) {
  console.error(`No animation named ${wanted}. Available:`, gltf.animations.map((a) => a.name));
  process.exit(1);
}

console.log(`=== ${anim.name} — rotation delta from bind pose (last keyframe) ===\n`);

for (const ch of anim.channels) {
  if (ch.target.path !== "rotation") continue;
  const name = nodeName(ch.target.node);
  if (!/^(index|midd|ring|pinky|thumb)_/.test(name)) continue;

  const sampler = anim.samplers[ch.sampler];
  const values = readAccessor(sampler.output);
  const posed = new THREE.Quaternion().fromArray(values[values.length - 1]);

  // delta = bind⁻¹ * posed, i.e. the local rotation the rigger applied.
  const delta = bindRot(ch.target.node).clone().invert().multiply(posed);
  const axis = new THREE.Vector3();
  let angle = 2 * Math.acos(Math.min(1, Math.abs(delta.w)));
  const s = Math.sqrt(1 - delta.w * delta.w);
  if (s < 1e-6) {
    axis.set(0, 0, 0);
    angle = 0;
  } else {
    axis.set(delta.x / s, delta.y / s, delta.z / s);
    if (delta.w < 0) axis.negate();
  }

  console.log(
    `${name.padEnd(14)} angle=${((angle * 180) / Math.PI).toFixed(1).padStart(6)}°  axis=[${axis
      .toArray()
      .map((v) => v.toFixed(2).padStart(5))
      .join(", ")}]`
  );
}
