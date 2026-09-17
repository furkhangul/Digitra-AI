// Local dev tool: works out which local axis (and sign) flexes the finger
// bones toward the palm, so the hero animation curls like a real hand
// instead of hyperextending or splaying sideways.
const fs = require("fs");
const path = require("path");
const THREE = require("three");

const glbPath = path.join(__dirname, "..", "public", "models", "hand.glb");
const buf = fs.readFileSync(glbPath);
const jsonLen = buf.readUInt32LE(12);
const gltf = JSON.parse(buf.slice(20, 20 + jsonLen).toString("utf8"));

// Rebuild the node hierarchy as three.js Object3Ds so we get real world matrices.
const objects = gltf.nodes.map((n) => {
  const o = new THREE.Object3D();
  o.name = n.name || "";
  if (n.translation) o.position.fromArray(n.translation);
  if (n.rotation) o.quaternion.fromArray(n.rotation);
  if (n.scale) o.scale.fromArray(n.scale);
  return o;
});
gltf.nodes.forEach((n, i) => {
  (n.children || []).forEach((c) => objects[i].add(objects[c]));
});
const root = objects[gltf.scenes[gltf.scene].nodes[0]];
root.updateMatrixWorld(true);

const byName = {};
objects.forEach((o) => (byName[o.name] = o));

const worldPos = (name) => byName[name].getWorldPosition(new THREE.Vector3());

// Hand frame: "up" runs wrist -> middle fingertip, "across" runs index -> pinky
// knuckle. Their cross product is the palm normal; a real finger curls toward
// the palm side, i.e. against that normal.
const wrist = worldPos("radius_ulna");
const midTip = worldPos("midd_dist");
const indexMcp = worldPos("index_prox");
const pinkyMcp = worldPos("pinky_prox");

const up = midTip.clone().sub(wrist).normalize();
const across = pinkyMcp.clone().sub(indexMcp).normalize();
const palmNormal = new THREE.Vector3().crossVectors(across, up).normalize();

console.log("up (wrist->midTip):", up.toArray().map((v) => v.toFixed(3)));
console.log("across (index->pinky):", across.toArray().map((v) => v.toFixed(3)));
console.log("palmNormal:", palmNormal.toArray().map((v) => v.toFixed(3)));
console.log("");

// For each candidate local axis, rotate the whole index chain and see which
// way the fingertip travels relative to the palm normal.
const CHAIN = (process.argv[2] || "index_prox,index_midd,index_dist").split(",");
console.log("chain:", CHAIN.join(" -> "), "\n");
const candidates = {
  "+X": new THREE.Vector3(1, 0, 0),
  "-X": new THREE.Vector3(-1, 0, 0),
  "+Z": new THREE.Vector3(0, 0, 1),
  "-Z": new THREE.Vector3(0, 0, -1),
};

const binds = {};
CHAIN.forEach((n) => (binds[n] = byName[n].quaternion.clone()));
const tipBefore = worldPos(CHAIN[CHAIN.length - 1]);

for (const [label, axis] of Object.entries(candidates)) {
  CHAIN.forEach((n, i) => {
    const factor = (i + 1) / CHAIN.length;
    const delta = new THREE.Quaternion().setFromAxisAngle(axis, factor * 1.2);
    byName[n].quaternion.copy(binds[n]).multiply(delta);
  });
  root.updateMatrixWorld(true);

  const tipAfter = worldPos(CHAIN[CHAIN.length - 1]);
  const motion = tipAfter.clone().sub(tipBefore);
  // Negative dot = moved toward the palm side => correct flexion.
  const towardPalm = -motion.dot(palmNormal);
  // How much it also dropped toward the wrist (curling shortens reach).
  const towardWrist = -motion.dot(up);
  const sideways = Math.abs(motion.dot(across));

  console.log(
    `${label}: towardPalm=${towardPalm.toFixed(2)}  towardWrist=${towardWrist.toFixed(2)}  sideways=${sideways.toFixed(2)}`
  );

  CHAIN.forEach((n) => byName[n].quaternion.copy(binds[n]));
  root.updateMatrixWorld(true);
}
