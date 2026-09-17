import * as THREE from "three";
import { FINGERS, rigSpec } from "./poses";
import { SIDES, type PairFrame } from "./motion";

// A closed analytic surface, without skin weights, an open wrist or polygon seams.
// Orthographic projection keeps the teaching view fixed and front-facing.
export function createHandUniforms() {
  return {
    uAspect: { value: 1 }, uScale: { value: 2 },
    uStart: { value: Array.from({ length: 80 }, () => new THREE.Vector4()) },
    uEnd: { value: Array.from({ length: 80 }, () => new THREE.Vector4()) },
    uFingerBounds: { value: Array.from({ length: 10 }, () => new THREE.Vector4()) },
    uDepth: { value: [new THREE.Vector2(), new THREE.Vector2()] },
    uInverse: { value: [new THREE.Matrix4(), new THREE.Matrix4()] },
    uBounds: { value: [new THREE.Vector4(), new THREE.Vector4()] },
    uPresence: { value: new THREE.Vector2(1, 1) },
  };
}
export type HandUniforms = ReturnType<typeof createHandUniforms>;
const z = new THREE.Vector3(0, 0, 1);
const y = new THREE.Vector3(0, 1, 0);

export function updateHandUniforms(uniforms: HandUniforms, frame: PairFrame) {
  SIDES.forEach((side, si) => {
    const hand = frame[side];
    const presence = Math.max(.001, hand.presence);
    uniforms.uPresence.value.setComponent(si, hand.presence);
    const matrix = new THREE.Matrix4().compose(hand.position, hand.rotation,
      new THREE.Vector3((side === "left" ? -1 : 1) * presence, presence, presence));
    uniforms.uInverse.value[si].copy(matrix).invert();
    const box = new THREE.Box3();
    for (const x of [-.53, .53]) for (const py of [-.18, 1.05]) for (const pz of [-.26, .26]) {
      box.expandByPoint(new THREE.Vector3(x, py, pz).applyMatrix4(matrix));
    }
    FINGERS.forEach((finger, fi) => {
      const spec = rigSpec[finger];
      const offset = fi * 5;
      const direction = new THREE.Vector3(...spec.direction);
      const axis = new THREE.Vector3().crossVectors(direction, z).normalize();
      const q = new THREE.Quaternion().setFromAxisAngle(z, THREE.MathUtils.degToRad(hand.angles[offset + 3]))
        .multiply(new THREE.Quaternion().setFromAxisAngle(y, THREE.MathUtils.degToRad(hand.angles[offset + 4])));
      const point = new THREE.Vector3(...spec.base);
      const points = [point.clone()];
      for (let joint = 0; joint < 3; joint++) {
        q.multiply(new THREE.Quaternion().setFromAxisAngle(axis, THREE.MathUtils.degToRad(hand.angles[offset + joint])));
        point.add(direction.clone().multiplyScalar(spec.lengths[joint]).applyQuaternion(q));
        points.push(point.clone());
      }
      const curve = new THREE.CatmullRomCurve3(points, false, "centripetal");
      let start = curve.getPoint(0).applyMatrix4(matrix);
      const fingerBox = new THREE.Box3().expandByPoint(start);
      for (let segment = 0; segment < 8; segment++) {
        const end = curve.getPoint((segment + 1) / 8).applyMatrix4(matrix);
        const i = si * 40 + fi * 8 + segment;
        const radius = spec.radius * .97 * presence;
        uniforms.uStart.value[i].set(start.x, start.y, start.z, radius);
        uniforms.uEnd.value[i].set(end.x, end.y, end.z, radius);
        box.expandByPoint(start).expandByPoint(end);
        fingerBox.expandByPoint(end);
        start = end;
      }
      const sphere = fingerBox.getBoundingSphere(new THREE.Sphere());
      uniforms.uFingerBounds.value[si * 5 + fi].set(sphere.center.x, sphere.center.y, sphere.center.z,
        sphere.radius + spec.radius * presence + .08 * presence);
    });
    box.expandByScalar(.17 * presence);
    uniforms.uBounds.value[si].set(box.min.x, box.min.y, box.max.x, box.max.y);
    uniforms.uDepth.value[si].set(box.min.z, box.max.z);
  });
}

export const handVertexShader = /* glsl */ `
  varying vec2 vUv;
  void main() { vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }
`;

export const handFragmentShader = /* glsl */ `
  precision highp float;
  varying vec2 vUv;
  uniform float uAspect;
  uniform float uScale;
  uniform vec4 uStart[80];
  uniform vec4 uEnd[80];
  uniform vec4 uFingerBounds[10];
  uniform vec2 uDepth[2];
  uniform mat4 uInverse[2];
  uniform vec4 uBounds[2];
  uniform vec2 uPresence;
  float smoothUnion(float a, float b, float k) {
    float h = max(k - abs(a - b), 0.0) / k;
    return min(a, b) - h * h * k * .25;
  }
  float ellipsoid(vec3 p, vec3 radius) {
    float k0 = length(p / radius);
    float k1 = length(p / (radius * radius));
    return k0 * (k0 - 1.0) / max(k1, .00001);
  }
  float capsule(vec3 p, vec4 a, vec4 b) {
    vec3 segment = b.xyz - a.xyz;
    float t = clamp(dot(p - a.xyz, segment) / max(dot(segment, segment), .000001), 0.0, 1.0);
    return length(p - a.xyz - segment * t) - mix(a.w, b.w, t);
  }
  float handDistance(vec3 p, int hand) {
    float presence = hand == 0 ? uPresence.x : uPresence.y;
    if (presence < .008) return 100.0;
    vec3 local = (uInverse[hand] * vec4(p, 1.0)).xyz;
    // Broad palm and an oval heel: no forearm or cut plane.
    float palm = ellipsoid(local - vec3(.005, .48, -.035), vec3(.455, .60, .205));
    float thenar = ellipsoid(local - vec3(-.235, .30, .045), vec3(.225, .32, .205));
    float d = smoothUnion(palm, thenar, .12) * presence;
    for (int f = 0; f < 5; f++) {
      vec4 bound = uFingerBounds[hand * 5 + f];
      if (length(p - bound.xyz) - bound.w > d + .07 * presence) continue;
      int i = hand * 40 + f * 8;
      float finger = capsule(p, uStart[i], uEnd[i]);
      for (int j = 1; j < 8; j++) finger = min(finger, capsule(p, uStart[i + j], uEnd[i + j]));

      d = smoothUnion(d, finger, .07 * presence);
    }
    return d;
  }
  bool inside(vec2 p, vec4 b) {
    return p.x > b.x && p.y > b.y && p.x < b.z && p.y < b.w;
  }
  void main() {
    vec2 xy = (vUv - .5) * vec2(uAspect, 1.0) * uScale * 2.0;
    bool right = uPresence.x > .008 && inside(xy, uBounds[0]);
    bool left = uPresence.y > .008 && inside(xy, uBounds[1]);
    if (!right && !left) discard;
    float nearZ = max(right ? uDepth[0].y : -100.0, left ? uDepth[1].y : -100.0);
    float farZ = min(right ? uDepth[0].x : 100.0, left ? uDepth[1].x : 100.0);
    vec3 p = vec3(xy, nearZ + .05);
    float material = 0.0;
    bool hit = false;
    for (int i = 0; i < 88; i++) {
      float a = right ? handDistance(p, 0) : 100.0;
      float b = left ? handDistance(p, 1) : 100.0;
      float d = min(a, b);
      material = a < b ? 0.0 : 1.0;
      if (d < .0015) { hit = true; break; }
      p.z -= max(d * .8, .001);
      if (p.z < farZ - .05) break;
    }
    if (!hit) discard;
    int hand = material < .5 ? 0 : 1;
    vec2 e = vec2(.002, -.002);
    vec3 n = normalize(e.xyy * handDistance(p + e.xyy, hand) + e.yyx * handDistance(p + e.yyx, hand)
      + e.yxy * handDistance(p + e.yxy, hand) + e.xxx * handDistance(p + e.xxx, hand));
    float ao = 1.0;
    for (int i = 1; i <= 3; i++) {
      float h = float(i) * .055;
      ao -= max(h - handDistance(p + n * h, hand), 0.0) * 1.5;
    }
    ao = clamp(ao, .55, 1.0);
    vec3 light = normalize(vec3(-.65, 1.0, 1.5));
    vec3 halfLight = normalize(light + vec3(0, 0, 1));
    float diffuse = max(dot(n, light), 0.0);
    float highlight = pow(max(dot(n, halfLight), 0.0), 38.0);
    float sheen = pow(max(dot(n, halfLight), 0.0), 8.0);
    float rim = pow(1.0 - max(n.z, 0.0), 3.0);
    vec3 base = mix(vec3(.96, .69, .37), vec3(.32, .12, .72), material);
    vec3 color = base * (.54 + diffuse * .57) * ao;
    color += vec3(1.0, .9, .77) * highlight * .43 + vec3(1.0, .78, .55) * sheen * .065;
    color += vec3(.55, .34, 1.0) * rim * .26;
    gl_FragColor = vec4(color, 1.0);
    #include <colorspace_fragment>
  }
`;



