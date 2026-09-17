import * as THREE from "three";
import { FINGERS, rigSpec } from "./poses";
import { SIDES, type PairFrame } from "./motion";

// A closed analytic surface, without skin weights, an open wrist or polygon seams.
// Orthographic projection keeps the teaching view fixed and front-facing.
//
// The shape is one continuous mass: six blended palm volumes (core, knuckle pad,
// heel, thenar, thumb web, hypothenar) joined to per-finger capsule chains whose
// radius follows an along-length profile, so fingers read as flesh, not tubes.
// The same constants drive the Blender build in scripts/build-tid-hand.py.

export const SEGMENTS = 8;
/** Radius multiplier along a finger, from base to tip. The small swells sit at
 * the two joints, so each finger reads as three segments rather than one tube. */
export const FINGER_PROFILE = [1.10, 1.03, 0.985, 0.975, 1.005, 0.955, 0.985, 0.945, 0.90];
/** The thumb keeps its bulk further along; its first segment is the metacarpal. */
export const THUMB_PROFILE = [1.00, 1.02, 1.03, 1.03, 1.02, 1.01, 1.00, 0.99, 0.96];
/** Palm volumes as [centre, radii], blended with PALM_BLEND. Mirrored in GLSL below. */
export const PALM_PARTS: [[number, number, number], [number, number, number]][] = [
  [[.005, .470, -.020], [.400, .500, .185]],
  [[.005, .800, -.005], [.465, .215, .160]],
  [[.020, .185, -.015], [.345, .245, .180]],
  [[-.235, .315, .100], [.268, .380, .232]],
  [[.325, .420, -.005], [.185, .325, .170]],
  [[-.315, .560, .045], [.150, .235, .150]],
];
export const PALM_BLEND = .26;
export const FINGER_BLEND = .105;
/** The thumb grows out of the thenar, so it needs a far wider joint than the fingers. */
export const THUMB_BLEND = .255;

function profileAt(u: number, profile: number[]) {
  const f = u * (profile.length - 1);
  const i = Math.min(Math.floor(f), profile.length - 2);
  return THREE.MathUtils.lerp(profile[i], profile[i + 1], f - i);
}

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

export function updateHandUniforms(uniforms: HandUniforms, frame: PairFrame, viewElevation = 0, viewAzimuth = 0) {
  // Ray marching uses view-space rays. Invert the orbiting camera transform
  // and apply it equally to surfaces and bounds, preserving finger contacts.
  const worldToView = new THREE.Matrix4().makeRotationX(THREE.MathUtils.degToRad(viewElevation))
    .multiply(new THREE.Matrix4().makeRotationY(THREE.MathUtils.degToRad(-viewAzimuth)));
  SIDES.forEach((side, si) => {
    const hand = frame[side];
    const presence = Math.max(.001, hand.presence);
    uniforms.uPresence.value.setComponent(si, hand.presence);
    const matrix = new THREE.Matrix4().compose(hand.position, hand.rotation,
      new THREE.Vector3((side === "left" ? -1 : 1) * presence, presence, presence)).premultiply(worldToView);
    uniforms.uInverse.value[si].copy(matrix).invert();
    const box = new THREE.Box3();
    for (const x of [-.56, .56]) for (const py of [-.20, 1.05]) for (const pz of [-.28, .30]) {
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
      const profile = finger === "thumb" ? THUMB_PROFILE : FINGER_PROFILE;
      let start = curve.getPoint(0).applyMatrix4(matrix);
      let startRadius = spec.radius * profileAt(0, profile) * presence;
      const fingerBox = new THREE.Box3().expandByPoint(start);
      for (let segment = 0; segment < SEGMENTS; segment++) {
        const u = (segment + 1) / SEGMENTS;
        const end = curve.getPoint(u).applyMatrix4(matrix);
        const endRadius = spec.radius * profileAt(u, profile) * presence;
        const i = si * 40 + fi * SEGMENTS + segment;
        uniforms.uStart.value[i].set(start.x, start.y, start.z, startRadius);
        uniforms.uEnd.value[i].set(end.x, end.y, end.z, endRadius);
        box.expandByPoint(start).expandByPoint(end);
        fingerBox.expandByPoint(end);
        start = end; startRadius = endRadius;
      }
      const sphere = fingerBox.getBoundingSphere(new THREE.Sphere());
      uniforms.uFingerBounds.value[si * 5 + fi].set(sphere.center.x, sphere.center.y, sphere.center.z,
        sphere.radius + spec.radius * 1.15 * presence + .10 * presence);
    });
    box.expandByScalar(.19 * presence);
    uniforms.uBounds.value[si].set(box.min.x, box.min.y, box.max.x, box.max.y);
    uniforms.uDepth.value[si].set(box.min.z, box.max.z);
  });
}

/** Half-extents of everything currently drawn, so the view can frame both hands. */
export function contentExtent(uniforms: HandUniforms) {
  let x = 0, y = 0;
  SIDES.forEach((_, si) => {
    if (uniforms.uPresence.value.getComponent(si) <= .008) return;
    const b = uniforms.uBounds.value[si];
    x = Math.max(x, Math.abs(b.x), Math.abs(b.z));
    y = Math.max(y, Math.abs(b.y), Math.abs(b.w));
  });
  return { x, y };
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
  // One soft mass: broad palm, full knuckle line, rounded heel that closes the wrist,
  // thenar and web carrying the thumb, hypothenar along the little-finger edge.
  float palmDistance(vec3 local) {
    float d = ellipsoid(local - vec3(.005, .470, -.020), vec3(.400, .500, .185));
    d = smoothUnion(d, ellipsoid(local - vec3(.005, .800, -.005), vec3(.465, .215, .160)), .26);
    d = smoothUnion(d, ellipsoid(local - vec3(.020, .185, -.015), vec3(.345, .245, .180)), .26);
    d = smoothUnion(d, ellipsoid(local - vec3(-.235, .315, .100), vec3(.268, .380, .232)), .26);
    d = smoothUnion(d, ellipsoid(local - vec3(.325, .420, -.005), vec3(.185, .325, .170)), .26);
    d = smoothUnion(d, ellipsoid(local - vec3(-.315, .560, .045), vec3(.150, .235, .150)), .26);
    return d;
  }
  float handDistance(vec3 p, int hand) {
    float presence = hand == 0 ? uPresence.x : uPresence.y;
    if (presence < .008) return 100.0;
    vec3 local = (uInverse[hand] * vec4(p, 1.0)).xyz;
    float palm = palmDistance(local) * presence;
    // Every finger blends into the PALM only, and the fingers are then combined
    // with a hard min. Two fingers lying against each other therefore keep a
    // crease and their own silhouettes instead of melting into one mass.
    float d = palm;
    for (int f = 0; f < 5; f++) {
      vec4 bound = uFingerBounds[hand * 5 + f];
      if (length(p - bound.xyz) - bound.w > d + .07 * presence) continue;
      int i = hand * 40 + f * 8;
      float finger = capsule(p, uStart[i], uEnd[i]);
      for (int j = 1; j < 8; j++) finger = min(finger, capsule(p, uStart[i + j], uEnd[i + j]));
      d = min(d, smoothUnion(palm, finger, (f == 0 ? .255 : .105) * presence));
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
    for (int i = 0; i < 96; i++) {
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
    // Contact shadow: deepens finger gaps and the palm hollow without hiding them.
    float ao = 1.0;
    for (int i = 1; i <= 4; i++) {
      float h = float(i) * .055;
      ao -= max(h - handDistance(p + n * h, hand), 0.0) * 1.25;
    }
    ao = clamp(ao, .45, 1.0);
    vec3 light = normalize(vec3(-.55, .95, 1.35));
    vec3 halfLight = normalize(light + vec3(0, 0, 1));
    float diffuse = max(dot(n, light), 0.0);
    // Soft skin, not plastic: a wide, low highlight instead of a tight hotspot.
    float highlight = pow(max(dot(n, halfLight), 0.0), 34.0);
    float sheen = pow(max(dot(n, halfLight), 0.0), 6.0);
    float rim = pow(1.0 - max(n.z, 0.0), 3.0);
    vec3 base = mix(vec3(.96, .69, .37), vec3(.32, .12, .72), material);
    vec3 color = base * (.50 + diffuse * .58) * ao;
    color += vec3(1.0, .92, .80) * highlight * .30 + vec3(1.0, .80, .58) * sheen * .05;
    color += vec3(.55, .34, 1.0) * rim * .22;
    gl_FragColor = vec4(color, 1.0);
    #include <colorspace_fragment>
  }
`;
