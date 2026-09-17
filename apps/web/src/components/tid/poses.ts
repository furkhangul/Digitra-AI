import * as THREE from "three";
import rigSpec from "./rig-spec.json";
import surfaceOffsets from "./surface-offsets.json";
import viewRoll from "./view-roll.json";

export const FINGERS = ["thumb", "index", "middle", "ring", "pinky"] as const;
export type Finger = typeof FINGERS[number];
export type Vec3 = [number, number, number];
export type FingerPose = { bend: Vec3; spread: number; oppose: number };
export type HandShape = Record<Finger, FingerPose>;
export type HandPose = { shape: HandShape; rotation: Vec3; position: Vec3; visible: boolean };
/** `contact` records which two anchors the sign actually brings together, so
 * calibration and collision checks judge the intended touch rather than any
 * two surfaces that happen to be nearest. */
export type PairPose = { right: HandPose; left: HandPose; contact?: { right: Anchor; left: Anchor } };
export type Anchor = "wrist" | "palm" | `${Finger}_${0 | 1 | 2 | "tip"}`;
export type Letter = {
  ch: string; tip: string; detail: string; pose: PairPose;
  /** Elevation of the viewing camera in degrees; leaves the authored pose intact. */
  viewElevation?: number;
  /** Positive azimuth moves the viewing camera to the right of the hands. */
  viewAzimuth?: number;
  motion?: "double-tap" | "thumb-tap" | "pinch" | "dot" | "hook" | "snap" | "double-snap" | "lower-snap" | "index-bob" | "j-trace";
  sourceVariant?: string;
};
export const TID_SOURCE = "https://linguistics.ankara.edu.tr/wp-content/uploads/sites/1078/2021/05/Dikyuva_H._Makaroglu_B._and_Arik_E._2015_compressed.pdf#page=92";
export const TID_G_SOURCE = "https://tdk.gov.tr/wp-content/uploads/2012/07/G.pdf";
export const TID_SOFT_G_SOURCE = "https://tdk.gov.tr/wp-content/uploads/2012/07/%C4%9E.pdf";
const rad = THREE.MathUtils.degToRad;
const X = new THREE.Vector3(1, 0, 0);
const Y = new THREE.Vector3(0, 1, 0);
const Z = new THREE.Vector3(0, 0, 1);

function finger(bend: Vec3 = [0, 0, 0], spread = 0, oppose = 0): FingerPose {
  return { bend, spread, oppose };
}
/** In a closed hand the thumb lies along the side of the folded index rather
 * than sticking out, so it is fitted against that finger instead of a fixed
 * point in space. Limits keep it inside a real thumb's range. */
function closedThumb(): FingerPose {
  const f = finger([20, 30, 25], -30, 30);
  const s = { thumb: f, index: finger([88, 100, 55]) } as HandShape;
  const target = localAnchor(s, "index_1").add(new THREE.Vector3(-.06, -.04, .13));
  const loss = () => localAnchor(s, "thumb_tip").distanceToSquared(target);
  for (const step of [16, 8, 4, 2, 1, .5]) for (let pass = 0; pass < 10; pass++) for (let k = 0; k < 5; k++) {
    const get = () => k < 3 ? f.bend[k] : k === 3 ? f.spread : f.oppose;
    const set = (v: number) => { if (k < 3) f.bend[k] = v; else if (k === 3) f.spread = v; else f.oppose = v; };
    const initial = get(); let best = initial, error = loss();
    for (const sign of [-1, 1]) {
      const min = k < 3 ? 0 : k === 3 ? -60 : 0;
      const max = k < 3 ? [55, 70, 75][k] : k === 3 ? 10 : 75;
      set(THREE.MathUtils.clamp(initial + step * sign, min, max));
      const e = loss(); if (e < error) { best = get(); error = e; }
    }
    set(best);
  }
  return f;
}
const CLOSED_THUMB=closedThumb();
function shape(extended: Finger[] = [], spread: Partial<Record<Finger, number>> = {}): HandShape {
  return Object.fromEntries(FINGERS.map((name) => [name, name === "thumb"
    ? (extended.includes(name) ? finger([0,0,0],12,0) : structuredClone(CLOSED_THUMB))
    : finger(extended.includes(name) ? [0, 0, 0] : [88, 100, 55], spread[name] ?? 0)])) as HandShape;
}
function change(base: HandShape, edits: Partial<HandShape>): HandShape { return { ...base, ...edits }; }
const POINT = shape(["index"]);
// Remove the index's natural fan before turning J's pointer towards the viewer.
const J_POINTER = change(POINT, { index: finger([0, 0, 0], -2.75083461) });
const K_FINGERS = shape(["index", "middle"], { index: -2.75083461 });
const V = shape(["index", "middle"], { index: 9, middle: -9 });
const V_SIGN = change(V, { thumb: finger([8, 36.5, 59.625], 64, 75) });
const Y_POINTER = change(POINT, { index: finger([0, 0, 0], -2.75083461) });
/** The two downward legs of A. Only a small spread: the source draws them close
 * and near-parallel, with the other index crossing in front like the letter's bar. */
const FORK = shape(["index", "middle"], { index: 6, middle: -6 });
// Cancel the rig's natural finger fan so H's index and pinky point straight
// down in parallel. Middle and ring stay folded into the palm.
const H_FINGERS = shape(["index", "pinky"], { index: -2.75083461, pinky: 5.27740634 });
const THREE_FINGERS = shape(["index", "middle", "ring"], { index: 5, ring: -5 });
const FIST = shape();
const L = shape(["index", "thumb"]);
// Side-view U: extend the thumb and keep the index straight beyond its base
// knuckle. The hand tilt brings both fingers upright around an open bowl.
const U_SHAPE = change(shape(["index", "thumb"]), {
  index: finger([55, 0, 0], -2.75083461),
  thumb: finger([0, 0, 0], 12, 50),
});
const C = change(shape(), {
  index: finger([20, 35, 30]),
  middle: finger([95, 100, 65]), ring: finger([95, 100, 65]), pinky: finger([95, 100, 65]),
  thumb: finger([25, 0, 0], 15, 65),
});
// G's thumbs form the lower bar in the TDK photograph and the user's chart.
// G/Ğ and the S variation share this opening; C retains its own shape.
const G_C = change(C, { thumb: finger([0, 0, 0], 12, 50) });
const R_POINTER = change(POINT, { index: finger([0, 0, 0], -2.75083461) });
const P_SHAPE = change(shape(["middle"]), {
  // Extend back at the knuckle, keeping both outer joints curled forwards.
  // The fingertip meets the side of middle_1; the left-hand side view puts
  // the rounded bowl to the right of the straight middle-finger stem.
  index: finger([-48, 68, 70], -4.5),
  thumb: finger([0, 70, 75], 10, 45),
});
/** D's bowl is wider than the C curve — the letter has a rounder loop. */
const D_BOWL = change(C, { index: finger([10, 20, 16]), thumb: finger([18, 0, 0], 12, 56) });

/** Forward kinematics in the same rest frame as the Blender-generated rig. */
export function localAnchor(s: HandShape, anchor: Anchor): THREE.Vector3 {
  if (anchor === "wrist") return new THREE.Vector3();
  if (anchor === "palm") return new THREE.Vector3(0, .42, .17);
  const [name, segment] = anchor.split("_") as [Finger, string];
  const spec = rigSpec[name];
  const p = new THREE.Vector3(...spec.base);
  const direction = new THREE.Vector3(...spec.direction);
  const axis = new THREE.Vector3().crossVectors(direction, Z).normalize();
  const f = s[name];
  const q = new THREE.Quaternion().setFromAxisAngle(Z, rad(f.spread))
    .multiply(new THREE.Quaternion().setFromAxisAngle(Y, rad(f.oppose)));
  const count = segment === "tip" ? 3 : Number(segment);
  for (let i = 0; i < count; i++) {
    q.multiply(new THREE.Quaternion().setFromAxisAngle(axis, rad(f.bend[i])));
    p.add(direction.clone().multiplyScalar(spec.lengths[i]).applyQuaternion(q));
  }
  return p;
}

// Solve thumb/index contact once while authoring poses, not in the render loop.
// The fitting targets are geometry constraints; language validation is separate.
function pinch(extended: Finger[]): HandShape {
  const s = change(shape(extended), { index: finger([26, 44, 44]), thumb: finger([4, 2, 2], -42, 12) });
  const names: Finger[] = ["index", "thumb"];
  // Bring the two finger PADS together — the fitted distance is the sum of the
  // two radii, so the surfaces touch instead of the bone tips coinciding — and
  // reward an open ring so the loop stays readable from the front.
  const pads = rigSpec.index.radius + rigSpec.thumb.radius;
  const loss = () => {
    const contact = localAnchor(s, "index_tip").distanceTo(localAnchor(s, "thumb_tip")) - pads;
    const opening = localAnchor(s, "index_1").distanceTo(localAnchor(s, "thumb_2"));
    return contact * contact * 40 - opening * .35 + (s.index.bend[0] ** 2 + s.index.bend[1] ** 2) * 2e-5;
  };
  for (const step of [16, 8, 4, 2, 1, .5]) {
    for (let pass = 0; pass < 12; pass++) {
      for (const name of names) {
        const f = s[name];
        for (let k = 0; k < 5; k++) {
          const get = () => k < 3 ? f.bend[k] : k === 3 ? f.spread : f.oppose;
          const set = (v: number) => { if (k < 3) f.bend[k] = v; else if (k === 3) f.spread = v; else f.oppose = v; };
          const initial = get(); let best = initial; let error = loss();
          for (const delta of [-step, step]) {
            if (name === "index" && k === 4) continue;
            // Anatomical limits: the thumb may swing away from the palm and
            // oppose towards it, but never fold back under its own metacarpal.
            const min = k < 3 ? 0 : name === "index" ? -20 : k === 3 ? -60 : 0;
            const max = k < 3 ? (name === "thumb" ? [55, 70, 75][k] : [85, 110, 80][k])
                              : name === "index" ? 20 : k === 3 ? 10 : 75;
            set(THREE.MathUtils.clamp(initial + delta, min, max));
            const next = loss(); if (next < error) { error = next; best = get(); }
          }
          set(best);
        }
      }
    }
  }
  return s;
}
const PINCH = pinch([]);
// Thumb/middle contact for the snapping left hand in Ç and İ. Fit the finger pads,
// keeping the index out of the contact so the snap remains visible from the front.
function snapShape(): HandShape {
  const s = change(shape(), {
    index: finger([30, 40, 20]), middle: finger([40, 34, 12], 10), thumb: finger([8, 14, 10], -26, 34),
  });
  const names: Finger[] = ["middle", "thumb"];
  // The two pads press together, so the fitted distance is the sum of the radii.
  // The middle finger is held only part-curled: it needs somewhere to flick TO.
  const pads = rigSpec.middle.radius + rigSpec.thumb.radius;
  const loss = () => {
    const press = localAnchor(s, "middle_tip").distanceTo(localAnchor(s, "thumb_tip")) - pads;
    const curl = s.middle.bend[0] + s.middle.bend[1] + s.middle.bend[2];
    return press * press * 60 + Math.max(0, curl - 95) ** 2 * 3e-4;
  };
  for (const step of [16, 8, 4, 2, 1, .5]) for (let pass = 0; pass < 12; pass++) for (const name of names) {
    const f = s[name];
    for (let k = 0; k < 5; k++) {
      if (name === "middle" && k === 4) continue;
      const get = () => k < 3 ? f.bend[k] : k === 3 ? f.spread : f.oppose;
      const set = (v: number) => { if (k < 3) f.bend[k] = v; else if (k === 3) f.spread = v; else f.oppose = v; };
      const initial = get(); let best = initial, error = loss();
      for (const sign of [-1, 1]) {
        // Same anatomical limits as the other fitted shapes: the thumb opposes
        // towards the palm, it never folds back under its own metacarpal.
        const min = k < 3 ? 0 : name === "middle" ? -22 : k === 3 ? -60 : 0;
        const max = k < 3 ? (name === "thumb" ? [55, 70, 75][k] : [90, 105, 70][k])
                          : name === "middle" ? 22 : k === 3 ? 10 : 75;
        set(THREE.MathUtils.clamp(initial + step * sign, min, max));
        const next = loss(); if (next < error) { best = get(); error = next; }
      }
      set(best);
    }
  }
  return s;
}
const SNAP = snapShape();
// Shared thumb/middle action; the caller decides whether the index and wrist
// may follow the strike or must hold an inter-hand contact (Ş).
function snapFingers(s: HandShape, load: number, travel: number) {
  const middle = s.middle, thumb = s.thumb;
  thumb.oppose += 6 * load;
  thumb.spread += 4 * load;
  middle.bend[0] += 5 * load;
  middle.bend = middle.bend.map((v, i) => v + ([96, 106, 60][i] - v) * travel) as Vec3;
  thumb.spread -= 9 * travel;
  thumb.oppose -= 5 * travel;
  thumb.bend[0] += 4 * travel;
}
const O = change(pinch(["middle", "ring", "pinky"]), {
  middle: finger([0,0,0],-2), ring: finger([0,0,0],-7), pinky: finger([0,0,0],-12),
});

function hand(s: HandShape, rotation: Vec3 = [0, 0, 0], position: Vec3 = [0, -.8, 0], visible = true): HandPose {
  return { shape: s, rotation, position, visible };
}
export const REST: PairPose = {
  right: hand(shape(FINGERS.slice()), [0, -12, -12], [-.82, -1.0, 0]),
  left: hand(shape(FINGERS.slice()), [0, 12, 12], [.82, -1.0, 0]),
};
export function orientation(rotation: Vec3) { return new THREE.Quaternion().setFromEuler(new THREE.Euler(...rotation.map(rad) as Vec3, "XYZ")); }
export function anchorOffset(h: HandPose, side: "left" | "right", name: Anchor) {
  const p = localAnchor(h.shape, name);
  if (side === "left") p.x *= -1;
  return p.applyQuaternion(orientation(h.rotation));
}
function solo(s: HandShape, rotation: Vec3 = [0, 0, 0], side: "right" | "left" = "right"): PairPose {
  const visible = hand(s, rotation);
  const hidden = hand(FIST, [0, 0, 0], [side === "right" ? 2.4 : -2.4, -2.8, -.5], false);
  return centre(side === "right" ? { right: visible, left: hidden } : { right: hidden, left: visible });
}
function centre(p: PairPose): PairPose {
  const box = new THREE.Box3();
  for (const side of ["right", "left"] as const) {
    const h = p[side]; if (!h.visible) continue;
    const translation = new THREE.Vector3(...h.position);
    for (const name of ["wrist", ...FINGERS.flatMap(f => [`${f}_0`, `${f}_1`, `${f}_2`, `${f}_tip`])] as Anchor[]) {
      box.expandByPoint(anchorOffset(h, side, name).add(translation));
    }
    box.expandByPoint(new THREE.Vector3(0,.035,0).applyQuaternion(orientation(h.rotation)).add(translation));
  }
  const center = box.getCenter(new THREE.Vector3());
  for (const side of ["right", "left"] as const) p[side].position = new THREE.Vector3(...p[side].position).sub(center).toArray() as Vec3;
  return p;
}
function joined(
  right: HandShape, left: HandShape, rr: Vec3, lr: Vec3,
  ra: Anchor, la: Anchor, gap: Vec3 = [0, 0, .09],
): PairPose {
  const r = hand(right, rr, [-.6, -.7, 0]);
  const l = hand(left, lr, [.6, -.7, -.1]);
  const rp = anchorOffset(r, "right", ra);
  const lp = anchorOffset(l, "left", la);
  r.position = lp.add(new THREE.Vector3(...l.position)).sub(rp).add(new THREE.Vector3(...gap)).toArray() as Vec3;
  return centre({ right: r, left: l, contact: { right: ra, left: la } });
}

// İ's dot is a left-hand snap above the upright right index. The anchor relation
// positions the hands; the intended contact is within the snapping hand itself.
function dottedI(): PairPose {
  const { right, left } = joined(POINT, SNAP, [0, 0, 0], [0, 55, 10],
    "index_tip", "middle_tip", [0, -.63, .05]);
  return { right, left };
}

// Ö keeps its right-hand ring in front while the upper left hand snaps twice.
// Contact is within each hand, so this placement has no inter-hand contact.
function dottedO(): PairPose {
  const { right, left } = joined(PINCH, SNAP, [0, -60, -50], [0, 55, 10],
    "index_tip", "middle_tip", [0, -.6, .2]);
  left.position[1] += .15;
  return { right, left };
}

// Ü keeps the reviewed side-view U below Ö's two-dot snapping gesture.
function dottedU(): PairPose {
  const { right, left } = joined(U_SHAPE, SNAP, [-90, 35, 90], [0, 55, 10],
    "index_tip", "middle_tip", [0, -.75, .2]);
  // Put the two dots over the U opening and bring the snapping hand forwards,
  // so its palm does not appear behind the upright fingers.
  left.position[0] += .22;
  left.position[1] += .65;
  left.position[2] += .60;
  return centre({ right, left });
}

const J_TRACE_START = localAnchor(L, "index_tip").add(new THREE.Vector3(-.255, 0, .495));

// Start level with the very top of the upright right index. The left
// fingertip faces the viewer throughout its open trace towards the thumb tip.
function tracingJ(): PairPose {
  const { right, left } = joined(L, J_POINTER, [0, 0, 0], [90, 0, 0],
    "index_tip", "index_tip", [.255, 0, -.495]);
  return { right, left };
}

function linkedS(): PairPose {
  const pose = joined(G_C, G_C, [133.00307153, -51.71009587, 126.20398975],
    [150.9680068, 61.09544437, -147.62441159], "index_tip", "thumb_tip", [.21619, .097, -.02229]);
  // Lower only the right hand slightly, sliding around the thumb pad so
  // the fingertip contact distance stays fixed and the left hand stays put.
  pose.right.position[1] -= .025;
  pose.right.position[0] += Math.sqrt(.21619 ** 2 + .097 ** 2 - .072 ** 2) - .21619;
  return pose;
}

/** Dikyuva, Makaroğlu & Arık (2015), printed p.91, frequent variants.
 * These are authored 3D reconstructions; expert linguistic review is pending.
 * No ASL fallback and no invented Turkish variants are used.
 */
export const TID_ALPHABET: Letter[] = [
  { ch: "A", tip: "İki parmağı aşağı uzat; diğer elin işaret parmağını önlerinden yatay geçir.", detail: "İki dik parmak A'nın bacakları, yatay parmak orta çubuğudur.", pose: joined(FORK, POINT, [180, 0, 0], [180, 0, -90], "index_1", "index_2", [-.27707, .01795, -.20679]) },
  { ch: "B", tip: "İki elin kıvrılmış baş ve işaret parmaklarını üst üste buluştur.", detail: "Üst ve alt halkayı ayrı ayrı incele.", sourceVariant: "B1", pose: joined(PINCH, PINCH, [0, -60, -50], [0, 60, 20], "index_tip", "index_tip", [.466, -.034, -.198]) },
  { ch: "C", tip: "Baş ve işaret parmağın arasını açık bırakarak C biçimini oluştur.", detail: "Parmak uçları birbirine değmez.", pose: solo(C, [0, -60, -15]) },
  { ch: "Ç", tip: "C biçimini koru; diğer elinle altında bir kez parmak şıklat.", detail: "Baş ve orta parmak önce buluşur; orta parmak hızla avuca iner. Hareketi 0.5× hızda inceleyebilirsin.", sourceVariant: "Ç — şıklatma", motion: "snap", pose: joined(C, SNAP, [0, -55, -10], [0, 55, 10], "wrist", "wrist", [.62, .60, 0]) },
  // Rotate the complete D pair 26° about the world Z axis to stand the left
  // index upright, preserving the bowl and both contacts. D's surface and
  // approach offsets use the same rotation (Euler XYZ is not additive).
  { ch: "D", tip: "Bir işaret parmağını dik tut; diğer elin kavisiyle uçlarını birleştir.", detail: "Dik parmak ve kavis birlikte D biçimini oluşturur.", pose: joined(D_BOWL, POINT, [43.2312665, -54.54625376, 75.09120646], [0, 0, 0], "index_tip", "index_tip", [.13109428, .09620441, .103]) },
  // Turn the right palm over while keeping its fingers horizontal. The left
  // hand turns 90° around Y and its index sits over the three finger roots.
  { ch: "E", tip: "Sağ elin üç parmağını sola doğru yatay uzat; sol işaret parmağını parmakların başladığı yere dik yerleştir.", detail: "Sol işaret parmağı, üç yatay parmağın köklerinde E'nin dik çizgisini oluşturur.", sourceVariant: "E1", pose: joined(THREE_FINGERS, POINT, [180, 0, 90], [0, 90, 0], "index_0", "index_tip", [.14, -.066, -.38]) },
  { ch: "F", tip: "Baş ve işaret parmağını aç; diğer işaret parmağını yatay yerleştir.", detail: "İki elin temas noktasına bak.", sourceVariant: "F1", pose: joined(POINT, L, [180, 0, 90], [0, 180, 180], "index_tip", "index_1") },
  // TDK/MEB p.111 and Harfler-EDT3, mirrored: left C outside, right index inside.
  // Y turns are +/-115°, followed by world-Z rolls of +30° (right), -15° (left).
  // These XYZ Euler angles preserve that order. Soft G gently moves the upper index.
  { ch: "G", tip: "İki eli C biçiminde tut; sağ işaret parmağını sol elin açık baş ve işaret parmakları arasına getir.", detail: "Başparmaklar üst üste temas eder; iki el de göğüs hizasındadır.", sourceVariant: "TDK/MEB 2012 — C el", pose: joined(G_C, G_C, [133.00307153, -51.71009587, 126.20398975], [150.9680068, 61.09544437, -147.62441159], "thumb_tip", "thumb_tip", [.02, -.31, 0]) },
  { ch: "Ğ", tip: "G biçimini koru; üstteki işaret parmağını çok hafif yukarı-aşağı hareket ettir.", detail: "Sol işaret parmağı küçük ve hızlı hareketlerle yukarı-aşağı oynar; el gövdeleri ve diğer parmaklar sabit kalır.", sourceVariant: "TDK/MEB 2012 — C el", motion: "index-bob", pose: joined(G_C, G_C, [133.00307153, -51.71009587, 126.20398975], [150.9680068, 61.09544437, -147.62441159], "thumb_tip", "thumb_tip", [.02, -.31, 0]) },
  // H uses A's mirrored hand arrangement, with index and pinky pointing down.
  { ch: "H", tip: "İşaret ve serçe parmağını düz aşağı uzat; diğer elin işaret parmağını önlerinden yatay geçir.", detail: "Orta ve yüzük parmaklarını avuca kapat; aşağı bakan işaret ve serçe parmağın birbirine paralel dursun.", pose: joined(H_FINGERS, POINT, [180, 0, 0], [180, 0, -90], "index_1", "index_2", [-.34207, .01795, -.20679]) },
  { ch: "I", tip: "İşaret parmağını dik tut; diğer parmakları kapat.", detail: "Başparmak kapalı parmakların yanında durur.", pose: solo(POINT) },
  { ch: "İ", tip: "Sağ işaret parmağını dik tut; sol elinle hemen üzerinde bir kez şıklat.", detail: "Alttaki sağ el sabit kalır; sol baş ve orta parmak buluşur, orta parmak hızla avuca iner.", motion: "snap", pose: dottedI() },
  { ch: "J", tip: "Sağ elini L biçiminde tut; sol işaret ucunu sağ işaret parmağının en üst ucundan aşağı, başparmağın ucuna doğru ilerlet.", detail: "Sol işaret ucu sana bakar; L'nin iç kenarını takip eder ve aynı yoldan başlangıca döner.", motion: "j-trace", pose: tracingJ() },
  { ch: "K", tip: "Sağ avucunu sana çevir; işaret ve orta parmağını bitişik, dik tut. Sol elindeki aynı iki parmağı bitiştirip sağ işaretin kök eklemine değdir.", detail: "İki elde de başparmak, yüzük ve serçe avuca kapalı kalsın.", sourceVariant: "K2", pose: joined(K_FINGERS, K_FINGERS, [0, 0, 0], [0, 0, -75], "index_0", "index_tip", [.28, 0, -.08]) },
  // Widen only L's thumb to about 82°; J retains its reviewed tracing opening.
  { ch: "L", tip: "İşaret parmağını dik, başparmağını yana açık tut.", detail: "Diğer üç parmak avuca kapanır.", pose: solo(change(L, { thumb: finger([0, 0, 0], 32) })) },
  { ch: "M", tip: "İki elin işaret ve orta parmaklarını aşağı aç; içteki uçları buluştur.", detail: "Dört parmakla oluşan M biçimini takip et.", pose: joined(V, V, [180, 0, 0], [180, 0, 0], "index_tip", "index_tip", [.18646, 0, 0]) },
  { ch: "N", tip: "Sağ elin işaret ve orta parmağını, sol elin yalnızca işaret parmağını aşağı uzat.", detail: "İki parmak sağda, tek parmak solda kalır.", pose: joined(V, POINT, [180, 0, 0], [180, 0, -26], "index_tip", "index_tip", [.18085, -.02381, .01413]) },
  { ch: "O", tip: "Baş ve işaret parmağının uçlarını birleştir; diğer üç parmağı açık tut.", detail: "Parmak uçları arasındaki halka görünür kalmalı.", pose: solo(O, [0, -60, 0]) },
  { ch: "Ö", tip: "Sağ elini önde sabit tut; sol elinle yukarıda şıklat, biraz sağa kayıp bir kez daha şıklat.", detail: "İki şıklatma, Ö'nün iki noktası gibi yan yana iki ayrı konumda yapılır.", motion: "double-snap", pose: dottedO() },
  { ch: "P", tip: "Sol orta parmağını düz aşağı uzat; işaret parmağını kökünden geriye alıp ucunu orta parmağın orta eklemine değdir.", detail: "İşaretin orta ve uç eklemleri kıvrık kalır; yandan bakınca sağdaki kavis ve düz orta parmak P biçimini oluşturur.", viewElevation: 10, pose: solo(P_SHAPE, [180, -90, 0], "left") },
  // Keep P's left hand and add the diagonal leg beneath its index fingertip.
  { ch: "R", tip: "Sol elini P biçiminde tut; sağ işaret parmağını sol işaretin ucuna getir.", detail: "Sağ işaret parmağı birleşimden sağ aşağıya uzanarak R'nin çapraz çizgisini tamamlar; diğer sağ parmaklar kapalı kalır.", viewElevation: 10, viewAzimuth: 10, pose: joined(R_POINTER, P_SHAPE, [0, 0, 40], [180, -90, 0], "index_tip", "index_tip", [.03, -.22, 0]) },
  // S keeps G's reviewed C hands, joining the right index to the left thumb.
  { ch: "S", tip: "Ellerini G'deki gibi C biçiminde tut; sol başparmak ucunu sağ işaret parmağı ucuyla birleştir.", detail: "İki parmak uç uca değer; kavislerin arasındaki açıklık görünür kalır.", pose: linkedS() },
  { ch: "Ş", tip: "S biçimini koru; sağ elinin alt kısmında baş ve orta parmağınla bir kez şıklat.", detail: "Sol el ve üstteki işaret–başparmak teması sabit kalır; sağ orta parmak hızla avuca iner, sonra el S biçimine döner.", motion: "lower-snap", pose: linkedS() },
  { ch: "T", tip: "Dik işaret parmağının ucuna diğer işaret parmağını yatay yerleştir.", detail: "İki parmak T biçiminde birleşir.", pose: joined(POINT, POINT, [0, 0, 90], [0, 0, 0], "index_1", "index_tip", [0, .06, .07]) },
  { ch: "U", tip: "Başparmak ve işaret parmağını açık tut; elini yandan U biçimi görünecek şekilde çevir.", detail: "İki açık parmak yukarı bakar; orta, yüzük ve serçe parmak avuca kapalı kalır.", pose: solo(U_SHAPE, [-90, 35, 90]) },
  { ch: "Ü", tip: "Sağ elinle U biçimini sabit tut; sol elinle üstte şıklat, biraz sağa kayıp bir kez daha şıklat.", detail: "Sağ başparmak ve işaret açık kalır; üstteki iki hızlı şıklatma Ü'nün iki noktasını oluşturur.", motion: "double-snap", pose: dottedU() },
  { ch: "V", tip: "İşaret ve orta parmağını birbirinden açarak dik tut.", detail: "Yüzük ve serçeyi avuca kapat; başparmak ucunu yüzüğün yanından orta parmağın köküne getir.", pose: solo(V_SIGN) },
  { ch: "Y", tip: "Sağ elini V biçiminde tut; sol işaret parmağını iki açık parmağın başladığı yere alttan, hafif soldan getir.", detail: "İki elin sırtı karşıya bakar. Sağ el hafif sağdan, sol işaret hafif soldan gelerek Y biçiminde birleşir; diğer sol parmaklar kapalı kalır.", pose: joined(V_SIGN, Y_POINTER, [0, 180, -9], [0, 180, 9], "index_0", "index_tip", [.12153, -.01619, -1.1]) },
  { ch: "Z", tip: "İki elin açık işaret ve orta parmaklarını yatay olarak karşılaştır.", detail: "Sağ el ekranın sağından sola, sol el ekranın solundan sağa uzanır.", pose: joined(V, V, [180, 0, 90], [180, 0, -90], "middle_tip", "index_tip", [.14524, .04923, .09852]) },
];

// Roll each hand about its own long axis so the palm is turned towards the other
// hand instead of flat to the camera. This is a viewing correction, not a change
// to the sign: rolling about the finger axis leaves every finger direction and
// every joint angle exactly where it was, it only decides how much palm the
// learner sees. Without it a sign reads as your own hand seen from above rather
// than as someone facing you. Values in degrees, per hand.
for (const letter of TID_ALPHABET) {
  const roll = (viewRoll as Record<string, { right?: number; left?: number }>)[letter.ch];
  if (!roll) continue;
  for (const side of ["right", "left"] as const) {
    if (roll[side] === undefined) continue;
    letter.pose[side].rotation[1] += roll[side]!;
  }
}

// Generated from posed surface intersections, not guessed bone distances.
for (const letter of TID_ALPHABET) {
  const offset = (surfaceOffsets as Record<string, number[]>)[letter.ch];
  if (!offset) continue;
  letter.pose.right.position = letter.pose.right.position.map((v,i)=>v+offset[i]/2) as Vec3;
  letter.pose.left.position = letter.pose.left.position.map((v,i)=>v-offset[i]/2) as Vec3;
}

/** Length of one motion loop, in seconds. */
export const LOOP = 3.4;

/** Runtime motion curves begin AND finish on the authored pose. */
export function samplePose(letter: Letter, elapsed: number): PairPose {
  const p = structuredClone(letter.pose);
  if (!letter.motion) return p;
  const t = (elapsed % LOOP) / LOOP;
  const pulse = (start: number, end: number) => t > start && t < end ? Math.sin(Math.PI * (t-start)/(end-start)) ** 2 : 0;
  if (letter.motion === "thumb-tap") {
    // Ğ2 marks a repeated up/down move on the raised thumb of the upper hand.
    const thumb = p.right.shape.thumb;
    const beat = pulse(.15, .38) + pulse(.44, .67);
    thumb.bend[0] += 26 * beat;
    thumb.bend[1] += 14 * beat;
  } else if (letter.motion === "index-bob") {
    // User-reviewed Ğ: the upper (left) index rises and falls four times per
    // 3.4-second loop (0.85 s per cycle), retaining the same small PIP bend,
    // keeping both hand bodies and the thumb contact completely still.
    // The two hands share G_C; detach the left shape before animating one finger.
    p.left.shape = structuredClone(p.left.shape);
    p.left.shape.index.bend[1] += 5 * Math.sin(t * Math.PI * 8);
  } else if (letter.motion === "j-trace") {
    // An open stroke from the index tip down along the inside edge of the thumb,
    // then a return over exactly the same path, never a circle in the opening.
    const smooth = (value: number) => {
      const u = THREE.MathUtils.clamp(value, 0, 1);
      return u * u * (3 - 2 * u);
    };
    const u = smooth((t - .08) / .44) * (1 - smooth((t - .68) / .26));
    // Extend the stroke up the full index; retain the lower bend and thumb end.
    const a = 3 * (1 - u) ** 2 * u, b = 3 * (1 - u) * u ** 2, c = u ** 3;
    p.left.position[0] += a * (-.6 - J_TRACE_START.x) + b * (-.68 - J_TRACE_START.x) + c * (-1.1 - J_TRACE_START.x);
    p.left.position[1] += a * (1.01 - J_TRACE_START.y) + b * (.665 - J_TRACE_START.y) + c * (.925 - J_TRACE_START.y);
  } else if (letter.motion === "double-tap") {
    p.right.shape.index.bend[1] -= 30 * (pulse(.15, .38) + pulse(.44, .67));
  } else if (letter.motion === "pinch") {
    p.right.position[0] += .17 * (pulse(.15, .4) + pulse(.48, .73));
  } else if (letter.motion === "dot") {
    p.right.position[1] -= .13 * pulse(.2, .6);
  } else if (letter.motion === "lower-snap") {
    // Ş starts and ends at S. Detach the right shape: linkedS shares G_C
    // between both hands, while the left hand must remain completely still.
    p.right.shape = structuredClone(p.right.shape);
    const seconds = t * LOOP;
    const ease = (x: number) => { const v = THREE.MathUtils.clamp(x, 0, 1); return v*v*v*(v*(v*6-15)+10); };
    const span = (a: number, b: number) => ease((seconds - a) / (b - a));
    const ready = span(.20, .70) * (1 - span(2.35, 2.80));
    // Prepare only the two lower fingers; the upper index anchors the S.
    for (const name of ["thumb", "middle"] as const) {
      const f = p.right.shape[name], target = SNAP[name];
      f.bend = f.bend.map((v, i) => THREE.MathUtils.lerp(v, target.bend[i], ready)) as Vec3;
      f.spread = THREE.MathUtils.lerp(f.spread, target.spread, ready);
      f.oppose = THREE.MathUtils.lerp(f.oppose, target.oppose, ready);
    }
    // Use Ç/İ's single snap timing and finger strike, without moving the
    // index or wrist away from the other hand's thumb.
    const load = span(.80, 1.12) * (1 - span(1.12, 1.16));
    const travel = span(1.16, 1.28) * (1 - span(1.75, 2.35));
    snapFingers(p.right.shape, load, travel);
  } else if (letter.motion === "snap" || letter.motion === "double-snap") {
    // A real snap, timed in seconds rather than as a symmetric ease: the pads
    // load against each other, the middle finger slips off the thumb and is
    // driven into the palm heel in about a tenth of a second, the strike recoils
    // the hand, and only then does everything drift back for the next one.
    const fastPair = letter.motion === "double-snap";
    const seconds = t * LOOP;
    // Two quick strokes 0.65 seconds apart, with a complete regrip between them.
    const s = fastPair ? seconds - (seconds < 1.15 ? .5 : 1.15) : seconds;
    const [loadStart, loadEnd, fireStart, fireEnd, backStart, backEnd] = fastPair
      ? [0, .15, .19, .27, .37, .59]
      : [.80, 1.12, 1.16, 1.28, 1.75, 2.35];
    const ease = (x: number) => { const v = THREE.MathUtils.clamp(x, 0, 1); return v*v*v*(v*(v*6-15)+10); };
    const span = (a: number, b: number) => ease((s - a) / (b - a));
    const load = span(loadStart, loadEnd) * (1 - span(loadEnd, fireStart));
    const fire = span(fireStart, fireEnd);                   // 0.12 s, or 0.08 s for Ö
    const back = span(backStart, backEnd);
    const travel = fire * (1 - back);
    // Impact ring-down: a decaying wobble, not a second animation.
    const since = s - fireEnd;
    const ring = since > 0 ? Math.exp(-since * 9) * Math.sin(since * 46) * (fastPair ? 1 - back : 1) : 0;
    const index = p.left.shape.index;
    snapFingers(p.left.shape, load, travel);
    index.bend[0] += 7 * travel;                              // the index rides along
    // The strike shakes the whole hand, then settles.
    p.left.rotation[2] -= 5 * travel + 4.5 * ring;
    p.left.rotation[0] += 3 * ring;
    p.left.position[1] -= .035 * travel + .028 * ring;
    if (fastPair) {
      // Move between the two dots during regrip, hold still for the second
      // snap, then return after both strokes so the loop closes smoothly.
      p.left.position[0] += .22 * ease((seconds - .92) / .25)
        * (1 - ease((seconds - 2) / .4));
    }
  }
  return p;
}

export { rigSpec, X, Y, Z };

