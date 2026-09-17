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
  motion?: "double-tap" | "thumb-tap" | "pinch" | "dot" | "hook" | "snap" | "side-sway";
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
const TWO = shape(["index", "middle"]);
const V = shape(["index", "middle"], { index: 9, middle: -9 });
/** The two downward legs of A. Only a small spread: the source draws them close
 * and near-parallel, with the other index crossing in front like the letter's bar. */
const FORK = shape(["index", "middle"], { index: 6, middle: -6 });
const THREE_FINGERS = shape(["index", "middle", "ring"], { index: 5, ring: -5 });
const FIST = shape();
const L = shape(["index", "thumb"]);
const C = change(shape(), {
  index: finger([20, 35, 30]),
  middle: finger([95, 100, 65]), ring: finger([95, 100, 65]), pinky: finger([95, 100, 65]),
  thumb: finger([25, 0, 0], 15, 65),
});
const HOOK = change(POINT, { index: finger([8, 95, 40]) });
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
// Thumb/middle contact for the lower hand in Ç. Fit the actual finger pads,
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
function solo(s: HandShape, rotation: Vec3 = [0, 0, 0]): PairPose {
  const p = { right: hand(s, rotation), left: hand(FIST, [0, 0, 0], [2.4, -2.8, -.5], false) };
  return centre(p);
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

/** Dikyuva, Makaroğlu & Arık (2015), printed p.91, frequent variants.
 * These are authored 3D reconstructions; expert linguistic review is pending.
 * No ASL fallback and no invented Turkish variants are used.
 */
export const TID_ALPHABET: Letter[] = [
  { ch: "A", tip: "İki parmağı aşağı uzat; diğer elin işaret parmağını önlerinden yatay geçir.", detail: "İki dik parmak A'nın bacakları, yatay parmak orta çubuğudur.", pose: joined(FORK, POINT, [180, 0, 0], [180, 0, -90], "index_1", "index_2", [-.27707, .01795, -.20679]) },
  { ch: "B", tip: "İki elin kıvrılmış baş ve işaret parmaklarını üst üste buluştur.", detail: "Üst ve alt halkayı ayrı ayrı incele.", sourceVariant: "B1", pose: joined(PINCH, PINCH, [0, -60, -50], [0, 60, 20], "index_tip", "index_tip", [.466, -.034, -.198]) },
  { ch: "C", tip: "Baş ve işaret parmağın arasını açık bırakarak C biçimini oluştur.", detail: "Parmak uçları birbirine değmez.", pose: solo(C, [0, -60, -15]) },
  { ch: "Ç", tip: "C biçimini koru; diğer elinle altında bir kez parmak şıklat.", detail: "Baş ve orta parmak önce buluşur; orta parmak hızla avuca iner. Hareketi 0.5× hızda inceleyebilirsin.", sourceVariant: "Ç — şıklatma", motion: "snap", pose: joined(C, SNAP, [0, -55, -10], [0, 55, 10], "wrist", "wrist", [.62, .60, 0]) },
  { ch: "D", tip: "Bir işaret parmağını dik tut; diğer elin kavisiyle uçlarını birleştir.", detail: "Dik parmak ve kavis birlikte D biçimini oluşturur.", pose: joined(D_BOWL, POINT, [0, -65, 26], [0, 0, -26], "index_tip", "index_tip", [.16, .029, .103]) },
  { ch: "E", tip: "Sağ elin üç parmağını sola doğru yatay uzat; sol işaret parmağını köklerine dik yerleştir.", detail: "Sağ el ekranın sağında kalır; üç yatay parmak sola bakar ve sol işaret parmağıyla birleşir.", sourceVariant: "E1", pose: joined(THREE_FINGERS, POINT, [0, 0, 90], [0, 0, 0], "index_0", "index_tip", [.10790, .18233, -.29028]) },
  { ch: "F", tip: "Baş ve işaret parmağını aç; diğer işaret parmağını yatay yerleştir.", detail: "İki elin temas noktasına bak.", sourceVariant: "F1", pose: joined(POINT, L, [0, 0, 90], [0, 0, 180], "index_tip", "index_1") },
  { ch: "G", tip: "İki eli C biçiminde tut; sağ işaret parmağını sol elin açık baş ve işaret parmakları arasına getir.", detail: "Başparmaklar üst üste temas eder; iki el de göğüs hizasındadır.", sourceVariant: "TDK/MEB 2012 — C el", pose: joined(C, C, [-30, -25, 0], [0, 55, 0], "thumb_tip", "thumb_tip", [.23962, 0, .08722]) },
  { ch: "Ğ", tip: "G'deki iki C el biçimini koru ve iki eli birlikte sağa-sola salla.", detail: "Sağ işaret parmağı sol C açıklığında, başparmaklar temas hâlindedir; hareket çene altında yapılır.", sourceVariant: "TDK/MEB 2012 — C el", motion: "side-sway", pose: joined(C, C, [-30, -25, 0], [0, 55, 0], "thumb_tip", "thumb_tip", [.23962, 0, .08722]) },
  { ch: "H", tip: "İki parmağı aşağı uzat; diğer işaret parmağını yatay olarak üzerlerine getir.", detail: "İki dik parmak ve yatay temas birlikte görülmeli.", pose: joined(POINT, TWO, [0, 0, 66], [0, 0, 180], "index_2", "index_1", [.387, -.234, .229]) },
  { ch: "I", tip: "İşaret parmağını dik tut; diğer parmakları kapat.", detail: "Başparmak kapalı parmakların yanında durur.", pose: solo(POINT) },
  { ch: "İ", tip: "Dik işaret parmağının üstüne diğer elin parmak ucunu getir.", detail: "Üstteki el nokta konumundadır.", pose: joined(PINCH, POINT, [0, -30, 180], [0, 0, 0], "index_tip", "index_tip", [.006, .283, .015]) },
  { ch: "J", tip: "Bir işaret parmağını dik tut; diğer elin kıvrık parmağını ucuna getir.", detail: "İki elin üstte kurduğu kıvrımı incele.", pose: joined(HOOK, POINT, [0, -60, 165], [0, 0, 0], "index_tip", "index_tip", [.06, .06, .07]) },
  { ch: "K", tip: "İki bitişik parmağı dik tut; diğer elin iki açık parmağını yanına getir.", detail: "Dik çizgi ile yana açılan iki parmağı karşılaştır.", sourceVariant: "K2", pose: joined(V, TWO, [0, 0, 90], [0, 0, 0], "index_tip", "middle_1", [0, 0, .1]) },
  { ch: "L", tip: "İşaret parmağını dik, başparmağını yana açık tut.", detail: "Diğer üç parmak avuca kapanır.", pose: solo(L) },
  { ch: "M", tip: "İki elin işaret ve orta parmaklarını aşağı aç; içteki uçları buluştur.", detail: "Dört parmakla oluşan M biçimini takip et.", pose: joined(V, V, [180, 0, 0], [180, 0, 0], "index_tip", "index_tip", [.18646, 0, 0]) },
  { ch: "N", tip: "Sağ işaret parmağını ve sol elin iki parmağını aşağı uzat.", detail: "Sağ el ekranın sağında, sol V eli ekranın solunda kalır.", pose: joined(POINT, V, [180, 0, 26], [180, 0, 0], "index_tip", "index_tip", [.18085, .02381, -.01413]) },
  { ch: "O", tip: "Baş ve işaret parmağının uçlarını birleştir; diğer üç parmağı açık tut.", detail: "Parmak uçları arasındaki halka görünür kalmalı.", pose: solo(O, [0, -60, 0]) },
  { ch: "Ö", tip: "İki elin kıvrık baş ve işaret parmaklarını karşılıklı yaklaştır.", detail: "Yaklaşma hareketini yavaşlatarak inceleyebilirsin.", motion: "pinch", pose: joined(PINCH, PINCH, [0, -60, -50], [0, 60, 20], "index_tip", "index_tip", [.30, .62, -.198]) },
  { ch: "P", tip: "İşaret parmağını aşağı uzat; diğer parmakları kıvır.", detail: "Elin aşağı yönelmesine dikkat et.", pose: solo(POINT, [0, -20, 165]) },
  { ch: "R", tip: "İki işaret parmağını birbirine kıvrım oluşturacak şekilde yaklaştır.", detail: "Parmakların oluşturduğu kıvrımı ve temas noktasını takip et.", pose: joined(HOOK, HOOK, [0, 0, 138], [0, 0, 18], "index_tip", "index_tip", [.11, .516, .331]) },
  { ch: "S", tip: "İki elin C biçimlerini ters yönlerde birbirine bağla.", detail: "Üst ve alt kavisler birlikte görünür kalmalı.", pose: joined(C, C, [0, 0, -10], [0, 0, 170], "index_tip", "thumb_tip", [.114, -.193, .225]) },
  { ch: "Ş", tip: "Üstteki C biçimini, alttaki elin işaret parmağıyla birleştir.", detail: "Alttaki elin konumu S harfinden farklıdır.", pose: joined(C, L, [0, -30, -10], [0, 0, 20], "thumb_tip", "index_tip", [.032, .217, -.081]) },
  { ch: "T", tip: "Dik işaret parmağının ucuna diğer işaret parmağını yatay yerleştir.", detail: "İki parmak T biçiminde birleşir.", pose: joined(POINT, POINT, [0, 0, 90], [0, 0, 0], "index_1", "index_tip", [0, .06, .07]) },
  { ch: "U", tip: "İşaret ve serçe parmağını açık tut; ortadaki iki parmağı kapat.", detail: "Açık iki parmak U biçimini oluşturur.", pose: solo(shape(["index", "pinky"]), [0, -15, 0]) },
  { ch: "Ü", tip: "İki elin kıvrılmış parmaklarını üst ve alt konumda karşılaştır.", detail: "Kaynakta gösterilen iki el biçimini birlikte incele.", pose: joined(C, shape(["index", "pinky"]), [0, 0, 85], [0, 0, 0], "thumb_tip", "index_tip", [.045, .212, -.088]) },
  { ch: "V", tip: "İşaret ve orta parmağını birbirinden açarak dik tut.", detail: "Diğer parmaklar avuca kapanır.", pose: solo(V) },
  { ch: "Y", tip: "V biçiminin ortasına diğer elin işaret parmağını getir.", detail: "İkinci işaret parmağı ortadaki çizgiyi tamamlar.", pose: joined(POINT, V, [0, -24, 0], [0, 0, -24], "index_tip", "middle_1", [.08, -.364, .437]) },
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
  } else if (letter.motion === "side-sway") {
    // TDK specifies that both C hands sway together left/right under the chin.
    // The source does not prescribe amplitude or tempo; this conservative loop
    // is a visualisation choice and keeps the authored contact rigid.
    const sway = .11 * Math.sin(t * Math.PI * 2);
    p.right.position[0] += sway;
    p.left.position[0] += sway;
  } else if (letter.motion === "double-tap") {
    p.right.shape.index.bend[1] -= 30 * (pulse(.15, .38) + pulse(.44, .67));
  } else if (letter.motion === "pinch") {
    p.right.position[0] += .17 * (pulse(.15, .4) + pulse(.48, .73));
  } else if (letter.motion === "dot") {
    p.right.position[1] -= .13 * pulse(.2, .6);
  } else if (letter.motion === "snap") {
    // A real snap, timed in seconds rather than as a symmetric ease: the pads
    // load against each other, the middle finger slips off the thumb and is
    // driven into the palm heel in about a tenth of a second, the strike recoils
    // the hand, and only then does everything drift back for the next one.
    const s = t * LOOP;
    const ease = (x: number) => { const v = THREE.MathUtils.clamp(x, 0, 1); return v*v*v*(v*(v*6-15)+10); };
    const span = (a: number, b: number) => ease((s - a) / (b - a));
    const load = span(.80, 1.12) * (1 - span(1.12, 1.16));   // press, then let go
    const fire = span(1.16, 1.28);                            // the flick, ~0.12 s
    const back = span(1.75, 2.35);                            // slow recovery
    const travel = fire * (1 - back);
    // Impact ring-down: a decaying wobble, not a second animation.
    const since = s - 1.28;
    const ring = since > 0 ? Math.exp(-since * 9) * Math.sin(since * 46) : 0;
    const middle = p.left.shape.middle;
    const index = p.left.shape.index;
    const thumb = p.left.shape.thumb;
    // Loading: the pads squeeze together and the thumb rides further across.
    thumb.oppose += 6 * load;
    thumb.spread += 4 * load;
    middle.bend[0] += 5 * load;
    // Release: the finger clears the thumb and slaps down onto the palm.
    middle.bend = middle.bend.map((v, i) => v + ([96, 106, 60][i] - v) * travel) as Vec3;
    index.bend[0] += 7 * travel;                              // the index rides along
    // The thumb barely moves in a real snap — it is the middle finger that
    // travels. Just enough recoil to show the pads parting.
    thumb.spread -= 9 * travel;
    thumb.oppose -= 5 * travel;
    thumb.bend[0] += 4 * travel;
    // The strike shakes the whole hand, then settles.
    p.left.rotation[2] -= 5 * travel + 4.5 * ring;
    p.left.rotation[0] += 3 * ring;
    p.left.position[1] -= .035 * travel + .028 * ring;
  }
  return p;
}

export { rigSpec, X, Y, Z };

