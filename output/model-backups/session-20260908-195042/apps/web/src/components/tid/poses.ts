import * as THREE from "three";
import rigSpec from "./rig-spec.json";
import surfaceOffsets from "./surface-offsets.json";

export const FINGERS = ["thumb", "index", "middle", "ring", "pinky"] as const;
export type Finger = typeof FINGERS[number];
export type Vec3 = [number, number, number];
export type FingerPose = { bend: Vec3; spread: number; oppose: number };
export type HandShape = Record<Finger, FingerPose>;
export type HandPose = { shape: HandShape; rotation: Vec3; position: Vec3; visible: boolean };
export type PairPose = { right: HandPose; left: HandPose };
export type Anchor = "wrist" | "palm" | `${Finger}_${0 | 1 | 2 | "tip"}`;
export type Letter = {
  ch: string; tip: string; detail: string; pose: PairPose;
  motion?: "double-tap" | "pinch" | "dot" | "hook" | "snap";
  sourceVariant?: string;
};
export const TID_SOURCE = "https://linguistics.ankara.edu.tr/wp-content/uploads/sites/1078/2021/05/Dikyuva_H._Makaroglu_B._and_Arik_E._2015_compressed.pdf#page=92";
const rad = THREE.MathUtils.degToRad;
const X = new THREE.Vector3(1, 0, 0);
const Y = new THREE.Vector3(0, 1, 0);
const Z = new THREE.Vector3(0, 0, 1);

function finger(bend: Vec3 = [0, 0, 0], spread = 0, oppose = 0): FingerPose {
  return { bend, spread, oppose };
}
function closedThumb(): FingerPose {
  const f=finger([25,35,20],-50,20);
  const s={thumb:f} as HandShape;
  const target=new THREE.Vector3(.05,.61,.40);
  const loss=()=>localAnchor(s,"thumb_tip").distanceToSquared(target);
  for (const step of [16,8,4,2,1,.5]) for(let pass=0;pass<10;pass++) for(let k=0;k<5;k++) {
    const get=()=>k<3?f.bend[k]:k===3?f.spread:f.oppose;
    const set=(v:number)=>{if(k<3)f.bend[k]=v;else if(k===3)f.spread=v;else f.oppose=v;};
    const initial=get();let best=initial,error=loss();
    for(const sign of [-1,1]) {
      set(THREE.MathUtils.clamp(initial+step*sign,k<3?0:k===3?-65:0,k<3?65:k===3?-30:35));
      const e=loss();if(e<error){best=get();error=e;}
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
const FORK = shape(["index", "middle"], { index: 18, middle: -18 });
const THREE_FINGERS = shape(["index", "middle", "ring"], { index: 5, ring: -5 });
const FIST = shape();
const L = shape(["index", "thumb"]);
const C = change(shape(), {
  index: finger([20, 35, 30]),
  middle: finger([95, 100, 65]), ring: finger([95, 100, 65]), pinky: finger([95, 100, 65]),
  thumb: finger([25, 0, 0], 15, 65),
});
const HOOK = change(POINT, { index: finger([8, 95, 40]) });

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
  const s = change(shape(extended), { index: finger([32, 66, 35]), thumb: finger([12, 25, 18], -30, 45) });
  const names: Finger[] = ["index", "thumb"];
  const target = new THREE.Vector3(-.38, .74, .40);
  const loss = () => names.reduce((sum, name) => {
    const contact = target.clone().add(new THREE.Vector3(0, name === "index" ? .035 : -.035, 0));
    return sum + localAnchor(s, `${name}_tip`).distanceToSquared(contact);
  }, 0);
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
            const min = k < 3 ? 0 : name === "index" ? -20 : -85;
            const max = k < 3 ? [85,110,80][k] : name === "index" ? 20 : 85;
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
  const s = change(shape(), { index: finger([25, 45, 30]), middle: finger([42, 58, 25], 12), thumb: finger([10, 25, 25], -45, 25) });
  const target = new THREE.Vector3(-.11, .83, .46);
  const names: Finger[] = ["middle", "thumb"];
  const loss = () => names.reduce((sum, name) => {
    const pad = target.clone().add(new THREE.Vector3(0, name === "middle" ? .115 : -.115, 0));
    return sum + localAnchor(s, `${name}_tip`).distanceToSquared(pad);
  }, 0);
  for (const step of [16, 8, 4, 2, 1, .5]) for (let pass = 0; pass < 12; pass++) for (const name of names) {
    const f = s[name];
    for (let k = 0; k < 5; k++) {
      if (name === "middle" && k === 4) continue;
      const get = () => k < 3 ? f.bend[k] : k === 3 ? f.spread : f.oppose;
      const set = (v: number) => { if (k < 3) f.bend[k] = v; else if (k === 3) f.spread = v; else f.oppose = v; };
      const initial = get(); let best = initial, error = loss();
      for (const sign of [-1, 1]) {
        const limit = name === "middle" ? 22 : 65;
        set(THREE.MathUtils.clamp(initial + step * sign, k < 3 ? 0 : -limit, k < 3 ? [90, 105, 70][k] : limit));
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
  return centre({ right: r, left: l });
}

/** Dikyuva, Makaroğlu & Arık (2015), printed p.91, frequent variants.
 * These are authored 3D reconstructions; expert linguistic review is pending.
 * No ASL fallback and no invented Turkish variants are used.
 */
export const TID_ALPHABET: Letter[] = [
  { ch: "A", tip: "İki parmağı aşağı uzat; diğer elin işaret parmağını aralarına yatay yerleştir.", detail: "İki elin oluşturduğu A biçimini takip et.", pose: joined(POINT, FORK, [0, 0, -90], [0, 0, 180], "index_2", "middle_1", [0, -.24, .02]) },
  { ch: "B", tip: "İki elin kıvrılmış baş ve işaret parmaklarını üst üste buluştur.", detail: "Üst ve alt halkayı ayrı ayrı incele.", sourceVariant: "B1", pose: joined(PINCH, PINCH, [0, 65, 175], [0, -65, 5], "index_tip", "index_tip", [0, .20, 0]) },
  { ch: "C", tip: "Baş ve işaret parmağın arasını açık bırakarak C biçimini oluştur.", detail: "Parmak uçları birbirine değmez.", pose: solo(C, [0, 65, -15]) },
  { ch: "Ç", tip: "C biçimini koru; diğer elinle altında bir kez parmak şıklat.", detail: "Baş ve orta parmak önce buluşur; orta parmak hızla avuca iner. Hareketi 0.5× hızda inceleyebilirsin.", sourceVariant: "Ç — şıklatma", motion: "snap", pose: joined(C, SNAP, [0, 65, -15], [0, -25, 10], "thumb_tip", "middle_tip", [0, .48, .03]) },
  { ch: "D", tip: "Bir işaret parmağını dik tut; diğer elin kavisiyle uçlarını birleştir.", detail: "Dik parmak ve kavis birlikte D biçimini oluşturur.", pose: joined(C, POINT, [0, -65, 0], [0, 0, 0], "index_tip", "index_tip", [.02, 0, .07]) },
  { ch: "E", tip: "Üç parmağı yatay uzat; diğer işaret parmağını köklerine dik yerleştir.", detail: "Üç yatay parmak görünür kalmalı.", sourceVariant: "E1", pose: joined(THREE_FINGERS, POINT, [0, 0, -90], [0, 0, 0], "index_0", "index_tip", [0, -.05, -.24]) },
  { ch: "F", tip: "Baş ve işaret parmağını aç; diğer işaret parmağını yatay yerleştir.", detail: "İki elin temas noktasına bak.", sourceVariant: "F1", pose: joined(POINT, L, [0, 0, 90], [0, 0, 180], "index_tip", "index_1") },
  { ch: "G", tip: "İki kapalı eli yatay tutup üst üste yerleştir.", detail: "Üstteki elin işaret parmağı hafif kıvrık durur.", sourceVariant: "G2", pose: joined(HOOK, FIST, [100, 0, -90], [-100, 0, 90], "middle_1", "middle_1", [0, .65, 0]) },
  { ch: "Ğ", tip: "G biçimini koruyup üstteki elin kıvrık parmağını hareket ettir.", detail: "G ve Ğ arasındaki hareket farkını izle.", sourceVariant: "Ğ2", motion: "double-tap", pose: joined(HOOK, FIST, [100, 0, -90], [-100, 0, 90], "middle_1", "middle_1", [0, .65, 0]) },
  { ch: "H", tip: "İki parmağı aşağı uzat; diğer işaret parmağını yatay olarak üzerlerine getir.", detail: "İki dik parmak ve yatay temas birlikte görülmeli.", pose: joined(POINT, TWO, [0, 0, 90], [0, 0, 180], "index_2", "index_1") },
  { ch: "I", tip: "İşaret parmağını dik tut; diğer parmakları kapat.", detail: "Başparmak kapalı parmakların yanında durur.", pose: solo(POINT) },
  { ch: "İ", tip: "Dik işaret parmağının üstüne diğer elin parmak ucunu getir.", detail: "Üstteki el nokta konumundadır.", pose: joined(PINCH, POINT, [0, 0, 180], [0, 0, 0], "index_tip", "index_tip", [0, .22, .04]) },
  { ch: "J", tip: "Bir işaret parmağını dik tut; diğer elin kıvrık parmağını ucuna getir.", detail: "İki elin üstte kurduğu kıvrımı incele.", pose: joined(HOOK, POINT, [0, -60, 165], [0, 0, 0], "index_tip", "index_tip", [.06, .06, .07]) },
  { ch: "K", tip: "İki bitişik parmağı dik tut; diğer elin iki açık parmağını yanına getir.", detail: "Dik çizgi ile yana açılan iki parmağı karşılaştır.", sourceVariant: "K2", pose: joined(V, TWO, [0, 0, 90], [0, 0, 0], "index_tip", "middle_1", [0, 0, .1]) },
  { ch: "L", tip: "İşaret parmağını dik, başparmağını yana açık tut.", detail: "Diğer üç parmak avuca kapanır.", pose: solo(L) },
  { ch: "M", tip: "İki elin işaret ve orta parmaklarını aşağı aç; içteki uçları buluştur.", detail: "Dört parmakla oluşan M biçimini takip et.", pose: joined(V, V, [0, 0, 180], [0, 0, 180], "index_tip", "index_tip", [.06, 0, 0]) },
  { ch: "N", tip: "Bir elin iki parmağını, diğer elin işaret parmağını aşağı uzat.", detail: "Üç parmağın oluşturduğu N biçimini incele.", pose: joined(V, POINT, [0, 0, 180], [0, 0, 180], "index_tip", "index_tip", [.06, 0, 0]) },
  { ch: "O", tip: "Baş ve işaret parmağının uçlarını birleştir; diğer üç parmağı açık tut.", detail: "Parmak uçları arasındaki halka görünür kalmalı.", pose: solo(O, [0, -60, 0]) },
  { ch: "Ö", tip: "İki elin kıvrık baş ve işaret parmaklarını karşılıklı yaklaştır.", detail: "Yaklaşma hareketini yavaşlatarak inceleyebilirsin.", motion: "pinch", pose: joined(PINCH, PINCH, [0, 65, 165], [0, -65, 15], "index_tip", "index_tip", [0, .45, 0]) },
  { ch: "P", tip: "İşaret parmağını aşağı uzat; diğer parmakları kıvır.", detail: "Elin aşağı yönelmesine dikkat et.", pose: solo(POINT, [0, -20, 165]) },
  { ch: "R", tip: "İki işaret parmağını birbirine kıvrım oluşturacak şekilde yaklaştır.", detail: "Parmakların oluşturduğu kıvrımı ve temas noktasını takip et.", pose: joined(HOOK, HOOK, [0, 0, 138], [0, 0, 42], "index_tip", "index_tip", [0, 0, .19]) },
  { ch: "S", tip: "İki elin C biçimlerini ters yönlerde birbirine bağla.", detail: "Üst ve alt kavisler birlikte görünür kalmalı.", pose: joined(C, C, [0, 65, -10], [0, 65, 170], "thumb_tip", "index_tip", [0, 0, .08]) },
  { ch: "Ş", tip: "Üstteki C biçimini, alttaki elin işaret parmağıyla birleştir.", detail: "Alttaki elin konumu S harfinden farklıdır.", pose: joined(C, L, [0, 65, -10], [0, 0, 20], "thumb_tip", "index_tip", [0, 0, .08]) },
  { ch: "T", tip: "Dik işaret parmağının ucuna diğer işaret parmağını yatay yerleştir.", detail: "İki parmak T biçiminde birleşir.", pose: joined(POINT, POINT, [0, 0, 90], [0, 0, 0], "index_1", "index_tip", [0, .06, .07]) },
  { ch: "U", tip: "İşaret ve serçe parmağını açık tut; ortadaki iki parmağı kapat.", detail: "Açık iki parmak U biçimini oluşturur.", pose: solo(shape(["index", "pinky"]), [0, -15, 0]) },
  { ch: "Ü", tip: "İki elin kıvrılmış parmaklarını üst ve alt konumda karşılaştır.", detail: "Kaynakta gösterilen iki el biçimini birlikte incele.", pose: joined(C, shape(["index", "pinky"]), [0, -65, 85], [0, -20, 0], "thumb_tip", "index_tip", [0, .08, .05]) },
  { ch: "V", tip: "İşaret ve orta parmağını birbirinden açarak dik tut.", detail: "Diğer parmaklar avuca kapanır.", pose: solo(V) },
  { ch: "Y", tip: "V biçiminin ortasına diğer elin işaret parmağını getir.", detail: "İkinci işaret parmağı ortadaki çizgiyi tamamlar.", pose: joined(POINT, V, [0, 0, 0], [0, 0, 0], "index_tip", "middle_1", [-.11, -.02, .19]) },
  { ch: "Z", tip: "İki elin açık işaret ve orta parmaklarını yatay olarak karşılaştır.", detail: "Parmakların oluşturduğu zikzak biçimini takip et.", pose: joined(V, V, [0, 0, -90], [0, 0, 90], "index_tip", "middle_tip", [0, 0, .09]) },
];

// Generated from posed surface intersections, not guessed bone distances.
for (const letter of TID_ALPHABET) {
  const offset = (surfaceOffsets as Record<string, number[]>)[letter.ch];
  if (!offset) continue;
  letter.pose.right.position = letter.pose.right.position.map((v,i)=>v+offset[i]/2) as Vec3;
  letter.pose.left.position = letter.pose.left.position.map((v,i)=>v-offset[i]/2) as Vec3;
}

/** Runtime motion curves begin AND finish on the authored pose. */
export function samplePose(letter: Letter, elapsed: number): PairPose {
  const p = structuredClone(letter.pose);
  if (!letter.motion) return p;
  const t = (elapsed % 3.4) / 3.4;
  const pulse = (start: number, end: number) => t > start && t < end ? Math.sin(Math.PI * (t-start)/(end-start)) ** 2 : 0;
  if (letter.motion === "double-tap") {
    p.right.shape.index.bend[1] -= 30 * (pulse(.15, .38) + pulse(.44, .67));
  } else if (letter.motion === "pinch") {
    p.right.position[0] += .17 * (pulse(.15, .4) + pulse(.48, .73));
  } else if (letter.motion === "dot") {
    p.right.position[1] -= .13 * pulse(.2, .6);
  } else if (letter.motion === "snap") {
    // One snap per loop: prepare, fast release, hold, then a gentle reset.
    const ease = (x: number) => { const v = THREE.MathUtils.clamp(x, 0, 1); return v*v*v*(v*(v*6-15)+10); };
    const release = ease((t-.36)/.065) * (1-ease((t-.66)/.25));
    const middle = p.left.shape.middle;
    const thumb = p.left.shape.thumb;
    middle.bend = middle.bend.map((v, i) => v + ([88, 100, 55][i] - v)*release) as Vec3;
    thumb.bend[0] += 9*release;
    thumb.spread -= 12*release;
    p.left.rotation[2] -= 4 * release;
  }
  return p;
}

export { rigSpec, X, Y, Z };

