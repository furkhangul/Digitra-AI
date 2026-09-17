import * as THREE from "three";
import { FINGERS, REST, orientation, samplePose, type PairPose, type Letter, type Vec3 } from "./poses";
import { ContactGuard } from "./contact-guard";
import approachOffsets from "./approach-offsets.json";

export type HandFrame = { position: THREE.Vector3; rotation: THREE.Quaternion; angles: number[]; presence: number };
export type PairFrame = { right: HandFrame; left: HandFrame };
export const SIDES = ["right", "left"] as const;
export function frameOf(pose: PairPose): PairFrame {
  return Object.fromEntries(SIDES.map(side => [side, {
    position: new THREE.Vector3(...pose[side].position),
    rotation: orientation(pose[side].rotation), presence: pose[side].visible ? 1 : 0,
    angles: FINGERS.flatMap(f => [...pose[side].shape[f].bend, pose[side].shape[f].spread, pose[side].shape[f].oppose]),
  }])) as PairFrame;
}
export function copyFrame(p: PairFrame): PairFrame {
  return Object.fromEntries(SIDES.map(side => [side, { ...p[side], position: p[side].position.clone(), rotation: p[side].rotation.clone(), angles: [...p[side].angles] }])) as PairFrame;
}
function smooth(t: number) { const x = THREE.MathUtils.clamp(t, 0, 1); return x*x*x*(x*(x*6-15)+10); }
function mix(out: PairFrame, a: PairFrame, b: PairFrame, t: number) {
  const u = smooth(t);
  for (const side of SIDES) {
    out[side].position.lerpVectors(a[side].position, b[side].position, u);
    out[side].rotation.slerpQuaternions(a[side].rotation, b[side].rotation, u).normalize();
    out[side].presence = THREE.MathUtils.lerp(a[side].presence, b[side].presence, u);
    for (let i=0; i<a[side].angles.length; i++) out[side].angles[i] = THREE.MathUtils.lerp(a[side].angles[i], b[side].angles[i], u);
  }
}

const PART_FALLBACK: Vec3 = [-.55, 0, .55];
/** Moves the pair apart along the sign's own parting direction. */
function part(frame: PairFrame, letter: string) {
  const o = (approachOffsets as Record<string, number[]>)[letter] ?? PART_FALLBACK;
  for (const side of SIDES) {
    const sign = side === "right" ? 1 : -1;
    frame[side].position.x += sign * o[0];
    frame[side].position.y += sign * o[1];
    frame[side].position.z += sign * o[2];
  }
}

/** Unit parting direction used for the mid-transition push; falls back to the
 * previous sign's direction when the new one is single-handed. */
function unitPart(letter: string, previous: string): Vec3 {
  const table = approachOffsets as Record<string, number[]>;
  let o = table[letter] ?? PART_FALLBACK;
  if (Math.hypot(o[0], o[1], o[2]) < 1e-4) o = table[previous] ?? PART_FALLBACK;
  const n = Math.hypot(o[0], o[1], o[2]) || 1;
  return [o[0]/n, o[1]/n, o[2]/n];
}

/** Interrupted transitions always restart from the last displayed frame.
 * Three phases release the contact, form the new sign apart, then approach.
 * No wall clock: pausing, changing playback speed and background tabs are stable.
 */
export class HandPlayer {
  current: PairFrame;
  private from: PairFrame;
  private release: PairFrame;
  private approach: PairFrame;
  private target: PairFrame;
  private letter: Letter;
  private elapsed = 0;
  private motionTime = 0;
  private duration = 1.35;
  private guard:ContactGuard | null;
  private spread: Vec3 = [0, 0, 0];
  constructor(letter: Letter, surfaceContact=false) {
    this.guard=surfaceContact?new ContactGuard():null;
    this.letter = letter;
    this.current = frameOf(letter.pose);
    this.from = copyFrame(this.current); this.release = copyFrame(this.current);
    this.approach = copyFrame(this.current); this.target = copyFrame(this.current);
    this.elapsed = this.duration;
  }
  select(letter: Letter, replay = false) {
    const previous = this.letter;
    this.letter = letter;
    this.from = copyFrame(this.current);
    this.target = frameOf(letter.pose);
    this.release = copyFrame(this.from);
    this.approach = copyFrame(this.target);
    // Part along the line the two hands actually separate on, measured on the
    // real surfaces per sign (scripts/tid_sdf/approach.py). A single fixed
    // diagonal does not open a sign whose hands are stacked, so the straight
    // move back in would drive one hand through the other.
    if (replay && !letter.motion) this.release = frameOf(REST);
    else part(this.release, previous.ch);
    part(this.approach, letter.ch);
    this.spread = unitPart(letter.ch, previous.ch);
    this.elapsed = 0;
    this.motionTime = 0;
  }
  tick(dt: number, playing = true, speed = 1, reducedMotion = false) {
    if (!playing) return false;
    const step = Math.min(Math.max(dt, 0), .05) * speed;
    if (reducedMotion && this.elapsed < this.duration) this.elapsed = this.duration;
    else this.elapsed = Math.min(this.duration, this.elapsed + step);
    if (this.elapsed < .27) mix(this.current, this.from, this.release, this.elapsed/.27);
    else if (this.elapsed < 1.02) {
      const u = (this.elapsed-.27)/.75;
      mix(this.current, this.release, this.approach, u);
      // Blending straight from one parted pose to the other can still sweep the
      // hands through each other when the two signs part in opposing directions.
      // An extra outward push, peaking mid-way, keeps the path clear.
      const boost = Math.sin(Math.PI * u) * .55;
      for (const side of SIDES) {
        const sign = side === "right" ? 1 : -1;
        this.current[side].position.x += sign * this.spread[0] * boost;
        this.current[side].position.y += sign * this.spread[1] * boost;
        this.current[side].position.z += sign * this.spread[2] * boost;
      }
    }
    else if (this.elapsed < this.duration) mix(this.current, this.approach, this.target, (this.elapsed-1.02)/.33);
    else {
      // Reduced-motion users see the authored contact pose without an endless loop.
      if (!reducedMotion) this.motionTime += step;
      else this.motionTime = 0;
      const target = frameOf(samplePose(this.letter, this.motionTime));
      mix(this.current, target, target, 1);
    }
    this.guard?.apply(this.current);
    return this.elapsed < this.duration || (!reducedMotion && Boolean(this.letter.motion));
  }
  get settled() { return this.elapsed >= this.duration; }
  dispose(){this.guard?.dispose();}
  get phase() { return this.settled ? "hold" : this.elapsed < .27 ? "release" : this.elapsed < 1.02 ? "shape" : "approach"; }
}

export function serialiseFrame(frame: PairFrame) {
  return Object.fromEntries(SIDES.map(side => [side, {
    position: frame[side].position.toArray() as Vec3,
    rotation: frame[side].rotation.toArray(), presence: frame[side].presence,
    angles: [...frame[side].angles],
  }]));
}
