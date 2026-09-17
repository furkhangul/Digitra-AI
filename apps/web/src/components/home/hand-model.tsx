"use client";

import { useEffect, useLayoutEffect, useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import { SkeletonUtils } from "three-stdlib";
import type { MotionValue } from "framer-motion";
import * as THREE from "three";
import {
  BASE_Y_ROTATION,
  CHAINS,
  HAND_CENTER_Y,
  ORBIT_TARGET_Y,
  LETTERS,
  SPREAD_AXIS,
  type FingerName,
  type Pose,
} from "./hand-poses";

const MODEL_URL = "/models/hand.glb";
const TARGET_SIZE = 5.6;

const TRANSITION = 0.55;
// Slightly uneven holds — an even metronome is the main thing that reads as
// machine rather than person.
const HOLDS = [1.35, 1.15, 1.5, 1.2, 1.4];
const SCHEDULE = HOLDS.map((h) => h + TRANSITION);
const TOTAL = SCHEDULE.reduce((a, b) => a + b, 0);

// Fingers don't fire in lockstep: the index leads and the pinky trails, as a
// fraction of the transition.
const FINGER_LAG: Record<FingerName, number> = {
  index: 0,
  thumb: 0.03,
  middle: 0.06,
  ring: 0.11,
  pinky: 0.16,
};

// Each finger gets its own tremor frequencies so no two ever sync up.
const TREMOR: Record<FingerName, [number, number, number]> = {
  index: [2.3, 3.7, 0.0],
  middle: [2.1, 4.1, 1.7],
  ring: [2.7, 3.3, 3.1],
  pinky: [1.9, 4.3, 4.4],
  thumb: [2.5, 3.1, 2.2],
};

function clamp(x: number, lo: number, hi: number) {
  return Math.min(hi, Math.max(lo, x));
}

// J and Z are the two ASL letters that aren't a static handshape — the
// fingertip actually draws the letter in the air. Since the hand's own
// finger curl can't trace a path, the whole hand is nudged instead: small
// translation + a touch of roll, on a continuous periodic curve so the loop
// never has a reset jump. Not a literal calligraphic trace — a legible
// approximation of "this shape moves like this".
function triangleWave(x: number) {
  return (2 / Math.PI) * Math.asin(Math.sin(x));
}

function traceOffset(letter: "J" | "Z", t: number): { dx: number; dy: number; rot: number } {
  if (letter === "J") {
    const w = (2 * Math.PI) / 2.4; // 2.4s loop
    const a = t * w;
    // sin/cos at the same frequency trace a small closed loop — reads as the
    // hook a pinky draws for J, repeating cleanly with no snap-back.
    return {
      dx: 0.16 * Math.sin(a),
      dy: -0.07 - 0.11 * (1 - Math.cos(a)),
      rot: 0.14 * Math.sin(a - 0.5),
    };
  }
  const w = (2 * Math.PI) / 2.6; // 2.6s loop
  const a = t * w;
  // Two triangle waves at a 1:2 ratio zig-zag back and forth — the closest a
  // smooth periodic curve gets to Z's three straight strokes.
  const tx = triangleWave(a);
  const ty = triangleWave(2 * a + Math.PI / 2);
  return { dx: 0.22 * tx, dy: 0.08 * ty, rot: 0.06 * tx };
}

// Gentle overshoot: a real finger reaching a shape slightly passes the target
// and settles back, where a lerp arrives and simply stops.
function easeOutBack(x: number) {
  const c1 = 0.7;
  const c3 = c1 + 1;
  const p = x - 1;
  return 1 + c3 * p * p * p + c1 * p * p;
}

export function HandModel({
  reducedMotion = false,
  pose,
  morphTo,
  morph,
  pitch,
  dolly,
  recenter,
  tipX,
  tipY,
  wristX,
  wristY,
  trace,
}: {
  reducedMotion?: boolean;
  /**
   * Hold a single handshape instead of cycling the fingerspelling sequence.
   * A full-screen poster wants one confident, still gesture — a hand swapping
   * letters at that size reads as fidgeting, not signing.
   */
  pose?: Pose;
  /** Handshape to blend toward as `morph` goes 0 → 1. */
  morphTo?: Pose;
  /** Blend amount, read per frame so it can be driven by scroll. */
  morph?: MotionValue<number>;
  /** Extra pitch in radians, added to the idle sway. */
  pitch?: MotionValue<number>;
  /** Moves the hand along +Z, toward the camera, for a real perspective approach. */
  dolly?: MotionValue<number>;
  /**
   * 0 → resting height, 1 → sitting on the camera's look-at point. The hand
   * normally hangs well above that point, and dollying an off-axis object
   * toward the camera throws it further off screen the closer it gets, so it
   * has to be centred before it can approach.
   */
  recenter?: MotionValue<number>;
  /**
   * Written each frame with the index fingertip's position as viewport
   * percentages, so DOM overlays can be pinned to it exactly. Scaling the
   * canvas in CSS instead would decouple the two.
   */
  tipX?: MotionValue<number>;
  tipY?: MotionValue<number>;
  /**
   * Same idea for the wrist, so the surface the hand emerges through can be
   * pinned to it and travel with the hand for the whole sequence.
   */
  wristX?: MotionValue<number>;
  wristY?: MotionValue<number>;
  /** J and Z are dynamic ASL letters — nudges the whole hand through a small
   * looping path approximating the shape traced in the air. */
  trace?: "J" | "Z";
}) {
  const group = useRef<THREE.Group>(null);
  const pitchGroup = useRef<THREE.Group>(null);
  const { scene: sourceScene } = useGLTF(MODEL_URL);
  // The GLTF cache returns the SAME scene object on every mount; if we
  // rotate its bones in place, hot-reloads (or a second instance) would
  // capture an already-animated pose as their "bind" reference. Clone with
  // SkeletonUtils so skinning stays correctly bound to our own copy.
  const scene = useMemo(() => SkeletonUtils.clone(sourceScene) as THREE.Object3D, [sourceScene]);

  // The authored material is a single flat, untextured color (no normal/AO
  // maps, no env reflections) — reads as plastic. Swap in a physical material
  // so the Environment light below actually has something to reflect off,
  // plus a faint sheen that fakes the soft skin scatter a flat material can't.
  useEffect(() => {
    scene.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (!mesh.isMesh) return;
      const source = mesh.material as THREE.MeshStandardMaterial;
      mesh.material = new THREE.MeshPhysicalMaterial({
        color: source?.color ?? new THREE.Color("#ffb38f"),
        roughness: 0.62,
        metalness: 0,
        clearcoat: 0.08,
        clearcoatRoughness: 0.45,
        sheen: 0.35,
        sheenRoughness: 0.8,
        sheenColor: new THREE.Color("#ff9d7a"),
        envMapIntensity: 0.65,
      });
    });
  }, [scene]);

  const bones = useRef<Record<string, THREE.Object3D>>({});
  const bindQuats = useRef<Record<string, THREE.Quaternion>>({});
  const tipVec = useMemo(() => new THREE.Vector3(), []);
  const tipDir = useMemo(() => new THREE.Vector3(), []);

  const { scale, offset } = useMemo(() => {
    const box = new THREE.Box3().setFromObject(scene);
    const size = new THREE.Vector3();
    box.getSize(size);
    const center = new THREE.Vector3();
    box.getCenter(center);
    const maxDim = Math.max(size.x, size.y, size.z) || 1;
    return { scale: TARGET_SIZE / maxDim, offset: center };
  }, [scene]);

  useLayoutEffect(() => {
    const found: Record<string, THREE.Object3D> = {};
    const quats: Record<string, THREE.Quaternion> = {};
    const names = Object.values(CHAINS)
      .flat()
      .map((j) => j.bone)
      // Not animated, but projected each frame to place the emergence surface.
      .concat("radius_ulna");
    scene.traverse((obj) => {
      if (names.includes(obj.name)) {
        found[obj.name] = obj;
        quats[obj.name] = obj.quaternion.clone();
      }
    });
    bones.current = found;
    bindQuats.current = quats;
  }, [scene]);

  useFrame((state) => {
    const t = state.clock.getElapsedTime();

    // Turning is the orbit camera's job now (auto-rotate plus drag), so the
    // model keeps only the small secondary motion that makes it feel alive.
    // Layered incommensurate sines: a single sine visibly loops, whereas
    // frequencies that never divide evenly read as unrepeating drift.
    const swayX = Math.cos(t * 0.15) * 0.05 + Math.sin(t * 0.41 + 0.7) * 0.02;
    const swayRoll = Math.sin(t * 0.27 + 2.1) * 0.04;
    const bob = Math.sin(t * 0.45) * 0.05 + Math.sin(t * 0.31 + 1.9) * 0.02;

    if (group.current) {
      group.current.rotation.y = BASE_Y_ROTATION;
      group.current.rotation.x = reducedMotion ? 0 : swayX - 0.05;
      group.current.rotation.z = reducedMotion ? 0 : swayRoll;
    }

    // Pitch lives on an outer group so it is applied *after* the yaw that turns
    // the palm to camera. Folded into the same Euler as the yaw it rotates
    // about an already-turned axis, which tips the hand sideways out of frame
    // instead of tilting the fingers toward the viewer.
    if (pitchGroup.current) {
      pitchGroup.current.rotation.x = pitch?.get() ?? 0;
      const rest = HAND_CENTER_Y + (reducedMotion ? 0 : bob);
      const toCentre = recenter ? Math.min(1, Math.max(0, recenter.get())) : 0;
      let traceDx = 0;
      let traceDy = 0;
      let traceRot = 0;
      if (trace && !reducedMotion) {
        ({ dx: traceDx, dy: traceDy, rot: traceRot } = traceOffset(trace, t));
      }
      pitchGroup.current.position.x = traceDx;
      pitchGroup.current.position.y = rest + (ORBIT_TARGET_Y - rest) * toCentre + traceDy;
      pitchGroup.current.position.z = dolly?.get() ?? 0;
      pitchGroup.current.rotation.z = traceRot;
    }

    // Reduced motion freezes the idle fingerspelling cycle (there's nothing
    // sensible to hold still on), but a fixed pose isn't motion — it must
    // still be applied, or every letter renders as the unposed bind shape.
    if (reducedMotion && !pose) return;

    // Walk the uneven schedule to find which letter we're on and how far
    // through its transition. Skipped entirely when holding a fixed pose.
    let idx = 0;
    let nextIdx = 0;
    let rawT = 0;
    if (!pose) {
      let cycleT = t % TOTAL;
      while (cycleT >= SCHEDULE[idx]) {
        cycleT -= SCHEDULE[idx];
        idx = (idx + 1) % LETTERS.length;
      }
      nextIdx = (idx + 1) % LETTERS.length;
      rawT = clamp((cycleT - HOLDS[idx]) / TRANSITION, 0, 1);
    }

    // A slow shared "breath" plus per-finger micro-tremor, so a held shape is
    // alive rather than frozen. Suppressed under reduced motion.
    const breath = reducedMotion ? 0 : Math.sin(t * 0.35) * 0.02;

    // Scroll-driven blend toward a target shape. Once it starts, it overrides
    // the fingerspelling underneath rather than fighting it.
    const morphAmount = morphTo && morph ? clamp(morph.get(), 0, 1) : 0;

    (Object.keys(CHAINS) as FingerName[]).forEach((finger) => {
      let curled: number;
      if (pose) {
        curled = pose[finger];
      } else {
        const lag = FINGER_LAG[finger];
        const staggered = clamp((rawT - lag) / (1 - lag), 0, 1);
        const from = LETTERS[idx].pose[finger];
        const to = LETTERS[nextIdx].pose[finger];
        curled = from + (to - from) * easeOutBack(staggered);
      }

      if (morphTo && morphAmount > 0) {
        curled = curled + (morphTo[finger] - curled) * morphAmount;
      }

      const [f1, f2, phase] = TREMOR[finger];
      const tremor = reducedMotion
        ? 0
        : Math.sin(t * f1 + phase) * 0.012 + Math.sin(t * f2 + phase * 1.7) * 0.008;
      // Allow a touch below zero: relaxed fingers sit slightly hyperextended.
      const curl = clamp(curled + tremor + breath, -0.04, 1.02);

      const spreadAngle = pose?.spread?.[finger] ?? 0;

      CHAINS[finger].forEach(({ bone: name, axis, maxBend }, jointIndex) => {
        const bone = bones.current[name];
        const bind = bindQuats.current[name];
        if (!bone || !bind) return;
        let delta = new THREE.Quaternion().setFromAxisAngle(axis, curl * maxBend);
        // Fan the finger sideways at its base joint only, before the curl —
        // this is what separates V/K/P/R from U, which curl identically.
        if (jointIndex === 0 && spreadAngle) {
          const spread = new THREE.Quaternion().setFromAxisAngle(SPREAD_AXIS, spreadAngle);
          delta = spread.multiply(delta);
        }
        bone.quaternion.copy(bind).multiply(delta);
      });
    });

    // Project the fingertip so an overlay can sit exactly on it. `index_dist`
    // is the base of the last phalanx, not the tip, so extrapolate one bone
    // further along the finger's own direction.
    if (tipX && tipY) {
      const distal = bones.current["index_dist"];
      const middle = bones.current["index_midd"];
      if (distal && middle) {
        distal.updateWorldMatrix(true, false);
        middle.updateWorldMatrix(true, false);
        tipVec.setFromMatrixPosition(distal.matrixWorld);
        tipDir.setFromMatrixPosition(middle.matrixWorld);
        const boneLength = tipVec.distanceTo(tipDir);
        tipDir.subVectors(tipVec, tipDir).normalize();
        tipVec.addScaledVector(tipDir, boneLength * 0.85).project(state.camera);
        tipX.set((tipVec.x * 0.5 + 0.5) * 100);
        tipY.set((-tipVec.y * 0.5 + 0.5) * 100);
      }
    }

    if (wristX && wristY) {
      // The forearm root, not the group origin: the origin is the pivot the
      // hand rotates about, so it stays put while the hand visibly swings, and
      // an emergence surface pinned to it would sit still and drift off the
      // wrist. This bone follows every transform.
      const wrist = bones.current["radius_ulna"];
      if (wrist) {
        wrist.updateWorldMatrix(true, false);
        tipVec.setFromMatrixPosition(wrist.matrixWorld).project(state.camera);
        wristX.set((tipVec.x * 0.5 + 0.5) * 100);
        wristY.set((-tipVec.y * 0.5 + 0.5) * 100);
      }
    }
  });

  return (
    <group ref={pitchGroup}>
      <group ref={group} scale={scale}>
        <primitive object={scene} position={[-offset.x, -offset.y, -offset.z]} />
      </group>
    </group>
  );
}

useGLTF.preload(MODEL_URL);
