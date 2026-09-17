"use client";

import { Canvas, useFrame } from "@react-three/fiber";
import { Environment, Lightformer, OrbitControls } from "@react-three/drei";
import { Suspense, useEffect, useRef } from "react";
import type { OrbitControls as OrbitControlsImpl } from "three-stdlib";
import { HandModel } from "./hand-model";
import { HandSkeleton } from "./hand-skeleton";
import { ParticleField } from "./particle-field";
import { ORBIT_TARGET_X, ORBIT_TARGET_Y, type Pose } from "./hand-poses";
import type { MotionValue } from "framer-motion";
import { usePrefersReducedMotion } from "@/lib/use-reduced-motion";

// Idle motion sweeps the camera within a flattering arc rather than spinning a
// full circle: seen exactly edge-on the hand collapses to a sliver with no
// readable fingers. Dragging is still unrestricted, so a viewer can take it all
// the way around; this only governs what it does on its own.
const IDLE_SPEED = 0.22;
const RESUME_DELAY = 2.5; // seconds of stillness before idle motion takes back over

function HandControls({
  reducedMotion,
  idleArc,
  settle,
}: {
  reducedMotion: boolean;
  idleArc: number;
  /**
   * 0 → free idle drift, 1 → hold dead centre. The intro raises this as the
   * hand turns to point at the viewer: with the camera still orbiting, a
   * finger aimed down the world Z axis is seen from the side and reads as
   * tilting away rather than coming at you.
   */
  settle?: MotionValue<number>;
}) {
  const ref = useRef<OrbitControlsImpl>(null);
  const dragging = useRef(false);
  const lastInteraction = useRef(0);

  useEffect(() => {
    const controls = ref.current;
    if (!controls) return;
    // OrbitControls sets `touch-action: none`, which would trap vertical
    // scrolling anywhere over the canvas on touch devices. `pan-y` gives the
    // page back its scroll while a horizontal drag still spins the hand.
    const el = controls.domElement as HTMLElement | undefined;
    if (el) el.style.touchAction = "pan-y";
  }, []);

  useFrame((state, delta) => {
    const controls = ref.current;
    if (!controls || reducedMotion || dragging.current) return;

    const now = state.clock.getElapsedTime();
    if (now - lastInteraction.current < RESUME_DELAY) return;

    // Chase the oscillating target rather than assigning it, so picking up
    // again after a drag eases in from wherever the viewer left the hand.
    const lock = settle ? Math.min(1, Math.max(0, settle.get())) : 0;
    const target = Math.sin(now * IDLE_SPEED) * idleArc * (1 - lock);
    const current = controls.getAzimuthalAngle();
    const rate = Math.min(1, delta * (1.1 + lock * 6));
    controls.setAzimuthalAngle(current + (target - current) * rate);
  });

  return (
    <OrbitControls
      ref={ref}
      target={[ORBIT_TARGET_X, ORBIT_TARGET_Y, 0]}
      enableZoom={false}
      enablePan={false}
      enableDamping
      dampingFactor={0.08}
      rotateSpeed={0.7}
      onStart={() => {
        dragging.current = true;
      }}
      onEnd={() => {
        dragging.current = false;
        lastInteraction.current = performance.now() / 1000;
      }}
      // Vertical is fenced off so the hand can't be dragged upside down or
      // viewed edge-on from directly above.
      minPolarAngle={Math.PI * 0.28}
      maxPolarAngle={Math.PI * 0.72}
    />
  );
}

export function HeroScene({
  pose,
  idleArc = 1.15,
  morphTo,
  morph,
  pitch,
  dolly,
  recenter,
  tipX,
  tipY,
  wristX,
  wristY,
  settle,
  trace,
}: {
  /** Hold one handshape (poster mode) instead of cycling fingerspelling. */
  pose?: Pose;
  /** Radians either side of front that the idle camera drifts through. */
  idleArc?: number;
  /** Handshape to blend toward as `morph` goes 0 → 1. */
  morphTo?: Pose;
  morph?: MotionValue<number>;
  /** Extra pitch in radians, added to the idle sway. */
  pitch?: MotionValue<number>;
  /** Moves the hand toward the camera for a real perspective approach. */
  dolly?: MotionValue<number>;
  /** 0 → resting height, 1 → sitting on the camera's look-at point. */
  recenter?: MotionValue<number>;
  /** Receive the index fingertip's viewport position, in percentages. */
  tipX?: MotionValue<number>;
  tipY?: MotionValue<number>;
  /** Same for the hand's own centre, for pinning the emergence surface. */
  wristX?: MotionValue<number>;
  wristY?: MotionValue<number>;
  /** 0 → free idle drift, 1 → hold the camera dead centre. */
  settle?: MotionValue<number>;
  /** J and Z are dynamic ASL letters — traces the shape in the air. */
  trace?: "J" | "Z";
} = {}) {
  const reducedMotion = usePrefersReducedMotion();

  return (
    <Canvas
      dpr={[1, 1.75]}
      camera={{ position: [0, 0, 6], fov: 42 }}
      gl={{ antialias: true, alpha: true }}
    >
      <ambientLight intensity={0.9} />
      <pointLight position={[4, 4, 4]} intensity={45} color="#ffffff" />
      <pointLight position={[-4, -2, 2]} intensity={22} color="#8f7bff" />
      {/* Kept dim: the intro tips the palm down toward this fill, and at any
          real intensity it washes the fingers mint green. */}
      <pointLight position={[0, -3, 4]} intensity={5} color="#00d4b8" />
      {/* Gives the skin something to reflect — without it there is no
          environment at all and even a good material reads as flat plastic.
          Built from simple virtual panels rather than a photographic HDRI:
          no network fetch, and a tiny 64px PMREM that won't crash a weak or
          software-rendered GPU the way a full preset environment can. */}
      <Environment resolution={64}>
        <Lightformer intensity={2.2} color="#ffffff" position={[2, 3, 4]} scale={[6, 6, 1]} />
        <Lightformer intensity={0.7} color="#8f7bff" position={[-4, 1, 2]} scale={[4, 4, 1]} />
        <Lightformer intensity={0.5} color="#00d4b8" position={[2, -3, -3]} scale={[4, 4, 1]} />
      </Environment>
      <Suspense fallback={<HandSkeleton reducedMotion={reducedMotion} />}>
        <ParticleField reducedMotion={reducedMotion} />
        <HandModel
          reducedMotion={reducedMotion}
          pose={pose}
          morphTo={morphTo}
          morph={morph}
          pitch={pitch}
          dolly={dolly}
          recenter={recenter}
          tipX={tipX}
          tipY={tipY}
          wristX={wristX}
          wristY={wristY}
          trace={trace}
        />
      </Suspense>
      <HandControls reducedMotion={reducedMotion} idleArc={idleArc} settle={settle} />
    </Canvas>
  );
}
