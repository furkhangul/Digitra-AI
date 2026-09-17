"use client";

import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, useGLTF } from "@react-three/drei";
import * as THREE from "three";
import { SkeletonUtils } from "three-stdlib";

// Chain layout of simplehand.glb: Bone (wrist) + five 4-bone chains. The
// loader sanitises "Bone.001" to "Bone001".
const CHAINS = {
  thumb: ["Bone001", "Bone002", "Bone003", "Bone004"],
  index: ["Bone005", "Bone006", "Bone007", "Bone008"],
  middle: ["Bone009", "Bone010", "Bone011", "Bone012"],
  ring: ["Bone013", "Bone014", "Bone015", "Bone016"],
  pinky: ["Bone017", "Bone018", "Bone019", "Bone020"],
} as const;

export type SimplePose = Record<keyof typeof CHAINS, number>;

const MAX_BENDS = [1.3, 1.5, 1.1, 0.8];

function SimpleHandModel({
  pose,
  axis,
  sign,
}: {
  pose: SimplePose;
  axis: "x" | "z";
  sign: 1 | -1;
}) {
  const { scene } = useGLTF("/models/simplehand.glb");
  const clone = useMemo(
    () => SkeletonUtils.clone(scene) as THREE.Group,
    [scene]
  );

  const root = useRef<THREE.Group>(null);
  // Bind-pose quaternions, captured once — re-capturing after a pose was
  // applied would compound every previous pose into the next one.
  const restRef = useRef<Map<string, THREE.Quaternion> | null>(null);

  useEffect(() => {
    // Normalize: skinned geometry bounds are pre-skinning (useless here), so
    // size the model from its bone world positions instead.
    if (!root.current) return;
    root.current.updateWorldMatrix(true, true);
    const box = new THREE.Box3();
    const v = new THREE.Vector3();
    clone.traverse((o) => {
      if ((o as THREE.Bone).isBone) box.expandByPoint((o as THREE.Bone).getWorldPosition(v));
    });
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const s = 2.0 / Math.max(size.x, size.y, size.z);
    console.log("[simplehand] bone bbox size", size.toArray(), "scale", s);
    root.current.scale.setScalar(s);
    root.current.position.set(-center.x * s, -center.y * s, -center.z * s);
  }, [clone]);

  useEffect(() => {
    const boneMap = new Map<string, THREE.Bone>();
    clone.traverse((o) => {
      if ((o as THREE.Bone).isBone) boneMap.set(o.name, o as THREE.Bone);
    });
    if (!restRef.current) {
      const m = new Map<string, THREE.Quaternion>();
      boneMap.forEach((b, name) => m.set(name, b.quaternion.clone()));
      restRef.current = m;
    }
    const rest = restRef.current;
    console.log("[simplehand] names:", [...boneMap.keys()].join(","));

    for (const [finger, bones] of Object.entries(CHAINS)) {
      const curl = pose[finger as keyof typeof CHAINS] ?? 0;
      bones.forEach((name, i) => {
        const b = boneMap.get(name);
        const q0 = rest.get(name);
        if (!b || !q0) return;
        const bend = curl * MAX_BENDS[i] * sign;
        const delta = new THREE.Quaternion().setFromAxisAngle(
          axis === "x" ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 0, 1),
          bend
        );
        b.quaternion.copy(q0).multiply(delta);
      });
    }
  }, [clone, pose, axis, sign]);

  return (
    <group ref={root}>
      <primitive object={clone} />
    </group>
  );
}

const POSES: Record<string, SimplePose> = {
  open: { thumb: 0, index: 0, middle: 0, ring: 0, pinky: 0 },
  fist: { thumb: 1, index: 1, middle: 1, ring: 1, pinky: 1 },
  point: { thumb: 0.85, index: 0, middle: 1, ring: 1, pinky: 1 },
  L: { thumb: 0, index: 0, middle: 1, ring: 1, pinky: 1 },
  Y: { thumb: 0, index: 1, middle: 1, ring: 1, pinky: 0 },
};

export default function DebugSimpleHand() {
  const [poseName, setPoseName] = useState("point");
  const [axis, setAxis] = useState<"x" | "z">("x");
  const [sign, setSign] = useState<1 | -1>(1);

  return (
    <div style={{ background: "#111", color: "#fff", minHeight: "100vh", padding: 24 }}>
      <h1>simplehand debug</h1>
      <div style={{ display: "flex", gap: 8, marginBottom: 12 }}>
        {Object.keys(POSES).map((k) => (
          <button key={k} onClick={() => setPoseName(k)} style={{ padding: "6px 12px" }}>
            {k}
          </button>
        ))}
        <button onClick={() => setAxis(axis === "x" ? "z" : "x")} style={{ padding: "6px 12px" }}>
          axis: {axis}
        </button>
        <button onClick={() => setSign(sign === 1 ? -1 : 1)} style={{ padding: "6px 12px" }}>
          sign: {sign}
        </button>
      </div>
      <div style={{ width: 700, height: 600 }}>
        <Canvas camera={{ position: [0, 0, 6.5], fov: 40 }}>
          <ambientLight intensity={1.2} />
          <directionalLight position={[3, 4, 5]} intensity={2} />
          <Suspense fallback={null}>
            <SimpleHandModel pose={POSES[poseName]} axis={axis} sign={sign} />
          </Suspense>
          <OrbitControls />
        </Canvas>
      </div>
    </div>
  );
}
