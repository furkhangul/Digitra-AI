"use client";

// Local diagnostic page (not linked from the site): renders every hero
// handshape at the hero's own camera/orientation, so poses can be checked by
// eye all at once instead of by timing screenshots against the animation.
import { Canvas } from "@react-three/fiber";
import { Suspense, useLayoutEffect, useMemo, useRef } from "react";
import { useGLTF } from "@react-three/drei";
import { SkeletonUtils } from "three-stdlib";
import * as THREE from "three";
import { CHAINS, LETTERS, BASE_Y_ROTATION } from "@/components/home/hand-poses";
import type { Pose } from "@/components/home/hand-poses";

const MODEL_URL = "/models/hand.glb";

function PosedHand({ pose }: { pose: Pose }) {
  const { scene: source } = useGLTF(MODEL_URL);
  const scene = useMemo(() => SkeletonUtils.clone(source) as THREE.Object3D, [source]);
  const ref = useRef<THREE.Group>(null);

  const { scale, offset } = useMemo(() => {
    const box = new THREE.Box3().setFromObject(scene);
    const size = new THREE.Vector3();
    box.getSize(size);
    const center = new THREE.Vector3();
    box.getCenter(center);
    return { scale: 4 / (Math.max(size.x, size.y, size.z) || 1), offset: center };
  }, [scene]);

  useLayoutEffect(() => {
    const byName: Record<string, THREE.Object3D> = {};
    scene.traverse((o) => (byName[o.name] = o));
    (Object.keys(CHAINS) as (keyof typeof CHAINS)[]).forEach((finger) => {
      CHAINS[finger].forEach(({ bone, axis, maxBend }) => {
        const b = byName[bone];
        if (!b) return;
        const delta = new THREE.Quaternion().setFromAxisAngle(axis, pose[finger] * maxBend);
        b.quaternion.multiply(delta);
      });
    });
  }, [scene, pose]);

  return (
    <group ref={ref} rotation={[-0.05, BASE_Y_ROTATION, 0]} scale={scale}>
      <primitive object={scene} position={[-offset.x, -offset.y, -offset.z]} />
    </group>
  );
}

export default function DebugHand() {
  return (
    <div className="grid grid-cols-3 bg-black">
      {LETTERS.map((l) => (
        <div key={l.label} className="relative aspect-square border border-white/20">
          <span className="absolute top-2 left-2 z-10 text-lg font-bold text-white">{l.label}</span>
          <Canvas camera={{ position: [0, 0, 6], fov: 42 }}>
            <ambientLight intensity={0.9} />
            <pointLight position={[4, 4, 4]} intensity={45} color="#ffffff" />
            <pointLight position={[-4, -2, 2]} intensity={22} color="#8f7bff" />
            <pointLight position={[0, -3, 4]} intensity={14} color="#00d4b8" />
            <Suspense fallback={null}>
              <PosedHand pose={l.pose} />
            </Suspense>
          </Canvas>
        </div>
      ))}
    </div>
  );
}
