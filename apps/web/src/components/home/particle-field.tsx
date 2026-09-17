"use client";

import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

export function ParticleField({ count = 500, reducedMotion = false }: { count?: number; reducedMotion?: boolean }) {
  const points = useRef<THREE.Points>(null);

  // Deterministic pseudo-random sequence (mulberry32) instead of Math.random:
  // React treats render-phase calls to Math.random as an impurity violation,
  // and this scene only ever needs a stable-looking, non-cryptographic scatter.
  const positions = useMemo(() => {
    let seed = 1337;
    const next = () => {
      seed = (seed + 0x6d2b79f5) | 0;
      let t = seed;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };

    const arr = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const r = 4 + next() * 6;
      const theta = next() * Math.PI * 2;
      const phi = Math.acos(2 * next() - 1);
      arr[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      arr[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta) * 0.6;
      arr[i * 3 + 2] = r * Math.cos(phi) - 2;
    }
    return arr;
  }, [count]);

  useFrame((state) => {
    if (!points.current || reducedMotion) return;
    const t = state.clock.getElapsedTime();
    points.current.rotation.y = t * 0.02;
    points.current.rotation.x = Math.sin(t * 0.05) * 0.05;
  });

  return (
    <points ref={points}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
      </bufferGeometry>
      <pointsMaterial color="#9c7bff" size={0.035} transparent opacity={0.5} sizeAttenuation />
    </points>
  );
}
