"use client";

import { useMemo, useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";

type Vec3 = [number, number, number];

// Realistic 21-point MediaPipe hand topology (open palm facing camera),
// derived from typical normalized MediaPipe landmark proportions.
const OPEN: Vec3[] = [
  [0, -1.2, 0], // 0 wrist
  [0.24, -0.96, 0.35], [0.43, -0.6, 0.42], [0.6, -0.31, 0.46], [0.72, -0.07, 0.48], // thumb
  [0.19, -0.29, 0.15], [0.22, 0.19, 0.18], [0.24, 0.53, 0.2], [0.26, 0.82, 0.22], // index
  [0, -0.24, 0.05], [0, 0.31, 0.08], [0, 0.7, 0.1], [0, 1.03, 0.12], // middle
  [-0.19, -0.29, 0], [-0.22, 0.24, 0.03], [-0.24, 0.6, 0.05], [-0.26, 0.91, 0.07], // ring
  [-0.36, -0.38, -0.05], [-0.41, 0, -0.03], [-0.43, 0.29, 0], [-0.46, 0.53, 0.02], // pinky
];

const CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

// Per-finger joint chains (base MCP/CMC index + the joints that curl toward it).
const FINGERS = [
  { base: 1, joints: [2, 3, 4], speed: 0.9, phase: 0.0, amount: 0.35 }, // thumb
  { base: 5, joints: [6, 7, 8], speed: 1.1, phase: 0.4, amount: 0.55 }, // index
  { base: 9, joints: [10, 11, 12], speed: 1.0, phase: 1.1, amount: 0.6 }, // middle
  { base: 13, joints: [14, 15, 16], speed: 1.2, phase: 1.8, amount: 0.55 }, // ring
  { base: 17, joints: [18, 19, 20], speed: 0.95, phase: 2.4, amount: 0.5 }, // pinky
];

function curledTarget(base: Vec3, open: Vec3, foldFrac: number): Vec3 {
  return [
    base[0] + (open[0] - base[0]) * (1 - foldFrac),
    base[1] + (open[1] - base[1]) * (1 - foldFrac),
    base[2] + (open[2] - base[2]) * (1 - foldFrac) + 0.32 * foldFrac,
  ];
}

export function HandSkeleton({ reducedMotion = false, scale = 1.35 }: { reducedMotion?: boolean; scale?: number }) {
  const group = useRef<THREE.Group>(null);
  const nodeRefs = useRef<THREE.Mesh[]>([]);
  const lineGeomRef = useRef<THREE.BufferGeometry>(null);
  const current = useRef<Vec3[]>(OPEN.map((p) => [...p] as Vec3));

  const initialLinePositions = useMemo(() => {
    const arr = new Float32Array(CONNECTIONS.length * 2 * 3);
    CONNECTIONS.forEach(([a, b], i) => {
      arr.set(OPEN[a], i * 6);
      arr.set(OPEN[b], i * 6 + 3);
    });
    return arr;
  }, []);

  useFrame((state) => {
    const t = state.clock.getElapsedTime();

    if (group.current && !reducedMotion) {
      group.current.rotation.y = Math.sin(t * 0.25) * 0.35 + 0.15;
      group.current.rotation.x = Math.cos(t * 0.2) * 0.08;
      group.current.position.y = Math.sin(t * 0.6) * 0.08;
    }

    const pts = current.current;
    for (let i = 0; i < OPEN.length; i++) pts[i] = OPEN[i];

    if (!reducedMotion) {
      for (const finger of FINGERS) {
        const curl = Math.max(0, Math.sin(t * finger.speed + finger.phase)) * finger.amount;
        const basePos = OPEN[finger.base];
        finger.joints.forEach((jointIndex, k) => {
          const foldFrac = curl * ((k + 1) / finger.joints.length);
          pts[jointIndex] = curledTarget(basePos, OPEN[jointIndex], foldFrac);
        });
      }
    }

    nodeRefs.current.forEach((mesh, i) => {
      if (!mesh) return;
      mesh.position.set(pts[i][0], pts[i][1], pts[i][2]);
      if (!reducedMotion) {
        const pulse = Math.sin(t * 1.4 + i * 0.6) * 0.06 + 1;
        mesh.scale.setScalar(pulse);
      }
    });

    const geom = lineGeomRef.current;
    if (geom) {
      const posAttr = geom.attributes.position as THREE.BufferAttribute;
      CONNECTIONS.forEach(([a, b], i) => {
        posAttr.setXYZ(i * 2, pts[a][0], pts[a][1], pts[a][2]);
        posAttr.setXYZ(i * 2 + 1, pts[b][0], pts[b][1], pts[b][2]);
      });
      posAttr.needsUpdate = true;
    }
  });

  return (
    <group ref={group} scale={scale}>
      <lineSegments>
        <bufferGeometry ref={lineGeomRef}>
          <bufferAttribute attach="attributes-position" args={[initialLinePositions, 3]} />
        </bufferGeometry>
        <lineBasicMaterial color="#8f7bff" transparent opacity={0.55} />
      </lineSegments>
      {OPEN.map((pos, i) => (
        <mesh
          key={i}
          position={pos}
          ref={(el) => {
            if (el) nodeRefs.current[i] = el;
          }}
        >
          <sphereGeometry args={[i === 0 ? 0.09 : 0.055, 16, 16]} />
          <meshStandardMaterial
            color={i === 0 ? "#00d4b8" : "#b8a6ff"}
            emissive={i === 0 ? "#00d4b8" : "#6d5bff"}
            emissiveIntensity={0.9}
            roughness={0.3}
            metalness={0.2}
          />
        </mesh>
      ))}
    </group>
  );
}
