"use client";

import { Suspense, useEffect, useMemo, useRef } from "react";
import { Canvas, useThree } from "@react-three/fiber";
import { useGLTF } from "@react-three/drei";
import * as THREE from "three";
import { SkeletonUtils, mergeVertices } from "three-stdlib";
import { cn } from "@/lib/utils";

// Chain layout of simplehand.glb: Bone (wrist) + five 4-bone chains. The
// glTF loader sanitises "Bone.001" to "Bone001".
const CHAINS = {
  thumb: ["Bone001", "Bone002", "Bone003", "Bone004"],
  index: ["Bone005", "Bone006", "Bone007", "Bone008"],
  middle: ["Bone009", "Bone010", "Bone011", "Bone012"],
  ring: ["Bone013", "Bone014", "Bone015", "Bone016"],
  pinky: ["Bone017", "Bone018", "Bone019", "Bone020"],
} as const;

export type SimplePose = Record<keyof typeof CHAINS, number> & {
  // Lateral fan (radians) per finger, applied at the base joint — negative
  // swings toward the thumb side, positive toward the pinky.
  spread?: Partial<Record<keyof typeof CHAINS, number>>;
};

// Per-joint flexion at full curl — fingers fold ~275° total (a real fist
// folds the tips back onto the palm), the thumb stays gentler. Measured
// against the trained classifier: weaker folds read as extended fingers.
const FINGER_BENDS = [1.5, 1.9, 1.4, 0.5];
const THUMB_BENDS = [0.9, 0.7, 0.5, 0.3];

type Joint = {
  bone: THREE.Bone;
  q0: THREE.Quaternion;
  parentQ: THREE.Quaternion;
  axis: THREE.Vector3;
  maxBend: number;
  isBase: boolean;
};

type Rig = {
  joints: Joint[];
  palmNormal: THREE.Vector3;
  box: THREE.Box3;
};

// Solve the rig once: walk the rest pose, measure every bone's world-space
// direction and derive each joint's anatomical flexion axis (bone direction ×
// palm normal) so fingers fold into the palm regardless of how the artist
// rolled the bones. Flip PALM_SIGN if the fist curls backwards.
const PALM_SIGN = -1;

// The authored rig's finger segments are ~2.8× the palm length (real hands ≈
// 0.95×). Shortening happens on the segment OFFSETS — bone scales would shear
// under rotation (the old stretching bug). Fingers 0.34, thumb 0.4.
const FINGER_SEG_SCALE = 0.34;
const THUMB_SEG_SCALE = 0.4;
// Seats the scaled-down finger bases into the palm so no gap shows between
// palm and fingers.
const ROOT_SINK = 0.002;

function buildRig(clone: THREE.Group): Rig {
  const bones = new Map<string, THREE.Bone>();
  clone.traverse((o) => {
    if ((o as THREE.Bone).isBone) bones.set(o.name, o as THREE.Bone);
  });

  for (const [finger, names] of Object.entries(CHAINS)) {
    const segScale = finger === "thumb" ? THUMB_SEG_SCALE : FINGER_SEG_SCALE;
    const rootBone = bones.get(names[0])!;
    rootBone.position.y -= ROOT_SINK;
    for (let i = 1; i < names.length; i++) {
      bones.get(names[i])!.position.multiplyScalar(segScale);
    }
    // Taper the tip bone — real fingers narrow toward the fingertip.
    bones.get(names[3])?.scale.setScalar(0.8);
  }

  clone.updateMatrixWorld(true);

  const pos = (name: string) =>
    bones.get(name)!.getWorldPosition(new THREE.Vector3());

  const wrist = pos("Bone");
  const along = pos("Bone009").sub(wrist).normalize(); // wrist → middle MCP
  // All five finger roots share one knuckle point, so measure the spread
  // between the fingertips, not the bases.
  const spread = pos("Bone020").sub(pos("Bone008")).normalize(); // index tip → pinky tip
  const palmNormal = new THREE.Vector3().crossVectors(spread, along).normalize().multiplyScalar(PALM_SIGN);

  const joints: Joint[] = [];
  for (const bonesOfFinger of Object.values(CHAINS)) {
    const points = bonesOfFinger.map((n) => pos(n));
    const bends = bonesOfFinger[0] === "Bone001" ? THUMB_BENDS : FINGER_BENDS;
    bonesOfFinger.forEach((name, i) => {
      const bone = bones.get(name)!;
      const dir = points[i + 1]
        ? points[i + 1].clone().sub(points[i])
        : points[i].clone().sub(points[i - 1]);
      dir.normalize();
      // Rotating around (dir × palmNormal) tilts the bone toward the palm —
      // exactly what finger flexion is.
      const axis = new THREE.Vector3().crossVectors(dir, palmNormal).normalize();
      joints.push({
        bone,
        q0: bone.quaternion.clone(),
        parentQ: bone.parent!.getWorldQuaternion(new THREE.Quaternion()),
        axis,
        maxBend: bends[i],
        isBase: i === 0,
      });
    });
  }

  const box = new THREE.Box3();
  const v = new THREE.Vector3();
  bones.forEach((b) => box.expandByPoint(b.getWorldPosition(v)));

  return { joints, palmNormal, box };
}

function HandMesh({
  pose,
  onLandmarks,
}: {
  pose: SimplePose;
  onLandmarks?: (points: [number, number, number][]) => void;
}) {
  const { scene } = useGLTF("/models/simplehand.glb");
  const invalidate = useThree((s) => s.invalidate);
  const clone = useMemo(() => SkeletonUtils.clone(scene) as THREE.Group, [scene]);
  const rig = useMemo(() => buildRig(clone), [clone]);
  const root = useRef<THREE.Group>(null);

  // The FBX export has hard face normals — weld duplicate vertices and
  // rebuild the normals so the surface shades smooth. Keep the authored skin
  // material; only the shading is touched. The forearm stub below the wrist
  // is cut away: palm and fingers only.
  useEffect(() => {
    const boneMap = new Map<string, THREE.Bone>();
    clone.traverse((o) => {
      if ((o as THREE.Bone).isBone) boneMap.set(o.name, o as THREE.Bone);
    });
    clone.updateWorldMatrix(true, true);

    clone.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (!mesh.isMesh) return;
      // The FBX ships hard per-face normals and UV seams — both keep
      // duplicate vertices apart, which would crack when the surface is
      // inflated. Drop them, weld, then rebuild smooth normals.
      const bare = mesh.geometry.clone();
      bare.deleteAttribute("normal");
      bare.deleteAttribute("uv");
      bare.deleteAttribute("uv1");
      bare.deleteAttribute("tangent");
      const geometry = mergeVertices(bare, 1e-4);
      geometry.computeVertexNormals();      // Inflate along the normals: restores finger thickness lost with the
      // uniform shortening scale.
      const position = geometry.attributes.position;
      const normal = geometry.attributes.normal;
      for (let i = 0; i < position.count; i++) {
        position.setXYZ(
          i,
          position.getX(i) + normal.getX(i) * 0.0009,
          position.getY(i) + normal.getY(i) * 0.0009,
          position.getZ(i) + normal.getZ(i) * 0.0009
        );
      }
      position.needsUpdate = true;

      // Cut the wrist: the palm mesh runs long past the knuckles, so keep
      // only the half above the midpoint between knuckle line and wrist bone.
      const knuckle = mesh.worldToLocal(
        boneMap.get("Bone009")!.getWorldPosition(new THREE.Vector3())
      );
      const wrist = mesh.worldToLocal(
        boneMap.get("Bone")!.getWorldPosition(new THREE.Vector3())
      );
      geometry.computeBoundingBox();
      const bbox = geometry.boundingBox!;
      const size = bbox.getSize(new THREE.Vector3());
      const axis: "x" | "y" | "z" =
        size.z >= size.x && size.z >= size.y ? "z" : size.y >= size.x ? "y" : "x";
      const cut = (knuckle[axis] + wrist[axis]) / 2;
      const index = geometry.index;
      if (index) {
        const keep: number[] = [];
        for (let i = 0; i < index.count; i += 3) {
          const a = index.getX(i);
          const b = index.getX(i + 1);
          const c = index.getX(i + 2);
          const centroid =
            (position.getX(a) + position.getX(b) + position.getX(c)) / 3;
          const centroidY =
            (position.getY(a) + position.getY(b) + position.getY(c)) / 3;
          const centroidZ =
            (position.getZ(a) + position.getZ(b) + position.getZ(c)) / 3;
          const along = axis === "x" ? centroid : axis === "y" ? centroidY : centroidZ;
          if ((along - cut) * Math.sign(knuckle[axis] - wrist[axis]) > 0) keep.push(a, b, c);
        }
        geometry.setIndex(keep);
      }
      const material = mesh.material as THREE.MeshStandardMaterial;
      if (material) material.side = THREE.DoubleSide;
      mesh.geometry = geometry;
    });
    invalidate();
  }, [clone, invalidate]);

  // Skinned geometry bounds are pre-skinning (useless), so size the model
  // from its bone world positions instead. Yaw the palm toward the camera.
  useEffect(() => {
    if (!root.current) return;
    const size = rig.box.getSize(new THREE.Vector3());
    const center = rig.box.getCenter(new THREE.Vector3());
    const s = 2.35 / Math.max(size.x, size.y, size.z);
    root.current.scale.setScalar(s);
    root.current.position.set(-center.x * s, -center.y * s, -center.z * s);
    const n = rig.palmNormal;
    root.current.rotation.set(0.06, -Math.atan2(n.x, n.z), 0);
    invalidate();
  }, [rig, invalidate]);

  useEffect(() => {
    // Pose pass: rotate each joint around its solved world-space flexion
    // axis. δ = parentQ⁻¹ · R · parentQ expresses that world rotation in the
    // bone's local frame; q_new = δ · q0.
    const fingerOf = new Map<string, keyof typeof CHAINS>();
    for (const [finger, names] of Object.entries(CHAINS)) {
      names.forEach((n) => fingerOf.set(n, finger as keyof typeof CHAINS));
    }
    // Gentle natural fan so neighbouring fingers never fold into each other;
    // a pose's own spread (V, K…) overrides these defaults. Folds are also
    // staggered per finger — real fists show fingertips at stepped heights
    // instead of one merged blob.
    const fan = { thumb: 0, index: -0.07, middle: 0, ring: 0.06, pinky: 0.14, ...pose.spread };
    const stagger = { thumb: 1, index: 0.96, middle: 1.0, ring: 1.03, pinky: 1.06 };
    for (const j of rig.joints) {
      const finger = fingerOf.get(j.bone.name);
      if (!finger) continue;
      const curl = (pose[finger] ?? 0) * stagger[finger];
      const R = new THREE.Quaternion().setFromAxisAngle(j.axis, curl * j.maxBend);
      const delta = j.parentQ.clone().invert().multiply(R).multiply(j.parentQ);
      if (j.isBase) {
        // Fan the finger laterally around the palm normal before curling.
        const angle = fan[finger] ?? 0;
        if (angle !== 0) {
          const Rs = new THREE.Quaternion().setFromAxisAngle(rig.palmNormal, angle);
          const deltaS = j.parentQ.clone().invert().multiply(Rs).multiply(j.parentQ);
          delta.premultiply(deltaS);
        }
      }
      j.bone.quaternion.copy(delta).multiply(j.q0);
    }
    // Demand mode renders one frame per React commit — the bone writes above
    // happen in an effect, after that frame, so force another one.
    invalidate();

    if (onLandmarks) {
      // MediaPipe landmark order: 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle,
      // 13-16 ring, 17-20 pinky — exactly the rig's chain layout.
      const boneMap = new Map<string, THREE.Bone>();
      clone.traverse((o) => {
        if ((o as THREE.Bone).isBone) boneMap.set(o.name, o as THREE.Bone);
      });
      clone.updateWorldMatrix(true, true);
      const v = new THREE.Vector3();
      const points = ["Bone", ...Object.values(CHAINS).flat()].map((name) => {
        const b = boneMap.get(name);
        b!.getWorldPosition(v);
        return [v.x, v.y, v.z] as [number, number, number];
      });
      onLandmarks(points);
    }
  }, [rig, pose, invalidate, onLandmarks, clone]);

  return (
    <group ref={root}>
      <primitive object={clone} />
    </group>
  );
}

export function SimpleHand({
  pose,
  className,
  onLandmarks,
}: {
  pose: SimplePose;
  className?: string;
  onLandmarks?: (points: [number, number, number][]) => void;
}) {
  return (
    <div className={cn("relative", className)}>
      <Canvas
        frameloop="demand"
        camera={{ position: [0, 0.1, 5.4], fov: 35 }}
        style={{ background: "transparent" }}
      >
        <ambientLight intensity={1.5} />
        <directionalLight position={[3, 4, 5]} intensity={1.6} />
        <directionalLight position={[-4, -2, 3]} intensity={0.5} color="#b9a8ff" />
        <Suspense fallback={null}>
          <HandMesh pose={pose} onLandmarks={onLandmarks} />
        </Suspense>
      </Canvas>
    </div>
  );
}

useGLTF.preload("/models/simplehand.glb");
