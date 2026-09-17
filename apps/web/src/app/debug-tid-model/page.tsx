"use client";

// Dev-only check that the exported tid-hand.glb loads, skins and animates in the
// browser. The education view does not use this file — it raymarches the same
// shape analytically — so this page is what keeps the exported asset honest.

import { Suspense, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, useGLTF } from "@react-three/drei";
import * as THREE from "three";

const MODEL_URL = "/models/tid-hand.glb";

type Report = {
  meshes: number; vertices: number; triangles: number;
  bones: string[]; animations: string[]; skinned: boolean;
};

function Model({ clip, onReport }: { clip: string; onReport: (r: Report) => void }) {
  const { scene, animations } = useGLTF(MODEL_URL);
  const mixer = useMemo(() => new THREE.AnimationMixer(scene), [scene]);
  const reported = useRef(false);

  useEffect(() => {
    if (reported.current) return;
    reported.current = true;
    let vertices = 0, triangles = 0, meshes = 0, skinned = false;
    const bones: string[] = [];
    scene.traverse((o) => {
      if ((o as THREE.SkinnedMesh).isSkinnedMesh) skinned = true;
      if ((o as THREE.Mesh).isMesh) {
        meshes++;
        const g = (o as THREE.Mesh).geometry;
        vertices += g.attributes.position.count;
        triangles += (g.index ? g.index.count : g.attributes.position.count) / 3;
      }
      if ((o as THREE.Bone).isBone) bones.push(o.name);
    });
    onReport({ meshes, vertices, triangles, bones, skinned, animations: animations.map(a => a.name) });
  }, [scene, animations, onReport]);

  useEffect(() => {
    mixer.stopAllAction();
    const found = animations.find(a => a.name === clip);
    if (found) mixer.clipAction(found).reset().play();
    return () => { mixer.stopAllAction(); };
  }, [clip, animations, mixer]);

  useFrame((_, dt) => mixer.update(dt));
  return <primitive object={scene} />;
}

export default function DebugTidModel() {
  const [report, setReport] = useState<Report | null>(null);
  const [clip, setClip] = useState("");
  return (
    <main className="min-h-screen bg-neutral-950 p-6 text-neutral-100">
      <h1 className="mb-4 text-lg font-semibold">tid-hand.glb — browser load check</h1>
      <div className="grid gap-6 lg:grid-cols-[420px_1fr]">
        <div className="h-[420px] rounded-xl bg-neutral-900">
          <Canvas camera={{ position: [0, 1.1, 3.4], fov: 35 }}>
            <ambientLight intensity={0.7} />
            <directionalLight position={[2, 4, 3]} intensity={2.2} />
            <directionalLight position={[-3, 1, -2]} intensity={0.6} />
            <Suspense fallback={null}>
              <Model clip={clip} onReport={setReport} />
            </Suspense>
            <OrbitControls target={[0, 0.9, 0]} />
          </Canvas>
        </div>
        <div>
          <pre data-report className="max-h-[420px] overflow-auto rounded-xl bg-neutral-900 p-4 text-xs">
            {report ? JSON.stringify({ ...report, animations: report.animations.length }, null, 2) : "loading…"}
          </pre>
          <div className="mt-3 flex flex-wrap gap-1">
            {report?.animations.map(a => (
              <button key={a} onClick={() => setClip(a)}
                className={`rounded px-2 py-1 text-xs ${clip === a ? "bg-violet-500" : "bg-neutral-800"}`}>
                {a}
              </button>
            ))}
          </div>
        </div>
      </div>
    </main>
  );
}
