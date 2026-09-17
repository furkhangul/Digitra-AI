"use client";

import { Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { SimpleHand } from "@/components/home/simple-hand";
import { ALPHABET } from "@/components/home/legacy-alphabet";

function Dump() {
  const params = useSearchParams();
  const letter = params.get("letter") ?? "A";
  const entry = ALPHABET.find((l) => l.ch === letter) ?? ALPHABET[0];

  return (
    <div style={{ background: "#111", color: "#fff", minHeight: "100vh", padding: 24 }}>
      <h1 style={{ fontSize: 18 }}>pose dump: {entry.ch}</h1>
      <SimpleHand
        pose={entry.pose}
        className="h-96 w-96"
        onLandmarks={(points) => {
          (window as unknown as { __LANDMARKS__: unknown }).__LANDMARKS__ = points;
        }}
      />
    </div>
  );
}

export default function DebugPoses() {
  return (
    <Suspense fallback={null}>
      <Dump />
    </Suspense>
  );
}
