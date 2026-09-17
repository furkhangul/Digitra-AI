"use client";

import { useEffect, useRef, useState } from "react";
import { SimpleHand } from "@/components/home/simple-hand";
import { ALPHABET } from "@/components/home/legacy-alphabet";

/**
 * Renders every alphabet pose in sequence and collects each one's 21 rig
 * landmarks into window.__ALL_LANDMARKS__, for the classifier verification
 * loop. One page load instead of one per letter.
 */
export default function DumpAll() {
  const [index, setIndex] = useState(0);
  const results = useRef<Record<string, [number, number, number][]>>({});
  const done = useRef<Set<string>>(new Set());

  useEffect(() => {
    if (index >= ALPHABET.length) {
      (window as unknown as { __ALL_DONE__: boolean }).__ALL_DONE__ = true;
      (window as unknown as { __ALL_LANDMARKS__: unknown }).__ALL_LANDMARKS__ =
        results.current;
    }
  }, [index]);

  const entry = ALPHABET[Math.min(index, ALPHABET.length - 1)];

  return (
    <div style={{ background: "#111", color: "#fff", minHeight: "100vh", padding: 24 }}>
      <h1 style={{ fontSize: 16 }}>
        dump-all {index + 1}/{ALPHABET.length}: {entry.ch}
      </h1>
      <SimpleHand
        pose={entry.pose}
        className="h-96 w-96"
        onLandmarks={(points) => {
          if (done.current.has(entry.ch)) return;
          done.current.add(entry.ch);
          results.current[entry.ch] = points;
          if (done.current.size >= ALPHABET.length) {
            (window as unknown as { __ALL_DONE__: boolean }).__ALL_DONE__ = true;
            (window as unknown as { __ALL_LANDMARKS__: unknown }).__ALL_LANDMARKS__ =
              results.current;
          } else {
            setTimeout(() => setIndex((i) => Math.min(i + 1, ALPHABET.length)), 60);
          }
        }}
      />
    </div>
  );
}
