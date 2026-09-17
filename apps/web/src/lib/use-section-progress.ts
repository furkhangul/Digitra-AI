"use client";

import { useEffect, useState, type RefObject } from "react";

/**
 * 0 → 1 scroll progress across a tall section's pinned (sticky) stretch:
 * 0 when the section's top reaches the viewport top, 1 when its bottom
 * reaches the viewport bottom. Plain scroll listener — no motion values.
 */
export function useSectionProgress(ref: RefObject<HTMLElement | null>, enabled = true) {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (!enabled) return;
    const onScroll = () => {
      const el = ref.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const total = rect.height - window.innerHeight;
      setProgress(total > 0 ? Math.min(1, Math.max(0, -rect.top / total)) : 1);
    };
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
    };
  }, [ref, enabled]);

  return progress;
}
