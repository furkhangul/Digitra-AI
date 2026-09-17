"use client";

import { useRef } from "react";
import { motion } from "framer-motion";
import { ArrowRight, Play } from "lucide-react";
import { Container } from "@/components/ui/container";
import { Magnetic } from "@/components/ui/magnetic";
import { Reveal } from "@/components/ui/reveal";
import { usePrefersReducedMotion } from "@/lib/use-reduced-motion";
import { useSectionProgress } from "@/lib/use-section-progress";

// The positioning line carries the whole "what is this" job, so each domain
// gets its own accent — the sentence doubles as the capability list.
const DOMAINS = [
  { label: "3D eller", color: "var(--primary)" },
  { label: "canlı pratik", color: "var(--secondary)" },
  { label: "harf tanıma", color: "var(--primary-2)" },
  { label: "sesli anlatım", color: "var(--warning)" },
];

// Where each word's reveal sits inside the pinned section's scroll progress:
// four staggered thresholds spread across the middle of the pin.
const REVEAL_START = 0.12;
const REVEAL_STEP = 0.17;

/**
 * One coloured capability word. Hidden until the pin's scroll progress
 * crosses the word's threshold, then a CSS transition fades it in and
 * rises it the rest of the way. The trailing punctuation rides along so a
 * lone comma is never left behind, while the explicit space between
 * phrases stays in the parent's flow — edge whitespace inside an
 * inline-block would collapse and glue phrases together.
 */
function ScrollWord({
  progress,
  index,
  separator,
  color,
  children,
  reducedMotion,
}: {
  progress: number;
  index: number;
  separator: string;
  color: string;
  children: string;
  reducedMotion: boolean;
}) {
  const shown = reducedMotion || progress >= REVEAL_START + index * REVEAL_STEP;

  return (
    <span>
      <span
        className={reducedMotion ? undefined : "inline-block transition-all duration-500 ease-out"}
        style={{
          color,
          opacity: shown ? 1 : 0,
          transform: shown ? undefined : "translateY(14px)",
        }}
      >
        {children}
        {separator}
      </span>
      {index < DOMAINS.length - 1 ? " " : ""}
    </span>
  );
}

export function Hero() {
  const ref = useRef<HTMLElement>(null);
  const reducedMotion = usePrefersReducedMotion();

  // The section is twice the viewport tall and its content sticks to the top:
  // progress 0 → 1 covers exactly the pinned stretch, so the words reveal
  // while the page holds still and the section releases when they are done.
  const progress = useSectionProgress(ref, !reducedMotion);

  return (
    <section
      ref={ref}
      className={
        reducedMotion
          ? "relative isolate overflow-hidden py-24 sm:py-32"
          : "relative isolate h-[200vh]"
      }
    >
      <div
        className={
          reducedMotion
            ? "relative"
            : "sticky top-0 flex min-h-[100svh] flex-col justify-center overflow-hidden py-24"
        }
      >
        <div
          className="animate-blob-1 pointer-events-none absolute -top-32 left-1/2 -z-10 h-[520px] w-[820px] -translate-x-1/2 rounded-full opacity-25 blur-3xl"
          style={{ background: "var(--grad-primary)" }}
        />

        <Container className="flex flex-col items-center text-center">
          <Reveal>
            <h1 className="max-w-4xl text-4xl leading-[1.15] font-semibold tracking-tight text-balance sm:text-5xl lg:text-6xl">
              Parmak alfabesiyle ilk adım:{" "}
              {DOMAINS.map((d, i) => (
                <ScrollWord
                  key={d.label}
                  progress={progress}
                  index={i}
                  separator={i < DOMAINS.length - 2 ? ", " : i === DOMAINS.length - 2 ? " ve " : "."}
                  color={d.color}
                  reducedMotion={reducedMotion}
                >
                  {d.label}
                </ScrollWord>
              ))}
            </h1>
          </Reveal>

          <Reveal delay={0.1}>
            <p className="mt-8 max-w-2xl text-lg leading-relaxed text-muted">
              TİD parmak alfabesini 3D ellerle keşfet, kamerayla pratik yap.
              Harfleri kendi hızında öğren, kelimeleri adım adım tamamla ve
              oluşturduğun metni sesli dinle.
            </p>
          </Reveal>

          <Reveal delay={0.18}>
            <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
              <Magnetic strength={0.4}>
                <a
                  href="#ceviri"
                  className="group inline-flex items-center gap-2 rounded-full bg-foreground px-7 py-4 text-sm font-medium text-background shadow-lg transition-transform duration-300 hover:scale-[1.03]"
                >
                  Harf tanımayı dene
                  <ArrowRight size={16} className="transition-transform duration-300 group-hover:translate-x-1" />
                </a>
              </Magnetic>
              <Magnetic strength={0.4}>
                <a
                  href="#how-it-works"
                  className="glass group inline-flex items-center gap-2 rounded-full px-7 py-4 text-sm font-medium transition-transform duration-300 hover:scale-[1.03]"
                >
                  <Play size={14} className="fill-current" />
                  Neden Digitra?
                </a>
              </Magnetic>
            </div>
          </Reveal>

          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8, delay: 0.3 }}
            className="mt-16 flex flex-wrap items-center justify-center gap-x-8 gap-y-3 text-xs text-muted"
          >
            <span>29 harf · TİD parmak alfabesi</span>
            <span className="h-1 w-1 rounded-full bg-border" />
            <span>Hareketli 3D el modelleri</span>
            <span className="h-1 w-1 rounded-full bg-border" />
            <span>Kamerayla adım adım pratik</span>
          </motion.div>
        </Container>
      </div>
    </section>
  );
}
