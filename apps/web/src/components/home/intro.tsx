"use client";

import { useRef, useState, type CSSProperties } from "react";
import dynamic from "next/dynamic";
import { motion, useMotionValue, useMotionValueEvent, useScroll, useTransform } from "framer-motion";
import { POINT } from "./hand-poses";
import { LogoMark } from "@/components/site/logo";
import { usePrefersReducedMotion } from "@/lib/use-reduced-motion";
import { useMediaQuery } from "@/lib/use-media-query";

const HeroScene = dynamic(() => import("./hero-scene").then((m) => m.HeroScene), {
  ssr: false,
});

/** Map `v` from the [a, b] window onto [from, to], clamped outside it. */
function ramp(v: number, a: number, b: number, from: number, to: number) {
  const t = Math.min(1, Math.max(0, (v - a) / (b - a)));
  return from + (to - from) * t;
}

/**
 * Opening panel that doesn't scroll away — it turns into the next section.
 *
 * A disc sitting over the hand is painted in the page background colour and
 * scaled up on scroll until it covers the viewport, so the reveal reads as
 * travelling *through* the hand rather than sliding past it. The tall outer
 * section supplies the scroll distance; the inner panel is sticky.
 *
 * The disc is deliberately the LAST child: a sticky panel stays partly on
 * screen while the tail of its section scrolls by, so the disc covering
 * everything above it is what guarantees no intro content bleeds into the
 * sections below — it does not depend on the fades landing.
 */
export function Intro() {
  const ref = useRef<HTMLDivElement>(null);
  const chromeRef = useRef<HTMLDivElement>(null);
  const handRef = useRef<HTMLDivElement>(null);
  const portalRef = useRef<HTMLDivElement>(null);
  const fogRef = useRef<HTMLDivElement>(null);
  const reducedMotion = usePrefersReducedMotion();
  const isDesktop = useMediaQuery("(min-width: 1024px)");

  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start start", "end end"],
  });

  // Ordering matters: centre, turn to point at the viewer, come at the camera,
  // and only then let the fingertip open into the next screen.
  //
  // The approach is a real 3D dolly rather than a CSS scale on the canvas: a
  // CSS scale enlarges the rendered image without perspective, and it also
  // decouples the canvas from the projected fingertip position the dot is
  // pinned to.
  //
  // These are plain motion values written by hand rather than useTransform
  // outputs. A derived value that is never bound to a motion component's
  // `style` has no subscriber and is never recomputed here, so reading it with
  // `.get()` from the render loop returns its initial value forever — measured:
  // scroll progress 0.79 with morph/pitch/dolly all still 0.
  const handMorph = useMotionValue(0);
  const handPitch = useMotionValue(0);
  const handRecenter = useMotionValue(0);
  const handDolly = useMotionValue(0);
  const handSettle = useMotionValue(0);

  // Fingertip position in viewport percentages, written by the scene.
  const tipX = useMotionValue(50);
  const tipY = useMotionValue(50);
  const discLeft = useTransform(tipX, (v) => `${v}%`);
  const discTop = useTransform(tipY, (v) => `${v}%`);

  // The emergence surface takes its X from the hand (so it stays under it) but
  // its Y from the mask's own fade line — the hand's projected centre sits near
  // the top of frame because the bounding box includes the forearm, which is
  // nowhere near where the hand visually dissolves.
  const wristX = useMotionValue(50);
  const wristY = useMotionValue(50);
  const portalY = useMotionValue(52);
  const portalLeft = useTransform(wristX, (v) => `${v}%`);
  const portalTop = useTransform(portalY, (v) => `${v}%`);

  // Held small so it reads as a dot on the fingertip, then blown out fast.
  const discScale = useTransform(
    scrollYProgress,
    [0.72, 0.84, 0.94, 1],
    [0, 0.5, 3, 34]
  );

  // On desktop the hand is parked in the right half so the headline gets its
  // own space. It has to travel back to the middle before it grows, or scaling
  // about a point at 75% width just pushes it off the right edge — which read
  // as the hand sliding away instead of coming at you.
  const handX = useTransform(
    scrollYProgress,
    [0, 0.28],
    [isDesktop ? "25%" : "0%", "0%"]
  );

  // Opacity is driven imperatively too: passing it alongside `scale` in the
  // same style object left it pinned at 1 while the transform updated fine.
  useMotionValueEvent(scrollYProgress, "change", (v) => {
    if (reducedMotion) return;

    // Blend into the pointing gesture, then turn the wrist so the extended
    // index aims out of the screen at the viewer. Positive pitch rotates the
    // fingers from +Y toward +Z, which is where the camera sits; negative
    // sends them into the page instead. It stops short of square-on — past
    // about 75° the whole hand is seen end-on and turns into an unreadable
    // cluster of stumps.
    handMorph.set(ramp(v, 0.16, 0.46, 0, 1));
    handPitch.set(ramp(v, 0.2, 0.58, 0, 1.3));
    // Centre before approaching, but only part of the way: pitching forward
    // swings the fingers up and the wrist down, so going the full distance
    // drops the hand out of the bottom of the frame.
    handRecenter.set(ramp(v, 0.24, 0.5, 0, 0.45));
    handDolly.set(ramp(v, 0.5, 0.92, 0, 1.6));
    // Stop the camera drifting before the finger turns, or it is seen from an
    // angle and reads as tilting away rather than coming at you.
    handSettle.set(ramp(v, 0.12, 0.3, 0, 1));

    if (chromeRef.current) {
      // Hold the message readable while the hand travels to centre, then clear
      // it out of the way before the disc opens.
      chromeRef.current.style.opacity = String(
        Math.max(0, 1 - Math.max(0, v - 0.12) / 0.28)
      );
    }
    if (handRef.current) {
      // Stays visible right up to the blow-out so the dot reads as sitting on
      // the fingertip rather than replacing the hand.
      handRef.current.style.opacity = String(Math.max(0, 1 - Math.max(0, v - 0.88) / 0.1));

      // The wrist fade is framing for the resting pose only. Once the hand
      // recentres and turns it sits lower and the same cut slices through the
      // fingers, so retract the fade off the bottom of the frame.
    }

    // Rides along until the very end, so the surface is still there while the
    // fingertip reaches the viewer, then clears with the hand.
    const portalFade = Math.max(0, 1 - Math.max(0, v - 0.86) / 0.1);
    if (portalRef.current) portalRef.current.style.opacity = String(portalFade);
    if (fogRef.current) {
      // Thins out as the hand turns and comes forward: at full strength it is
      // sized to swallow a forearm, and once the hand is centred and pitched
      // that same cloud starts eating the palm instead.
      fogRef.current.style.opacity = String(portalFade * (1 - ramp(v, 0.15, 0.5, 0, 0.7)));
    }
  });

  // Both the fade and the surface follow the projected forearm, so the hand
  // dissolves exactly where it comes through for the whole sequence — driving
  // them off scroll progress instead left the effect behind the moment the
  // hand started moving.
  useMotionValueEvent(wristY, "change", (wy) => {
    if (reducedMotion) return;
    // The mist over the wrist does the hiding now, so the alpha cut only has to
    // deal with whatever forearm sticks out below it — and it starts well below
    // the wrist, keeping the hand solid where the ripples pass so they stay
    // behind it instead of showing through.
    const cut = Math.max(0, Math.min(96, wy + 13));
    const gradient = `linear-gradient(to bottom, black ${cut}%, transparent ${Math.min(100, cut + 10)}%)`;
    if (handRef.current) {
      handRef.current.style.maskImage = gradient;
      handRef.current.style.webkitMaskImage = gradient;
    }
    portalY.set(wy + 3);
  });

  // The portal's fallback coordinates are the centre of frame, so until the
  // scene reports the real projected wrist the rings would sit mid-screen on
  // load. Keep them unmounted until that first report lands.
  const [portalPinned, setPortalPinned] = useState(false);
  useMotionValueEvent(wristY, "change", () => {
    setPortalPinned(true);
  });

  return (
    <section
      ref={ref}
      className={reducedMotion ? "relative" : "relative h-[240vh]"}
      aria-label="Digitra tanıtım"
    >
      <div className="sticky top-0 h-[100svh] overflow-hidden">
        {/* Brand field. One deep, committed colour rather than a full-spectrum
            sweep — the sweep read as washed out at this size, and the flatter
            field makes the cut to the page background land harder. */}
        <div
          className="absolute inset-0"
          style={{
            background:
              "radial-gradient(120% 110% at 50% 18%, #453c94 0%, #2b2470 45%, #0d0a33 100%)",
          }}
        />
        {/* Galaxy backdrop: nebulae, a tilted milky-way band, two drifting
            starfields and the occasional shooting star — pure backdrop, sits
            under the noise, hand and portal. */}
        <div className="pointer-events-none absolute inset-0 overflow-hidden">
          <div className="nebula nebula-a absolute" />
          <div className="nebula nebula-b absolute" />
          <div className="galaxy-band absolute" />
          <div className="star-field star-field-a absolute" />
          <div className="star-field star-field-b absolute" />
          {/* Negative delays start each star mid-cycle, so streaks begin the
              moment the page opens and recur every few seconds. */}
          <span
            className="shooting-star absolute"
            style={{ top: "16%", right: "8%", animationDelay: "-11.2s", animationDuration: "13s", "--angle": "-26deg" } as CSSProperties}
          />
          <span
            className="shooting-star absolute"
            style={{ top: "44%", right: "-4%", animationDelay: "-13.6s", animationDuration: "17s", "--angle": "-33deg" } as CSSProperties}
          />
          <span
            className="shooting-star absolute"
            style={{ top: "5%", right: "30%", animationDelay: "-9s", animationDuration: "15s", "--angle": "-21deg" } as CSSProperties}
          />
          <span
            className="shooting-star absolute"
            style={{ top: "60%", right: "18%", animationDelay: "-5.5s", animationDuration: "19s", "--angle": "-28deg" } as CSSProperties}
          />
        </div>
        <div className="noise-overlay absolute inset-0" />

        {/* Hand. Given its own half of the screen on desktop: laying the
            headline over it needed a scrim heavy enough to turn the hand into
            a ghost, and the two fought instead of composing. Below lg there
            isn't room to split, so it sits behind the text with a light scrim. */}
        {/* Both the canvas and the portal live inside this one translated
            wrapper. The portal is positioned from coordinates the scene
            projects, so it has to share every transform the canvas gets or the
            two drift apart. */}
        <motion.div
          style={reducedMotion ? undefined : { x: handX }}
          className="absolute inset-0"
        >
          {/* The surface the hand comes through, pinned to the hand's own
              centre so it travels with it for the whole sequence. Sits under
              the canvas so the hand reads as passing out through it.
              Unmounted until the scene pins it to the real wrist — its
              fallback coordinates would park it mid-screen on load. */}
          {portalPinned && (
            <motion.div
              ref={portalRef}
              style={{ left: portalLeft, top: portalTop, x: "-50%", y: "-50%" }}
              className="pointer-events-none absolute h-[28vmin] w-[28vmin]"
            >
              <span className="portal-glow" />
              <span className="portal-ring" />
              <span className="portal-ring" style={{ animationDelay: "1.3s" }} />
              <span className="portal-ring" style={{ animationDelay: "2.6s" }} />
            </motion.div>
          )}

          <motion.div
            ref={handRef}
            style={{
              // The rig ends in a forearm stub that would otherwise stop dead
              // in mid-air. A long, soft fade dissolves it into the portal
              // instead, so the hand looks like it is coming through rather
              // than being cut off.
              maskImage: "linear-gradient(to bottom, black 75%, transparent 85%)",
              WebkitMaskImage: "linear-gradient(to bottom, black 75%, transparent 85%)",
            }}
            className="absolute inset-0 cursor-grab active:cursor-grabbing"
          >
            <HeroScene
              idleArc={0.4}
              morphTo={POINT}
              morph={handMorph}
              pitch={handPitch}
              dolly={handDolly}
              recenter={handRecenter}
              tipX={tipX}
              tipY={tipY}
              wristX={wristX}
              wristY={wristY}
              settle={handSettle}
            />
          </motion.div>

          {/* Mist painted over the wrist, after the hand so it covers it. */}
          {portalPinned && (
            <motion.div
              ref={fogRef}
              style={{ left: portalLeft, top: portalTop, x: "-50%", y: "-50%" }}
              className="pointer-events-none absolute h-[28vmin] w-[28vmin]"
            >
              <span className="portal-fog" />
            </motion.div>
          )}
        </motion.div>

        <div ref={chromeRef} className="absolute inset-0">
          {/* Only needed where the text sits on top of the hand. */}
          <div
            className="pointer-events-none absolute inset-0 lg:hidden"
            style={{
              background:
                "radial-gradient(70% 45% at 50% 50%, rgba(10,6,40,0.75) 0%, rgba(10,6,40,0.4) 60%, transparent 100%)",
            }}
          />

          <div className="pointer-events-none relative mx-auto flex h-full w-full max-w-7xl flex-col justify-center px-6 text-center sm:px-10 lg:w-1/2 lg:pr-0 lg:text-left">
            <motion.h1
              initial={{ opacity: 0, y: 24 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.9, ease: [0.16, 1, 0.3, 1] }}
              className="text-5xl leading-[1.02] font-semibold tracking-tight text-balance text-white sm:text-7xl lg:text-[5rem]"
            >
              Ellerle öğren.
              <br />
              <span className="text-white/65">Harflerle anlat.</span>
            </motion.h1>

            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.9, delay: 0.15, ease: [0.16, 1, 0.3, 1] }}
              className="mx-auto mt-7 max-w-xl text-base leading-relaxed text-white/75 sm:text-lg lg:mx-0"
            >
              Türk İşaret Dili parmak alfabesini 3D ellerle keşfet.
              Kameranla pratik yap, tanınan harfleri metne ve sese dönüştür.
            </motion.p>
          </div>

          {/* Corner anchors. */}
          <button
            type="button"
            onClick={() => window.scrollTo({ top: 0, behavior: reducedMotion ? "auto" : "smooth" })}
            className="pointer-events-auto absolute top-6 left-6 flex cursor-pointer items-center gap-2 text-lg font-semibold tracking-tight text-white sm:top-10 sm:left-10"
            aria-label="Sayfanın başına dön"
          >
            <LogoMark className="h-8 w-8" />
            Digitra
          </button>

          <div className="pointer-events-none absolute bottom-6 left-1/2 hidden -translate-x-1/2 flex-col items-center gap-2 text-xs text-white/70 lg:flex">
            <span>Digitra’yı keşfet</span>
            <motion.span
              animate={{ y: [0, 8, 0] }}
              transition={{ duration: 1.8, repeat: Infinity, ease: "easeInOut" }}
              className="block h-8 w-5 rounded-full border border-white/40 p-1"
            >
              <span className="block h-1.5 w-1.5 rounded-full bg-white" />
            </motion.span>
          </div>
        </div>

        {/* The disc that becomes the next screen — topmost on purpose, and
            centred on the hand so the reveal grows out of it. */}
        <motion.div
          style={
            reducedMotion
              ? { scale: 0 }
              : { left: discLeft, top: discTop, x: "-50%", y: "-50%", scale: discScale }
          }
          className="pointer-events-none absolute h-[10vmin] w-[10vmin] rounded-full bg-background"
        />
      </div>
    </section>
  );
}
