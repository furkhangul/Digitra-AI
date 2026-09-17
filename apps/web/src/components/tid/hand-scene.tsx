"use client";

import { Component, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";
import type { ShaderMaterial } from "three";
import { TID_ALPHABET, type Letter } from "./poses";
import { HandPlayer, serialiseFrame } from "./motion";
import { createHandUniforms, updateHandUniforms, contentExtent, handFragmentShader, handVertexShader, type HandUniforms } from "./rounded-hand";
import { usePrefersReducedMotion } from "@/lib/use-reduced-motion";

type Props = { letter: Letter; playing: boolean; speed: number; replay: number; debug?: boolean; hideLetter?: boolean };

function Hands(props: Props & { active: boolean; reducedMotion: boolean }) {
  const uniforms = useMemo(() => createHandUniforms(), []);
  const material = useRef<ShaderMaterial>(null);
  const frameTimes = useRef<number[]>([]);
  const player = useRef<HandPlayer | null>(null);
  const previous = useRef({ letter: props.letter.ch, replay: props.replay });
  const zoom = useRef(1.8);
  const elevation = useRef(props.letter.viewElevation ?? 0);
  const azimuth = useRef(props.letter.viewAzimuth ?? 0);
  const invalidate = useThree(s => s.invalidate);
  useEffect(() => {
    if (!player.current) player.current = new HandPlayer(props.letter);
    if (previous.current.letter !== props.letter.ch || previous.current.replay !== props.replay) {
      player.current.select(props.letter, previous.current.letter === props.letter.ch && previous.current.replay !== props.replay);
      previous.current = { letter: props.letter.ch, replay: props.replay };
    }
    invalidate();
  }, [props.letter, props.replay, props.playing, props.speed, props.active, props.reducedMotion, invalidate]);
  useEffect(() => () => { player.current?.dispose(); player.current = null; }, []);

  useFrame((state, dt) => {
    const p = player.current;
    if (!p || !material.current) return;
    const values = material.current.uniforms as HandUniforms;
    const moving = p.tick(dt, props.playing && props.active, props.speed, props.reducedMotion);
    const targetElevation = props.letter.viewElevation ?? 0;
    elevation.current = props.reducedMotion ? targetElevation
      : THREE.MathUtils.damp(elevation.current, targetElevation, 6, dt);
    const targetAzimuth = props.letter.viewAzimuth ?? 0;
    azimuth.current = props.reducedMotion ? targetAzimuth
      : THREE.MathUtils.damp(azimuth.current, targetAzimuth, 6, dt);
    const viewMoving = Math.abs(elevation.current - targetElevation) > .01
      || Math.abs(azimuth.current - targetAzimuth) > .01;
    if (!viewMoving) { elevation.current = targetElevation; azimuth.current = targetAzimuth; }
    updateHandUniforms(values, p.current, elevation.current, azimuth.current);
    const aspect = state.size.width / state.size.height;
    values.uAspect.value = aspect;
    // Frame both hands: widen at once when a sign would clip, close in gently
    // afterwards, so nothing leaves the frame and the view never pops.
    const extent = contentExtent(values);
    const needed = Math.max(1.15, extent.y, extent.x / Math.max(aspect, .35)) * 1.05;
    const scale = zoom.current;
    zoom.current = needed > scale
      ? THREE.MathUtils.damp(scale, needed, 14, dt)
      : THREE.MathUtils.damp(scale, needed, 1.6, dt);
    values.uScale.value = zoom.current;
    if (props.debug) {
      const canvas = state.gl.domElement;
      canvas.setAttribute("data-letter", props.letter.ch);
      canvas.setAttribute("data-settled", String(p.settled));
      canvas.setAttribute("data-phase", p.phase);
      canvas.setAttribute("data-pose", JSON.stringify(serialiseFrame(p.current)));
      canvas.setAttribute("data-renderer", "rounded-front");
      if (moving && props.active && props.playing && dt < .1) {
        frameTimes.current.push(dt * 1000);
        if (frameTimes.current.length > 90) frameTimes.current.shift();
        canvas.setAttribute("data-frame-ms", String(frameTimes.current.reduce((a, b) => a + b, 0) / frameTimes.current.length));
      }
    }
    // Keep drawing while the framing is still settling, not only while the hands move.
    if (props.active && ((moving && props.playing) || viewMoving || Math.abs(needed - zoom.current) > 2e-3)) invalidate();
  });
  return <mesh frustumCulled={false}>
    <planeGeometry args={[2, 2]} />
    <shaderMaterial ref={material} uniforms={uniforms} vertexShader={handVertexShader} fragmentShader={handFragmentShader}
      transparent depthTest={false} depthWrite={false} />
  </mesh>;
}

class SceneBoundary extends Component<{ children: ReactNode }, { failed: boolean }> {
  state = { failed: false };
  static getDerivedStateFromError() { return { failed: true }; }
  render() {
    return this.state.failed ? <p role="alert" className="flex h-full items-center justify-center p-8 text-center text-sm text-white/75">El gösterimi yüklenemedi. Sayfayı yenileyerek tekrar deneyebilirsin.</p> : this.props.children;
  }
}

export default function HandScene(props: Props) {
  const container = useRef<HTMLDivElement>(null);
  const [active, setActive] = useState(true);
  const reducedMotion = usePrefersReducedMotion();
  useEffect(() => {
    let visible = true;
    const update = () => setActive(visible && !document.hidden);
    const observer = new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; update(); }, { rootMargin: "80px" });
    if (container.current) observer.observe(container.current);
    document.addEventListener("visibilitychange", update);
    return () => { observer.disconnect(); document.removeEventListener("visibilitychange", update); };
  }, []);
  return <div ref={container} className="absolute inset-0" role="img" aria-label={props.hideLetter ? "Tahmin edilecek hareketli TİD harfi" : `${props.letter.ch} harfinin önden hareketli TİD gösterimi`}>
    <SceneBoundary>
      <Canvas frameloop="demand" dpr={[1, 1.25]} orthographic camera={{ position: [0, 0, 1] }}
        gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
        fallback={<p className="p-8 text-white">Bu cihaz el animasyonunu desteklemiyor.</p>}>
        <Hands {...props} active={active} reducedMotion={reducedMotion} />
      </Canvas>
    </SceneBoundary>
  </div>;
}

export { TID_ALPHABET };
