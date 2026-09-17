"use client";

import { useEffect, useRef, useState } from "react";
import { Check, Clock3, RotateCcw, SkipForward } from "lucide-react";
import { LiveRecognition } from "@/components/home/live-recognition";
import { cn } from "@/lib/utils";
import { TID_ALPHABET } from "@/components/tid/poses";
import dynamic from "next/dynamic";

const HandScene = dynamic(() => import("@/components/tid/hand-scene"), {
  ssr: false,
  loading: () => <div className="absolute inset-0 flex items-center justify-center text-sm text-white/60">Eller hazırlanıyor…</div>,
});

const PRACTICE_WORDS = ["AYŞE", "ANNE", "BABA", "OKUL", "MERHABA", "DİJİTRA"] as const;

type Feedback = "idle" | "correct" | "wrong" | "complete";

export function LiveWordTraining() {
  const [wordIndex, setWordIndex] = useState(0);
  const [letterIndex, setLetterIndex] = useState(0);
  const [feedback, setFeedback] = useState<Feedback>("idle");
  const [score, setScore] = useState(0);
  const [correctLetters, setCorrectLetters] = useState(0);
  const [attempts, setAttempts] = useState(0);
  const [dominantHand, setDominantHand] = useState<"left" | "right">("right");
  const [cameraRunning, setCameraRunning] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const advanceRef = useRef(false);
  const timerRef = useRef<number | null>(null);

  const word = PRACTICE_WORDS[wordIndex];
  const letters = [...word];
  const expected = letters[letterIndex] ?? letters[0];
  const reference = TID_ALPHABET.find((letter) => letter.ch === expected) ?? TID_ALPHABET[0];

  const reset = () => {
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    timerRef.current = null;
    advanceRef.current = false;
    setLetterIndex(0);
    setFeedback("idle");
  };

  const handleLetter = (letter: string) => {
    if (advanceRef.current) return;
    const normalized = letter.toLocaleUpperCase("tr-TR");
    setAttempts((value) => value + 1);
    if (normalized !== expected) {
      setFeedback("wrong");
      return;
    }

    advanceRef.current = true;
    setCorrectLetters((value) => value + 1);
    setFeedback("correct");
    timerRef.current = window.setTimeout(() => {
      timerRef.current = null;
      const isLastLetter = letterIndex >= letters.length - 1;
      if (!isLastLetter) {
        setLetterIndex((value) => value + 1);
        setFeedback("idle");
        advanceRef.current = false;
        return;
      }

      setScore((value) => value + 1);
      setFeedback("complete");
      timerRef.current = window.setTimeout(() => {
        setWordIndex((value) => (value + 1) % PRACTICE_WORDS.length);
        setLetterIndex(0);
        setFeedback("idle");
        advanceRef.current = false;
        timerRef.current = null;
      }, 900);
    }, 650);
  };

  useEffect(() => () => {
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
  }, []);

  useEffect(() => {
    if (!cameraRunning) return;
    const interval = window.setInterval(() => setElapsedSeconds((value) => value + 1), 1000);
    return () => window.clearInterval(interval);
  }, [cameraRunning]);

  const skipLetter = () => {
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    timerRef.current = null;
    advanceRef.current = false;
    if (letterIndex >= letters.length - 1) {
      setWordIndex((value) => (value + 1) % PRACTICE_WORDS.length);
      setLetterIndex(0);
    } else {
      setLetterIndex((value) => value + 1);
    }
    setFeedback("idle");
  };

  const timeLabel = `${String(Math.floor(elapsedSeconds / 60)).padStart(2, "0")}:${String(elapsedSeconds % 60).padStart(2, "0")}`;
  const accuracy = attempts ? Math.round((correctLetters / attempts) * 100) : 0;

  return (
    <div className="grid overflow-hidden rounded-[28px] border border-border bg-surface shadow-[var(--shadow-soft)] lg:grid-cols-[minmax(360px,0.82fr)_minmax(0,1.18fr)]">
      <div className="border-b border-border lg:border-b-0 lg:border-r">
        <div className="border-b border-border bg-surface-2 px-6 py-5">
          <div className="mb-5 flex items-center justify-between gap-4 text-xs text-muted">
            <span className="rounded-full border border-border px-3 py-1.5">Seviye 1</span>
            <span>Kelime {wordIndex + 1} / {PRACTICE_WORDS.length}</span>
          </div>
          <div className="flex items-end justify-between gap-4">
            <div>
              <p className="text-xs font-semibold uppercase tracking-[.18em] text-primary">Hedef kelime</p>
              <h2 className="mt-2 text-4xl font-semibold tracking-[.16em]">{word}</h2>
            </div>
            <p className="text-right text-xs text-muted">{score} kelime<br />tamamlandı</p>
          </div>
          <div className="mt-5 flex flex-wrap gap-2" aria-label={`${word} kelime ilerlemesi`}>
            {letters.map((letter, index) => (
              <span key={`${letter}-${index}`} className={cn(
                "flex h-10 w-10 items-center justify-center rounded-xl border text-lg font-semibold",
                index < letterIndex ? "border-emerald-400/40 bg-emerald-400/10 text-emerald-500" :
                  index === letterIndex ? "border-primary bg-primary/10 text-primary" : "border-border text-muted",
              )}>
                {letter}
              </span>
            ))}
          </div>
          <div className="mt-5 flex items-center justify-between gap-3 border-t border-border pt-4">
            <div>
              <p className="text-xs font-medium text-foreground">Hangi elini kullanıyorsun?</p>
              <p className="mt-1 text-[11px] text-muted">Baskın elini seç</p>
            </div>
            <div className="flex gap-1 rounded-xl border border-border bg-surface p-1">
              {(["left", "right"] as const).map((hand) => (
                <button key={hand} type="button" onClick={() => setDominantHand(hand)} aria-pressed={dominantHand === hand} className={cn("rounded-lg px-3 py-1.5 text-xs font-medium", dominantHand === hand ? "bg-foreground text-background" : "text-muted hover:text-foreground")}>
                  {hand === "left" ? "Sol" : "Sağ"}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="relative h-[360px] overflow-hidden bg-[#171824] sm:h-[430px]" data-testid="live-training-reference">
          <div className="pointer-events-none absolute inset-0" style={{ background: "radial-gradient(ellipse at 45% 35%, #8555ff 0%, #7441f5 62%, #6533df 100%)" }} />
          <HandScene letter={reference} playing speed={1} replay={letterIndex + wordIndex * 10} />
          <div className="pointer-events-none absolute left-5 top-5 rounded-full border border-white/10 bg-black/15 px-3 py-1.5 text-[11px] tracking-wide text-white/70">
            Parmak alfabesi modeli · {expected}
          </div>
          <p className="pointer-events-none absolute bottom-5 left-5 text-xs text-white/60">Modeli taklit et</p>
        </div>

        <div className="space-y-3 px-6 py-5" aria-live="polite">
          <p className={cn(
            "text-sm font-medium",
            feedback === "correct" || feedback === "complete" ? "text-emerald-500" : feedback === "wrong" ? "text-rose-500" : "text-foreground",
          )}>
            {feedback === "complete" ? "Kelime tamamlandı! Yeni kelime geliyor…" :
              feedback === "correct" ? "Doğru! Sıradaki harfe geç." :
                feedback === "wrong" ? `Bu harf değil. “${expected}” harfini göster.` :
                  `Şimdi “${expected}” harfini göster.`}
          </p>
          <p className="text-xs leading-relaxed text-muted">Harf kararlı algılandığında otomatik ilerler. Aynı harfi tekrar etmek için elini kısa süre indir.</p>
          <div className="flex flex-wrap gap-4">
            <button type="button" onClick={reset} className="inline-flex items-center gap-2 text-xs text-muted hover:text-foreground">
              <RotateCcw size={13} /> Kelimeyi baştan başlat
            </button>
            <button type="button" onClick={skipLetter} className="inline-flex items-center gap-2 text-xs text-muted hover:text-foreground">
              <SkipForward size={13} /> Harfi atla
            </button>
          </div>
        </div>
      </div>

      <div className="min-w-0 bg-black p-3 sm:p-5">
        <div className="mb-3 grid grid-cols-3 gap-2 text-center text-xs text-white/70">
          <div className="rounded-xl border border-white/10 bg-black px-3 py-2"><Clock3 size={14} className="mx-auto mb-1 text-white/50" />{timeLabel}</div>
          <div className="rounded-xl border border-white/10 bg-black px-3 py-2"><Check size={14} className="mx-auto mb-1 text-emerald-300" />{score} puan</div>
          <div className="rounded-xl border border-white/10 bg-black px-3 py-2">{accuracy}%<span className="mt-1 block text-[10px] text-white/45">eşleşme</span></div>
        </div>
        <LiveRecognition apiBase="http://127.0.0.1:8006" modelName="TİD V6" onConfirmedLetter={handleLetter} onRunningChange={setCameraRunning} variant="training" />
      </div>
    </div>
  );
}
