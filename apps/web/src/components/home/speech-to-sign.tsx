"use client";

import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";
import dynamic from "next/dynamic";
import { Info, LoaderCircle, Mic, MicOff, Pause, Play, RotateCcw, Sparkles } from "lucide-react";
import { TID_ALPHABET, LOOP, type Letter } from "@/components/tid/poses";
import { planFingerspelling, type SpellToken } from "@/lib/tid-text";
import { getRecognitionConstructor, recognitionErrorMessage, type BrowserRecognition } from "@/lib/speech";
import { cn } from "@/lib/utils";

const HandScene = dynamic(() => import("@/components/tid/hand-scene"), {
  ssr: false,
  loading: () => <div className="absolute inset-0 flex items-center justify-center text-sm text-white/60">Eller hazırlanıyor…</div>,
});

const LETTERS = new Map(TID_ALPHABET.map((letter) => [letter.ch, letter]));
// HandPlayer needs 1.35 s to move into a sign; static signs then hold briefly,
// motion signs play one full loop so the movement is seen at least once.
const TRANSITION_S = 1.35;
const STATIC_HOLD_S = 0.9;
const WORD_GAP_S = 0.8;
const EXAMPLES = ["MERHABA", "ANNE", "TEŞEKKÜRLER", "DİGİTRA"];

function tokenDuration(token: SpellToken, speed: number) {
  if (token.kind === "space") return WORD_GAP_S / speed;
  const letter = LETTERS.get(token.ch);
  return (TRANSITION_S + (letter?.motion ? LOOP : STATIC_HOLD_S)) / speed;
}

export type SpellRequest = { text: string; nonce: number };

const noopSubscribe = () => () => {};

export function SpeechToSign({ request }: { request?: SpellRequest | null }) {
  const [input, setInput] = useState("");
  const [spelled, setSpelled] = useState("");
  const [position, setPosition] = useState(-1);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [replay, setReplay] = useState(0);
  const [listening, setListening] = useState(false);
  const [interim, setInterim] = useState("");
  const [micError, setMicError] = useState("");
  const micSupported = useSyncExternalStore(noopSubscribe, () => Boolean(getRecognitionConstructor()), () => true);
  const [handledNonce, setHandledNonce] = useState<number | null>(null);
  const recognitionRef = useRef<BrowserRecognition | null>(null);
  const timerRef = useRef<number | null>(null);

  const plan = useMemo(() => planFingerspelling(spelled), [spelled]);
  const token = position >= 0 ? plan.tokens[position] : undefined;
  const lastLetter = useMemo<Letter>(() => {
    for (let i = Math.min(position, plan.tokens.length - 1); i >= 0; i--) {
      const candidate = plan.tokens[i];
      if (candidate.kind === "letter") return LETTERS.get(candidate.ch) ?? TID_ALPHABET[0];
    }
    const first = plan.tokens.find((candidate) => candidate.kind === "letter");
    return (first?.kind === "letter" && LETTERS.get(first.ch)) || TID_ALPHABET[0];
  }, [plan, position]);

  const clearTimer = () => {
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    timerRef.current = null;
  };

  const spell = useCallback((text: string) => {
    const trimmed = text.trim();
    if (!trimmed) return;
    setInput(trimmed);
    setSpelled(trimmed);
    setPosition(0);
    setReplay((value) => value + 1);
    setPlaying(true);
  }, []);

  // Advance through the sequence; each step bumps `replay` so a repeated
  // letter (the two N's in ANNE) visibly re-forms instead of standing still.
  useEffect(() => {
    clearTimer();
    if (!playing || !token) return;
    timerRef.current = window.setTimeout(() => {
      if (position + 1 >= plan.tokens.length) {
        setPlaying(false);
        return;
      }
      const next = plan.tokens[position + 1];
      setPosition(position + 1);
      if (next.kind === "letter") setReplay((value) => value + 1);
    }, tokenDuration(token, speed) * 1000);
    return clearTimer;
  }, [playing, position, plan, speed, token]);

  // A new request from the assistant replaces whatever is playing.
  if (request && request.nonce !== handledNonce) {
    setHandledNonce(request.nonce);
    spell(request.text);
  }

  useEffect(() => () => { clearTimer(); recognitionRef.current?.abort(); }, []);

  const toggleMic = () => {
    if (listening) { recognitionRef.current?.stop(); return; }
    const Recognition = getRecognitionConstructor();
    if (!Recognition) return;
    const recognition = new Recognition();
    recognition.lang = "tr-TR";
    recognition.interimResults = true;
    recognition.continuous = false;
    recognition.maxAlternatives = 1;
    let finalText = "";
    recognition.onstart = () => { setListening(true); setMicError(""); setInterim(""); };
    recognition.onresult = (event) => {
      let pending = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) finalText += result[0].transcript;
        else pending += result[0].transcript;
      }
      setInterim(pending);
    };
    recognition.onerror = (event) => { if (event.error !== "aborted") setMicError(recognitionErrorMessage(event.error)); };
    recognition.onend = () => {
      setListening(false);
      setInterim("");
      recognitionRef.current = null;
      if (finalText.trim()) spell(finalText);
    };
    recognitionRef.current = recognition;
    recognition.start();
  };

  const finished = !playing && position >= plan.tokens.length - 1 && plan.tokens.length > 0;

  return (
    <div className="overflow-hidden rounded-3xl border border-border bg-surface shadow-[var(--shadow-soft)]">
      <div className="relative h-[340px] overflow-hidden bg-[#171824] sm:h-[400px]" data-testid="speech-to-sign-stage">
        <div className="pointer-events-none absolute inset-0" style={{ background: "radial-gradient(ellipse at 45% 35%, #8555ff 0%, #7441f5 62%, #6533df 100%)" }} />
        <HandScene letter={lastLetter} playing speed={speed} replay={replay} />
        <div className="pointer-events-none absolute left-5 top-5 flex items-center gap-2 rounded-full border border-white/10 bg-black/15 px-3 py-1.5 text-[11px] tracking-wide text-white/70">
          <Sparkles size={12} /> Parmak alfabesi · harf harf
        </div>
        {plan.tokens.length > 0 && (
          <div className="pointer-events-none absolute right-6 top-4 text-right">
            <p className="text-[68px] font-medium leading-none tracking-tight text-white/90 sm:text-[82px]" aria-live="polite">
              {token?.kind === "letter" ? token.ch : "␣"}
            </p>
            {token?.kind === "letter" && token.source !== token.ch && (
              <p className="mt-1 text-[11px] text-white/55">{token.source} → {token.ch}</p>
            )}
          </div>
        )}
        {plan.tokens.length === 0 && (
          <p className="pointer-events-none absolute inset-x-0 bottom-6 text-center text-sm text-white/70">Bir kelime yaz ya da söyle; eller harf harf göstersin.</p>
        )}
      </div>

      {plan.tokens.length > 0 && (
        <div className="flex flex-wrap gap-1.5 border-b border-border px-5 py-4" aria-label="Hecelenen harfler">
          {plan.tokens.map((item, index) => item.kind === "space"
            ? <span key={index} className="w-3" aria-hidden />
            : (
              <button
                key={index}
                type="button"
                onClick={() => { setPosition(index); setReplay((value) => value + 1); setPlaying(false); }}
                aria-label={`${item.ch} harfine git`}
                className={cn(
                  "flex h-9 min-w-9 cursor-pointer items-center justify-center rounded-lg border px-2 text-base font-semibold transition-colors",
                  index === position ? "border-primary bg-primary text-white" :
                    index < position ? "border-primary/25 bg-primary/10 text-primary" : "border-border text-muted hover:border-primary/40",
                )}
              >
                {item.ch}
              </button>
            ))}
        </div>
      )}

      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-border px-5 py-3">
        <div className="flex items-center gap-1">
          <button
            type="button"
            disabled={!plan.tokens.length}
            onClick={() => {
              if (finished) { setPosition(0); setReplay((value) => value + 1); setPlaying(true); }
              else setPlaying((value) => !value);
            }}
            aria-label={playing ? "Duraklat" : "Oynat"}
            className="flex h-9 w-9 cursor-pointer items-center justify-center rounded-full bg-foreground text-background disabled:opacity-35"
          >
            {playing ? <Pause size={15} /> : <Play size={15} />}
          </button>
          <button
            type="button"
            disabled={!plan.tokens.length}
            onClick={() => { setPosition(0); setReplay((value) => value + 1); setPlaying(true); }}
            aria-label="Baştan oynat"
            className="flex h-9 w-9 cursor-pointer items-center justify-center rounded-full text-muted hover:bg-surface-2 disabled:opacity-35"
          >
            <RotateCcw size={16} />
          </button>
          <div className="mx-2 h-5 w-px bg-border" />
          {[0.5, 1, 1.5].map((value) => (
            <button
              key={value}
              type="button"
              aria-pressed={speed === value}
              onClick={() => setSpeed(value)}
              className={cn("cursor-pointer rounded-lg px-2.5 py-1.5 text-xs font-medium", speed === value ? "bg-primary/10 text-primary" : "text-muted hover:bg-surface-2")}
            >
              {value}×
            </button>
          ))}
        </div>
        <span className="text-xs text-muted">
          {plan.tokens.length ? `${Math.max(position + 1, 0)} / ${plan.tokens.length}` : "Hazır"}
        </span>
      </div>

      <form
        className="space-y-3 p-5"
        onSubmit={(event) => { event.preventDefault(); spell(input); }}
      >
        <label htmlFor="speech-to-sign-input" className="text-sm font-medium">Kelime ya da kısa cümle</label>
        <div className="flex gap-2">
          <input
            id="speech-to-sign-input"
            value={listening ? interim || input : input}
            onChange={(event) => setInput(event.target.value)}
            maxLength={80}
            placeholder={listening ? "Dinliyorum…" : "Örneğin: merhaba"}
            className="min-w-0 flex-1 rounded-xl border border-border bg-surface-2 px-4 py-2.5 text-sm outline-none focus:border-primary"
          />
          <button
            type="button"
            onClick={toggleMic}
            disabled={!micSupported}
            aria-pressed={listening}
            aria-label={listening ? "Dinlemeyi durdur" : "Söyleyerek yaz"}
            title={micSupported ? undefined : "Bu tarayıcı ses tanımayı desteklemiyor (Chrome ya da Edge önerilir)"}
            className={cn(
              "flex h-11 w-11 shrink-0 cursor-pointer items-center justify-center rounded-xl border transition disabled:cursor-not-allowed disabled:opacity-40",
              listening ? "animate-pulse border-rose-400 bg-rose-500 text-white" : "border-border text-muted hover:border-primary/40 hover:text-primary",
            )}
          >
            {listening ? <MicOff size={18} /> : micSupported ? <Mic size={18} /> : <LoaderCircle size={18} />}
          </button>
          <button
            type="submit"
            disabled={!input.trim()}
            className="shrink-0 cursor-pointer rounded-xl px-5 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50"
            style={{ background: "var(--grad-primary)" }}
          >
            İşaretle
          </button>
        </div>
        <div className="flex flex-wrap gap-2">
          {EXAMPLES.map((example) => (
            <button key={example} type="button" onClick={() => spell(example)} className="cursor-pointer rounded-full border border-border px-3 py-1 text-xs text-muted hover:border-primary/40 hover:text-foreground">
              {example}
            </button>
          ))}
        </div>
        {micError && <p role="alert" className="text-xs text-rose-500">{micError}</p>}
        {!micSupported && <p className="text-xs text-muted">Bu tarayıcı ses tanımayı desteklemiyor; kelimeyi yazarak gösterebilirsin. Sesle kullanmak için Chrome ya da Edge önerilir.</p>}
        {(plan.substituted.length > 0 || plan.skipped.length > 0) && (
          <p className="flex items-start gap-2 text-xs text-muted">
            <Info size={14} className="mt-0.5 shrink-0" />
            <span>
              {plan.substituted.length > 0 && `TİD alfabesinde olmayan ${plan.substituted.join(", ")} Türkçe okunuşuyla hecelendi. `}
              {plan.skipped.length > 0 && `Gösterilemeyen karakterler atlandı: ${plan.skipped.join(" ")}.`}
            </span>
          </p>
        )}
        <p className="text-xs leading-relaxed text-muted">
          Bu bir parmak alfabesi hecelemesidir, tam TİD çevirisi değildir. Sayılar yazıyla hecelenir.
        </p>
      </form>
    </div>
  );
}
