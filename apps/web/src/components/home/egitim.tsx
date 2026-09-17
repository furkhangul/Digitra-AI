"use client";

import { useEffect, useRef, useState } from "react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { ArrowLeft, ArrowRight, BookOpen, Camera, Check, Hand, Pause, Play, RotateCcw, Trophy } from "lucide-react";
import { Container, Section } from "@/components/ui/container";
import { TID_ALPHABET, TID_G_SOURCE, TID_SOFT_G_SOURCE, TID_SOURCE } from "@/components/tid/poses";
import { cn } from "@/lib/utils";
import { LiveRecognition } from "@/components/home/live-recognition";

const HandScene = dynamic(() => import("@/components/tid/hand-scene"), { ssr: false, loading: () => <div className="absolute inset-0 flex items-center justify-center text-sm text-white/60">Eller hazırlanıyor…</div> });
function quizOptions(index: number) {
  const all = TID_ALPHABET.map((_, i) => i).filter(i => i !== index);
  for (let i=all.length-1; i>0; i--) { const j = Math.floor(Math.random()*(i+1)); [all[i],all[j]]=[all[j],all[i]]; }
  const choices = [index, ...all.slice(0,3)];
  for (let i=choices.length-1; i>0; i--) { const j = Math.floor(Math.random()*(i+1)); [choices[i],choices[j]]=[choices[j],choices[i]]; }
  return choices;
}

const PRACTICE_WORDS = ["AYŞE", "ANNE", "BABA", "OKUL", "MERHABA", "DİJİTRA"] as const;

function WordPractice({
  wordIndex,
  letterIndex,
  feedback,
  score,
  onLetter,
  onReset,
}: {
  wordIndex: number;
  letterIndex: number;
  feedback: "idle" | "correct" | "wrong" | "complete";
  score: number;
  onLetter: (letter: string) => void;
  onReset: () => void;
}) {
  const word = PRACTICE_WORDS[wordIndex];
  const letters = [...word];
  const expected = letters[letterIndex] ?? letters[0];
  const reference = TID_ALPHABET.find((letter) => letter.ch === expected) ?? TID_ALPHABET[0];

  return (
    <div className="grid overflow-hidden rounded-[28px] border border-border bg-surface shadow-[var(--shadow-soft)] lg:grid-cols-[minmax(0,0.85fr)_minmax(0,1.15fr)]">
      <div className="border-b border-border lg:border-b-0 lg:border-r">
        <div className="border-b border-border bg-surface-2 px-6 py-5">
          <p className="text-xs font-semibold uppercase tracking-[.18em] text-primary">Kelime çalışması</p>
          <div className="mt-3 flex items-end justify-between gap-4">
            <div>
              <p className="text-xs text-muted">Hedef kelime</p>
              <h3 className="mt-1 text-4xl font-semibold tracking-[.16em]">{word}</h3>
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
        </div>

        <div className="relative h-[300px] overflow-hidden bg-[#171824] sm:h-[360px]" data-testid="word-reference-stage">
          <div className="pointer-events-none absolute inset-0" style={{ background: "radial-gradient(ellipse at 45% 35%, #8555ff 0%, #7441f5 62%, #6533df 100%)" }} />
          <HandScene letter={reference} playing speed={1} replay={letterIndex + wordIndex * 10} />
          <div className="pointer-events-none absolute left-5 top-5 rounded-full border border-white/10 bg-black/15 px-3 py-1.5 text-[11px] tracking-wide text-white/70">
            Model referansı · {expected}
          </div>
          <p className="pointer-events-none absolute bottom-5 left-5 text-xs text-white/60">Soldaki modeli taklit et</p>
        </div>

        <div className="space-y-3 px-6 py-5" aria-live="polite">
          <p className={cn(
            "text-sm font-medium",
            feedback === "correct" || feedback === "complete" ? "text-emerald-500" : feedback === "wrong" ? "text-rose-500" : "text-foreground",
          )}>
            {feedback === "complete" ? "Kelime tamamlandı! Yeni kelime hazırlanıyor…" :
              feedback === "correct" ? "Doğru! Sıradaki harfe geç." :
                feedback === "wrong" ? `Bu harf değil. Şimdi “${expected}” harfini göster.` :
                  `Şimdi “${expected}” harfini kameraya göster.`}
          </p>
          <p className="text-xs leading-relaxed text-muted">Harf tanındığında otomatik olarak ilerler. Aynı harfi tekrar göstermek için elini kısa süre indir.</p>
          <button type="button" onClick={onReset} className="inline-flex items-center gap-2 text-xs text-muted hover:text-foreground">
            <RotateCcw size={13} /> Kelimeyi baştan başlat
          </button>
        </div>
      </div>

      <div className="min-w-0 p-3 sm:p-5">
        <LiveRecognition apiBase="http://127.0.0.1:8006" modelName="TİD V6" onConfirmedLetter={onLetter} />
      </div>
    </div>
  );
}

export function Egitim({ debug = false }: { debug?: boolean }) {
  const [selected, setSelected] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [replay, setReplay] = useState(0);


  const [mode, setMode] = useState<"learn" | "quiz" | "word">("learn");
  const [options, setOptions] = useState<number[]>([0, 2, 10, 26]);
  const [picked, setPicked] = useState<number | null>(null);
  const [score, setScore] = useState(0);
  const [asked, setAsked] = useState(0);
  const [auto, setAuto] = useState(false);
  const [wordIndex, setWordIndex] = useState(0);
  const [wordLetterIndex, setWordLetterIndex] = useState(0);
  const [wordFeedback, setWordFeedback] = useState<"idle" | "correct" | "wrong" | "complete">("idle");
  const [wordScore, setWordScore] = useState(0);
  const wordAdvanceRef = useRef(false);
  const wordTimerRef = useRef<number | null>(null);
  const active = TID_ALPHABET[selected];
  const sourceHref = active.ch === "G" ? TID_G_SOURCE : active.ch === "Ğ" ? TID_SOFT_G_SOURCE : TID_SOURCE;
  const choose = (index: number) => {
    setSelected((index + TID_ALPHABET.length) % TID_ALPHABET.length);
    setPlaying(true); setReplay(v => v+1); setPicked(null);
  };
  useEffect(() => {
    if (!auto || !playing || mode !== "learn") return;
    const id = window.setInterval(() => setSelected(i => (i+1)%TID_ALPHABET.length), 6500/speed);
    return () => window.clearInterval(id);
  }, [auto, playing, mode, speed]);
  useEffect(() => () => {
    if (wordTimerRef.current !== null) window.clearTimeout(wordTimerRef.current);
  }, []);
  const nextQuestion = () => {
    const index = (selected+1+Math.floor(Math.random()*(TID_ALPHABET.length-1)))%TID_ALPHABET.length;
    choose(index); setOptions(quizOptions(index));
  };

  const resetWord = () => {
    if (wordTimerRef.current !== null) {
      window.clearTimeout(wordTimerRef.current);
      wordTimerRef.current = null;
    }
    wordAdvanceRef.current = false;
    setWordLetterIndex(0);
    setWordFeedback("idle");
  };

  const handlePracticeLetter = (letter: string) => {
    if (wordAdvanceRef.current) return;
    const expected = [...PRACTICE_WORDS[wordIndex]][wordLetterIndex];
    const normalized = letter.toLocaleUpperCase("tr-TR");
    if (normalized !== expected) {
      setWordFeedback("wrong");
      return;
    }

    wordAdvanceRef.current = true;
    setWordFeedback("correct");
    wordTimerRef.current = window.setTimeout(() => {
      wordTimerRef.current = null;
      const isLastLetter = wordLetterIndex >= [...PRACTICE_WORDS[wordIndex]].length - 1;
      if (isLastLetter) {
        setWordScore((value) => value + 1);
        setWordFeedback("complete");
        wordTimerRef.current = window.setTimeout(() => {
          setWordIndex((value) => (value + 1) % PRACTICE_WORDS.length);
          setWordLetterIndex(0);
          setWordFeedback("idle");
          wordAdvanceRef.current = false;
          wordTimerRef.current = null;
        }, 900);
      } else {
        setWordLetterIndex((value) => value + 1);
        setWordFeedback("idle");
        wordAdvanceRef.current = false;
      }
    }, 650);
  };

  return <Section id="egitim" className="overflow-hidden"><Container>
    <div className="mb-10 flex flex-col justify-between gap-6 md:flex-row md:items-end">
      <div className="max-w-2xl">
        <p className="mb-4 flex items-center gap-2 text-xs font-semibold uppercase tracking-[.2em] text-primary"><span className="h-1.5 w-1.5 rounded-full bg-primary" /> Türk İşaret Dili · Parmak alfabesi</p>
        <h2 className="text-4xl font-semibold tracking-tight sm:text-5xl">29 harf. Kendi hızında öğren.</h2>
        <p className="mt-4 max-w-xl text-base leading-relaxed text-muted">3D ellerde parmakların yerleşimini incele, hareketi yavaşlat ve tekrar et. Harfleri öğrendikçe kendini sına; canlı eğitimde kelimeleri harf harf çalış.</p>
      </div>
      <div className="flex w-fit gap-1 rounded-full border border-border bg-surface-2 p-1" aria-label="Eğitim modu">
        <button type="button" aria-pressed={mode === "learn"} onClick={() => { resetWord(); setMode("learn"); setPicked(null); }} className={cn("flex cursor-pointer items-center gap-2 rounded-full px-5 py-2.5 text-sm font-medium", mode === "learn" ? "bg-foreground text-background shadow-sm" : "text-muted")}><BookOpen size={15} /> Öğren</button>
        <button type="button" aria-pressed={mode === "quiz"} onClick={() => { resetWord(); setMode("quiz"); setAuto(false); nextQuestion(); }} className={cn("flex cursor-pointer items-center gap-2 rounded-full px-5 py-2.5 text-sm font-medium", mode === "quiz" ? "bg-foreground text-background shadow-sm" : "text-muted")}><Trophy size={15} /> Kendini dene</button>
        <Link href="/egitim/canli" className="flex cursor-pointer items-center gap-2 rounded-full px-5 py-2.5 text-sm font-medium text-muted transition-colors hover:bg-surface hover:text-foreground"><Camera size={15} /> Canlı eğitim</Link>
      </div>
    </div>
    {mode === "word" && <WordPractice wordIndex={wordIndex} letterIndex={wordLetterIndex} feedback={wordFeedback} score={wordScore} onLetter={handlePracticeLetter} onReset={resetWord} />}
    <div className={cn("grid overflow-hidden rounded-[28px] border border-border bg-surface shadow-[var(--shadow-soft)] lg:grid-cols-[minmax(0,1fr)_340px]", mode === "word" && "hidden")}>
      <div className="min-w-0 border-border lg:border-r">
        <div className="relative h-[360px] overflow-hidden bg-[#171824] sm:h-[470px]" data-testid="tid-stage">
          <div className="pointer-events-none absolute inset-0" style={{ background: "radial-gradient(ellipse at 45% 35%, #8555ff 0%, #7441f5 62%, #6533df 100%)" }} />
          <div className="pointer-events-none absolute inset-x-0 bottom-0 h-24" style={{ background: "radial-gradient(ellipse at center, #4321a53d 0%, transparent 66%)" }} />
          <HandScene letter={active} playing={playing} speed={speed} replay={replay} debug={debug} hideLetter={mode === "quiz" && picked === null} />
          <div className="pointer-events-none absolute left-5 top-5 flex items-center gap-2 rounded-full border border-white/10 bg-black/15 px-3 py-1.5 text-[11px] tracking-wide text-white/65"><span className="h-1.5 w-1.5 rounded-full bg-[#c4b5fd]" /> {active.pose.right.visible && active.pose.left.visible ? "İKİ EL" : "TEK EL"} · TİD</div>
          {mode === "learn" && <div className="pointer-events-none absolute right-6 top-4 text-right"><p className="text-[68px] font-medium leading-none tracking-tight text-white/90 sm:text-[82px]" data-testid="current-letter">{active.ch}</p><p className="mt-2 text-[11px] tracking-[.18em] text-white/40">{String(selected+1).padStart(2,"0")} / 29</p></div>}
          <div className="pointer-events-none absolute bottom-5 left-5 flex gap-4 text-[11px] text-white/60">{active.pose.right.visible && <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-[#f0dfc6]" /> Sağ el</span>}{active.pose.left.visible && <span className="flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-[#a18ae8]" /> Sol el</span>}</div>
          <p className="pointer-events-none absolute bottom-5 right-5 hidden items-center gap-1.5 text-[11px] text-white/40 sm:flex">Sabit ön görünüm</p>
        </div>
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-t border-border px-4 py-3 sm:px-5">
          <div className="flex items-center gap-1">
            <button type="button" aria-label={playing ? "Hareketi duraklat" : "Hareketi oynat"} title={playing ? "Duraklat" : "Oynat"} onClick={() => setPlaying(v=>!v)} className="flex h-9 w-9 cursor-pointer items-center justify-center rounded-full bg-foreground text-background">{playing ? <Pause size={15} /> : <Play size={15} />}</button>
            <button type="button" aria-label="Hareketi tekrarla" title="Tekrarla" onClick={() => {setReplay(v=>v+1); setPlaying(true);}} className="flex h-9 w-9 cursor-pointer items-center justify-center rounded-full text-muted hover:bg-surface-2"><RotateCcw size={16} /></button>
            <div className="mx-2 h-5 w-px bg-border" />
            {[.5, 1, 1.5].map(s => <button key={s} type="button" aria-label={`${s} kat hız`} aria-pressed={speed===s} onClick={()=>setSpeed(s)} className={cn("cursor-pointer rounded-lg px-2.5 py-1.5 text-xs font-medium", speed===s ? "bg-primary/10 text-primary" : "text-muted hover:bg-surface-2")}>{s}×</button>)}
          </div>
          <span className="text-xs text-muted">Önden gösterim</span>
        </div>
        <div className="flex min-h-[112px] items-start gap-3 px-5 py-5" aria-live="polite"><div className="mt-0.5 rounded-xl bg-primary/10 p-2.5 text-primary"><Hand size={18} /></div><div>{mode === "learn" || picked !== null ? <><p className="text-sm font-medium leading-relaxed">{active.tip}</p><p className="mt-1 text-xs leading-relaxed text-muted">{active.detail}</p></> : <><p className="text-sm font-medium">Bu hareket hangi harfi gösteriyor?</p><p className="mt-1 text-xs text-muted">İki elin konumuna bak. İstersen hareketi yavaşlat.</p></>}</div></div>
      </div>
      <aside className="flex flex-col border-t border-border p-6 lg:border-t-0">
        {mode === "learn" ? <>
          <div className="mb-5 flex items-center justify-between"><h3 className="text-sm font-semibold">Alfabeyi keşfet</h3><span className="text-xs text-muted">29 harf</span></div>
          <div className="grid grid-cols-6 gap-2 lg:grid-cols-5" aria-label="TİD alfabesi">{TID_ALPHABET.map((l,i)=><button key={l.ch} type="button" aria-label={`${l.ch} harfini göster`} aria-pressed={selected===i} onClick={()=>{setAuto(false);choose(i);}} className={cn("relative flex aspect-square cursor-pointer items-center justify-center rounded-xl border text-base font-semibold transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-primary",selected===i ? "border-primary bg-primary text-white shadow-[0_4px_14px_#6d5bff35]" : "border-border bg-surface hover:border-primary/40 hover:bg-primary/5")}>{l.ch}{l.motion && <span className={cn("absolute bottom-1.5 h-1 w-1 rounded-full",selected===i?"bg-white/60":"bg-primary/50")} />}</button>)}</div>
          <div className="mt-5 flex items-center justify-between text-xs text-muted"><span className="flex items-center gap-1.5"><span className="h-1 w-1 rounded-full bg-primary/50" /> Hareketli gösterim</span><a href={sourceHref} target="_blank" rel="noreferrer" className="hover:text-primary">Kaynak ↗</a></div>
          <div className="mt-auto pt-7"><button type="button" aria-pressed={auto} onClick={()=>{setAuto(v=>!v);setPlaying(true);}} className={cn("flex w-full cursor-pointer items-center justify-center gap-2 rounded-xl border py-3 text-sm",auto?"border-primary/25 bg-primary/10 text-primary":"border-border bg-surface-2 text-foreground")}><Play size={14}/>{auto?"Sırayla oynatılıyor":"Alfabeyi sırayla oynat"}</button><div className="mt-3 flex items-center justify-between"><button type="button" aria-label="Önceki harf" onClick={()=>{setAuto(false);choose(selected-1);}} className="flex cursor-pointer items-center gap-2 py-2 text-xs text-muted hover:text-foreground"><ArrowLeft size={14}/> Önceki</button><button type="button" aria-label="Sonraki harf" onClick={()=>{setAuto(false);choose(selected+1);}} className="flex cursor-pointer items-center gap-2 py-2 text-xs text-muted hover:text-foreground">Sonraki <ArrowRight size={14}/></button></div></div>
        </> : <>
          <p className="text-xs uppercase tracking-widest text-muted">Kendini dene</p><p className="mt-3 text-4xl font-semibold">{score}<span className="text-lg font-normal text-muted"> / {asked}</span></p><p className="mt-2 text-sm text-muted">doğru yanıt</p>
          <div className="mt-7 grid grid-cols-2 gap-3">{options.map(i=><button key={i} type="button" disabled={picked!==null} onClick={()=>{setPicked(i);setAsked(n=>n+1);if(i===selected)setScore(n=>n+1);}} className={cn("flex h-20 cursor-pointer items-center justify-center rounded-2xl border text-2xl font-semibold",picked!==null&&i===selected?"border-success bg-success/10 text-success":picked===i?"border-danger bg-danger/10 text-danger":"border-border hover:bg-surface-2")}>{TID_ALPHABET[i].ch}{picked!==null&&i===selected&&<Check className="ml-2" size={18}/>}</button>)}</div>
          <div aria-live="polite" className="mt-5 text-sm">{picked!==null&&(picked===selected?"Doğru bildin.":`Bu hareket ${active.ch} harfini gösteriyor.`)}</div>
          {picked!==null&&<button type="button" onClick={nextQuestion} className="mt-4 flex cursor-pointer items-center justify-center gap-2 rounded-xl bg-primary py-3 text-sm text-white">Sonraki soru <ArrowRight size={15}/></button>}
          <button type="button" onClick={()=>{setAsked(0);setScore(0);nextQuestion();}} className="mt-auto cursor-pointer pt-6 text-xs text-muted">Skoru sıfırla</button>
        </>}
      </aside>
    </div>
    <p className="mt-4 text-xs leading-relaxed text-muted">Kaynak: genel alfabe için TİD Dilbilgisi Kitabı (2015, s. 91); G/Ğ için MEB tarafından hazırlanan TDK Türk İşaret Dili Sözlüğü (2012). Üç boyutlu uyarlamanın TİD uzmanı doğrulaması bekleniyor.</p>
  </Container></Section>;
}


