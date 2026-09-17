"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { ArrowLeft, ArrowRight, Check, Clock3, Hand, Layers3, Pause, Play, RotateCcw, SkipForward, Trophy, X } from "lucide-react";
import { LiveRecognition } from "@/components/home/live-recognition";
import { LogoMark } from "@/components/site/logo";
import { TID_ALPHABET } from "@/components/tid/poses";
import { cn } from "@/lib/utils";
import { TID_API_BASE, TID_MODEL_NAME } from "@/lib/recognition-config";
import styles from "./digitra-training.module.css";

const HandScene = dynamic(() => import("@/components/tid/hand-scene"), {
  ssr: false,
  loading: () => <span className={styles.loading}>Eller hazırlanıyor…</span>,
});

const LESSONS = [
  { name: "İlk kelimeler", note: "Tanıdık isimler, kısa kelimeler.", words: ["AYŞE", "ANNE", "BABA", "EL", "EV", "OKUL"] },
  { name: "Günlük hayat", note: "Her gün karşılaştığın kelimeler.", words: ["KALEM", "KİTAP", "MASA", "KAPI", "BARDAK", "MERHABA"] },
  { name: "Türkçeye özgü harfler", note: "Ç, Ş, Ö, Ü ve İ içeren kelimeler.", words: ["ÇAY", "ŞİŞE", "ÜZÜM", "GÖZ", "ÇİÇEK", "ÖYKÜ"] },
  { name: "Hareket zamanı", note: "J ve Ğ ile hareketli harf pratiği.", words: ["JALE", "DAĞ", "YAĞMUR", "AĞAÇ", "JEL", "DOĞA"] },
] as const;
type Screen = "welcome" | "levels" | "intro" | "practice" | "complete";
type Feedback = "idle" | "correct" | "wrong" | "word";

function formatTime(seconds: number) {
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}

export function FingerspellingTraining() {
  const [screen, setScreen] = useState<Screen>("welcome");
  const [level, setLevel] = useState(0);
  const [wordIndex, setWordIndex] = useState(0);
  const [letterIndex, setLetterIndex] = useState(0);
  const [feedback, setFeedback] = useState<Feedback>("idle");
  const [correct, setCorrect] = useState(0);
  const [completedWords, setCompletedWords] = useState(0);
  const [skipped, setSkipped] = useState<number[]>([]);
  const [seconds, setSeconds] = useState(0);
  const [cameraRunning, setCameraRunning] = useState(false);
  const [playing, setPlaying] = useState(true);
  const [slow, setSlow] = useState(false);
  const [replay, setReplay] = useState(0);
  const advanceRef = useRef(false);
  const timerRef = useRef<number | null>(null);

  const lesson = LESSONS[level];
  const word = lesson.words[wordIndex];
  const letters = [...word];
  const expected = letters[letterIndex];
  const reference = TID_ALPHABET.find((letter) => letter.ch === expected) ?? TID_ALPHABET[0];
  const practicing = screen === "practice";
  const points = correct * 10;

  useEffect(() => {
    return () => {
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    };
  }, []);

  useEffect(() => {
    if (!cameraRunning || !practicing) return;
    const id = window.setInterval(() => setSeconds((value) => value + 1), 1000);
    return () => window.clearInterval(id);
  }, [cameraRunning, practicing]);

  const clearAdvance = () => {
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    timerRef.current = null;
    advanceRef.current = false;
  };

  const navigate = (next: Screen) => {
    clearAdvance();
    setFeedback("idle");
    setScreen(next);
  };

  const resetRound = () => {
    clearAdvance();
    setWordIndex(0);
    setLetterIndex(0);
    setCorrect(0);
    setCompletedWords(0);
    setSkipped([]);
    setSeconds(0);
    setFeedback("idle");
    setPlaying(true);
  };

  const selectLevel = (index: number) => {
    resetRound();
    setLevel(index);
    setScreen("intro");
  };

  const nextWord = () => {
    setSkipped([]);
    setFeedback("idle");
    advanceRef.current = false;
    if (wordIndex === lesson.words.length - 1) {
      setScreen("complete");
    } else {
      setWordIndex((value) => value + 1);
      setLetterIndex(0);
    }
  };

  const skipLetter = () => {
    if (advanceRef.current) return;
    if (letterIndex === letters.length - 1) {
      nextWord();
    } else {
      setSkipped((value) => [...value, letterIndex]);
      setLetterIndex((value) => value + 1);
      setFeedback("idle");
    }
  };

  const handleLetter = (letter: string) => {
    if (!practicing || !cameraRunning || advanceRef.current) return;
    if (letter.toLocaleUpperCase("tr-TR") !== expected) {
      setFeedback("wrong");
      return;
    }
    advanceRef.current = true;
    setCorrect((value) => value + 1);
    const lastLetter = letterIndex === letters.length - 1;
    setFeedback(lastLetter ? "word" : "correct");
    if (lastLetter && skipped.length === 0) setCompletedWords((value) => value + 1);
    timerRef.current = window.setTimeout(() => {
      timerRef.current = null;
      if (lastLetter) nextWord();
      else {
        setLetterIndex((value) => value + 1);
        setFeedback("idle");
        advanceRef.current = false;
      }
    }, lastLetter ? 1200 : 650);
  };

  const back = () => {
    if (screen === "intro") navigate("levels");
    else if (screen === "practice") navigate("intro");
    else navigate("welcome");
  };

  const message = feedback === "word"
    ? (skipped.length ? "Kelime bitti. Sıradaki geliyor." : "Doğru! Kelime tamamlandı.")
    : feedback === "correct" ? "Doğru! Ellerini kısa süre indir."
    : feedback === "wrong" ? `“${expected}” henüz eşleşmedi. Modeli tekrar incele.`
    : cameraRunning ? (reference.motion ? `“${expected}” hareketini modeldeki gibi tamamla.` : `“${expected}” harfini göster ve kısa süre koru.`) : "Kameranı aç; ilk harfi birlikte deneyelim.";

  return (
    <div className={styles.app} data-training-screen={screen}>
      <header className={styles.header}>
        <Link href="/#egitim" className={styles.brand} aria-label="Digitra eğitim bölümüne dön">
          <LogoMark className="h-8 w-8" /><span>digitra<span className={styles.brandDot}>.</span></span>
        </Link>
        <div className={styles.headerTitle}><span className={styles.liveDot} /> Canlı atölye <span className={styles.headerDivider} /> TİD</div>
        <Link href="/#egitim" className={styles.exit}><span>Eğitime dön</span><X size={17} /></Link>
      </header>

      <main className={styles.main}>
        {screen === "welcome" && <div className={styles.welcome}>
          <section className={styles.welcomeCopy}>
            <p className={styles.eyebrow}>ELLERİNLE ÖĞREN</p>
            <h1>Bir harf.<br />Bir hareket.<br /><em>Yeni bir bağ.</em></h1>
            <p className={styles.description}>3D elleri izle, kamerayla dene.<br />Kelimeleri TİD parmak alfabesiyle harf harf çalış.</p>
            <button className={styles.primaryButton} onClick={() => navigate("levels")}>Atölyeye gir <ArrowRight size={19} /></button>
            <div className={styles.welcomeFacts}><span><Layers3 size={15} /> 4 çalışma</span><span><Hand size={15} /> İki elle pratik</span></div>
          </section>
          <div className={styles.welcomeArt}>
            <div className={styles.orbit} /><div className={styles.orbitInner} />
            <span className={styles.artLetter}>A</span>
            <div className={styles.heroHands}><HandScene letter={TID_ALPHABET[0]} playing speed={0.8} replay={0} /></div>
            <div className={styles.artCaption}><span className={styles.liveDot} /> Her kelimeyi harf harf keşfet.</div>
            <span className={styles.artIndex}>01 / 29</span>
          </div>
        </div>}

        {screen === "levels" && <section className={styles.levels}>
          <button className={styles.backButton} onClick={back}><ArrowLeft size={17} /> Geri</button>
          <div className={styles.sectionHeading}><p className={styles.eyebrow}>KENDİ HIZINDA İLERLE</p><h1>Bugün ne çalışalım?</h1><p>Kısa kelimelerle başla, hareketli harflere doğru ilerle.</p></div>
          <div className={styles.lessonGrid}>
            {LESSONS.map((item, index) => <button key={item.name} className={styles.lesson} onClick={() => selectLevel(index)}>
              <span className={styles.lessonNumber}>0{index + 1}<ArrowRight size={20} /></span>
              <span className={styles.lessonLetters}>{index === 0 ? "A Y" : index === 1 ? "K M" : index === 2 ? "Ç Ş" : "J Ğ"}</span>
              <strong>{item.name}</strong><span>{item.note}</span><small>6 kelime · Harf pratiği</small>
            </button>)}
          </div>
        </section>}

        {screen === "intro" && <section className={styles.intro}>
          <div className={styles.introCopy}>
            <button className={styles.backButton} onClick={back}><ArrowLeft size={17} /> Çalışmalar</button>
            <p className={styles.eyebrow}>ÇALIŞMA 0{level + 1}</p><h1>{lesson.name}<span>.</span></h1>
            <p className={styles.description}>{lesson.note} Soldaki modeli takip et; doğru harf algılandığında sıradaki harfe geçeceğiz.</p>
            <div className={styles.wordList}>{lesson.words.map((value) => <span key={value}>{value}</span>)}</div>
            <button className={styles.primaryButton} onClick={() => navigate("practice")}>Çalışmayı aç <ArrowRight size={19} /></button>
            <p className={styles.smallNote}>Kameranı bir sonraki ekranda açabilirsin.</p>
          </div>
          <div className={styles.introArt}>
            <span className={styles.introBigLetter}>{expected}</span>
            <div className={styles.heroHands}><HandScene letter={reference} playing speed={0.8} replay={level} /></div>
            <span className={styles.artCaption}>İzle. Dene. Öğren.</span>
          </div>
        </section>}

        {practicing && <div className={styles.practice}>
          <section className={styles.modelPanel} aria-label="Parmak alfabesi modeli">
            <div className={styles.panelTop}>
              <button className={styles.backButton} onClick={back} aria-label="Çalışma bilgisine dön"><ArrowLeft size={18} /></button>
              <span><span className={styles.eyebrow}>MODELİ TAKİP ET</span><strong>{lesson.name}</strong></span>
              <span className={styles.stepCount}>{wordIndex + 1} / {lesson.words.length}</span>
            </div>
            <div className={styles.modelStage}>
              <span className={styles.ghostLetter} aria-hidden="true">{expected}</span>
              <div className={styles.stageHalo} />
              <div className={styles.modelHands}><HandScene letter={reference} playing={playing} speed={slow ? 0.5 : 1} replay={replay + letterIndex + wordIndex * 10} /></div>
              <div className={styles.modelTools}>
                <button onClick={() => setPlaying((value) => !value)} aria-label={playing ? "Modeli duraklat" : "Modeli oynat"}>{playing ? <Pause size={15} /> : <Play size={15} />}</button>
                <button onClick={() => { setReplay((value) => value + 1); setPlaying(true); }} aria-label="Hareketi tekrar göster"><RotateCcw size={15} /></button>
                <button onClick={() => setSlow((value) => !value)} aria-pressed={slow} aria-label="Yavaş gösterim">{slow ? "0.5×" : "1×"}</button>
              </div>
            </div>
            <div className={styles.wordDock}>
              <div className={styles.wordHeading}><span>HARF HARF TAMAMLA</span><span>{letterIndex + 1} / {letters.length} harf</span></div>
              <div className={styles.wordProgress} aria-label={`Hedef kelime: ${word}`}>
                {letters.map((letter, index) => <span key={index} aria-current={index === letterIndex ? "step" : undefined} className={cn(
                  index < letterIndex && !skipped.includes(index) && styles.doneLetter,
                  index === letterIndex && styles.currentLetter,
                  skipped.includes(index) && styles.skippedLetter,
                  (feedback === "correct" || feedback === "word") && index === letterIndex && styles.doneLetter,
                )}>{letter}</span>)}
              </div>
              <div className={styles.wordFooter}><p>{reference.tip}</p><button onClick={skipLetter} disabled={feedback === "correct" || feedback === "word"}>Harfi atla <SkipForward size={14} /></button></div>
            </div>
          </section>
          <section className={styles.cameraPanel} aria-label="Canlı kamera">
            <LiveRecognition apiBase={TID_API_BASE} modelName={TID_MODEL_NAME} variant="training"
              expectedLetter={expected} onConfirmedLetter={handleLetter} onRunningChange={setCameraRunning} />
            <div className={styles.cameraStats}><span><Clock3 size={14} />{formatTime(seconds)}</span><span><span className={styles.mintDot} />{points} puan</span></div>
            <div className={cn(styles.feedback, (feedback === "correct" || feedback === "word") && styles.feedbackSuccess)} role="status">
              <span className={styles.feedbackIcon}>{feedback === "correct" || feedback === "word" ? <Check size={20} /> : <Hand size={20} />}</span>
              <div><strong>{message}</strong><span>{feedback === "word" ? "Sıradaki kelime hazırlanıyor." : "İki elin de görüntüde olsun."}</span></div>
            </div>
          </section>
        </div>}

        {screen === "complete" && <section className={styles.complete}>
          <div className={styles.trophy}><Trophy size={42} strokeWidth={1.4} /></div>
          <p className={styles.eyebrow}>ÇALIŞMA TAMAMLANDI</p><h1>Ellerine sağlık<span>!</span></h1>
          <p>{lesson.name} çalışmasının sonuna geldin.</p>
          <div className={styles.results}><div><strong>{points}</strong><span>puan</span></div><div><strong>{completedWords}/{lesson.words.length}</strong><span>tamamlanan kelime</span></div><div><strong>{formatTime(seconds)}</strong><span>çalışma süresi</span></div></div>
          <div className={styles.resultActions}><button className={styles.secondaryButton} onClick={() => { resetRound(); setScreen("practice"); }}><RotateCcw size={16} /> Tekrar çalış</button><button className={styles.primaryButton} onClick={() => navigate("levels")}>Çalışmalar <ArrowRight size={18} /></button></div>
        </section>}
      </main>

      <footer className={styles.footer}><span>DIGITRA <span className={styles.footerDivider}>/</span> ÖĞRENME ATÖLYESİ</span><span>{practicing ? "Doğru harf → sıradaki adım" : "TİD parmak alfabesiyle adım adım."}</span></footer>
    </div>
  );
}
