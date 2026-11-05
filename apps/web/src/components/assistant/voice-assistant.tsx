"use client";

import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react";
import { usePathname, useRouter } from "next/navigation";
import { AnimatePresence, motion } from "framer-motion";
import { AudioLines, Ear, EarOff, LoaderCircle, Mic, MicOff, Send, Volume2, VolumeX, X } from "lucide-react";
import { useTheme } from "@/components/site/theme-provider";
import { DIGITRA_API_BASE } from "@/lib/api-config";
import { getLastRecognition, sendCommand } from "@/lib/assistant-bus";
import { HELP_TEXT, SUGGESTIONS, parseIntent, stripWakeWord, type SectionId } from "@/lib/assistant-intents";
import {
  getRecognitionConstructor,
  recognitionErrorMessage,
  speak,
  speechSynthesisSupported,
  stopSpeaking,
  type BrowserRecognition,
} from "@/lib/speech";
import { cn } from "@/lib/utils";

type Message = { id: number; role: "user" | "assistant"; text: string };
type Status = "idle" | "listening" | "thinking" | "speaking";
type ServerAction = { type: "navigate" | "spell" | "camera_on" | "camera_off" | "quiz" | "open_live_training"; target: string; text: string };

const PREFS_KEY = "digitra-assistant-prefs";
const SECTION_ORDER: SectionId[] = ["ceviri", "egitim", "how-it-works", "architecture", "science", "datasets", "faq"];
const WAKE_FOLLOW_UP_MS = 7000;

const PREFS_EVENT = "digitra-assistant-prefs-change";
type Prefs = { voice: boolean; handsFree: boolean };

function readPrefsRaw() {
  try { return window.localStorage.getItem(PREFS_KEY) ?? ""; } catch { return ""; }
}

function parsePrefs(raw: string): Prefs {
  try {
    const parsed = JSON.parse(raw || "{}");
    return { voice: parsed.voice !== false, handsFree: parsed.handsFree === true };
  } catch {
    return { voice: true, handsFree: false };
  }
}

function writePrefs(prefs: Prefs) {
  try { window.localStorage.setItem(PREFS_KEY, JSON.stringify(prefs)); } catch { /* private mode */ }
  window.dispatchEvent(new Event(PREFS_EVENT));
}

function subscribePrefs(callback: () => void) {
  window.addEventListener(PREFS_EVENT, callback);
  window.addEventListener("storage", callback);
  return () => {
    window.removeEventListener(PREFS_EVENT, callback);
    window.removeEventListener("storage", callback);
  };
}

const noopSubscribe = () => () => {};

function currentSection(): SectionId | null {
  const middle = window.innerHeight / 2;
  for (const id of SECTION_ORDER) {
    const rect = document.getElementById(id)?.getBoundingClientRect();
    if (rect && rect.top <= middle && rect.bottom >= middle) return id;
  }
  return null;
}

export function VoiceAssistant() {
  const router = useRouter();
  const pathname = usePathname();
  const { setTheme } = useTheme();

  const [open, setOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [status, setStatus] = useState<Status>("idle");
  const [interim, setInterim] = useState("");
  const [draft, setDraft] = useState("");
  const [error, setError] = useState("");
  const prefs = parsePrefs(useSyncExternalStore(subscribePrefs, readPrefsRaw, () => ""));
  const micSupported = useSyncExternalStore(noopSubscribe, () => Boolean(getRecognitionConstructor()), () => true);
  const voiceReplies = prefs.voice;
  const handsFree = prefs.handsFree && micSupported;
  const [llmReady, setLlmReady] = useState<boolean | null>(null);

  const idRef = useRef(0);
  const historyRef = useRef<Message[]>([]);
  const recognitionRef = useRef<BrowserRecognition | null>(null);
  const handsFreeRef = useRef(false);
  const speakingRef = useRef(false);
  const awaitingCommandUntilRef = useRef(0);
  const listRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const voiceRepliesRef = useRef(true);
  const handleRef = useRef<(text: string) => void>(() => {});
  const startHandsFreeRef = useRef<() => void>(() => {});

  useEffect(() => { voiceRepliesRef.current = voiceReplies; }, [voiceReplies]);

  useEffect(() => {
    if (!open || llmReady !== null) return;
    fetch(`${DIGITRA_API_BASE}/assistant/status`)
      .then((response) => (response.ok ? response.json() : { llm: false }))
      .then((data: { llm?: boolean }) => setLlmReady(Boolean(data.llm)))
      .catch(() => setLlmReady(false));
  }, [open, llmReady]);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, interim, status]);

  const addMessage = useCallback((role: Message["role"], text: string) => {
    const message = { id: ++idRef.current, role, text };
    historyRef.current = [...historyRef.current, message].slice(-16);
    setMessages((current) => [...current, message].slice(-40));
  }, []);

  // ---- Speech output -----------------------------------------------------
  const say = useCallback((text: string) => {
    addMessage("assistant", text);
    if (!voiceRepliesRef.current || !speechSynthesisSupported()) { setStatus("idle"); return; }
    speakingRef.current = true;
    // Mute hands-free listening while talking so the assistant never hears itself.
    recognitionRef.current?.abort();
    setStatus("speaking");
    speak(text, {
      onEnd: () => {
        speakingRef.current = false;
        setStatus((current) => (current === "speaking" ? "idle" : current));
        if (handsFreeRef.current) startHandsFreeRef.current();
      },
    });
  }, [addMessage]);

  // ---- Site actions -------------------------------------------------------
  const goToSection = useCallback((section: SectionId) => {
    if (pathname !== "/") {
      router.push(section === "top" ? "/" : `/#${section}`);
      return;
    }
    // scrollIntoView is unreliable next to the sections' overflow-hidden
    // wrappers, so scroll the window to the section's absolute offset.
    const target = section === "top" ? null : document.getElementById(section);
    const top = target ? target.getBoundingClientRect().top + window.scrollY : 0;
    // Smooth scrolling never runs in a hidden tab (hands-free use); jump there.
    window.scrollTo({ top, behavior: document.hidden ? "instant" : "smooth" });
  }, [pathname, router]);

  const spellWord = useCallback((text: string) => {
    if (pathname !== "/") {
      router.push(`/?isaret=${encodeURIComponent(text)}#ceviri`);
    } else {
      sendCommand({ type: "spell", text });
      goToSection("ceviri");
    }
    say(`"${text}" kelimesini parmak alfabesiyle harf harf gösteriyorum.`);
  }, [goToSection, pathname, router, say]);

  const camera = useCallback((on: boolean) => {
    if (pathname === "/egitim/canli") {
      sendCommand({ type: "camera", on });
      say(on ? "Kamerayı açıyorum. Elini kadraja getir." : "Kamerayı kapattım.");
      return;
    }
    if (pathname !== "/") {
      router.push("/#ceviri");
      say(on ? "Harf tanıma bölümüne gidiyorum. Kamerayı başlatmak için tekrar 'kamerayı aç' de." : "Kamera zaten kapalı.");
      return;
    }
    if (on) goToSection("ceviri");
    sendCommand({ type: "camera", on });
    say(on ? "Kamerayı açıyorum. Harfi bir iki saniye sabit tut." : "Kamerayı kapattım.");
  }, [goToSection, pathname, router, say]);

  const sendFeedback = useCallback(async (correct: string) => {
    const last = getLastRecognition();
    if (!last?.prediction) {
      say("Düzeltilecek bir tahmin bulamadım. Önce kamerayla bir harf göster.");
      return;
    }
    if (last.prediction === correct) {
      say(`Son tahmin zaten ${correct} harfiydi.`);
      return;
    }
    try {
      const response = await fetch(`${DIGITRA_API_BASE}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ predicted_label: last.prediction, correct_label: correct, model_version: last.model, source: "assistant" }),
      });
      if (!response.ok) throw new Error(String(response.status));
      say(`Teşekkürler. ${last.prediction} yerine ${correct} olması gerektiğini inceleme listesine ekledim. Görüntün kaydedilmedi.`);
    } catch {
      say("Geri bildirimi gönderemedim. Digitra API çalışıyor mu kontrol et.");
    }
  }, [say]);

  const runServerAction = useCallback((action: ServerAction | null | undefined) => {
    if (!action) return;
    if (action.type === "navigate" && action.target) goToSection(action.target as SectionId);
    else if (action.type === "spell" && action.text) {
      if (pathname !== "/") router.push(`/?isaret=${encodeURIComponent(action.text)}#ceviri`);
      else { sendCommand({ type: "spell", text: action.text }); goToSection("ceviri"); }
    } else if (action.type === "camera_on" || action.type === "camera_off") {
      sendCommand({ type: "camera", on: action.type === "camera_on" });
      if (action.type === "camera_on") goToSection("ceviri");
    } else if (action.type === "quiz") { goToSection("egitim"); sendCommand({ type: "quiz" }); }
    else if (action.type === "open_live_training") router.push("/egitim/canli");
  }, [goToSection, pathname, router]);

  const askServer = useCallback(async (text: string) => {
    setStatus("thinking");
    try {
      const response = await fetch(`${DIGITRA_API_BASE}/assistant`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          history: historyRef.current.slice(0, -1).slice(-8).map(({ role, text: content }) => ({ role, content })),
          section: currentSection(),
          recognized_text: getLastRecognition()?.text || null,
        }),
      });
      if (response.status === 503) {
        setLlmReady(false);
        say("Bunu henüz yanıtlayamıyorum. Serbest sorular için Claude bağlantısı kapalı; ama site komutlarını kullanabilirsin. 'Ne yapabilirsin' diye sorabilirsin.");
        return;
      }
      if (!response.ok) throw new Error(String(response.status));
      const data = (await response.json()) as { reply: string; action?: ServerAction | null };
      runServerAction(data.action);
      say(data.reply || "Bunu anlayamadım, farklı bir şekilde sorar mısın?");
    } catch {
      say("Digitra sunucusuna ulaşamadım. Komutlar yine de çalışır: örneğin 'merhaba kelimesini işaretle'.");
    }
  }, [runServerAction, say]);

  const handle = useCallback((raw: string) => {
    const text = raw.trim();
    if (!text) return;
    setError("");
    addMessage("user", text);
    const intent = parseIntent(text);
    switch (intent.kind) {
      case "stop":
        stopSpeaking();
        speakingRef.current = false;
        setStatus("idle");
        if (handsFreeRef.current) startHandsFreeRef.current();
        return;
      case "help": say(HELP_TEXT); return;
      case "answer": say(intent.reply); return;
      case "navigate": goToSection(intent.section); say(intent.reply); return;
      case "spell": spellWord(intent.text); return;
      case "camera": camera(intent.on); return;
      case "text": {
        sendCommand({ type: "text", op: intent.op });
        say(intent.op === "clear" ? "Metni temizledim." : intent.op === "backspace" ? "Son harfi sildim." : "Boşluk ekledim.");
        return;
      }
      case "read": {
        const recognized = getLastRecognition()?.text?.trim();
        say(recognized ? `Yazdığın metin: ${recognized}` : "Henüz tanınan bir metin yok. Kamerayı açıp harfleri gösterebilirsin.");
        return;
      }
      case "feedback": void sendFeedback(intent.correct); return;
      case "theme": setTheme(intent.theme); say(intent.theme === "dark" ? "Karanlık temaya geçtim." : "Aydınlık temaya geçtim."); return;
      case "quiz":
        goToSection("egitim");
        if (pathname === "/") sendCommand({ type: "quiz" });
        say("Test başlıyor. Ellerin gösterdiği harfi seçeneklerden bul.");
        return;
      case "live-training": router.push("/egitim/canli"); say("Canlı eğitimi açıyorum. Modeli izle ve harfi kameraya göster."); return;
      case "time": say(`Saat ${new Date().toLocaleTimeString("tr-TR", { hour: "2-digit", minute: "2-digit" })}.`); return;
      case "date": say(`Bugün ${new Date().toLocaleDateString("tr-TR", { weekday: "long", day: "numeric", month: "long", year: "numeric" })}.`); return;
      case "unknown": void askServer(intent.text); return;
    }
  }, [addMessage, askServer, camera, goToSection, pathname, router, say, sendFeedback, setTheme, spellWord]);

  useEffect(() => { handleRef.current = handle; }, [handle]);

  // ---- Speech input --------------------------------------------------------
  const stopListening = useCallback(() => {
    recognitionRef.current?.abort();
    recognitionRef.current = null;
    setInterim("");
    setStatus((current) => (current === "listening" ? "idle" : current));
  }, []);

  const listenOnce = useCallback(() => {
    const Recognition = getRecognitionConstructor();
    if (!Recognition) return;
    stopSpeaking();
    speakingRef.current = false;
    recognitionRef.current?.abort();
    const recognition = new Recognition();
    recognition.lang = "tr-TR";
    recognition.interimResults = true;
    recognition.continuous = false;
    recognition.maxAlternatives = 1;
    let finalText = "";
    recognition.onstart = () => { setStatus("listening"); setError(""); };
    recognition.onresult = (event) => {
      let pending = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        if (event.results[i].isFinal) finalText += event.results[i][0].transcript;
        else pending += event.results[i][0].transcript;
      }
      setInterim(pending);
    };
    recognition.onerror = (event) => {
      if (event.error !== "aborted" && event.error !== "no-speech") setError(recognitionErrorMessage(event.error));
    };
    recognition.onend = () => {
      if (recognitionRef.current === recognition) recognitionRef.current = null;
      setInterim("");
      setStatus((current) => (current === "listening" ? "idle" : current));
      if (finalText.trim()) handleRef.current(finalText);
      else if (handsFreeRef.current) startHandsFreeRef.current();
    };
    recognitionRef.current = recognition;
    try { recognition.start(); } catch { setError("Mikrofon başlatılamadı. Tekrar dene."); }
  }, []);

  // Hands-free: listen continuously, act only on "Digitra, …".
  const startHandsFree = useCallback(() => {
    const Recognition = getRecognitionConstructor();
    if (!Recognition || !handsFreeRef.current || speakingRef.current || recognitionRef.current) return;
    const recognition = new Recognition();
    recognition.lang = "tr-TR";
    recognition.interimResults = true;
    recognition.continuous = true;
    recognition.maxAlternatives = 1;
    recognition.onresult = (event) => {
      let pending = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (!event.results[i].isFinal) { pending += transcript; continue; }
        const { woke, rest } = stripWakeWord(transcript);
        if (woke && rest) { setOpen(true); handleRef.current(rest); }
        else if (woke) {
          awaitingCommandUntilRef.current = Date.now() + WAKE_FOLLOW_UP_MS;
          setOpen(true);
          setStatus("listening");
        } else if (Date.now() < awaitingCommandUntilRef.current && transcript.trim()) {
          awaitingCommandUntilRef.current = 0;
          handleRef.current(transcript);
        }
      }
      setInterim(Date.now() < awaitingCommandUntilRef.current || stripWakeWord(pending).woke ? pending : "");
    };
    recognition.onerror = (event) => {
      if (event.error === "not-allowed" || event.error === "service-not-allowed") {
        handsFreeRef.current = false;
        writePrefs({ voice: voiceRepliesRef.current, handsFree: false });
        setError(recognitionErrorMessage(event.error));
      }
    };
    recognition.onend = () => {
      if (recognitionRef.current === recognition) recognitionRef.current = null;
      setInterim("");
      // Browsers end continuous sessions after a silence; keep the ear open.
      if (handsFreeRef.current && !speakingRef.current) window.setTimeout(() => startHandsFreeRef.current(), 300);
    };
    recognitionRef.current = recognition;
    try { recognition.start(); } catch { recognitionRef.current = null; }
  }, []);
  useEffect(() => { startHandsFreeRef.current = startHandsFree; }, [startHandsFree]);

  useEffect(() => {
    handsFreeRef.current = handsFree;
    if (handsFree) startHandsFree();
    else if (recognitionRef.current?.continuous) stopListening();
  }, [handsFree, startHandsFree, stopListening]);

  useEffect(() => () => { recognitionRef.current?.abort(); stopSpeaking(); }, []);

  const toggleMic = () => {
    if (status === "listening" && !recognitionRef.current?.continuous) { recognitionRef.current?.stop(); return; }
    if (recognitionRef.current?.continuous) {
      // Hands-free is on: treat the click as "I'm talking to you now".
      awaitingCommandUntilRef.current = Date.now() + WAKE_FOLLOW_UP_MS;
      setStatus("listening");
      return;
    }
    listenOnce();
  };

  // Alt+V anywhere: open the panel and start listening.
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.altKey && !event.ctrlKey && !event.metaKey && event.code === "KeyV") {
        event.preventDefault();
        setOpen(true);
        listenOnce();
      } else if (event.key === "Escape") {
        setOpen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [listenOnce]);

  useEffect(() => { if (open) inputRef.current?.focus({ preventScroll: true }); }, [open]);

  const toggleVoice = () => {
    const next = !voiceReplies;
    voiceRepliesRef.current = next;
    if (!next) { stopSpeaking(); speakingRef.current = false; setStatus((current) => (current === "speaking" ? "idle" : current)); }
    writePrefs({ voice: next, handsFree });
  };

  const toggleHandsFree = () => {
    const next = !handsFree;
    writePrefs({ voice: voiceReplies, handsFree: next });
    if (next) say("Eller serbest mod açık. Bana 'Digitra' diye başlayarak seslenebilirsin.");
  };

  const statusLabel =
    status === "listening" ? "Dinliyorum…" :
      status === "thinking" ? "Düşünüyorum…" :
        status === "speaking" ? "Konuşuyorum…" :
          handsFree ? "“Digitra” de ve konuş" : "Hazır";

  return (
    <>
      <AnimatePresence>
        {open && (
          <motion.div
            role="dialog"
            aria-label="Digitra sesli asistanı"
            initial={{ opacity: 0, y: 16, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 16, scale: 0.98 }}
            transition={{ duration: 0.22, ease: [0.16, 1, 0.3, 1] }}
            className="fixed bottom-24 right-4 z-[70] flex max-h-[min(640px,calc(100dvh-8rem))] w-[calc(100vw-2rem)] max-w-[380px] flex-col overflow-hidden rounded-3xl border border-border bg-surface shadow-2xl sm:right-6"
          >
            <header className="flex items-center justify-between gap-3 border-b border-border px-4 py-3">
              <div className="flex items-center gap-3">
                <span className="relative flex h-9 w-9 items-center justify-center rounded-full text-white" style={{ background: "var(--grad-primary)" }}>
                  <AudioLines size={17} />
                  {status !== "idle" && <span className="absolute inset-0 animate-ping rounded-full bg-primary/30" />}
                </span>
                <div>
                  <p className="text-sm font-semibold leading-tight">Digitra asistanı</p>
                  <p className="text-xs text-muted" aria-live="polite">{statusLabel}</p>
                </div>
              </div>
              <div className="flex items-center gap-1">
                <button
                  type="button"
                  onClick={toggleHandsFree}
                  disabled={!micSupported}
                  aria-pressed={handsFree}
                  title={handsFree ? "Eller serbest modu kapat" : "Eller serbest mod: “Digitra, …” diye seslen"}
                  className={cn("flex h-8 w-8 cursor-pointer items-center justify-center rounded-full disabled:opacity-35", handsFree ? "bg-primary/10 text-primary" : "text-muted hover:bg-surface-2")}
                >
                  {handsFree ? <Ear size={16} /> : <EarOff size={16} />}
                </button>
                <button
                  type="button"
                  onClick={toggleVoice}
                  aria-pressed={voiceReplies}
                  title={voiceReplies ? "Sesli yanıtları kapat" : "Sesli yanıtları aç"}
                  className={cn("flex h-8 w-8 cursor-pointer items-center justify-center rounded-full", voiceReplies ? "text-primary" : "text-muted hover:bg-surface-2")}
                >
                  {voiceReplies ? <Volume2 size={16} /> : <VolumeX size={16} />}
                </button>
                <button type="button" onClick={() => setOpen(false)} aria-label="Asistanı kapat" className="flex h-8 w-8 cursor-pointer items-center justify-center rounded-full text-muted hover:bg-surface-2">
                  <X size={16} />
                </button>
              </div>
            </header>

            <div ref={listRef} className="min-h-[180px] flex-1 space-y-3 overflow-y-auto px-4 py-4" aria-live="polite">
              {messages.length === 0 && (
                <div className="space-y-3">
                  <p className="text-sm leading-relaxed text-muted">
                    Merhaba! Mikrofona dokunup konuşabilir ya da yazabilirsin. Kelimeleri 3D ellerle
                    işaretlerim, kamerayı yönetirim ve seni bölümlere götürürüm.
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {SUGGESTIONS.map((suggestion) => (
                      <button key={suggestion} type="button" onClick={() => handle(suggestion)} className="cursor-pointer rounded-full border border-border px-3 py-1.5 text-xs text-muted transition hover:border-primary/40 hover:text-foreground">
                        {suggestion}
                      </button>
                    ))}
                  </div>
                </div>
              )}
              {messages.map((message) => (
                <div key={message.id} className={cn("flex", message.role === "user" ? "justify-end" : "justify-start")}>
                  <p className={cn(
                    "max-w-[85%] rounded-2xl px-3.5 py-2 text-sm leading-relaxed",
                    message.role === "user" ? "rounded-br-md bg-primary text-white" : "rounded-bl-md bg-surface-2",
                  )}>
                    {message.text}
                  </p>
                </div>
              ))}
              {interim && (
                <div className="flex justify-end">
                  <p className="max-w-[85%] rounded-2xl rounded-br-md border border-dashed border-primary/40 px-3.5 py-2 text-sm italic text-muted">{interim}</p>
                </div>
              )}
              {status === "thinking" && (
                <div className="flex items-center gap-2 text-xs text-muted"><LoaderCircle size={14} className="animate-spin" /> Yanıt hazırlanıyor</div>
              )}
            </div>

            {error && <p role="alert" className="border-t border-border px-4 py-2 text-xs text-rose-500">{error}</p>}

            <form
              onSubmit={(event) => { event.preventDefault(); handle(draft); setDraft(""); }}
              className="flex items-center gap-2 border-t border-border p-3"
            >
              <button
                type="button"
                onClick={toggleMic}
                disabled={!micSupported}
                aria-label={status === "listening" ? "Dinlemeyi durdur" : "Konuş"}
                title={micSupported ? "Konuş (Alt+V)" : "Bu tarayıcı ses tanımayı desteklemiyor (Chrome ya da Edge önerilir)"}
                className={cn(
                  "flex h-10 w-10 shrink-0 cursor-pointer items-center justify-center rounded-full text-white transition disabled:cursor-not-allowed disabled:opacity-40",
                  status === "listening" ? "animate-pulse bg-rose-500" : "",
                )}
                style={status === "listening" ? undefined : { background: "var(--grad-primary)" }}
              >
                {status === "listening" ? <MicOff size={17} /> : <Mic size={17} />}
              </button>
              <input
                ref={inputRef}
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                maxLength={500}
                placeholder="Bir şey sor ya da komut ver…"
                aria-label="Asistana mesaj"
                className="min-w-0 flex-1 rounded-full border border-border bg-surface-2 px-4 py-2.5 text-sm outline-none focus:border-primary"
              />
              <button type="submit" disabled={!draft.trim()} aria-label="Gönder" className="flex h-10 w-10 shrink-0 cursor-pointer items-center justify-center rounded-full text-muted hover:text-primary disabled:opacity-35">
                <Send size={17} />
              </button>
            </form>
            <p className="px-4 pb-3 text-[11px] leading-relaxed text-muted">
              {micSupported ? "Sesin tarayıcının ses tanımasıyla işlenir; Digitra kaydetmez." : "Ses tanıma bu tarayıcıda yok; yazarak kullanabilirsin."}
              {llmReady === false && " Serbest sorular kapalı, komutlar çalışır."}
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        aria-label={open ? "Asistanı kapat" : "Sesli asistanı aç"}
        aria-expanded={open}
        title="Sesli asistan (Alt+V)"
        className="fixed bottom-5 right-4 z-[70] flex h-14 w-14 cursor-pointer items-center justify-center rounded-full text-white shadow-xl transition-transform duration-300 hover:-translate-y-0.5 sm:right-6"
        style={{ background: "var(--grad-primary)" }}
      >
        {status === "listening" || status === "speaking"
          ? <span className="absolute inset-0 animate-ping rounded-full bg-primary/35" aria-hidden />
          : null}
        {open ? <X size={22} /> : <AudioLines size={22} />}
        {handsFree && !open && <span className="absolute right-0.5 top-0.5 h-3 w-3 rounded-full border-2 border-white bg-emerald-400" aria-hidden />}
      </button>
    </>
  );
}
