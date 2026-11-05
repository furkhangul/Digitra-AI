// Browser speech: Web Speech API recognition (tr-TR) and speechSynthesis.
// Both run client-side; nothing is recorded or stored by Digitra.

type RecognitionAlternative = { transcript: string; confidence: number };
type RecognitionResultItem = { isFinal: boolean; length: number; [index: number]: RecognitionAlternative };
export type RecognitionEvent = { resultIndex: number; results: { length: number; [index: number]: RecognitionResultItem } };
export type RecognitionErrorEvent = { error: string };

export interface BrowserRecognition {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  maxAlternatives: number;
  onresult: ((event: RecognitionEvent) => void) | null;
  onerror: ((event: RecognitionErrorEvent) => void) | null;
  onend: (() => void) | null;
  onstart: (() => void) | null;
  start(): void;
  stop(): void;
  abort(): void;
}

type RecognitionConstructor = new () => BrowserRecognition;

export function getRecognitionConstructor(): RecognitionConstructor | null {
  if (typeof window === "undefined") return null;
  const w = window as unknown as { SpeechRecognition?: RecognitionConstructor; webkitSpeechRecognition?: RecognitionConstructor };
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null;
}

export function recognitionErrorMessage(code: string) {
  switch (code) {
    case "not-allowed":
    case "service-not-allowed":
      return "Mikrofon izni verilmedi. Adres çubuğundaki kilit simgesinden mikrofona izin ver.";
    case "no-speech":
      return "Ses algılanmadı. Mikrofona biraz daha yakın konuşmayı dene.";
    case "audio-capture":
      return "Mikrofon bulunamadı. Bağlı olduğundan emin ol.";
    case "network":
      return "Ses tanıma servisine ulaşılamadı. İnternet bağlantını kontrol et.";
    case "language-not-supported":
      return "Tarayıcın Türkçe ses tanımayı desteklemiyor.";
    default:
      return "Ses tanıma durdu. Tekrar dene.";
  }
}

export function speechSynthesisSupported() {
  return typeof window !== "undefined" && "speechSynthesis" in window;
}

let cachedVoice: SpeechSynthesisVoice | null | undefined;

function rankVoice(voice: SpeechSynthesisVoice) {
  let score = 0;
  if (/natural|online|neural/i.test(voice.name)) score += 4;
  if (/google/i.test(voice.name)) score += 2;
  if (voice.localService) score += 1;
  return score;
}

export function pickTurkishVoice(): SpeechSynthesisVoice | null {
  if (!speechSynthesisSupported()) return null;
  if (cachedVoice !== undefined) return cachedVoice;
  const voices = window.speechSynthesis.getVoices();
  if (!voices.length) return null; // Not loaded yet; retry on next call.
  const turkish = voices.filter((voice) => voice.lang.toLowerCase().replace("_", "-").startsWith("tr"));
  cachedVoice = turkish.sort((a, b) => rankVoice(b) - rankVoice(a))[0] ?? null;
  return cachedVoice;
}

if (speechSynthesisSupported()) {
  window.speechSynthesis.addEventListener?.("voiceschanged", () => { cachedVoice = undefined; });
}

export function speak(text: string, { rate = 1, onEnd }: { rate?: number; onEnd?: () => void } = {}) {
  if (!speechSynthesisSupported() || !text.trim()) { onEnd?.(); return; }
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "tr-TR";
  utterance.rate = rate;
  const voice = pickTurkishVoice();
  if (voice) utterance.voice = voice;
  let finished = false;
  const done = () => { if (!finished) { finished = true; onEnd?.(); } };
  utterance.onend = done;
  utterance.onerror = done;
  window.speechSynthesis.speak(utterance);
}

export function stopSpeaking() {
  if (speechSynthesisSupported()) window.speechSynthesis.cancel();
}
