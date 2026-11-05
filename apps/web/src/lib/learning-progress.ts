// Per-letter learning progress, kept only in this browser. Quiz answers and
// live-training attempts feed it; the 3D lesson uses it to suggest the letters
// that need the most practice.

import { useSyncExternalStore } from "react";
import { TID_LETTERS } from "@/lib/tid-text";

export type LetterStats = { attempts: number; correct: number; last: number };
export type Progress = Record<string, LetterStats>;
export type Recommendation = { letter: string; reason: string };

const KEY = "digitra-progress-v1";
const EVENT = "digitra-progress-change";
const EMPTY: Progress = {};

let cachedRaw: string | null = null;
let cachedProgress: Progress = EMPTY;

function readRaw() {
  try { return window.localStorage.getItem(KEY) ?? ""; } catch { return ""; }
}

function parse(raw: string): Progress {
  if (raw === cachedRaw) return cachedProgress;
  cachedRaw = raw;
  try {
    const value = JSON.parse(raw || "{}");
    cachedProgress = value && typeof value === "object" ? value : EMPTY;
  } catch {
    cachedProgress = EMPTY;
  }
  return cachedProgress;
}

function subscribe(callback: () => void) {
  window.addEventListener(EVENT, callback);
  window.addEventListener("storage", callback);
  return () => {
    window.removeEventListener(EVENT, callback);
    window.removeEventListener("storage", callback);
  };
}

function write(progress: Progress) {
  try { window.localStorage.setItem(KEY, JSON.stringify(progress)); } catch { /* private mode: progress stays in-session only */ }
  window.dispatchEvent(new Event(EVENT));
}

export function recordAttempt(letter: string, correct: boolean) {
  const key = letter.toLocaleUpperCase("tr-TR");
  if (!(TID_LETTERS as readonly string[]).includes(key)) return;
  const progress = { ...parse(readRaw()) };
  const previous = progress[key] ?? { attempts: 0, correct: 0, last: 0 };
  progress[key] = { attempts: previous.attempts + 1, correct: previous.correct + (correct ? 1 : 0), last: Date.now() };
  write(progress);
}

export function resetProgress() {
  write({});
}

export function useLearningProgress(): Progress {
  return parse(useSyncExternalStore(subscribe, readRaw, () => ""));
}

export function accuracy(stats: LetterStats | undefined) {
  return stats && stats.attempts ? stats.correct / stats.attempts : null;
}

/** Weakest practised letters first, then letters never tried, in alphabet order. */
export function recommendLetters(progress: Progress, count = 3): Recommendation[] {
  const practised = TID_LETTERS
    .filter((letter) => progress[letter]?.attempts)
    .map((letter) => ({ letter, rate: accuracy(progress[letter]) ?? 0, attempts: progress[letter].attempts }))
    .filter(({ rate }) => rate < 0.8)
    .sort((a, b) => a.rate - b.rate || b.attempts - a.attempts)
    .map(({ letter, rate }) => ({ letter, reason: `%${Math.round(rate * 100)} doğru` }));
  const unseen = TID_LETTERS
    .filter((letter) => !progress[letter]?.attempts)
    .map((letter) => ({ letter, reason: "Henüz çalışılmadı" }));
  return [...practised, ...unseen].slice(0, count);
}

export function progressSummary(progress: Progress) {
  const practised = TID_LETTERS.filter((letter) => progress[letter]?.attempts);
  const mastered = practised.filter((letter) => (accuracy(progress[letter]) ?? 0) >= 0.8 && progress[letter].attempts >= 2);
  return { practised: practised.length, mastered: mastered.length, total: TID_LETTERS.length };
}
