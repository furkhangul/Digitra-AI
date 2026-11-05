// Turns free text into the letter sequence the 3D fingerspelling player can
// show. TİD's manual alphabet has 29 letters; everything else is either
// mapped to its closest TİD spelling or reported as skipped.

export const TID_LETTERS = [
  "A", "B", "C", "Ç", "D", "E", "F", "G", "Ğ", "H", "I", "İ", "J", "K", "L",
  "M", "N", "O", "Ö", "P", "R", "S", "Ş", "T", "U", "Ü", "V", "Y", "Z",
] as const;

const TID_SET = new Set<string>(TID_LETTERS);

// Loan letters are spelled the way Turkish writes them.
const SUBSTITUTIONS: Record<string, string> = { Â: "A", Î: "İ", Û: "U", Q: "K", W: "V", X: "KS" };

const DIGITS = ["SIFIR", "BİR", "İKİ", "ÜÇ", "DÖRT", "BEŞ", "ALTI", "YEDİ", "SEKİZ", "DOKUZ"];

export type SpellToken =
  | { kind: "letter"; ch: string; source: string }
  | { kind: "space" };

export type SpellPlan = { tokens: SpellToken[]; skipped: string[]; substituted: string[] };

export function planFingerspelling(text: string): SpellPlan {
  const tokens: SpellToken[] = [];
  const skipped: string[] = [];
  const substituted: string[] = [];
  const pushSpace = () => {
    if (tokens.length && tokens[tokens.length - 1].kind !== "space") tokens.push({ kind: "space" });
  };

  for (const raw of text.normalize("NFC").toLocaleUpperCase("tr-TR")) {
    if (/\s/.test(raw)) { pushSpace(); continue; }
    if (TID_SET.has(raw)) { tokens.push({ kind: "letter", ch: raw, source: raw }); continue; }
    if (SUBSTITUTIONS[raw]) {
      for (const ch of SUBSTITUTIONS[raw]) tokens.push({ kind: "letter", ch, source: raw });
      substituted.push(raw);
      continue;
    }
    if (/[0-9]/.test(raw)) {
      pushSpace();
      for (const ch of DIGITS[Number(raw)]) tokens.push({ kind: "letter", ch, source: raw });
      pushSpace();
      continue;
    }
    // Apostrophes join suffixes ("Ayşe'ye"); other punctuation only separates.
    if (/['’]/.test(raw)) continue;
    if (/[\p{L}]/u.test(raw)) skipped.push(raw);
    else pushSpace();
  }
  while (tokens.length && tokens[tokens.length - 1].kind === "space") tokens.pop();
  return { tokens, skipped: [...new Set(skipped)], substituted: [...new Set(substituted)] };
}
