// Archived single-hand diagnostic poses; NOT used for TID education.
type Pose = {
  thumb: number;
  index: number;
  middle: number;
  ring: number;
  pinky: number;
  spread?: { index?: number; middle?: number; ring?: number; pinky?: number };
};
type Letter = {
  ch: string;
  pose: Pose;
  tip: string;
  // G/Ğ/H are actually signed with the wrist turned side-on — the rig has no
  // wrist orientation, but rotating the rendered view in 2D reproduces the
  // same silhouette without fighting the 3D camera framing.
  screenRotate?: number;
};

// Simplified single-hand finger-alphabet shapes. Curl 0 = straight, 1 = fully
// folded. Approximations for the training UI — not TİD reference material.
export const ALPHABET: Letter[] = [
  { ch: "A", pose: { thumb: 0.02, index: 0.72, middle: 0.75, ring: 0.78, pinky: 0.8 }, tip: "Yumruk; başparmak yanda dik durur." },
  { ch: "B", pose: { thumb: 1.1, index: 0, middle: 0, ring: 0, pinky: 0 }, tip: "Dört parmak düz; başparmak avuca kapanır." },
  { ch: "C", pose: { thumb: 0.7, index: 0.56, middle: 0.56, ring: 0.56, pinky: 0.74 }, tip: "Tüm parmaklar C kavisinde açılır." },
  { ch: "Ç", pose: { thumb: 0.73, index: 0.6, middle: 0.6, ring: 0.6, pinky: 0.78 }, tip: "C şekli; el ileri doğru iki kez vurulur." },
  { ch: "D", pose: { thumb: 0.9, index: 0, middle: 1.1, ring: 1.1, pinky: 1.1 }, tip: "İşaret parmağı dik; diğerleri başparmağa değer." },
  { ch: "E", pose: { thumb: 1.1, index: 0.95, middle: 1, ring: 1, pinky: 0.98 }, tip: "Parmaklar başparmağın üzerine kıvrılır." },
  { ch: "F", pose: { thumb: 1, index: 1.04, middle: 0, ring: 0, pinky: 0 }, tip: "Başparmak–işaret halkası; üç parmak dik." },
  { ch: "G", pose: { thumb: 0.1, index: 0, middle: 1.1, ring: 1.1, pinky: 1.1 }, tip: "İşaret parmağı ile başparmak yatay açılır.", screenRotate: 90 },
  { ch: "Ğ", pose: { thumb: 0.1, index: 0, middle: 1.1, ring: 1.1, pinky: 1.1 }, tip: "G gibi yapılır; ses gırtlaktan uzatılarak hissedilir.", screenRotate: 90 },
  { ch: "H", pose: { thumb: 1.1, index: 0.08, middle: 0.08, ring: 1.1, pinky: 1.1 }, tip: "İşaret ve orta parmak yatay, bitişik durur.", screenRotate: 90 },
  { ch: "I", pose: { thumb: 1.1, index: 1.1, middle: 1.1, ring: 1.1, pinky: 0 }, tip: "Yalnızca serçe parmak dik kalır." },
  { ch: "İ", pose: { thumb: 1.1, index: 1.1, middle: 1.1, ring: 1.1, pinky: 0 }, tip: "I gibi; el öne doğru bir kez hareket eder." },
  { ch: "J", pose: { thumb: 1, index: 1, middle: 1, ring: 1, pinky: 0 }, tip: "I gibi; serçe parmakla havada J çizilir." },
  { ch: "K", pose: { thumb: 0.7, index: 0, middle: 0, ring: 1.1, pinky: 1.1, spread: { index: -0.28, middle: 0.28 } }, tip: "İşaret–orta dik V; başparmak aralarına geçer." },
  { ch: "L", pose: { thumb: 0, index: 0, middle: 1.1, ring: 1.1, pinky: 1.1 }, tip: "İşaret dik, başparmak yana açık — L şekli." },
  { ch: "M", pose: { thumb: 0.9, index: 0.85, middle: 1.1, ring: 1.1, pinky: 1.1 }, tip: "Üç parmak başparmağın üzerinden sarılır." },
  { ch: "N", pose: { thumb: 1.05, index: 0.82, middle: 1.1, ring: 1.1, pinky: 1.1 }, tip: "İki parmak başparmağın üzerinden sarılır." },
  { ch: "O", pose: { thumb: 0.8, index: 0.84, middle: 0.84, ring: 0.84, pinky: 0.84 }, tip: "Tüm parmaklar O halkası oluşturur." },
  { ch: "Ö", pose: { thumb: 0.8, index: 0.84, middle: 0.84, ring: 0.84, pinky: 0.84 }, tip: "O gibi; el öne doğru uzatılır." },
  { ch: "P", pose: { thumb: 0.65, index: 0.03, middle: 0.13, ring: 1.1, pinky: 1.1, spread: { index: -0.28, middle: 0.28 } }, tip: "K gibi yapılır; el aşağı bakar." },
  { ch: "R", pose: { thumb: 1.1, index: 0, middle: 0, ring: 1.1, pinky: 1.1, spread: { index: 0.16, middle: -0.16 } }, tip: "İşaret ve orta parmak birbirine çapraz." },
  { ch: "S", pose: { thumb: 1, index: 0.78, middle: 0.8, ring: 0.82, pinky: 0.84 }, tip: "Yumruk; başparmak önde parmakları kapatır." },
  { ch: "Ş", pose: { thumb: 0.35, index: 0.88, middle: 0.88, ring: 0.88, pinky: 0.88 }, tip: "Yumruk; başparmak ileri doğru uzatılır." },
  { ch: "T", pose: { thumb: 0.7, index: 1.04, middle: 1.1, ring: 1.1, pinky: 1.1 }, tip: "Başparmak işaret–orta parmağın arasına girer." },
  { ch: "U", pose: { thumb: 1.1, index: 0, middle: 0, ring: 1.1, pinky: 1.1 }, tip: "İşaret ve orta parmak dik, bitişik." },
  { ch: "Ü", pose: { thumb: 1.1, index: 0, middle: 0, ring: 1.1, pinky: 1.1 }, tip: "U gibi; el küçük daireler çizer." },
  { ch: "V", pose: { thumb: 1.1, index: 0, middle: 0, ring: 1.1, pinky: 1.1, spread: { index: -0.32, middle: 0.32 } }, tip: "İşaret ve orta parmak dik ve açık — V." },
  { ch: "Y", pose: { thumb: 0, index: 1.1, middle: 1.1, ring: 1.1, pinky: 0 }, tip: "Başparmak ve serçe açık, diğerleri kapalı." },
  { ch: "Z", pose: { thumb: 1, index: 0.15, middle: 1, ring: 1, pinky: 1 }, tip: "İşaret parmağı havada Z çizer."},
];

