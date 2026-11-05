// Rule-based Turkish intent parser for the voice assistant. It covers every
// action the site can take so the assistant works without any LLM; only
// utterances that match nothing are forwarded to the /assistant endpoint.

export type SectionId = "top" | "ceviri" | "egitim" | "how-it-works" | "architecture" | "science" | "datasets" | "faq";

export type Intent =
  | { kind: "stop" }
  | { kind: "help" }
  | { kind: "answer"; reply: string; section?: SectionId }
  | { kind: "navigate"; section: SectionId; reply: string }
  | { kind: "spell"; text: string }
  | { kind: "camera"; on: boolean }
  | { kind: "text"; op: "clear" | "backspace" | "space" }
  | { kind: "read" }
  | { kind: "feedback"; correct: string }
  | { kind: "theme"; theme: "light" | "dark" }
  | { kind: "quiz" }
  | { kind: "live-training" }
  | { kind: "time" }
  | { kind: "date" }
  | { kind: "unknown"; text: string };

export const SECTION_NAMES: Record<SectionId, string> = {
  top: "ana sayfa",
  ceviri: "harf tanıma",
  egitim: "3D eğitim",
  "how-it-works": "Neden Digitra",
  architecture: "teknoloji",
  science: "doğrulama",
  datasets: "veri kaynakları",
  faq: "sık sorulan sorular",
};

export const HELP_TEXT =
  "Şunları yapabilirim: bir kelimeyi 3D ellerle işaretlerim, örneğin 'merhaba kelimesini işaretle'. " +
  "Kamerayı açıp kapatırım, tanınan metni okurum ya da silerim. Bölümlere götürürüm, testi başlatırım, " +
  "temayı değiştiririm ve Digitra hakkındaki sorularını yanıtlarım.";

export const SUGGESTIONS = [
  "Merhaba kelimesini işaretle",
  "Kamerayı aç",
  "Yazdığım metni oku",
  "Beni test et",
  "Ne yapabilirsin?",
];

const WAKE_WORD = /^(hey |hey,? )?(di[gj]itra|digit ra|dijit ra)[,.!?:\s]*/;
const POLITE = /(?<=^|\s)(lütfen|rica etsem|bana|benim için|acaba)(?=\s|$)/g;

export function stripWakeWord(text: string) {
  const lowered = text.toLocaleLowerCase("tr-TR").trim();
  const match = lowered.match(WAKE_WORD);
  return match ? { woke: true, rest: text.trim().slice(match[0].length) } : { woke: false, rest: text.trim() };
}

function normalize(text: string) {
  return text
    .toLocaleLowerCase("tr-TR")
    .replace(/[“”"«»]/g, '"')
    .replace(/[.!?,;:]+(\s|$)/g, " ")
    .replace(POLITE, " ")
    .replace(/\s+/g, " ")
    .trim();
}

// "Merhaba'yı", "anneyi" → word itself. Only strips the accusative after an
// apostrophe or a vowel-buffer "y", which never removes part of a real stem
// in the short words people fingerspell.
function stripAccusative(word: string) {
  const apostrophe = word.match(/^(.+?)['’][a-zçğıöşü]*$/);
  if (apostrophe) return apostrophe[1];
  const buffered = word.match(/^(.*[aeıioöuü])y[ıiuü]$/);
  return buffered ? buffered[1] : word;
}

function cleanSpellTarget(raw: string) {
  const quoted = raw.match(/"([^"]+)"/);
  const text = (quoted ? quoted[1] : raw)
    .replace(/(?<=^|\s)(kelimesini|kelimesi|sözcüğünü|sözcüğü|ismini|adını|yazısını|cümlesini|harfini|harfi)(?=\s|$)/g, " ")
    .replace(/(?<=^|\s)(parmak alfabesiyle|işaret diliyle|tid ile|3d|ellerle|elle)(?=\s|$)/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (!text) return "";
  const words = text.split(" ");
  // Only the last word carries the case suffix in "X Y'yi işaretle".
  words[words.length - 1] = stripAccusative(words[words.length - 1]);
  return words.join(" ");
}

const SECTION_KEYWORDS: [RegExp, SectionId][] = [
  [/(harf tanıma|çeviri|tanıma bölüm|canlı tanıma)/, "ceviri"],
  [/(3d eğitim|eğitim|alfabe(yi)?( göster| aç)?$|öğren(me)? bölüm)/, "egitim"],
  [/(neden digitra|nasıl çalışır|nasıl çalışıyor)/, "how-it-works"],
  [/(teknoloji|mimari)/, "architecture"],
  [/(doğrulama|bilimsel|başarı oranı bölüm)/, "science"],
  [/(veri kaynak|veri seti|veriler)/, "datasets"],
  [/(sss|sık sorulan|soru cevap)/, "faq"],
  [/(en üst|başa dön|ana sayfa|yukarı çık|en başa)/, "top"],
];

const NAV_VERB = /(git|götür|aç|geç|göster|gel|dön|in|çık|kaydır|bölüm)/;

const LETTER = "(a|b|c|ç|d|e|f|g|ğ|h|ı|i|j|k|l|m|n|o|ö|p|r|s|ş|t|u|ü|v|y|z)";

export function parseIntent(input: string): Intent {
  const t = normalize(input);
  if (!t) return { kind: "unknown", text: input };

  if (/^(dur|sus|sessiz ol|yeter|kes|tamam dur|konuşmayı kes)$/.test(t)) return { kind: "stop" };
  if (/(ne(ler)? yapabil|yardım|komutlar|nasıl kullan|neler sorabilirim)/.test(t)) return { kind: "help" };

  // Recognition feedback: "yanlış tanıdı, doğrusu B" / "o harf B olacaktı".
  const feedback = t.match(new RegExp(`(yanlış|hatalı).*?(doğrusu|aslında|olmalıydı|olacaktı)?\\s${LETTER}( harfi)?( olmalıydı| olacaktı)?$`));
  if (feedback && /(yanlış|hatalı)/.test(t)) return { kind: "feedback", correct: feedback[3].toLocaleUpperCase("tr-TR") };

  // Camera.
  if (/(kamera|tanıma|harf tanımayı)(yı|y[ıi])?\s*(aç|başlat|çalıştır|etkinleştir)/.test(t)) return { kind: "camera", on: true };
  if (/(kamera|tanıma|harf tanımayı)(yı|y[ıi])?\s*(kapat|durdur|bitir)/.test(t)) return { kind: "camera", on: false };

  // Recognized text buffer.
  if (/((metn?i|metnimi|yazıyı|yazdıklarımı|tanınanları)\s*(temizle|sil|sıfırla)|hepsini sil)/.test(t)) return { kind: "text", op: "clear" };
  if (/(son harfi sil|harfi sil|geri al|bir harf sil)/.test(t)) return { kind: "text", op: "backspace" };
  if (/(boşluk (ekle|bırak|koy))/.test(t)) return { kind: "text", op: "space" };
  if (/(ne yazdım|(metn?i|metnimi|yazdığımı|yazdığım metni|yazıyı|tanınan metni|tanınanları)\s*(oku|söyle|seslendir|dinlet))/.test(t)) return { kind: "read" };

  // Theme.
  if (/(karanlık|koyu|gece)\s*(mod|tema|görünüm)/.test(t)) return { kind: "theme", theme: "dark" };
  if (/(aydınlık|açık|gündüz)\s*(mod|tema|görünüm)/.test(t)) return { kind: "theme", theme: "light" };

  if (/(beni (test et|sına|dene)|kendimi dene|test başlat|sınav|quiz|bilgimi ölç)/.test(t)) return { kind: "quiz" };
  if (/(canlı (eğitim|pratik|ders|alıştırma)|kelime pratiği|pratik yap)/.test(t)) return { kind: "live-training" };

  // Explicit spelling: "X kelimesini işaretle", "X'i parmak alfabesiyle göster",
  // "hecele X", "X nasıl işaretlenir".
  const spellAfter = t.match(/^(.+?)\s+(işaretle|işaret et|hecele|parmak alfabesiyle göster|işaret diliyle göster|elle göster|ellerle göster|nasıl (işaretlenir|yapılır|gösterilir|hecelenir|söylenir işaret dilinde))( bakalım)?$/);
  if (spellAfter) {
    const text = cleanSpellTarget(spellAfter[1]);
    if (text) return { kind: "spell", text };
  }
  const spellBefore = t.match(/^(işaretle|hecele|parmak alfabesiyle göster|işaret et)\s+(.+)$/);
  if (spellBefore) {
    const text = cleanSpellTarget(spellBefore[2]);
    if (text) return { kind: "spell", text };
  }
  const withWord = t.match(/^(.+?)\s+(kelimesini|sözcüğünü|ismini|adını|cümlesini)\s+(göster|oynat|yap)$/);
  if (withWord) return { kind: "spell", text: cleanSpellTarget(withWord[1]) };
  const letter = t.match(new RegExp(`^${LETTER}\\s*harfi(ni)?\\s*(göster|nasıl( yapılır| gösterilir)?|oynat|öğret)`));
  if (letter) return { kind: "spell", text: letter[1] };

  // Navigation.
  for (const [pattern, section] of SECTION_KEYWORDS) {
    if (pattern.test(t) && (NAV_VERB.test(t) || t.split(" ").length <= 3)) {
      return { kind: "navigate", section, reply: `${SECTION_NAMES[section][0].toLocaleUpperCase("tr-TR")}${SECTION_NAMES[section].slice(1)} bölümüne geçiyorum.` };
    }
  }

  // Knowledge the assistant can answer on its own.
  if (/^(merhaba|selam|selamlar|günaydın|iyi akşamlar|hey)$/.test(t)) {
    return { kind: "answer", reply: "Merhaba! Ben Digitra asistanı. Bir kelimeyi işaretlememi, kamerayı açmamı ya da seni bir bölüme götürmemi isteyebilirsin." };
  }
  if (/(teşekkür|sağ ol|eyvallah)/.test(t)) return { kind: "answer", reply: "Rica ederim. Başka bir şey istersen buradayım." };
  if (/(sen kimsin|adın ne|kimsin)/.test(t)) return { kind: "answer", reply: "Ben Digitra'nın sesli asistanıyım. Türk İşaret Dili parmak alfabesini öğrenmene ve siteyi sesle kullanmana yardım ediyorum." };
  if (/(digitra nedir|digitra ne(dir)? |bu site ne|ne işe yarıyor|amacı ne)/.test(t)) {
    return { kind: "answer", reply: "Digitra, Türk İşaret Dili parmak alfabesini 3D ellerle öğreten ve kamerayla gösterdiğin harfleri tanıyıp metne çeviren bir platform." };
  }
  if (/(kaç harf|harf sayısı|alfabede kaç)/.test(t)) {
    return { kind: "answer", reply: "TİD parmak alfabesinde 29 harf var. Q, W ve X bulunmaz; bunları K, V ve KS olarak hecelerim.", section: "egitim" };
  }
  if (/(hareketli harf|iki elli|hangi harfler hareket)/.test(t)) {
    return { kind: "answer", reply: "Ç, Ğ, İ, J, Ö, Ş ve Ü hareketli ya da iki elli gösterilir. Hareket tanıma şu an deneme aşamasında." };
  }
  if (/(doğruluk|başarı oranı|ne kadar doğru|isabet)/.test(t)) {
    return { kind: "answer", reply: "Harf tanımada kullanılan V6 modeli, 431 gerçek test görüntüsünde yüzde 90,3 doğruluk ve yüzde 90,2 macro F1 aldı. Bu test yeni kişilerle yapılmadığı için kişi bağımsız bir ölçüm değil.", section: "science" };
  }
  if (/(gizlilik|kaydediliyor mu|kayıt ediyor mu|görüntüm|verilerim|kamera güvenli)/.test(t)) {
    return { kind: "answer", reply: "Kameranın tamamı gönderilmez. Yalnızca el çevresindeki küçük kırpım yerel tanıma servisinde bellekte işlenir ve kaydedilmez. Sesin de tarayıcının ses tanımasıyla işlenir." };
  }
  if (/(cümle çevir|tam çeviri|türk işaret diline çevir|işaret diline çevir)/.test(t)) {
    return { kind: "answer", reply: "Digitra şimdilik tam TİD cümle çevirisi yapmıyor. Söylediğin kelimeleri parmak alfabesiyle harf harf gösterebilirim." };
  }
  if (/(saat kaç|saati söyle)/.test(t)) return { kind: "time" };
  if (/(bugün (ayın kaçı|günlerden ne|ne günü)|tarih ne|bugünün tarihi)/.test(t)) return { kind: "date" };

  // Bare "X göster" as a last structured guess: fingerspell it.
  const bareShow = t.match(/^(.{1,40}?)\s+(göster|oynat)$/);
  if (bareShow && !/(kamera|bölüm|sayfa|menü)/.test(bareShow[1])) {
    const text = cleanSpellTarget(bareShow[1]);
    if (text) return { kind: "spell", text };
  }

  return { kind: "unknown", text: input };
}
