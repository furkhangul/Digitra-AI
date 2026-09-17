"use client";

import { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown } from "lucide-react";
import { Container, Section } from "@/components/ui/container";
import { Reveal } from "@/components/ui/reveal";
import { cn } from "@/lib/utils";

const FAQS = [
  {
    q: "Digitra ile neler yapabilirim?",
    a: "TİD parmak alfabesindeki 29 harfi 3D ellerle inceleyebilir, gösterimi yavaşlatıp tekrarlayabilir ve kamerayla kelimeleri harf harf çalışabilirsin. Canlı tanımada oluşan metni kontrol edip sesli dinleyebilirsin. Öğretimdeki 3D modeller, tanıma modelinin eğitimine yardımcı görseller üretmek için de kullanılır. Kişisel hareket düzeltmeleri sunan 3D eğitmen ise geliştirme hedefimizdir. Digitra şu an parmak alfabesine odaklanır; TİD kelime işaretlerini veya cümle çevirisini kapsamaz.",
  },
  {
    q: "Kamera görüntülerim kaydediliyor mu?",
    a: "Mevcut tanıma akışında görüntüler kaydedilmez. El tespiti tarayıcıda yapılır; elleri çevreleyen görüntü kırpımı ve el noktaları tanıma servisine gönderilip bellekte işlenir. Tam kamera karesi gönderilmez. Kamerayı paneldeki durdur düğmesiyle kapatabilirsin.",
  },
  {
    q: "Harf tanıma ne kadar başarılı?",
    a: "V6 görüntü modeli, 431 gerçek görüntüden oluşan tarihsel testte %90,26 doğruluk ve %90,23 macro F1 elde etti. Bu grup daha önce V5 için de kullanıldı; yeni kişilerle yapılmış bağımsız bir test değil. Sonuçlar canlı kamera veya hareket tanıma başarısını göstermez. Işık, kamera açısı ve ellerin görünürlüğü tahminleri etkileyebilir.",
  },
  {
    q: "İlk kez kullanıyorum. Nereden başlamalıyım?",
    a: "Eğitim bölümünden bir harf seç. 3D ellerde parmakların yerleşimini incele; hareketi yavaşlatıp tekrar oynat. Kendini dene bölümünde harfleri ayırt etmeyi çalış, ardından canlı eğitimde kısa kelimeleri kamerayla harf harf tamamla. 3D gösterimler kaynaklardan uyarlanmıştır; TİD uzmanı doğrulaması henüz tamamlanmamıştır.",
  },
  {
    q: "Kamerayla pratik için ne gerekiyor?",
    a: "Kameralı bir cihaz, kamera izni verilmiş bir tarayıcı ve çalışan tanıma servisi gerekiyor. Ellerin tamamını kadrajda tut ve yeterli ışık kullan. Model bağlantısı kapalıysa canlı tanıma çalışmaz. Mevcut sürüm, tamamen tarayıcı içinde çalışan çevrimdışı tanıma sunmuyor.",
  },
];

export function FAQ() {
  const [open, setOpen] = useState<number | null>(0);

  return (
    <Section id="faq">
      <Container className="mx-auto max-w-3xl">
        <Reveal className="text-center">
          <span className="text-xs font-medium uppercase tracking-[0.2em] text-primary">
            Sık Sorulan Sorular
          </span>
          <h2 className="mt-4 text-4xl font-semibold tracking-tight sm:text-5xl">
            Merak edilenler.
          </h2>
        </Reveal>

        <Reveal delay={0.1} className="mt-14 flex flex-col gap-3">
          {FAQS.map((item, i) => {
            const isOpen = open === i;
            return (
              <div key={item.q} className="overflow-hidden rounded-2xl border border-border bg-surface">
                <button
                  onClick={() => setOpen(isOpen ? null : i)}
                  aria-expanded={isOpen}
                  className="flex w-full items-center justify-between gap-4 px-6 py-5 text-left"
                >
                  <span className="text-sm font-medium sm:text-base">{item.q}</span>
                  <ChevronDown
                    size={18}
                    className={cn("shrink-0 text-muted transition-transform duration-300", isOpen && "rotate-180 text-primary")}
                  />
                </button>
                <AnimatePresence initial={false}>
                  {isOpen && (
                    <motion.div
                      initial={{ height: 0, opacity: 0 }}
                      animate={{ height: "auto", opacity: 1 }}
                      exit={{ height: 0, opacity: 0 }}
                      transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
                    >
                      <p className="px-6 pb-5 text-sm leading-relaxed text-muted">{item.a}</p>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            );
          })}
        </Reveal>
      </Container>
    </Section>
  );
}
