"use client";

import { Camera, Hand, Volume2, PersonStanding } from "lucide-react";
import { Container } from "@/components/ui/container";
import { Reveal } from "@/components/ui/reveal";

// Sticky offsets increase per card, so each one parks slightly lower than the
// last and the stack stays readable as an edge-on deck while you scroll.
const CARDS = [
  {
    title: "3D Rehberle Canlı Pratik",
    body: "TİD harfini 3D ellerde incele, kendi ellerinle uygula ve kamerayla eşleşmesini takip et. Canlı eğitimde her eşleşme seni sıradaki harfe taşır; kelimeyi adım adım tamamlarsın.",
    icon: Camera,
    bg: "linear-gradient(135deg, #6d5bff 0%, #8f7bff 100%)",
    fg: "#ffffff",
  },
  {
    title: "Aynı Eller, İki Görev",
    body: "Sana rehberlik eden 3D eller, tanıma modelinin eğitimine de katkı sağlar. Aynı el pozlarından ürettiğimiz görseller, gerçek el görüntülerini destekler. Öğretim ve model geliştirme ortak bir kaynaktan beslenir.",
    icon: Hand,
    bg: "linear-gradient(135deg, #00d4b8 0%, #2fe6c8 100%)",
    fg: "#03211d",
  },
  {
    title: "Pratikten Anlatıma",
    body: "Çalıştığın harfleri canlı tanıma bölümünde birleştirerek kendi metnini oluştur. Metni kontrol et, düzelt ve istersen tarayıcının Türkçe konuşma desteğiyle seslendir.",
    icon: Volume2,
    bg: "linear-gradient(135deg, #b8a6ff 0%, #d8ccff 100%)",
    fg: "#1a1140",
  },
  {
    title: "Sırada: Sana Özel Rehberlik",
    body: "Digitra’yı, el konumuna ve hareket yönüne göre kişisel öneriler sunan bir 3D eğitmene dönüştürmeyi hedefliyoruz. “İki elini yaklaştır” gibi düzeltmeler bu geliştirme hedefinin parçası; henüz kullanıma açık değil.",
    icon: PersonStanding,
    bg: "linear-gradient(135deg, #f5a524 0%, #ffc866 100%)",
    fg: "#2b1a00",
  },
];

export function StackingCards() {
  return (
    <section className="relative py-20 sm:py-28">
      <Container>
        <Reveal className="mx-auto max-w-2xl text-center">
          <span className="text-xs font-medium uppercase tracking-[0.2em] text-primary">
            Öğren, uygula, anlat
          </span>
          <h2 className="mt-4 text-4xl font-semibold tracking-tight sm:text-5xl">
            Bir harfle başlayan yeni bir beceri.
          </h2>
        </Reveal>

        <div className="mt-16">
          {CARDS.map((card, i) => (
            <div
              key={card.title}
              className="sticky"
              style={{ top: `${7 + i * 2.5}rem`, paddingBottom: "1.5rem" }}
            >
              <div
                className="relative overflow-hidden rounded-[2rem] p-8 shadow-xl sm:p-12"
                style={{ background: card.bg, color: card.fg }}
              >
                <div className="noise-overlay absolute inset-0" />
                <div className="relative flex flex-col gap-6 sm:flex-row sm:items-start sm:justify-between">
                  <div className="max-w-xl">
                    <span className="text-sm font-medium opacity-70">0{i + 1}</span>
                    <h3 className="mt-2 text-3xl font-semibold tracking-tight sm:text-5xl">
                      {card.title}
                    </h3>
                    <p className="mt-4 text-base leading-relaxed opacity-85 sm:text-lg">
                      {card.body}
                    </p>
                  </div>
                  <card.icon
                    size={72}
                    strokeWidth={1.2}
                    className="shrink-0 opacity-90"
                    aria-hidden
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      </Container>
    </section>
  );
}
