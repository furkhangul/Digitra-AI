"use client";

import { motion } from "framer-motion";
import { Keyboard, Eye, Contrast, Captions, Waves, Ear } from "lucide-react";
import { Container, Section } from "@/components/ui/container";
import { Reveal, RevealGroup, revealItem } from "@/components/ui/reveal";

const FEATURES = [
  { icon: Keyboard, title: "Klavye ile kullanım", desc: "Harf seçimi ve oynatma düğmeleri gibi temel kontrolleri klavyenle kullanabilirsin." },
  { icon: Eye, title: "Açıklayıcı etiketler", desc: "Kontrol etiketleri ve metinle verilen durum bilgileri, ekran okuyucuyla kullanımı destekler." },
  { icon: Contrast, title: "Açık ve koyu tema", desc: "Görünümü kendin seçebilir veya cihazının tema tercihini kullanabilirsin." },
  { icon: Captions, title: "Metinle takip", desc: "Tanınan harfleri ve çalışma yönergelerini ekranda okuyabilirsin; seslendirme isteğe bağlıdır." },
  { icon: Waves, title: "Kendi hızında öğren", desc: "3D harf gösterimini duraklatabilir, yavaşlatabilir ve istediğin kadar tekrar edebilirsin." },
  { icon: Ear, title: "Görünür geri bildirim", desc: "Harf eşleşmelerini ve kamera durumunu yazılı mesajlarla takip edebilirsin." },
];

export function Accessibility() {
  return (
    <Section className="mesh-bg">
      <Container>
        <Reveal className="mx-auto max-w-2xl text-center">
          <span className="text-xs font-medium uppercase tracking-[0.2em] text-primary">
            Erişilebilirlik
          </span>
          <h2 className="mt-4 text-4xl font-semibold tracking-tight sm:text-5xl">
            Öğrenme deneyimini kendine uyarla.
          </h2>
          <p className="mt-4 text-lg text-muted">
            Görerek incele, metinle takip et, istersen sesli dinle.
            Digitra, parmak alfabesini kendi hızında çalışabilmen için farklı
            kullanım seçeneklerini bir araya getirir.
          </p>
        </Reveal>

        <RevealGroup className="mt-16 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {FEATURES.map((f) => (
            <motion.div
              key={f.title}
              variants={revealItem}
              className="card-hover rounded-3xl border border-border bg-surface p-6"
            >
              <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-primary/10 text-primary">
                <f.icon size={20} strokeWidth={1.75} />
              </div>
              <h3 className="mt-4 text-base font-semibold">{f.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-muted">{f.desc}</p>
            </motion.div>
          ))}
        </RevealGroup>
      </Container>
    </Section>
  );
}
