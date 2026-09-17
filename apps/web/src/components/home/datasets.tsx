"use client";

import { motion } from "framer-motion";
import { Database, ExternalLink } from "lucide-react";
import { Container, Section } from "@/components/ui/container";
import { Reveal, RevealGroup, revealItem } from "@/components/ui/reveal";
import { Badge } from "@/components/ui/badge";
import { TiltCard } from "@/components/ui/tilt-card";

const DATASETS = [
  {
    id: "A",
    name: "TİD görüntüleri",
    source: "Kaggle · feronial",
    role: "29 harfi kapsayan gerçek el görüntülerinden 1.969 örnek, V6 eğitiminin temelini oluşturur.",
    usage: "Eğitim",
  },
  {
    id: "B",
    name: "Doğrulama verisi",
    source: "TİD · 574 gerçek görüntü",
    role: "Model seçimi ve tanıma ayarları, eğitimden ayrı tutulan bu bölüm üzerinden değerlendirilir.",
    usage: "Model seçimi",
  },
  {
    id: "C",
    name: "Tarihsel test",
    source: "TİD · 431 gerçek görüntü",
    role: "V5 ve V6 görüntü modellerinin raporlanan sonuçları bu gruba aittir. Yeni bir dış test değildir.",
    usage: "Değerlendirme",
  },
  {
    id: "D",
    name: "3D el görselleri",
    source: "Digitra · 330 yardımcı görüntü",
    role: "Sana harfleri gösteren aynı 3D el pozlarından üretilir. Gerçek veriyi desteklemek için model eğitiminde düşük ağırlıkla kullanılır.",
    usage: "Eğitim desteği",
  },
  {
    id: "E",
    name: "ASL araştırmaları",
    source: "Digitra · Önceki model çalışmaları",
    role: "İlk el noktası ve görüntü modelleri ASL ile geliştirildi. Bu sonuçlar TİD başarısı olarak sunulmaz.",
    usage: "Araştırma geçmişi",
  },
];

export function Datasets() {
  return (
    <Section id="datasets">
      <Container>
        <Reveal className="mx-auto max-w-2xl text-center">
          <span className="text-xs font-medium uppercase tracking-[0.2em] text-primary">
            Verinin kaynağı
          </span>
          <h2 className="mt-4 text-4xl font-semibold tracking-tight sm:text-5xl">
            Gerçek ellerle öğrenir, 3D ile desteklenir.
          </h2>
          <p className="mt-4 text-lg text-muted">
            Sana rehberlik eden 3D eller, yapay zekâ eğitimine de yardımcı olur.
            Aynı el pozlarından farklı açı ve ışıklarda görseller üretiyoruz.
            Bu görseller, V6&apos;nın temelini oluşturan 2.974 gerçek görüntünün
            eğitim bölümünü destekler; doğrulama ve test gerçek görüntülerle yapılır.
          </p>
        </Reveal>

        <RevealGroup className="mt-16 grid grid-cols-1 gap-4 lg:grid-cols-5">
          {DATASETS.map((d) => (
            <motion.div key={d.id} variants={revealItem}>
              <TiltCard className="card-hover flex h-full flex-col rounded-3xl border border-border bg-surface p-6">
                <div className="flex items-center justify-between">
                  <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-sm font-semibold text-primary">
                    {d.id}
                  </div>
                  <Database size={16} className="text-muted" />
                </div>
                <h3 className="mt-4 text-base font-semibold">{d.name}</h3>
                <p className="mt-1 text-xs text-muted">{d.source}</p>
                <p className="mt-3 flex-1 text-sm leading-relaxed text-muted">{d.role}</p>
                <Badge variant="outline" className="mt-4 w-fit">
                  {d.usage}
                </Badge>
              </TiltCard>
            </motion.div>
          ))}
        </RevealGroup>

        <Reveal delay={0.1} className="mt-10 flex items-start gap-3 rounded-2xl border border-border bg-surface-2 p-5">
          <ExternalLink size={16} className="mt-0.5 shrink-0 text-muted" />
          <p className="text-sm leading-relaxed text-muted">
            3D el gösterimleri TİD alfabe kaynaklarından uyarlanmıştır; uzman doğrulaması
            devam eden bir geliştirme alanıdır. Görüntü testi, hareket tanıma ve öğrenme
            deneyiminin değerlendirmeleri ayrı ele alınır.
          </p>
        </Reveal>
      </Container>
    </Section>
  );
}
