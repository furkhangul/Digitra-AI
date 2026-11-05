"use client";

import { motion } from "framer-motion";
import { BarChart3, Layers3, Timer, Zap } from "lucide-react";
import { Container, Section } from "@/components/ui/container";
import { Reveal, RevealGroup, revealItem } from "@/components/ui/reveal";
import { Badge } from "@/components/ui/badge";

// Source: models/digitra-tid-augmented-v6/evaluation.json (locked test split).
const METRICS = [
  { icon: BarChart3, value: "%90,26", label: "Test doğruluğu", note: "%95 GA: %87,2–%93,0" },
  { icon: Layers3, value: "%90,23", label: "Test macro F1", note: "29 harf ortalaması" },
  { icon: Timer, value: "20,4 ms", label: "Model gecikmesi", note: "Medyan · p95 31,4 ms" },
  { icon: Zap, value: "431", label: "Gerçek test görüntüsü", note: "Eğitimde kullanılmadı" },
];

export function LiveStatistics() {
  return (
    <Section className="relative">
      <Container>
        <div className="grid gap-14 lg:grid-cols-[0.9fr_1.1fr] lg:items-center">
          <Reveal>
            <span className="text-xs font-medium uppercase tracking-[0.2em] text-primary">
              Ölçülen sonuçlar
            </span>
            <h2 className="mt-4 text-4xl font-semibold tracking-tight sm:text-5xl">
              V6 sonuçları, kapsamıyla birlikte.
            </h2>
            <p className="mt-5 text-lg leading-relaxed text-muted">
              V6 görüntü modeli, 29 harfi kapsayan 431 gerçek görüntüde değerlendirildi.
              Bu grup V5&apos;te de kullanılan tarihsel testtir; yeni kişi veya ortamlarla
              yapılmış bağımsız bir test değildir. Canlı kamera ve hareket tanıma
              başarısı ayrıca ölçülmelidir.
            </p>
            <div className="mt-6">
              <Badge variant="demo">V6 model kartı · Statik görüntü testi</Badge>
            </div>
          </Reveal>

          <Reveal delay={0.1}>
            <div className="glass card-hover rounded-[2rem] p-6 shadow-[var(--shadow-soft)] sm:p-8">
              <RevealGroup className="grid grid-cols-2 gap-4">
                {METRICS.map((metric) => (
                  <motion.div
                    key={metric.label}
                    variants={revealItem}
                    className="rounded-2xl border border-border bg-surface p-5"
                  >
                    <metric.icon size={18} className="text-primary" strokeWidth={1.75} />
                    <p className="mt-4 text-3xl font-semibold tracking-tight tabular-nums sm:text-4xl">{metric.value}</p>
                    <p className="mt-1 text-sm font-medium">{metric.label}</p>
                    <p className="mt-1 text-xs leading-snug text-muted">{metric.note}</p>
                  </motion.div>
                ))}
              </RevealGroup>
              <p className="mt-6 text-center text-xs text-muted">
                Gecikme yalnızca model çıkarımını kapsar; kamera ve ağ süresi dahil değildir.
                Canlı oturum değerleri harf tanıma panelinde gösterilir.
              </p>
            </div>
          </Reveal>
        </div>
      </Container>
    </Section>
  );
}
