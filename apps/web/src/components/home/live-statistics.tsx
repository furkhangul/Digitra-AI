"use client";

import { motion } from "framer-motion";
import { BarChart3, Layers3, Timer, Zap } from "lucide-react";
import { Container, Section } from "@/components/ui/container";
import { Reveal, RevealGroup, revealItem } from "@/components/ui/reveal";
import { Badge } from "@/components/ui/badge";

const METRICS = [
  { icon: BarChart3, label: "%90,26 · Test doğruluğu" },
  { icon: Layers3, label: "%90,23 · Test macro F1" },
  { icon: Timer, label: "20,4 ms · Model gecikmesi (medyan)" },
  { icon: Zap, label: "431 · Gerçek test görüntüsü" },
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
                    <div className="mt-4 h-8 w-20 overflow-hidden rounded-md bg-surface-2">
                      <motion.div
                        className="h-full w-1/3"
                        style={{ background: "var(--grad-primary)", opacity: 0.5 }}
                        animate={{ x: ["-100%", "260%"] }}
                        transition={{ duration: 2.2, repeat: Infinity, ease: "easeInOut" }}
                      />
                    </div>
                    <p className="mt-3 text-xs leading-snug text-muted">{metric.label}</p>
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
