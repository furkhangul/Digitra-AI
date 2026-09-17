"use client";

import { motion } from "framer-motion";
import { ScanEye, Layers, GitMerge, Gauge, History, ShieldCheck } from "lucide-react";
import { Container, Section } from "@/components/ui/container";
import { Reveal, RevealGroup, revealItem } from "@/components/ui/reveal";
import { Badge } from "@/components/ui/badge";
import { TiltCard } from "@/components/ui/tilt-card";

const STAGES = [
  {
    icon: ScanEye,
    title: "MediaPipe ile el tespiti",
    desc: "Tarayıcı, bir veya iki eli bulur ve her el için 21 noktanın konumunu çıkarır. Böylece el bölgesi ve parmak hareketleri takip edilir.",
  },
  {
    icon: Layers,
    title: "El bölgesine odaklanma",
    desc: "Tam kamera karesi yerine elleri çevreleyen görüntü kırpılır. Tanıma servisi, bu görüntüyü ve el noktalarını bellekte işler.",
  },
  {
    icon: GitMerge,
    title: "EfficientNet-B0 · V6",
    desc: "Güncel görüntü modeli, TİD alfabesindeki 29 harf için tahmin üretir. Gerçek el görüntüleriyle eğitilir; 3D ellerden üretilen görseller eğitimi destekler.",
  },
  {
    icon: Gauge,
    title: "Hareket takibi",
    desc: "Ç, Ğ, İ, J, Ö, Ş ve Ü için el noktalarının zaman içindeki değişimi de izlenir. Bu deneysel katmanın canlı kamera başarısı henüz ölçülmedi.",
  },
  {
    icon: History,
    title: "Kararlı harf seçimi",
    desc: "Ardışık tahminler, güven düzeyi ve adaylar arasındaki fark birlikte değerlendirilir. Harf, kararlılık koşulları sağlandığında metne eklenir.",
  },
  {
    icon: ShieldCheck,
    title: "Kontrol sende",
    desc: "El görünmediğinde veya sonuç yeterince kararlı olmadığında sistem bekler. Tanınan metni kontrol edebilir, düzeltebilir ve seslendirebilirsin.",
  },
];

export function Architecture() {
  return (
    <Section id="architecture">
      <Container>
        <Reveal className="mx-auto max-w-2xl text-center">
          <span className="text-xs font-medium uppercase tracking-[0.2em] text-primary">
            Tanıma teknolojisi
          </span>
          <h2 className="mt-4 text-4xl font-semibold tracking-tight sm:text-5xl">
            Elin şeklini ve hareketini birlikte izler.
          </h2>
          <p className="mt-4 text-lg text-muted">
            ASL ile başlayan araştırma, TİD görüntü modelleriyle ilerledi.
            MobileNetV3 ve EfficientNet-B0 birleşimini kullanan V5&apos;in ardından,
            güncel V6 akışı EfficientNet-B0 görüntü modelini el ve hareket takibiyle
            birlikte kullanıyor.
          </p>
          <div className="mt-6 flex justify-center">
            <Badge variant="demo">TİD V6 · Görüntü modeli + deneysel hareket takibi</Badge>
          </div>
        </Reveal>

        <RevealGroup className="mt-16 grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {STAGES.map((stage, i) => (
            <motion.div key={stage.title} variants={revealItem}>
              <TiltCard className="card-hover group relative overflow-hidden rounded-3xl border border-border bg-surface p-6">
                <div
                  className="pointer-events-none absolute -top-16 -right-16 h-40 w-40 rounded-full opacity-0 blur-3xl transition-opacity duration-500 group-hover:opacity-20"
                  style={{ background: "var(--grad-primary)" }}
                />
                <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-primary/10 text-primary">
                  <stage.icon size={20} strokeWidth={1.75} />
                </div>
                <span className="mt-4 block text-xs font-medium text-muted">Aşama {i + 1}</span>
                <h3 className="mt-1 text-lg font-semibold">{stage.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-muted">{stage.desc}</p>
              </TiltCard>
            </motion.div>
          ))}
        </RevealGroup>
      </Container>
    </Section>
  );
}
