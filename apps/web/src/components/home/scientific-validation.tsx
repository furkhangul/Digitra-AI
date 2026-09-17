"use client";

import { motion } from "framer-motion";
import { CheckCircle2, FlaskConical } from "lucide-react";
import { Container, Section } from "@/components/ui/container";
import { Reveal, RevealGroup, revealItem } from "@/components/ui/reveal";

const PRINCIPLES = [
  "Eğitim, doğrulama ve test görüntüleri ayrı tutulur; mevcut ardışık kare grupları korunur.",
  "Eğitimden önce dosya bütünlüğü, aynı görüntüler ve yakın kopyalar denetlenir.",
  "Işık, açı ve görüntü kalitesi çeşitlendirmeleri yalnızca eğitim verisine uygulanır.",
  "3D el görselleri yardımcı eğitim verisidir; test sonuçları gerçek görüntüler üzerinden hesaplanır.",
  "Model seçimi doğrulama verisiyle yapılır; son test için seçilen model sabitlenir.",
  "Doğrulukla birlikte harf bazlı sonuçlar, macro F1 ve farklı görüntü koşulları incelenir.",
  "Kaynak veride kişi kimliği yoktur; mevcut test, yeni kullanıcılardaki başarıyı kanıtlamaz.",
  "Görüntü modelinin test sonucu, deneysel hareket takibinin başarı oranı olarak kullanılmaz.",
];

export function ScientificValidation() {
  return (
    <Section id="science">
      <Container>
        <div className="grid gap-14 lg:grid-cols-[0.9fr_1.1fr] lg:items-start">
          <Reveal className="lg:sticky lg:top-32">
            <span className="text-xs font-medium uppercase tracking-[0.2em] text-primary">
              Nasıl değerlendiriyoruz?
            </span>
            <h2 className="mt-4 text-4xl font-semibold tracking-tight sm:text-5xl">
              Her sonucun bir ölçüm koşulu var.
            </h2>
            <p className="mt-5 text-lg leading-relaxed text-muted">
              Bir testteki başarı, her kamerada aynı sonucu almak anlamına gelmez.
              Digitra&apos;nın modellerini geliştirirken neyi ölçtüğümüzü, hangi veriyi
              kullandığımızı ve hangi alanların ek doğrulama gerektirdiğini açıkça belirtiyoruz.
            </p>
            <div className="mt-6 inline-flex items-center gap-2 rounded-2xl border border-border px-4 py-3 text-sm text-muted">
              <FlaskConical size={16} className="text-primary" />
              V6 sonuçları, model kartı ve deney kayıtlarıyla izlenir.
            </div>
          </Reveal>

          <RevealGroup className="grid grid-cols-1 gap-3 sm:grid-cols-2">
            {PRINCIPLES.map((principle) => (
              <motion.div
                key={principle}
                variants={revealItem}
                className="card-hover flex items-start gap-3 rounded-2xl border border-border bg-surface p-5"
              >
                <CheckCircle2 size={18} className="mt-0.5 shrink-0 text-success" />
                <p className="text-sm leading-relaxed">{principle}</p>
              </motion.div>
            ))}
          </RevealGroup>
        </div>
      </Container>
    </Section>
  );
}
