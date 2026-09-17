"use client";

import { useState } from "react";
import { ArrowRight, Mic, ShieldCheck, Sparkles } from "lucide-react";
import { LiveRecognition } from "@/components/home/live-recognition";
import { Container } from "@/components/ui/container";
import { Reveal } from "@/components/ui/reveal";
import { cn } from "@/lib/utils";
import { TID_API_BASE, TID_MODEL_NAME } from "@/lib/recognition-config";

type Direction = "hand-to-text" | "speech-to-sign";

const LIVE_STEPS = [
  ["Ellerini göster", "İki elli harflerde her iki elini de kadrajda tut."],
  ["Harfi tamamla", "Sabit harfi kısa süre koru; hareketli harfte hareketi tamamla."],
  ["Sonucu takip et", "Tahmin kararlı hale geldiğinde harf metne eklenir."],
  ["Devam et", "Yeni harf için ellerini kısa süre indir. Metnini istersen seslendir."],
];

export function Ceviri() {
  const [direction, setDirection] = useState<Direction>("hand-to-text");

  return (
    <section id="ceviri" className="mesh-bg relative overflow-hidden py-24 sm:py-32">
      <Container>
        <div className="grid items-start gap-12 lg:grid-cols-[0.82fr_1.18fr] lg:gap-16">
          <div className="lg:sticky lg:top-28">
            <Reveal>
              <span className="text-xs font-medium uppercase tracking-[0.2em] text-primary">
                Canlı harf tanıma
              </span>
              <h2 className="mt-4 text-4xl font-semibold tracking-tight sm:text-5xl">
                Harfi göster. Metne dönüşsün.
              </h2>
              <p className="mt-5 text-lg leading-relaxed text-muted">
                TİD parmak alfabesindeki harfleri kamerayla dene ve kendi metnini
                oluştur. Otomatik mod sabit harflerle hareketli harfleri birlikte
                izler; hareket tanıma özelliği deneme aşamasındadır.
              </p>
            </Reveal>

            <Reveal delay={0.06}>
              <div className="mt-8 inline-flex rounded-full border border-border bg-surface-2 p-1">
                <button
                  type="button"
                  onClick={() => setDirection("hand-to-text")}
                  className={cn(
                    "cursor-pointer rounded-full px-4 py-2 text-sm font-medium transition",
                    direction === "hand-to-text" ? "text-white shadow" : "text-muted hover:text-foreground",
                  )}
                  style={direction === "hand-to-text" ? { background: "var(--grad-primary)" } : undefined}
                >
                  Harf → Metin
                </button>
                <button
                  type="button"
                  onClick={() => setDirection("speech-to-sign")}
                  className={cn(
                    "cursor-pointer rounded-full px-4 py-2 text-sm font-medium transition",
                    direction === "speech-to-sign" ? "text-white shadow" : "text-muted hover:text-foreground",
                  )}
                  style={direction === "speech-to-sign" ? { background: "var(--grad-primary)" } : undefined}
                >
                  Ses → İşaret
                </button>
              </div>
            </Reveal>

            {direction === "hand-to-text" ? (
              <ol className="mt-9 space-y-5">
                {LIVE_STEPS.map(([title, description], index) => (
                  <li key={title} className="flex gap-4">
                    <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-primary/10 text-sm font-semibold text-primary">
                      {index + 1}
                    </span>
                    <div>
                      <p className="font-medium">{title}</p>
                      <p className="mt-0.5 text-sm leading-relaxed text-muted">{description}</p>
                    </div>
                  </li>
                ))}
              </ol>
            ) : (
              <div className="mt-9 rounded-2xl border border-border bg-surface/70 p-5">
                <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-primary/10 text-primary">
                  <Mic size={20} />
                </div>
                <h3 className="mt-4 font-semibold">Ses → işaret: geliştirme hedefi</h3>
                <p className="mt-2 text-sm leading-relaxed text-muted">
                  Konuşmayı TİD anlatımına dönüştürmek için dil eşlemesi ve
                  doğrulanmış avatar hareketleri üzerinde çalışılması gerekiyor.
                  Bu özellik henüz kullanıma açık değil.
                </p>
              </div>
            )}

            <Reveal delay={0.1}>
              <div className="mt-8 flex items-start gap-3 border-t border-border pt-6 text-sm text-muted">
                <ShieldCheck size={18} className="mt-0.5 shrink-0 text-emerald-500" />
                <p>
                  El tespiti tarayıcıda yapılır. El bölgesinin görüntüsü ve el noktaları
                  tanıma servisine iletilir; mevcut akışta kaydedilmeden işlenir.
                </p>
              </div>
            </Reveal>
          </div>

          <Reveal delay={0.08}>
            {direction === "hand-to-text" ? (
              <LiveRecognition apiBase={TID_API_BASE} modelName={TID_MODEL_NAME} />
            ) : (
              <div className="flex min-h-[560px] flex-col items-center justify-center rounded-3xl border border-border bg-surface p-8 text-center shadow-[var(--shadow-soft)]">
                <span className="flex h-16 w-16 items-center justify-center rounded-2xl bg-primary/10 text-primary">
                  <Sparkles size={26} />
                </span>
                <p className="mt-6 text-xs font-medium uppercase tracking-[0.18em] text-primary">
                  Araştırma aşamasında
                </p>
                <h3 className="mt-3 max-w-md text-2xl font-semibold">
                  Hedefimiz, konuşmadan işaret diline bir köprü kurmak.
                </h3>
                <p className="mt-4 max-w-lg leading-relaxed text-muted">
                  TİD&apos;in kendine özgü dil yapısı, yüz ifadeleri ve beden hareketleri var.
                  Bu özellik, dil uzmanlarıyla doğrulanmış bir çeviri akışı gerektiriyor.
                  Bugün 3D eğitim bölümünde parmak alfabesini keşfedebilirsin.
                </p>
                <a href="#architecture" className="mt-7 inline-flex items-center gap-2 text-sm font-medium text-primary hover:underline">
                  Tanıma teknolojisini incele <ArrowRight size={15} />
                </a>
              </div>
            )}
          </Reveal>
        </div>
      </Container>
    </section>
  );
}
