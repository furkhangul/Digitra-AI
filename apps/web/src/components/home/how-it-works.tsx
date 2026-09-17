"use client";

import dynamic from "next/dynamic";
import Link from "next/link";
import { useRef } from "react";
import { useInView } from "framer-motion";
import { ArrowDownRight, ArrowUpRight, Check, MoveRight } from "lucide-react";
import { Container } from "@/components/ui/container";
import { Reveal } from "@/components/ui/reveal";
import styles from "./digitra-highlights.module.css";

const HandPreview = dynamic(() => import("./digitra-feature-preview"), {
  ssr: false,
  loading: () => <p className={styles.loading}>3D eller hazırlanıyor…</p>,
});

export function HowItWorks() {
  const previewRef = useRef<HTMLDivElement>(null);
  const showPreview = useInView(previewRef, { once: true, margin: "240px" });

  return (
    <section id="how-it-works" className={styles.section} aria-labelledby="digitra-highlights-title">
      <Container>
        <Reveal className={styles.heading}>
          <div>
            <p className={styles.eyebrow}><span /> Neden Digitra?</p>
            <h2 id="digitra-highlights-title">Öğrenmek,<br /><em>elinde.</em></h2>
          </div>
          <div className={styles.headingAside}>
            <p>Bir harfin nasıl yapıldığını gör. Kendi ellerinle dene.
              Öğrendiklerini metne ve sese dönüştür.</p>
            <span className={styles.headingNote}>Türk İşaret Dili parmak alfabesi <ArrowDownRight size={20} aria-hidden /></span>
          </div>
        </Reveal>

        <Reveal className={styles.grid}>
          <article className={styles.learnCard}>
            <div className={styles.cardTop}>
              <span className={styles.kicker}>01 / Görerek öğren</span>
              <span className={styles.lightBadge}>29 harf · 3D rehber</span>
            </div>
            <h3>Her harfi<br />hareketiyle keşfet.</h3>
            <p className={styles.learnDescription}>Parmakların yerleşimini incele.<br />Bir harf seç, kendin dene.</p>
            <div ref={previewRef} className={styles.preview}>
              {showPreview ? <HandPreview /> : <p className={styles.loading}>3D harf önizlemesi</p>}
            </div>
            <a href="#egitim" className={styles.learnLink}>Alfabenin tamamını keşfet <ArrowUpRight size={20} aria-hidden /></a>
          </article>

          <article className={styles.practiceCard}>
            <div className={styles.cardTop}>
              <span className={styles.kicker}>02 / Yaparak pekiştir</span>
              <span className={styles.darkBadge}><span /> Kamerayla pratik</span>
            </div>
            <h3>Sıra senin<br />ellerinde.</h3>
            <p>3D rehberi taklit et, harfi kameraya göster.
              Eşleştiğinde bir sonraki harfe geç ve kelimeyi tamamla.</p>
            <div className={styles.wordExample} aria-label="Örnek çalışma: AYŞE kelimesinde Y harfi">
              <div className={styles.wordTiles} aria-hidden>
                <span className={styles.completedLetter}>A<Check size={10} /></span>
                <span className={styles.currentLetter}>Y</span><span>Ş</span><span>E</span>
              </div>
              <span className={styles.exampleCaption}>Örnek çalışma<br /><strong>Harf harf, kendi hızında.</strong></span>
            </div>
            <Link href="/egitim/canli" className={styles.darkLink}>Canlı pratiğe geç <ArrowUpRight size={19} aria-hidden /></Link>
          </article>

          <article className={styles.dataCard}>
            <div className={styles.cardTop}><span className={styles.kicker}>03 / Digitra’nın yaklaşımı</span><span className={styles.smallTag}>Ortak 3D kaynak</span></div>
            <h3>Aynı eller.<br />İki farklı katkı.</h3>
            <p>Sana harfleri gösteren 3D modeller, tanıma modelinin eğitimine yardımcı görseller de üretir.</p>
            <div className={styles.dataFlow}>
              <div><span>Senin için</span><strong>3D harf rehberi</strong></div>
              <span className={styles.flowDivider} aria-hidden>+</span>
              <div><span>Model için</span><strong>Eğitim görselleri</strong></div>
            </div>
            <a href="#datasets" className={styles.textLink}>Nasıl geliştiriyoruz? <MoveRight size={17} aria-hidden /></a>
          </article>

          <article className={styles.voiceCard}>
            <div className={styles.cardTop}><span className={styles.kicker}>04 / Kendini ifade et</span><span className={styles.smallTag}>Metin → Ses</span></div>
            <h3>Harfleri birleştir.<br />Sesini duyur.</h3>
            <p>Tanınan harflerle metnini oluştur. Kontrol et, düzelt ve tarayıcının Türkçe sesiyle dinle.</p>
            <div className={styles.voiceExample} aria-hidden>
              <span>Merhaba<span className={styles.textCursor} /></span>
              <div className={styles.waveform}>{[12, 24, 17, 36, 25, 44, 30, 18, 32, 14].map((height, i) => <i key={i} style={{ height }} />)}</div>
            </div>
            <a href="#ceviri" className={styles.textLink}>Harf tanımayı dene <MoveRight size={17} aria-hidden /></a>
          </article>

          <article className={styles.futureCard}>
            <div className={styles.cardTop}><span className={styles.kicker}>05 / Sırada ne var?</span><span className={styles.futureBadge}>Geliştirme hedefi</span></div>
            <h3>Sana özel<br />bir yol gösterici.</h3>
            <p>Bir sonraki hedefimiz, el konumuna ve hareketine göre kişisel düzeltmeler sunan bir 3D eğitmen.</p>
            <div className={styles.futureNote}><span className={styles.futureMark} aria-hidden>↗</span><span>Daha kişisel bir öğrenme deneyimi.<small>Bu özellik henüz kullanıma açık değil.</small></span></div>
          </article>
        </Reveal>
      </Container>
    </section>
  );
}
