"use client";

import { useState } from "react";
import { Pause, Play } from "lucide-react";
import HandScene from "@/components/tid/hand-scene";
import { TID_ALPHABET } from "@/components/tid/poses";
import styles from "./digitra-highlights.module.css";

const LETTERS = TID_ALPHABET.filter(({ ch }) => ["A", "B", "C", "Ç", "Y"].includes(ch));

export default function DigitraFeaturePreview() {
  const [selected, setSelected] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [replay, setReplay] = useState(0);
  const letter = LETTERS[selected];

  return (
    <>
      <div className={styles.handStage}>
        <span className={styles.backdropLetter} aria-hidden>{letter.ch}</span>
        <HandScene letter={letter} playing={playing} speed={0.8} replay={replay} />
        <span className={styles.stageCaption}>TİD · {letter.ch} harfi</span>
      </div>
      <div className={styles.previewControls}>
        <div className={styles.letterPicker} role="group" aria-label="3D önizlemede harf seç">
          {LETTERS.map((item, i) => <button type="button" key={item.ch} aria-label={`Önizlemede ${item.ch} harfini göster`} aria-pressed={i === selected} onClick={() => { setSelected(i); setPlaying(true); setReplay(value => value + 1); }}>{item.ch}</button>)}
        </div>
        <button type="button" className={styles.pauseButton} aria-label={playing ? "Önizlemeyi duraklat" : "Önizlemeyi oynat"} onClick={() => setPlaying(value => !value)}>
          {playing ? <Pause size={16} aria-hidden /> : <Play size={16} aria-hidden />}
        </button>
      </div>
    </>
  );
}
