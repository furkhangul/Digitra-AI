import Link from "next/link";
import { LiveRecognition } from "@/components/home/live-recognition";
import { Egitim } from "@/components/home/egitim";
import { TID_API_BASE, TID_MODEL_NAME } from "@/lib/recognition-config";

export default function V6CameraTest() {
  return (
    <div className="pt-28 pb-16">
      <section className="mx-auto max-w-5xl px-6">
        <Link href="/" className="text-sm text-muted hover:text-primary">
          ← Digitra ana sayfa
        </Link>
        <p className="mt-8 text-sm font-medium text-primary">Canlı kamera testi · V6</p>
        <h1 className="mt-2 text-3xl font-semibold tracking-tight">Birlikte deneyelim.</h1>
        <p className="mt-3 mb-8 max-w-2xl leading-relaxed text-muted">
          V6 otomatik modda sabit harf görüntüsünü ve J, Ğ gibi hareketli
          harflerin sekansını birlikte izler. İki elini de görüntüde tutup
          hareketi tamamla; her harften sonra ellerini kısa süre indir.
          Harflerin 3D gösterimleri aşağıda.
        </p>
        <LiveRecognition apiBase={TID_API_BASE} modelName={TID_MODEL_NAME} />
      </section>
      <Egitim />
    </div>
  );
}
