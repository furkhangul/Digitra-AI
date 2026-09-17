# Eğitim el gösterimi

Eğitim ekranı sabit önden, ortografik bir görünüm kullanır. El hacmi ve ışıklandırma korunur; kullanıcı kamerayı döndürmez. Bu nedenle uygulama 2.5D bir sunumdur, düz bir SVG veya video değildir. WebGL gerekir.

`src/components/tid/rounded-hand.ts` avuç ve başparmak kökünü kapalı oval yüzeylerle, parmakları eğri boyunca yuvarlak kesitlerle oluşturur. Elin altında açık bilek veya kesilmiş bir düzlem bulunmaz. Parmakların kendi segmentleri minimum uzaklıkla birleştirilir; segment birleşimlerinde gereksiz yumuşak birleştirme uygulamak düz parmaklarda halka izleri oluşturduğu için kullanılmaz. Parmak/avuç bağlantıları yumuşatılır. Parmak sınır küreleri ve el derinlik aralıkları boş bölgelerdeki hesaplamayı azaltır.

`hand-scene.tsx` mevcut zaman tabanlı oynatıcıya bağlanır. Duraklatma, tekrar, hız, görünmeyen sekmede durma ve azaltılmış hareket tercihi desteklenir. Durağan pozlarda sürekli çizim yapılmaz. Blender/GLB dosyaları korunmuştur; eğitim ekranı bu dosyaları yüklemez.

Ç için üstte C sabit tutulur; alt elin baş ve orta parmakları buluşur, orta parmak hızla avuca iner, ardından daha yavaş geri döner. Başlangıç ve bitiş pozları aynıdır. Temas pozunda orta parmak yana aşırı açılmaz. Mevcut TİD kaynak notu ve uzman doğrulaması bekleniyor açıklaması korunur. Görsel/kinematik testler dilbilimsel doğrulama yerine geçmez.

Doğrulama:

- `node scripts/test-rounded-hand.cjs`: Ç döngüsü, parmak ucu teması, sabit üst el, hız, azaltılmış hareket ve 29 harfin yüzey verileri.
- `node scripts/test-tid.cjs`: 841 harf geçişi, 151.380 kare, duraklatma ve yarıda kesilen geçişler.
- `npm run build` ve değişen bileşenler üzerinde ESLint.
- `/debug-tid`: yerel görsel inceleme; canvas üzerinde kare süresi ve poz verileri bulunur. Kare süresi ölçümü cihaz ve tarayıcıya bağlıdır.

Mevcut kaynak pozları ve pozlara ait derinlik/yerleşim düzeltmeleri kullanılır. Görüntüde örtüşen yüzeyleri derinlik belirler; bu gösterim anatomik bir çarpışma simülasyonu değildir.
