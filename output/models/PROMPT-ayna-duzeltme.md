# Görev: TİD parmak alfabesinde ayna görünümü tutarsızlığını gider

## 1. Problem

Bir Türk İşaret Dili (TİD) parmak alfabesi 3B gösterimi var. 29 harf, iki el,
sabit önden kamera. Harflerin bir kısmı **ayna görünümüyle**, bir kısmı
**birinci-şahıs görünümüyle** yazılmış. Hepsi ayna görünümü olmalı.

### Kabul ölçütü (ayna kuralı)

Kullanıcı işareti kendisi yapıp aynaya baktığında, ekrandaki görüntü aynadaki
görüntüyle **birebir aynı** olmalı.

Aynanın iki sonucu vardır, ikisi de gereklidir:

1. **Taraf korunur.** Kişinin sağ eli, ayna görüntüsünde de görüntünün sağ
   tarafındadır. (Ekranda: sağ el sağda, sol el solda.)
2. **El yönü (chirality) tersine döner.** Aynadaki "sağ el", geometrik olarak
   bir *sol el* gibi görünür. Ayna ön-arkayı çevirir, sağ-solu değil.

### Pratik test (her harf için tek tek uygula)

> Ekranda "sağ el" olarak gösterilen pozu, **kendi sağ elinle**, aynaya bakıyormuş
> gibi alabiliyor musun? Bilek ve parmak eklemleri insan sınırları içinde kalarak.
> Alamıyorsan o harf yanlıştır.

Kullanıcının tespiti tam olarak buydu: A harfinde ekrandaki sağ el, **sadece bir
sol elin alabileceği** bir açıda duruyor. "Sağ elimi ekrana doğru bu şekilde
çevirmem imkânsız; sol elim döner ama sağ dönmez."

## 2. Harf harf mevcut durum (kullanıcı gözlemi)

| Durum | Harfler |
|---|---|
| **Doğru, ayna** — dokunma | B, C, Ç, D, F |
| **Yanlış** | A, E |
| **Ayna doğru ama el şekli yanlış** (ayrı sorun, bkz. §5) | G, Ğ |
| **Denetlenmedi** | H'den Z'ye kalanlar |

Ayrıntılar:

- **A**: Sağ el imkânsız açıda. Ayna uygulanmamış.
- **E**: El −Z ekseninden geliyor, aynayla ilgisi yok. Olması gereken: sağ el
  sola bakacak şekilde üç parmak uzanır.
- **G, Ğ**: Ayna tamam ama TİD'e göre el şekli bu değil.

H'den Z'ye kadar olan harfleri de aynı testle tek tek denetle ve raporla.

## 3. Kod yapısı

Next.js + TypeScript. İlgili dosyalar:

```
apps/web/src/components/tid/
  poses.ts               29 harfin tanımı (asıl düzenlenecek dosya)
  rig-spec.json          parmak uzunlukları, yarıçapları, kök noktaları
  rounded-hand.ts        ekrana çizen analitik SDF ray-marcher
  surface-offsets.json   harf başına iki eli ayıran düzeltme vektörü
  approach-offsets.json   geçişte ellerin ayrılma yönü
  motion.ts              geçiş/animasyon motoru
```

### Poz veri modeli

Her harf: `pose: { right: HandPose, left: HandPose, contact?: {...} }`

```ts
HandPose = {
  shape:    Record<Finger, { bend: [number,number,number]; spread: number; oppose: number }>, // derece
  rotation: [number, number, number],   // Euler XYZ, DERECE
  position: [number, number, number],
  visible:  boolean,
}
```

- Parmaklar: `thumb, index, middle, ring, pinky`.
- `bend[j]`: j. eklemin bükülmesi. `spread`: yana açılma. `oppose`: başparmak karşılaması.
- El yerel uzayı: **+Y** parmak yönü, **+Z** avuç normali (avucun baktığı yön),
  **+X** serçe tarafı. Avuç boyu (bilek → orta parmak boğumu) = **1.0 birim**;
  bütün mesafeler bu birimdedir.

### Pozların kurulması

```ts
solo(shape, rotation)                                   // tek el
joined(rightShape, leftShape, rightRot, leftRot,
       rightAnchor, leftAnchor, gap)                    // iki el
```

`joined`, iki elin belirtilen çapa noktalarını çakıştırır, `gap` kadar kaydırır,
sonra `centre()` ile sahneyi ortalar. Çapa adları: `wrist`, `palm`,
`{finger}_0|1|2|tip`. `pose.contact` işaretin **hangi iki noktayı** buluşturduğunu
kaydeder — çakışma değil, kastedilen temas budur.

### Chirality nereden geliyor

`rounded-hand.ts` içinde sol el `scale(-1, 1, 1)` ile aynalanır, sağ el
aynalanmaz. Yani ekrandaki iki el aynı geometrinin ayna çiftidir.

**Önemli:** `Euler XYZ` sırası nedeniyle `rotation[1]`'e (ry) açı eklemek elin
kendi ekseni etrafında değil, **dünya dikey ekseni etrafında** döndürür. Elin
kendi uzun ekseni etrafında döndürmek istiyorsan bunu hesaba kat.

## 4. Kısıtlar

1. **Parmak eklem açılarını (`shape`) değiştirme.** Sorun yönelim/ayna, el şekli
   değil. Tek istisna: G ve Ğ (§5).
2. **Kastedilen temas korunmalı.** `pose.contact`'taki iki çapa noktası yüzeyde
   birbirine değmeli (mesafe ≈ 0). Ç ve Ö'de eller kasten **ayrık** durur
   (sırasıyla ~0.09 ve ~0.11 birim) — onları birbirine yapıştırma.
3. **Çakışma olmasın.** İki elin yüzeyleri arasındaki en küçük mesafe
   **> −0.02 birim**. Kemik merkezlerinin uzaklığına bakma, gerçek yüzeye bak.
4. **Sağ el ekranın sağında, sol el solunda.** (Şu an M, N, Z'de ters — onlar
   da düzeltilmeli.)
5. **Harfin dilsel yapısını bozma.** Çakışmayı gidermek için elleri rastgele
   uzaklaştırma; el şekli, avuç yönü, parmak yönü ve iki elin ilişkisi
   kaynaktaki işaretle uyumlu kalmalı.
6. Yönelim değiştiğinde `surface-offsets.json` ve `approach-offsets.json`
   yeniden hesaplanmalı.

## 5. Ayrı sorun: G ve Ğ'nin el şekli

Şu anki uygulama: iki kapalı yumruk üst üste, üstteki elin başparmağı dik
yukarı; Ğ'de başparmak yukarı-aşağı tekrarlı hareket eder. Kaynak olarak
Dikyuva, Makaroğlu & Arık (2015), *Türk İşaret Dili Dilbilgisi Kitabı*,
basılı s. 91, G2/Ğ2 çizimleri kullanılmış.

Kullanıcı bunun TİD'e göre **yanlış** olduğunu söylüyor. Yapılması gereken:
güvenilir bir TİD kaynağından (uzman, video, sözlük) G ve Ğ'nin doğru el
şeklini, avuç yönünü ve hareketini doğrula; sonra düzelt. Statik çizimden
çıkaramadığın hareket ayrıntılarını **uydurma**, açıkça belirt.

## 6. Doğrulama

Projede hazır araçlar var, bunları kullan:

```
node apps/web/scripts/dump-tid-poses.cjs     # 29 harfin son pozunu JSON'a döker
node apps/web/scripts/test-tid.cjs           # 841 geçiş × 151.380 kare testi
python apps/web/scripts/tid_sdf/verify.py    # yüzey teması + çakışma raporu
```

`apps/web/scripts/tid_sdf/letters.py` aynı SDF'i Python'da kurar; iki elin
yüzeyleri arasındaki gerçek mesafeyi ölçmek için kullanılabilir
(numpy/scipy/scikit-image gerekir).

Her harf için şu üç sayıyı ayrı ayrı raporla:
- kastedilen temas mesafesi
- başka her yerdeki en derin çakışma
- sağ/sol el taraf sırası doğru mu

Birini diğerinin yerine geçirme: temasın çözülmüş olması başka bir yerde
çakışma olmadığını göstermez.

## 7. Çalışma yöntemi — bu önemli

**Harf harf ilerle.** Her harfte: değişikliği yap → yukarıdaki üç sayıyı ölç →
ekran görüntüsü al → ayna testini uygula → sonra diğerine geç.

Daha önce 23 harfe aynı anda toplu bir döndürme uygulandı; 11 harf bozuldu
(eller 0.3–0.4 birim iç içe geçti, bazı eller yukarıdan görünür hale geldi).
**Toplu dönüşüm uygulama.**

## 8. Teslim

1. Hangi harflerin değiştiği, her biri için eski → yeni `rotation` / `position` /
   `gap` değerleri ve **neden** değiştiği.
2. Değişmeyen harflerin listesi ve ayna testinden geçtiklerinin doğrulaması.
3. Her harf için §6'daki üç sayı.
4. G ve Ğ için kullanılan TİD kaynağı ve doğrulanamayan noktalar.
5. Düzeltilemeyen veya belirsiz kalan her şey açıkça listelensin. "Sıfır hata"
   veya "kusursuz" deme; ölçtüğünü söyle, ölçmediğini ölçmedim de.
