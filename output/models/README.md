# Digitra TİD el modeli

Eğitim bölümündeki el yeniden yapıldı. Bu belge ne yapıldığını, neyin
doğrulandığını ve nelerin hâlâ açık olduğunu ayrı ayrı anlatır.

## Elin nasıl tanımlandığı

El artık **tek bir analitik yüzey** olarak tanımlıdır: altı avuç kütlesi
(avuç çekirdeği, boğum yastığı, avuç tabanı, tenar, başparmak perdesi,
hipotenar) yumuşak birleşimle kaynaşır; her parmak kendi kapsül zinciriyle
**yalnızca avuca** yumuşak bağlanır, parmaklar birbirine **sert birleşimle**
(hard min) eklenir.

Bu ayrım önemli: parmaklar birbirine de yumuşak bağlanırsa yan yana gelen
parmaklar oyun hamuru gibi tek kütleye erir. Sert birleşimle her parmak kendi
siluetini ve aradaki çizgiyi korur. Parmak yarıçapı boy boyunca bir profille
değişir; iki eklemde küçük şişkinlikler vardır, böylece parmak tek boru değil
üç boğumlu bir parmak gibi okunur.

Aynı tanım tek kaynaktan üç yere gider:

- **Web**: `apps/web/src/components/tid/rounded-hand.ts` içindeki fragment
  shader bu alanı doğrudan ışın yürüterek çizer. Eğitim bölümünde ekranda
  görünen budur.
- **Blender/GLB**: `apps/web/scripts/tid_sdf/make_mesh.py` aynı alanı marching
  cubes ile ağa çevirir; `apps/web/scripts/build-tid-sdf-hand.py` bunu
  kemiklendirir, ağırlıklandırır ve dışa aktarır.
- **Doğrulama**: `apps/web/scripts/tid_sdf/` altındaki Python modülleri aynı
  alanı kullanarak yüzey temaslarını ölçer.

> **Açık olalım:** eğitim ekranı GLB'yi yüklemez. GLB ve `.blend`, aynı biçimin
> düzenlenebilir/aktarılabilir kopyalarıdır. Bu, `tid-hand.glb`'nin sitede
> kullanıldığı izlenimi verilmesin diye yazılmıştır.

## Anatomi düzeltmeleri

Ölçüler avuç boyu (bilek → orta parmak boğumu) = 1.0 birimiyle verilir.

| Ne | Önce | Sonra | Neden |
|---|---|---|---|
| Başparmak toplam boyu | 0.87 | 1.05 | Eski başparmak işaret parmağına **hiç ulaşamıyordu**; O/B/Ö/İ gibi halka işaretleri geometrik olarak imkânsızdı (en yakın mesafe 0.476, temas için gereken 0.245). |
| Başparmak kökü | y 0.10 | y 0.24, avuca doğru z 0.13 | Kök bilekten değil avucun içinden çıkıyor; tenar kütlesi artık doğal devam ediyor. |
| Parmak yarıçapları | 0.087–0.106 | 0.104–0.140 | Daha dolgun, referanstaki tokluğa yakın. |
| Serçe boyu | 0.82 | 0.70 | Diğer parmaklara göre orantısız uzundu. |
| İşaret/orta/yüzük | 0.93/1.02/0.95 | 0.86/0.94/0.88 | Avuca göre dengelendi. |

Kapalı elde başparmak artık boşluğa değil, **kapanmış işaret parmağının
yanına** oturtularak çözülüyor; eklem sınırları başparmağın kendi altına
katlanmasını engelliyor (`closedThumb`, `pinch` içindeki limitler).

## Harf düzeltmeleri

- **G, Ğ**: Kaynak çizimde (G2/Ğ2) **iki kapalı yumruk üst üste** durur ve
  **üstteki elin başparmağı dik yukarı** bakar. Önceki uygulama bunu "kıvrık
  işaret parmağı" olarak modellemişti — bu bir okuma hatasıydı, düzeltildi.
  Ğ'nin farkı yalnızca hareket: kaynakta başparmağın yanındaki çift yönlü ok.
  Bu yüzden `thumb-tap` hareketi eklendi; eski `double-tap` gizli kalan işaret
  parmağını oynattığı için **ekranda hiçbir hareket görünmüyordu**.
- **B**: İki el kıskaç yapıp uçları birleştirir. Eller artık her iki halka da
  kameradan görünecek biçimde yerleştirildi; önceki düzende bir el diğerini
  tamamen kapatıyordu.
- **Ö**: Aynı kıskaç, ama kaynağa uygun olarak eller **ayrık** durur ve üstteki
  el yaklaşma hareketi yapar. B (temas) ile Ö (ayrık + hareket) artık ayırt
  edilebilir.
- **O**: Halka açık ve okunur; üç parmak yukarıda.
- **A**: Kaynak çizim aslında **harfin kendi biçimini** gösteriyor — aşağı
  bakan iki parmak A'nın bacakları, yatay işaret parmağı da orta çubuğu.
  Önceki düzende yatay parmak bacakların **altında** kalıyordu (otomatik
  çarpışma araması eli 0.74 birim aşağı iterek çakışmayı "çözmüş", ama harfin
  yapısını bozmuştu). Yeniden kuruldu: bacaklar tam aşağı bakıyor, aralarındaki
  açıklık ±18°'den ±6°'ye indirildi (kaynakta neredeyse paralel), yatay parmak
  bacakların önünden %45 yüksekliğinde geçiyor, **her ikisine de değiyor**
  (temas payı +0.005) ve ucu uzak bacağı geçerek görünüyor. Çakışma +0.006,
  yani kesişme yok.
- **D, E, H, M, N, R, Y**: İstenen temas kurulduğunda ellerin başka
  yerlerinin birbirine girdiği tespit edildi. Her biri için el şekli ve temas
  noktaları sabit tutularak yalnızca bilek yönü ve yaklaşma yönü aranıp
  çarpışmasız bir düzen bulundu (`apps/web/scripts/tid_sdf/fixpose.py`).

## Temas nasıl çözülüyor

`surface-offsets.json` artık **gerçek yüzey** üzerinden üretiliyor, kemik
merkezlerinden değil. Her işaret için:

1. İşaretin **hangi iki noktayı** birleştirdiği `poses.ts` içinde
   `pose.contact` olarak kayıtlı (ör. B için `index_tip`↔`index_tip`).
2. Yüzey noktaları örnekleniyor, o iki bölgenin arası tam temasa getiriliyor.
3. **Ayrı olarak** ellerin başka her yerindeki en derin çakışma ölçülüyor.

İkisi ayrı raporlanır. Bir temasın çözülmüş olması geri kalan yerlerin temiz
olduğunu göstermez; bu yüzden aynı sayı iki kez sunulmaz.

## Doğrulanan sonuçlar

`output/models/surface-contact-report.json` (üretim: `python
apps/web/scripts/tid_sdf/verify.py`)

**Geometri** — `output/models/sdf-build-report.json`
- 11.262 köşe, 22.520 üçgen
- **0 açık kenar, 0 manifold olmayan kenar** → kapalı yüzey; bilek ucu
  yuvarlatılmış ve kapalı, kesik boru ağzı yok
- 21 kemik (bilek + 5×3 eklem + 5 uç işaretçisi), ağırlık toplamları 1.0
- Köşelerin %55,9'u baskın olarak bir parmak kemiğine bağlı

**İki el teması** — 22 çift elli harf
- İstenen temas noktası çözülmüş
- **Derin çakışma (< −0.03) kalan harf yok**; en kötü değer −0.016
- Yalnızca **Ö** kasıtlı olarak ayrık (+0.11)

**Geçiş sırasında çakışma** — 29 geçişin ara kareleri, tam yüzey örneklemesi
- Her harfe giren geçişin her 4. karesi ölçüldü
- **Derin çakışma yok**; en kötü değer −0.016
- Bu ölçüm ilk turda B, D, S, Ş harflerinde 0.667 s civarında gerçek çakışma
  buldu (en kötüsü −0.249); ayrılma yönü ölçülerek ve geçişin ortasına dışa
  doğru ek bir itme eklenerek giderildi

**Hareket motoru** — `output/models/motion-test-report.json`
- 29×29 = 841 geçiş, 151.380 kare: değerler sonlu, dönüşler normalize, konum
  adımı sınırların içinde; kesme, duraklatma, hız değişimi ve arka plan
  senaryoları dâhil

**Yaklaşık çarpışma vekili** — `test-tid-contact.cjs`, 3.000 kare, 0 hata.
Bu **yaklaşık** bir vekildir (621 köşeli düşük çözünürlüklü kopya) ve yukarıdaki
gerçek yüzey ölçümleriyle karıştırılmamalıdır; ayrı raporlanmasının nedeni budur.
Vekil, yeni ağdan yeniden üretildi (eskisi eski geometriye aitti, 176 KB → 62 KB).

**Tarayıcı**
- `/debug-tid`: yeni yüzey ekranda; çevrimdışı hesaplanan aynı poz ile birebir
  aynı görüntü
- `/debug-tid-model`: `tid-hand.glb` yükleniyor — 1 deri ağı, 11.262 köşe,
  21 kemik, 51 animasyon; iskelet deformasyonu çalışıyor

## Geçişler

Her geçiş şu sırayı izler: teması bırak → ayrıl → yeni şekli havada kur →
yaklaş → temas → okunabilir duruş. Ayrılma yönü artık **her harf için ayrı
ölçülüyor** (`approach-offsets.json`, üretim `tid_sdf/approach.py`): iki elin
gerçekten ayrıldığı doğrultuda, yüzeyler arasında 0.16 birim boşluk kalana
kadar. Önceki sabit çapraz kaçış, elleri üst üste duran harflerde (G, Ö gibi)
hiç ayırmıyordu; geri dönüşte eller birbirinin içinden geçiyordu.

Kesilen geçiş ekranda görünen kareden devam eder; duraklatma, 0.5×/1×/1.5× hız
ve arka plan sekmesi durumları testlerde ayrıca kontrol edilir.

## Kadraj

Görüntü artık **kendini ölçekliyor**: bir işaret kadraja sığmayacaksa görüş
hızla açılır, sonra yavaşça toparlar. Önceki sabit ölçekte üstteki el
kırpılıyordu.

## Kalan sorunlar ve doğrulama sınırları

Bunlar giderilmedi; "sıfır hata" değildir:

1. **Başparmağın arkadan görünümü.** Yumruk ve kanca pozlarında, elin
   arkasından ~250° açıyla bakıldığında başparmak ucunun kapanmış parmaklara
   girdiği yerde bir oyuk okunuyor. Eğitim görünümü önden olduğu için ekranda
   görünmez, ama GLB'yi serbest döndürürseniz görürsünüz.
2. **Tek elin kendi içindeki çakışma sayısal olarak ölçülmedi.** Tek el tek bir
   yumuşak yüzey olduğu için "kesişme" yerine "kaynaşma" olarak görünür; bunu
   görsel kontrolle değerlendirdim, otomatik ölçmedim.
3. **29×29 geçişin tamamı yüzey düzeyinde taranmadı.** Hareket motorunun
   841 geçişinin tamamı sayısal testlerden geçti, ancak yüzey çakışması
   yalnızca 29 geçişte (her harfe bir giriş, her 4. kare) ölçüldü. Kalan
   812 geçiş yüzey düzeyinde denetlenmedi.
4. **Dilsel doğruluk uzman onaylı değildir.** Pozlar Dikyuva, Makaroğlu ve
   Arık'ın (2015) kitabındaki s.91 çizimlerinden yeniden kuruldu. Statik
   çizimler hareketin tamamını vermez. Özellikle şunlar TİD uzmanı veya
   güvenilir video kaynağı ile doğrulanmalıdır:
   - **O**: kaynak çizimde halkanın içinde yukarı bakan bir ok var. Bunun
     küçük bir yukarı hareket mi yoksa yalnızca el yönü işareti mi olduğunu
     çizimden çıkaramadım; **uydurmadım**, poz şu an durağan.
   - **Ö**: gölgeli ikinci el hareketi gösteriyor; hareketin yönü ve tekrar
     sayısı çizimden kesin okunmuyor.
   - **Ç**: şıklatma hareketinin zamanlaması anlatımdan uyarlandı.
   - Sağ/sol el dağılımı ve kameraya göre ayna yönü bir tercihtir; kitaptaki
     çizimler okuyucu bakışıyla verilmiştir.
5. **Kameradaki sınıflandırıcı bu değişiklikle eğitilmedi**; mevcut ASL
   prototipi olarak kalır ve buradaki TİD gösteriminden bağımsızdır.
6. Yaklaşık (kapsül kabuğu) örnekleme yalnızca arama sırasında kullanıldı;
   raporlanan bütün sayılar gerçek yüzeye Newton adımlarıyla oturtulmuş
   noktalarla ölçüldü. İkisi karıştırılmadı.

## Dosyalar

- `digitra-tid-hands.blend` — düzenlenebilir model: ağ, iskelet, elle
  hesaplanmış ağırlıklar ve 51 harf eylemi
- `tid-hand.glb` — dışa aktarılmış deri + iskelet + eylemler (≈0,95 MB)
- `tid-pose-review.png` — 29 harfin görsel kontrol tablosu
- `sdf-build-report.json`, `surface-contact-report.json`,
  `motion-test-report.json` — sayısal raporlar
- `../hand-check/` — yakın plan deformasyon kontrolleri (açık el, yayvan,
  yarım kıvrım, yumruk, halka, kanca × önden/60°/arkadan/başparmak arkası) ve
  `motion-B/G/Ğ/O/Ö.png` yakın plan hareket önizlemeleri

## Yeniden üretme

Proje kökünden:

```
cd apps/web/scripts/tid_sdf && python make_mesh.py 0.0075 hand_rest.obj
blender -b --factory-startup --python apps/web/scripts/build-tid-sdf-hand.py -- <yol>/hand_rest.obj
blender -b --factory-startup --python apps/web/scripts/render-tid-check.py
cd apps/web && node scripts/dump-tid-poses.cjs && node scripts/test-tid.cjs
cd apps/web/scripts/tid_sdf && python approach.py
cd apps/web && node scripts/dump-tid-motion.cjs A B C Ç D E F G Ğ H I İ J K L M N O Ö P R S Ş T U Ü V Y Z
cd apps/web/scripts/tid_sdf && python verify.py && python mksheet.py final && python motion_sheet.py
```

Python tarafı numpy, scipy, scikit-image ve Pillow ister (Blender'ın kendi
Python'unda değil, sistem Python'unda çalışır).

## Kaynak ve lisans

- El şekli tamamen bu proje için üretildi; dış varlık kullanılmadı.
  Önceki sürümdeki WebXR Input Profiles `generic-hand` (MIT, Amazon) geometrisi
  **artık kullanılmıyor**; lisans metni geçmiş sürümler için
  `THIRD_PARTY_HAND_LICENSE.md` içinde bırakıldı.
- fingerspelling.xyz yalnızca görsel ve hareket kalitesi için **referans olarak
  incelendi**; hiçbir varlığı bu projeye kopyalanmadı. Referansın el modeli
  ASL içindir ve ASL harfleri kullanılmamıştır.
- TİD kaynağı: Dikyuva, H., Makaroğlu, B. & Arık, E. (2015), *Türk İşaret Dili
  Dilbilgisi Kitabı*, basılı s. 91 —
  https://linguistics.ankara.edu.tr/wp-content/uploads/sites/1078/2021/05/Dikyuva_H._Makaroglu_B._and_Arik_E._2015_compressed.pdf#page=92
