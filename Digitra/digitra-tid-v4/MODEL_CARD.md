# Digitra TİD Robust V5 — model kartı

## Önerilen model

Doğrulama macro F1 sonucuna göre seçilen ensemble:

- %25 MobileNetV3 Large + %75 EfficientNet-B0
- Çok ölçekli ve aynalanmış test-time augmentation
- Tavsiye edilen güven eşiği: 0,28
- Eşiğin doğrulama kapsaması: %97,91
- Kabul edilen doğrulama tahminlerinde doğruluk: %90,04

## Veri ve değerlendirme protokolü

- Veri: 2.974 TİD parmak-yazımı görüntüsü, 29 sınıf.
- Split: ardışık 10 kare aynı pseudo-oturumda tutuldu.
- Model ve ensemble ağırlıkları yalnızca validation sonucuyla seçildi.
- Aşağıdaki sayılar daha önce model seçimi için kullanılmayan kilitli test
  bölümü üzerindedir.

## Kilitli test sonuçları

| Koşul | Doğruluk | Macro F1 |
|---|---:|---:|
| Temiz | %89,10 | %88,98 |
| Yatay ayna (diğer el benzetimi) | %89,33 | %89,22 |
| Uzak — el %55 ölçek | %85,61 | %85,01 |
| Yakın — %132 yakınlaştırma | %85,85 | %85,73 |
| Uzak ve kadraj dışına kaymış | %87,01 | %85,95 |
| Düşük ışık | %89,33 | %89,12 |

Önceki en iyi HOG + landmark fusion baseline'ı %75,41 doğruluk ve %74,15
macro F1 idi. Yeni ensemble temiz testte doğruluğu **13,69 yüzde puan**, macro
F1'i **14,83 yüzde puan** yükseltti.

## Dayanıklılık yaklaşımı

- Eğitimde yatay aynalama, %55–125 ölçek, %16 konum kayması, dönüş,
  perspektif, ışık/renk, bulanıklık ve silme artırmaları uygulanır.
- Görüntü oranı bozulmadan letterbox ile modele verilir.
- Tahminde 0,82 / 1,00 / 1,18 ölçeklerin düz ve aynalanmış olasılıkları
  ortalanır; bu nedenle sağ ve sol el tahminleri pratikte aynı davranır.
- Webcam ön işlemesi bir veya iki eli algılar, ortak kareye alır ve uzak eller
  için çok ölçekli algılama dener.
- Güveni 0,28'in altındaki sonuçlar harf olarak zorlanmaz, `?` döndürülür.

## Dürüst sınırlar

Bu bir araştırma modelidir, ürün modeli değildir. Kaynak veri kişi ve oturum
kimliği taşımadığı için test signer-independent değildir; pseudo-oturum split'i
ardışık kare sızıntısını azaltır ancak tamamen yeni bir kullanıcıdaki başarıyı
kanıtlamaz. Uzak/yakın sonuçları kontrollü görüntü dönüşümleridir; gerçek kamera,
arka plan, hareket bulanıklığı ve farklı kullanıcılarla ayrıca test edilmelidir.

En güvenilir sonraki adım, çok sayıda kişiden hem sağ hem sol el, farklı
mesafe, açı, ışık ve arka plan kayıtları toplayıp kişiye göre ayrılmış kapalı
bir test seti oluşturmaktır.

## Lisans

Kaynak Kaggle veri seti CC BY-NC-SA 4.0 lisanslıdır. Bu model ticari kullanım
için sunulmamalıdır.
