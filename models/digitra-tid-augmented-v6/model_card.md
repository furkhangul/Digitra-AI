# Digitra TİD V6 — augmentation ile eğitilmiş araştırma modeli

## Kullanım

29 harflik TİD parmak alfabesi için statik görüntü sınıflandırıcı. RGB el görüntüsü, oranı korunarak 224×224 boyutuna getirilir. Mevcut Digitra API yükleyicisiyle uyumludur; bu paket canlı model seçimini değiştirmez.

## Ölçülen sonuçlar

| Bölüm | Doğruluk | Macro F1 |
|---|---:|---:|
| Temiz gerçek eğitim | 99,75% | 99,75% |
| Doğrulama | 91,99% | 91,68% |
| Tarihsel test | 90,26% | 90,23% |

Seçilen epoch: 15 (raw). Test: 431 gerçek görüntü.
Tek model çıkarım gecikmesi medyanı: 20.4 ms; P95: 31.4 ms. Kamera ve ağ gecikmesi dahil değildir.
Kaydetme/yükleme uyumunda en büyük olasılık farkı: 0.000003.

## Dayanıklılık testleri

| Koşul | Doğruluk | Macro F1 |
|---|---:|---:|
| clean | 90,26% | 90,23% |
| mirror | 90,26% | 90,23% |
| far | 86,54% | 86,73% |
| shift | 89,79% | 89,76% |
| rotation | 89,33% | 89,21% |
| low_light | 90,95% | 90,80% |
| blur | 89,10% | 89,14% |
| jpeg | 89,33% | 89,24% |

## Sınırlar

- Bu test grubu V5 için daha önce raporlanmıştır; yeni dış test değildir.
- Kaynakta kişi kimliği yoktur. Sonuçlar kişi bağımsız başarı kanıtı değildir.
- Statik kareler, yalnızca hareketle ayrılan harfler için yeterli olmayabilir.
- 3D eller yardımcı eğitim verisidir; 3D görseller üzerindeki başarı gerçek kamera başarısı olarak raporlanmaz.
- Gerçek veri CC BY-NC-SA 4.0 lisanslıdır; paket araştırma/ticari olmayan kullanım içindir.

## Tekrar üretme

Eğitim kodu: `ml/training/tid_v6/`. Parametreler, veri manifesti kimliği ve kullanılan sürümler `config.json` dosyasındadır. Doğrulama sonucu ile seçim `selection.json` içinde kilitlenmiştir. Eğitim ve test verileri pakete kopyalanmaz.
