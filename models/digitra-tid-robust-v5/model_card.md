# Digitra TİD Robust v5.0.0

## Kullanım amacı

- Digitra canlı kamera panelinde 29 harflik TİD parmak alfabesi tanıma.
- Sağ ve sol el ile farklı mesafe, kadraj ve ışık koşullarına dayanıklılık.
- Doğrulama macro F1 sonucuyla seçilmiş %25 MobileNetV3 Large ve %75
  EfficientNet-B0 ensemble'ı.

## Girdi ve gizlilik

Tarayıcı MediaPipe ile en fazla iki eli bulur ve yalnızca elleri çevreleyen
kare JPEG kırpımını yerel FastAPI servisine gönderir. Tam kamera karesi
gönderilmez; kırpım bellekte işlenir, diske veya geri bildirim sistemine
kaydedilmez.

## Ölçülen sonuç

- Kilitli test doğruluğu: %89,10; macro F1: %88,98.
- Aynalanmış test doğruluğu: %89,33.
- Elin %55 ölçeğe küçültüldüğü test: %85,61.
- %132 yakınlaştırma testi: %85,85.
- Uzak ve kaydırılmış el testi: %87,01.
- Düşük ışık testi: %89,33.

Test split'i 10 ardışık karelik pseudo-oturum bloklarını birlikte tutar.
Kaynak veri güvenilir kişi kimliği içermediğinden sonuç kişi bağımsız değildir.

## Çalışma zamanı

- Görüntü oranını koruyan 224×224 letterbox ön işleme.
- Her model için 0,82 / 1,00 / 1,18 ölçek ve yatay ayna TTA'sı.
- Beş tahminlik olasılık yumuşatma ve en az üç kare kararlılık kontrolü.
- Doğrulamada seçilen güven eşiği: 0,28.
- Beklenen çalışma yolu yerel CUDA GPU'dur; CUDA yoksa CPU'ya düşer.

## Sınırlar ve lisans

Bu bir araştırma adayıdır ve tamamen yeni kullanıcılarda doğrulanmış ürün modeli
değildir. Hareketli işaret/kelime tanıma yapmaz; tek karelik parmak alfabesi
sınıflandırır. Kaynak Kaggle veri seti CC BY-NC-SA 4.0 lisanslı olduğundan model
ticari kullanım için sunulmamalıdır.
