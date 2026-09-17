# Digitra V3 Hibrit Model

## Ölçülen son sonuç

- Kapsam: 24 statik ASL parmak alfabesi sınıfı; J ve Z dinamik modele aittir.
- P8 geliştirme doğruluğu: %99,5417.
- Fresh locked P9-P10 test doğruluğu: %94,1458.
- Macro F1: %92,5553.
- Test örnekleri: 4.800; hata: 281.
- P9: %94,7917; P10: %93,50.

Bu değerler notebook çıktılarından korunmuştur. %99,5417 geliştirme sonucudur;
nihai test sonucu değildir.

## Mimari

1. 384 piksel DINOv3 ViT-B/16 RGB kolu.
2. MediaPipe Hand Landmarker ile 21 nokta ve 422 boyutlu kanonik geometri.
3. ExtraTrees tabanlı landmark sınıflandırıcısı.
4. N/S ve R/U zor sınıfları için çift uzmanları.
5. Dondurulmuş RGB-landmark yönlendirme kuralları.

## Veri protokolü

- ASL-HG P1-P7: eğitim.
- ASL-HG P8: geliştirme ve kalibrasyon.
- ASL-HG P9-P10: model ve kurallar dondurulduktan sonra tek seferlik kilitli test.
- J ve Z statik sınıflandırmaya dahil edilmedi.

## Yeniden üretim dosyaları

- `artifacts/DIGITRA_V3_RAW_999_RESEARCH.ipynb`
- `ml/training/train_digitra_v3_external.py`
- `ml/training/extract_digitra_v3_landmarks.py`
- `ml/training/train_digitra_v3_landmarks.py`
- `ml/training/extract_digitra_v3_locked_landmarks.py`
- `ml/training/evaluate_digitra_v3_hybrid_locked.py`

## Kritik durum

V3 ağırlıkları Colab'ın geçici `/content` diskinde tutulmuştu ve yerel diske veya
Google Drive'a kopyalanmadan çalışma zamanı sıfırlandı. Bu nedenle V3 checkpoint
dosyaları şu an mevcut değildir; yukarıdaki notebook ve betikler kullanılarak
yeniden eğitilmelidir. Yeniden üretim ve doğrulama tamamlanıncaya kadar
`artifacts/DIGITRA_V2_RELEASE.zip` çalışan geri dönüş paketi olarak korunmalıdır.
