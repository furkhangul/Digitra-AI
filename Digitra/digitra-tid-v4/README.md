# Digitra TİD Robust V5

Türk İşaret Dili parmak yazımı için sağ/sol el, uzaklık, kadraj ve ışık
değişimlerine dayanıklı görüntü sınıflandırma hattı. Eski `digitra-landmark`
denemelerinden ayrı tutulur; ASL ve TİD metrikleri karıştırılmaz.

## Veri kaynakları

- İlk araştırma baseline'ı: Kaggle
  `feronial/turkish-sign-languagefinger-spelling` (2.974 görüntü,
  CC BY-NC-SA 4.0; ticari kullanım için uygun değildir).
- Hedef kelime modeli: TurkSign446 (446 sözcük, 10 kişi, 44.600 video,
  CC BY 4.0; Zenodo erişimi şu anda kısıtlıdır).
- Alternatif akademik benchmark: AUTSL (226 işaret, 43 kişi, 38.336 video;
  ticari kullanımı yasaktır).

Ham veri, çıkarılmış özellikler ve model dosyaları yeniden üretilebilir olduğu
için Git'e eklenmez.

## Çalıştırma

```powershell
.\.venv\Scripts\python.exe scripts\download_kaggle.py
.\.venv\Scripts\python.exe scripts\audit_dataset.py
.\.venv\Scripts\python.exe scripts\extract_landmarks.py
.\.venv\Scripts\python.exe scripts\train_static.py
.\.venv\Scripts\python.exe scripts\train_image_hog.py
.\.venv\Scripts\python.exe scripts\train_image_cnn.py --architecture mobilenet_v3_large
.\.venv\Scripts\python.exe scripts\train_image_cnn.py --architecture efficientnet_b0
.\.venv\Scripts\python.exe scripts\select_cnn_ensemble.py
.\.venv\Scripts\python.exe scripts\evaluate_robustness.py
```

NVIDIA GPU ile eğitim için temel bağımlılıklardan sonra CUDA tekerleklerini
kurun:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps -r requirements-cuda.txt
```

Canlı kamera demosu varsayılan olarak doğrulama setinde seçilmiş en iyi
ensemble'ı (%25 MobileNetV3 Large + %75 EfficientNet-B0) kullanır:

```powershell
.\.venv\Scripts\python.exe scripts\webcam_demo.py
```

Robust V5 hattı eli bozmadan kareye yerleştirir; sağ/sol el için yatay
aynalama, uzaklık için geniş ölçek ve konum artırması kullanır. Tahminde üç
ölçeğin hem düz hem aynalanmış görünümleri birleştirilir. Webcam katmanı da
iki eli birlikte bulup ortak bir kareye kırpar. Düşük güvenli tahminler harf
yerine `?` olarak gösterilir.

Kilitli test sonucu %89,10 doğruluk ve %88,98 macro F1'dir. Aynalanmış testte
%89,33, eli görüntünün %55'ine küçülten uzaklık stres testinde %85,61 ve
%132 yakınlaştırma testinde %85,85 doğruluk ölçülmüştür. Ayrıntılar
`MODEL_CARD.md` dosyasındadır.

Tek komut:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py
```

İlk ölçülen sonuçlar ve sınırlamalar için `MODEL_CARD.md`, veri kaynağı kararı
için `DATASET_SOURCES.md` dosyasına bakın.

## Bilimsel sınır

Kaggle setinde güvenilir kişi/oturum kimliği yoksa rastgele görüntü doğruluğu
ürün doğruluğu olarak raporlanmaz. Nihai ölçüm, Digitra'nın kendi veri toplama
protokolündeki tamamen görülmemiş kişiler üzerinde yapılmalıdır.
