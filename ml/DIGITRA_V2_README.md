# Digitra V2 model hattı

Bu klasördeki V2 çalışması mevcut `DIGITRA_LANDMARK_V1` modelini değiştirmez. V1, aynı örneğin 422 boyutlu geometrik görünümü olarak hibrit modele dahil edilir.

## Kapsam

- Statik model: ASL parmak alfabesindeki 24 statik harf (`J` ve `Z` hariç).
- Temporal model: `J`, `Z` ve `OTHER`.
- Bu çalışma Türk İşaret Dili modeli değildir.

## Veri protokolü

- Statik kaynak: `piotrpopis/asl-hands`, 16 signer ve 37.891 görüntü.
- Geliştirme split'i: train signer `0–9,12,13`; validation signer `10,11`; locked test signer `14,15`.
- Final split: signer `10` eğitime eklenir, signer `11` kalibrasyonda kalır, signer `14,15` kilitli test olarak korunur.
- Exact SHA-256 ve perceptual dHash bölümler arası sızıntı denetimi yapılır.
- J/Z kaynağı: `signnteam/asl-sign-language-alphabet-videos-j-z`, 712 video. Kaynak signer kimliği yayımlamadığı için video-group ayrımı signer-disjoint olarak sunulmaz.

## Mimari

```text
RGB el ROI ── DINOv3 ViT-S/16 ───────────────┐
aynı örnek ── V1 canonical 422D landmark ────┼─ validation-based fusion
                                              └─ temperature calibration + reject

32 × 130D temporal landmark ── TCN ── BiGRU ── attention ── J/Z/OTHER
```

Yüksek güven modu düşük güvenli örnekleri `UNKNOWN` olarak reddeder. Seçici doğruluk yalnız kabul edilen örneklerdeki doğruluktur; bütün kamera akışının ham doğruluğu değildir.

## Dosyalar

- `training/prepare_digitra_v2_static.py`: ROI, aynı-örnek landmark ve sızıntı denetimi.
- `training/train_digitra_v2_static.py`: DINOv3 + V1 füzyon, kalibrasyon ve kilitli test.
- `training/prepare_digitra_v2_dynamic.py`: J/Z temporal landmark çıkarımı.
- `training/train_digitra_v2_dynamic.py`: J/Z/OTHER temporal kapısı ve TorchScript parity.
- `export/export_digitra_v2_static.py`: ONNX export, gerçek örneklerle runtime parity ve checksum.
- `notebooks/DIGITRA_V2_RESEARCH_CLEAN.ipynb`: yalnız gerekli dokuz hücrelik Colab akışı.

## Üretim öncesi zorunlu sonraki veri

Gerçek kamera genellemesi için farklı cilt tonları, ışıklar, kameralar, sağ/sol el ve `OTHER` hareketleri içeren izinli Digitra kayıtları gerekir. Düşük güvenli örnekler otomatik etiketlenmez; kullanıcı onayı ve insan doğrulamasıyla active-learning havuzuna alınır.
