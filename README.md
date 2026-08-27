# Digitra-AI

ASL (American Sign Language) el landmark tanima projesi - MediaPipe + MLP

## Proje Yapisi

```
Digitra/
├── digitra-landmark v2/          # Ana model v2 (aktif)
│   ├── asl_landmarks_*.csv        # Ham ve geometrik dataset (LFS)
│   ├── train_v4.csv / val_v4.csv / test_v4.csv  # Split dataset
│   ├── build_geometric_features_v3.py  # 63 ham + ~50 geometrik feature uretimi
│   ├── train_mlp_v5.py           # MLP (256,128,64) + StandardScaler
│   ├── digitra_asl_*.pkl         # Egitilmis model, scaler, encoder (LFS)
│   ├── collect_webcam_landmarks_v9.py  # Webcam veri toplama (G,H,K,M,P,Q,R,S,T)
│   ├── webcam_test_v6.py / v7.py # Canli test
│   ├── webcam_evaluation_v8.py   # Degerlendirme
│   ├── hand_landmarker.task      # MediaPipe model (LFS)
│   └── confusion_matrix_v5.csv   # Sonuclar
│
│   # Buyuk ham goruntu verisi .gitignore'da (1GB+ her biri)
│   # archive/ ve dataset_split/ lokalde mevcut, GitHub'a push edilmedi
│   # Ihtiyac halinde Release veya harici storage ile paylasilir
│
└── digitra-landmark v1/          # Eski v1 (v2 ile degistirildi, history'de mevcut)
```

## Model v2 Detay

- Girdi: MediaPipe 21 nokta (x,y,z = 63 feature) + geometrik feature'lar (aci, mesafe, avuc orani, parmak extension vb.) = toplam ~113 feature
- Model: MLPClassifier(hidden_layer_sizes=(256,128,64), early_stopping=True) - digitra-landmark v2/train_mlp_v5.py:89
- On isleme: StandardScaler + LabelEncoder (26 harf)
- Ozellik uretimi: digitra-landmark v2/build_geometric_features_v3.py:93 (create_geometric_features)

## Kurulum

```bash
pip install mediapipe opencv-python pandas numpy scikit-learn joblib
```

## Kullanim

```bash
# Geometrik feature uret
python "digitra-landmark v2/build_geometric_features_v3.py"

# Egit
python "digitra-landmark v2/train_mlp_v5.py"

# Canli test
python "digitra-landmark v2/webcam_test_v7.py"

# Webcam ile veri topla
python "digitra-landmark v2/collect_webcam_landmarks_v9.py"
```

## Git LFS

Buyuk CSV / pkl / task dosyalari Git LFS ile takip ediliyor:

```
*.csv filter=lfs
*.pkl filter=lfs
*.task filter=lfs
```

Ilk klonda: `git lfs pull`

## Notlar

- archive/ ve dataset_split/ (~2GB, 174k goruntu) repo boyutunu kucultmek icin .gitignore'da birakildi. Lokal digitra-landmark v2/ altinda mevcut.
- v1 gecmisi git log icinde korunuyor, ana branch artik v2'yi gosteriyor.

## Lisans

MIT
