# DIGITRA

## Uçtan Uca Parmak Alfabesi, İşaret Dili, Görüntü İşleme, Ses ve 3D Avatar Platformu

### Master Proje Rehberi ve Teknik Gereksinim Dokümanı

> **Belge amacı:** Bu dosya Digitra projesinin tek ana rehberidir.
> Google Colab üzerinde veri ediniminden başlayarak dört farklı modelin
> eğitilmesi, veri sızıntısının önlenmesi, modellerin
> karşılaştırılması/birleştirilmesi, gerçek zamanlı kamera sistemi,
> metin/ses dönüşümleri, 3D avatar, web arayüzü, test, güvenlik, MLOps
> ve deployment dahil bütün geliştirme sürecini tarif eder.
>
> **Temel kural:** Gerçek ölçüm yapılmadan hiçbir başarı oranı, doğruluk
> veya performans değeri uydurulmayacaktır.

------------------------------------------------------------------------

# 1. Proje Özeti

**Digitra**, kameradan gelen el/parmak görüntülerini analiz ederek
parmak alfabesindeki karakterleri tanımayı; bu karakterleri metne ve
sese dönüştürmeyi; ters yönde ses veya metni işaret/3D avatar
hareketlerine çevirmeyi hedefleyen çok modlu bir iletişim platformudur.

Nihai akış:

``` text
KAMERA
  ↓
El Tespiti
  ↓
Görüntü + Landmark
  ↓
Birden Fazla AI Modeli
  ↓
Model Fusion / Karar Katmanı
  ↓
Temporal Smoothing
  ↓
Harf → Kelime → Cümle
  ↓
Metin
  ↓
TTS → Ses
```

Ters yön:

``` text
MİKROFON / METİN
  ↓
Speech-to-Text (gerekirse)
  ↓
Metin
  ↓
Dil/İşaret Eşleme Katmanı
  ↓
İşaret Dizisi
  ↓
3D Avatar
  ↓
El/Kol/Yüz Animasyonu
```

------------------------------------------------------------------------

# 2. Projenin Bilimsel Sınırı

İlk prototipte aşağıdaki veri kaynakları **ASL --- American Sign
Language** içindir. Bunlar **Türk İşaret Dili (TİD)** değildir.

ASL modeli hiçbir yerde TİD modeli olarak sunulmamalıdır.

Digitra'nın nihai hedefi TİD ise:

1.  Önce ASL ile teknik pipeline doğrulanır.
2.  Sonra güvenilir TİD veri seti toplanır/edinilir.
3.  Aynı leakage-free eğitim pipeline'ı TİD verisine uygulanır.
4.  TİD sonuçları ayrı raporlanır.
5.  Dil uzmanı/işaret dili uzmanı doğrulaması yapılır.

------------------------------------------------------------------------

# 3. Kullanılacak Veri Kaynakları

## Dataset A --- Ana Görüntü Dataseti

**Kaggle: ASL Alphabet --- grassknoted**

https://www.kaggle.com/datasets/grassknoted/asl-alphabet

Amaç: - Custom CNN - EfficientNet/MobileNet - Görüntü branch'i -
MediaPipe landmark çıkarımı

Kaggle sayfasına göre: - yaklaşık 87.000 training görüntüsü - 200×200 -
29 sınıf - A--Z - SPACE - DELETE - NOTHING

**Digitra'da ana statik görüntü training kaynağı budur.**

KaggleHub örneği:

``` python
!pip install -q kagglehub

import kagglehub

dataset_a = kagglehub.dataset_download(
    "grassknoted/asl-alphabet"
)

print(dataset_a)
```

Path hiçbir zaman elle tahmin edilmemeli; indirme sonrası Python ile
keşfedilmelidir.

------------------------------------------------------------------------

## Dataset B --- Harici Test / Domain Shift Dataseti

**Kaggle: ASL Alphabet Test --- danrasband**

https://www.kaggle.com/datasets/danrasband/asl-alphabet-test

Amaç: - Ana training'e karıştırmamak - External test - Farklı arka
planlarda genelleme testi

Bu veri kaynağı, ana ASL Alphabet modelinin preprocessing ve
genellemesini farklı görüntüler üzerinde sınamak için tasarlanmıştır.

Örnek:

``` python
dataset_b = kagglehub.dataset_download(
    "danrasband/asl-alphabet-test"
)
```

**Bu dataset hyperparameter seçimi için kullanılmayacak.**

------------------------------------------------------------------------

## Dataset C --- İkinci ASL Görüntü Kaynağı

**Kaggle: American Sign Language Alphabet Dataset --- rupaul007**

https://www.kaggle.com/datasets/rupaul007/american-sign-language-alphabet-dataset

Amaç: - Dataset yapısı ve lisansı doğrulandıktan sonra external
validation/test adayı - Domain-shift analizi - Gerektiğinde yalnızca
kontrollü ikinci deney

Örnek:

``` python
dataset_c = kagglehub.dataset_download(
    "rupaul007/american-sign-language-alphabet-dataset"
)
```

**Dataset A ile körlemesine birleştirilmemeli.** Önce duplicate, label
ve provenance kontrolü yapılmalıdır.

------------------------------------------------------------------------

## Dataset D --- MediaPipe Landmark Dataseti

**Kaggle: MediaPipe Processed ASL Dataset --- vignonantoine**

https://www.kaggle.com/datasets/vignonantoine/mediapipe-processed-asl-dataset

Amaç: - Landmark tabanlı model - 21 el noktası üzerinden özellik
çıkarımı - Görüntüden bağımsız hızlı model - CNN ile hibrit kullanım

Dataset açıklamasına göre MediaPipe ile işlenmiş el/palm/finger konum
özellikleri içerir; ASL harfleri ve sayıları kapsar.

Örnek:

``` python
dataset_d = kagglehub.dataset_download(
    "vignonantoine/mediapipe-processed-asl-dataset"
)
```

------------------------------------------------------------------------

## Dataset E --- Dinamik İşaret / Sequence Dataseti

**Google --- Isolated Sign Language Recognition**

https://www.kaggle.com/competitions/asl-signs/data

Amaç: - V1 statik harf modeline doğrudan katılmayacak - Daha sonraki
video/sequence modelinin ana kaynaklarından biri olacak -
Transformer/LSTM/GRU/Temporal CNN araştırması - Participant/signer bazlı
split

Veri: - MediaPipe Holistic landmark sequence'ları - participant_id -
sequence_id - sign - frame - face / pose / left_hand / right_hand
landmarkları

**Çok önemli:** `participant_id` bulunduğu için random sequence split
yerine signer/participant holdout kullanılmalıdır.

------------------------------------------------------------------------

# 4. Datasetlerin Görev Dağılımı

  Kaynak      Ana Görev                   Training?       Final Test?
  ----------- ------------------ ------------------ -----------------
  Dataset A   Statik görüntü                   Evet        İç holdout
  Dataset B   Farklı arka plan                Hayır              Evet
  Dataset C   İkinci domain        Varsayılan hayır              Aday
  Dataset D   Landmark model                   Evet   Kendi holdout'u
  Dataset E   Dinamik işaret                  V2/V3    Signer holdout

**Kural:** "Daha fazla veri = her şeyi birleştir" yaklaşımı
kullanılmayacak.

------------------------------------------------------------------------

# 5. Google Colab Çalışma Düzeni

Her model için ayrı notebook önerilir:

``` text
01_digitra_data_audit.ipynb
02_digitra_custom_cnn.ipynb
03_digitra_transfer_learning.ipynb
04_digitra_landmark.ipynb
05_digitra_hybrid.ipynb
06_digitra_ensemble.ipynb
07_digitra_dynamic_sign.ipynb
08_digitra_export_benchmark.ipynb
```

Google Drive:

``` text
MyDrive/
└── Digitra/
    ├── datasets/
    ├── manifests/
    ├── splits/
    ├── experiments/
    ├── models/
    ├── reports/
    ├── exports/
    └── logs/
```

Raw datasetleri Drive'a kopyalamak zorunlu değildir. KaggleHub cache
kullanılabilir. Kalıcı tutulması gerekenler: - split manifestleri -
deney sonuçları - modeller - scaler/config - raporlar

------------------------------------------------------------------------

# 6. Colab Environment Kontrolü

İlk hücre:

``` python
import sys, os, platform
print(sys.version)
print(platform.platform())
```

GPU:

``` python
!nvidia-smi
```

TensorFlow kullanılıyorsa:

``` python
import tensorflow as tf
print(tf.__version__)
print(tf.config.list_physical_devices("GPU"))
```

PyTorch kullanılıyorsa:

``` python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "GPU yok")
```

GPU yoksa büyük görüntü modelinin training'i başlatılmadan kullanıcı
uyarılmalıdır.

------------------------------------------------------------------------

# 7. Reproducibility

Tek seed:

``` python
SEED = 42
```

Şunlarda sabitlenir: - Python random - NumPy - TensorFlow/PyTorch -
split - DataLoader shuffle - hyperparameter experiment

Her deneyde: - seed - package versions - dataset version - model
version - config kaydedilir.

------------------------------------------------------------------------

# 8. Dataset Audit --- Eğitimden Önce Zorunlu

Her dataset için manifest oluştur:

``` text
filepath
label
width
height
channels
filesize
sha256
phash
dataset_source
subject_id (varsa)
session_id (varsa)
sequence_id (varsa)
```

Kontroller: - toplam dosya - toplam sınıf - sınıf isimleri - sınıf
başına örnek - çözünürlük - RGB/grayscale - bozuk dosya - sıfır byte -
okunamayan JPEG/PNG - eksik label - beklenmeyen label - duplicate - near
duplicate - class imbalance

Çıktılar:

``` text
dataset_manifest.csv
dataset_summary.json
corrupt_images.csv
duplicate_report.csv
class_distribution.png
resolution_distribution.png
```

------------------------------------------------------------------------

# 9. Data Leakage --- Kesin Kurallar

## Yasak

``` text
Augmentation
↓
Train/Test Split
```

## Doğru

``` text
RAW
↓
Quality Check
↓
Exact Duplicate Check
↓
Near-Duplicate Check
↓
Subject/Session/Video Grouping
↓
TRAIN / VALIDATION / TEST
↓
Augmentation yalnızca TRAIN
```

Ek yasaklar: - testte scaler fit - testte normalization parametresi
öğrenmek - test sonucuna bakıp LR/dropout seçmek - external test
sonucuna göre modeli tune etmek - aynı signer'ı train ve testte tutmak
(metadata varsa) - aynı videonun frame'lerini farklı splitlere
dağıtmak - aynı orijinal görüntünün augment edilmiş kopyalarını farklı
splitlere koymak

------------------------------------------------------------------------

# 10. Exact Duplicate Kontrolü

SHA-256 gibi hash kullan.

``` python
import hashlib

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
```

Aynı hash: - aynı dosya - farklı isim - farklı klasör olsa bile
duplicate kabul edilir.

------------------------------------------------------------------------

# 11. Near-Duplicate Kontrolü

Exact hash yeterli değildir.

Araştır: - pHash - dHash - aHash - image embedding similarity

Near-duplicate kümeleri split'ten önce group olarak ele alınmalıdır.

Özellikle ardışık çekilmiş çok benzer kareler validation skorunu yapay
olarak yükseltebilir.

------------------------------------------------------------------------

# 12. Signer / Subject Leakage

Metadata varsa:

``` text
Signer 01 → TRAIN
Signer 02 → TRAIN
Signer 03 → VALIDATION
Signer 04 → TEST
```

aynı signer birden fazla splitte bulunmamalı.

Dataset E'deki `participant_id` bu nedenle kritik bir grouping alanıdır.

------------------------------------------------------------------------

# 13. Video Leakage

Bir video:

``` text
video_001
 ├ frame_001
 ├ frame_002
 ├ ...
 └ frame_300
```

ise tüm frame'ler aynı splitte kalmalıdır.

------------------------------------------------------------------------

# 14. Split Stratejisi

Metadata yeterliyse:

**Group-aware / signer-aware split**.

Metadata yoksa: - stratified split - duplicate group awareness

Başlangıç oranı:

``` text
Train      70%
Validation 15%
Test       15%
```

Alternatif:

``` text
80 / 10 / 10
```

ama oran deney başlamadan belirlenmeli.

`test.csv` oluşturulduktan sonra final aşamaya kadar açılmaması tercih
edilir.

Çıktı:

``` text
train_manifest.csv
val_manifest.csv
test_manifest.csv
```

------------------------------------------------------------------------

# 15. Class Imbalance

Önce raporla.

Sonra gerekirse: 1. veri toplama 2. class weight 3. weighted loss 4.
kontrollü oversampling

**Validation ve test oversample edilmez.**

Class weights yalnızca TRAIN'den hesaplanır.

------------------------------------------------------------------------

# 16. Görüntü Preprocessing

Başlangıç: - RGB - 224×224 - modelin gerektirdiği normalization

Custom CNN:

``` text
0..255 → 0..1
```

EfficientNet/MobileNet: - resmi preprocessing fonksiyonu

Bounding/cropping: - MediaPipe ile el bölgesi bulunabilir - crop padding
uygulanabilir - crop işlemi train/val/testte aynı deterministic pipeline
ile yapılır

------------------------------------------------------------------------

# 17. Augmentation

Sadece TRAIN.

Aday: - küçük rotation - küçük translation - küçük zoom - brightness -
contrast - hafif blur/noise

**Horizontal flip varsayılan olarak kapalı.** İşaretin yönünü/handedness
bilgisini değiştirebilir.

Her augmentation deney olarak kaydedilir.

------------------------------------------------------------------------

# 18. MODEL 1 --- Custom CNN

Amaç: Baseline.

``` text
224×224×3
↓
Conv
↓
BatchNorm
↓
Activation
↓
Pool
↓
Conv Blocks
↓
Global Average Pooling
↓
Dropout
↓
Dense
↓
Softmax
```

Zorunlu: - parameter count - early stopping - checkpoint - LR
scheduler - training curves

Dosya:

``` text
digitra_cnn_v1.keras
digitra_cnn_v1.onnx
```

------------------------------------------------------------------------

# 19. MODEL 2 --- Transfer Learning

Aday: - EfficientNetB0 - MobileNetV3

Amaç: - daha iyi genelleme - daha hızlı convergence - production için
dengeli model

Aşama 1:

``` text
ImageNet Backbone = Frozen
Classifier = Train
```

Aşama 2:

``` text
Son backbone bloklarının bir kısmı açılır
Learning rate düşürülür
Fine-tuning yapılır
```

Tüm backbone ilk anda açılmaz.

Dosya:

``` text
digitra_efficientnet_v1.keras
digitra_efficientnet_v1.onnx
```

------------------------------------------------------------------------

# 20. MODEL 3 --- MediaPipe Landmark Modeli

MediaPipe Hands: - 21 landmark - x, y, z - handedness

Temel feature:

``` text
21 × 3 = 63
```

Ek feature deneyleri: - wrist-relative coordinates - scale
normalization - joint angle - fingertip distance - palm orientation

Normalization:

``` text
scaler.fit(TRAIN)
scaler.transform(VAL)
scaler.transform(TEST)
```

Testte `fit` yasaktır.

Aday classifier: - MLP - SVM - Random Forest - gerektiğinde gradient
boosting

Amaç: - küçük model - yüksek FPS - browser/mobile

Dosyalar:

``` text
digitra_landmark_v1.keras / .pkl
landmark_scaler.pkl
landmark_config.json
```

------------------------------------------------------------------------

# 21. MODEL 4 --- Hybrid Image + Landmark

``` text
RGB Image
   ↓
EfficientNet
   ↓
Visual Vector ───────────┐
                         ├─ Concatenate → Dense → Softmax
Landmarks                │
   ↓                     │
MLP                       │
   ↓                     │
Landmark Vector ─────────┘
```

Bu model iki bilgi kaynağını birlikte kullanır: - görünüş - geometrik el
yapısı

Ablation zorunlu: 1. Image only 2. Landmark only 3. Image + Landmark

Hybrid gerçekten katkı sağlamıyorsa sırf karmaşık olduğu için
production'a alınmamalıdır.

Dosya:

``` text
digitra_hybrid_v1.keras
digitra_hybrid_v1.onnx
```

------------------------------------------------------------------------

# 22. MODEL 5 --- Dinamik İşaret Modeli (İleri Faz)

Dataset E kullanılır.

Amaç: - tek frame ile anlaşılamayan hareketler - dinamik sign
recognition

Girdiler:

``` text
T × landmarks
```

Adaylar: - GRU - LSTM - Temporal CNN - Transformer

Split: **participant_id / signer holdout**.

Bu model statik harf classifier ile karıştırılmamalıdır.

------------------------------------------------------------------------

# 23. Modelleri "Birbirine Bağlama" --- Doğru Yaklaşım

Dört modeli körlemesine zincirleme bağlama.

Önce her biri bağımsız benchmark edilir.

Sonra üç production stratejisi değerlendirilir.

## A. Routing

``` text
MediaPipe Hand Detector
        ↓
Statik mi / dinamik mi?
  ↓                 ↓
Static Model    Sequence Model
```

## B. Hybrid Feature Fusion

Görüntü + landmark aynı neural network içinde birleşir.

## C. Ensemble / Decision Fusion

Örnek:

``` text
EfficientNet → probability vector ┐
                                  │
CNN → probability vector ─────────┼→ Fusion → Final Probability
                                  │
Landmark → probability vector ────┘
```

Basit başlangıç:

``` text
weighted average
```

Örnek:

``` text
final = 0.50 * efficientnet
      + 0.20 * cnn
      + 0.30 * landmark
```

**Bu ağırlıklar örnektir; gerçek ağırlıklar validation üzerinde
belirlenir. Test üzerinde değil.**

Daha ileri: - stacking - meta-classifier - confidence-aware gating

Meta-classifier eğitilecekse base modelin training verisi üzerinde
in-sample tahmin kullanarak leakage oluşturulmamalıdır. Out-of-fold
predictions kullanılmalıdır.

------------------------------------------------------------------------

# 24. Nihai Tahmin Pipeline'ı

``` text
Webcam Frame
   ↓
Quality Gate
   ↓
Hand Detection
   ↓
ROI Crop
   ↓
┌───────────────────────┐
│                       │
↓                       ↓
Image Model        MediaPipe Landmarks
│                       │
↓                       ↓
Image Prob.        Landmark Prob.
└──────────┬────────────┘
           ↓
     Fusion / Gating
           ↓
   Confidence Calibration
           ↓
    Temporal Smoothing
           ↓
 Stability / Debounce
           ↓
     Final Character
           ↓
      Text Buffer
```

------------------------------------------------------------------------

# 25. Quality Gate

Modelden önce kontrol: - el var mı? - el yeterince büyük mü? - görüntü
aşırı karanlık mı? - blur fazla mı? - el crop dışında mı? - birden fazla
el var mı?

Kalite yetersizse tahmin zorlanmamalı.

UI mesajı:

``` text
Işık yetersiz.
Elinizi kameraya yaklaştırın.
El algılanamadı.
```

------------------------------------------------------------------------

# 26. Confidence Calibration

Softmax confidence doğrudan "doğruluk" değildir.

Ölç: - reliability diagram - Expected Calibration Error - Brier score

Gerekirse: - temperature scaling

Production threshold validation setinde belirlenir.

------------------------------------------------------------------------

# 27. Temporal Smoothing

Tek frame → doğrudan harf değildir.

Örnek:

``` text
A A A E A A
```

çıktısında E reddedilebilir.

Yöntemler: - majority vote - probability moving average - exponential
smoothing - stability window - hysteresis - debounce

Bir karakterin buffer'a eklenmesi için: - minimum confidence - minimum
stability - minimum süre gibi koşullar olabilir.

------------------------------------------------------------------------

# 28. Repeated Letter Problemi

Kullanıcı `LL` yazmak isterse sistem aynı işareti sonsuza kadar tekrar
eklememelidir.

Çözüm: - neutral/no-hand state - explicit SPACE/NEXT gesture -
cooldown - gesture release detection

UX test edilerek seçilir.

------------------------------------------------------------------------

# 29. Harf → Kelime → Cümle

Character buffer:

``` text
H
HE
HEL
HELL
HELLO
```

Kontroller: - geri al - sil - boşluk - temizle - seslendir

NLP correction opsiyonel.

Orijinal çıktı saklanmalı:

``` text
Raw: ...
AI corrected: ...
```

AI düzeltmesi model tahmini gibi gösterilmemeli.

------------------------------------------------------------------------

# 30. Speech-to-Text

``` text
Microphone
↓
STT
↓
Text
```

Provider abstraction kullanılmalı.

API key frontend'e koyulmaz.

Browser native STT veya backend/provider seçenekleri ayrı
değerlendirilebilir.

------------------------------------------------------------------------

# 31. Text-to-Speech

``` text
Digitra Text
↓
TTS
↓
Audio
```

Kontroller: - voice - language - speed - pitch - autoplay

------------------------------------------------------------------------

# 32. Metin/Ses → İşaret

Bu problem basit kelime değiştirme değildir.

``` text
Speech
↓
STT
↓
Text
↓
Language Processing
↓
Sign Representation / Gloss
↓
Animation Sequence
↓
3D Avatar
```

TİD için Türkçe cümleyi kelime kelime birebir işarete çevirmek doğru
kabul edilmemelidir. Dilbilgisi/işaret dili uzmanlığı gerektirir.

------------------------------------------------------------------------

# 33. 3D Avatar

Teknoloji adayları: - Blender - glTF / GLB - Three.js - React Three
Fiber - Mixamo yalnızca uygun animasyonlarda - özel hand rig

Avatar gereksinimleri: - iki el rig - parmak kemikleri - kol - omuz -
yüz - baş - mimik - blendshape - animation blending

İşaret dilinde yüz/beden bilgisi önemli olabileceği için yalnızca el
animasyonu nihai çözüm sayılmamalıdır.

------------------------------------------------------------------------

# 34. Animation Library

``` text
signs/
├── A.glb
├── B.glb
├── ...
└── metadata.json
```

veya tek rig + animation clips.

Metadata:

``` json
{
  "id": "sign_x",
  "language": "ASL",
  "type": "static",
  "duration_ms": 1200,
  "clip": "..."
}
```

Lisans metadata'sı ayrıca tutulmalı.

------------------------------------------------------------------------

# 35. UI/UX Vizyonu

Digitra: - premium - bilimsel - modern - sade - erişilebilir -
teknolojik

olmalı.

Apple esintili fakat kopya olmayan: - sticky scroll sections -
scroll-linked animation - 3D hero - cinematic transitions - büyük
typography - whitespace - progressive disclosure

------------------------------------------------------------------------

# 36. Ana Sayfa

Öneri:

``` text
Navigation
↓
3D Hero
↓
Live Demo CTA
↓
How Digitra Works
↓
Vision + Landmark Visualization
↓
Four Model Architecture
↓
Live Statistics
↓
Speech ↔ Sign
↓
3D Avatar
↓
Scientific Validation
↓
Accessibility
↓
Datasets & Methodology
↓
FAQ
↓
Footer
```

------------------------------------------------------------------------

# 37. Tema

-   Light
-   Dark
-   System
-   isteğe bağlı High Contrast

Design tokens:

``` text
background
foreground
surface
primary
secondary
muted
border
success
warning
danger
```

------------------------------------------------------------------------

# 38. Live Camera UI

Göster: - video - hand skeleton - bounding box - final prediction -
top-3 predictions - confidence - stability - FPS - latency - active
model - sentence buffer

Normal kullanıcı için teknik metrikler gizlenebilir; Research Mode'da
açılır.

------------------------------------------------------------------------

# 39. Model Modları

## Fast

Landmark model.

## Balanced

MobileNet/EfficientNet.

## Accurate

Hybrid/ensemble.

Gerçek performans ölçülmeden hangi modelin hangi moda atanacağı
kesinleştirilmez.

------------------------------------------------------------------------

# 40. Research Mode

Göster:

``` text
Prediction
Top-3
Confidence
Calibrated confidence
Stability
Latency
FPS
Model version
Input size
Handedness
```

Ayrıca: - Grad-CAM - landmark visualization - probability chart

------------------------------------------------------------------------

# 41. Dashboard

Kullanıcı/araştırma dashboard'u:

-   toplam inference
-   model version
-   ortalama latency
-   FPS
-   confidence dağılımı
-   harf dağılımı
-   düşük güven oranı
-   feedback accuracy (varsa)
-   model comparison

Sahte demo istatistikleri gerçek production metriği gibi sunulmaz.

------------------------------------------------------------------------

# 42. Eğitim Metrikleri

Zorunlu: - accuracy - macro precision - macro recall - macro F1 -
weighted F1 - confusion matrix - per-class report

Ek: - calibration - latency - throughput - model size - parameter count

------------------------------------------------------------------------

# 43. Overfitting

Belirti:

``` text
Train çok iyi
Validation kötüleşiyor
```

Kontrol: - loss curves - accuracy curves - macro F1 curves

Çözüm adayları: - augmentation - dropout - L2/weight decay - smaller
model - early stopping - fine-tuning depth

Her değişiklik deney olarak kaydedilir.

------------------------------------------------------------------------

# 44. Underfitting

Belirti:

``` text
Train düşük
Validation düşük
```

Kontrol: - model kapasitesi - LR - preprocessing - training duration -
feature quality - crop kalitesi

------------------------------------------------------------------------

# 45. Hyperparameter Tuning

Test seti kullanılmaz.

Validation üzerinden: - learning rate - dropout - weight decay - batch
size - augmentation - fine-tuning depth

denenir.

Deney tablosu:

``` text
experiment_id
model
seed
lr
batch
dropout
augmentation
best_epoch
val_macro_f1
val_loss
```

------------------------------------------------------------------------

# 46. Error Analysis

En az: - 20 random error - 20 high-confidence error - class confusion
pairs

Her hata:

``` text
image
true
predicted
confidence
top3
```

İnsan gözüyle nedenleri sınıflandır: - blur - lighting - crop - similar
handshape - occlusion - label issue - background - model failure

------------------------------------------------------------------------

# 47. Robustness Suite

Ayrı test seti: - düşük ışık - yüksek ışık - karmaşık arka plan - farklı
cilt tonları - sağ/sol el - farklı kamera - farklı mesafe - farklı açı -
kısmi kapanma - motion blur - farklı kıyafet/background

Robustness test training tuning amacıyla sürekli kullanılmamalıdır.

------------------------------------------------------------------------

# 48. External Test

Ana training bittikten sonra Dataset B/C.

Amaç:

``` text
Aynı distribution'da mı iyi,
yoksa gerçek farklı görüntüde de iyi mi?
```

External skorları model seçiminde ayrı raporlanır.

------------------------------------------------------------------------

# 49. Final Model Selection

Tek accuracy yasak.

Önerilen sıralama: 1. external macro F1 2. internal test macro F1 3.
robustness 4. calibration 5. latency 6. FPS 7. model size 8. accuracy

------------------------------------------------------------------------

# 50. Model Export

Her production model:

``` text
model.keras / .pt
model.onnx
labels.json
preprocessing.json
metrics.json
model_card.md
```

Landmark:

``` text
scaler.pkl
feature_config.json
```

ONNX export sonrası aynı test örneklerinde source model vs ONNX output
farkı kontrol edilmelidir.

------------------------------------------------------------------------

# 51. Browser Deployment

Tercih sırası:

``` text
ONNX Runtime Web + WebGPU
↓
WASM fallback
↓
Server inference fallback
```

Ama browser compatibility gerçek cihazlarda test edilir.

Kamera mümkünse browser içinde işlenir.

------------------------------------------------------------------------

# 52. Privacy by Design

Varsayılan:

``` text
Camera Frame
↓
Inference
↓
Result
↓
Frame discarded
```

Görüntü/video kaydetme yok.

Kayıt gerekiyorsa: - explicit consent - retention - delete -
encryption - privacy notice

------------------------------------------------------------------------

# 53. Backend

Gerekirse FastAPI/Node.

Örnek endpoint:

``` text
GET  /health
GET  /models
POST /predict
POST /stt
POST /tts
POST /feedback
GET  /model-metrics
```

Her frame'i server'a HTTP POST etmek gerçek zamanlı sistem için ilk
tercih değildir.

------------------------------------------------------------------------

# 54. Frontend

Öneri: - React - TypeScript - Vite veya Next.js - Tailwind - Framer
Motion - Three.js - React Three Fiber

Gereksiz dependency ekleme.

------------------------------------------------------------------------

# 55. Security

-   `.env`
-   secret manager
-   API key frontend'e yok
-   CORS
-   CSP
-   rate limit
-   input validation
-   secure headers
-   dependency scanning

`.gitignore`:

``` text
.env
credentials*
*.key
*.pem
__pycache__
models/large/*
datasets/*
```

------------------------------------------------------------------------

# 56. Accessibility

-   keyboard navigation
-   focus indicators
-   ARIA
-   screen reader
-   high contrast
-   scalable font
-   captions
-   reduced motion
-   no-color-only feedback

`prefers-reduced-motion` desteklenmeli.

------------------------------------------------------------------------

# 57. Responsive

-   desktop
-   laptop
-   tablet
-   mobile

Mobilde: - kamera full/near-full screen - büyük kontroller - 3D kalite
düşürülebilir - model Fast moda geçebilir

------------------------------------------------------------------------

# 58. Performance

Web: - lazy loading - code splitting - AVIF/WebP - compressed GLB -
texture optimization - LOD - model lazy load - worker thread mümkünse

UI 60 FPS hedefi. AI inference donanıma göre ayrı benchmark edilir.

------------------------------------------------------------------------

# 59. Testing

## ML

-   preprocessing consistency
-   label mapping
-   input shape
-   no-hand
-   invalid image
-   deterministic test
-   split overlap assertion

## Backend

-   API
-   validation
-   auth
-   error handling

## Frontend

-   component
-   camera permission
-   state
-   buffer

## E2E

``` text
Camera → Hand → Prediction → Text → TTS
```

------------------------------------------------------------------------

# 60. Split Integrity Test

Otomatik test:

``` python
assert set(train_ids).isdisjoint(val_ids)
assert set(train_ids).isdisjoint(test_ids)
assert set(val_ids).isdisjoint(test_ids)
```

Aynı kontroller: - SHA hash - duplicate group - signer - video/sequence
için yapılır.

Fail olursa training başlamaz.

------------------------------------------------------------------------

# 61. MLOps

Her deney: - config - seed - git commit - dataset version - metric -
artifact

ile loglanır.

Araç adayları: - TensorBoard - MLflow - Weights & Biases

Secret güvenliği korunur.

------------------------------------------------------------------------

# 62. Model Registry

Durum:

``` text
Development
Candidate
Validated
Production
Archived
```

Production promotion manuel/onaylı olabilir.

------------------------------------------------------------------------

# 63. Model Card

İçerik: - model - version - purpose - dataset - split - metrics -
external metrics - limitations - intended use - out-of-scope -
bias/robustness - license - preprocessing

------------------------------------------------------------------------

# 64. Data Card

Her dataset: - source - source URL - license - version - labels - size -
known limitations - provenance - allowed usage

Lisans kontrol edilmeden ticari kullanım varsayılmaz.

------------------------------------------------------------------------

# 65. Monitoring

Production: - inference latency - errors - confidence distribution -
predicted class distribution - model version - user correction feedback

Kamera frame'i loglanmaz.

------------------------------------------------------------------------

# 66. Active Learning

``` text
Low Confidence / User says Wrong
↓
Optional consented sample
↓
Review queue
↓
Human label
↓
Quality control
↓
Dataset candidate
↓
Next model version
```

Model otomatik kendi tahminini gerçek label olarak dataset'e eklemez.

------------------------------------------------------------------------

# 67. Drift

Takip: - confidence drift - class distribution drift - latency drift -
feedback error rate

Drift = otomatik retrain zorunluluğu değildir; inceleme tetikler.

------------------------------------------------------------------------

# 68. Yasaklar / Anti-Patterns

-   ASL = TİD demek
-   sahte accuracy
-   test leakage
-   random signer leakage
-   video frame leakage
-   augmentation before split
-   testte scaler fit
-   testle hyperparameter tuning
-   düşük confidence'ı zorla label yapmak
-   tek frame'i doğrudan harf buffer'a basmak
-   raw camera storage by default
-   secrets frontend
-   lisanssız asset
-   ağır 3D ile mobil siteyi kilitlemek
-   sırf daha karmaşık diye hybrid'i seçmek
-   accuracy tek başına kullanmak
-   model export edip inference doğrulamamak

------------------------------------------------------------------------

# 69. Fark Yaratacak Özellikler

1.  **Confidence Ring**
2.  **Hand Skeleton Overlay**
3.  **Prediction Timeline**
4.  **Stability Meter**
5.  **Top-3 Prediction**
6.  **Fast/Balanced/Accurate**
7.  **Research Mode**
8.  **Grad-CAM**
9.  **Offline inference**
10. **WebGPU**
11. **Low-light warning**
12. **Model comparison dashboard**
13. **3D sign avatar**
14. **Bidirectional Speech ↔ Sign**
15. **User correction feedback**
16. **Calibration visualization**
17. **Robustness report**
18. **Model/data cards**
19. **Accessibility mode**
20. **Reduced motion**

------------------------------------------------------------------------

# 70. Git/Repository Yapısı

``` text
digitra/
├── apps/
│   ├── web/
│   └── api/
├── ml/
│   ├── notebooks/
│   ├── data_pipeline/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   ├── export/
│   └── configs/
├── models/
├── packages/
├── assets/
│   └── avatar/
├── docs/
├── tests/
├── scripts/
├── .env.example
├── .gitignore
├── README.md
└── CHANGELOG.md
```

------------------------------------------------------------------------

# 71. Colab Notebook Hücre Standardı

Her training notebook:

``` text
CELL 01  Project Config
CELL 02  Environment
CELL 03  GPU
CELL 04  Seeds
CELL 05  Dataset Download
CELL 06  Dataset Discovery
CELL 07  Manifest
CELL 08  Quality Audit
CELL 09  Duplicate Audit
CELL 10  Group/Signer Audit
CELL 11  Split
CELL 12  Leakage Assertions
CELL 13  Distribution
CELL 14  Preprocessing
CELL 15  Augmentation
CELL 16  Model
CELL 17  Training
CELL 18  Curves
CELL 19  Validation Analysis
CELL 20  Controlled Experiments
CELL 21  Best Checkpoint
CELL 22  Final Test
CELL 23  Classification Report
CELL 24  Confusion Matrix
CELL 25  Error Analysis
CELL 26  Calibration
CELL 27  Robustness
CELL 28  External Test
CELL 29  Latency
CELL 30  Export
CELL 31  Export Validation
CELL 32  Model Card
CELL 33  Final Report
```

------------------------------------------------------------------------

# 72. Geliştirme Fazları

## Faz 0 --- Proje ve Veri Denetimi

-   repo
-   Colab
-   dataset kaynakları
-   lisans
-   manifest
-   leakage planı

## Faz 1 --- CNN Baseline

-   Dataset A
-   clean split
-   CNN
-   evaluation
-   export

## Faz 2 --- Transfer Learning

-   EfficientNet/MobileNet
-   frozen training
-   fine tuning
-   external test

## Faz 3 --- Landmark

-   Dataset D + Dataset A'dan landmark extraction
-   MLP/SVM/RF
-   latency

## Faz 4 --- Hybrid

-   image + landmark
-   ablation

## Faz 5 --- Ensemble/Gating

-   validation predictions
-   weighted fusion
-   OOF stacking gerekiyorsa
-   final comparison

## Faz 6 --- Real-time

-   webcam
-   MediaPipe
-   quality gate
-   smoothing
-   buffer

## Faz 7 --- Speech

-   STT
-   TTS

## Faz 8 --- Dynamic Signs

-   Dataset E
-   signer holdout
-   temporal model

## Faz 9 --- Avatar

-   rig
-   animation library
-   text/gloss → clips

## Faz 10 --- Web Product

-   premium UI
-   dashboard
-   settings
-   research mode

## Faz 11 --- Production

-   browser inference
-   API fallback
-   monitoring
-   security
-   privacy

## Faz 12 --- TİD

-   verified TİD data
-   expert validation
-   TİD-specific models
-   TİD avatar grammar

------------------------------------------------------------------------

# 73. Milestone Acceptance Criteria

## ML Foundation

-   [ ] Dataset source documented
-   [ ] License documented
-   [ ] Manifest generated
-   [ ] Corrupt files handled
-   [ ] Exact duplicates checked
-   [ ] Near duplicates checked
-   [ ] Signer/session leakage checked
-   [ ] Split manifests saved
-   [ ] Split overlap test passes
-   [ ] Train-only augmentation
-   [ ] Train-only fitted preprocessing
-   [ ] Class balance reported
-   [ ] Reproducible seed
-   [ ] Best checkpoint saved
-   [ ] Macro F1 reported
-   [ ] Confusion matrix generated
-   [ ] Error analysis completed
-   [ ] Calibration analyzed
-   [ ] External test completed
-   [ ] Model exported
-   [ ] Export validated

## Real-time

-   [ ] Camera permissions
-   [ ] Hand detection
-   [ ] No-hand state
-   [ ] Low-light state
-   [ ] Prediction
-   [ ] Confidence
-   [ ] Temporal smoothing
-   [ ] Repeated-letter logic
-   [ ] Text buffer
-   [ ] Latency benchmark

## Product

-   [ ] Light/Dark/System
-   [ ] Responsive
-   [ ] Accessibility
-   [ ] Reduced motion
-   [ ] 3D hero
-   [ ] Live demo
-   [ ] Dashboard
-   [ ] Model comparison
-   [ ] Research page
-   [ ] Privacy notice
-   [ ] Security headers
-   [ ] Error/loading states
-   [ ] Tests
-   [ ] Documentation

------------------------------------------------------------------------

# 74. Final Model Comparison Tablosu

Gerçek eğitim sonunda doldur:

  -------------------------------------------------------------------------------------------------
  Model            Internal   Macro   External   Robustness     ECE  CPU ms  GPU ms     FPS Size MB
                        Acc      F1         F1                                              
  -------------- ---------- ------- ---------- ------------ ------- ------- ------- ------- -------
  Custom CNN            TBD     TBD        TBD          TBD     TBD     TBD     TBD     TBD     TBD

  EfficientNet          TBD     TBD        TBD          TBD     TBD     TBD     TBD     TBD     TBD

  Landmark              TBD     TBD        TBD          TBD     TBD     TBD     TBD     TBD     TBD

  Hybrid                TBD     TBD        TBD          TBD     TBD     TBD     TBD     TBD     TBD

  Ensemble              TBD     TBD        TBD          TBD     TBD     TBD     TBD     TBD     TBD
  -------------------------------------------------------------------------------------------------

`TBD` değerleri gerçek deneyler olmadan doldurulmaz.

------------------------------------------------------------------------

# 75. Claude Code / AI Coding Agent Ana Talimatı

Bu dosyayı kullanan coding agent:

1.  Önce dosyanın tamamını oku.
2.  Hemen bütün projeyi kodlamaya başlama.
3.  Mevcut repo/dosya yapısını incele.
4.  Bir milestone seç.
5.  Planı yaz.
6.  Gerekli dependency'leri doğrula.
7.  Dataset pathlerini varsayma.
8.  Gerçek dosya yapısını keşfet.
9.  Leakage assertion'larını training'den önce çalıştır.
10. Her hücre/işlem sonrası sonucu doğrula.
11. Hata varsa önce root cause bul.
12. Çalışan environment'ı gereksiz package upgrade ile bozma.
13. Test etmeden "tamamlandı" deme.
14. Metrik uydurma.
15. Test setini tuning için kullanma.
16. Her milestone sonunda rapor ver.

Milestone raporu:

``` text
Completed:
Tests:
Artifacts:
Metrics:
Known Issues:
Next:
```

------------------------------------------------------------------------

# 76. İlk Yapılacak İş --- Kesin Sıra

Digitra'ya bugün başlanıyorsa:

``` text
1. Colab aç
2. GPU kontrol
3. KaggleHub kur
4. Dataset A'yı uzaktan çek
5. Dataset manifest oluştur
6. Görüntüleri audit et
7. Exact duplicate
8. Near duplicate
9. Signer/session bilgisi araştır
10. Leakage-safe split
11. Split manifestlerini kaydet
12. Custom CNN baseline
13. Validation
14. Kontrollü tuning
15. Final internal test
16. Dataset B external test
17. Export
18. EfficientNet
19. Landmark
20. Hybrid
21. Ensemble/gating
22. Hepsini aynı protokolde karşılaştır
23. Production model/modları seç
24. Kamera pipeline
25. Temporal smoothing
26. Text buffer
27. TTS/STT
28. 3D avatar
29. Dynamic sign model
30. Web deployment
```

------------------------------------------------------------------------

# 77. Nihai Digitra Mimarisi

``` text
                         ┌───────────────────┐
                         │     USER INPUT    │
                         └─────────┬─────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    │                             │
                  CAMERA                       SPEECH/TEXT
                    │                             │
             Quality Gate                        │
                    │                             ↓
             Hand Detection                  STT / NLP
                    │                             │
       ┌────────────┴────────────┐                │
       │                         │                ↓
 RGB / ROI                  Landmarks        Sign Mapping
       │                         │                │
       ↓                         ↓                ↓
EfficientNet/CNN             MLP Model        3D Avatar
       │                         │
       └────────────┬────────────┘
                    ↓
            Hybrid / Ensemble
                    ↓
               Calibration
                    ↓
            Temporal Smoothing
                    ↓
             Stable Character
                    ↓
                Text Buffer
                    ↓
              Word/Sentence
                    ↓
              ┌─────┴─────┐
              ↓           ↓
            TEXT          TTS
                           ↓
                         AUDIO
```

------------------------------------------------------------------------

# 78. Son İlke

Digitra'nın başarısı:

> "Model %99 accuracy aldı."

cümlesi değildir.

Başarı:

-   yeni kişilere genelleme,
-   farklı ortamlarda çalışma,
-   leakage-free değerlendirme,
-   düşük latency,
-   güvenilir confidence,
-   kullanıcı gizliliği,
-   erişilebilirlik,
-   bilimsel şeffaflık,
-   doğru dil/işaret temsili,
-   gerçek zamanlı kullanılabilirlik

birlikte sağlandığında elde edilir.

**Digitra bir demo classifier değil; görüntü, hareket, dil, ses ve 3D
insan etkileşimini tek erişilebilir iletişim sisteminde birleştiren uzun
vadeli bir platform olarak tasarlanmalıdır.**
