# DIGITRA --- AI / MODEL TEKNOLOJİ REHBERİ

## Amaç

Bu belge `DIGITRA_MASTER_REHBER.md` dosyasının AI/ML tamamlayıcısıdır.
Digitra'da hangi yapay zekânın nerede kullanılacağını, hangi datasetle
eğitileceğini, modellerin nasıl karşılaştırılıp birleştirileceğini ve
nihai gerçek-zamanlı karar motorunun nasıl kurulacağını tarif eder.

## 1. Temel Mimari

Digitra tek bir dev model kullanmayacaktır:

``` text
KAMERA
  ↓
MediaPipe Hands / Hand Detector
  ↓
┌─────────────────────┬────────────────────┐
│ RGB/ROI             │ 21 Hand Landmarks  │
↓                     ↓
Image Model            Landmark Model
CNN/EfficientNet       MLP/SVM/RF
│                     │
└──────────┬──────────┘
           ↓
Hybrid / Ensemble / Gating
           ↓
Confidence Calibration
           ↓
Temporal Smoothing
           ↓
Reject / Unknown Logic
           ↓
Final Harf / İşaret
           ↓
Harf → Kelime → Cümle
           ↓
Text → TTS → Ses
```

Ters yön:

``` text
Mikrofon → STT → Metin → Dil/İşaret Eşleme → İşaret Dizisi → 3D Avatar
```

## 2. Kullanılacak AI Bileşenleri

  Bileşen                  Görev
  ------------------------ -------------------------------------------------------
  MediaPipe Hands          El, handedness ve 21 landmark
  Custom CNN               Görüntü baseline modeli
  EfficientNetB0           Güçlü transfer-learning görüntü modeli
  MobileNetV3              Hafif/browser-mobile görüntü modeli
  Landmark MLP/SVM/RF      Landmark tabanlı hızlı sınıflandırma
  Hybrid Model             Aynı örneğin RGB + landmark özelliklerini birleştirme
  Ensemble/Gating          Bağımsız modellerin final kararını birleştirme
  GRU/LSTM/Transformer     Dinamik işaret/video sequence tanıma
  NLP/LLM                  Ham metni isteğe bağlı dil seviyesinde düzenleme
  Whisper/alternatif STT   Ses → metin
  TTS servisi              Metin → ses
  Sign Mapping Engine      Metin/gloss → doğrulanmış işaret dizisi
  3D Animation Engine      İşaret dizisi → avatar hareketi

## 3. Dataset → Model Eşleşmesi

### Dataset A --- Ana statik görüntü

https://www.kaggle.com/datasets/grassknoted/asl-alphabet

Kullan: - Custom CNN - EfficientNetB0 - MobileNetV3 - Aynı görüntülerden
MediaPipe landmark çıkarımı - Hybrid model

KaggleHub:

``` python
import kagglehub
dataset_a = kagglehub.dataset_download("grassknoted/asl-alphabet")
```

### Dataset B --- External test

https://www.kaggle.com/datasets/danrasband/asl-alphabet-test

Kullan: - Ana görüntü modelini farklı görüntüler/arka planlarda
sınamak - Training veya hyperparameter tuning için kullanma

``` python
dataset_b = kagglehub.dataset_download("danrasband/asl-alphabet-test")
```

### Dataset C --- İkinci domain adayı

https://www.kaggle.com/datasets/rupaul007/american-sign-language-alphabet-dataset

Kullan: - Kaynak, label ve duplicate analizi sonrası
external/domain-shift testi - Dataset A ile otomatik birleştirme

``` python
dataset_c = kagglehub.dataset_download(
    "rupaul007/american-sign-language-alphabet-dataset"
)
```

### Dataset D --- Landmark

https://www.kaggle.com/datasets/vignonantoine/mediapipe-processed-asl-dataset

Kullan: - Landmark model benchmarkı - MLP/SVM/Random Forest
karşılaştırması

``` python
dataset_d = kagglehub.dataset_download(
    "vignonantoine/mediapipe-processed-asl-dataset"
)
```

### Dataset E --- Dinamik işaret

https://www.kaggle.com/competitions/asl-signs/data

Kullan: - GRU/LSTM/Transformer - sequence recognition -
participant/signer holdout

`participant_id` train ve test arasında ayrılmalıdır.

## 4. Kritik Dil Kuralı

Bu kaynaklar ağırlıklı olarak ASL içindir. ASL modeli **Türk İşaret Dili
modeli** olarak sunulmayacaktır. TİD aşamasında ayrı, doğrulanmış TİD
verisi ve uzman doğrulaması gerekir.

## 5. Model 1 --- Custom CNN

Amaç: bilimsel baseline.

``` text
224×224 RGB
↓
Conv2D
↓
BatchNorm
↓
Activation
↓
Pooling
↓
Conv Blocks
↓
GlobalAveragePooling
↓
Dropout
↓
Dense/Softmax
```

Üret: - `digitra_cnn.keras` - `digitra_cnn.onnx` - `labels.json` -
`metrics.json` - `model_card.md`

Production'a otomatik seçme. Baseline olarak diğer modellerle
karşılaştır.

## 6. Model 2 --- EfficientNetB0 / MobileNetV3

### EfficientNetB0

Güçlü accuracy/size dengesi için ana aday.

### MobileNetV3

Browser/mobile ve düşük latency için ana aday.

Training:

``` text
ImageNet pretrained backbone
↓
Backbone frozen
↓
Classifier training
↓
Validation
↓
Son blokları kontrollü aç
↓
Daha düşük LR ile fine-tune
```

İkisini aynı split üzerinde benchmark et. Gerçek sonuç olmadan
hangisinin iyi olduğunu varsayma.

## 7. Model 3 --- Landmark AI

MediaPipe Hands:

``` text
21 × (x,y,z) = 63 temel feature
```

Araştırılacak ek feature'lar: - wrist-relative coordinate - scale
normalization - joint angles - fingertip distances - palm orientation

Model sırası: 1. Logistic Regression baseline 2. SVM 3. Random Forest 4.
MLP

Scaler:

``` text
fit(TRAIN)
transform(VAL)
transform(TEST)
```

Test üzerinde `fit` kesinlikle yasaktır.

Bu model FAST MODE için güçlü adaydır.

## 8. Model 4 --- Hybrid AI

**Aynı örneğin** görüntüsü ve landmarkı kullanılmalıdır:

``` text
image_001.jpg
↓
MediaPipe
↓
landmarks_001

(image_001, landmarks_001, label)
```

Yanlış:

``` text
Dataset A'daki A görüntüsü
+
Dataset D'deki rastgele A landmarkı
```

Hybrid mimari:

``` text
RGB → EfficientNet → Visual Vector ┐
                                   ├→ Concatenate → Dense → Softmax
Landmarks → MLP → Geometry Vector ┘
```

Zorunlu ablation: - image only - landmark only - image + landmark

Hybrid katkı sağlamıyorsa sırf daha karmaşık olduğu için production'a
alınmaz.

## 9. Ensemble / Model Fusion

Bağımsız modellerin probability çıktıları birleştirilebilir:

``` text
EfficientNet probabilities ┐
CNN probabilities ─────────┼→ Fusion → Final Probability
Landmark probabilities ────┘
```

İlk yöntem: - validation üzerinde belirlenen weighted average

Örnek oranlar gerçek oran değildir:

``` python
final = w1 * p_image + w2 * p_landmark
```

`w1`, `w2` test setinde seçilmez.

İleri: - confidence-aware gating - stacking - meta-classifier

Stacking'de base modelin training sample'larındaki in-sample
prediction'ları meta-model training için kullanılmaz. Out-of-fold
predictions kullanılmalıdır.

## 10. Nihai Static Karar Motoru

``` text
Frame
↓
Image Quality Gate
↓
MediaPipe Hand Detection
↓
ROI + Landmarks
↓
Image Model + Landmark Model
↓
Fusion
↓
Confidence Calibration
↓
Temporal Smoothing
↓
Stability Check
↓
Unknown/Reject Check
↓
Final Character
```

## 11. Quality Gate

Inference öncesi: - el var mı? - el yeterince büyük mü? - görüntü aşırı
karanlık mı? - blur var mı? - el crop dışında mı? - birden fazla el var
mı?

Kalite yetersizse model harf uydurmamalıdır.

Durumlar: - `NO_HAND` - `LOW_LIGHT` - `BLURRY` - `MULTIPLE_HANDS` -
`LOW_CONFIDENCE` - `UNKNOWN`

## 12. Confidence

Softmax confidence = accuracy değildir.

Ölç: - reliability diagram - Expected Calibration Error - Brier score

Gerekirse: - temperature scaling

Confidence threshold validation setinde belirlenir.

## 13. Temporal Smoothing

Tek frame doğrudan harf değildir.

``` text
A A A E A A
```

gibi bir dizide tek E filtrelenebilir.

Araştır: - majority voting - moving probability average - exponential
smoothing - stability window - hysteresis - debounce

Karakter ancak yeterli güven + kararlılık + süre sağlanınca buffer'a
eklenir.

## 14. Tekrarlanan Harfler

`LL` gibi durumlarda aynı işaretin sürekli yazılmasını engelle.

Aday: - neutral/no-hand state - cooldown - release detection - NEXT
gesture

UX testiyle karar ver.

## 15. Dynamic Sign AI

Dataset: https://www.kaggle.com/competitions/asl-signs/data

Girdi:

``` text
T × landmarks
```

Benchmark: 1. GRU 2. LSTM 3. Transformer/Temporal model

Transformer otomatik "en iyi" kabul edilmez.

Split: - `participant_id` holdout - aynı participant train/testte yok

## 16. Static / Dynamic Routing

İleri sürüm:

``` text
Input
↓
Gesture State / Mode
↓
Static → Static AI
Dynamic → Temporal AI
↓
Unified Sign Output
```

İlk sürümde kullanıcı modu manuel seçebilir.

## 17. NLP / LLM

LLM kamera classifier değildir.

Görev: - ham karakter dizisini isteğe bağlı düzeltmek - punctuation -
yazım önerisi - doğal dil katmanı

UI iki çıktıyı ayırmalıdır:

``` text
Raw Recognition
AI Suggested Text
```

LLM'nin düzelttiği metin "model doğrudan bunu gördü" diye sunulmaz.

## 18. LLM Provider Mimarisi

Hard-code etme:

``` text
LanguageProcessor
├ CloudLLMProvider
├ LocalLLMProvider
└ RuleBasedProvider
```

Basit işlemler için LLM zorunlu değildir. Dictionary/rule-based çözüm
tercih edilebilir.

## 19. Speech-to-Text

Aday: - Whisper ailesi - browser speech recognition - cloud STT

Mimari:

``` text
STTService
├ LocalWhisper
├ BrowserSTT
└ CloudSTT
```

Ölç: - Word Error Rate - latency - noisy-room performance - Türkçe
performansı

## 20. Text-to-Speech

Başlangıç: - browser TTS

İleri: - neural cloud/local TTS

``` text
TTSService
├ BrowserTTS
├ CloudTTS
└ LocalTTS
```

Kontroller: - dil - ses - hız - pitch - autoplay

## 21. Text/Speech → Sign

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
Validated Sign Sequence
↓
3D Avatar
```

Türkçe kelimeleri birebir TİD işaretlerine çevirmek doğru kabul edilmez.
TİD dilbilgisi ve uzman doğrulaması gerekir.

## 22. 3D Avatar AI

V1'de generative motion zorunlu değildir.

Daha güvenli başlangıç:

``` text
sign_id
↓
verified animation clip
↓
rigged avatar
```

İleri aşamada motion generation araştırılabilir.

Avatar: - el/parmak rig - kol - omuz - baş - yüz/mimik desteklemelidir.

## 23. Model Eğitim Kuralları

Her model: - reproducible seed - data audit - exact duplicate - near
duplicate - signer/session/video leakage check - leakage-safe
train/val/test - train-only augmentation - train-only fitted
preprocessing - class balance analysis - early stopping - checkpoint -
LR scheduling - training curves - macro F1 - confusion matrix - error
analysis - calibration - robustness - external test - latency - export
validation

içermelidir.

## 24. Model Seçimi

Tek accuracy yasak.

Öncelik: 1. external macro F1 2. internal test macro F1 3. signer/user
generalization 4. robustness 5. calibration 6. latency 7. FPS 8. model
size 9. accuracy

## 25. Fast / Balanced / Accurate

Benchmark sonrası:

### FAST

Muhtemel aday: `MediaPipe + Landmark MLP`

### BALANCED

Muhtemel aday: `MobileNetV3` veya `EfficientNetB0`

### ACCURATE

Muhtemel aday: `Hybrid` veya `Image + Landmark Ensemble`

Bunlar hedef atamalardır; gerçek benchmark sonrası kesinleşir.

## 26. Quantization / Optimization

Model doğrulandıktan sonra: - FP16 - INT8 - quantization - pruning

araştır.

Her optimizasyondan sonra yeniden: - macro F1 - accuracy - latency -
size ölç.

## 27. ONNX

Export sonrası parity testi:

``` text
Same input
↓
Original Model → P1
ONNX Model → P2
↓
Numerical difference check
```

Export edilen model test edilmeden production'a alınmaz.

## 28. Browser AI

Öncelik:

``` text
ONNX Runtime Web + WebGPU
↓
WASM fallback
↓
Server inference
```

Gerçek cihaz/browser benchmarkı zorunludur.

## 29. Reject / OOD

Model her şeyi A-Z'den biri olarak zorlamamalıdır.

Kullan: - hand detector - confidence threshold - UNKNOWN - NOTHING -
NO_HAND

İleri: - OOD detection

## 30. Handedness / Multi-Hand

Sağ ve sol el ayrı test edilir.

Horizontal flip otomatik çözüm değildir.

İki el gereken işaretler için ileri model:

``` text
Left Hand
+
Right Hand
+
Pose
↓
Temporal Model
```

## 31. Face + Pose

Tam işaret dili aşamasında yalnızca el yeterli olmayabilir.

İleri TİD/dynamic sisteminde: - hands - face - pose araştırılmalıdır.

## 32. Explainability

Image: - Grad-CAM

Landmark: - feature ablation/importance

Explainability çıktısı modelin "kesin düşüncesi" olarak sunulmaz;
araştırma/debug aracıdır.

## 33. Bias / Robustness

Test: - farklı cilt tonları - farklı ışık - farklı kamera - farklı arka
plan - farklı el boyutu - sağ/sol el - açı - mesafe - blur - occlusion

Dataset kapsamıyorsa limitation olarak yaz.

## 34. Monitoring

Production: - model version - latency - confidence distribution - reject
rate - prediction distribution - user correction rate

Raw camera frame varsayılan olarak loglanmaz.

## 35. Active Learning

``` text
Low confidence / Wrong feedback
↓
Explicit consent if image needed
↓
Review
↓
Human label
↓
Quality control
↓
New dataset version
```

Model kendi tahminini otomatik gerçek label olarak training datasına
eklemez.

## 36. Model Versioning

``` text
digitra-cnn-v1.0.0
digitra-efficientnet-v1.0.0
digitra-landmark-v1.0.0
digitra-hybrid-v1.0.0
digitra-dynamic-v1.0.0
```

## 37. Model Artifact Standardı

``` text
models/
└── digitra-hybrid-v1.0.0/
    ├── model.keras
    ├── model.onnx
    ├── labels.json
    ├── preprocessing.json
    ├── calibration.json
    ├── metrics.json
    ├── model_card.md
    └── checksum.sha256
```

Landmark model ayrıca: - `scaler.pkl` - `feature_config.json`

içerir.

## 38. Standart Inference Output

``` json
{
  "model": "digitra-hybrid",
  "version": "1.0.0",
  "prediction": "A",
  "confidence": null,
  "stable": false,
  "top_k": [],
  "latency_ms": null,
  "hand_detected": true
}
```

`null` alanları gerçek inference sırasında doldurulur; sahte değer
yazılmaz.

## 39. AI Testleri

-   valid hand
-   no hand
-   blank image
-   dark image
-   corrupted input
-   invalid shape
-   left/right hand
-   two hands
-   low confidence
-   repeated character
-   ONNX parity
-   preprocessing parity
-   split integrity

## 40. AI Sağlayıcı Bağımsızlığı

Interface'ler:

``` text
HandDetector
StaticVisionClassifier
LandmarkClassifier
TemporalSignClassifier
FusionEngine
SpeechRecognizer
SpeechSynthesizer
LanguageProcessor
SignMapper
AvatarController
```

Tek provider'a bağımlı kod yazma.

## 41. Kesin Geliştirme Sırası

``` text
01 Dataset Audit
02 Custom CNN
03 EfficientNet/MobileNet
04 Landmark Model
05 Hybrid
06 Ablation
07 Ensemble/Gating
08 Calibration
09 Robustness
10 External Test
11 ONNX
12 Browser Benchmark
13 Real-time Camera
14 Temporal Smoothing
15 Text Buffer
16 STT
17 TTS
18 Dynamic Sign Model
19 NLP
20 Sign Mapping
21 3D Avatar
22 TİD-specific AI
```

## 42. İlk Production AI

İlk gerçek kullanılabilir Digitra sürümünün hedef mimarisi:

``` text
MediaPipe Hands
+
EfficientNet/MobileNet
+
Landmark MLP
+
Validation-based Fusion
+
Confidence Calibration
+
Temporal Smoothing
+
Reject Logic
```

Dynamic sign, LLM ve avatar ilk statik harf modelinin bitmesini
engellememelidir.

## 43. TİD Fazı

1.  Güvenilir TİD sınıfları belirle.
2.  İşaret dili uzmanı ile doğrula.
3.  Farklı signer'lardan veri topla.
4.  Signer-independent test oluştur.
5.  Static/dynamic ayrımı yap.
6.  Hands + pose + face gereksinimini değerlendir.
7.  Türkçe → TİD mapping'i uzmanla geliştir.
8.  Avatar hareketlerini uzmanla doğrula.
9.  ASL ve TİD metriklerini tamamen ayrı tut.
10. Model card'da kapsam ve sınırlamaları açıkla.

## 44. Yapılmaması Gerekenler

-   LLM'yi görüntü classifier yerine kullanmak
-   Dataset A görüntüsü ile Dataset D'den rastgele landmarkı hybrid
    sample yapmak
-   test üzerinde ensemble weight seçmek
-   test üzerinde confidence threshold seçmek
-   stacking'de in-sample prediction leakage
-   softmax confidence'ı accuracy sanmak
-   her görüntüyü zorla harfe çevirmek
-   MediaPipe'ı tek başına Digitra classifier sanmak
-   "Transformer daha yeni, o halde daha iyi" varsaymak
-   quantization sonrası metriği tekrar ölçmemek
-   ONNX export'u test etmeden kullanmak
-   ASL modelini TİD modeli olarak sunmak
-   TİD çevirisini uzman doğrulaması olmadan kesin doğru diye göstermek

## 45. Son İlke

Digitra AI geliştirme felsefesi:

``` text
Doğru Veri
↓
Leakage-Free Split
↓
Basit Baseline
↓
Transfer Learning
↓
Geometrik Landmark Model
↓
Hybrid
↓
Ensemble
↓
Calibration
↓
Temporal Stability
↓
Robustness
↓
Real-Time Deployment
↓
Dynamic Signs
↓
Speech + Language + Avatar
```

**En karmaşık model değil; yeni kullanıcıda en güvenilir, en hızlı, en
iyi kalibre edilmiş ve bilimsel olarak en savunulabilir model
production'a alınacaktır.**
