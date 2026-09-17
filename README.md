# Digitra

Uçtan uca parmak alfabesi / işaret dili tanıma, ses ve 3D avatar platformu.
Proje kuralları için bkz. `DIGITRA_MASTER_REHBER.md` ve `DIGITRA_AI_MODEL_REHBERI.md`.

## Repo Yapısı

```
apps/
  web/   Next.js + TypeScript + Tailwind + Framer Motion + React Three Fiber
  api/   FastAPI backend iskeleti (health, models, predict, stt, tts, feedback)
ml/
  notebooks/   Colab/Jupyter eğitim defterleri (01_digitra_data_audit.ipynb ile başlar)
  ...          data_pipeline, training, evaluation, inference, export, configs
models/        Eğitilmiş model artefaktları (bkz. MASTER_REHBER #37/#50)
assets/avatar/ 3D avatar rig ve animasyon dosyaları
docs/          Model/veri kartları, ek dokümantasyon
```

## Web'i çalıştırma

```bash
cd apps/web
npm install
npm run dev   # http://localhost:3000
```

Kurulum tamamlandıktan sonra Windows'ta kökteki `run_digitra_ai.cmd`
dosyasına çift tıklayarak API, model ve web'i birlikte açın. Başlatıcı servisleri
hazır olana kadar bekler; bir hata oluşursa ayrıntıyı ekranda ve
`output/runtime-logs/` altında bırakır.

## API'yi çalıştırma

```bash
cd apps/api
python -m venv .venv-digitra
.venv-digitra/Scripts/python -m pip install -r requirements.txt
.venv-digitra/Scripts/python -m pip install --force-reinstall --no-deps -r requirements-cuda.txt
cp .env.example .env
.venv-digitra/Scripts/python -m uvicorn app.main:app --port 8001 --reload
# http://localhost:8001
```

## Canlı TİD Robust V5

- **Web:** `#ceviri` paneli MediaPipe ile bir veya iki eli bulur, ortak kareye
  kırpar ve ilk üç aday, güven, kararlılık, gecikme, FPS ile metin tamponunu gösterir.
- **API:** `/ws/recognize` her tarayıcı için ayrı smoothing/text oturumu oluşturur.
  `/predict` endpoint'i base64 JPEG/PNG/WEBP el kırpımı kabul eder; görüntü yoksa
  eski landmark istemcileri için geriye uyumlu tahmin yolu çalışır.
- **Model:** %25 MobileNetV3 Large + %75 EfficientNet-B0; çok ölçekli ve aynalı
  TTA ile 29 harflik TİD parmak alfabesi.
- **Ölçüm:** Kilitli test doğruluğu %89,10, macro F1 %88,98; kişi kimliği
  bulunmadığından bu sonuç signer-independent değildir.
- **Gizlilik:** Tam kamera karesi gönderilmez. Yalnızca el çevresindeki 320×320
  JPEG kırpımı yerel API'de bellekte işlenir ve kaydedilmez.
- **Kullanım:** Canlı panelde önce “Model bağlantısı açık” durumunu kontrol edin,
  kamerayı başlatın ve harfi 1–2 saniye sabit tutun. MediaPipe örtüşen iki elden
  yalnızca birini bulsa bile kırpıcı ikinci elin bağlamını korur.
- **Fallback:** Görüntü göndermeyen eski istemcilerde bir el için ASL landmark,
  iki el özel harfler için kişisel temporal model korunmuştur.
- Model kartı ve gerçek ölçüm kapsamları
  `models/digitra-tid-robust-v5/` altında tutulur.
