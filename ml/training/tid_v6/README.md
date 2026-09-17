# Digitra TİD V6 araştırma eğitimi

Bu çalışma, mevcut web sitesini ve eğitim bölümündeki el geometrilerini değiştirmeden yeni bir **görüntüden harf tanıma modeli** üretir. Ana giriş noktası `train.py` dosyasıdır. Mevcut canlı V5 modeli otomatik olarak değiştirilmez.

## Veriler ve hedef

- 29 harf; 1.969 gerçek eğitim, 574 doğrulama, 431 tarihsel test görüntüsü.
- Sitedeki mevcut el pozlarından 5 küçük bakış açısı ve 3 ten/ışık varyasyonu ile 435 görüntü.
- Statik pozlara ait 330 görüntü, eğitimde 0,20 kayıp ağırlığıyla yardımcı veri olarak kullanılır. Hareketli harflerin 105 görüntüsü statik sınıflandırma eğitimine alınmaz.
- Ölçülen V6 hedefi: temiz gerçek eğitimde %99,75, doğrulamada %91,99 ve tarihsel testte %90,26. Bunlar bu veri bölümü için ölçümlerdir; yeni kişi/ortam doğruluğu garanti etmez.

Test, V5 için daha önce raporlanan test grubudur; yeni dış test değildir. Kişi kimlikleri bulunmadığından kişi bağımsız genelleme sonucu değildir. Görüntülerden harf tanınır; işaret dili cümle çevirisi yapılmaz.

## Eğitim özellikleri

1. ImageNet başlangıç ağırlıkları; V5'in TİD ağırlıklarından devam edilmez.
2. Eğitim öncesi dosya bütünlüğü, SHA-256 ve pHash/yakın kopya denetimi.
3. Ardışık 10 karelik mevcut grupları koruyan veri ayrımı.
4. Yalnızca eğitim verisine uygulanan, kademeli güçlenen augmentation.
5. Tam el çiftini aynalama, sınırlı dönüş, kaydırma, ölçek, eğme ve perspektif.
6. Parlaklık, kontrast, doygunluk, renk tonu, gamma ve beyaz dengesi.
7. Gamma/pozlama, yüksek ve düşük ışık, renk sıcaklığı, beyaz dengesi, arkadan ışık, yumuşak gölge, sensör gürültüsü, JPEG bozulması, düşük çözünürlük, hareket ve Gaussian bulanıklığı.
8. Seyrek gri tonlama ve küçük alan kapatma; parmakların büyük bölümünü silen agresif dönüşümler uygulanmaz.
9. Sentetik veride etiketlerden bağımsız arka plan ve ten/ışık değişimi.
10. İlk iki epoch omurgayı dondurma, sonra düşük öğrenme hızıyla ince ayar.
11. AdamW, cosine öğrenme hızı, karma hassasiyet, gradyan sınırlandırma.
12. Etiket yumuşatma, gerçek eğitim sınıf sayılarına göre kayıp ağırlıkları.
13. EMA ve normal ağırlıklar arasında yalnızca doğrulama macro F1 ile seçim.
14. Erken durdurma, atomik en iyi/son kayıtlar ve yeniden devam etme desteği.
15. TTA, sıcaklık kalibrasyonu ve güven eşiğini yalnızca doğrulamada seçme.
16. Seçim kaydı kilitlendikten sonra son test; temiz görüntü, ışık, konum, dönüş, bulanıklık ve sıkıştırma dahil koşullarda dayanıklılık ölçümü.
17. Eğitim grafikleri, karışıklık matrisi, harf bazlı F1, hatalı örnekler ve doğruluk güven aralığı.
18. Kaydedilen modelin mevcut API yükleyicisiyle uyum ve gecikme kontrolü.

Augmentation tek başına yeni kişi/ortam verisinin yerini tutmaz. Aynalama, sağ/sol el çiftinin birlikte yansıtılmasıdır; tek el bağımsız çevrilmez. 3D el uyarlamalarının dil uzmanı doğrulaması tamamlanmadığı için yardımcı veri düşük ağırlıklıdır.

## Çalıştırma (proje kökünden PowerShell)

```powershell
node ml/training/tid_v6/export_hands.cjs
& 'Digitra/digitra-tid-v4/.venv/Scripts/python.exe' -X utf8 ml/training/tid_v6/render_hands.py --source artifacts/tid-v6/hands.json --output artifacts/tid-v6/synthetic
& 'Digitra/digitra-tid-v4/.venv/Scripts/python.exe' -X utf8 ml/training/tid_v6/audit.py --manifest 'Digitra/digitra-tid-v4/artifacts/manifest.csv' --output artifacts/tid-v6/data
& 'Digitra/digitra-tid-v4/.venv/Scripts/python.exe' -m pytest ml/training/tid_v6/test_training.py -q
& 'Digitra/digitra-tid-v4/.venv/Scripts/python.exe' -X utf8 -u ml/training/tid_v6/train.py --output artifacts/tid-v6/runs/efficientnet-seed42 --epochs 30 --batch-size 48 --workers 2
```

Kesilen eğitim aynı parametrelerle `--resume` eklenerek son tamamlanan epoch'tan devam eder. `--finalize-only`, doğrulama ile seçilmiş `best.pt` dosyasını yeni epoch çalıştırmadan test eder. Teste açılmış bir çalışma tekrar eğitim için kullanılamaz. Yeni deney ayrı çıktı klasörü gerektirir.

## Çıktılar

`artifacts/tid-v6/runs/efficientnet-seed42/` altında:

- `status.json`: mevcut aşama, epoch ve ilerleme.
- `history.json`: her epoch'un gerçek eğitim kaybı ve doğrulama sonuçları.
- `config.json`: veri kimliği, parametreler, cihaz ve özellik listesi.
- `best.pt`, `last.pt`: seçilen model ve devam kaydı.
- `model.pt`, `robust_cnn_ensemble.json`: API yükleyicisiyle uyumlu araştırma adayı.
- `selection.json`, `test_started.json`: testten önce kilitlenen seçim.
- `evaluation.json`, `per_class.json`, `test_predictions.csv`: ölçülen sonuçlar.
- `training_curves.png`, `confusion_matrix.png`: grafikler.

Gerçek veri lisansı CC BY-NC-SA 4.0'dır. Çıktılar araştırma/ticari olmayan kullanım içindir.

Tamamlanan ve değerlendirmesi biten çalışmayı, dosya kimlikleri ve model kartıyla paketlemek için:

```powershell
& 'Digitra/digitra-tid-v4/.venv/Scripts/python.exe' -X utf8 ml/training/tid_v6/package_model.py --run artifacts/tid-v6/runs/efficientnet-seed42 --output models/digitra-tid-augmented-v6
```

Paketleme, mevcut bir model klasörünün üzerine yazmaz. `model_card.md` gerçek ölçümleri, `manifest.json` paket dosyalarının SHA-256 kimliklerini içerir.

Teknik başvuru: [Torchvision dönüşümleri](https://docs.pytorch.org/vision/stable/transforms.html), [PyTorch AveragedModel](https://docs.pytorch.org/docs/stable/generated/torch.optim.swa_utils.AveragedModel.html).

## Yerel hareket testi

`/test-v6` ekranındaki **Hareketli harfler · deneme** seçeneği, görüntü modelinden
ayrı bir akış kullanır. Tarayıcı her algılama karesinin el noktalarını zaman
damgasıyla paketler; görüntü tahmininin daha yavaş aralığı hareketleri düşürmez.
İki el görünürken 2,6 saniyelik geçmiş 30 örneğe yeniden örneklenir. Sabit duruş,
küçük titreme ve tüm el çiftinin birlikte taşınması tek başına yeterli değildir.

- Ç, Ğ, İ, Ö, Ş, Ü: mevcut `digitra-landmark-personal-v1.0.0` dizi modeli;
  0,70 güven, 0,18 sınıf farkı ve üç ardışık karar eşiği. Tek kişinin tek
  oturumundan eğitilmiştir; açık küme yanlış kabul oranı bilinmiyor.
- J: sitedeki iki elli L boyunca parmak izi için sıralı geometrik kontrol.
  Eğitilmiş J modeli değildir; olasılık yüzdesi gösterilmez. Ayna, ölçek ve
  el sırası değişimleri desteklenir. Örtüşme ve kenardan bakışta reddedebilir.
- Tamamlanan hareket metin ekleme süresi için kısa süre tutulur. El kaybı,
  uzun kare arası boşluk ve mod değişimi geçmişi temizler. Metin mod değişiminde korunur.
- Sabit görüntü modu bu yedi harfi hareket tamamlanmış gibi otomatik eklemez.

Hareket modunun gerçek kamera doğruluğu **ölçülmedi**. V6'nın %90,25 statik
test sonucu bu moda ait değildir. J ve farklı kişiler için güvenilir bir
eğitilmiş model, etiketli gerçek hareket dizileri ve ayrı oturum testi gerektirir.
Kamera görüntüsü veya el noktaları diske kaydedilmez. 3D el kaynakları değiştirilmez.

```powershell
node ml/training/tid_v6/export_motion_fixtures.cjs
& 'apps/api/.venv-digitra/Scripts/python.exe' -X utf8 -m pytest ml/training/tid_v6/test_motion_recognition.py ml/training/tid_v6/test_live_acceptance.py -q
& 'apps/api/.venv-digitra/Scripts/python.exe' -X utf8 -m uvicorn live_api:app --app-dir ml/training/tid_v6 --host 127.0.0.1 --port 8006 --ws-max-size 1000000
```

Üretilen animasyon dizileri yalnızca yazılım testidir; gerçek kamera verisi veya
eğitim/test doğruluğu olarak kullanılmaz.

## 15 Eylül: Ğ / N canlı tanıma düzeltmesi

- Çeviri, canlı eğitim ve `/test-v6` aynı `NEXT_PUBLIC_DIGITRA_V6_API_URL`
  adresini kullanır (varsayılan `127.0.0.1:8006`). Başlatma betiği V6'yı da açar.
- N için iki eldeki 2+1 açık parmak, aşağı yön ve uç yakınlığı denetlenir;
  en az üç kare ve 450 ms kararlılık gerekir. A, M ve Y'nin farklı yön/parmak
  düzenleri ayrıca reddedilir. Sonuç `pose_rule` olarak etiketlenir, güven
  yüzdesi üretilmez. Kişiye/ortama göre başarısı henüz ölçülmemiştir.
- Ğ, G el şekli korunurken yerel parmak hareketinin eksenini ve anlamlı yön
  değişimlerini izler. Eski dar hareket büyüklüğü aralığı genişletilmiştir.
  Hareket tamamlanmadan G'nin erken eklenmesi bekletilir; aynı bırakılmamış
  işaretin G tabanı metne eklenmişse tamamlanan hareket onu Ğ olarak düzeltir.
- Kısa el örtüşmelerinde yalnızca takip kimliği korunur; eksik karelere el
  noktası uydurulmaz. Eğitimde aday harf, görünen el sayısı ve harekete özel
  yönlendirme görünür.
- Kontrol: 35 Python davranış testi, 8 tarayıcı el filtresi testi; ayrıca
  `smoke_live_api.py` çalışan WebSocket üzerinden üretilmiş N/Ğ dizilerini
  doğrular. Bunlar gerçek kullanıcı doğruluğu değildir. Model ağırlıkları ve
  eski eğitim/test ölçümleri değiştirilmemiştir.

```powershell
& 'apps/api/.venv-digitra/Scripts/python.exe' -X utf8 -m pytest ml/training/tid_v6/test_motion_recognition.py ml/training/tid_v6/test_live_acceptance.py ml/training/tid_v6/test_g_n_recognition.py -q
& 'apps/api/.venv-digitra/Scripts/python.exe' -X utf8 ml/training/tid_v6/smoke_live_api.py
```
