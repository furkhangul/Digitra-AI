# Digitra TİD el modeli

El geometrisi MIT lisanslı WebXR Input Profiles `generic-hand` modelinden
uyarlandı. Hazır el yüzeyi ve eklem ağırlıkları, mevcut TİD iskeletine taşındı;
parmaklar hafif dolgunlaştırıldı ve yüzey iki kademe yumuşatıldı. Avuç ve parmak
araları artık kaynak modelin anatomik yüzey bağlantılarını kullanır.
Kullanıcının fildişi ana el ve koyu mor yardımcı el renkleri korunur.
Bu model fingerspelling.xyz'nin birebir modeli değildir.
Önceki düzenlenmiş model `output/model-backups/before-webxr` klasöründe saklanır.

## Açılacak dosyalar

- `digitra-tid-hands.blend`: İki bağımsız iskelet, 58 düzenlenebilir harf pozu ve 2.240 karelik ortak alfabe zaman çizelgesi. Blender 5.2 ile hazırlandı. Zaman çizelgesinde Space tuşuyla oynatın; harf işaretçileri konumları gösterir. Tek harfi düzenlemek için ilgili elin Action Editor bölümünde `TID_XX_HARF_right/left` eylemini seçin. `TID_Alphabet_right/left` tam zaman çizelgesidir.
- `tid-hand.glb`: Web uygulamasında kullanılan uyarlanmış tek el ve iskeleti (27.770 köşe, yaklaşık 1,83 MB). İkinci el, ayrı iskelet kopyasıyla aynalanır. Harfler ve geçişler web kodunda yönetilir.
- `tid-alphabet-animated.glb`: İki eli ve oynatılabilir alfabe animasyonunu birlikte içeren bağımsız GLB. Blender animasyonu 24 FPS; GLB örneklemesi iki karede bir yapılır ve arada interpolasyon uygulanır.
- `tid-pose-review.png`: 29 harfin Blender görsel kontrol tablosu. Animasyonları değerlendirmek için canlı önizlemeyi veya Blender zaman çizelgesini kullanın.
- `motion-test-report.json`: Geçiş motorunun sayısal test sonuçları.

## Web önizlemesi

Geliştirme sunucusu çalışırken `http://127.0.0.1:3000/debug-tid` doğrudan eğitim ekranını açar. Ana sayfanın `#egitim` bölümü de aynı yeni bileşeni kullanır. Üstteki dekoratif el değiştirilmedi.

29 harf, iki el koordinasyonu, bağımsız üç parmak eklemi, başparmak karşılaması, bilek yönleri, kesilen geçişe mevcut pozdan devam etme, duraklatma, tekrar, 0.5×/1×/1.5× hız, önden/çapraz görünüş, sürükleyerek döndürme ve sınav arayüzü bulunur.

## Önceki poz düzenlemeleri (korundu)

Harf biçimleri ve iki elin ilişkisi kaynaktaki gibi bırakıldı; değişen yalnızca
elin geometrisi ve kameraya göre duruş/aralıktır.

- **G, Ğ**: İki kapalı el üst üste dururken tek bir yumruya karışıyordu. Eller dikleştirildi ve düşey aralık, alttaki elin üstüne oturacak biçimde yeniden hesaplandı; üstteki elin kıvrık işaret parmağı artık siluette görünür.
- **A**: Yatay işaret parmağı, aşağı uzanan iki parmağın önünden geçiyordu. İki parmak daha çok açıldı ve yatay parmak aynı derinlikte, parmakların ortasına indirildi; "aralarına" ilişkisi görünür oldu.
- **B, Ö**: İki sıkıştırma halkası kameraya yanlamasına bakıyordu. Y dönüşleri 65°'den 30°'ye indirildi ve dikey aralık açıldı; her iki halka da görünür.
- **J**: Önden İ harfinden ayırt edilemiyordu. Kıvrık el 60° çevrildi; kanca biçimi artık profilden okunur.
- **R**: İki kanca tek bir düğüm gibi görünüyordu. Eller 138°/42° ile karşılıklı kilitlendi ve derinlik aralığı artırıldı.
- **E**: Alttaki elin işaret parmağı, üstteki elin avucunun arkasında kalıyordu. Üç parmaklı el arkaya alındı.
- **Ç**: İki el birbirine yapışıyordu. Dönüşler yumuşatıldı ve düşey aralık .25'ten .36'ya çıkarıldı.
- **C, O, P, M, N** ve diğerleri: Ayrı bir poz değişikliği gerekmedi; yeni el geometrisi ve renk ayrımıyla okunur hâle geldi.

## Referans ve doğrulama sınırı

Pozlar Dikyuva, Makaroğlu ve Arık'ın (2015) *Türk İşaret Dili Dilbilgisi Kitabı*, basılı sayfa 91'deki yaygın 29 harflik el abecesinden yeniden modellenmiştir. Kaynak görselleri modele gömülmemiştir. El yüzeyi WebXR Input Profiles generic-hand varlığından uyarlanmıştır; kaynak ve MIT lisansı `THIRD_PARTY_HAND_LICENSE.md` dosyasında bulunur. fingerspelling.xyz varlıkları kullanılmamıştır.

Kaynak: https://linguistics.ankara.edu.tr/wp-content/uploads/sites/1078/2021/05/Dikyuva_H._Makaroglu_B._and_Arik_E._2015_compressed.pdf#page=92

Bu teslim, uzman tarafından onaylanmış TİD öğretim materyali değildir. Çizimlerden oluşturulmuş 3B önizlemedir; özellikle temas konumları, el yönleri, bölgesel değişkeler ve hareket zamanlaması TİD uzmanı incelemesini bekler. Statik çizimler bütün hareket ayrıntılarını vermez. Otomatik testler dilsel doğruluğu veya bütün geçişlerde yüzeylerin hiçbir noktada kesişmediğini kanıtlamaz.

Webdeki kamera sınıflandırıcısı bu değişiklikle yeniden eğitilmedi; mevcut ASL prototipi olarak kalır. Eğitimdeki TİD gösterimi bu sınıflandırıcıdan bağımsızdır.

## Yapılan kontroller

- 29 × 29 = 841 sıralı geçiş, 151.380 karede sonlu değerler, normalize dönüşler ve konum adımı kontrolü.
- Duraklatma, hızlı seçimle animasyonu kesme, hız değişimi, arka plandan dönme ve hareket döngülerinin uç noktaları.
- Yeni elin dinlenme, yarım kıvrım ve yumruk pozlarında önden, arkadan, yandan ve çaprazdan Blender kontrol renderları.
- 29 harflik Blender render tablosunun yeniden üretimi ve harf harf görsel karşılaştırma.
- `digitra-tid-hands.blend` içeriğinin doğrulanması: 21 kemik, 16 vertex grubu, 58 harf eylemi, 2 alfabe eylemi, 30 zaman çizelgesi işaretçisi ve 2.240 karelik oynatım.
- Tarayıcıda `/debug-tid` üzerinde yeni GLB'nin yüklenmesi ve harf gösterimi.
- TypeScript, değişen bileşenlerde ESLint ve Next.js üretim derlemesi.

Not: Tarayıcı sekmesi arka plandayken `requestAnimationFrame` kısıtlandığı için
geçiş animasyonu donmuş görünebilir; sekme öne alındığında normal ilerler.

## Yeniden üretme

Proje kökünden:

1. Blender arka plan modunda `apps/web/scripts/build-tid-webxr.py` dosyasını çalıştırın. Önceki prosedürel `build-tid-hand.py` kullanıcı düzenlemeleriyle arşiv olarak korunur.
2. `node apps/web/scripts/test-tid.cjs --dump --timeline` çalıştırın.
3. Blender ile `apps/web/scripts/render-tid-review.py` çalıştırın.
4. Sistem Python'u ile `python apps/web/scripts/assemble-tid-review.py` çalıştırın (Pillow gerekir; Blender'ın Python'unda bulunmaz).
5. Blender ile `apps/web/scripts/bake-tid-timeline.py` çalıştırın.

İlk adım oluşturulan Blender dosyasını yeniden yazar. Elle yapılan model değişiklikleri varsa önce ayrı isimle kaydedin.
