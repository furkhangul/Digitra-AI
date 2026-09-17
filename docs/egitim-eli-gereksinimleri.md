# Digitra eğitim eli: inceleme ve üretim şartnamesi

Tarih: 7 Eylül 2026. Kapsam: sayfanın eğitim bölümündeki el ve harf animasyonları. Bu belge bir üretim planıdır; yeni model veya çalışan animasyon teslimi değildir.

## İncelemede doğrulanan sorunlar

- `apps/web/src/components/home/egitim.tsx`, eğitim için `HeroScene` kullanıyor. Bu sahne `hand.glb` modelini `HandModel` üzerinden yönetiyor; `simplehand.glb` mevcut eğitim yolunda kullanılmıyor.
- `hand.glb` 21 eklem ve üç animasyon içeriyor: `Do_HandRiggedAction`, `Pose_OK`, `Pose_OKHand`. Tam alfabe animasyon paketi bulunmuyor. Kemik sayısı tek başına deformasyon kalitesini kanıtlamaz.
- `hand-poses.ts` her parmağı tek kapanma değeriyle temsil ediyor. `hand-model.ts` bu değeri bütün parmak eklemlerine sabit katsayılarla uyguluyor. Başparmağın avuç içinde karşıya uzanması ve her eklemin bağımsız pozlanması yeterince temsil edilmiyor.
- Seçili harf değiştiğinde yeni değer doğrudan uygulanıyor; kullanıcı seçimi için mevcut pozdan hedef poza zaman tabanlı geçiş bulunmuyor. Üst bölümdeki otomatik döngünün geçişleri bu yolu çözmüyor.
- G/Ğ/H yönü 3B bilek pozu yerine HTML kapsayıcısının döndürülmesiyle taklit ediliyor.
- J/Z yolları kodda açıkça yaklaşık hareketler olarak tanımlanmış. Ç/İ/Ö/Ü açıklamalarındaki hareketler için eğitim sahnesine ayrı animasyon verilmiyor.
- M, N, S, Ğ, İ, J, Ö ve Ü sınav havuzundan çıkarılmış. Kod gerekçesi, bazı şekillerin ayırt edilememesi ve tekrarlar.
- Alfabe kodu, pozların yaklaşık olduğunu ve TİD referansı olmadığını belirtiyor. Harf biçimlerinin doğruluğu model kalitesinden ayrı doğrulanmalı.

Yerel kaynak kodu ve GLB metaverisi incelendi. Tarayıcı kontrolü zaman aşımına uğradığı için bu turda canlı animasyonların görsel değerlendirmesi tamamlanamadı.

## Kesinleşen hedef: Türk İşaret Dili (TİD)

Kullanıcı Türk İşaret Dili (TİD) istediğini doğruladı. Eğitim iki el ve 29 harflik TİD parmak alfabesi üzerinden hazırlanacak. Fingerspelling.xyz ASL öğretir; burada görsel kalite ve akıcılık referansıdır. Mevcut yaklaşık Türkçe harf tablosu doğrulanmış kaynak olarak kullanılmamalı. Bu belgedeki koşullu TİD gereksinimleri artık zorunlu kapsamdır.

## Önerilen üretim

1. Eğitim için bağımsız sahne ve hareket oynatıcı oluştur.
2. Önce mevcut rig üzerinde zor pozlarla deformasyon denemesi yap. Uygunsa yeniden kullan; yetersizse Blender'da sade, yuvarlak hatlı, net parmak siluetleri olan bir el üret veya yeniden rigle.
3. Dört parmağın taban, orta ve uç eklemlerini ayrı kontrol et. Parmak açılımı, başparmak karşılaması, avuç kavisi ve bilek dönüşü ayrı kontroller olsun. TİD seçilirse iki bileğin konum ve yönünü birlikte yönet.
4. Her harfe doğrulanmış eklem pozları, bilek yönü, gerekiyorsa hareket eğrisi ve görünüş açısı tanımla. Parmak temaslarını ve çapraz parmakları elle ayarla.
5. Geçişleri zaman tabanlı çalıştır. Hızlı harf değişiminde ekrandaki mevcut pozdan devam et. Dönüşlerde quaternion interpolasyonu kullan; kesişen parmaklar için gerektiğinde araya güvenli pozlar ekle. Yalnızca yumuşak interpolasyon çarpışmasız hareket garantisi değildir.
6. Hareketli harflerde hazırlık, esas hareket, bekleme ve yumuşak dönüş aşamaları oluştur. Dekoratif salınım yerine öğretilecek hareketi göster.
7. Kullanıcıya tekrar, hız ayarı, duraklatma, görünüşü döndürme ve doğru öğrenme açısına dönme kontrolleri sun.

## Kullanıcının sağlayabileceği girdiler

- Alfabe seçimi tamamlandı: TİD. Öncelikli kapsam, 29 harfin doğru gösterimi ve bunların arasındaki kesintisiz geçişlerdir. Alfabe dışındaki işaretler ayrı bir referans hareket listesiyle genişletilebilir.
- Projeye özel hareketler varsa, harf adıyla eşleştirilmiş referans fotoğraf ve videolar. Hareketli harflerde başlangıç, hareket ve bitiş görünmeli; el ve bilek kadrajda olmalı. Önden ve yandan çekimler temasları çözmeyi kolaylaştırır. Standart alfabe için kamuya açık yetkin kaynaklar araştırılabilir; ayrıca çekim zorunlu değildir.
- Blender gerekiyorsa resmi Windows sürümü. PATH ve standart Program Files konumunda bulunamadı; başka konuma kuruluysa `blender.exe` yolu yeterli. Blender eklentisi veya ücretli program zorunlu değil.
- Hazır model kullanılacaksa modelin kaynak bağlantısı ve lisansı. Mevcut arşiv yalnızca GLB içeriyor; lisans belgesi içermiyor. Sıfırdan özgün model üretimi de seçenek.

Kullanıcının hazır model satın alması şu aşamada gerekli değil. Fingerspelling.xyz modelinin birebir kullanımı istenirse kaynak dosya ve kullanım izni ayrıca edinilmeli; benzer görsel yaklaşım için birebir varlık gerekmez.

## Bir 3B sanatçıya verilebilecek teslim tarifi

“Web üzerinde parmak alfabesi öğreten bir uygulama için stilize el rig'i. Gerçekçi cilt zorunlu değil; parmakların ve temasların okunaklı olması öncelikli. Her parmak eklemi bağımsız kontrol edilebilir, başparmak avuca karşılanabilir, parmaklar açılabilir ve çaprazlanabilir, bilek üç eksende dönebilir olmalı. Yumruk, halka ve çapraz parmak pozlarında yüzey çökmemeli veya belirgin biçimde kesişmemeli. Hedef alfabenin doğrulanmış bütün harfleri ve hareketli harf animasyonları hazırlanmalı. Düzenlenebilir .blend, web için .glb, kemik isimleri, animasyon listesi ve lisans teslim edilmeli. Kontrol rig'i animasyonları GLB'de çalışacak kemik anahtar karelerine aktarılmalı. TİD seçilirse iki el birlikte hazırlanmalı.”

## Kabul ölçütleri

- Hedef alfabedeki bütün harfler referansla karşılaştırılır; şekli bozuk harfler kapsamdan çıkarılarak tamamlandı sayılmaz.
- Benzer harfler etiketsiz görsel karşılaştırmada ayırt edilebilir olmalı. Dilsel doğruluk için yetkin bir işaret dili kullanıcısının incelemesi gerekir.
- Parmak uçlarında, başparmak temaslarında ve avuç içinde belirgin kesişme veya çökme olmamalı.
- Tüm sıralı harf çiftleri ve hızlı kullanıcı seçimleri denenmeli; poz sıçraması, döngü başında atlama, eski animasyonun geri gelmesi olmamalı.
- Hareketli harflerin yönü, çizdiği yol ve tekrar sayısı referansla uyuşmalı.
- Masaüstü ve mobil boyutlarda el kadraj içinde kalmalı. Performans hedef cihazlarda ölçülmeli; 60 FPS hedefi test sonucu görülmeden garanti olarak sunulmamalı.
- Son teslim: kaynak model, web modeli, poz/hareket kütüphanesi, eğitim entegrasyonu ve görsel doğrulama kaydı.

## Kaynaklar

- [Fingerspelling.xyz: ASL eğitim aracı](https://fingerspelling.xyz/)
- [Geliştirici röportajı: three.js, TweenMax ve ASL uzmanıyla doğrulama](https://www.commarts.com/webpicks/fingerspelling-xyz)
- [Ankara EGO: İşaret Dili Öğreniyorum, iki elle TİD parmak alfabesi](https://isaretdili.ego.gov.tr/wp-content/uploads/2021/09/isaret-dili-web.pdf)
- [Blender resmi indirme sayfası](https://www.blender.org/download/)
