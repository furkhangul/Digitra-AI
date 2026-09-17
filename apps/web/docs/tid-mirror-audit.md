# TİD parmak alfabesi — ayna görünümü denetimi
Tarih: 10 Eylül 2026

## D için son görünüm düzeltmesi

Kullanıcı incelemesinden sonra D'nin iki eli, sabit ön görünümde birlikte
26° döndürüldü. Sol bileğin `-26°` eğimi sıfırlandı; işaret parmağında yalnızca
elin doğal `2.75°` açılımı kaldı. Sağ elin kavisi, parmak şekilleri ve iki elin
temas geometrisi korundu. D'nin yüzey ve yaklaşma vektörleri de aynı dönüşümü
aldı; böylece yerleşim ve geçiş yönü yeni poza uydu.

Diğer 28 harfin pozları ve düzeltme vektörleri önceki kayıtla aynı.
Tarayıcıda D kontrol edildi; 841 harf geçişi (151.380 kare), yuvarlak el testleri,
TypeScript ve değişen poz dosyasının ESLint kontrolü geçti. İki elin bütün
çapalarında aynı dönüşümden sapma `6.9e-9` birimden küçüktür.
Aşağıdaki toplu denetim, bu D düzeltmesinden önceki kaydı anlatır.

## E için ayna görünümü ve sol işaret parmağı

A, B, C, Ç ve D kullanıcı tarafından tamamlandı olarak kabul edildi; bu pozlar
korundu. E'de sağ elin yatay yönü korunarak avuç yönü 180° çevrildi
(`rotation: [180,0,90]`); sol el Y ekseninde 90° döndürüldü (`[0,90,0]`).
Sol işaret parmağı, kullanıcının son yönlendirmesiyle sağa, üç yatay parmağın
başladığı yere taşındı. El boyutları ve parmak şekilleri değişmedi.

Yeni ilişki `index_0 / index_tip`, gap `[0.14,-0.066,-0.38]`; eski E yüzey
düzeltmesi sıfırlandı. Yaklaşma `[0,0,-0.525]` ile derinlik yönünden yapılır.
Uçlara yapılan önceki hizalama kaldırıldı. Sol el, parmak köklerinin önünde
kalacak derinliğe alındı. 14.240 yüzey örneğinde en derin örtüşme `-0.0041`
birimdir. Üst işaret-parmağı temas aralığı `0.1023` birim olduğundan bu yerleşim
önden kök hizasını sağlar; üst uçlar arasında fiziksel temas doğrulanmış değildir.
Ayna sırası artık parmak uçlarının ortalaması yerine avuç merkezleriyle ölçülür:
sağ avuç X=`0.526`, sol avuç X=`0.066`. E'de yatay parmakların öbür eli geçmesi
el gövdelerinin yer değiştirdiği anlamına gelmez.
841 harf geçişi, yuvarlak el testleri ve TypeScript kontrolü geçti.
E dışındaki 28 harfin pozları önceki kayıtla birebir aynı.

## F için el yönleri

Kullanıcının istediği üzere F'de sağ el X ekseninde 180°, sol el Y ekseninde
180° döndürüldü. Son rotasyonlar sağ `[180,0,90]`, sol `[0,180,180]`;
parmak şekilleri ve el boyutları korundu. Temas yerleşimi bu yönlerle yeniden
hesaplandı. 11.823 yüzey örneğinde amaçlanan temas aralığı `-0.0032`, temas
dışı en yakın aralık `0.1916` birimdir. Ayna yönü, temas ve 841 geçiş kontrolü
geçti. Bu adımda F dışındaki 28 harf değişmedi.

## G için fotoğraf ve çizim üzerinden düzeltme

Önceki 30° karşılıklı dönüş yalnızca başparmak temasını sağlıyordu; sağ işaret
parmağının sol C açıklığına girdiğini doğrulamıyordu. Yerleşim yeniden kuruldu.
[TDK/MEB G kaynağının](https://tdk.gov.tr/wp-content/uploads/2012/07/G.pdf)
111. sayfasındaki fotoğraf ve kullanıcının verdiği `Harfler-EDT3.jpg` çiziminin
aynalanmış karşılığı birlikte incelendi: sol C dış kavistir, sağ işaret parmağı
açıklığa uzanır, iki başparmak altta üst üste gelir.

Sağ el Y'de -115°, ardından dünya Z ekseninde +30°; sol el Y'de +115°,
ardından dünya Z ekseninde -15° döndürülür. Son XYZ Euler değerleri sağ
`[133.00307153,-51.71009587,126.20398975]`, sol
`[150.9680068,61.09544437,-147.62441159]` şeklindedir.
G/Ğ'ye özel başparmak bükümü `[0,0,0]`, açılımı `12°`, karşılaması `50°`;
el boyutları ve diğer parmakların şekilleri korunur. `thumb_tip / thumb_tip`
ilişkisi `[-0.1,-0.31,0]`, ek yüzey düzeltmesi sıfırdır. Yaklaşma yeniden
hesaplandı; Ğ'nin ortak pozu ve iki eli birlikte sallayan hareketi korundu.

14.134 yüzey örneğinde amaçlanan temas aralığı `0.0002`, temas dışındaki
en yakın aralık `0.0476` birimdir; avuçların ayna sırası doğrudur.
Sağ işaret ucu, sol başparmak-işaret ucu açıklığının %66,88 hizasındadır;
bu açıklık çizgisine üç boyutlu uzaklığı `0.1374` birimdir. Yeni kontrol,
yalnızca önden üst üste görünmeyi yeterli saymaz; parmağın açıklığa derinlikte
de yaklaşmasını sınar. Bu ölçümler uzman dilsel doğrulamanın yerini tutmaz.
Tarayıcıdaki G görünümü iki referansla karşılaştırıldı. 841 geçiş, yuvarlak el
testleri, TypeScript ve poz dosyasının ESLint kontrolü geçti.
G/Ğ dışındaki 27 harfin pozları bu adım öncesiyle aynıdır.

Son görsel incelemede kullanıcı bu biçimi beğenip yalnızca çok küçük bir
ayrılma istedi. Ardından aynı küçük adım bir kez daha uygulandı.
G/Ğ'nin yatay aralığı iki adımda toplam `0.12` birim artırıldı:
ilişki `[0.02,-0.31,0]` oldu. Her el dışa toplam `0.06` birim kayar; yönler,
boyutlar ve parmak biçimleri aynıdır. Bu son konumda işaret ucu açıklığın
%64,38 hizasındadır, çizgiye uzaklığı `0.1294` birimdir; el testleri geçti.
Yukarıdaki yüzey ölçümleri bu küçük ayrılmadan önceki kaynak uyarlamasına aittir.

## Ğ için üst işaret parmağının hafif hareketi

Kullanıcının son yönlendirmesiyle Ğ'nin bütün eli sağa-sola sallayan hareketi,
yalnızca üstteki sol işaret parmağını hafifçe yukarı-aşağı oynatan hareketle
değiştirildi. G ile aynı temel poz ve son açılan aralık korunur. İşaret parmağının
orta ekleminde ±5° sinüs salınımı vardır. Son hız isteğiyle salınım dört kat
hızlandırıldı: 3,4 saniyelik ana döngüde dört kez iner-kalkar; bir salınım
0,85 saniye sürer. Hareket aralığı aynı kalır.
Parmak ucunun toplam dikey hareketi `0.0757` birimdir. Sağ el, iki elin
konumu/yönü ve öteki parmaklar sabittir. Paylaşılan G el şekli, parmak hareketi
uygulanmadan önce sol el için ayrılır; böylece sağ işaret parmağı etkilenmez.
Yönerge metni de bu hareketi anlatır. Döngü, sabit G, hareket azaltma ve
841 geçiş kontrolleri geçti.

## H için A ile aynı yerleşim ve dar parmak aralığı

Kullanıcının isteğiyle H, onaylanan A'nın sağda aşağı uzanan iki parmağı ve
soldan yatay geçen işaret parmağıyla aynı ayna yerleşimine alındı. Dönüşler
sağ `[180,0,0]`, sol `[180,0,-90]`; temas ilişkisi `index_1 / index_2`.
Son kullanıcı düzeltmesiyle aşağı uzanan parmaklar **işaret ve serçe** olarak
değiştirildi; orta ve yüzük avuca kapanır. Kullanıcı ardından bu iki parmağın
eğimini kaldırmak istedi. İşaret `-2.75083461°`, serçe `5.27740634°` ile rig'in
doğal yelpaze açısı giderildi; ikisi de dünya ekseninde tam aşağı ve paralel
uzanır. Uç yönlerinin X/Z sapması `1e-10` birimden küçüktür. El boyutları
ve yatay işaret parmağının temas ilişkisi aynıdır.
H'nin yüzey düzeltmesi A'nın yerleşimine uyarlandı, yaklaşması yeniden hesaplandı.
Parmakları düzleştirmeden önce 14.134 yüzey örneğinde temas aralığı `-0.0002`, temas dışındaki en yakın aralık
`0.0201` birimdir; ayna sırası doğrudur. H dışındaki 28 harf aynı kaldı.
841 geçiş ve el testleri geçti. Son düzleştirmede yönler sayısal olarak ve
tarayıcıda kontrol edildi; el testleri tekrar geçti.

Son hizalamada sol el, sağ ele göre yatayda `0.065` birim yaklaştırıldı;
ilişki `[-0.34207,0.01795,-0.20679]` oldu. Sol işaret ucu iki dik parmağın
arasında kalır. Son 14.135 yüzey örneğinde temas aralığı `-0.0047`, en derin
diğer örtüşme `-0.0074` birimdir; temas ve ayna kontrolleri geçti.

## İ için altta sağ işaret ve üstte sol şıklatma

Kullanıcının isteğiyle İ'nin sağ eli altta sabit POINT, sol eli üstte SNAP
olarak düzenlendi. Sağ dönüş `[0,0,0]`, sol dönüş Ç'deki gibi `[0,55,10]`.
Sağ işaret ucu ile sol orta parmak ucu ilişkisi `[0,-0.63,0.05]`; bu ilişki
yerleşim içindir, iki el arasında temas beklenmez. Şıklatmanın asıl teması
sol baş ve orta parmak arasındadır. İ'nin eski yüzey düzeltmesi sıfırlandı ve
yaklaşma yeniden hesaplandı.

İ doğrudan Ç'nin SNAP şeklini ve `snap` hareket eğrisini kullanır: hazırlık,
0,12 saniyelik hızlı orta-parmak vuruşu, elin küçük geri tepmesi ve geri dönüş
aynıdır. Sağ el bütün döngü boyunca sabit kalır. Dokuz hareket anında ölçülen
eller arası yüzey aralığı `0.0200–0.0639` birimdir; çakışma görülmedi.
Şıklatma şekli ve zamanlamasının Ç ile eşitliği, sabit sağ el, döngü sonu,
841 harf geçişi, TypeScript ve ESLint kontrolleri geçti. İ dışındaki 28 harfin
pozları aynı kaldı; Ç'nin hareket eğrisi değiştirilmedi.

## J için L kenarını izleyen sol işaret parmağı

Sağ el L biçiminde sabit kalır; sol el `[90,0,0]` yönündedir. Sol işaretin
doğal açılımı giderildiği için ucu bütün hareket boyunca izleyiciye bakar.
Kullanıcının son düzeltmesiyle hareket, sağ işaretin en üst ucuyla aynı
yükseklikten başlar; L'nin iç kenarından aşağı inip başparmağın ucuna ulaşır.
Kısa duraklamadan sonra aynı açık yol üzerinden geri döner. Başlangıç
ilişkisi `index_tip / index_tip`, gap `[0.255,0,-0.495]`;
iki el arasında temas beklenmez.

J'nin yüzey düzeltmesi sıfırlandı ve yaklaşması yeniden hesaplandı. Yolun
13 anında örneklenen SDF yüzey aralığı `0.0239–0.1146` birimdir; bu örneklerde
çakışma görülmedi. Sabit sağ L, izleyiciye bakan sol işaret, işaretin üst ucundan
başparmak ucuna erişim ve aynı yoldan dönüş testleri geçti. 841 harf geçişi,
TypeScript ve ESLint kontrolleri de geçti. J dışındaki 28 harf aynı kaldı.

## K için dik sağ el ve ilk açısına dönen sol el

K'de sağ avuç izleyiciye bakar; işaret ve orta parmak dik ve paraleldir.
Sol elde de aynı bitişik parmak şekli kullanılır. İki elde başparmak, yüzük
ve serçe avuca kapanır. Kullanıcının geri alma isteğiyle X/Y eğimleri
kaldırıldı; sol el ilk `[0,0,-75]` açısına döndü. Parmakların bitişik şekli korunur.
Sağ el `[0,0,0]` yönündedir. Sol işaret ucu, sağ işaretin kök ekleminin
yanına `index_0 / index_tip`, gap `[0.28,0,-0.08]` ilişkisiyle yerleşir.

Eski K yüzey düzeltmesi sıfırlandı; yaklaşma yeni yöne göre yeniden hesaplandı.
13.510 yüzey örneğinde amaçlanan temas aralığı `-0.0044`, temas dışındaki en
yakın aralık `0.0145` birimdir; mevcut `-0.02` örtüşme toleransı içinde kaldı.
Ayna sırası, el testleri, 841 harf geçişi, TypeScript ve ESLint kontrolleri
geçti. K dışındaki 28 harf bu düzenlemeden önceki haliyle aynıdır.

## L için başparmak açıklığı

Yalnızca L'nin başparmak açılımı `12°` yerine `32°` yapıldı. Son kullanıcı
isteğiyle 85°'lik açıklık biraz daraltıldı. İşaret ile başparmak arasındaki
açı önden görünümde `82.04°`, üç boyutta `82.74°` oldu
(önceki üç boyutlu açı `64.68°`). Elin yönü ve diğer parmaklar korundu.
J'nin onaylanan L biçimi ve hareket yolu dahil diğer 28 harf değişmedi.
El testleri ve açı ölçümü geçti.

## N için el görevlerinin değişmesi

Kullanıcının isteğiyle N'deki iki parmak sağ ele, tek işaret parmağı sol ele
alındı. Önceki poz yatayda aynalandı; temas geometrisi korundu. Dönüşler sağ
`[180,0,0]`, sol `[180,0,-26]`; temas ilişkisi
`index_tip / index_tip`, gap `[0.18085,-0.02381,0.01413]` oldu.
Yüzey ve yaklaşma vektörleri de aynı yansımaya uyarlandı. N dışındaki 28 harf
aynı kaldı. El testleri ve 841 harf geçişi geçti; tarayıcıda yerleşim kontrol edildi.

## Ö için üstte iki ayrı noktada hızlı şıklatma

Sağ elin mevcut PINCH şekli ve `[0,-60,-50]` yönü korunur; sağ el önde
sabittir. Sol el Ç'deki SNAP şeklini, `[0,55,10]` yönünü kullanır ve üstte
yerleşir. Son kullanıcı isteğiyle sol el ayrıca `0.15` birim yükseltildi.
Sağ işaret ucu ile sol orta parmak ucu arasındaki düşey fark `0.75` birimdir;
eller arasında temas amaçlanmaz. Ö'nün eski yüzey düzeltmesi sıfırlandı ve
yaklaşması yeniden hesaplandı.

Sol orta parmak `0.69–0.77` ve `1.34–1.42` saniyelerinde iki hızlı vuruş
yapar. Arada el `0.22` birim sağa kayar; şıklatmalar Ö'nün iki noktası gibi
ayrı konumlardadır. İkinci şıklatmadan sonra başlangıç konumuna döner.
Ç ve İ'nin tek şıklatma zamanlaması korunur. Tam iki vuruş, sabit sağ el,
iki ayrı konum, yeniden kavrama ve döngü kapanışı testleri geçti.
19 hareket anında örneklenen en yakın yüzey aralığı `0.0019–0.1806` birimdir;
örneklerde çakışma görülmedi. 841 harf geçişi, TypeScript ve ESLint kontrolleri
geçti. Ö dışındaki 28 harfin pozları değişmedi.

## Sol elde P: işaret ucunun orta parmak eklemine teması

P yalnızca sol elle, `[180,-90,0]` yan görünümünde yapılır. Orta parmak
düz aşağı uzanır. İşaret kök ekleminden geriye alınır; orta ve uç eklemleri
öne kıvrılmaya devam eder. İşaretin `[-48,68,70]` bükümü ve `-4.5°`
açılımı, parmak kökünü taşımadan ucunu düz orta parmağın gerçek orta
ekleminin (`middle_1`, PIP) yanına getirir. Uç sola, gövdeye doğru döner;
sağdaki kavis ve soldaki düz parmak yandan P biçimini oluşturur.

İşaret ucu ile hedef eklemin ekrandaki hizalama farkı `0.0062` birim,
üç boyutlu merkez uzaklığı `0.21995` birimdir. Uzaklık yaklaşık iki yüzey
yarıçapının toplamıdır; kemik merkezleri üst üste bindirilmez. İşaretin
ilk boğumu gövdenin `0.2896` birim sağına çıkarak kavis açıklığını korur.
Başparmak önceki `[0,70,75]` büküm, `10°` açılım ve `45°` karşılama
konumundadır; yüzük ve serçe kapalı kalır.

Tek el yardımcı işlevi artık sol eli de destekler. Eğitim alanının
el sayısı ve renk açıklaması görünür ellere göre hesaplanır; P için
"TEK EL" ve yalnızca "Sol el" gösterilir. P testleri görünür eli,
geriye alınan kökü, öne kıvrık dış eklemleri, hedef eklemdeki yüzey
yakınlığını ve sağda kalan kavis açıklığını doğrular.

P'nin kamerası 10° yükseltildi (`viewElevation`). Bu yalnızca görüntüleme
dönüşümüdür; el pozu ve parmak teması aynı kalır. Harf değişiminde açı
yumuşak geçer; azaltılmış hareket tercihinde doğrudan uygulanır.
R de P'nin 10° kamera yüksekliğini kullanır; kalan harfler 0°'de kalır.

[TDK/MEB sözlüğünün P bölümü](https://tdk.gov.tr/wp-content/uploads/2012/07/P.pdf)
basılı 224 ve 229. sayfalardaki yakın el fotoğraflarıyla karşılaştırıldı.
224. sayfanın P açıklaması parmak adlarını ters verirken, 229. sayfadaki
PENALTI açıklaması orta parmağı açık, işaret parmağını kıvrık tarif eder.
Kullanıcının isteğiyle bu parmak rolleri sol ele aktarıldı; fotoğraftaki
düz parmak ile kavis arasındaki açıklık referans olarak tutuldu.

P dışındaki 28 harf değişmedi. Tarayıcıda yan görünüm incelendi; el testleri,
841 harf geçişi, TypeScript ve ESLint kontrolleri geçti.

## R: sol P ve sağ işaretle çapraz devam

R'nin sol eli P ile aynı el biçimini ve `[180,-90,0]` yönelimini kullanır.
Sağ işaret düz, diğer sağ parmaklar kapalıdır. Sağ el `[0,0,40]`
yöneliminde, işaret uçları arasında `[0.03,-0.22,0]` farkla yerleşir.
Sağ işaret, sol işaretin bittiği noktadan sağ aşağı uzanan R bacağını
oluşturur. Eski iki kıvrık işaret pozu ve yüzey düzeltmesi kaldırıldı.

İki yönden örneklenen yüzey aralığı `0.0059–0.0096` birimdir;
örneklerde el yüzeyleri iç içe geçmez. R'nin yaklaşma/ayrılma yönü bu
yerleşime göre yeniden ölçüldü: `[0.328,-0.4092,-0.0246]`.
Kamera P gibi 10° yukarıdadır; kullanıcının sağa bakış isteğiyle ayrıca
10° sağa alınmıştır (`viewAzimuth`). Kamera dönüşü temas geometrisini
değiştirmez ve harf geçişinde yumuşak uygulanır.

R dışındaki 28 harfin pozları ve kamera ayarları değişmedi. Sol P biçimi,
sağ işaretin düzlüğü, uçların yakınlığı ve çapraz bacak yönü test edildi.

## S: G biçimiyle sol başparmak–sağ işaret teması

S, G'nin iki el biçimini ve yönelimlerini aynen kullanır. Temas hedefi
sağ `index_tip` ile sol `thumb_tip` olarak ayarlandı. Uç merkezleri
arasındaki `[0.22575,0.072,-0.02229]` farkın uzunluğu `0.238` birimdir;
bu, uç yüzeylerinin yarıçap toplamına karşılık gelir. Merkezler üst üste
bindirilmez. Eski S pozu ve ona ait yüzey düzeltmesi kaldırıldı.

Sağ el son ince ayarda `0.025` birim aşağı alındı. Uç temas mesafesini
korumak için sağa `0.00956` birim kayar; sol elin konumu sabittir.

Bu ince ayardan önce iki yönden örneklenen yüzey aralığı `0.0048–0.0092` birimdir;
örneklerde çakışma görülmedi. Bunlar sonlu örneklemenin sonuçlarıdır;
uç merkezleri ve yarıçaplar ayrıca doğrudan kontrol edilir. S'nin
yaklaşma/ayrılma yönü yeni yerleşime göre yeniden ölçüldü.

S dışında kalan 28 harf değişmedi. G ile aynı el biçimleri ve yönelimler,
doğru iki parmak ucu ve uçların yüzey teması test edildi.

## Ş: S duruşunda sağ alt parmaklarla şıklatma

Ş'nin başlangıç pozu son S ayarıyla aynıdır (`linkedS`). Sol el, sağ
işaret, sağ yüzük/serçe ve el gövdeleri hareket boyunca sabit kalır.
Böylece sol başparmak–sağ işaret teması korunur. Sağ el şekli örneklenirken
kopyalanır; G/S ile paylaşılan şekil veya sol el değişmez.

Sağ başparmak ve orta parmak 0.20–0.70 saniyede şıklatma hazırlığına gelir.
Baş/orta parmak hareketi Ç ve İ ile aynı ortak işlevi kullanır: 1.16–1.28
saniyede hızlı vuruş yapılır. Ş'de işaret ve bilek vuruşa eşlik etmez.
2.35–2.80 saniyede yeniden S duruşuna dönülür; 3.4 saniyelik döngü
atlama olmadan kapanır. Azaltılmış hareket tercihinde S duruşu sabit gösterilir.

Ş'nin yüzey düzeltmesi sıfırlandı, yaklaşma yönü S ile eşitlendi.
Ş dışındaki 28 harfin duruşları ve 69 zaman örneğindeki hareketleri önceki
sürümle aynı kaldı. Sol elin sabitliği, üst temas, hazırlıkta baş/orta
parmak teması, Ç ile aynı hızlı vuruş ve S'ye dönüş testleri geçti.
841 harf geçişi, TypeScript ve ESLint kontrolleri de geçti. Hareketin 13
anında örneklenen eller arası en yakın yüzey aralığı `0.0052` birim kaldı;
örneklerde çakışma görülmedi.

## U: yandan açık başparmak ve işaret

U sağ elde başparmak ve işaret parmağıyla oluşturulur; orta, yüzük ve
serçe avuca kapalıdır. İşaret kökü 55° bükülürken diğer iki eklemi düz
kalır; doğal parmak açılımı `-2.75083461°` ile giderilir. Başparmak
`[0,0,0]` büküm, 12° açılım ve 50° karşılama ile açık tutulur.

Elin `[-90,35,90]` yan görünümü iki parmağı yukarı getirir: uç yönlerinin
düşey bileşenleri işarette `1`, başparmakta `0.963` olur. Uçlar yatayda
`0.673` birim ayrıdır; yükseklik farkı `0.134` birimdir. Böylece başparmak
ve işaret arasındaki açıklık U'nun iki kolunu oluşturur. Ü dahil diğer
28 harfin pozu ve hareket ayarları değiştirilmedi.

## Ü: sabit U üzerinde Ö'nün iki hızlı şıklatması

Ü'nün sağ eli son U biçimini ve `[-90,35,90]` yönelimini aynen kullanır.
Sol el Ö ile aynı `SNAP` biçiminde ve `[0,55,10]` yönelimindedir.
İki el arasında temas hedefi yoktur. Sol el U açıklığının üstüne ve önüne
alınır; başlangıç yerleşimine göre `[0.22,0.65,0.60]` kaydırıldıktan sonra
iki el birlikte kadraja ortalanır. Böylece şıklatan avuç U parmaklarının
arkasında görünmez ve vuruşta aşağı inerken de onlara girmez.

`double-snap` hareketi Ö ile ortaktır. Vuruşlar 0.69–0.77 ve 1.34–1.42
saniyelerde gerçekleşir; ikinci nokta için sol el 0.22 birim sağa kayar.
Sağ U hareket boyunca sabittir. İki vuruştan sonra sol el başlangıç
yerine döner; 3.4 saniyelik döngü kesintisiz kapanır.

Sol elin her örnekteki parmakları, dönüşü ve konum değişimi Ö ile
karşılaştırıldı; tam iki vuruş ve sabit sağ U testleri geçti. Son yaklaşma
yönüyle 841 harf geçişi doğrulandı. Hareketin 19 anında sağ elin yüzeyinden
örneklenen en yakın sol el aralığı `0.037–0.163` birimdir; örneklerde
çakışma görülmedi. Ü dışındaki 28 harfin duruş ve hareketleri korundu.

## V: başparmak yüzüğün yanında, orta parmak kökünde

V'nin başparmağı `[8,36.5,59.625]` büküm, 64° açılım ve 75° karşılama ile
yüzük parmağının yanından orta parmağın köküne getirildi. Yerel uç
konumu `[-0.0802,0.9305,0.5192]` olur; ekranda orta parmak kökünün hemen
altında durur. Yüzüğün kıvrılan ilk eklemine uzaklığı `0.2432` birimdir;
bu yaklaşık parmak yüzeylerinin toplam yarıçapıdır.
İşaret ve orta parmağın açıklığı, yüzük/serçe duruşu ve elin yönelimi
korunur. Değişiklik yalnızca V girdisindedir; M/N/Y'nin kullandığı ortak
el biçimi değiştirilmez.

## Y: sağ V ve köküne alttan gelen sol işaret

Kullanıcının yönlendirmesiyle sağ el, son V biçimiyle aynı parmakları kullanır.
Y ekseninde 180° çevrilen sağ ele son istekte 9° sağdan geliş eğimi verildi
(`[0,180,-9]`); el sırtı izleyiciye bakar. Sol elde yalnızca işaret parmağı açıktır.
Sol el de Y ekseninde 180° çevrildi; iki elin sırtı izleyiciye bakar.
Sol elin hafif soldan geliş eğimi 9°'dir (`[0,180,9]`).
Sol işaret dik konuma yakın kalır; ucu
`[0.07765,0.96402,1.095]` noktasında (sağ elin konumuna göre dünya eksenlerinde),
önden bakıldığında iki V parmağının kökünün ortasına hizalanır. Dönen elin
kökleriyle birlikte yatay hizası güncellendi; sol işaretin ucu aynı hedefte tutuldu.
Sol elin dönüşünde kapalı parmakların sağ ele girmesini önlemek için sol el
yalnız derinlikte `0.250` birim öne alındı. İki yönlü yüzey örnekleriyle
yaklaşık `0.0049` birim minimum açıklık sağlandı; önden birleşim hizası korundu.
Bu dönüşte sol avuç işaretin sağında kalır; Y'de el gövdeleri ekranda
çapraz sıralanır. Genel ayna sıralama kontrolü bu istenen Y düzenini ayrıca doğrular.
Y'nin eski yüzey düzeltmesi sıfırlandı ve yaklaşma vektörü yeniden hesaplandı.
V için ortak `V_SIGN` tanımı kullanılır; diğer harflerin pozları korunur.

Aşağıdaki toplu kayıt, yukarıdaki harf düzeltmelerinden önceki denetimdir.
Bu rapor sabit önden kamera için 29 harfin tamamını denetler. Kabul kuralı,
semantik sağ elin ekranda sağda kalması ve pozun kullanıcının aynadaki karşılığı
olarak anatomik biçimde alınabilmesidir. Mesafeler avuç boyu birimindedir;
negatif değer iki SDF yüzeyinin örtüştüğünü gösterir.

## Sonuç

- Değiştirilen harfler: **A, E, G, Ğ, M, N, Z**.
- Değiştirilmeyen ve önden ayna/anatomi kontrolünden geçen harfler:
  **B, C, Ç, D, F, H, I, İ, J, K, L, O, Ö, P, R, S, Ş, T, U, Ü, V, Y**.
- İki elli 22 harfte yanlış taraf sırası kalmadı.
- Amaçlanan temas hedefi dışında kalan harf yok.
- Temas dışındaki en derin örtüşme `-0.0158` (F); kabul sınırı `>-0.02`.
- Ç ve Ö kasıtlı olarak ayrık bırakıldı: sırasıyla `0.1016` ve `0.1083`.

Son SDF görüntüleri:

- [29 harf kontrol sayfası](../scripts/tid_sdf/sheet/mirror-final-alphabet.png)
- Her harfin ayrı karesi aynı klasörde `mirror-final-{harf}.png` adıyla bulunur.

## Değişen pozlar

`position` değerleri `centre()` ve kalibre edilmiş `surface-offsets.json`
uygulandıktan sonraki son değerlerdir. `gap`, `joined()` fonksiyonuna yazılan
ham ilişki vektörüdür.

| Harf | Rotation eski → yeni (sağ / sol) | Position eski → yeni (sağ / sol) | Gap eski → yeni |
|---|---|---|---|
| A | `[0,0,180] / [0,0,-90]` → `[180,0,0] / [0,0,-90]` | `[0.433252,0.943488,-0.655268] / [-0.905744,0.015566,-0.055268]` → `[0.669729,0.943531,-0.106632] / [-1.018721,0.017573,0.106632]` | `[0.20,0.02,-0.60]` → `[-0.27707,0.01795,-0.20679]` |
| E | `[0,-55,-116] / [0,35,-15]` → `[0,0,90] / [0,0,0]` | `[-0.341546,0.803056,-0.917489] / [-0.560168,-0.924299,0.602536]` → `[0.949390,0.980512,-0.506642] / [-0.455507,-1.329504,-0.203893]` | `[0.086,0.051,-0.25]` → `[0.10790,0.18233,-0.29028]` |
| G | `[0,0,-90] / [0,0,114]` → `[-30,-25,0] / [0,55,0]` | `[-0.079975,0.211815,-0.311472] / [0.069858,-0.585632,-0.356064]` → `[0.575545,-0.891011,0.128840] / [-0.688163,-0.536981,-0.184510]` | `[-0.20,0.753,0.043]` → `[0.23962,0,0.08722]` |
| Ğ | `[0,0,-90] / [0,0,114]` → `[-30,-25,0] / [0,55,0]` | `[-0.079975,0.211815,-0.311472] / [0.069858,-0.585632,-0.356064]` → `[0.575545,-0.891011,0.128840] / [-0.688163,-0.536981,-0.184510]` | `[-0.20,0.753,0.043]` → `[0.23962,0,0.08722]` |
| M | `[0,0,180] / [0,0,180]` → `[180,0,0] / [180,0,0]` | `[-0.617956,0.940355,-0.355268] / [0.617956,0.940355,-0.355268]` → `[0.617950,0.940355,0.355268] / [-0.617950,0.940355,0.355268]` | `[-0.191,0,0]` → `[0.18646,0,0]` |
| N | `[0,0,180] / [0,0,154]` → `[180,0,26] / [180,0,0]` | `[-0.762724,0.940737,-0.363016] / [1.067430,0.588398,-0.347520]` → `[1.067520,0.634950,0.347512] / [-0.762815,0.940994,0.363023]` | `[-0.196,0.023,-0.015]` → `[0.18085,0.02381,-0.01413]` |
| Z | `[0,0,-90] / [0,0,90]` → `[180,0,90] / [180,0,-90]` | `[-1.889646,-0.387235,-0.302691] / [1.889646,0.220563,-0.407844]` → `[1.889013,0.220351,0.407809] / [-1.889013,-0.387022,0.302727]` | `[0,0,0.09]` → `[0.14524,0.04923,0.09852]` |

N'de var olan `POINT` ve `V` şekilleri semantik ellere ters atanmıştı;
`right: V / left: POINT` yerine `right: POINT / left: V` kullanıldı. Parmak
eklem açıları değiştirilmedi. Z'de aynı nedenle temas çapaları
`right:index_tip / left:middle_tip` yerine
`right:middle_tip / left:index_tip` oldu.

G ve Ğ tek şekil istisnasıdır: `THUMB_UP/FIST` yerine `C/C`, bilek teması
yerine `thumb_tip/thumb_tip` teması kullanıldı. Ğ hareketi de başparmağı tek
başına bükmekten çıkarılıp iki eli birlikte sağa-sola taşıyan `side-sway`
hareketine çevrildi.

## Neden değişti

- **A:** Sağ elin `rz=180` dönüşü parmakları aşağı çevirse de avuç yönünü
  değiştirmiyordu ve sağ bileği ayna testinde alınamaz hâle getiriyordu.
  `rx=180` aynı aşağı yönü koruyup avuç/back yönünü fiziksel ayna karşılığına
  çevirdi; sol yatay çubuk şekli değişmedi.
- **E:** Eski iki bilek de Z derinliğine kaçıyordu. Sağdaki üç açık parmak
  artık doğrudan sola, soldaki işaret parmağı doğrudan yukarı bakıyor.
- **M/N/Z:** Yalnız pozisyonları takas etmek M/N'de `-0.32` ile `-0.40`
  arası ağır iç içe geçme üretti. Bunun yerine parmak yönünü koruyan tam bilek
  ayna dönüşümü uygulandı; N'nin semantik şekilleri ve Z'nin temas çapaları da
  doğru ellere taşındı.
- **G/Ğ:** 2015 çizelgesindeki G2/Ğ2 yumruk varyantı, kullanıcı bildirimi ve
  TDK/MEB sözlük tarifi doğrultusunda iki C el varyantıyla değiştirildi.

## Kalibrasyon vektörleri

| Harf | Surface offset eski → yeni | Approach offset eski → yeni |
|---|---|---|
| A | `[0,0,0]` → `[0.019987,0.000086,-0.016474]` | `[0.3296,0.0032,-0.4086]` → `[0.4051,0.0017,-0.3339]` |
| E | `[0.004825,0.006739,-0.009028]` → `[0.006779,0.040651,-0.012469]` | `[0.2067,0.2887,-0.3867]` → `[0.0827,0.4957,-0.1520]` |
| G | `[0.050167,0.044447,0.001592]` → `[0.017440,-0.006698,-0.000434]` | `[0.3928,0.3481,0.0125]` → `[0.4900,-0.1882,-0.0122]` |
| Ğ | `[0.050167,0.044447,0.001592]` → `[0.017440,-0.006698,-0.000434]` | `[0.3928,0.3481,0.0125]` → `[0.4900,-0.1882,-0.0122]` |
| M | `[-0.013584,0,0]` → `[0.018113,0,0]` | `[-0.5250,0,0]` → `[0.5250,0,0]` |
| N | `[-0.008601,0.000763,-0.000496]` → `[0.023932,-0.001278,-0.001381]` | `[-0.5235,0.0224,-0.0327]` → `[0.5234,-0.0280,-0.0302]` |
| Z | `[-0.173144,-0.058684,0.010153]` → `[0.026639,0.009029,0.001562]` | `[-0.4965,-0.1683,0.0291]` → `[0.4965,0.1683,0.0291]` |

## Harf başına yüzey ölçümü

`Kastedilen temas`, yalnız `pose.contact` çevresinde ölçülür.
`Başka yerde en derin`, bu temas çevresi dışındaki minimum SDF değeridir.
Tek elli harflerde iki-el teması ve taraf sırası uygulanamaz.

| Harf | Kastedilen temas | Başka yerde en derin | Taraf sırası | Sonuç |
|---|---:|---:|---|---|
| A | -0.0015 | 0.0037 | doğru | geçti |
| B | 0.0056 | 0.0038 | doğru | geçti |
| C | — | — | tek el | geçti |
| Ç | 0.1016 | 0.0907 | doğru | geçti; kasıtlı ayrık |
| D | 0.0009 | 0.0809 | doğru | geçti |
| E | -0.0011 | 0.0009 | doğru | geçti |
| F | -0.0002 | -0.0158 | doğru | geçti |
| G | 0.0013 | 0.1356 | doğru | geçti |
| Ğ | 0.0013 | 0.1356 | doğru | geçti |
| H | 0.0004 | -0.0006 | doğru | geçti |
| I | — | — | tek el | geçti |
| İ | -0.0036 | 0.1786 | doğru | geçti |
| J | -0.0007 | 0.1684 | doğru | geçti |
| K | -0.0044 | -0.0047 | doğru | geçti |
| L | — | — | tek el | geçti |
| M | -0.0021 | 0.0840 | doğru | geçti |
| N | 0.0011 | 0.1908 | doğru | geçti |
| O | — | — | tek el | geçti |
| Ö | 0.1083 | 0.1076 | doğru | geçti; kasıtlı ayrık |
| P | — | — | tek el | geçti |
| R | 0.0006 | 0.0008 | doğru | geçti |
| S | 0.0046 | 0.0908 | doğru | geçti |
| Ş | -0.0069 | 0.1927 | doğru | geçti |
| T | -0.0009 | 0.1566 | doğru | geçti |
| U | — | — | tek el | geçti |
| Ü | -0.0191 | 0.2291 | doğru | geçti; temas sınırın 0.0009 üstünde |
| V | — | — | tek el | geçti |
| Y | 0.0003 | 0.0005 | doğru | geçti |
| Z | 0.0001 | 0.3929 | doğru | geçti |

## G/Ğ kaynağı ve belirsizlik

TDK'nin yayımladığı, MEB tarafından okullar için hazırlanmış Türk İşaret Dili
Sözlüğü G maddesi iki eli “işaret ve başparmaklar açık, kıvrık; öbür parmaklar
kapalı (C el)” diye tarif eder. Sağ işaret parmağının sol elin iki açık parmağı
arasına getirildiğini ve başparmakların üst üste konduğunu açıkça belirtir:

- [TDK — Türk İşaret Dili Sözlüğü](https://tdk.gov.tr/icerik/basindan/turk-isaret-dili-sozlugu/)
- [TDK/MEB — G maddesi, s.111](https://tdk.gov.tr/wp-content/uploads/2012/07/G.pdf)
- [TDK/MEB — Ğ maddesi, s.118](https://tdk.gov.tr/wp-content/uploads/2012/07/%C4%9E.pdf)

Ğ maddesi aynı iki C el biçimini korur ve iki elin çene altında sağa-sola
sallandığını söyler. Kaynak genlik, tempo ve tekrar sayısı vermediğinden,
uygulamadaki `0.11` birimlik tek yumuşak salınım ve `3.4 s` döngü yalnız
görselleştirme tercihidir; dilsel kaynak ölçümü değildir.

Kaynaklar arasında varyant farkı vardır: Dikyuva, Makaroğlu & Arık (2015)
çizelgesi ile EGO öğrenim materyali G2/Ğ2 yumruk varyantını gösterir. Bu çalışma,
kullanıcı geri bildirimi üzerine TDK/MEB sözlük varyantını seçmiştir. Önden
izdüşümde sağ işaret ucu sol C açıklığına `0.0127` birim uzaklıkta ve açıklığın
`0.4406` oranındaki noktasındadır. Kaynak videosu olmadığı için tam hareket
genliği/ritmi ve parmakların derinlikteki kesin sırası uzmanla doğrulanmış
sayılmamalıdır.

## Doğrulama kayıtları

- `node scripts/dump-tid-poses.cjs`: 29 harf, unsettled yok.
- `node scripts/test-tid.cjs`: 841 çift, 151,380 kare, geçti.
  En büyük konum adımı `0.240488 < 0.3`; en büyük eklem adımı `18.255780°`.
- `python scripts/tid_sdf/verify.py`: 22 iki-elli harf; temas hatası yok,
  `wrongSideOrder=[]`, `penetrating=[]`.
- Değişen yedi harfin geçiş SDF örneklemesi: en kötü `-0.0141`,
  `-0.02` altında geçiş yok.
- `node scripts/test-rounded-hand.cjs`: kapalı yüzeyler, G/Ğ C-el eşitliği ve
  Ğ'nin teması bozmayan ortak salınımı geçti.
- `npm run lint`, `npx tsc --noEmit`, `npm run build`: geçti.

SDF taraması yoğun fakat sonlu, tohumlanmış yüzey örneklemesidir; bu sayılar
matematiksel sürekli-yüzey ispatı veya TİD uzman onayı değildir.
