"""Package a completed, measured run without switching the live API model."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    status=json.loads((args.run/'status.json').read_text(encoding='utf-8'))
    if status['status']!='complete':raise RuntimeError('Only a completed run can be packaged')
    result=json.loads((args.run/'evaluation.json').read_text(encoding='utf-8'))
    if args.output.exists():raise FileExistsError('Choose a new release directory; never overwrite a model')
    args.output.mkdir(parents=True)
    names=['model.pt','robust_cnn_ensemble.json','config.json','selection.json','evaluation.json',
           'per_class.json','training_curves.png','confusion_matrix.png','history.json',
           'test_started.json','test_predictions.csv']
    if (args.run/'provenance.json').exists():names.append('provenance.json')
    hashes={}
    for name in names:
        target=args.output/name
        shutil.copyfile(args.run/name,target)
        hashes[name]={'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size}
    def pct(value):return f'{value*100:.2f}'.replace('.',',')+'%'
    lines=['# Digitra TİD V6 — augmentation ile eğitilmiş araştırma modeli','',
        '## Kullanım','',
        '29 harflik TİD parmak alfabesi için statik görüntü sınıflandırıcı. RGB el görüntüsü, oranı korunarak 224×224 boyutuna getirilir. Mevcut Digitra API yükleyicisiyle uyumludur; bu paket canlı model seçimini değiştirmez.','',
        '## Ölçülen sonuçlar','',
        '| Bölüm | Doğruluk | Macro F1 |','|---|---:|---:|']
    for key,name in [('clean_real_train','Temiz gerçek eğitim'),('validation','Doğrulama'),('historical_test','Tarihsel test')]:
        lines.append(f"| {name} | {pct(result[key]['accuracy'])} | {pct(result[key]['macro_f1'])} |")
    lines+=['',f"Seçilen epoch: {result['selected_epoch']} ({result['selected_source']}). Test: {result['test_samples']} gerçek görüntü.",
        f"Tek model çıkarım gecikmesi medyanı: {result['runtime_latency_ms_median']:.1f} ms; P95: {result['runtime_latency_ms_p95']:.1f} ms. Kamera ve ağ gecikmesi dahil değildir.",
        f"Kaydetme/yükleme uyumunda en büyük olasılık farkı: {result['export_parity_max_probability_error']:.6f}.",'',
        '## Dayanıklılık testleri','', '| Koşul | Doğruluk | Macro F1 |','|---|---:|---:|']
    for item in result['robustness']:lines.append(f"| {item['variant']} | {pct(item['accuracy'])} | {pct(item['macro_f1'])} |")
    lines+=['','## Sınırlar','',
        '- Bu test grubu V5 için daha önce raporlanmıştır; yeni dış test değildir.',
        '- Kaynakta kişi kimliği yoktur. Sonuçlar kişi bağımsız başarı kanıtı değildir.',
        '- Statik kareler, yalnızca hareketle ayrılan harfler için yeterli olmayabilir.',
        '- 3D eller yardımcı eğitim verisidir; 3D görseller üzerindeki başarı gerçek kamera başarısı olarak raporlanmaz.',
        '- Gerçek veri CC BY-NC-SA 4.0 lisanslıdır; paket araştırma/ticari olmayan kullanım içindir.','',
        '## Tekrar üretme','',
        'Eğitim kodu: `ml/training/tid_v6/`. Parametreler, veri manifesti kimliği ve kullanılan sürümler `config.json` dosyasındadır. Doğrulama sonucu ile seçim `selection.json` içinde kilitlenmiştir. Eğitim ve test verileri pakete kopyalanmaz.','']
    (args.output/'model_card.md').write_text('\n'.join(lines),encoding='utf-8')
    hashes['model_card.md']={'sha256':hashlib.sha256((args.output/'model_card.md').read_bytes()).hexdigest(),'bytes':(args.output/'model_card.md').stat().st_size}
    (args.output/'manifest.json').write_text(json.dumps({'name':'digitra-tid-augmented-v6','files':hashes},indent=2),encoding='utf-8')
    print(json.dumps({'package':str(args.output.resolve()),'files':len(hashes)}))


if __name__=='__main__':main()
