"""Reproducible TID V6 training. Only validation data selects the checkpoint.

The historical V5 test set stays untouched until finalization. The deployed
model and teaching-hand assets are never written by this script.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'Digitra/digitra-tid-v4'))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from src.cnn import build_model, set_backbone_trainable, tta_logits
from augment import TrainingTransform, evaluation_transform, stress_transform


def save_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def save_torch(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(data, temp)
    temp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def score(targets, logits):
    predicted = np.asarray(logits).argmax(1)
    return {'accuracy': float(accuracy_score(targets, predicted)),
            'macro_f1': float(f1_score(targets, predicted, average='macro', zero_division=0))}


class Images(Dataset):
    def __init__(self, frame, labels, transform, synthetic_weight=.20, variant=None):
        self.rows = frame.to_dict('records')
        self.mapping = {label: i for i, label in enumerate(labels)}
        self.transform = transform
        self.synthetic_weight = synthetic_weight
        self.variant = variant

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        synthetic = row.get('source') == 'digitra_3d'
        with Image.open(row['path']) as opened:
            image = opened.convert('RGBA' if synthetic else 'RGB')
        tensor = stress_transform(image, self.variant) if self.variant else self.transform(image)
        return tensor, self.mapping[str(row['label'])], self.synthetic_weight if synthetic else 1.


def loader(dataset, batch_size, shuffle=False, seed=42, workers=2):
    # PIL/OpenCV augmentation is CPU-bound on the laptop GPU. A small worker
    # pool keeps the CUDA stream fed without changing sample order or seeds.
    kwargs = dict(batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                  pin_memory=torch.cuda.is_available(), generator=torch.Generator().manual_seed(seed))
    if workers:
        kwargs.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(dataset, **kwargs)


@torch.inference_mode()
def evaluate(model, data, device, scales=(1.,), mirror=False):
    model.eval()
    chunks, truths = [], []
    for images, targets, _ in data:
        images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            logits = tta_logits(model, images, scales, mirror)
        chunks.append(logits.float().cpu())
        truths.append(targets)
    return torch.cat(chunks).numpy(), torch.cat(truths).numpy()


def calibration(logits, truth):
    data = torch.tensor(logits, dtype=torch.float64)
    labels = torch.tensor(truth, dtype=torch.long)
    values = np.geomspace(.4, 3.0, 81)
    losses = [float(nn.functional.cross_entropy(data / float(t), labels)) for t in values]
    temperature = float(values[np.argmin(losses)])
    probabilities = torch.softmax(data / temperature, dim=1).numpy()
    confidence = probabilities.max(1)
    correct = probabilities.argmax(1) == truth
    options = []
    for t in np.concatenate(([0.], np.linspace(.2, .95, 76))):
        accepted = confidence >= t
        if accepted.sum() >= max(30, int(len(truth) * .5)):
            options.append({'threshold': float(t), 'coverage': float(accepted.mean()),
                            'accuracy': float(correct[accepted].mean())})
    eligible = [item for item in options if item['accuracy'] >= .9]
    threshold = max(eligible, key=lambda x: x['coverage']) if eligible else max(options, key=lambda x: (x['accuracy'], x['coverage']))
    return temperature, threshold


def ece(probabilities, truth, bins=15):
    confidence = probabilities.max(1)
    correct = probabilities.argmax(1) == truth
    error = 0.
    for i in range(bins):
        mask = (confidence > i / bins) & (confidence <= (i + 1) / bins)
        if mask.any():
            error += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return error


def plots(output, history, truth, prediction, labels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [r['epoch'] for r in history]
    axes[0].plot(epochs, [r['train_loss'] for r in history], label='Training loss')
    axes[0].set(xlabel='Epoch', ylabel='Loss')
    for key, label in [('train_aug_accuracy', 'Augmented training'), ('val_accuracy', 'Validation'), ('ema_val_accuracy', 'EMA validation')]:
        axes[1].plot(epochs, [r[key] * 100 for r in history], label=label)
    axes[1].set(xlabel='Epoch', ylabel='Accuracy (%)', ylim=(0, 101))
    for ax in axes: ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); fig.savefig(output / 'training_curves.png', dpi=150); plt.close(fig)
    cm = confusion_matrix(truth, prediction, labels=np.arange(len(labels)))
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set(xlabel='Predicted', ylabel='True label', title='Historical real-image holdout')
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]: ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=6)
    fig.colorbar(im, ax=ax); fig.tight_layout(); fig.savefig(output / 'confusion_matrix.png', dpi=150); plt.close(fig)


def finalize(args, model, frame, labels, history, device, fingerprint):
    output = args.output
    if (output / 'test_started.json').exists():
        raise RuntimeError('This run already opened the test set. Do not reuse it for further model selection.')
    checkpoint = torch.load(output / 'best.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])
    val_data = loader(Images(frame[frame.split == 'val'], labels, evaluation_transform), args.batch_size)
    options = []
    for scales, mirror in [((1.,), False), ((1.,), True), ((.85, 1., 1.15), True)]:
        logits, truth = evaluate(model, val_data, device, scales, mirror)
        options.append({'scales': scales, 'mirror': mirror, 'logits': logits, 'truth': truth, **score(truth, logits)})
    chosen = max(options, key=lambda r: (r['macro_f1'], r['accuracy']))
    temperature, selective = calibration(chosen['logits'], chosen['truth'])
    checkpoint.update({'tta_scales': list(chosen['scales']), 'tta_mirror': chosen['mirror'],
                       'temperature': temperature, 'recommended_confidence': selective['threshold']})
    save_torch(output / 'model.pt', checkpoint)
    save_json(output / 'selection.json', {'checkpoint_epoch': checkpoint['epoch'], 'checkpoint_source': checkpoint['source'],
              'validation': {k: v for k, v in chosen.items() if k not in ['logits', 'truth']},
              'temperature': temperature, 'selective_validation': selective, 'manifest_sha256': fingerprint})
    # Selection is durably locked before any test predictions are generated.
    save_json(output / 'test_started.json', {'time': time.time(), 'model_sha256': hashlib.sha256((output / 'model.pt').read_bytes()).hexdigest()})
    clean_train = loader(Images(frame[frame.split == 'train'], labels, evaluation_transform), args.batch_size)
    train_logits, train_truth = evaluate(model, clean_train, device, chosen['scales'], chosen['mirror'])
    test_frame = frame[frame.split == 'test'].copy()
    robustness = []
    test_logits = test_truth = None
    for variant in ['clean', 'mirror', 'far', 'shift', 'rotation', 'low_light', 'high_key', 'backlight', 'color_temperature', 'blur', 'jpeg']:
        data = loader(Images(test_frame, labels, evaluation_transform, variant=None if variant == 'clean' else variant), args.batch_size)
        logits, truth = evaluate(model, data, device, chosen['scales'], chosen['mirror'])
        metrics = {'variant': variant, **score(truth, logits)}
        robustness.append(metrics)
        print(json.dumps({'stage': 'final_test', **metrics}), flush=True)
        if variant == 'clean': test_logits, test_truth = logits, truth
    probabilities = torch.softmax(torch.from_numpy(test_logits) / temperature, dim=1).numpy()
    prediction = test_logits.argmax(1)
    correct = prediction == test_truth
    rng = np.random.default_rng(args.seed)
    bootstrap = correct[rng.integers(0, len(correct), size=(2000, len(correct)))].mean(1)
    test_frame['predicted'] = [labels[i] for i in prediction]
    test_frame['correct'] = correct
    test_frame['confidence'] = probabilities.max(1)
    test_frame.to_csv(output / 'test_predictions.csv', encoding='utf-8-sig', index=False)
    save_json(output / 'per_class.json', classification_report(test_truth, prediction, labels=list(range(len(labels))), target_names=labels, output_dict=True, zero_division=0))
    # Check the serialized model through the same loader used by the current API.
    sys.path.insert(0, str(ROOT / 'apps/api'))
    from app.services.digitra_tid.image_model import _CheckpointClassifier
    restored = _CheckpointClassifier(output / 'model.pt', device)
    sample = next(iter(val_data))[0][:1].to(device, memory_format=torch.channels_last)
    model.eval()
    with torch.inference_mode():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            expected = torch.softmax(tta_logits(model, sample, chosen['scales'], chosen['mirror']) / temperature, 1)[0].float().cpu().numpy()
        actual = restored.probability_vector(sample)
    difference = float(np.abs(actual - expected).max())
    if int(actual.argmax()) != int(expected.argmax()) or difference > .015:
        raise RuntimeError(f'Runtime export mismatch: {difference}')
    timings = []
    for i in range(25):
        if device.type == 'cuda': torch.cuda.synchronize()
        start = time.perf_counter(); restored.probability_vector(sample)
        if device.type == 'cuda': torch.cuda.synchronize()
        if i >= 5: timings.append((time.perf_counter() - start) * 1000)
    train_metrics = score(train_truth, train_logits)
    summary = {'name': 'digitra-tid-v6-research', 'architecture': args.architecture,
        'clean_real_train': train_metrics, 'validation': {k: chosen[k] for k in ['accuracy', 'macro_f1']},
        'historical_test': score(test_truth, test_logits), 'test_accuracy_95ci': np.quantile(bootstrap, [.025, .975]).tolist(),
        'test_ece': ece(probabilities, test_truth), 'robustness': robustness, 'temperature': temperature,
        'selected_epoch': checkpoint['epoch'], 'selected_source': checkpoint['source'], 'epochs_ran': len(history),
        'export_parity_max_probability_error': difference, 'runtime_latency_ms_median': float(np.median(timings)),
        'runtime_latency_ms_p95': float(np.quantile(timings, .95)), 'gpu': torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu',
        'targets': {'clean_train_90': train_metrics['accuracy'] >= .9, 'validation_90': chosen['accuracy'] >= .9,
                    'test_80': float(correct.mean()) >= .8},
        'limitations': ['Historical test already reported for V5; not a new external evaluation.',
                       'No signer identities; test is not signer-independent.',
                       'Synthetic hands are auxiliary training data, not evidence of real-world performance.',
                       'Static image classifier; moving letters need temporal context for reliable distinction.',
                       'Real dataset license: CC BY-NC-SA 4.0; research/non-commercial use.'],
        'source_manifest_sha256': fingerprint, 'test_samples': len(test_frame), 'classes': len(labels)}
    save_json(output / 'evaluation.json', summary)
    save_json(output / 'robust_cnn_ensemble.json', {'format_version': 1, 'recommended_confidence': selective['threshold'],
        'selection': 'V6 validation-selected single member', 'members': [{'checkpoint': 'model.pt', 'weight': 1.0}]})
    plots(output, history, test_truth, prediction, labels)
    save_json(output / 'status.json', {'status': 'complete', 'evaluation': str(output / 'evaluation.json'), **summary['targets']})
    print(json.dumps({'stage': 'complete', **summary}, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, default=ROOT / 'artifacts/tid-v6/data/manifest.csv')
    p.add_argument('--synthetic', type=Path, default=ROOT / 'artifacts/tid-v6/synthetic/manifest.csv')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--architecture', choices=['efficientnet_b0', 'mobilenet_v3_large'], default='efficientnet_b0')
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=24)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--patience', type=int, default=9)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--synthetic-weight', type=float, default=.20)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--finalize-only', action='store_true', help='Evaluate the validation-selected best.pt without another epoch.')
    p.add_argument('--skip-final-test', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'test_started.json').exists(): raise RuntimeError('Test opened: run is frozen.')
    if (args.output / 'last.pt').exists() and not args.resume and not args.finalize_only: raise RuntimeError('Run exists; use --resume.')
    seed_all(args.seed)
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda': raise RuntimeError('GPU training requires CUDA; no silent CPU fallback.')
    torch.backends.cudnn.benchmark = True
    fingerprint = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    frame = pd.read_csv(args.manifest, encoding='utf-8-sig')
    labels = sorted(str(x) for x in frame.label.unique())
    train_frame = frame[frame.split == 'train'].copy()
    if args.synthetic_weight > 0:
        synthetic = pd.read_csv(args.synthetic, encoding='utf-8-sig')
        synthetic = synthetic[synthetic.split == 'train'].copy()
        if not set(synthetic.label).issubset(labels): raise ValueError('Synthetic labels mismatch')
        train_frame = pd.concat([train_frame, synthetic], ignore_index=True)
    transform = TrainingTransform()
    dataset = Images(train_frame, labels, transform, synthetic_weight=args.synthetic_weight)
    val_data = loader(Images(frame[frame.split == 'val'], labels, evaluation_transform), args.batch_size, workers=args.workers)
    model = build_model(args.architecture, len(labels), pretrained=True).to(device, memory_format=torch.channels_last)
    if args.finalize_only:
        history_path = args.output / 'history.json'
        if not (args.output / 'best.pt').exists() or not history_path.exists():
            raise RuntimeError('Finalize-only requires a completed validation run with best.pt and history.json.')
        history = json.loads(history_path.read_text(encoding='utf-8'))
        finalize(args, model, frame, labels, history, device, fingerprint)
        return
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(.98), use_buffers=True)
    optimizer = torch.optim.AdamW([{'params': model.features.parameters(), 'lr': 2e-4},
                                  {'params': model.classifier.parameters(), 'lr': 8e-4}], weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=5e-6)
    scaler = torch.amp.GradScaler('cuda')
    counts = frame[frame.split == 'train'].label.value_counts()
    weights = np.array([1/math.sqrt(counts[label]) for label in labels]); weights /= weights.mean()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device), label_smoothing=.04, reduction='none')
    history, best_score, stale, first_epoch = [], -1., 0, 0
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update({'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(),
        'manifest_sha256': fingerprint, 'training_images': len(dataset), 'real_training_images': int((frame.split == 'train').sum()),
        'seed': args.seed, 'selection': 'maximum real-validation macro F1 among raw/EMA checkpoints',
        'augmentation': ['whole-pair mirror', 'rotation', 'translation', 'scale', 'shear', 'perspective', 'brightness', 'contrast', 'saturation', 'hue', 'gamma', 'exposure', 'high-key light', 'color temperature', 'white balance', 'sensor noise', 'soft shadow', 'backlight', 'JPEG compression', 'low resolution', 'motion blur', 'Gaussian blur', 'grayscale', 'small occlusion'],
        'training_features': ['ImageNet initialization', 'backbone warmup', 'AdamW', 'cosine LR schedule', 'mixed precision', 'gradient clipping', 'label smoothing', 'class weighting', 'EMA', 'early stopping', 'augmentation curriculum', 'atomic best/latest checkpoints', 'resume RNG state', 'low-weight synthetic auxiliary data']})
    if args.resume:
        saved = torch.load(args.output / 'last.pt', map_location='cpu', weights_only=True)
        if saved['manifest_sha256'] != fingerprint or saved['epochs'] != args.epochs: raise ValueError('Resume data/schedule mismatch')
        model.load_state_dict(saved['model']); ema.load_state_dict(saved['ema']); optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']); scaler.load_state_dict(saved['scaler'])
        history, best_score, stale, first_epoch = saved['history'], saved['best_score'], saved['stale'], saved['epoch']
        random.setstate(saved['python_rng']); np.random.set_state(('MT19937', saved['numpy_rng'].numpy().astype(np.uint32), saved['numpy_pos'], saved['numpy_has_gauss'], saved['numpy_cached']))
        torch.set_rng_state(saved['torch_rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
    else: save_json(args.output / 'config.json', config)
    print(json.dumps({'stage': 'started', **config}, ensure_ascii=False), flush=True)
    started = time.perf_counter()
    for epoch in range(first_epoch, args.epochs):
        seed_all(args.seed + epoch)
        set_backbone_trainable(model, epoch >= args.warmup)
        transform.strength = min(1., .45 + epoch * .09)
        train_data = loader(dataset, args.batch_size, True, args.seed + epoch, args.workers)
        model.train()
        if epoch < args.warmup: model.features.eval()
        loss_sum, correct, seen, real_seen = 0., 0, 0, 0
        epoch_start = time.perf_counter()
        for step, (images, targets, sample_weights) in enumerate(train_data):
            images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)
            sample_weights = sample_weights.to(device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(images)
                loss = (loss_fn(logits, targets) * sample_weights).sum() / sample_weights.sum()
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.)
            scaler.step(optimizer); scaler.update(); ema.update_parameters(model)
            loss_sum += float(loss.detach()) * len(targets); seen += len(targets)
            real = sample_weights == 1.
            correct += int(((logits.argmax(1) == targets) & real).sum()); real_seen += int(real.sum())
            if step % 25 == 0:
                save_json(args.output / 'status.json', {'status': 'training', 'epoch': epoch + 1, 'epochs': args.epochs,
                    'batch': step + 1, 'batches': len(train_data), 'loss': loss_sum / seen, 'elapsed_seconds': time.perf_counter() - started})
        scheduler.step()
        raw_logits, val_truth = evaluate(model, val_data, device)
        ema_logits, _ = evaluate(ema.module, val_data, device)
        raw_metrics, ema_metrics = score(val_truth, raw_logits), score(val_truth, ema_logits)
        source, chosen_model, current = ('ema', ema.module, ema_metrics) if ema_metrics['macro_f1'] > raw_metrics['macro_f1'] else ('raw', model, raw_metrics)
        row = {'epoch': epoch + 1, 'train_loss': loss_sum / seen, 'train_aug_accuracy': correct / real_seen,
            'val_accuracy': raw_metrics['accuracy'], 'val_macro_f1': raw_metrics['macro_f1'],
            'ema_val_accuracy': ema_metrics['accuracy'], 'ema_val_macro_f1': ema_metrics['macro_f1'],
            'seconds': time.perf_counter() - epoch_start, 'augmentation_strength': transform.strength,
            'lr_backbone': optimizer.param_groups[0]['lr'], 'selected_source': source}
        history.append(row)
        if current['macro_f1'] > best_score:
            best_score, stale = current['macro_f1'], 0
            save_torch(args.output / 'best.pt', {'format_version': 1, 'name': 'digitra-tid-v6-research',
                'architecture': args.architecture, 'labels': labels, 'image_size': 224,
                'state_dict': {k: v.detach().cpu().clone() for k, v in chosen_model.state_dict().items()},
                'epoch': epoch + 1, 'source': source, 'validation': current, 'manifest_sha256': fingerprint})
        else: stale += 1
        save_json(args.output / 'history.json', history)
        nr = np.random.get_state()
        save_torch(args.output / 'last.pt', {'model': model.state_dict(), 'ema': ema.state_dict(), 'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch + 1, 'epochs': args.epochs,
            'history': history, 'best_score': best_score, 'stale': stale, 'manifest_sha256': fingerprint,
            'python_rng': random.getstate(), 'numpy_rng': torch.from_numpy(nr[1].astype(np.int64)), 'numpy_pos': nr[2],
            'numpy_has_gauss': nr[3], 'numpy_cached': nr[4], 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()})
        print(json.dumps({'stage': 'epoch', **row, 'best_val_macro_f1': best_score}, ensure_ascii=False), flush=True)
        if stale >= args.patience:
            print(json.dumps({'stage': 'early_stop', 'epoch': epoch + 1}), flush=True); break
    if args.skip_final_test:
        save_json(args.output / 'status.json', {'status': 'trained_validation_only', 'best_val_macro_f1': best_score})
    else:
        del ema, optimizer, scaler
        torch.cuda.empty_cache()
        finalize(args, model, frame, labels, history, device, fingerprint)


if __name__ == '__main__':
    main()
