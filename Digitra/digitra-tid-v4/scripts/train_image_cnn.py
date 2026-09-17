from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.cnn import (
    IMAGE_SIZE,
    SUPPORTED_ARCHITECTURES,
    build_model,
    evaluation_transform,
    set_backbone_trainable,
    training_transform,
    tta_logits,
)
from src.config import ARTIFACTS_DIR, MODELS_DIR, REPORTS_DIR, SEED, ensure_dirs


class TidDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, labels: list[str], transform) -> None:
        self.paths = frame.path.astype(str).tolist()
        indices = {label: index for index, label in enumerate(labels)}
        self.targets = [indices[str(label)] for label in frame.label]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        with Image.open(self.paths[index]) as opened:
            image = opened.convert("RGB")
        return self.transform(image), self.targets[index], self.paths[index]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, batch_size: int, shuffle: bool, workers: int, device) -> DataLoader:
    generator = torch.Generator().manual_seed(SEED)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=generator,
    )


def metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
    }


@torch.inference_mode()
def evaluate(model, loader, device, *, use_tta: bool) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_parts = []
    target_parts = []
    for images, targets, _ in tqdm(loader, desc="Değerlendirme", leave=False):
        images = images.to(device, non_blocking=True)
        logits = tta_logits(model, images) if use_tta else model(images)
        logits_parts.append(logits.float().cpu())
        target_parts.append(targets)
    logits = torch.cat(logits_parts).numpy()
    targets = torch.cat(target_parts).numpy()
    predictions = logits.argmax(axis=1)
    return metrics(targets, predictions), logits, targets, predictions


def select_temperature(logits: np.ndarray, targets: np.ndarray) -> float:
    tensor = torch.from_numpy(logits).double()
    truth = torch.from_numpy(targets).long()
    candidates = np.geomspace(0.35, 4.0, 120)
    losses = [
        float(nn.functional.cross_entropy(tensor / float(value), truth))
        for value in candidates
    ]
    return float(candidates[int(np.argmin(losses))])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=SUPPORTED_ARCHITECTURES, default="mobilenet_v3_large")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=7)
    args = parser.parse_args()

    ensure_dirs()
    seed_everything(SEED)
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Aygıt: {device}")

    manifest = pd.read_csv(ARTIFACTS_DIR / "manifest.csv", encoding="utf-8-sig")
    labels = sorted(str(label) for label in manifest.label.unique())
    parts = {name: manifest[manifest.split == name].copy() for name in ("train", "val", "test")}
    loaders = {
        "train": make_loader(
            TidDataset(parts["train"], labels, training_transform()),
            args.batch_size,
            True,
            args.workers,
            device,
        ),
        "val": make_loader(
            TidDataset(parts["val"], labels, evaluation_transform()),
            args.batch_size,
            False,
            args.workers,
            device,
        ),
        "test": make_loader(
            TidDataset(parts["test"], labels, evaluation_transform()),
            args.batch_size,
            False,
            args.workers,
            device,
        ),
    }

    model = build_model(args.architecture, len(labels), pretrained=True).to(device)
    set_backbone_trainable(model, False)
    optimizer = AdamW(
        [
            {"params": model.features.parameters(), "lr": 8e-5},
            {"params": model.classifier.parameters(), "lr": 8e-4},
        ],
        weight_decay=0.015,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, args.epochs), eta_min=8e-6)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.08)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best = None
    best_state = None
    history = []
    stale = 0
    started = time.perf_counter()
    for epoch in range(args.epochs):
        if epoch == args.warmup_epochs:
            set_backbone_trainable(model, True)
            print("Backbone ince ayara açıldı.")
        model.train()
        if epoch < args.warmup_epochs:
            model.features.eval()
        running_loss = 0.0
        seen = 0
        progress = tqdm(loaders["train"], desc=f"Epoch {epoch + 1}/{args.epochs}")
        for images, targets, _ in progress:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(images)
                loss = loss_fn(logits, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss) * len(targets)
            seen += len(targets)
            progress.set_postfix(loss=f"{running_loss / max(1, seen):.4f}")
        scheduler.step()

        val_metrics, _, _, _ = evaluate(model, loaders["val"], device, use_tta=True)
        row = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(1, seen),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "backbone_trainable": epoch >= args.warmup_epochs,
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_classifier": optimizer.param_groups[1]["lr"],
        }
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))
        score = (row["val_macro_f1"], row["val_accuracy"])
        if best is None or score > best:
            best = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Erken durdurma: {args.patience} epoch gelişme yok.")
                break

    if best_state is None:
        raise RuntimeError("Eğitim sonucu oluşmadı")
    model.load_state_dict(best_state)
    val_metrics, val_logits, val_targets, _ = evaluate(model, loaders["val"], device, use_tta=True)
    temperature = select_temperature(val_logits, val_targets)
    test_metrics, test_logits, test_targets, test_predictions = evaluate(
        model, loaders["test"], device, use_tta=True
    )

    checkpoint_path = MODELS_DIR / f"static_{args.architecture}_robust.pt"
    torch.save(
        {
            "format_version": 1,
            "name": f"digitra-tid-{args.architecture}-robust-v5-research",
            "architecture": args.architecture,
            "state_dict": best_state,
            "labels": labels,
            "image_size": IMAGE_SIZE,
            "tta_scales": [0.82, 1.0, 1.18],
            "tta_mirror": True,
            "temperature": temperature,
        },
        checkpoint_path,
    )
    report = classification_report(
        test_targets,
        test_predictions,
        labels=list(range(len(labels))),
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    metadata = {
        "name": f"digitra-tid-{args.architecture}-robust-v5-research",
        "architecture": args.architecture,
        "device": str(device),
        "labels": labels,
        "augmentation": {
            "horizontal_flip": True,
            "scale": [0.55, 1.25],
            "translate": 0.16,
            "rotation_degrees": 12,
            "lighting_and_camera_jitter": True,
        },
        "inference_tta": {"mirror": True, "scales": [0.82, 1.0, 1.18]},
        "temperature": temperature,
        "best_validation": val_metrics,
        "locked_test": test_metrics,
        "epochs_ran": len(history),
        "train_seconds": time.perf_counter() - started,
        "checkpoint": str(checkpoint_path.resolve()),
        "split_protocol": "10-frame pseudo-session blocks; not signer-independent",
        "dataset_license": "CC BY-NC-SA 4.0 (non-commercial)",
        "production_ready": False,
    }
    metadata_path = MODELS_DIR / f"static_{args.architecture}_robust_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (REPORTS_DIR / f"{args.architecture}_history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / f"{args.architecture}_classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
