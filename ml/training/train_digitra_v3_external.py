"""Train Digitra V3 on a fresh participant-disjoint ASL-HG benchmark.

The script deliberately keeps ASL-HG participants P9 and P10 locked until the
very end.  Hyperparameters are fixed before the test set is evaluated.  It
trains only the 24 static ASL fingerspelling letters; J and Z remain temporal.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


ImageFile.LOAD_TRUNCATED_IMAGES = False
SEED = 2027
STATIC_LABELS = [chr(ord("A") + index) for index in range(26) if index not in (9, 25)]
LABEL_TO_INDEX = {label: index for index, label in enumerate(STATIC_LABELS)}
PARTICIPANT_RE = re.compile(r"^(P(?:10|[1-9]))_([A-Z]|[0-9])_(\d+)\.jpe?g$", re.I)
TARGET_ACCURACIES = (0.95, 0.98, 0.99, 0.995, 0.999)


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def split_for_participant(participant: str) -> str:
    number = int(participant[1:])
    if number <= 7:
        return "train"
    if number == 8:
        return "calibration"
    return "test"


def image_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def difference_hash(path: Path) -> str:
    with Image.open(path) as opened:
        gray = opened.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    return f"{int(''.join('1' if bit else '0' for bit in bits.ravel()), 2):016x}"


def discover_rows(processed_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(processed_root.rglob("*.jpg")):
        match = PARTICIPANT_RE.match(path.name)
        if not match:
            continue
        participant, label, sample_number = match.groups()
        label = label.upper()
        if label not in LABEL_TO_INDEX:
            continue
        rows.append(
            {
                "path": str(path.resolve()),
                "participant": participant.upper(),
                "label": label,
                "target": str(LABEL_TO_INDEX[label]),
                "sample_number": sample_number,
                "sample_id": hashlib.sha256(
                    f"asl-hg-v1:{participant.upper()}:{label}:{sample_number}".encode("utf-8")
                ).hexdigest()[:24],
                "split": split_for_participant(participant.upper()),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No ASL-HG images found below {processed_root}")
    return rows


def audit_rows(rows: list[dict[str, str]], output: Path) -> dict:
    expected = {
        "train": 7 * len(STATIC_LABELS) * 100,
        "calibration": len(STATIC_LABELS) * 100,
        "test": 2 * len(STATIC_LABELS) * 100,
    }
    split_counts = Counter(row["split"] for row in rows)
    if dict(split_counts) != expected:
        raise RuntimeError(f"Unexpected participant split counts: {split_counts}; expected {expected}")

    participant_sets = {
        split: {row["participant"] for row in rows if row["split"] == split}
        for split in expected
    }
    assert participant_sets["train"].isdisjoint(participant_sets["calibration"])
    assert participant_sets["train"].isdisjoint(participant_sets["test"])
    assert participant_sets["calibration"].isdisjoint(participant_sets["test"])

    class_participant_counts = Counter((row["participant"], row["label"]) for row in rows)
    bad_class_counts = {
        f"{participant}:{label}": count
        for (participant, label), count in class_participant_counts.items()
        if count != 100
    }
    if bad_class_counts:
        raise RuntimeError(f"Expected 100 images per participant/class: {bad_class_counts}")

    # Exact-byte duplicates must never cross splits. dHash collisions are reported
    # as an audit signal but are not automatically deleted because static handshape
    # sequences legitimately contain many nearly identical frames.
    sha_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    dhash_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for index, row in enumerate(rows, start=1):
        path = Path(row["path"])
        sha_groups[image_sha256(path)].append(row)
        dhash_groups[difference_hash(path)].append(row)
        if index % 2000 == 0:
            print("AUDIT_HASHED", index, "of", len(rows), flush=True)

    cross_split_exact = []
    for digest, group in sha_groups.items():
        splits = sorted({item["split"] for item in group})
        if len(splits) > 1:
            cross_split_exact.append(
                {"sha256": digest, "splits": splits, "sample_ids": [item["sample_id"] for item in group]}
            )
    if cross_split_exact:
        raise RuntimeError(f"Exact cross-split duplicates found: {len(cross_split_exact)}")

    cross_split_dhash = []
    for digest, group in dhash_groups.items():
        splits = sorted({item["split"] for item in group})
        if len(splits) > 1:
            cross_split_dhash.append(
                {
                    "dhash": digest,
                    "splits": splits,
                    "count": len(group),
                    "sample_ids": [item["sample_id"] for item in group[:20]],
                }
            )

    audit = {
        "dataset": "ASL-HG v1",
        "source_doi": "10.17632/j4y5w2c8w9.1",
        "license": "CC BY 4.0",
        "scope": "24 static ASL fingerspelling letters; J/Z excluded",
        "rows": len(rows),
        "split_counts": dict(split_counts),
        "participants": {key: sorted(value) for key, value in participant_sets.items()},
        "exact_cross_split_duplicates": 0,
        "identical_dhash_cross_split_groups": len(cross_split_dhash),
        "identical_dhash_cross_split_examples": cross_split_dhash[:100],
        "test_opened_before_final_evaluation": False,
    }
    (output / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


class StaticDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], transform):
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row["path"]) as opened:
            image = opened.convert("RGB")
        return self.transform(image), int(row["target"]), row["sample_id"], row["participant"]


def build_transforms(image_size: int, mean, std):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.72, 1.0),
                ratio=(0.82, 1.18),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.RandomAffine(14, translate=(0.08, 0.08), scale=(0.9, 1.1), shear=4)],
                p=0.72,
            ),
            transforms.RandomPerspective(distortion_scale=0.12, p=0.22),
            transforms.RandomApply([transforms.ColorJitter(0.28, 0.28, 0.12, 0.02)], p=0.7),
            transforms.RandomGrayscale(p=0.035),
            transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 1.1))], p=0.16),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.11, scale=(0.008, 0.05), ratio=(0.5, 2.0), value="random"),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(image_size + 32, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def evaluate(model, loader, device: torch.device, tta_flip: bool = True) -> dict:
    model.eval()
    logits_parts, labels_parts, sample_ids, participants = [], [], [], []
    with torch.inference_mode():
        for images, labels, batch_ids, batch_participants in loader:
            images = images.to(device, non_blocking=True)
            with autocast_context(device):
                logits = model(images)
                if tta_flip:
                    logits = 0.5 * logits + 0.5 * model(torch.flip(images, dims=[3]))
            logits_parts.append(logits.float().cpu())
            labels_parts.append(labels)
            sample_ids.extend(batch_ids)
            participants.extend(batch_participants)
    logits = torch.cat(logits_parts).numpy()
    labels = torch.cat(labels_parts).numpy()
    predictions = logits.argmax(axis=1)
    return {
        "logits": logits,
        "labels": labels,
        "predictions": predictions,
        "sample_ids": np.asarray(sample_ids),
        "participants": np.asarray(participants),
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def train_phase(
    model,
    train_loader,
    calibration_loader,
    device: torch.device,
    output: Path,
    phase: str,
    epochs: int,
    optimizer,
    accumulation_steps: int,
    best: dict,
) -> list[dict]:
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    optimizer_steps = math.ceil(len(train_loader) / accumulation_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[group["lr"] for group in optimizer.param_groups],
        epochs=epochs,
        steps_per_epoch=optimizer_steps,
        pct_start=0.14,
        div_factor=8.0,
        final_div_factor=120.0,
    )
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_seen = 0
        for step, (images, labels, _, _) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with autocast_context(device):
                logits = model(images)
                loss = criterion(logits, labels) / accumulation_steps
            loss.backward()
            should_step = step % accumulation_steps == 0 or step == len(train_loader)
            if should_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            total_loss += float(loss.detach()) * accumulation_steps * len(labels)
            total_seen += len(labels)

        calibration = evaluate(model, calibration_loader, device, tta_flip=True)
        score = (calibration["macro_f1"], calibration["accuracy"])
        record = {
            "phase": phase,
            "epoch": epoch,
            "train_loss": total_loss / max(total_seen, 1),
            "calibration_accuracy": calibration["accuracy"],
            "calibration_macro_f1": calibration["macro_f1"],
            "lr": [group["lr"] for group in optimizer.param_groups],
        }
        history.append(record)
        print("EPOCH", json.dumps(record), flush=True)
        if score > best["score"]:
            best.update({"score": score, "phase": phase, "epoch": epoch})
            torch.save(model.state_dict(), output / "best_model_state.pt")
    return history


def optimize_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    log_temperature = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=80)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = criterion(logits_tensor / temperature, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            value += selected.mean() * abs(float(correct[selected].mean()) - float(confidence[selected].mean()))
    return float(value)


def choose_threshold(confidence: np.ndarray, correct: np.ndarray, target: float, minimum: int = 50):
    order = np.argsort(-confidence)
    ordered_correct = correct[order]
    cumulative = np.cumsum(ordered_correct)
    counts = np.arange(1, len(order) + 1)
    accuracy = cumulative / counts
    valid = np.where((counts >= minimum) & (accuracy >= target))[0]
    if not len(valid):
        return None
    chosen = int(valid[-1])
    threshold = float(confidence[order[chosen]])
    accepted = confidence >= threshold
    return {
        "threshold": threshold,
        "accepted": int(accepted.sum()),
        "coverage": float(accepted.mean()),
        "accuracy": float(correct[accepted].mean()),
        "errors": int((~correct[accepted]).sum()),
    }


def evaluate_threshold(confidence: np.ndarray, correct: np.ndarray, threshold: float) -> dict:
    accepted = confidence >= threshold
    return {
        "threshold": float(threshold),
        "accepted": int(accepted.sum()),
        "coverage": float(accepted.mean()),
        "accuracy": float(correct[accepted].mean()) if accepted.any() else None,
        "errors": int((~correct[accepted]).sum()),
    }


def result_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict:
    predictions = probabilities.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "ece_15": expected_calibration_error(probabilities, labels),
        "errors": int((predictions != labels).sum()),
        "samples": int(len(labels)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", default="vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--head-epochs", type=int, default=1)
    parser.add_argument("--full-epochs", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument(
        "--dev-only",
        action="store_true",
        help="Stop after P8 calibration; never instantiate or evaluate the locked P9/P10 test loader.",
    )
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing V3 run: {args.output}")
    args.output.mkdir(parents=True)
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Digitra V3 training requires CUDA")
    print("DEVICE", torch.cuda.get_device_name(0), flush=True)

    rows = discover_rows(args.processed_root)
    audit = audit_rows(rows, args.output)
    by_split = {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "calibration", "test")
    }
    with (args.output / "dataset_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    model = timm.create_model(
        args.backbone,
        pretrained=True,
        num_classes=len(STATIC_LABELS),
        img_size=args.image_size,
        drop_rate=0.06,
        drop_path_rate=0.14,
    ).to(device)
    config = timm.data.resolve_model_data_config(model)
    mean = tuple(float(value) for value in config["mean"])
    std = tuple(float(value) for value in config["std"])
    train_transform, eval_transform = build_transforms(args.image_size, mean, std)
    datasets = {
        "train": StaticDataset(by_split["train"], train_transform),
        "calibration": StaticDataset(by_split["calibration"], eval_transform),
        "test": StaticDataset(by_split["test"], eval_transform),
    }
    loaders = {
        "train": DataLoader(
            datasets["train"], batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
            drop_last=True, generator=torch.Generator().manual_seed(SEED),
        ),
        "calibration": DataLoader(
            datasets["calibration"], batch_size=args.batch_size * 2, shuffle=False,
            num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
        ),
        "test": DataLoader(
            datasets["test"], batch_size=args.batch_size * 2, shuffle=False,
            num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
        ),
    }

    classifier = model.get_classifier()
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in classifier.parameters():
        parameter.requires_grad = True
    best = {"score": (-1.0, -1.0), "phase": None, "epoch": None}
    history: list[dict] = []
    head_optimizer = torch.optim.AdamW(classifier.parameters(), lr=1.1e-3, weight_decay=0.012)
    history += train_phase(
        model, loaders["train"], loaders["calibration"], device, args.output,
        "head", args.head_epochs, head_optimizer, args.accumulation_steps, best,
    )

    for parameter in model.parameters():
        parameter.requires_grad = True
    classifier_ids = {id(parameter) for parameter in classifier.parameters()}
    backbone_parameters = [parameter for parameter in model.parameters() if id(parameter) not in classifier_ids]
    full_optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": 2.0e-5, "weight_decay": 0.05},
            {"params": classifier.parameters(), "lr": 2.2e-4, "weight_decay": 0.012},
        ]
    )
    history += train_phase(
        model, loaders["train"], loaders["calibration"], device, args.output,
        "full", args.full_epochs, full_optimizer, args.accumulation_steps, best,
    )
    (args.output / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    model.load_state_dict(torch.load(args.output / "best_model_state.pt", map_location=device))
    model.eval()
    calibration = evaluate(model, loaders["calibration"], device, tta_flip=True)

    if args.dev_only:
        temperature = optimize_temperature(calibration["logits"], calibration["labels"])
        calibration_probability = softmax(calibration["logits"] / temperature)
        dev_metrics = {
            "name": "Digitra Static V3 External — development run",
            "scope": "24 static ASL fingerspelling letters; J/Z excluded",
            "architecture": args.backbone,
            "image_size": args.image_size,
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "protocol": {
                "train_participants": audit["participants"]["train"],
                "calibration_participants": audit["participants"]["calibration"],
                "locked_test_participants": audit["participants"]["test"],
                "locked_test_opened": False,
            },
            "best_checkpoint": best,
            "temperature": temperature,
            "calibration": result_metrics(calibration_probability, calibration["labels"]),
            "claims": {
                "raw_99_9_proven": False,
                "development_only": True,
                "asl_not_tid": True,
            },
        }
        (args.output / "development_metrics.json").write_text(
            json.dumps(dev_metrics, indent=2), encoding="utf-8"
        )
        print("V3_DEVELOPMENT_ONLY", json.dumps(dev_metrics["calibration"]), flush=True)
        print("LOCKED_TEST_P9_P10_OPENED", False, flush=True)
        print("V3_OUTPUT", args.output, flush=True)
        return

    # The locked test is opened exactly once, after checkpoint selection is complete.
    print("OPENING_FRESH_LOCKED_TEST_P9_P10", flush=True)
    locked_test = evaluate(model, loaders["test"], device, tta_flip=True)
    temperature = optimize_temperature(calibration["logits"], calibration["labels"])
    calibration_probability = softmax(calibration["logits"] / temperature)
    test_probability = softmax(locked_test["logits"] / temperature)
    calibration_prediction = calibration_probability.argmax(axis=1)
    test_prediction = test_probability.argmax(axis=1)
    calibration_correct = calibration_prediction == calibration["labels"]
    test_correct = test_prediction == locked_test["labels"]

    selective = {}
    for target in TARGET_ACCURACIES:
        selected = choose_threshold(calibration_probability.max(axis=1), calibration_correct, target)
        selective[f"target_{target:.3f}"] = {
            "calibration": selected,
            "locked_test": (
                evaluate_threshold(test_probability.max(axis=1), test_correct, selected["threshold"])
                if selected else None
            ),
        }

    cm = confusion_matrix(locked_test["labels"], test_prediction, labels=list(range(len(STATIC_LABELS))))
    per_class = {
        label: {
            "correct": int(cm[index, index]),
            "total": int(cm[index].sum()),
            "accuracy": float(cm[index, index] / max(cm[index].sum(), 1)),
        }
        for index, label in enumerate(STATIC_LABELS)
    }
    per_participant = {}
    for participant in sorted(set(locked_test["participants"])):
        selected = locked_test["participants"] == participant
        per_participant[participant] = {
            "accuracy": float(accuracy_score(locked_test["labels"][selected], test_prediction[selected])),
            "samples": int(selected.sum()),
            "errors": int((locked_test["labels"][selected] != test_prediction[selected]).sum()),
        }

    metrics = {
        "name": "Digitra Static V3 External",
        "scope": "24 static ASL fingerspelling letters; J/Z excluded",
        "architecture": args.backbone,
        "image_size": args.image_size,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "protocol": {
            "train_participants": audit["participants"]["train"],
            "calibration_participants": audit["participants"]["calibration"],
            "fresh_locked_test_participants": audit["participants"]["test"],
            "checkpoint_selected_on": "P8 calibration only",
            "test_used_for_checkpoint_or_hyperparameters": False,
        },
        "best_checkpoint": best,
        "temperature": temperature,
        "calibration": result_metrics(calibration_probability, calibration["labels"]),
        "fresh_locked_test": result_metrics(test_probability, locked_test["labels"]),
        "fresh_locked_test_per_participant": per_participant,
        "fresh_locked_test_per_class": per_class,
        "fresh_locked_test_confusion_matrix": cm.tolist(),
        "raw_99_9_target_passed": bool(float(test_correct.mean()) >= 0.999),
        "selective_accuracy": selective,
        "claims": {
            "universal_99_9_guarantee": False,
            "dataset_specific_locked_test_only": True,
            "asl_not_tid": True,
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output / "labels.json").write_text(json.dumps(STATIC_LABELS, indent=2), encoding="utf-8")
    (args.output / "preprocessing.json").write_text(
        json.dumps(
            {
                "input": "RGB MediaPipe hand crop",
                "size": [args.image_size, args.image_size],
                "resize": args.image_size + 32,
                "crop": "center",
                "mean": mean,
                "std": std,
                "tta": "mean of original and horizontal flip logits",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "backbone": args.backbone,
            "classes": STATIC_LABELS,
            "image_size": args.image_size,
            "mean": mean,
            "std": std,
            "temperature": temperature,
        },
        args.output / "digitra_static_v3.pt",
    )

    with (args.output / "fresh_locked_test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "participant", "true", "prediction", "confidence", "correct"])
        for index, sample_id in enumerate(locked_test["sample_ids"]):
            writer.writerow(
                [
                    sample_id,
                    locked_test["participants"][index],
                    STATIC_LABELS[int(locked_test["labels"][index])],
                    STATIC_LABELS[int(test_prediction[index])],
                    float(test_probability[index].max()),
                    bool(test_correct[index]),
                ]
            )

    model_card = "# Digitra Static V3 External\n\n"
    model_card += "ASL-HG v1 (CC BY 4.0) üzerinde katılımcı-bazlı değerlendirme. "
    model_card += "P1-P7 eğitim, P8 kalibrasyon, daha önce açılmamış P9-P10 kilitli testtir. "
    model_card += "J/Z temporal modele aittir. Sonuç yalnız bu kilitli test dağılımı için geçerlidir; evrensel %99,9 garantisi değildir.\n"
    (args.output / "model_card.md").write_text(model_card, encoding="utf-8")

    print("V3_FRESH_LOCKED_TEST", json.dumps(metrics["fresh_locked_test"]), flush=True)
    print("V3_RAW_99_9_TARGET_PASSED", metrics["raw_99_9_target_passed"], flush=True)
    print("V3_OUTPUT", args.output, flush=True)


if __name__ == "__main__":
    main()
