"""Train and evaluate the leakage-safe Digitra V2 static hybrid model."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


SEED = 2026
STATIC_SOURCE_INDICES = [index for index in range(26) if index not in (9, 25)]
STATIC_NAMES = [chr(ord("A") + index) for index in STATIC_SOURCE_INDICES]
SOURCE_TO_STATIC = {source: target for target, source in enumerate(STATIC_SOURCE_INDICES)}
TARGET_ACCURACIES = (0.95, 0.98, 0.99, 0.995, 0.999)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class StaticDataset(Dataset):
    def __init__(self, rows, transform):
        self.rows = rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["crop_path"]) as opened:
            image = opened.convert("RGB")
        return (
            self.transform(image),
            SOURCE_TO_STATIC[int(row["class_index"])],
            row["sample_id"],
            row["signer"],
        )


def read_manifest(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    usable = [
        row for row in rows
        if row["detected"].lower() == "true"
        and row["status"] == "OK"
        and int(row["class_index"]) in STATIC_SOURCE_INDICES
    ]
    by_split = {
        split: [row for row in usable if row["split"] == split]
        for split in ("train", "val", "test")
    }
    total_by_split = {
        split: sum(
            row["split"] == split
            and int(row["class_index"]) in STATIC_SOURCE_INDICES
            and not row["status"].startswith("DROPPED")
            for row in rows
        )
        for split in ("train", "val", "test")
    }
    signer_sets = {
        split: {row["signer"] for row in split_rows}
        for split, split_rows in by_split.items()
    }
    assert signer_sets["train"].isdisjoint(signer_sets["val"])
    assert signer_sets["train"].isdisjoint(signer_sets["test"])
    assert signer_sets["val"].isdisjoint(signer_sets["test"])
    return rows, by_split, total_by_split, signer_sets


def build_final_train_calibration_split(
    by_split, calibration_fraction, calibration_signers
):
    """Fold development-val signers into final training without opening test.

    Hyperparameters must already have been frozen by a signer-disjoint development
    run.  This helper then reserves deterministic, sample-disjoint calibration
    examples from all non-test signers.  Calibration is used only for probability
    fusion, temperature, and reject thresholds; the final checkpoint can be fixed
    to the requested epoch with ``--fixed-final-checkpoint``.
    """
    candidates = list(by_split["train"]) + list(by_split["val"])
    final_train, calibration = [], []
    for row in candidates:
        if calibration_signers and row["signer"] in calibration_signers:
            calibration.append(row)
        elif calibration_signers:
            final_train.append(row)
        else:
            digest = hashlib.sha256(row["sample_id"].encode("utf-8")).digest()
            fraction = int.from_bytes(digest[:8], "big") / float(2**64)
            if fraction < calibration_fraction:
                calibration.append(row)
            else:
                final_train.append(row)
    train_ids = {row["sample_id"] for row in final_train}
    calibration_ids = {row["sample_id"] for row in calibration}
    test_ids = {row["sample_id"] for row in by_split["test"]}
    assert train_ids.isdisjoint(calibration_ids)
    assert train_ids.isdisjoint(test_ids)
    assert calibration_ids.isdisjoint(test_ids)
    if not final_train or not calibration:
        raise RuntimeError("Final train/calibration split is empty")
    return {"train": final_train, "val": calibration, "test": by_split["test"]}


def build_transforms(image_size, mean, std):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.78, 1.0),
                ratio=(0.9, 1.1),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.RandomAffine(12, translate=(0.07, 0.07), scale=(0.9, 1.1), shear=5)],
                p=0.7,
            ),
            transforms.RandomPerspective(distortion_scale=0.14, p=0.25),
            transforms.RandomApply(
                [transforms.ColorJitter(0.32, 0.32, 0.18, 0.025)], p=0.8
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(5, sigma=(0.1, 1.4))], p=0.22
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(
                p=0.16, scale=(0.01, 0.08), ratio=(0.4, 2.5), value="random"
            ),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(
                image_size + 32,
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_transform, eval_transform


def build_loaders(by_split, train_transform, eval_transform, batch_size, workers):
    datasets = {
        "train": StaticDataset(by_split["train"], train_transform),
        "val": StaticDataset(by_split["val"], eval_transform),
        "test": StaticDataset(by_split["test"], eval_transform),
    }
    counts = Counter(
        SOURCE_TO_STATIC[int(row["class_index"])] for row in by_split["train"]
    )
    weights = [
        1.0 / counts[SOURCE_TO_STATIC[int(row["class_index"])]]
        for row in by_split["train"]
    ]
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            sampler=sampler,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
            drop_last=True,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
        ),
    }
    return datasets, loaders


def autocast_context(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def evaluate(model, loader, device):
    model.eval()
    logits_parts, labels_parts, sample_ids, signers = [], [], [], []
    with torch.inference_mode():
        for images, labels, batch_ids, batch_signers in loader:
            images = images.to(device, non_blocking=True)
            with autocast_context(device):
                logits = model(images)
            logits_parts.append(logits.float().cpu())
            labels_parts.append(labels)
            sample_ids.extend(batch_ids)
            signers.extend(batch_signers)
    logits = torch.cat(logits_parts).numpy()
    labels = torch.cat(labels_parts).numpy()
    prediction = logits.argmax(axis=1)
    return {
        "logits": logits,
        "labels": labels,
        "sample_ids": np.asarray(sample_ids),
        "signers": np.asarray(signers),
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
    }


def train_phase(
    model,
    train_loader,
    val_loader,
    device,
    output_dir,
    phase_name,
    epochs,
    optimizer,
    best_state,
    patience,
):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.045)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[group["lr"] for group in optimizer.param_groups],
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.15,
        div_factor=8.0,
        final_div_factor=100.0,
    )
    stale = 0
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for images, labels, _, _ in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                logits = model(images)
                loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach()) * len(labels)
            total_seen += len(labels)
        val_result = evaluate(model, val_loader, device)
        score = (val_result["macro_f1"], val_result["accuracy"])
        record = {
            "phase": phase_name,
            "epoch": epoch,
            "train_loss": total_loss / max(total_seen, 1),
            "val_accuracy": val_result["accuracy"],
            "val_macro_f1": val_result["macro_f1"],
            "lr": [group["lr"] for group in optimizer.param_groups],
        }
        history.append(record)
        print("EPOCH", json.dumps(record), flush=True)
        if score > best_state["score"]:
            best_state["score"] = score
            best_state["phase"] = phase_name
            best_state["epoch"] = epoch
            torch.save(model.state_dict(), output_dir / "best_image_model.pt")
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            print("EARLY_STOP", phase_name, epoch, flush=True)
            break
    return history


def optimize_temperature(logits, labels):
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


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def load_v1(bundle_dir: Path, device):
    spec = importlib.util.spec_from_file_location("digitra_v1_for_hybrid", bundle_dir / "inference.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("V1 inference module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.DigitraRecognizer(bundle_dir, device=str(device), initialize_detector=False)


def v1_probabilities(recognizer, features, batch_size=2048):
    scaled = recognizer.scaler.transform(features).astype(np.float32)
    geometry_parts, graph_parts = [], []
    for start in range(0, len(features), batch_size):
        stop = min(start + batch_size, len(features))
        tensor = torch.from_numpy(scaled[start:stop]).to(recognizer.device)
        with torch.inference_mode():
            geometry_parts.append(
                np.mean(
                    [
                        torch.softmax(model(tensor), dim=1).cpu().numpy()
                        for model in recognizer.geometry_models
                    ],
                    axis=0,
                )
            )
            graph_parts.append(
                torch.softmax(recognizer.graph_model(tensor), dim=1).cpu().numpy()
            )
    geometry = np.concatenate(geometry_parts)
    graph = np.concatenate(graph_parts)
    et_small = recognizer.extra_trees.predict_proba(features)
    et = np.zeros((len(features), len(recognizer.classes)), dtype=np.float64)
    et[:, np.asarray(recognizer.extra_trees.classes_, dtype=int)] = et_small
    probability = (
        recognizer.geometry_weight * geometry
        + recognizer.graph_weight * graph
        + recognizer.extra_trees_weight * et
    )
    selected = probability[:, STATIC_SOURCE_INDICES]
    return selected / np.clip(selected.sum(axis=1, keepdims=True), 1e-9, None)


def aligned_landmark_probabilities(result, feature_lookup, recognizer):
    features = np.stack([feature_lookup[sample_id] for sample_id in result["sample_ids"]])
    return v1_probabilities(recognizer, features)


def expected_calibration_error(probability, labels, bins=15):
    confidence = probability.max(axis=1)
    prediction = probability.argmax(axis=1)
    correct = prediction == labels
    result = 0.0
    for left, right in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        mask = (confidence > left) & (confidence <= right)
        if mask.any():
            result += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(result)


def brier_score(probability, labels):
    one_hot = np.eye(probability.shape[1], dtype=np.float64)[labels]
    return float(np.mean(np.sum((probability - one_hot) ** 2, axis=1)))


def metrics(probability, labels):
    prediction = probability.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "ece_15": expected_calibration_error(probability, labels),
        "brier": brier_score(probability, labels),
    }


def search_fusion(image_probability, landmark_probability, labels):
    best = None
    for alpha in np.linspace(0.0, 1.0, 101):
        probability = alpha * image_probability + (1.0 - alpha) * landmark_probability
        prediction = probability.argmax(axis=1)
        candidate = (
            float(f1_score(labels, prediction, average="macro", zero_division=0)),
            float(accuracy_score(labels, prediction)),
            -abs(alpha - 0.5),
        )
        if best is None or candidate > best[0]:
            best = (candidate, float(alpha))
    return best[1]


def choose_threshold(confidence, correct, target, minimum_accepted=50):
    best = None
    for threshold in np.unique(confidence):
        accepted = confidence >= threshold
        count = int(accepted.sum())
        if count < minimum_accepted:
            continue
        accuracy = float(correct[accepted].mean())
        if accuracy >= target and (best is None or count > best["accepted"]):
            best = {
                "threshold": float(threshold),
                "accepted": count,
                "coverage": float(count / len(confidence)),
                "accuracy": accuracy,
            }
    return best


def evaluate_threshold(confidence, correct, threshold):
    accepted = confidence >= threshold
    count = int(accepted.sum())
    return {
        "accepted": count,
        "coverage": float(count / len(confidence)),
        "accuracy": float(correct[accepted].mean()) if count else None,
        "errors": int((~correct[accepted]).sum()) if count else 0,
    }


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_onnx(model, image_size, output_path, device):
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
        do_constant_folding=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--v1-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", default="vit_small_patch16_dinov3.lvd1689m")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--head-epochs", type=int, default=2)
    parser.add_argument("--full-epochs", type=int, default=12)
    parser.add_argument("--include-development-val", action="store_true")
    parser.add_argument("--calibration-fraction", type=float, default=0.14)
    parser.add_argument(
        "--calibration-signers",
        default="11",
        help="Comma-separated held-out signer IDs; empty uses sample hashing.",
    )
    parser.add_argument("--fixed-final-checkpoint", action="store_true")
    parser.add_argument("--skip-onnx", action="store_true")
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {args.output}")
    args.output.mkdir(parents=True)
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Digitra V2 accurate training requires a CUDA GPU")
    print("DEVICE", torch.cuda.get_device_name(0), flush=True)

    all_rows, by_split, total_by_split, signer_sets = read_manifest(args.manifest)
    protocol = "signer_disjoint_development"
    if args.include_development_val:
        if not 0.05 <= args.calibration_fraction <= 0.35:
            raise ValueError("calibration-fraction must be between 0.05 and 0.35")
        calibration_signers = {
            value.strip() for value in args.calibration_signers.split(",")
            if value.strip()
        }
        by_split = build_final_train_calibration_split(
            by_split, args.calibration_fraction, calibration_signers
        )
        final_total = {"train": 0, "val": 0, "test": 0}
        for row in all_rows:
            if (
                int(row["class_index"]) not in STATIC_SOURCE_INDICES
                or row["status"].startswith("DROPPED")
            ):
                continue
            if row["split"] == "test":
                final_total["test"] += 1
            elif calibration_signers and row["signer"] in calibration_signers:
                final_total["val"] += 1
            elif calibration_signers:
                final_total["train"] += 1
            else:
                # Sample-hash calibration can only include successfully detected rows;
                # retain the original no-hand counts in train for end-to-end reporting.
                final_total["train"] += 1
        if not calibration_signers:
            final_total["val"] = len(by_split["val"])
            final_total["train"] = (
                total_by_split["train"] + total_by_split["val"] - final_total["val"]
            )
        total_by_split = final_total
        signer_sets = {
            split: {row["signer"] for row in split_rows}
            for split, split_rows in by_split.items()
        }
        if not signer_sets["train"].isdisjoint(signer_sets["test"]):
            raise RuntimeError("Final train signers leaked into locked test")
        if not signer_sets["val"].isdisjoint(signer_sets["test"]):
            raise RuntimeError("Calibration signers leaked into locked test")
        protocol = (
            "fixed_hyperparameters_final_train_with_signer_disjoint_calibration"
            if calibration_signers
            else "fixed_hyperparameters_final_train_with_sample_disjoint_calibration"
        )
    print("DETECTED_SPLITS", {key: len(value) for key, value in by_split.items()}, flush=True)
    print("TOTAL_SPLITS", total_by_split, flush=True)
    print("SIGNERS", {key: sorted(value) for key, value in signer_sets.items()}, flush=True)

    model = timm.create_model(
        args.backbone,
        pretrained=True,
        num_classes=len(STATIC_NAMES),
        img_size=args.image_size,
        drop_rate=0.08,
        drop_path_rate=0.12,
    ).to(device)
    data_config = timm.data.resolve_model_data_config(model)
    mean = tuple(float(value) for value in data_config["mean"])
    std = tuple(float(value) for value in data_config["std"])
    train_transform, eval_transform = build_transforms(args.image_size, mean, std)
    _, loaders = build_loaders(
        by_split, train_transform, eval_transform, args.batch_size, args.workers
    )

    classifier = model.get_classifier()
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in classifier.parameters():
        parameter.requires_grad = True
    best_state = {"score": (-1.0, -1.0), "phase": None, "epoch": None}
    history = []
    head_optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1.2e-3,
        weight_decay=0.015,
    )
    history += train_phase(
        model, loaders["train"], loaders["val"], device, args.output,
        "head", args.head_epochs, head_optimizer, best_state, patience=args.head_epochs + 1,
    )

    for parameter in model.parameters():
        parameter.requires_grad = True
    classifier_ids = {id(parameter) for parameter in classifier.parameters()}
    backbone_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in classifier_ids
    ]
    full_optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": 2.2e-5, "weight_decay": 0.05},
            {"params": list(classifier.parameters()), "lr": 2.5e-4, "weight_decay": 0.015},
        ]
    )
    history += train_phase(
        model, loaders["train"], loaders["val"], device, args.output,
        "full", args.full_epochs, full_optimizer, best_state, patience=4,
    )
    if args.fixed_final_checkpoint:
        torch.save(model.state_dict(), args.output / "best_image_model.pt")
        best_state = {
            "score": None,
            "phase": "full_fixed_final_epoch",
            "epoch": args.full_epochs,
        }
    (args.output / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    model.load_state_dict(torch.load(args.output / "best_image_model.pt", map_location=device))
    model.eval()

    val_result = evaluate(model, loaders["val"], device)
    test_result = evaluate(model, loaders["test"], device)
    image_temperature = optimize_temperature(val_result["logits"], val_result["labels"])
    val_image_probability = softmax(val_result["logits"] / image_temperature)
    test_image_probability = softmax(test_result["logits"] / image_temperature)

    feature_data = np.load(args.features, allow_pickle=False)
    feature_lookup = {
        sample_id: feature
        for sample_id, feature in zip(feature_data["sample_id"].astype(str), feature_data["X"])
    }
    recognizer = load_v1(args.v1_bundle, device)
    try:
        val_landmark_probability = aligned_landmark_probabilities(
            val_result, feature_lookup, recognizer
        )
        test_landmark_probability = aligned_landmark_probabilities(
            test_result, feature_lookup, recognizer
        )
    finally:
        recognizer.close()

    landmark_temperature = optimize_temperature(
        np.log(np.clip(val_landmark_probability, 1e-8, 1.0)), val_result["labels"]
    )
    val_landmark_probability = softmax(
        np.log(np.clip(val_landmark_probability, 1e-8, 1.0)) / landmark_temperature
    )
    test_landmark_probability = softmax(
        np.log(np.clip(test_landmark_probability, 1e-8, 1.0)) / landmark_temperature
    )
    image_alpha = search_fusion(
        val_image_probability, val_landmark_probability, val_result["labels"]
    )
    val_hybrid = image_alpha * val_image_probability + (1.0 - image_alpha) * val_landmark_probability
    test_hybrid = image_alpha * test_image_probability + (1.0 - image_alpha) * test_landmark_probability

    hybrid_temperature = optimize_temperature(
        np.log(np.clip(val_hybrid, 1e-8, 1.0)), val_result["labels"]
    )
    val_hybrid = softmax(np.log(np.clip(val_hybrid, 1e-8, 1.0)) / hybrid_temperature)
    test_hybrid = softmax(np.log(np.clip(test_hybrid, 1e-8, 1.0)) / hybrid_temperature)

    val_prediction = val_hybrid.argmax(axis=1)
    test_prediction = test_hybrid.argmax(axis=1)
    val_correct = val_prediction == val_result["labels"]
    test_correct = test_prediction == test_result["labels"]
    selective = {}
    for target in TARGET_ACCURACIES:
        selected = choose_threshold(val_hybrid.max(axis=1), val_correct, target)
        key = f"target_{target:.3f}"
        selective[key] = {
            "validation": selected,
            "locked_test": (
                evaluate_threshold(test_hybrid.max(axis=1), test_correct, selected["threshold"])
                if selected else None
            ),
        }

    result_metrics = {
        "name": "Digitra Static Hybrid V2",
        "scope": "ASL static fingerspelling letters excluding dynamic J and Z",
        "backbone": args.backbone,
        "data_protocol": protocol,
        "image_size": args.image_size,
        "best_checkpoint": best_state,
        "signers": {key: sorted(value) for key, value in signer_sets.items()},
        "detected_samples": {key: len(value) for key, value in by_split.items()},
        "total_samples": total_by_split,
        "validation": {
            "image": metrics(val_image_probability, val_result["labels"]),
            "landmark": metrics(val_landmark_probability, val_result["labels"]),
            "hybrid": metrics(val_hybrid, val_result["labels"]),
        },
        "locked_test": {
            "image": metrics(test_image_probability, test_result["labels"]),
            "landmark": metrics(test_landmark_probability, test_result["labels"]),
            "hybrid_detected": metrics(test_hybrid, test_result["labels"]),
            "hybrid_end_to_end_accuracy": float(test_correct.sum() / total_by_split["test"]),
            "detector_misses": int(total_by_split["test"] - len(test_correct)),
        },
        "fusion": {
            "image_alpha": image_alpha,
            "landmark_alpha": 1.0 - image_alpha,
        },
        "calibration": {
            "image_temperature": image_temperature,
            "landmark_temperature": landmark_temperature,
            "hybrid_temperature": hybrid_temperature,
        },
        "selective_accuracy": selective,
        "test_used_for_training_or_threshold": False,
    }
    (args.output / "metrics.json").write_text(
        json.dumps(result_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output / "labels.json").write_text(
        json.dumps(STATIC_NAMES, indent=2), encoding="utf-8"
    )
    preprocessing = {
        "input": "RGB hand ROI",
        "size": [args.image_size, args.image_size],
        "mean": mean,
        "std": std,
        "resize": args.image_size + 32,
        "crop": "center",
        "roi_source": "MediaPipe HandLandmarker with 42% square margin",
    }
    (args.output / "preprocessing.json").write_text(
        json.dumps(preprocessing, indent=2), encoding="utf-8"
    )
    torch.save(
        {
            "state_dict": model.state_dict(),
            "backbone": args.backbone,
            "classes": STATIC_NAMES,
            "source_indices": STATIC_SOURCE_INDICES,
            "image_size": args.image_size,
            "mean": mean,
            "std": std,
        },
        args.output / "image_model_v2.pt",
    )

    with (args.output / "locked_test_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["sample_id", "signer", "true", "image", "landmark", "hybrid", "confidence"]
        )
        for index, sample_id in enumerate(test_result["sample_ids"]):
            writer.writerow(
                [
                    sample_id,
                    test_result["signers"][index],
                    STATIC_NAMES[test_result["labels"][index]],
                    STATIC_NAMES[test_image_probability[index].argmax()],
                    STATIC_NAMES[test_landmark_probability[index].argmax()],
                    STATIC_NAMES[test_prediction[index]],
                    float(test_hybrid[index].max()),
                ]
            )

    v1_copy = args.output / "landmark_v1"
    shutil.copytree(args.v1_bundle, v1_copy)
    model_card = f"""# Digitra Static Hybrid V2

## Intended use
ASL parmak alfabesinin 24 statik harfi. J ve Z bu modelin kapsamı dışındadır ve temporal modele yönlendirilir.

## Architecture
`{args.backbone}` RGB hand ROI + Digitra Landmark V1 422D geometry; validation-selected probability fusion.

## Evaluation protocol
Protocol: {protocol}. Train signers: {sorted(signer_sets['train'])}; validation/calibration signers: {sorted(signer_sets['val'])}; locked test signers: {sorted(signer_sets['test'])}. Locked test was not used for checkpoint, fusion, calibration, or thresholds.

## Results
See `metrics.json`. Accuracy is reported separately for detected-hand samples and end-to-end samples including detector misses.

## Limitations
This is ASL fingerspelling, not Turkish Sign Language. J/Z require temporal recognition. A high-confidence reject mode trades coverage for precision; confidence is not raw accuracy.
"""
    (args.output / "model_card.md").write_text(model_card, encoding="utf-8")

    if not args.skip_onnx:
        export_onnx(model, args.image_size, args.output / "image_model_v2.onnx", device)

    manifest = {}
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            manifest[path.relative_to(args.output).as_posix()] = {
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
    (args.output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    archive = Path(shutil.make_archive(str(args.output), "zip", args.output))
    print("V2_STATIC_RELEASE_READY", archive, flush=True)
    print("V2_STATIC_METRICS", json.dumps(result_metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
