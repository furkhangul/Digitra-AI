"""Train Digitra V2's temporal J/Z/OTHER gate without touching V1 artifacts.

The public SigNN clips provide positive J and Z trajectories.  The OTHER class is
built independently inside every split from static fingerspelling poses, invalid
pose transitions, and time-reversed J/Z trajectories.  Validation chooses every
checkpoint, calibration value, and reject threshold; the locked test is opened
once at the end.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler


SEED = 2026
CLASSES = ["J", "Z", "OTHER"]
SEQ_LEN = 32
FRAME_DIM = 130


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def temporal_resample(sequence: np.ndarray, positions: np.ndarray) -> np.ndarray:
    source = np.arange(len(sequence), dtype=np.float32)
    return np.column_stack(
        [np.interp(positions, source, sequence[:, column]) for column in range(FRAME_DIM)]
    ).astype(np.float32)


def motion_energy(sequence: np.ndarray) -> float:
    # Dynamic layout: 0:63 canonical image, 63:126 canonical world,
    # 126:128 normalized wrist trajectory, 128 log scale, 129 handedness.
    index_tip = sequence[:, 8 * 3 : 8 * 3 + 3]
    pinky_tip = sequence[:, 20 * 3 : 20 * 3 + 3]
    wrist = sequence[:, 126:128]
    velocity = (
        np.linalg.norm(np.diff(index_tip, axis=0), axis=1)
        + np.linalg.norm(np.diff(pinky_tip, axis=0), axis=1)
        + 0.35 * np.linalg.norm(np.diff(wrist, axis=0), axis=1)
    )
    return float(np.mean(velocity))


def static_to_frame(feature: np.ndarray) -> np.ndarray:
    frame = np.zeros(FRAME_DIM, dtype=np.float32)
    frame[:126] = feature[:126]
    frame[129] = feature[420]
    return frame


def smoothstep(values: np.ndarray) -> np.ndarray:
    return values * values * (3.0 - 2.0 * values)


def make_other_sequences(
    positive: np.ndarray,
    static_features: np.ndarray,
    static_labels: np.ndarray,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create split-local negatives; no sample can cross train/val/test."""
    if len(static_features) < 2:
        raise RuntimeError("Need at least two static samples to synthesize OTHER")
    rng = np.random.default_rng(seed)
    output, kinds = [], []
    time = np.linspace(0.0, 1.0, SEQ_LEN, dtype=np.float32)
    type_cycle = ("hold", "transition", "random_motion", "reverse")
    for item_index in range(count):
        kind = type_cycle[item_index % len(type_cycle)]
        if kind == "reverse" and len(positive):
            source = positive[int(rng.integers(len(positive)))]
            sequence = source[::-1].copy()
        else:
            first_index = int(rng.integers(len(static_features)))
            first = static_to_frame(static_features[first_index])
            if kind == "hold":
                sequence = np.repeat(first[None, :], SEQ_LEN, axis=0)
                pose_noise = rng.normal(0.0, 0.0025, (SEQ_LEN, 126)).astype(np.float32)
                pose_noise = np.cumsum(pose_noise, axis=0)
                pose_noise -= pose_noise.mean(axis=0, keepdims=True)
                sequence[:, :126] += pose_noise
                wrist = np.cumsum(
                    rng.normal(0.0, 0.002, (SEQ_LEN, 2)).astype(np.float32), axis=0
                )
                sequence[:, 126:128] = wrist - wrist[0]
            elif kind == "transition":
                candidates = np.flatnonzero(static_labels != static_labels[first_index])
                second_index = int(rng.choice(candidates))
                second = static_to_frame(static_features[second_index])
                blend = smoothstep(time)[:, None]
                sequence = (1.0 - blend) * first + blend * second
                angle = float(rng.uniform(-np.pi, np.pi))
                radius = float(rng.uniform(0.05, 0.20))
                sequence[:, 126] = radius * np.sin(2.0 * np.pi * time + angle)
                sequence[:, 127] = radius * np.sin(3.0 * np.pi * time + 0.4 * angle)
            else:
                sequence = np.repeat(first[None, :], SEQ_LEN, axis=0)
                control = rng.normal(0.0, 0.04, (5, 126)).astype(np.float32)
                control_positions = np.linspace(0.0, 1.0, len(control))
                sequence[:, :126] += np.column_stack(
                    [np.interp(time, control_positions, control[:, dim]) for dim in range(126)]
                )
                turns = float(rng.uniform(0.6, 1.8))
                phase = float(rng.uniform(-np.pi, np.pi))
                radius = float(rng.uniform(0.08, 0.28))
                sequence[:, 126] = radius * np.cos(turns * 2 * np.pi * time + phase)
                sequence[:, 127] = radius * np.sin(turns * 2 * np.pi * time + phase)
        output.append(sequence.astype(np.float32))
        kinds.append(kind)
    return np.stack(output), np.asarray(kinds)


def load_static_split(
    manifest_path: Path, features_path: Path
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    feature_data = np.load(features_path, allow_pickle=False)
    lookup = {
        sample_id: feature
        for sample_id, feature in zip(feature_data["sample_id"].astype(str), feature_data["X"])
    }
    rows_by_split: dict[str, list[dict[str, str]]] = {key: [] for key in ("train", "val", "test")}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["status"] == "OK"
                and row["detected"].lower() == "true"
                and int(row["class_index"]) not in (9, 25)
                and row["sample_id"] in lookup
            ):
                rows_by_split[row["split"]].append(row)
    result = {}
    for split, rows in rows_by_split.items():
        result[split] = (
            np.stack([lookup[row["sample_id"]] for row in rows]).astype(np.float32),
            np.asarray([int(row["class_index"]) for row in rows], dtype=np.int64),
        )
    return result


def build_splits(
    positive_path: Path, manifest_path: Path, static_features_path: Path
) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    positive_data = np.load(positive_path, allow_pickle=False)
    positive_x = positive_data["X"].astype(np.float32)
    positive_y = positive_data["y"].astype(np.int64)
    positive_split = positive_data["split"].astype(str)
    positive_groups = positive_data["group_id"].astype(np.int64)
    if positive_x.shape[1:] != (SEQ_LEN, FRAME_DIM):
        raise RuntimeError(f"Unexpected dynamic input shape: {positive_x.shape}")
    static_by_split = load_static_split(manifest_path, static_features_path)
    output, audit = {}, {}
    for split_index, split in enumerate(("train", "val", "test")):
        mask = positive_split == split
        pos_x, pos_y, groups = positive_x[mask], positive_y[mask], positive_groups[mask]
        static_x, static_y = static_by_split[split]
        other_x, kinds = make_other_sequences(
            pos_x, static_x, static_y, count=len(pos_x), seed=SEED + split_index * 1000
        )
        x = np.concatenate([pos_x, other_x]).astype(np.float32)
        y = np.concatenate([pos_y, np.full(len(other_x), 2, dtype=np.int64)])
        source = np.concatenate(
            [np.asarray([f"positive_group_{value}" for value in groups]), kinds]
        )
        output[split] = {"X": x, "y": y, "source": source}
        audit[split] = {
            "total": int(len(x)),
            "class_counts": {
                CLASSES[index]: int((y == index).sum()) for index in range(len(CLASSES))
            },
            "negative_kinds": dict(Counter(kinds.tolist())),
            "positive_groups": int(len(set(groups.tolist()))),
            "static_source_samples": int(len(static_x)),
        }
    return output, audit


class TemporalDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, augment: bool):
        self.x = x
        self.y = y
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, index: int):
        sequence = self.x[index].copy()
        if self.augment:
            # Monotonic timing jitter preserves the intended trajectory direction.
            increments = np.exp(np.random.normal(0.0, 0.16, SEQ_LEN - 1))
            positions = np.concatenate([[0.0], np.cumsum(increments)])
            positions = positions / positions[-1] * (SEQ_LEN - 1)
            sequence = temporal_resample(sequence, positions.astype(np.float32))
            sequence[:, :126] += np.random.normal(0.0, 0.004, (SEQ_LEN, 126)).astype(np.float32)
            sequence[:, 126:129] *= np.random.uniform(0.82, 1.18)
            if np.random.random() < 0.35:
                drop_count = np.random.randint(1, 5)
                for frame in np.random.choice(np.arange(1, SEQ_LEN), drop_count, replace=False):
                    sequence[frame] = sequence[frame - 1]
            if np.random.random() < 0.5:
                sequence[:, 129] = 1.0 - sequence[:, 129]
        return torch.from_numpy(sequence), int(self.y[index]), index


class ResidualTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        padding = 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 5, padding=padding, dilation=dilation),
            nn.GroupNorm(16, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(16, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class DigitraTemporalV2(nn.Module):
    def __init__(self, mean: np.ndarray, std: np.ndarray):
        super().__init__()
        self.register_buffer("feature_mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("feature_std", torch.tensor(std, dtype=torch.float32))
        self.frame_encoder = nn.Sequential(
            nn.Linear(FRAME_DIM, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.10)
        )
        self.tcn = nn.Sequential(
            ResidualTemporalBlock(256, 1, 0.10),
            ResidualTemporalBlock(256, 2, 0.12),
            ResidualTemporalBlock(256, 4, 0.14),
        )
        self.gru = nn.GRU(
            256, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.12
        )
        self.attention = nn.Sequential(nn.Linear(256, 96), nn.Tanh(), nn.Linear(96, 1))
        self.head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Linear(768, 384),
            nn.GELU(),
            nn.Dropout(0.24),
            nn.Linear(384, len(CLASSES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.feature_mean) / self.feature_std
        x = self.frame_encoder(x)
        x = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        x, _ = self.gru(x)
        attention = torch.softmax(self.attention(x), dim=1)
        attended = torch.sum(attention * x, dim=1)
        pooled = torch.cat([attended, x.mean(dim=1), x.amax(dim=1)], dim=1)
        return self.head(pooled)


def build_loaders(splits: dict, batch_size: int, workers: int):
    datasets = {
        split: TemporalDataset(data["X"], data["y"], augment=split == "train")
        for split, data in splits.items()
    }
    counts = Counter(splits["train"]["y"].tolist())
    weights = [1.0 / counts[int(label)] for label in splits["train"]["y"]]
    sampler = WeightedRandomSampler(
        weights,
        num_samples=max(len(weights) * 2, batch_size * 10),
        replacement=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    return {
        "train": DataLoader(
            datasets["train"], batch_size=batch_size, sampler=sampler,
            num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
            drop_last=True,
        ),
        "val": DataLoader(
            datasets["val"], batch_size=batch_size * 2, shuffle=False,
            num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
        ),
        "test": DataLoader(
            datasets["test"], batch_size=batch_size * 2, shuffle=False,
            num_workers=workers, pin_memory=True, persistent_workers=workers > 0,
        ),
    }


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    logits_parts, labels_parts, indices = [], [], []
    with torch.inference_mode():
        for sequences, labels, batch_indices in loader:
            sequences = sequences.to(device, non_blocking=True)
            with autocast_context(device):
                logits = model(sequences)
            logits_parts.append(logits.float().cpu())
            labels_parts.append(labels)
            indices.extend(batch_indices.tolist())
    return (
        torch.cat(logits_parts).numpy(),
        torch.cat(labels_parts).numpy(),
        np.asarray(indices, dtype=np.int64),
    )


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
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def classification_metrics(probability: np.ndarray, labels: np.ndarray) -> dict:
    prediction = probability.argmax(axis=1)
    matrix = confusion_matrix(labels, prediction, labels=np.arange(len(CLASSES)))
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(f1_score(labels, prediction, average="macro", zero_division=0)),
        "confusion_matrix": matrix.tolist(),
        "per_class_accuracy": {
            CLASSES[index]: float(matrix[index, index] / max(matrix[index].sum(), 1))
            for index in range(len(CLASSES))
        },
    }


def choose_reject_threshold(
    probability: np.ndarray, labels: np.ndarray, target: float, minimum_accepted: int = 30
) -> dict | None:
    prediction = probability.argmax(axis=1)
    confidence = probability.max(axis=1)
    correct = prediction == labels
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
                "coverage": float(count / len(labels)),
                "accuracy": accuracy,
            }
    return best


def apply_reject_threshold(probability: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    prediction = probability.argmax(axis=1)
    accepted = probability.max(axis=1) >= threshold
    correct = prediction == labels
    return {
        "accepted": int(accepted.sum()),
        "coverage": float(accepted.mean()),
        "accuracy": float(correct[accepted].mean()) if accepted.any() else None,
        "errors": int((~correct[accepted]).sum()),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--positive", type=Path, required=True)
    parser.add_argument("--static-manifest", type=Path, required=True)
    parser.add_argument("--static-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {args.output}")
    args.output.mkdir(parents=True)
    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Digitra temporal V2 training requires a CUDA GPU")

    splits, data_audit = build_splits(
        args.positive, args.static_manifest, args.static_features
    )
    train_x = splits["train"]["X"]
    mean = train_x.reshape(-1, FRAME_DIM).mean(axis=0).astype(np.float32)
    std = train_x.reshape(-1, FRAME_DIM).std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-3)
    loaders = build_loaders(splits, args.batch_size, args.workers)
    model = DigitraTemporalV2(mean, std).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.5e-4, weight_decay=0.035)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3.5e-4, epochs=args.epochs,
        steps_per_epoch=len(loaders["train"]), pct_start=0.12,
        div_factor=10.0, final_div_factor=100.0,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.035)
    best_score = (-1.0, -1.0)
    best_epoch, stale, history = 0, 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_seen = 0.0, 0
        for sequences, labels, _ in loaders["train"]:
            sequences = sequences.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                logits = model(sequences)
                loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach()) * len(labels)
            total_seen += len(labels)
        val_logits, val_labels, _ = evaluate(model, loaders["val"], device)
        val_probability = softmax(val_logits)
        val_metrics = classification_metrics(val_probability, val_labels)
        score = (val_metrics["macro_f1"], val_metrics["accuracy"])
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_seen, 1),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        print("DYNAMIC_EPOCH", json.dumps(record), flush=True)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), args.output / "best_temporal_v2.pt")
        else:
            stale += 1
        if epoch >= 24 and stale >= 14:
            print("DYNAMIC_EARLY_STOP", epoch, flush=True)
            break

    (args.output / "training_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    model.load_state_dict(torch.load(args.output / "best_temporal_v2.pt", map_location=device))

    # Validation is used for calibration and reject policy.
    val_logits, val_labels, val_indices = evaluate(model, loaders["val"], device)
    temperature = optimize_temperature(val_logits, val_labels)
    val_probability = softmax(val_logits / temperature)
    policies = {}
    for target in (0.95, 0.98, 0.99, 0.995, 0.999):
        policies[f"target_{target:.3f}"] = choose_reject_threshold(
            val_probability, val_labels, target
        )

    # Locked test is evaluated exactly once, after all choices above are frozen.
    test_logits, test_labels, test_indices = evaluate(model, loaders["test"], device)
    test_probability = softmax(test_logits / temperature)
    selective = {}
    for key, policy in policies.items():
        selective[key] = {
            "validation": policy,
            "locked_test": (
                apply_reject_threshold(test_probability, test_labels, policy["threshold"])
                if policy else None
            ),
        }

    metrics = {
        "name": "Digitra Temporal V2",
        "scope": "J, Z and synthetic OTHER temporal gate",
        "best_epoch": best_epoch,
        "data": data_audit,
        "calibration_temperature": temperature,
        "validation": classification_metrics(val_probability, val_labels),
        "locked_test": classification_metrics(test_probability, test_labels),
        "selective_accuracy": selective,
        "test_used_for_training_or_threshold": False,
        "positive_split_limitation": (
            "SigNN does not publish signer IDs; label-stratified contiguous video-ID "
            "blocks are disjoint, but a signer-disjoint positive split cannot be proven."
        ),
        "other_class_limitation": (
            "OTHER is synthetic and must later be expanded with consented real webcam "
            "non-J/Z motion from deployment conditions."
        ),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output / "labels.json").write_text(json.dumps(CLASSES, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.output / "normalization_v2.npz", mean=mean, std=std,
        sequence_length=np.asarray([SEQ_LEN]), frame_dim=np.asarray([FRAME_DIM])
    )
    torch.save(
        {
            "state_dict": model.state_dict(), "classes": CLASSES,
            "mean": mean, "std": std, "sequence_length": SEQ_LEN,
            "frame_dim": FRAME_DIM, "calibration_temperature": temperature,
            "reject_policies": policies,
        },
        args.output / "temporal_model_v2.pt",
    )
    traced = torch.jit.trace(model.eval(), torch.zeros(1, SEQ_LEN, FRAME_DIM, device=device))
    traced.save(str(args.output / "temporal_model_v2.torchscript"))
    probe = torch.from_numpy(splits["test"]["X"][: min(64, len(splits["test"]["X"]))]).to(device)
    with torch.inference_mode():
        reference_logits = model(probe).float().cpu().numpy()
        traced_logits = traced(probe).float().cpu().numpy()
    parity = {
        "samples": int(len(probe)),
        "max_abs_logit_difference": float(np.max(np.abs(reference_logits - traced_logits))),
        "argmax_matches": int(
            np.sum(reference_logits.argmax(axis=1) == traced_logits.argmax(axis=1))
        ),
    }
    if parity["argmax_matches"] != parity["samples"]:
        raise RuntimeError(f"TorchScript prediction parity failed: {parity}")
    (args.output / "torchscript_parity.json").write_text(
        json.dumps(parity, indent=2), encoding="utf-8"
    )

    with (args.output / "locked_test_predictions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["row", "source", "true", "predicted", "confidence", "motion_energy"])
        for row_index, data_index in enumerate(test_indices):
            sequence = splits["test"]["X"][data_index]
            writer.writerow(
                [
                    row_index,
                    splits["test"]["source"][data_index],
                    CLASSES[test_labels[row_index]],
                    CLASSES[test_probability[row_index].argmax()],
                    float(test_probability[row_index].max()),
                    motion_energy(sequence),
                ]
            )

    model_card = f"""# Digitra Temporal V2\n\n## Intended use\nJ ve Z dinamik ASL parmak harflerini, statik veya ilgisiz hareketlerden ayıran üç sınıflı temporal kapı.\n\n## Architecture\n130D/frame x 32 frame; frame encoder + dilated TCN + bidirectional GRU + attention pooling.\n\n## Evaluation protocol\nCheckpoint, temperature and reject threshold validation üzerinde seçildi. Locked test yalnızca bir kez, seçimler dondurulduktan sonra çalıştırıldı. En iyi epoch: {best_epoch}. Pozitif J/Z örnekleri label-stratified contiguous video-ID bloklarıyla ayrıldı.\n\n## Critical limitations\nPublic SigNN kaynağı signer kimliği yayımlamadığından video-ID-disjoint split'in signer-disjoint olduğu kanıtlanamaz. OTHER sınıfı sentetiktir; üretim öncesinde izinli gerçek kamera negatifleriyle genişletilmelidir. %99.9 evrensel gerçek-dünya doğruluğu garanti edilmez.\n"""
    (args.output / "model_card.md").write_text(model_card, encoding="utf-8")

    manifest = {}
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            manifest[path.relative_to(args.output).as_posix()] = {
                "sha256": sha256(path), "bytes": path.stat().st_size
            }
    (args.output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    archive = Path(shutil.make_archive(str(args.output), "zip", args.output))
    print("V2_DYNAMIC_RELEASE_READY", archive, flush=True)
    print("V2_DYNAMIC_METRICS", json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
