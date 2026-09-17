"""Train a participant-disjoint Digitra V3 geometry ensemble.

P1-P7 are used for fitting. P8 is development-only. P9/P10 are absent from
the input feature archive and are never evaluated by this script.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset


SEED = 2027
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class GeometryMLP(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.17),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.12),
            nn.Linear(256, n_out),
        )

    def forward(self, values):
        return self.net(values)


def normalized_adjacency() -> np.ndarray:
    adjacency = np.eye(21, dtype=np.float32)
    for left, right in HAND_EDGES:
        adjacency[left, right] = adjacency[right, left] = 1.0
    degree = adjacency.sum(axis=1)
    return adjacency / np.sqrt(degree[:, None] * degree[None, :])


class GraphGeometryNet(nn.Module):
    def __init__(self, n_out: int):
        super().__init__()
        self.register_buffer("adj", torch.tensor(normalized_adjacency()))
        self.node_enc = nn.Sequential(nn.Linear(6, 128), nn.LayerNorm(128), nn.GELU())
        self.g1 = nn.Sequential(
            nn.Linear(128, 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(0.10)
        )
        self.g2 = nn.Sequential(
            nn.Linear(192, 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(0.10)
        )
        self.attn = nn.Linear(192, 1)
        self.global_branch = nn.Sequential(
            nn.Linear(296, 384), nn.BatchNorm1d(384), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(384, 256), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(192 * 3 + 256, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.17),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.10), nn.Linear(256, n_out),
        )

    def forward(self, values):
        image_points = values[:, :63].reshape(-1, 21, 3)
        world_points = values[:, 63:126].reshape(-1, 21, 3)
        hidden = self.node_enc(torch.cat([image_points, world_points], dim=2))
        hidden = self.g1(torch.einsum("ij,bjd->bid", self.adj, hidden))
        hidden = hidden + self.g2(torch.einsum("ij,bjd->bid", self.adj, hidden))
        weights = torch.softmax(self.attn(hidden), dim=1)
        pooled = torch.cat(
            [hidden.mean(1), hidden.amax(1), (weights * hidden).sum(1)], dim=1
        )
        global_values = self.global_branch(values[:, 126:])
        return self.head(torch.cat([pooled, global_values], dim=1))


def probabilities(model, values: np.ndarray, device: torch.device, batch_size: int = 2048):
    model.eval()
    parts = []
    with torch.inference_mode():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start:start + batch_size]).to(device)
            parts.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(parts)


def train_neural(
    model,
    seed: int,
    train_values: np.ndarray,
    train_labels: np.ndarray,
    dev_values: np.ndarray,
    dev_labels: np.ndarray,
    device: torch.device,
    max_epochs: int,
    name: str,
):
    set_seed(seed)
    model = model.to(device)
    dataset = TensorDataset(
        torch.from_numpy(train_values), torch.from_numpy(train_labels).long()
    )
    loader = DataLoader(
        dataset,
        batch_size=640,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed),
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.025)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.2e-3, weight_decay=2.5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs * len(loader), eta_min=2.0e-5
    )
    best = {"score": (-1.0, -1.0), "epoch": 0, "state": None}
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = 0.0
        seen = 0
        for batch_values, batch_labels in loader:
            batch_values = batch_values.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            batch_values = batch_values + torch.randn_like(batch_values) * 0.014
            batch_values = batch_values.masked_fill(
                torch.rand_like(batch_values) < 0.009, 0.0
            )
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_values), batch_labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            optimizer.step()
            scheduler.step()
            loss_sum += float(loss.detach()) * len(batch_labels)
            seen += len(batch_labels)
        dev_probability = probabilities(model, dev_values, device)
        dev_prediction = dev_probability.argmax(axis=1)
        accuracy = float(accuracy_score(dev_labels, dev_prediction))
        macro_f1 = float(f1_score(dev_labels, dev_prediction, average="macro", zero_division=0))
        record = {
            "epoch": epoch,
            "loss": loss_sum / max(seen, 1),
            "dev_accuracy": accuracy,
            "dev_macro_f1": macro_f1,
        }
        history.append(record)
        score = (macro_f1, accuracy)
        if score > best["score"]:
            best = {
                "score": score,
                "epoch": epoch,
                "state": copy.deepcopy(model.state_dict()),
            }
        if epoch == 1 or epoch % 5 == 0:
            print(name, seed, json.dumps(record), flush=True)
    model.load_state_dict(best["state"])
    model.eval()
    return model, {"epoch": best["epoch"], "score": best["score"], "history": history}


def metric(probability: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    prediction = probability.argmax(axis=1)
    return (
        float(f1_score(labels, prediction, average="macro", zero_division=0)),
        float(accuracy_score(labels, prediction)),
    )


def blend_search(parts: list[np.ndarray], labels: np.ndarray):
    best = {"score": (-1.0, -1.0), "weights": None, "probability": None}
    # Coarse simplex search.
    for a in np.arange(0.0, 1.0001, 0.10):
        for b in np.arange(0.0, 1.0001 - a, 0.10):
            for c in np.arange(0.0, 1.0001 - a - b, 0.10):
                d = 1.0 - a - b - c
                if d < -1e-8:
                    continue
                weights = np.asarray([a, b, c, max(d, 0.0)])
                probability = sum(weight * part for weight, part in zip(weights, parts))
                score = metric(probability, labels)
                if score > best["score"]:
                    best = {"score": score, "weights": weights, "probability": probability}
    # Local refinement around the coarse optimum.
    center = best["weights"]
    for a in np.arange(max(0, center[0] - 0.12), min(1, center[0] + 0.12) + 1e-6, 0.025):
        for b in np.arange(max(0, center[1] - 0.12), min(1, center[1] + 0.12) + 1e-6, 0.025):
            for c in np.arange(max(0, center[2] - 0.12), min(1, center[2] + 0.12) + 1e-6, 0.025):
                d = 1.0 - a - b - c
                if d < 0 or abs(d - center[3]) > 0.18:
                    continue
                weights = np.asarray([a, b, c, d])
                probability = sum(weight * part for weight, part in zip(weights, parts))
                score = metric(probability, labels)
                if score > best["score"]:
                    best = {"score": score, "weights": weights, "probability": probability}
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=35)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite V3 landmark run: {args.output}")
    args.output.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required")

    archive = np.load(args.features, allow_pickle=False)
    values = archive["X"].astype(np.float32)
    labels = archive["y"].astype(np.int64)
    participants = archive["participant"].astype(str)
    sample_ids = archive["sample_id"].astype(str)
    classes = archive["classes"].astype(str).tolist()
    if set(participants) != {f"P{index}" for index in range(1, 9)}:
        raise RuntimeError(f"Unexpected participants: {sorted(set(participants))}")
    train_mask = np.isin(participants, [f"P{index}" for index in range(1, 8)])
    dev_mask = participants == "P8"
    train_values, train_labels = values[train_mask], labels[train_mask]
    dev_values, dev_labels = values[dev_mask], labels[dev_mask]
    dev_sample_ids = sample_ids[dev_mask]
    if set(participants[dev_mask]) != {"P8"}:
        raise RuntimeError("P8 development split contract failed")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_values).astype(np.float32)
    dev_scaled = scaler.transform(dev_values).astype(np.float32)
    joblib.dump(scaler, args.output / "feature_scaler.joblib")

    extra_trees = ExtraTreesClassifier(
        n_estimators=900,
        criterion="entropy",
        max_features=0.48,
        min_samples_leaf=1,
        class_weight="balanced",
        n_jobs=-1,
        random_state=SEED,
    )
    extra_trees.fit(train_values, train_labels)
    joblib.dump(extra_trees, args.output / "extra_trees.joblib", compress=3)
    extra_probability = extra_trees.predict_proba(dev_values)
    print("EXTRA_TREES_DEV", metric(extra_probability, dev_labels), flush=True)

    svc = SVC(
        C=8.0,
        gamma="scale",
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        cache_size=6000,
        random_state=SEED,
    )
    svc.fit(train_scaled, train_labels)
    joblib.dump(svc, args.output / "rbf_svc.joblib", compress=3)
    svc_probability = svc.predict_proba(dev_scaled)
    print("RBF_SVC_DEV", metric(svc_probability, dev_labels), flush=True)

    neural_models = []
    neural_specs = []
    for seed in (2027, 3407):
        model, spec = train_neural(
            GeometryMLP(422, len(classes)), seed,
            train_scaled, train_labels, dev_scaled, dev_labels,
            device, args.epochs, "GEOMETRY",
        )
        neural_models.append(model)
        neural_specs.append(spec)
    geometry_probability = np.mean(
        [probabilities(model, dev_scaled, device) for model in neural_models], axis=0
    )

    graph_model, graph_spec = train_neural(
        GraphGeometryNet(len(classes)), 5521,
        train_scaled, train_labels, dev_scaled, dev_labels,
        device, args.epochs, "GRAPH",
    )
    graph_probability = probabilities(graph_model, dev_scaled, device)

    blend = blend_search(
        [geometry_probability, graph_probability, extra_probability, svc_probability],
        dev_labels,
    )
    blended_probability = blend["probability"]
    base_prediction = blended_probability.argmax(axis=1)
    print(
        "BLEND_DEV",
        {"weights": blend["weights"].tolist(), "macro_f1_accuracy": blend["score"]},
        flush=True,
    )

    candidate_pairs = [
        ("M", "N"), ("M", "T"), ("N", "T"), ("N", "S"), ("S", "T"),
        ("R", "U"), ("U", "V"), ("K", "V"), ("G", "H"), ("P", "Q"),
        ("C", "O"), ("A", "E"), ("D", "I"), ("F", "W"),
    ]
    prediction = base_prediction.copy()
    kept_experts = []
    current_score = (
        float(f1_score(dev_labels, prediction, average="macro", zero_division=0)),
        float(accuracy_score(dev_labels, prediction)),
    )
    expert_models = {}
    for left_name, right_name in candidate_pairs:
        pair = (classes.index(left_name), classes.index(right_name))
        pair_train = np.isin(train_labels, pair)
        expert = SVC(
            C=10.0,
            gamma="scale",
            kernel="rbf",
            class_weight="balanced",
            cache_size=2500,
        )
        expert.fit(train_scaled[pair_train], train_labels[pair_train])
        active = np.isin(prediction, pair)
        if not active.any():
            continue
        candidate = prediction.copy()
        candidate[active] = expert.predict(dev_scaled[active])
        candidate_score = (
            float(f1_score(dev_labels, candidate, average="macro", zero_division=0)),
            float(accuracy_score(dev_labels, candidate)),
        )
        if candidate_score > current_score:
            prediction = candidate
            current_score = candidate_score
            kept_experts.append([left_name, right_name])
            expert_models[f"{left_name}_{right_name}"] = expert
            print("EXPERT_KEPT", left_name, right_name, current_score, flush=True)
    joblib.dump(expert_models, args.output / "pair_experts.joblib", compress=3)

    detected_accuracy = float(accuracy_score(dev_labels, prediction))
    detected_macro_f1 = float(
        f1_score(dev_labels, prediction, average="macro", zero_division=0)
    )
    correct = int((prediction == dev_labels).sum())
    input_total = len(classes) * 100
    end_to_end_accuracy = correct / input_total
    cm = confusion_matrix(dev_labels, prediction, labels=list(range(len(classes))))
    per_class = {
        label: {
            "correct": int(cm[index, index]),
            "detected": int(cm[index].sum()),
            "input_total": 100,
            "end_to_end_accuracy": int(cm[index, index]) / 100,
        }
        for index, label in enumerate(classes)
    }
    confusions = []
    for true_index, true_name in enumerate(classes):
        for pred_index, pred_name in enumerate(classes):
            count = int(cm[true_index, pred_index])
            if true_index != pred_index and count:
                confusions.append({"true": true_name, "pred": pred_name, "count": count})
    confusions.sort(key=lambda row: row["count"], reverse=True)

    torch.save(
        {
            "geometry_states": [model.state_dict() for model in neural_models],
            "graph_state": graph_model.state_dict(),
            "classes": classes,
            "blend_weights": blend["weights"].tolist(),
        },
        args.output / "neural_ensemble.pt",
    )
    metrics = {
        "scope": "ASL-HG P8 development only; 24 static letters",
        "train_participants": [f"P{index}" for index in range(1, 8)],
        "development_participant": "P8",
        "locked_test_participants_read": False,
        "train_detected": int(train_mask.sum()),
        "development_detected": int(dev_mask.sum()),
        "development_input_total": input_total,
        "development_detector_misses": input_total - int(dev_mask.sum()),
        "development_detected_accuracy": detected_accuracy,
        "development_macro_f1": detected_macro_f1,
        "development_end_to_end_accuracy": end_to_end_accuracy,
        "development_errors_end_to_end": input_total - correct,
        "blend_weights": {
            "geometry": float(blend["weights"][0]),
            "graph": float(blend["weights"][1]),
            "extra_trees": float(blend["weights"][2]),
            "rbf_svc": float(blend["weights"][3]),
        },
        "kept_pair_experts": kept_experts,
        "neural_selection": [
            {"seed": seed, "best_epoch": spec["epoch"], "score": spec["score"]}
            for seed, spec in zip((2027, 3407), neural_specs)
        ],
        "graph_selection": {
            "seed": 5521,
            "best_epoch": graph_spec["epoch"],
            "score": graph_spec["score"],
        },
        "per_class": per_class,
        "top_confusions": confusions[:30],
        "claims": {
            "raw_99_proven": False,
            "development_only": True,
            "asl_not_tid": True,
        },
    }
    (args.output / "development_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output / "p8_predictions.npz",
        sample_id=dev_sample_ids,
        true_index=dev_labels,
        prediction=prediction,
        probability=blended_probability,
    )
    print("V3_LANDMARK_DEVELOPMENT", json.dumps({
        "detected_accuracy": detected_accuracy,
        "macro_f1": detected_macro_f1,
        "end_to_end_accuracy": end_to_end_accuracy,
        "errors_end_to_end": input_total - correct,
        "S": per_class["S"], "T": per_class["T"],
        "N": per_class["N"], "U": per_class["U"],
        "top_confusions": confusions[:10],
    }), flush=True)
    print("LOCKED_TEST_P9_P10_OPENED", False, flush=True)


if __name__ == "__main__":
    main()
