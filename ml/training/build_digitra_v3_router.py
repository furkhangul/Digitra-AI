"""Build the frozen Digitra V3 RGB/geometry router on ASL-HG P8.

The component models are fitted on P1-P7.  This script only uses P8 to
reproduce the already selected, fixed router configuration.  P9/P10 are not
read here.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import joblib
import numpy as np
import timm
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("digitra_v3_training", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--rgb-run", type=Path, required=True)
    parser.add_argument("--landmark-run", type=Path, required=True)
    parser.add_argument("--rgb-script", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite router run: {args.output}")
    args.output.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required")

    dv3 = load_module(args.rgb_script)
    labels = list(dv3.STATIC_LABELS)
    rgb_metrics = json.loads((args.rgb_run / "development_metrics.json").read_text())
    architecture = rgb_metrics["architecture"]
    image_size = int(rgb_metrics["image_size"])
    temperature = float(rgb_metrics["temperature"])

    model = timm.create_model(
        architecture,
        pretrained=False,
        num_classes=len(labels),
        img_size=image_size,
        drop_rate=0.06,
        drop_path_rate=0.14,
    ).to(device)
    model.load_state_dict(
        torch.load(args.rgb_run / "best_model_state.pt", map_location=device)
    )
    config = timm.data.resolve_model_data_config(model)
    _, eval_transform = dv3.build_transforms(
        image_size, tuple(config["mean"]), tuple(config["std"])
    )
    p8_rows = [
        row for row in dv3.discover_rows(args.processed_root)
        if row["participant"] == "P8"
    ]
    if len(p8_rows) != 2400:
        raise RuntimeError(f"Expected 2400 P8 rows, got {len(p8_rows)}")
    loader = DataLoader(
        dv3.StaticDataset(p8_rows, eval_transform),
        batch_size=80,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    result = dv3.evaluate(model, loader, device, tta_flip=True)
    rgb_probability = softmax(np.asarray(result["logits"], dtype=np.float64) / temperature)
    rgb_prediction = rgb_probability.argmax(axis=1)
    true_index = np.asarray(result["labels"], dtype=np.int64)

    key_to_index = {
        (row["participant"].upper(), row["label"].upper(), int(row["sample_number"])): index
        for index, row in enumerate(p8_rows)
    }
    landmark_result = np.load(args.landmark_run / "p8_predictions.npz", allow_pickle=False)
    landmark_prediction = np.full(len(p8_rows), -1, dtype=np.int64)
    landmark_truth = np.full(len(p8_rows), -1, dtype=np.int64)
    landmark_detected = np.zeros(len(p8_rows), dtype=bool)
    for index, sample_id in enumerate(landmark_result["sample_id"].astype(str)):
        parts = sample_id.split("-")
        key = (parts[1].upper(), parts[2].upper(), int(parts[3]))
        image_index = key_to_index[key]
        landmark_prediction[image_index] = int(landmark_result["prediction"][index])
        landmark_truth[image_index] = int(landmark_result["true_index"][index])
        landmark_detected[image_index] = True
    if not np.array_equal(true_index[landmark_detected], landmark_truth[landmark_detected]):
        raise RuntimeError("RGB/landmark P8 alignment failed")

    archive = np.load(args.features, allow_pickle=False)
    values = archive["X"].astype(np.float32)
    targets = archive["y"].astype(np.int64)
    participants = archive["participant"].astype(str)
    sample_ids = archive["sample_id"].astype(str)
    r_index, u_index = labels.index("R"), labels.index("U")
    train_mask = np.isin(participants, [f"P{index}" for index in range(1, 8)])
    pair_mask = train_mask & np.isin(targets, [r_index, u_index])
    specialist = Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(
            C=0.20,
            max_iter=4000,
            class_weight="balanced",
            random_state=2036,
        )),
    ])
    specialist.fit(values[pair_mask], targets[pair_mask])

    p8_values = np.zeros((len(p8_rows), values.shape[1]), dtype=np.float32)
    feature_detected = np.zeros(len(p8_rows), dtype=bool)
    for feature_index in np.flatnonzero(participants == "P8"):
        parts = sample_ids[feature_index].split("-")
        key = (parts[1].upper(), parts[2].upper(), int(parts[3]))
        image_index = key_to_index[key]
        p8_values[image_index] = values[feature_index]
        feature_detected[image_index] = True
    if not np.array_equal(feature_detected, landmark_detected):
        raise RuntimeError("Landmark feature/prediction alignment failed")

    r_column = list(specialist.classes_).index(r_index)
    specialist_r = np.zeros(len(p8_rows), dtype=np.float64)
    specialist_r[landmark_detected] = specialist.predict_proba(
        p8_values[landmark_detected]
    )[:, r_column]
    rgb_pair_r = rgb_probability[:, r_index] / np.clip(
        rgb_probability[:, r_index] + rgb_probability[:, u_index], 1e-12, None
    )
    blended_r = 0.53 * rgb_pair_r + 0.47 * specialist_r
    ru_prediction = np.where(blended_r >= 0.5, r_index, u_index)

    hybrid_prediction = rgb_prediction.copy()
    disagree = landmark_detected & (rgb_prediction != landmark_prediction)
    for index in np.flatnonzero(disagree):
        rgb_name = labels[int(rgb_prediction[index])]
        landmark_name = labels[int(landmark_prediction[index])]
        if rgb_name in {"M", "N"} and landmark_name in {"A", "N", "S", "T"}:
            hybrid_prediction[index] = landmark_prediction[index]
        elif rgb_name == "R" and landmark_name == "U":
            hybrid_prediction[index] = ru_prediction[index]

    conflict = disagree & (rgb_prediction == r_index) & (landmark_prediction == u_index)
    metrics = {
        "scope": "ASL-HG P8 development; fixed V3 router reproduction",
        "samples": int(len(true_index)),
        "accuracy": float(accuracy_score(true_index, hybrid_prediction)),
        "macro_f1": float(f1_score(true_index, hybrid_prediction, average="macro", zero_division=0)),
        "errors": int((true_index != hybrid_prediction).sum()),
        "rgb_accuracy": float(accuracy_score(true_index, rgb_prediction)),
        "landmark_detected": int(landmark_detected.sum()),
        "ru_conflicts": int(conflict.sum()),
        "ru_conflict_errors": int((true_index[conflict] != ru_prediction[conflict]).sum()),
        "locked_test_P9_P10_opened": False,
    }
    router = {
        "classes": labels,
        "rules": [
            "if RGB in {M,N} and landmark in {A,N,S,T}: landmark",
            "if RGB=R and landmark=U: R/U specialist blend",
            "otherwise: RGB",
        ],
        "ru": {
            "name": "logreg",
            "rgb_weight": 0.53,
            "specialist_weight": 0.47,
            "threshold": 0.5,
        },
        "development_only": True,
    }
    joblib.dump(specialist, args.output / "ru_specialist_p1_p7.joblib", compress=3)
    (args.output / "router_config.json").write_text(json.dumps(router, indent=2))
    (args.output / "development_metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez_compressed(
        args.output / "p8_hybrid_predictions.npz",
        true_index=true_index,
        prediction=hybrid_prediction,
        rgb_prediction=rgb_prediction,
        landmark_prediction=landmark_prediction,
        landmark_detected=landmark_detected,
    )
    print("DIGITRA_V3_ROUTER", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
