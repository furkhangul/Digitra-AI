"""Evaluate the frozen Digitra V3 hybrid once on locked ASL-HG P9/P10.

No threshold, model, rule, or checkpoint is selected in this script.  Every
decision comes from the pre-test frozen manifest and router configuration.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import timm
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader

import train_digitra_v3_external as dv3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument("--landmarks", type=Path, required=True)
    parser.add_argument("--rgb-run", type=Path, required=True)
    parser.add_argument("--landmark-run", type=Path, required=True)
    parser.add_argument("--router-run", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--opened-marker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite locked evaluation: {args.output}")
    marker = json.loads(args.opened_marker.read_text(encoding="utf-8"))
    if marker.get("stage") != "landmarks_extracted":
        raise RuntimeError(f"Unexpected locked-test stage: {marker.get('stage')}")
    frozen = json.loads(args.frozen_manifest.read_text(encoding="utf-8"))
    router = json.loads((args.router_run / "router_config.json").read_text(encoding="utf-8"))
    if sha256(args.router_run / "router_config.json") != frozen["files"]["router_config.json"]:
        raise RuntimeError("Router changed after protocol freeze")
    if sha256(args.rgb_run / "best_model_state.pt") != frozen["files"]["best_model_state.pt"]:
        raise RuntimeError("RGB checkpoint changed after protocol freeze")
    args.output.mkdir(parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for locked evaluation")
    development = json.loads((args.rgb_run / "development_metrics.json").read_text(encoding="utf-8"))
    backbone = development["architecture"]
    image_size = int(development["image_size"])
    model = timm.create_model(
        backbone, pretrained=False, num_classes=len(dv3.STATIC_LABELS),
        img_size=image_size, drop_rate=0.06, drop_path_rate=0.14,
    ).to(device)
    model.load_state_dict(torch.load(args.rgb_run / "best_model_state.pt", map_location=device))
    model.eval()
    config = timm.data.resolve_model_data_config(model)
    _, eval_transform = dv3.build_transforms(
        image_size, tuple(config["mean"]), tuple(config["std"])
    )

    rows = dv3.discover_rows(args.processed_root)
    test_rows = [row for row in rows if row["participant"] in {"P9", "P10"}]
    if len(test_rows) != 4800:
        raise RuntimeError(f"Expected 4800 locked rows, got {len(test_rows)}")
    loader = DataLoader(
        dv3.StaticDataset(test_rows, eval_transform),
        batch_size=80, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
    )
    rgb = dv3.evaluate(model, loader, device, tta_flip=True)
    true_index = rgb["labels"].astype(np.int64)
    rgb_probability = softmax(rgb["logits"] / float(development["temperature"]))
    rgb_prediction = rgb_probability.argmax(axis=1)

    key_to_index = {}
    for index, row in enumerate(test_rows):
        key = (row["participant"], row["label"], int(row["sample_number"]))
        if key in key_to_index:
            raise RuntimeError(f"Duplicate locked RGB key: {key}")
        key_to_index[key] = index

    archive = np.load(args.landmarks, allow_pickle=False)
    landmark_probability = np.zeros_like(rgb_probability)
    landmark_prediction = np.full(len(test_rows), -1, dtype=np.int64)
    landmark_true = np.full(len(test_rows), -1, dtype=np.int64)
    detected = np.zeros(len(test_rows), dtype=bool)
    X = archive["X"].astype(np.float32)

    extra = joblib.load(args.landmark_run / "extra_trees.joblib")
    extra_probability = extra.predict_proba(X)
    extra_classes = np.asarray(extra.classes_, dtype=np.int64)
    detected_prediction = extra_classes[extra_probability.argmax(axis=1)]
    scaler = joblib.load(args.landmark_run / "feature_scaler.joblib")
    scaled = scaler.transform(X).astype(np.float32)
    pair_experts = joblib.load(args.landmark_run / "pair_experts.joblib")
    for pair_name, expert in pair_experts.items():
        pair = np.asarray(expert.classes_, dtype=np.int64)
        active = np.isin(detected_prediction, pair)
        if active.any():
            detected_prediction[active] = expert.predict(scaled[active])

    ru_model = joblib.load(args.router_run / "ru_specialist_p1_p7.joblib")
    ru_classes = list(ru_model.classes_)
    r_index = dv3.STATIC_LABELS.index("R")
    u_index = dv3.STATIC_LABELS.index("U")
    ru_pR_detected = ru_model.predict_proba(X)[:, ru_classes.index(r_index)]
    ru_pR = np.zeros(len(test_rows), dtype=np.float64)

    for lm_index, sample_id in enumerate(archive["sample_id"].astype(str)):
        parts = sample_id.split("-")
        key = (parts[1].upper(), parts[2].upper(), int(parts[3]))
        image_index = key_to_index.get(key)
        if image_index is None:
            raise RuntimeError(f"Locked landmark row absent from RGB split: {key}")
        # ExtraTrees uses all 24 classes in sorted integer order.
        landmark_probability[image_index, extra_classes] = extra_probability[lm_index]
        landmark_prediction[image_index] = int(detected_prediction[lm_index])
        landmark_true[image_index] = int(archive["y"][lm_index])
        ru_pR[image_index] = float(ru_pR_detected[lm_index])
        detected[image_index] = True
    if not np.array_equal(true_index[detected], landmark_true[detected]):
        raise RuntimeError("Locked RGB/landmark label alignment failed")

    rgb_pair_pR = rgb_probability[:, r_index] / np.clip(
        rgb_probability[:, r_index] + rgb_probability[:, u_index], 1e-12, None
    )
    ru_config = router["ru"]
    blended_pR = (
        float(ru_config["rgb_weight"]) * rgb_pair_pR
        + (1.0 - float(ru_config["rgb_weight"])) * ru_pR
    )
    ru_prediction = np.where(blended_pR >= float(ru_config["threshold"]), r_index, u_index)

    hybrid_prediction = rgb_prediction.copy()
    disagreement = detected & (rgb_prediction != landmark_prediction)
    for index in np.flatnonzero(disagreement):
        rgb_name = dv3.STATIC_LABELS[int(rgb_prediction[index])]
        landmark_name = dv3.STATIC_LABELS[int(landmark_prediction[index])]
        if rgb_name in {"M", "N"} and landmark_name in {"A", "N", "S", "T"}:
            hybrid_prediction[index] = landmark_prediction[index]
        elif rgb_name == "R" and landmark_name == "U":
            hybrid_prediction[index] = ru_prediction[index]

    correct = hybrid_prediction == true_index
    cm = confusion_matrix(true_index, hybrid_prediction, labels=list(range(len(dv3.STATIC_LABELS))))
    per_class = {
        label: {
            "correct": int(cm[index, index]),
            "total": int(cm[index].sum()),
            "accuracy": float(cm[index, index] / max(cm[index].sum(), 1)),
        }
        for index, label in enumerate(dv3.STATIC_LABELS)
    }
    participants = np.asarray(rgb["participants"]).astype(str)
    per_participant = {}
    for participant in ("P9", "P10"):
        selected = participants == participant
        per_participant[participant] = {
            "accuracy": float(accuracy_score(true_index[selected], hybrid_prediction[selected])),
            "samples": int(selected.sum()),
            "errors": int((true_index[selected] != hybrid_prediction[selected]).sum()),
        }

    metrics = {
        "name": "Digitra V3 frozen hybrid — fresh locked test",
        "scope": "24 static ASL fingerspelling letters; J/Z excluded as temporal",
        "protocol": {
            "train_participants": [f"P{i}" for i in range(1, 8)],
            "development_participant": "P8",
            "fresh_locked_test_participants": ["P9", "P10"],
            "models_and_router_frozen_before_test": True,
            "test_used_for_selection_or_tuning": False,
        },
        "fresh_locked_test": {
            "accuracy": float(accuracy_score(true_index, hybrid_prediction)),
            "macro_f1": float(f1_score(true_index, hybrid_prediction, average="macro", zero_division=0)),
            "errors": int((~correct).sum()),
            "samples": int(len(true_index)),
        },
        "branches": {
            "rgb_accuracy": float(accuracy_score(true_index, rgb_prediction)),
            "landmark_end_to_end_accuracy": float((detected & (landmark_prediction == true_index)).mean()),
            "landmark_detection_rate": float(detected.mean()),
            "disagreements": int(disagreement.sum()),
        },
        "fresh_locked_test_per_participant": per_participant,
        "fresh_locked_test_per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "raw_99_target_passed": bool(correct.mean() >= 0.99),
        "raw_99_9_target_passed": bool(correct.mean() >= 0.999),
        "claims": {
            "dataset_specific_locked_test_only": True,
            "universal_accuracy_guarantee": False,
            "asl_not_tid": True,
        },
    }
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (args.output / "labels.json").write_text(json.dumps(dv3.STATIC_LABELS, indent=2), encoding="utf-8")
    with (args.output / "fresh_locked_test_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "sample_id", "participant", "true", "prediction", "correct",
            "rgb_prediction", "landmark_prediction", "landmark_detected",
        ])
        for index, row in enumerate(test_rows):
            writer.writerow([
                f"ASLHG-{row['participant']}-{row['label']}-{int(row['sample_number']):04d}",
                row["participant"], dv3.STATIC_LABELS[int(true_index[index])],
                dv3.STATIC_LABELS[int(hybrid_prediction[index])], bool(correct[index]),
                dv3.STATIC_LABELS[int(rgb_prediction[index])],
                None if landmark_prediction[index] < 0 else dv3.STATIC_LABELS[int(landmark_prediction[index])],
                bool(detected[index]),
            ])

    marker["stage"] = "evaluation_completed"
    marker["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    marker["result"] = metrics["fresh_locked_test"]
    args.opened_marker.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    print("V3_FRESH_LOCKED_HYBRID_TEST", json.dumps(metrics["fresh_locked_test"]), flush=True)
    print("V3_RAW_99_TARGET_PASSED", metrics["raw_99_target_passed"], flush=True)
    print("V3_OUTPUT", args.output, flush=True)


if __name__ == "__main__":
    main()
