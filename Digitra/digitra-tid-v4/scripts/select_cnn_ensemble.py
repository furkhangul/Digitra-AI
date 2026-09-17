from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from scripts.evaluate_robustness import VARIANTS, VariantDataset
from src.cnn import TidImageClassifier, tta_logits
from src.config import ARTIFACTS_DIR, MODELS_DIR, REPORTS_DIR


@torch.inference_mode()
def probabilities(classifier: TidImageClassifier, loader: DataLoader):
    chunks = []
    targets = []
    for images, truth, _ in tqdm(loader, desc=classifier.model.__class__.__name__, leave=False):
        images = images.to(classifier.device)
        logits = tta_logits(classifier.model, images, classifier.scales, classifier.mirror)
        chunks.append(torch.softmax(logits / classifier.temperature, dim=1).cpu().numpy())
        targets.append(truth.numpy())
    return np.concatenate(chunks), np.concatenate(targets)


def score(truth: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    prediction = scores.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
    }


def loader(frame: pd.DataFrame, labels: list[str], variant: str, batch_size: int):
    return DataLoader(
        VariantDataset(frame, labels, variant),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mobilenet",
        type=Path,
        default=MODELS_DIR / "static_mobilenet_v3_large_robust.pt",
    )
    parser.add_argument(
        "--efficientnet",
        type=Path,
        default=MODELS_DIR / "static_efficientnet_b0_robust.pt",
    )
    parser.add_argument("--batch-size", type=int, default=24)
    args = parser.parse_args()

    mobile = TidImageClassifier(args.mobilenet)
    efficient = TidImageClassifier(args.efficientnet)
    if mobile.labels != efficient.labels:
        raise RuntimeError("Model sınıf sıraları aynı değil")
    labels = mobile.labels
    manifest = pd.read_csv(ARTIFACTS_DIR / "manifest.csv", encoding="utf-8-sig")
    validation = manifest[manifest.split == "val"].copy()
    test = manifest[manifest.split == "test"].copy()

    val_loader = loader(validation, labels, "clean", args.batch_size)
    mobile_val, truth = probabilities(mobile, val_loader)
    efficient_val, truth_again = probabilities(efficient, val_loader)
    np.testing.assert_array_equal(truth, truth_again)

    candidates = []
    for mobile_weight in np.linspace(0.0, 1.0, 21):
        combined = mobile_weight * mobile_val + (1.0 - mobile_weight) * efficient_val
        candidates.append({"mobilenet_weight": float(mobile_weight), **score(truth, combined)})
    best = max(candidates, key=lambda row: (row["macro_f1"], row["accuracy"]))
    weight = best["mobilenet_weight"]
    combined_validation = weight * mobile_val + (1.0 - weight) * efficient_val
    validation_prediction = combined_validation.argmax(axis=1)
    validation_confidence = combined_validation.max(axis=1)
    threshold_candidates = []
    for threshold in np.linspace(0.20, 0.95, 76):
        accepted = validation_confidence >= threshold
        if not accepted.any():
            continue
        threshold_candidates.append(
            {
                "threshold": float(threshold),
                "coverage": float(accepted.mean()),
                "accepted_accuracy": float((validation_prediction[accepted] == truth[accepted]).mean()),
            }
        )
    eligible = [row for row in threshold_candidates if row["accepted_accuracy"] >= 0.90]
    selective = max(eligible, key=lambda row: row["coverage"]) if eligible else max(
        threshold_candidates, key=lambda row: (row["accepted_accuracy"], row["coverage"])
    )

    rows = []
    for variant in VARIANTS:
        current_loader = loader(test, labels, variant, args.batch_size)
        mobile_scores, target = probabilities(mobile, current_loader)
        efficient_scores, target_again = probabilities(efficient, current_loader)
        np.testing.assert_array_equal(target, target_again)
        combined = weight * mobile_scores + (1.0 - weight) * efficient_scores
        rows.extend(
            [
                {"model": "mobilenet_v3_large", "variant": variant, **score(target, mobile_scores)},
                {"model": "efficientnet_b0", "variant": variant, **score(target, efficient_scores)},
                {"model": "selected_ensemble", "variant": variant, **score(target, combined)},
            ]
        )

    config = {
        "format_version": 1,
        "selection": "weight selected only on validation macro-F1",
        "validation": best,
        "recommended_confidence": selective["threshold"],
        "validation_selective_metrics": selective,
        "members": [
            {"checkpoint": args.mobilenet.name, "weight": weight},
            {"checkpoint": args.efficientnet.name, "weight": 1.0 - weight},
        ],
    }
    config_path = MODELS_DIR / "robust_cnn_ensemble.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    frame = pd.DataFrame(rows)
    frame.to_csv(REPORTS_DIR / "cnn_ensemble_robustness.csv", index=False)
    (REPORTS_DIR / "cnn_ensemble_robustness.json").write_text(
        json.dumps({"selection": config, "results": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(config, ensure_ascii=False, indent=2))
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
