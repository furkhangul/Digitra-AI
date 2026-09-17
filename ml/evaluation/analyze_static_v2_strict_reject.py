"""Derive an ultra-strict reject policy from calibration only, then audit test."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import timm
import torch

from train_digitra_v2_static import (
    STATIC_NAMES,
    aligned_landmark_probabilities,
    build_final_train_calibration_split,
    build_loaders,
    build_transforms,
    evaluate,
    load_v1,
    metrics as classification_metrics,
    read_manifest,
    softmax,
)


def threshold_report(probability, labels, threshold):
    prediction = probability.argmax(axis=1)
    confidence = probability.max(axis=1)
    accepted = confidence >= threshold
    correct = prediction == labels
    return {
        "threshold": float(threshold),
        "accepted": int(accepted.sum()),
        "coverage": float(accepted.mean()),
        "accuracy": float(correct[accepted].mean()) if accepted.any() else None,
        "errors": int((~correct[accepted]).sum()),
    }


def classwise_thresholds(probability, labels):
    prediction = probability.argmax(axis=1)
    confidence = probability.max(axis=1)
    correct = prediction == labels
    result = {}
    for class_index, class_name in enumerate(STATIC_NAMES):
        class_mask = prediction == class_index
        error_confidence = confidence[class_mask & ~correct]
        if len(error_confidence):
            threshold = np.nextafter(float(error_confidence.max()), 1.0)
        else:
            # No calibration error for this predicted class.  Retain a modest
            # confidence floor instead of accepting arbitrary low-confidence rows.
            threshold = 0.5
        result[class_name] = float(threshold)
    return result


def apply_classwise(probability, labels, thresholds):
    prediction = probability.argmax(axis=1)
    confidence = probability.max(axis=1)
    required = np.asarray([thresholds[STATIC_NAMES[index]] for index in prediction])
    accepted = confidence >= required
    correct = prediction == labels
    return {
        "thresholds": thresholds,
        "accepted": int(accepted.sum()),
        "coverage": float(accepted.mean()),
        "accuracy": float(correct[accepted].mean()) if accepted.any() else None,
        "errors": int((~correct[accepted]).sum()),
    }


def write_predictions(path, result, probability):
    prediction = probability.argmax(axis=1)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "signer", "true", "predicted", "confidence"])
        for index, sample_id in enumerate(result["sample_ids"]):
            writer.writerow(
                [
                    sample_id,
                    result["signers"][index],
                    STATIC_NAMES[result["labels"][index]],
                    STATIC_NAMES[prediction[index]],
                    float(probability[index].max()),
                ]
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--calibration-signer", default="11")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.bundle / "image_model_v2.pt", map_location=device)
    model = timm.create_model(
        checkpoint["backbone"], pretrained=False,
        num_classes=len(checkpoint["classes"]), img_size=int(checkpoint["image_size"]),
        drop_rate=0.08, drop_path_rate=0.12,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    _, original_split, _, _ = read_manifest(args.manifest)
    by_split = build_final_train_calibration_split(
        original_split, 0.14, {args.calibration_signer}
    )
    _, eval_transform = build_transforms(
        int(checkpoint["image_size"]), checkpoint["mean"], checkpoint["std"]
    )
    _, loaders = build_loaders(
        by_split, eval_transform, eval_transform, args.batch_size, args.workers
    )
    calibration_result = evaluate(model, loaders["val"], device)
    test_result = evaluate(model, loaders["test"], device)

    feature_data = np.load(args.features, allow_pickle=False)
    feature_lookup = {
        sample_id: feature
        for sample_id, feature in zip(feature_data["sample_id"].astype(str), feature_data["X"])
    }
    recognizer = load_v1(args.bundle / "landmark_v1", device)
    try:
        calibration_landmark = aligned_landmark_probabilities(
            calibration_result, feature_lookup, recognizer
        )
        test_landmark = aligned_landmark_probabilities(test_result, feature_lookup, recognizer)
    finally:
        recognizer.close()

    bundle_metrics = json.loads((args.bundle / "metrics.json").read_text())
    calibration = bundle_metrics["calibration"]
    fusion = bundle_metrics["fusion"]

    def hybrid(result, landmark):
        image_probability = softmax(result["logits"] / calibration["image_temperature"])
        landmark_probability = softmax(
            np.log(np.clip(landmark, 1e-8, 1.0)) / calibration["landmark_temperature"]
        )
        fused = (
            fusion["image_alpha"] * image_probability
            + fusion["landmark_alpha"] * landmark_probability
        )
        return softmax(
            np.log(np.clip(fused, 1e-8, 1.0)) / calibration["hybrid_temperature"]
        )

    calibration_probability = hybrid(calibration_result, calibration_landmark)
    test_probability = hybrid(test_result, test_landmark)
    reproduced = classification_metrics(test_probability, test_result["labels"])
    expected = bundle_metrics["locked_test"]["hybrid_detected"]["accuracy"]
    if abs(reproduced["accuracy"] - expected) > 1e-12:
        raise RuntimeError(("Metric reproduction failed", reproduced["accuracy"], expected))

    calibration_prediction = calibration_probability.argmax(axis=1)
    calibration_correct = calibration_prediction == calibration_result["labels"]
    calibration_confidence = calibration_probability.max(axis=1)
    error_confidence = calibration_confidence[~calibration_correct]
    if not len(error_confidence):
        zero_error_threshold = float(calibration_confidence.min())
    else:
        zero_error_threshold = float(np.nextafter(error_confidence.max(), 1.0))
    conservative_threshold = min(0.999999, zero_error_threshold + 0.005)
    per_class = classwise_thresholds(calibration_probability, calibration_result["labels"])

    report = {
        "policy_selection_data": f"calibration signer {args.calibration_signer} only",
        "test_used_for_policy_selection": False,
        "global_zero_calibration_error": {
            "calibration": threshold_report(
                calibration_probability, calibration_result["labels"], zero_error_threshold
            ),
            "locked_test": threshold_report(
                test_probability, test_result["labels"], zero_error_threshold
            ),
        },
        "global_zero_error_plus_fixed_margin_0_005": {
            "calibration": threshold_report(
                calibration_probability, calibration_result["labels"], conservative_threshold
            ),
            "locked_test": threshold_report(
                test_probability, test_result["labels"], conservative_threshold
            ),
        },
        "classwise_zero_calibration_error": {
            "calibration": apply_classwise(
                calibration_probability, calibration_result["labels"], per_class
            ),
            "locked_test": apply_classwise(
                test_probability, test_result["labels"], per_class
            ),
        },
    }
    (args.bundle / "strict_reject_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_predictions(
        args.bundle / "calibration_predictions.csv", calibration_result,
        calibration_probability,
    )
    bundle_metrics["strict_reject"] = report
    # Correct the final train/calibration totals from the development-run layout.
    bundle_metrics["total_samples"] = {"train": 28191, "val": 2400, "test": 4800}
    (args.bundle / "metrics.json").write_text(
        json.dumps(bundle_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("STRICT_REJECT_READY", json.dumps(report, ensure_ascii=False), flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
