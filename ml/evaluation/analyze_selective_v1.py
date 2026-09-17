"""Leakage-safe selective-accuracy audit for Digitra Landmark V1.

The acceptance threshold is selected only on the validation signers. The
locked test signers are evaluated once with that fixed threshold.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


TARGETS = (0.95, 0.98, 0.99, 0.995, 0.999)


def load_recognizer(bundle_dir: Path, device: str):
    spec = importlib.util.spec_from_file_location(
        "digitra_v1_runtime", bundle_dir / "inference.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Digitra inference module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.DigitraRecognizer(
        bundle_dir, device=device, initialize_detector=False
    )


def predict_components(recognizer, raw: np.ndarray, batch_size: int = 2048):
    scaled = recognizer.scaler.transform(raw).astype(np.float32)
    geometry_parts = [[] for _ in recognizer.geometry_models]
    graph_parts = []

    for start in range(0, len(raw), batch_size):
        stop = min(start + batch_size, len(raw))
        tensor = torch.from_numpy(scaled[start:stop]).to(recognizer.device)
        with torch.inference_mode():
            for index, model in enumerate(recognizer.geometry_models):
                geometry_parts[index].append(
                    torch.softmax(model(tensor), dim=1).cpu().numpy()
                )
            graph_parts.append(
                torch.softmax(recognizer.graph_model(tensor), dim=1).cpu().numpy()
            )

    geometry_probabilities = [np.concatenate(parts) for parts in geometry_parts]
    geometry_mean = np.mean(geometry_probabilities, axis=0)
    graph_probability = np.concatenate(graph_parts)

    et_small = recognizer.extra_trees.predict_proba(raw)
    et_probability = np.zeros(
        (len(raw), len(recognizer.classes)), dtype=np.float64
    )
    et_probability[:, np.asarray(recognizer.extra_trees.classes_, dtype=int)] = et_small

    probability = (
        recognizer.geometry_weight * geometry_mean
        + recognizer.extra_trees_weight * et_probability
        + recognizer.graph_weight * graph_probability
    )
    base_prediction = probability.argmax(axis=1)
    final_prediction = base_prediction.copy()
    expert_margin = np.ones(len(raw), dtype=np.float32)
    expert_applied = np.zeros(len(raw), dtype=bool)

    for pair in recognizer.expert_pairs:
        active = np.isin(final_prediction, pair)
        if not active.any():
            continue
        pair_spec = recognizer.expert_specs[pair]
        if pair_spec["mode"] == "raw_subset":
            expert_input = raw[active][:, np.asarray(pair_spec["idx"], dtype=int)]
        elif pair_spec["mode"] == "scaled_all":
            expert_input = scaled[active]
        else:
            raise RuntimeError(f"Unknown expert mode: {pair_spec['mode']}")
        expert = pair_spec["model"]
        final_prediction[active] = expert.predict(expert_input)
        decision = np.asarray(expert.decision_function(expert_input)).reshape(-1)
        expert_margin[active] = np.tanh(np.abs(decision)).astype(np.float32)
        expert_applied[active] = True

    component_predictions = np.column_stack(
        [part.argmax(axis=1) for part in geometry_probabilities]
        + [graph_probability.argmax(axis=1), et_probability.argmax(axis=1)]
    )
    agreement = np.mean(
        component_predictions == final_prediction[:, None], axis=1
    ).astype(np.float32)

    sorted_probability = np.sort(probability, axis=1)
    top1 = sorted_probability[:, -1]
    margin = top1 - sorted_probability[:, -2]
    final_probability = probability[
        np.arange(len(probability)), final_prediction
    ]
    entropy = -np.sum(
        probability * np.log(np.clip(probability, 1e-9, 1.0)), axis=1
    ) / math.log(probability.shape[1])

    confidence_features = np.column_stack(
        [
            top1,
            final_probability,
            margin,
            1.0 - entropy,
            agreement,
            (base_prediction == final_prediction).astype(np.float32),
            expert_margin,
            expert_applied.astype(np.float32),
        ]
    ).astype(np.float32)
    return final_prediction, probability, confidence_features


def choose_threshold(scores, correct, target, minimum_accepted=50):
    candidates = np.unique(scores)
    best = None
    for threshold in candidates:
        accepted = scores >= threshold
        count = int(accepted.sum())
        if count < minimum_accepted:
            continue
        accuracy = float(correct[accepted].mean())
        if accuracy >= target and (best is None or count > best["accepted"]):
            best = {
                "threshold": float(threshold),
                "accepted": count,
                "coverage": float(count / len(scores)),
                "accuracy": accuracy,
            }
    return best


def evaluate_threshold(scores, correct, threshold):
    accepted = scores >= threshold
    count = int(accepted.sum())
    return {
        "accepted": count,
        "coverage": float(count / len(scores)),
        "accuracy": float(correct[accepted].mean()) if count else None,
        "errors": int((~correct[accepted]).sum()) if count else 0,
    }


def per_class_accuracy(labels, prediction, class_names):
    result = {}
    for index, name in enumerate(class_names[:26]):
        mask = labels == index
        result[name] = (
            float((prediction[mask] == labels[mask]).mean()) if mask.any() else None
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data = np.load(args.data, allow_pickle=False)
    classes = data["classes"].astype(str).tolist()
    x_val = data["X_val"].astype(np.float32)
    y_val = data["y_val"].astype(np.int32)
    x_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.int32)
    source_val = data["source_val"].astype(str)
    source_test = data["source_test"].astype(str)

    recognizer = load_recognizer(args.bundle, args.device)
    try:
        val_prediction, _, val_features = predict_components(recognizer, x_val)
        test_prediction, _, test_features = predict_components(recognizer, x_test)
    finally:
        recognizer.close()

    val_correct = val_prediction == y_val
    test_correct = test_prediction == y_test
    calibrator = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.35, max_iter=3000, random_state=2026),
    )
    calibrator.fit(val_features, val_correct.astype(np.int32))
    val_score = calibrator.predict_proba(val_features)[:, 1]
    test_score = calibrator.predict_proba(test_features)[:, 1]

    report = {
        "protocol": {
            "threshold_source": sorted(np.unique(source_val).tolist()),
            "locked_test_signers": sorted(np.unique(source_test).tolist()),
            "test_used_for_threshold": False,
        },
        "baseline": {
            "validation_accuracy": float(accuracy_score(y_val, val_prediction)),
            "validation_macro_f1": float(
                f1_score(y_val, val_prediction, labels=list(range(26)), average="macro", zero_division=0)
            ),
            "locked_test_accuracy": float(accuracy_score(y_test, test_prediction)),
            "locked_test_macro_f1": float(
                f1_score(y_test, test_prediction, labels=list(range(26)), average="macro", zero_division=0)
            ),
        },
        "selective": {},
        "locked_test_per_class": per_class_accuracy(y_test, test_prediction, classes),
    }

    for target in TARGETS:
        selected = choose_threshold(val_score, val_correct, target)
        key = f"target_{target:.3f}"
        if selected is None:
            report["selective"][key] = {"validation": None, "locked_test": None}
            continue
        report["selective"][key] = {
            "validation": selected,
            "locked_test": evaluate_threshold(
                test_score, test_correct, selected["threshold"]
            ),
        }

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
