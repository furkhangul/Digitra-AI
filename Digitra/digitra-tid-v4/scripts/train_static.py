from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.config import ARTIFACTS_DIR, FEATURES_DIR, MODELS_DIR, REPORTS_DIR, SEED, ensure_dirs


def evaluate(model, x, y):
    pred = model.predict(x)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "pred": pred,
    }


def main() -> None:
    ensure_dirs()
    data_path = FEATURES_DIR / "static_two_hand.parquet"
    if not data_path.exists():
        raise FileNotFoundError("Önce scripts/extract_landmarks.py çalıştırın")
    frame = pd.read_parquet(data_path)
    feature_columns = [c for c in frame.columns if c.startswith("f")]
    parts = {name: frame[frame.split == name].copy() for name in ("train", "val", "test")}
    if any(part.empty for part in parts.values()):
        raise RuntimeError({name: len(part) for name, part in parts.items()})

    models = {
        "extra_trees": ExtraTreesClassifier(
            n_estimators=700,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            n_jobs=-1,
            random_state=SEED,
        ),
        "rbf_svc": make_pipeline(
            StandardScaler(),
            SVC(C=12.0, gamma="scale", probability=True, class_weight="balanced", random_state=SEED),
        ),
    }
    x_train = parts["train"][feature_columns].to_numpy(np.float32)
    y_train = parts["train"].label.to_numpy()
    x_val = parts["val"][feature_columns].to_numpy(np.float32)
    y_val = parts["val"].label.to_numpy()

    comparison = []
    fitted = {}
    for name, model in models.items():
        started = time.perf_counter()
        model.fit(x_train, y_train)
        metrics = evaluate(model, x_val, y_val)
        comparison.append({
            "model": name,
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
            "fit_seconds": time.perf_counter() - started,
        })
        fitted[name] = model
        print(comparison[-1])

    comparison.sort(key=lambda row: (row["val_macro_f1"], row["val_accuracy"]), reverse=True)
    chosen_name = comparison[0]["model"]
    chosen = fitted[chosen_name]

    x_test = parts["test"][feature_columns].to_numpy(np.float32)
    y_test = parts["test"].label.to_numpy()
    test = evaluate(chosen, x_test, y_test)

    manifest = pd.read_csv(ARTIFACTS_DIR / "manifest.csv", encoding="utf-8-sig")
    test_input_total = int((manifest.split == "test").sum())
    end_to_end_correct = int((test["pred"] == y_test).sum())
    end_to_end_accuracy = end_to_end_correct / max(1, test_input_total)

    report = classification_report(y_test, test["pred"], output_dict=True, zero_division=0)
    labels = sorted(frame.label.unique())
    cm = confusion_matrix(y_test, test["pred"], labels=labels)
    plt.figure(figsize=(15, 13))
    sns.heatmap(cm, cmap="Blues", xticklabels=labels, yticklabels=labels, square=True)
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    joblib.dump(chosen, MODELS_DIR / "static_two_hand.joblib")
    metadata = {
        "name": "digitra-tid-static-two-hand-v4-research",
        "selected_model": chosen_name,
        "labels": labels,
        "feature_columns": feature_columns,
        "feature_count": len(feature_columns),
        "split_protocol": "10-frame pseudo-session blocks; source has no signer IDs",
        "dataset_license": "CC BY-NC-SA 4.0 (non-commercial)",
        "comparison": comparison,
        "test_detected": {
            "samples": len(y_test),
            "accuracy": test["accuracy"],
            "macro_f1": test["macro_f1"],
        },
        "test_end_to_end": {
            "input_samples": test_input_total,
            "correct": end_to_end_correct,
            "accuracy": end_to_end_accuracy,
        },
        "claims": {
            "tid_alphabet": True,
            "signer_independent": False,
            "production_ready": False,
            "commercial_use_allowed": False,
        },
    }
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(comparison).to_csv(REPORTS_DIR / "model_comparison.csv", index=False)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
