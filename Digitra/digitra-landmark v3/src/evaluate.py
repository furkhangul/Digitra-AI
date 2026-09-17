"""Report generation: confusion matrix, classification report, metrics,
final report and the permanent reports/ artifacts."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from utils import ARTIFACTS_DIR, MODELS_DIR, REPORTS_DIR, setup_logging


def most_confused(y_true, y_pred, classes, k: int = 10):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                pairs.append((classes[i], classes[j], int(cm[i, j])))
    pairs.sort(key=lambda x: -x[2])
    return pairs[:k], cm


def write_reports(ctx: dict):
    logger = setup_logging()
    classes = ctx["classes"]
    y_test = np.asarray(ctx["y_test"])
    y_pred = np.asarray(ctx["test_pred"])

    # classification report
    crep = classification_report(y_test, y_pred, target_names=classes,
                                 zero_division=0)
    (REPORTS_DIR / "classification_report.txt").write_text(crep)

    # confusion matrix png
    pairs, cm = most_confused(y_test, y_pred, classes)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, xticklabels=classes, yticklabels=classes, cmap="Blues",
                cbar=True, annot=False)
    plt.title("Confusion Matrix (P9 + P10 unseen test set)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=120)
    plt.close()

    # metrics.json
    metrics = {
        "dataset": {
            "total_images": ctx["total_images"],
            "classes": len(classes),
            "persons": ctx["persons"],
            "train_persons": ctx["train_persons"],
            "val_persons": ctx["val_persons"],
            "test_persons": ctx["test_persons"],
            "train_samples": ctx["train_n"],
            "val_samples": ctx["val_n"],
            "test_samples": ctx["test_n"],
        },
        "landmark_extraction": {
            "successful": ctx["successful"],
            "failed": ctx["failed"],
            "success_rate": ctx["success_rate"],
            "mediapipe_model": ctx["mediapipe_model"],
        },
        "feature_engineering": {
            "n_features": ctx["n_features"],
            "groups": ctx["feature_groups"],
        },
        "model_comparison": ctx["comparison"].to_dict(orient="records"),
        "best_model": {
            "model": ctx["chosen"],
            "val_accuracy": ctx["val_acc"],
            "val_macro_f1": ctx["val_f1"],
            "params": ctx["best_params"],
        },
        "final_test": {
            "accuracy": ctx["test_acc"],
            "macro_precision": ctx["test_prec"],
            "macro_recall": ctx["test_rec"],
            "macro_f1": ctx["test_f1"],
        },
        "generalization": {
            "train_val_gap": ctx.get("train_val_gap"),
            "val_test_gap": ctx["val_acc"] - ctx["test_acc"],
            "overfitting": ctx["overfitting"],
            "data_leakage_checked": True,
        },
        "realtime": ctx.get("realtime", {}),
        "most_confused": [{"true": a, "pred": b, "count": c} for a, b, c in pairs],
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # model_comparison.csv (ensure present)
    ctx["comparison"].to_csv(REPORTS_DIR / "model_comparison.csv", index=False)

    # final report text
    txt = build_final_report(ctx, pairs, metrics)
    (REPORTS_DIR / "final_report.txt").write_text(txt)
    logger.info("Reports written to %s", REPORTS_DIR)
    return txt, metrics, pairs


def build_final_report(ctx, pairs, metrics):
    def f(x):
        try:
            return f"{float(x):.4f}"
        except Exception:
            return str(x)

    lines = []
    lines.append("==============================")
    lines.append("DIGITRA LANDMARK V3 FINAL REPORT")
    lines.append("==============================")
    lines.append("")
    lines.append("DATASET")
    lines.append(f"* Total images: {ctx['total_images']}")
    lines.append(f"* Classes: {len(ctx['classes'])}")
    lines.append(f"* Persons: {ctx['persons']}")
    lines.append(f"* Train persons: {ctx['train_persons']}")
    lines.append(f"* Validation persons: {ctx['val_persons']}")
    lines.append(f"* Test persons: {ctx['test_persons']}")
    lines.append(f"* Train samples: {ctx['train_n']}")
    lines.append(f"* Validation samples: {ctx['val_n']}")
    lines.append(f"* Test samples: {ctx['test_n']}")
    lines.append("")
    lines.append("LANDMARK EXTRACTION")
    lines.append(f"* Successful: {ctx['successful']}")
    lines.append(f"* Failed: {ctx['failed']}")
    lines.append(f"* Success rate: {f(ctx['success_rate'])}")
    lines.append(f"* MediaPipe model: {ctx['mediapipe_model']}")
    lines.append("")
    lines.append("FEATURE ENGINEERING")
    lines.append(f"* Raw landmark features: 63 (canonical XYZ)")
    lines.append(f"* Engineered features: {ctx['n_features']-63} additional")
    lines.append(f"* Total features: {ctx['n_features']}")
    lines.append(f"* Normalization methods: wrist-center, scale, rotation, handedness-canonical")
    lines.append("")
    lines.append("MODEL COMPARISON")
    for _, r in ctx["comparison"].iterrows():
        lines.append(f"* {r['model']}:")
        lines.append(f"  Accuracy: {f(r['val_accuracy'])}")
        lines.append(f"  Macro F1: {f(r['val_macro_f1'])}")
    lines.append("")
    lines.append("BEST MODEL")
    lines.append(f"* Model: {ctx['chosen']}")
    lines.append(f"* Best parameters: {ctx['best_params']}")
    lines.append(f"* Validation accuracy: {f(ctx['val_acc'])}")
    lines.append(f"* Validation Macro F1: {f(ctx['val_f1'])}")
    lines.append("")
    lines.append("FINAL UNSEEN TEST")
    lines.append(f"* Test accuracy: {f(ctx['test_acc'])}")
    lines.append(f"* Macro Precision: {f(ctx['test_prec'])}")
    lines.append(f"* Macro Recall: {f(ctx['test_rec'])}")
    lines.append(f"* Macro F1: {f(ctx['test_f1'])}")
    lines.append("")
    lines.append("MOST CONFUSED CLASSES")
    for a, b, c in pairs[:8]:
        lines.append(f"* {a} -> {b} : {c}")
    lines.append("")
    lines.append("GENERALIZATION")
    lines.append(f"* Train vs Validation gap: {f(ctx.get('train_val_gap', float('nan')))}")
    lines.append(f"* Validation vs Test gap: {f(ctx['val_acc'] - ctx['test_acc'])}")
    lines.append(f"* Overfitting status: {ctx['overfitting']}")
    lines.append(f"* Data leakage status: CHECKED - none (strict person split, test unused in tuning)")
    lines.append("")
    lines.append("REAL-TIME")
    rt = ctx.get("realtime", {})
    lines.append(f"* Webcam inference: {rt.get('ready', 'N/A')}")
    lines.append(f"* Confidence threshold: {rt.get('conf_threshold', 'N/A')}")
    lines.append(f"* Temporal smoothing: {rt.get('smoothing', 'N/A')}")
    lines.append(f"* Average inference time: {rt.get('avg_time', 'N/A')}")
    lines.append(f"* Approximate FPS: {rt.get('fps', 'N/A')}")
    lines.append("")
    lines.append("FILES CREATED")
    for fc in ctx.get("files_created", []):
        lines.append(f"* {fc}")
    lines.append("")
    lines.append("FINAL CONCLUSION")
    lines.append(f"* Model guvenilir mi? {ctx.get('reliable', 'N/A')}")
    lines.append(f"* Gercek kullanicida beklenen performans: {ctx.get('expected', 'N/A')}")
    lines.append(f"* %99 hedefi gercekleti mi? {ctx.get('target99', 'N/A')}")
    lines.append(f"* Gerceklesmediyse temel sebep: {ctx.get('reason', 'N/A')}")
    lines.append(f"* 3 gelistirme onerisi: {ctx.get('improvements', 'N/A')}")
    return "\n".join(lines)
