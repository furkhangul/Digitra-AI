from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score

from src.config import ARTIFACTS_DIR, FEATURES_DIR, MODELS_DIR


def aligned_scores(model, x: np.ndarray, labels: list[str], kind: str) -> np.ndarray:
    raw = model.decision_function(x) if kind == "decision" else model.predict_proba(x)
    probs = softmax(raw, axis=1) if kind == "decision" else raw
    output = np.zeros((len(x), len(labels)), dtype=np.float64)
    for source_index, label in enumerate(model.classes_):
        output[:, labels.index(str(label))] = probs[:, source_index]
    return output


def metric(y_true, scores, labels):
    pred = np.asarray(labels)[np.argmax(scores, axis=1)]
    return float(accuracy_score(y_true, pred)), float(f1_score(y_true, pred, average="macro"))


def main() -> None:
    manifest = pd.read_csv(ARTIFACTS_DIR / "manifest.csv", encoding="utf-8-sig")
    landmarks = pd.read_parquet(FEATURES_DIR / "static_two_hand.parquet")
    packed = np.load(FEATURES_DIR / "static_hog.npz", allow_pickle=True)
    hog_model = joblib.load(MODELS_DIR / "static_hog.joblib")
    landmark_model = joblib.load(MODELS_DIR / "static_two_hand.joblib")
    labels = sorted(manifest.label.unique().tolist())

    hog_scores = aligned_scores(hog_model, packed["x"], labels, "decision")
    score_by_path = {path: score for path, score in zip(manifest.path, hog_scores)}
    feature_columns = [c for c in landmarks.columns if c.startswith("f")]

    results = []
    for split in ("val", "test"):
        part = landmarks[landmarks.split == split]
        x_landmark = part[feature_columns].to_numpy(np.float32)
        lm_scores = aligned_scores(landmark_model, x_landmark, labels, "probability")
        lm_by_path = {path: score for path, score in zip(part.path, lm_scores)}
        all_part = manifest[manifest.split == split]
        image_scores = np.stack([score_by_path[path] for path in all_part.path])
        y = all_part.label.to_numpy()
        split_rows = []
        for alpha in np.linspace(0.0, 1.0, 21):
            scores = image_scores.copy()
            for index, path in enumerate(all_part.path):
                if path in lm_by_path:
                    scores[index] = alpha * image_scores[index] + (1.0 - alpha) * lm_by_path[path]
            acc, macro_f1 = metric(y, scores, labels)
            split_rows.append({"hog_weight": float(alpha), "accuracy": acc, "macro_f1": macro_f1})
        results.append((split, split_rows))

    validation = results[0][1]
    best = max(validation, key=lambda row: (row["macro_f1"], row["accuracy"]))
    test = next(row for row in results[1][1] if row["hog_weight"] == best["hog_weight"])
    report = {
        "scope": "HOG tüm örneklerde; landmark bulunduğunda HOG + iki-el landmark fusion",
        "selection": "Ağırlık validation üzerinde seçildi; testte donduruldu",
        "best_validation": best,
        "locked_test_end_to_end": test,
        "all_validation_weights": validation,
        "recommendation": "use_fusion" if best["hog_weight"] < 1.0 else "hog_only",
    }
    (MODELS_DIR / "fusion_config.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
