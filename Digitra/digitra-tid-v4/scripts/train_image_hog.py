from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.svm import LinearSVC
from tqdm import tqdm

from src.config import ARTIFACTS_DIR, FEATURES_DIR, MODELS_DIR, REPORTS_DIR, SEED, ensure_dirs

WIN_SIZE = (96, 96)
HOG = cv2.HOGDescriptor(WIN_SIZE, (16, 16), (8, 8), (8, 8), 9)


def letterbox_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = min(WIN_SIZE[0] / w, WIN_SIZE[1] / h)
    resized = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))))
    canvas = np.zeros(WIN_SIZE, dtype=np.uint8)
    y = (WIN_SIZE[1] - resized.shape[0]) // 2
    x = (WIN_SIZE[0] - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(canvas)


def main() -> None:
    ensure_dirs()
    manifest = pd.read_csv(ARTIFACTS_DIR / "manifest.csv", encoding="utf-8-sig")
    cache = FEATURES_DIR / "static_hog.npz"
    if cache.exists():
        packed = np.load(cache, allow_pickle=True)
        x = packed["x"]
        labels = packed["labels"]
        splits = packed["splits"]
    else:
        vectors = []
        for path in tqdm(manifest.path, desc="HOG"):
            image = cv2.imread(path)
            if image is None:
                raise RuntimeError(f"Görüntü okunamadı: {path}")
            vectors.append(HOG.compute(letterbox_gray(image)).reshape(-1))
        x = np.asarray(vectors, dtype=np.float32)
        labels = manifest.label.to_numpy()
        splits = manifest.split.to_numpy()
        np.savez_compressed(cache, x=x, labels=labels, splits=splits)

    train = splits == "train"
    val = splits == "val"
    test = splits == "test"
    candidates = []
    fitted = {}
    for c in (0.03, 0.1, 0.3, 1.0):
        model = LinearSVC(C=c, class_weight="balanced", dual="auto", random_state=SEED)
        model.fit(x[train], labels[train])
        pred = model.predict(x[val])
        row = {
            "C": c,
            "val_accuracy": float(accuracy_score(labels[val], pred)),
            "val_macro_f1": float(f1_score(labels[val], pred, average="macro")),
        }
        candidates.append(row)
        fitted[c] = model
        print(row)

    candidates.sort(key=lambda row: (row["val_macro_f1"], row["val_accuracy"]), reverse=True)
    best_c = candidates[0]["C"]
    model = fitted[best_c]
    pred = model.predict(x[test])
    metrics = {
        "name": "digitra-tid-static-hog-v4-research",
        "selected_C": best_c,
        "feature": "OpenCV HOG 96x96 grayscale CLAHE",
        "comparison": candidates,
        "test": {
            "samples": int(test.sum()),
            "accuracy": float(accuracy_score(labels[test], pred)),
            "macro_f1": float(f1_score(labels[test], pred, average="macro")),
        },
        "split_protocol": "10-frame pseudo-session blocks; not signer-independent",
        "dataset_license": "CC BY-NC-SA 4.0 (non-commercial)",
        "production_ready": False,
    }
    joblib.dump(model, MODELS_DIR / "static_hog.joblib")
    (MODELS_DIR / "static_hog_metadata.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report = classification_report(labels[test], pred, output_dict=True, zero_division=0)
    (REPORTS_DIR / "hog_classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
