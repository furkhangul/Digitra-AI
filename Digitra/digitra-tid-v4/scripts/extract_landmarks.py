from __future__ import annotations

import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import ARTIFACTS_DIR, FEATURES_DIR, ROOT, ensure_dirs
from src.features import N_FEATURES, build_two_hand_features

_LANDMARKER = None


def model_path() -> Path:
    target = ARTIFACTS_DIR / "hand_landmarker.task"
    if target.exists():
        return target
    candidates = [
        ROOT.parent / "digitra-landmark v3" / "artifacts" / "hand_landmarker.task",
        ROOT.parents[1] / "artifacts" / "DIGITRA_TRAINING_KIT_V1_1" / "hand_landmarker.task",
    ]
    for source in candidates:
        if source.exists():
            shutil.copy2(source, target)
            return target
    raise FileNotFoundError("hand_landmarker.task bulunamadı")


def variants(image: np.ndarray):
    yield "original", image
    if max(image.shape[:2]) < 900:
        yield "upscale", cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    yield "clahe", cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def detect_best(landmarker, image: np.ndarray):
    best = None
    best_name = ""
    best_score = (-1, -1.0)
    for name, candidate in variants(image):
        rgb = cv2.cvtColor(candidate, cv2.COLOR_BGR2RGB)
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        count = len(result.hand_landmarks)
        confidence = float(np.mean([
            cats[0].score for cats in result.handedness if cats
        ])) if result.handedness else 0.0
        if (count, confidence) > best_score:
            best, best_name, best_score = result, name, (count, confidence)
        if count >= 2:
            break
    return best, best_name, best_score


def init_worker(asset_path: str) -> None:
    global _LANDMARKER
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=asset_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.18,
        min_hand_presence_confidence=0.18,
        min_tracking_confidence=0.18,
    )
    _LANDMARKER = mp.tasks.vision.HandLandmarker.create_from_options(options)


def process_record(record: dict):
    image = cv2.imread(record["path"])
    if image is None:
        return None, {"path": record["path"], "reason": "unreadable"}
    result, variant, score = detect_best(_LANDMARKER, image)
    hands = [
        np.asarray([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)
        for hand in result.hand_landmarks
    ] if result is not None else []
    if not hands:
        return None, {"path": record["path"], "reason": "no_hand"}
    vector = build_two_hand_features(hands)
    if len(vector) != N_FEATURES:
        raise RuntimeError(f"Feature boyutu {len(vector)} != {N_FEATURES}")
    row = {
        "path": record["path"],
        "label": record["label"],
        "split": record["split"],
        "pseudo_session": record["pseudo_session"],
        "hand_count": len(hands),
        "detector_confidence": score[1],
        "preprocess": variant,
    }
    row.update({f"f{i}": float(value) for i, value in enumerate(vector)})
    return row, None


def main() -> None:
    ensure_dirs()
    manifest_path = ARTIFACTS_DIR / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError("Önce scripts/audit_dataset.py çalıştırın")
    manifest = pd.read_csv(manifest_path, encoding="utf-8-sig")
    output = FEATURES_DIR / "static_two_hand.parquet"
    failures = []
    rows = []

    records = manifest[["path", "label", "split", "pseudo_session"]].to_dict("records")
    with ProcessPoolExecutor(
        max_workers=4,
        initializer=init_worker,
        initargs=(str(model_path()),),
    ) as executor:
        results = executor.map(process_record, records, chunksize=6)
        for row, failure in tqdm(results, total=len(records), desc="İki el landmark"):
            if failure:
                failures.append(failure)
            else:
                rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_parquet(output, index=False)
    failure_path = ARTIFACTS_DIR / "landmark_failures.csv"
    pd.DataFrame(failures, columns=["path", "reason"]).to_csv(failure_path, index=False)
    summary = {
        "input": len(manifest),
        "extracted": len(frame),
        "failed": len(failures),
        "detection_rate": len(frame) / max(1, len(manifest)),
        "two_hand_rate_detected": float((frame.hand_count == 2).mean()),
        "n_features": N_FEATURES,
        "output": str(output),
    }
    (ARTIFACTS_DIR / "extraction_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
