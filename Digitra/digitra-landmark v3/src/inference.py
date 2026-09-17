"""Real-time webcam inference using the exact same preprocessing/features
as training. Supports ensemble and single best model, confidence threshold
and temporal smoothing (probability averaging across recent frames).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

import joblib

from utils import ARTIFACTS_DIR, MODELS_DIR, get_model_path, setup_logging
from feature_engineering import compute_features

SMOOTH_WINDOW = 8
DEFAULT_THRESHOLD = 0.55


def load_pipeline():
    logger = setup_logging()
    meta_path = MODELS_DIR / "pipeline_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError("Run the pipeline first (no pipeline_meta.json).")
    meta = json.load(open(meta_path))

    if meta.get("model") == "ensemble":
        members = meta["members"]
        models = []
        for n in members:
            m = joblib.load(MODELS_DIR / f"ensemble_{n}.pkl")
            scp = MODELS_DIR / f"ensemble_{n}_scaler.pkl"
            sc = joblib.load(scp) if scp.exists() else None
            models.append((m, sc))
        classes = meta["label_classes"]

        def predict(feat):
            probas = []
            for m, sc in models:
                x = sc.transform(feat.reshape(1, -1)) if sc else feat.reshape(1, -1)
                probas.append(m.predict_proba(x)[0])
            return np.mean(probas, axis=0)
    else:
        m = joblib.load(MODELS_DIR / "best_model.pkl")
        scp = MODELS_DIR / "scaler.pkl"
        sc = joblib.load(scp) if scp.exists() else None
        le = joblib.load(MODELS_DIR / "label_encoder.pkl")
        classes = list(le.classes_)

        def predict(feat):
            x = sc.transform(feat.reshape(1, -1)) if sc else feat.reshape(1, -1)
            return m.predict_proba(x)[0]
    logger.info("Pipeline loaded: %s | %d classes", meta.get("model"), len(classes))
    return predict, classes


def build_detector(model_path):
    base = mp_python.BaseOptions(model_asset_path=str(model_path))
    opt = vision.HandLandmarkerOptions(
        base_options=base, num_hands=1,
        running_mode=vision.RunningMode.IMAGE,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5)
    return vision.HandLandmarker.create_from_options(opt)


def frame_to_features(result):
    if not result.hand_landmarks:
        return None
    lm = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float32)
    wl = result.hand_world_landmarks
    world = np.array([[p.x, p.y, p.z] for p in wl[0]], dtype=np.float32) if wl else None
    handed = result.handedness[0][0].category_name if result.handedness else "Unknown"
    return compute_features(lm, world, handed)


def run_webcam(threshold: float = DEFAULT_THRESHOLD, window: int = SMOOTH_WINDOW):
    logger = setup_logging()
    predict, classes = load_pipeline()
    detector = build_detector(get_model_path(logger))
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Webcam could not be opened (no camera on this machine?).")
        return
    history = []
    t0 = time.time(); frames = 0
    logger.info("Webcam started. Press ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect(img)
        label, conf = "UNKNOWN", 0.0
        feat = frame_to_features(res)
        if feat is not None:
            proba = predict(feat)
            history.append(proba)
            if len(history) > window:
                history.pop(0)
            avg = np.mean(history, axis=0)
            idx = int(np.argmax(avg))
            conf = float(avg[idx])
            if conf >= threshold:
                label = classes[idx]
        # draw
        cv2.putText(frame, f"{label} ({conf:.2f})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
        if res.hand_landmarks:
            for lm in res.hand_landmarks[0]:
                cx, cy = int(lm.x * frame.shape[1]), int(lm.y * frame.shape[0])
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.imshow("Digitra ASL", frame)
        frames += 1
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    fps = frames / max(1e-6, time.time() - t0)
    logger.info("Avg FPS (incl. detection): %.1f", fps)


def benchmark(threshold: float = DEFAULT_THRESHOLD, n: int = 60):
    logger = setup_logging()
    predict, classes = load_pipeline()
    detector = build_detector(get_model_path(logger))
    # pick a sample image from the dataset
    from utils import DATASET_PATH
    import glob
    sample = glob.glob(str(Path(DATASET_PATH) / "**" / "*.jpg"), recursive=True)
    if not sample:
        logger.error("No sample image for benchmark.")
        return None
    img_path = sample[0]
    image = mp.Image.create_from_file(img_path)
    # warmup
    detector.detect(image)
    t0 = time.time()
    for _ in range(n):
        res = detector.detect(image)
        feat = frame_to_features(res)
        if feat is not None:
            predict(feat)
    dt = (time.time() - t0) / n
    logger.info("Benchmark: avg %.1f ms/frame -> ~%.1f FPS", dt * 1000, 1 / dt)
    return {"avg_time_ms": dt * 1000, "fps": 1 / dt}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true", help="measure speed only")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--window", type=int, default=SMOOTH_WINDOW)
    args = ap.parse_args()
    if args.benchmark:
        benchmark(args.threshold)
    else:
        run_webcam(args.threshold, args.window)
