"""Landmark extraction with MediaPipe Hand Landmarker.

Extracts 21 hand landmarks (normalized + world) and handedness for every
image, caches the result to artifacts/landmarks.parquet and logs failures.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

# Quiet third-party C++/TF logs (also effective inside worker subprocesses).
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from tqdm import tqdm

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MAX_DIM = 512  # resize large images before detection (faster, normalized coords unaffected)

from utils import (
    ARTIFACTS_DIR,
    DATASET_PATH,
    ensure_dirs,
    get_model_path,
    parse_person,
    set_seed,
    setup_logging,
)

LANDMARK_DIM = 21 * 3  # x,y,z per landmark


def build_detector(model_path: Path, num_hands: int = 1):
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=num_hands,
        running_mode=vision.RunningMode.IMAGE,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return vision.HandLandmarker.create_from_options(options)


def process_image(detector, image_path: str):
    """Return (landmarks 21x3 or None, world 21x3 or None, handedness str)."""
    try:
        bgr = cv2.imread(image_path)
        if bgr is None:
            return None, None, None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if max(h, w) > MAX_DIM:
            s = MAX_DIM / max(h, w)
            rgb = cv2.resize(rgb, (int(w * s), int(h * s)),
                             interpolation=cv2.INTER_AREA)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    except Exception:
        return None, None, None
    try:
        result = detector.detect(image)
    except Exception:
        return None, None, None

    if not result.hand_landmarks:
        return None, None, None

    lm = result.hand_landmarks[0]
    wl = result.hand_world_landmarks[0] if result.hand_world_landmarks else None
    handed = None
    if result.handedness and result.handedness[0]:
        handed = result.handedness[0][0].category_name

    coords = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
    world = None
    if wl:
        world = np.array([[p.x, p.y, p.z] for p in wl], dtype=np.float32)
    return coords, world, handed


def _worker(files):
    """Process a chunk of files; returns (rows, failed_paths)."""
    from utils import get_model_path, setup_logging
    detector = build_detector(get_model_path(setup_logging()), num_hands=1)
    rows, failed = [], []
    for fp in files:
        p = Path(fp)
        coords, world, handed = process_image(detector, str(fp))
        if coords is None:
            failed.append(str(fp))
            continue
        rows.append({
            "path": str(fp),
            "person": parse_person("", p.name),
            "label": p.parent.name,
            "landmarks": coords.flatten(),
            "world_landmarks": world.flatten() if world is not None else np.full(LANDMARK_DIM, np.nan, dtype=np.float32),
            "handedness": handed or "Unknown",
            "has_hand": True,
        })
    return rows, failed


def run(limit: int | None = None, num_hands: int = 1,
        force: bool = False, n_proc: int | None = None) -> pd.DataFrame:
    logger = setup_logging()
    ensure_dirs()
    set_seed()

    out_path = ARTIFACTS_DIR / "landmarks.parquet"
    total_files = len(list(Path(DATASET_PATH).rglob("*.jpg")))
    if out_path.exists() and not force:
        cached = pd.read_parquet(out_path)
        # cached is valid if it covers the dataset minus expected detection
        # failures (some images may have no detected hand).
        if len(cached) >= total_files - 50:
            logger.info("Cached landmarks found (%d images, %d expected): %s",
                        len(cached), total_files, out_path)
            return cached
        logger.info("Cached landmarks incomplete (%d < %d) -> re-extracting",
                    len(cached), total_files)

    files = sorted(str(p) for p in Path(DATASET_PATH).rglob("*.jpg"))
    if limit:
        files = files[:limit]
    logger.info("Processing %d images from %s", len(files), DATASET_PATH)

    n_proc = n_proc or min(10, (os.cpu_count() or 4))
    if n_proc <= 1:
        results = [_worker(files)]
    else:
        chunks = [files[i::n_proc] for i in range(n_proc)]
        import multiprocessing as mp
        with mp.Pool(n_proc) as pool:
            results = list(tqdm(pool.imap(_worker, chunks),
                                total=n_proc, desc="Workers"))
    rows, failed = [], []
    for r, f in results:
        rows.extend(r)
        failed.extend(f)

    df = pd.DataFrame(rows)
    pd.DataFrame({"path": failed}).to_csv(
        ARTIFACTS_DIR / "failed_images.csv", index=False)
    df.to_parquet(out_path, index=False)

    logger.info("Done | successful=%d failed=%d success_rate=%.4f",
                len(df), len(failed),
                len(df) / max(1, len(df) + len(failed)))
    logger.info("Cached -> %s", out_path)
    logger.info("Failed images -> %s", ARTIFACTS_DIR / "failed_images.csv")
    return df


if __name__ == "__main__":
    run()
