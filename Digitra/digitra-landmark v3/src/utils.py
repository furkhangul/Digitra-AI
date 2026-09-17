"""Shared utilities: logging, deterministic seeding, model download, helpers."""
from __future__ import annotations

import logging
import os
import random
import urllib.request
import warnings
from pathlib import Path

import numpy as np

# Keep third-party / framework noise out of the console for a clean session.
os.environ.setdefault("GLOG_minloglevel", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore")
for _n in ("absl", "tensorflow", "mediapipe"):
    try:
        __import__(_n)
    except Exception:
        pass

SEED = 42

DATASET_PATH = (
    r"C:\Users\Furkan\Desktop\Digitra\digitra-landmark v3\ASL_Raw_Images\asl_dataset"
)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    warnings.filterwarnings("ignore")
    logger = logging.getLogger("digitra")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s",
                            datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    return logger


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dirs() -> None:
    for d in (ARTIFACTS_DIR, MODELS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def get_model_path(logger: logging.Logger | None = None) -> Path:
    """Return path to hand_landmarker.task, downloading it if missing."""
    logger = logger or setup_logging()
    model_path = ARTIFACTS_DIR / "hand_landmarker.task"
    if model_path.exists() and model_path.stat().st_size > 1_000_000:
        logger.info("Hand Landmarker model already present: %s", model_path)
        return model_path
    logger.info("Downloading Hand Landmarker model from official source...")
    try:
        urllib.request.urlretrieve(MODEL_URL, model_path)
    except Exception as exc:  # pragma: no cover - network
        logger.error("Automatic download failed: %s", exc)
        raise
    logger.info("Downloaded model -> %s (%.1f MB)",
                model_path, model_path.stat().st_size / 1e6)
    return model_path


def parse_person(label: str, filename: str) -> str:
    """Extract person id (P1..P10) from a filename like P10_A_901.jpg.

    Filename pattern: P<digits>_<label>_<index>.jpg
    """
    import re
    stem = Path(filename).stem
    m = re.match(r"^(P\d+)", stem)
    if m:
        return m.group(1).upper()
    parts = Path(filename).parts
    for p in parts:
        m = re.match(r"^(P\d+)", p)
        if m:
            return m.group(1).upper()
    raise ValueError(f"Cannot parse person from {filename}")
