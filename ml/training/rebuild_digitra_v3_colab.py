"""One-command Colab rebuild for the Digitra V3 model package."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

import torch


CONTENT = Path("/content")
OUTER = CONTENT / "ASL_HG_36000_CC_BY_4.zip"
NESTED = CONTENT / "asl_hg_nested"
PROCESSED = CONTENT / "asl_hg_processed"
RAW = CONTENT / "asl_hg_raw"
V1_ZIP = CONTENT / "DIGITRA_LANDMARK_V1_RELEASE_CLEAN.zip"
V1_ROOT = CONTENT / "DIGITRA_LANDMARK_V1"


def run(*parts: str) -> None:
    command = [str(part) for part in parts]
    print("RUN", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def extract_dataset() -> None:
    if not OUTER.exists():
        run(
            "wget", "-q", "--show-progress",
            "https://data.mendeley.com/public-api/zip/j4y5w2c8w9/download/1",
            "-O", str(OUTER),
        )
    if not NESTED.exists():
        NESTED.mkdir()
        with zipfile.ZipFile(OUTER) as archive:
            archive.extractall(NESTED)
    processed_zip = next(NESTED.rglob("ASL_Processed_Images.zip"))
    raw_zip = next(NESTED.rglob("ASL_Raw_Images.zip"))
    if not PROCESSED.exists():
        PROCESSED.mkdir()
        with zipfile.ZipFile(processed_zip) as archive:
            archive.extractall(PROCESSED)
    if not RAW.exists():
        RAW.mkdir()
        with zipfile.ZipFile(raw_zip) as archive:
            archive.extractall(RAW)
    processed_count = sum(1 for _ in PROCESSED.rglob("*.jpg"))
    raw_count = sum(1 for _ in RAW.rglob("*.jpg"))
    if (processed_count, raw_count) != (36000, 36000):
        raise RuntimeError(f"Dataset extraction failed: processed={processed_count}, raw={raw_count}")
    print("DATASET_READY", processed_count, raw_count, flush=True)


def extract_task() -> Path:
    if not V1_ZIP.exists():
        raise FileNotFoundError(V1_ZIP)
    V1_ROOT.mkdir(exist_ok=True)
    task = V1_ROOT / "hand_landmarker.task"
    if not task.exists():
        with zipfile.ZipFile(V1_ZIP) as archive:
            archive.extract("hand_landmarker.task", V1_ROOT)
    return task


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Colab GPU runtime is required")
    print("GPU", torch.cuda.get_device_name(0), flush=True)
    extract_dataset()
    task = extract_task()

    rgb = CONTENT / "DIGITRA_V3_ASL_HG_BASE_DEV1"
    landmark_features = CONTENT / "DIGITRA_V3_LANDMARK_DEV1"
    landmark_model = CONTENT / "DIGITRA_V3_LANDMARK_MODEL_DEV1"
    router = CONTENT / "DIGITRA_V3_HYBRID_ROUTER_DEV1"
    package = CONTENT / "DIGITRA_V3_HYBRID_9415"

    run(
        sys.executable, "-u", CONTENT / "train_digitra_v3_external.py",
        "--processed-root", PROCESSED,
        "--output", rgb,
        "--backbone", "vit_base_patch16_dinov3.lvd1689m",
        "--image-size", "384", "--batch-size", "24", "--workers", "6",
        "--head-epochs", "1", "--full-epochs", "8",
        "--accumulation-steps", "4", "--dev-only",
    )
    run(
        sys.executable, "-u", CONTENT / "extract_digitra_v3_landmarks.py",
        "--images-root", RAW, "--task", task, "--output", landmark_features,
        "--workers", "8",
    )
    run(
        sys.executable, "-u", CONTENT / "train_digitra_v3_landmarks.py",
        "--features", landmark_features / "asl_hg_p1_p8_landmarks_422d.npz",
        "--output", landmark_model, "--epochs", "35",
    )
    run(
        sys.executable, "-u", CONTENT / "build_digitra_v3_router.py",
        "--processed-root", PROCESSED,
        "--features", landmark_features / "asl_hg_p1_p8_landmarks_422d.npz",
        "--rgb-run", rgb, "--landmark-run", landmark_model,
        "--rgb-script", CONTENT / "train_digitra_v3_external.py",
        "--output", router, "--workers", "6",
    )
    run(
        sys.executable, "-u", CONTENT / "package_digitra_v3.py",
        "--rgb-run", rgb, "--landmark-run", landmark_model,
        "--router-run", router, "--task", task,
        "--inference", CONTENT / "inference_digitra_v3.py",
        "--output", package,
    )
    print("REBUILD_COMPLETE", package.with_suffix(".zip"), flush=True)


if __name__ == "__main__":
    main()
