from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    steps = [
        "download_kaggle.py",
        "audit_dataset.py",
        "extract_landmarks.py",
        "train_static.py",
        "train_image_hog.py",
        "evaluate_fusion.py",
        "train_image_cnn.py",
        "evaluate_robustness.py",
    ]
    for step in steps:
        print(f"\n=== {step} ===", flush=True)
        subprocess.run(
            [sys.executable, str(root / "scripts" / step)],
            cwd=root,
            check=True,
        )


if __name__ == "__main__":
    main()
