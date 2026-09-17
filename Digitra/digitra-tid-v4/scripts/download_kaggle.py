from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kagglehub

from src.config import DATA_DIR, KAGGLE_DATASET, RAW_DIR, ensure_dirs


def main() -> None:
    ensure_dirs()
    os.environ.setdefault("KAGGLEHUB_CACHE", str(DATA_DIR / "kagglehub"))
    source = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    target = RAW_DIR / "tid_fingerspelling"
    if target.exists() and any(target.rglob("*")):
        print(f"Veri zaten mevcut: {target}")
        return
    shutil.copytree(source, target, dirs_exist_ok=True)
    print(f"Veri indirildi: {target}")


if __name__ == "__main__":
    main()
