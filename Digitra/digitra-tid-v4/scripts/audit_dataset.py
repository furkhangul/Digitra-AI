from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import pandas as pd

from src.config import ARTIFACTS_DIR, IMAGE_EXTENSIONS, LABEL_MAP, SEED, ensure_dirs, fingerspelling_image_dir


def parse(path: Path) -> tuple[str, int]:
    match = re.match(r"^(.+?) \((\d+)\)$", path.stem)
    if not match:
        raise ValueError(f"Beklenmeyen dosya adı: {path.name}")
    raw_label, index = match.groups()
    return LABEL_MAP.get(raw_label, raw_label), int(index)


def assign_split(block: int) -> str:
    # Adjacent frames remain together. The source has no signer metadata, so
    # this is explicitly a pseudo-session split, not a signer-independent one.
    bucket = (block * 2654435761 + SEED) % 10
    if bucket < 7:
        return "train"
    if bucket < 9:
        return "val"
    return "test"


def main() -> None:
    ensure_dirs()
    image_dir = fingerspelling_image_dir()
    paths = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    rows = []
    exact = defaultdict(list)
    unreadable = []
    for path in paths:
        label, index = parse(path)
        image = cv2.imread(str(path))
        if image is None:
            unreadable.append(str(path))
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        exact[digest].append(str(path))
        block = (index - 1) // 10
        rows.append({
            "path": str(path.resolve()),
            "filename": path.name,
            "label": label,
            "source_index": index,
            "pseudo_session": f"{label}-{block:02d}",
            "split": assign_split(block),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "sha256": digest,
        })

    frame = pd.DataFrame(rows).sort_values(["label", "source_index"])
    manifest = ARTIFACTS_DIR / "manifest.csv"
    frame.to_csv(manifest, index=False, encoding="utf-8-sig")

    duplicate_groups = [members for members in exact.values() if len(members) > 1]
    cross_label_duplicates = []
    path_to_label = dict(zip(frame.path, frame.label))
    for members in duplicate_groups:
        labels = {path_to_label[m] for m in members if m in path_to_label}
        if len(labels) > 1:
            cross_label_duplicates.append({"labels": sorted(labels), "paths": members})

    report = {
        "source_dir": str(image_dir.resolve()),
        "samples": int(len(frame)),
        "classes": int(frame.label.nunique()),
        "class_counts": {str(k): int(v) for k, v in Counter(frame.label).items()},
        "split_counts": {str(k): int(v) for k, v in Counter(frame.split).items()},
        "pseudo_sessions": int(frame.pseudo_session.nunique()),
        "unreadable": unreadable,
        "exact_duplicate_groups": len(duplicate_groups),
        "cross_label_duplicate_groups": cross_label_duplicates,
        "limitations": [
            "Kaynak veri kişi ve gerçek oturum kimliği içermiyor.",
            "Split, ardışık 10 karelik pseudo-session bloklarıyla sızıntıyı azaltır.",
            "Bu test signer-independent ürün testi değildir.",
            "Lisans CC BY-NC-SA 4.0; ticari ürün eğitiminde kullanılamaz.",
        ],
    }
    (ARTIFACTS_DIR / "audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
