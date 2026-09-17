"""Extract the frozen Digitra V3 landmark branch's locked P9/P10 test inputs.

This script is intentionally separate from development extraction.  It may run
only after the hybrid protocol has been frozen, and it records that the locked
test has been opened before reading the first image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from extract_digitra_v3_landmarks import (
    LABEL_TO_INDEX,
    NAME_RE,
    STATIC_LABELS,
    process_one,
    worker_init,
)


LOCKED_PARTICIPANTS = {"P9", "P10"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--opened-marker", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite locked extraction: {args.output}")
    if args.opened_marker.exists():
        raise FileExistsError(f"Locked test was already opened: {args.opened_marker}")
    if not args.frozen_manifest.exists():
        raise FileNotFoundError("Freeze the development protocol before opening P9/P10")
    frozen = json.loads(args.frozen_manifest.read_text(encoding="utf-8"))
    if frozen.get("locked_test_opened") is not False:
        raise RuntimeError("Frozen manifest does not assert an unopened locked test")
    args.output.mkdir(parents=True)

    payloads = []
    input_counts = Counter()
    for path in sorted(args.images_root.rglob("*.jpg")):
        match = NAME_RE.match(path.name)
        if not match:
            continue
        participant, label, number = match.groups()
        participant = participant.upper()
        label = label.upper()
        if participant not in LOCKED_PARTICIPANTS or label not in LABEL_TO_INDEX:
            continue
        sample_id = f"ASLHG-{participant}-{label}-{int(number):04d}"
        payloads.append(
            (
                str(path.resolve()), participant, label,
                LABEL_TO_INDEX[label], sample_id,
            )
        )
        input_counts[(participant, label)] += 1

    expected = len(LOCKED_PARTICIPANTS) * len(STATIC_LABELS) * 100
    if len(payloads) != expected or set(input_counts.values()) != {100}:
        raise RuntimeError(
            f"Locked split contract failed: rows={len(payloads)}, "
            f"per participant/class={Counter(input_counts.values())}"
        )

    marker = {
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "landmark_extraction_started",
        "participants": sorted(LOCKED_PARTICIPANTS),
        "frozen_manifest": str(args.frozen_manifest),
        "frozen_manifest_sha256": sha256(args.frozen_manifest),
        "development_configuration_was_frozen_first": True,
    }
    args.opened_marker.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    print("OPENING_FRESH_LOCKED_TEST_P9_P10", flush=True)

    results = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=worker_init,
        initargs=(str(args.task.resolve()),),
    ) as executor:
        for index, result in enumerate(executor.map(process_one, payloads, chunksize=8), start=1):
            results.append(result)
            if index % 500 == 0 or index == len(payloads):
                detected = sum(item[-1] is not None for item in results)
                print("LOCKED_LANDMARK", index, "of", len(payloads), "detected", detected, flush=True)

    detected_rows = [row for row in results if row[-1] is not None]
    np.savez_compressed(
        args.output / "asl_hg_p9_p10_landmarks_422d.npz",
        X=np.stack([row[-1] for row in detected_rows]).astype(np.float32),
        y=np.asarray([row[2] for row in detected_rows], dtype=np.int64),
        participant=np.asarray([row[0] for row in detected_rows]),
        label=np.asarray([row[1] for row in detected_rows]),
        sample_id=np.asarray([row[3] for row in detected_rows]),
        path=np.asarray([row[4] for row in detected_rows]),
        classes=np.asarray(STATIC_LABELS),
        feature_version=np.asarray(["canonical_422_v1"]),
    )
    status_counts = Counter(row[5] for row in results)
    audit = {
        "scope": "Fresh locked ASL-HG P9/P10; 24 static ASL letters",
        "inputs": len(payloads),
        "detected": len(detected_rows),
        "detector_misses": len(payloads) - len(detected_rows),
        "detection_rate": len(detected_rows) / len(payloads),
        "status_counts": dict(status_counts),
        "participants": sorted(LOCKED_PARTICIPANTS),
        "development_configuration_frozen_before_read": True,
    }
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    marker["stage"] = "landmarks_extracted"
    marker["landmark_audit"] = audit
    args.opened_marker.write_text(json.dumps(marker, indent=2), encoding="utf-8")
    print("LOCKED_LANDMARK_READY", json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
