"""Prepare signer-disjoint ROI crops and paired 422D features for Digitra V2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


TRAIN_SIGNERS = {str(value) for value in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13]}
VAL_SIGNERS = {"10", "11"}
TEST_SIGNERS = {"14", "15"}
EXPECTED_SIGNERS = TRAIN_SIGNERS | VAL_SIGNERS | TEST_SIGNERS
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]
FINGER_CHAINS = [
    [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16], [0, 17, 18, 19, 20],
]
TIP_IDS = [4, 8, 12, 16, 20]

_DETECTOR = None
_OUTPUT_ROOT = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_for_signer(signer: str) -> str:
    if signer in TRAIN_SIGNERS:
        return "train"
    if signer in VAL_SIGNERS:
        return "val"
    if signer in TEST_SIGNERS:
        return "test"
    raise ValueError(f"Unexpected signer: {signer}")


def unit_vector(vector, epsilon=1e-7):
    return vector / max(float(np.linalg.norm(vector)), epsilon)


def canonical_points(points):
    points = np.asarray(points, dtype=np.float32)
    centered = points - points[0]
    ex = unit_vector(centered[5] - centered[17])
    y_hint = unit_vector(centered[9])
    ez = unit_vector(np.cross(ex, y_hint))
    if np.linalg.norm(ez) < 1e-5:
        ez = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    ey = unit_vector(np.cross(ez, ex))
    coordinates = np.stack(
        [centered @ ex, centered @ ey, centered @ ez], axis=1
    )
    scale = np.mean([np.linalg.norm(centered[index]) for index in [5, 9, 13, 17]])
    return coordinates / max(float(scale), 1e-6)


def landmark_feature(image_points, world_points, handedness, handedness_score):
    canonical_image = canonical_points(image_points)
    canonical_world = canonical_points(world_points)
    pairwise = [
        np.linalg.norm(canonical_world[left] - canonical_world[right])
        for left in range(21) for right in range(left + 1, 21)
    ]
    bone_vectors = []
    for left, right in HAND_EDGES:
        bone_vectors.extend(
            unit_vector(canonical_world[right] - canonical_world[left]).tolist()
        )
    joint_cosines = []
    for chain in FINGER_CHAINS:
        for index in range(1, len(chain) - 1):
            first = unit_vector(
                canonical_world[chain[index - 1]] - canonical_world[chain[index]]
            )
            second = unit_vector(
                canonical_world[chain[index + 1]] - canonical_world[chain[index]]
            )
            joint_cosines.append(float(np.clip(np.dot(first, second), -1.0, 1.0)))
    tip_geometry = [float(np.linalg.norm(canonical_world[index])) for index in TIP_IDS]
    tip_geometry += [
        float(np.linalg.norm(canonical_world[4] - canonical_world[index]))
        for index in TIP_IDS[1:]
    ]
    handed = [
        1.0 if str(handedness).lower() == "left" else 0.0,
        float(handedness_score),
    ]
    feature = np.concatenate(
        [
            canonical_image.reshape(-1), canonical_world.reshape(-1),
            np.asarray(pairwise, dtype=np.float32),
            np.asarray(bone_vectors, dtype=np.float32),
            np.asarray(joint_cosines, dtype=np.float32),
            np.asarray(tip_geometry, dtype=np.float32),
            np.asarray(handed, dtype=np.float32),
        ]
    ).astype(np.float32)
    if feature.shape != (422,):
        raise RuntimeError(f"Expected a 422D feature, got {feature.shape}")
    return feature


def difference_hash(image: Image.Image) -> str:
    pixels = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    bits = pixels[:, 1:] > pixels[:, :-1]
    return f"{int(''.join('1' if bit else '0' for bit in bits.flat), 2):016x}"


def _worker_init(task_path: str, output_root: str):
    global _DETECTOR, _OUTPUT_ROOT
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import mediapipe as mp

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=task_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.25,
        min_hand_presence_confidence=0.25,
        min_tracking_confidence=0.25,
    )
    _DETECTOR = mp.tasks.vision.HandLandmarker.create_from_options(options)
    _OUTPUT_ROOT = Path(output_root)


def _square_roi(image: Image.Image, points: np.ndarray, margin: float = 0.42):
    width, height = image.size
    xs = points[:, 0] * width
    ys = points[:, 1] * height
    hand_width = max(float(xs.max() - xs.min()), 1.0)
    hand_height = max(float(ys.max() - ys.min()), 1.0)
    side = max(hand_width, hand_height) * (1.0 + 2.0 * margin)
    center_x = float((xs.min() + xs.max()) / 2.0)
    center_y = float((ys.min() + ys.max()) / 2.0)
    left = int(round(center_x - side / 2.0))
    top = int(round(center_y - side / 2.0))
    right = int(round(center_x + side / 2.0))
    bottom = int(round(center_y + side / 2.0))
    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - width)
    pad_bottom = max(0, bottom - height)
    if any([pad_left, pad_top, pad_right, pad_bottom]):
        image = ImageOps.expand(
            image,
            border=(pad_left, pad_top, pad_right, pad_bottom),
            fill=(114, 114, 114),
        )
        left += pad_left
        right += pad_left
        top += pad_top
        bottom += pad_top
    return image.crop((left, top, right, bottom))


def _process_one(payload):
    import mediapipe as mp

    row_index, source_path, sample_id, split, signer, label_index = payload
    try:
        with Image.open(source_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        rgb = np.ascontiguousarray(np.asarray(image))
        result = _DETECTOR.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        )
        if not result.hand_landmarks:
            return row_index, False, "NO_HAND", None, None, None
        image_points = np.asarray(
            [[point.x, point.y, point.z] for point in result.hand_landmarks[0]],
            dtype=np.float32,
        )
        world_points = np.asarray(
            [[point.x, point.y, point.z] for point in result.hand_world_landmarks[0]],
            dtype=np.float32,
        )
        category = result.handedness[0][0]
        feature = landmark_feature(
            image_points,
            world_points,
            category.category_name,
            float(category.score),
        )
        crop = _square_roi(image, image_points).resize(
            (384, 384), Image.Resampling.LANCZOS
        )
        output_path = _OUTPUT_ROOT / split / str(label_index) / f"{sample_id}.webp"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(output_path, format="WEBP", quality=94, method=4)
        return (
            row_index,
            True,
            "OK",
            str(output_path),
            difference_hash(crop),
            feature,
        )
    except Exception as error:
        return row_index, False, f"ERROR:{type(error).__name__}:{error}", None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    crop_root = args.output / "crops"
    image_paths = sorted(
        path for path in args.images_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )
    rows = []
    class_signer_counts = Counter()
    for path in image_paths:
        relative = path.relative_to(args.images_root)
        if len(relative.parts) < 3:
            raise RuntimeError(f"Unexpected source path: {relative}")
        signer, class_text = relative.parts[0], relative.parts[1]
        if signer not in EXPECTED_SIGNERS:
            raise RuntimeError(f"Unexpected signer in {relative}")
        class_index = int(class_text)
        if not 0 <= class_index < 26:
            raise RuntimeError(f"Unexpected class in {relative}")
        split = split_for_signer(signer)
        sample_id = hashlib.sha1(str(relative).encode("utf-8")).hexdigest()[:20]
        row = {
            "sample_id": sample_id,
            "source_path": str(path),
            "source_relative": relative.as_posix(),
            "signer": signer,
            "class_index": class_index,
            "class_name": chr(ord("A") + class_index),
            "split": split,
            "source_sha256": sha256_file(path),
            "detected": False,
            "status": "PENDING",
            "crop_path": "",
            "dhash": "",
        }
        rows.append(row)
        class_signer_counts[(split, signer, class_index)] += 1

    observed_signers = {row["signer"] for row in rows}
    if observed_signers != EXPECTED_SIGNERS:
        raise RuntimeError(
            f"Signer contract failed: expected {sorted(EXPECTED_SIGNERS)}, got {sorted(observed_signers)}"
        )

    exact_groups = defaultdict(list)
    for index, row in enumerate(rows):
        exact_groups[row["source_sha256"]].append(index)
    dropped_exact = set()
    priority = {"test": 3, "val": 2, "train": 1}
    for group in exact_groups.values():
        splits = {rows[index]["split"] for index in group}
        if len(splits) <= 1:
            continue
        keeper = max(group, key=lambda index: priority[rows[index]["split"]])
        dropped_exact.update(index for index in group if index != keeper)
    for index in dropped_exact:
        rows[index]["status"] = "DROPPED_CROSS_SPLIT_EXACT_DUPLICATE"

    payloads = [
        (
            index,
            row["source_path"],
            row["sample_id"],
            row["split"],
            row["signer"],
            row["class_index"],
        )
        for index, row in enumerate(rows)
        if index not in dropped_exact
    ]
    feature_rows = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(str(args.task), str(crop_root)),
    ) as executor:
        for completed, result in enumerate(executor.map(_process_one, payloads, chunksize=12), 1):
            row_index, detected, status, crop_path, dhash, feature = result
            rows[row_index]["detected"] = bool(detected)
            rows[row_index]["status"] = status
            rows[row_index]["crop_path"] = crop_path or ""
            rows[row_index]["dhash"] = dhash or ""
            if feature is not None:
                feature_rows.append((rows[row_index]["sample_id"], feature))
            if completed % 1000 == 0 or completed == len(payloads):
                print(f"CROPS {completed}/{len(payloads)}", flush=True)

    feature_rows.sort(key=lambda item: item[0])
    np.savez_compressed(
        args.output / "paired_landmark_features_v2.npz",
        sample_id=np.asarray([item[0] for item in feature_rows]),
        X=np.stack([item[1] for item in feature_rows]).astype(np.float32),
        feature_version=np.asarray(["canonical_422_v1"]),
    )

    manifest_path = args.output / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    split_total = Counter(row["split"] for row in rows if not row["status"].startswith("DROPPED"))
    split_detected = Counter(row["split"] for row in rows if row["detected"])
    dhash_splits = defaultdict(set)
    for row in rows:
        if row["dhash"]:
            dhash_splits[row["dhash"]].add(row["split"])
    cross_split_identical_dhash = sum(len(splits) > 1 for splits in dhash_splits.values())
    audit = {
        "source_root": str(args.images_root),
        "source_images": len(rows),
        "signers": sorted(observed_signers),
        "split_signers": {
            "train": sorted(TRAIN_SIGNERS),
            "val": sorted(VAL_SIGNERS),
            "test": sorted(TEST_SIGNERS),
        },
        "split_total": dict(split_total),
        "split_detected": dict(split_detected),
        "split_detection_rate": {
            split: split_detected[split] / split_total[split] for split in split_total
        },
        "cross_split_exact_source_duplicates_removed": len(dropped_exact),
        "cross_split_identical_dhash_groups": cross_split_identical_dhash,
        "feature_rows": len(feature_rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "class_signer_min": min(class_signer_counts.values()),
        "class_signer_max": max(class_signer_counts.values()),
        "leakage_assertions": {
            "train_val_signers_disjoint": TRAIN_SIGNERS.isdisjoint(VAL_SIGNERS),
            "train_test_signers_disjoint": TRAIN_SIGNERS.isdisjoint(TEST_SIGNERS),
            "val_test_signers_disjoint": VAL_SIGNERS.isdisjoint(TEST_SIGNERS),
        },
    }
    (args.output / "audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)
    print("STATIC_V2_PREP_READY", manifest_path, flush=True)


if __name__ == "__main__":
    main()
