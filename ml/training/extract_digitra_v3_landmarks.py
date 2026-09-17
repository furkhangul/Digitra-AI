"""Extract leakage-safe 422D MediaPipe features for Digitra V3 development.

Only ASL-HG participants P1-P8 are read. P1-P7 are training participants and
P8 is the development participant. Locked participants P9/P10 are deliberately
excluded so that development cannot contaminate the final benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


STATIC_LABELS = [chr(ord("A") + index) for index in range(26) if index not in (9, 25)]
LABEL_TO_INDEX = {label: index for index, label in enumerate(STATIC_LABELS)}
NAME_RE = re.compile(r"^(P(?:10|[1-9]))_([A-Z])_(\d+)\.jpe?g$", re.I)
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


def unit_vector(vector, epsilon: float = 1e-7):
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
    coordinates = np.stack([centered @ ex, centered @ ey, centered @ ez], axis=1)
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
        bone_vectors.extend(unit_vector(canonical_world[right] - canonical_world[left]).tolist())
    joint_cosines = []
    for chain in FINGER_CHAINS:
        for index in range(1, len(chain) - 1):
            first = unit_vector(canonical_world[chain[index - 1]] - canonical_world[chain[index]])
            second = unit_vector(canonical_world[chain[index + 1]] - canonical_world[chain[index]])
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
        raise RuntimeError(f"Expected 422 features, received {feature.shape}")
    return feature


def worker_init(task_path: str):
    global _DETECTOR
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import mediapipe as mp

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=task_path),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.22,
        min_hand_presence_confidence=0.22,
        min_tracking_confidence=0.22,
    )
    _DETECTOR = mp.tasks.vision.HandLandmarker.create_from_options(options)


def process_one(payload):
    import mediapipe as mp

    path_text, participant, label, target, sample_id = payload
    try:
        with Image.open(path_text) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        rgb = np.ascontiguousarray(np.asarray(image))
        result = _DETECTOR.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.hand_landmarks:
            return participant, label, target, sample_id, path_text, "NO_HAND", None
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
        return participant, label, target, sample_id, path_text, "OK", feature
    except Exception as error:
        return (
            participant, label, target, sample_id, path_text,
            f"ERROR:{type(error).__name__}:{error}", None,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite V3 landmark extraction: {args.output}")
    args.output.mkdir(parents=True)
    if not args.task.exists():
        raise FileNotFoundError(args.task)

    payloads = []
    input_counts = Counter()
    for path in sorted(args.images_root.rglob("*.jpg")):
        match = NAME_RE.match(path.name)
        if not match:
            continue
        participant, label, number = match.groups()
        participant = participant.upper()
        label = label.upper()
        if label not in LABEL_TO_INDEX or int(participant[1:]) > 8:
            continue
        target = LABEL_TO_INDEX[label]
        sample_id = f"ASLHG-{participant}-{label}-{int(number):04d}"
        payloads.append((str(path.resolve()), participant, label, target, sample_id))
        input_counts[(participant, label)] += 1

    expected = 8 * len(STATIC_LABELS) * 100
    if len(payloads) != expected:
        raise RuntimeError(f"Expected {expected} P1-P8 static inputs, got {len(payloads)}")
    if set(input_counts.values()) != {100}:
        raise RuntimeError(f"Participant/class imbalance: {Counter(input_counts.values())}")

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
                print("LANDMARK_V3", index, "of", len(payloads), "detected", detected, flush=True)

    detected_rows = [row for row in results if row[-1] is not None]
    features = np.stack([row[-1] for row in detected_rows]).astype(np.float32)
    participants = np.asarray([row[0] for row in detected_rows])
    labels = np.asarray([row[1] for row in detected_rows])
    targets = np.asarray([row[2] for row in detected_rows], dtype=np.int64)
    sample_ids = np.asarray([row[3] for row in detected_rows])
    paths = np.asarray([row[4] for row in detected_rows])
    statuses = Counter(row[5] for row in results)
    detected_counts = Counter((row[0], row[1]) for row in detected_rows)

    np.savez_compressed(
        args.output / "asl_hg_p1_p8_landmarks_422d.npz",
        X=features,
        y=targets,
        participant=participants,
        label=labels,
        sample_id=sample_ids,
        path=paths,
        classes=np.asarray(STATIC_LABELS),
        feature_version=np.asarray(["canonical_422_v1"]),
    )
    audit = {
        "source": "ASL-HG raw v1",
        "license": "CC BY 4.0",
        "scope": "P1-P8 and 24 static letters only; P9/P10 never read",
        "inputs": len(payloads),
        "detected": len(detected_rows),
        "detection_rate": len(detected_rows) / len(payloads),
        "status_counts": dict(statuses),
        "participants": sorted(set(participants)),
        "per_participant_class_min_detected": min(detected_counts.values()),
        "per_participant_class_max_detected": max(detected_counts.values()),
        "locked_test_participants_read": False,
    }
    (args.output / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("LANDMARK_V3_READY", json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
