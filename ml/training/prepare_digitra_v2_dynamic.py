"""Extract regulated temporal landmarks from the public SigNN J/Z videos."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from prepare_digitra_v2_static import canonical_points


SEQUENCE_LENGTH = 32
FRAME_DIM = 130
_DETECTOR = None
_NEXT_TIMESTAMP_MS = None


def _worker_init(task_path: str):
    global _DETECTOR, _NEXT_TIMESTAMP_MS
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import mediapipe as mp

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=task_path),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.22,
        min_hand_presence_confidence=0.22,
        min_tracking_confidence=0.22,
    )
    _DETECTOR = mp.tasks.vision.HandLandmarker.create_from_options(options)
    _NEXT_TIMESTAMP_MS = 0


def frame_feature(image_points, world_points, handedness):
    canonical_image = canonical_points(image_points).reshape(-1)
    canonical_world = canonical_points(world_points).reshape(-1)
    width = float(np.linalg.norm(image_points[5, :2] - image_points[17, :2]))
    height = float(np.linalg.norm(image_points[0, :2] - image_points[9, :2]))
    scale = max((width + height) / 2.0, 1e-5)
    feature = np.concatenate(
        [
            canonical_image,
            canonical_world,
            np.asarray(
                [
                    image_points[0, 0],
                    image_points[0, 1],
                    scale,
                    1.0 if str(handedness).lower() == "left" else 0.0,
                ],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)
    if feature.shape != (FRAME_DIM,):
        raise RuntimeError(f"Expected {FRAME_DIM} frame features, got {feature.shape}")
    return feature


def regulate(features, positions, output_length=SEQUENCE_LENGTH):
    positions = np.asarray(positions, dtype=np.float32)
    positions = (positions - positions.min()) / max(float(np.ptp(positions)), 1e-6)
    targets = np.linspace(0.0, 1.0, output_length, dtype=np.float32)
    source = np.asarray(features, dtype=np.float32)
    output = np.column_stack(
        [np.interp(targets, positions, source[:, column]) for column in range(source.shape[1])]
    ).astype(np.float32)
    median_scale = max(float(np.median(output[:, 128])), 1e-5)
    output[:, 126:128] = (output[:, 126:128] - output[0:1, 126:128]) / median_scale
    output[:, 128] = np.log(np.clip(output[:, 128] / median_scale, 1e-4, 1e4))
    return output


def motion_energy(sequence):
    index_tip = sequence[:, 8 * 3 : 8 * 3 + 3]
    pinky_tip = sequence[:, 20 * 3 : 20 * 3 + 3]
    wrist_trajectory = sequence[:, 126:128]
    velocity = (
        np.linalg.norm(np.diff(index_tip, axis=0), axis=1)
        + np.linalg.norm(np.diff(pinky_tip, axis=0), axis=1)
        + 0.35 * np.linalg.norm(np.diff(wrist_trajectory, axis=0), axis=1)
    )
    return float(np.mean(velocity))


def _extract_video(payload):
    global _NEXT_TIMESTAMP_MS
    import cv2
    import mediapipe as mp

    path_text, label, group_id = payload
    path = Path(path_text)
    capture = cv2.VideoCapture(str(path))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    features, positions = [], []
    frame_index = 0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            # A VIDEO-mode landmarker persists across tasks inside each worker and
            # therefore requires timestamps to stay increasing between videos too.
            timestamp_ms = int(_NEXT_TIMESTAMP_MS)
            _NEXT_TIMESTAMP_MS += max(1, int(round(1000.0 / fps)))
            result = _DETECTOR.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
            )
            if result.hand_landmarks:
                image_points = np.asarray(
                    [[point.x, point.y, point.z] for point in result.hand_landmarks[0]],
                    dtype=np.float32,
                )
                world_points = np.asarray(
                    [[point.x, point.y, point.z] for point in result.hand_world_landmarks[0]],
                    dtype=np.float32,
                )
                category = result.handedness[0][0]
                features.append(
                    frame_feature(image_points, world_points, category.category_name)
                )
                positions.append(frame_index)
            frame_index += 1
    finally:
        capture.release()
    if len(features) < 8:
        return {
            "ok": False,
            "path": str(path),
            "label": label,
            "group_id": group_id,
            "frames": frame_index,
            "detected_frames": len(features),
            "status": "TOO_FEW_DETECTED_FRAMES",
        }
    sequence = regulate(features, positions)
    return {
        "ok": True,
        "path": str(path),
        "label": label,
        "group_id": group_id,
        "frames": frame_index,
        "detected_frames": len(features),
        "detection_rate": len(features) / max(frame_index, 1),
        "motion_energy": motion_energy(sequence),
        "sequence": sequence,
        "status": "OK",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    payloads = []
    for label_name, label_index in [("J", 0), ("Z", 1)]:
        for path in sorted((args.video_root / label_name).glob("*.avi")):
            try:
                group_id = int(path.stem)
            except ValueError as error:
                raise RuntimeError(f"Non-numeric SigNN video name: {path}") from error
            payloads.append((str(path), label_index, group_id))
    if len(payloads) != 712:
        raise RuntimeError(f"Expected 712 J/Z videos, found {len(payloads)}")

    results = []
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_worker_init,
        initargs=(str(args.task),),
    ) as executor:
        for completed, result in enumerate(executor.map(_extract_video, payloads, chunksize=2), 1):
            results.append(result)
            if completed % 25 == 0 or completed == len(payloads):
                print(f"VIDEOS {completed}/{len(payloads)}", flush=True)

    valid = [result for result in results if result["ok"]]
    # Numeric filenames are not a shared paired-ID namespace: the J and Z ranges
    # overlap only partially.  Split each class contiguously by its own recording
    # order so every split contains both labels while nearby recording blocks stay
    # together as much as the public metadata permits.
    train_groups, val_groups, test_groups = set(), set(), set()
    for label in (0, 1):
        label_groups = sorted(
            {result["group_id"] for result in valid if result["label"] == label}
        )
        train_end = int(round(len(label_groups) * 0.70))
        val_end = int(round(len(label_groups) * 0.85))
        train_groups.update((label, group_id) for group_id in label_groups[:train_end])
        val_groups.update((label, group_id) for group_id in label_groups[train_end:val_end])
        test_groups.update((label, group_id) for group_id in label_groups[val_end:])
    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    def split_for(label, group_id):
        key = (label, group_id)
        if key in train_groups:
            return "train"
        if key in val_groups:
            return "val"
        return "test"

    np.savez_compressed(
        args.output / "dynamic_positive_v2.npz",
        X=np.stack([result["sequence"] for result in valid]).astype(np.float32),
        y=np.asarray([result["label"] for result in valid], dtype=np.int64),
        group_id=np.asarray([result["group_id"] for result in valid], dtype=np.int64),
        split=np.asarray(
            [split_for(result["label"], result["group_id"]) for result in valid]
        ),
        motion_energy=np.asarray([result["motion_energy"] for result in valid], dtype=np.float32),
        source_path=np.asarray([result["path"] for result in valid]),
        class_names=np.asarray(["J", "Z", "OTHER"]),
        frame_dim=np.asarray([FRAME_DIM]),
        sequence_length=np.asarray([SEQUENCE_LENGTH]),
    )
    serializable_results = [
        {key: value for key, value in result.items() if key != "sequence"}
        for result in results
    ]
    audit = {
        "source": str(args.video_root),
        "license": "CC0 Public Domain (Kaggle dataset card)",
        "source_videos": len(results),
        "valid_videos": len(valid),
        "status_counts": dict(Counter(result["status"] for result in results)),
        "label_counts": dict(Counter("J" if result["label"] == 0 else "Z" for result in valid)),
        "split_counts": dict(
            Counter(split_for(result["label"], result["group_id"]) for result in valid)
        ),
        "split_label_counts": {
            split: {
                label_name: sum(
                    split_for(result["label"], result["group_id"]) == split
                    and result["label"] == label
                    for result in valid
                )
                for label_name, label in [("J", 0), ("Z", 1)]
            }
            for split in ("train", "val", "test")
        },
        "split_group_counts": {
            "train": len(train_groups), "val": len(val_groups), "test": len(test_groups)
        },
        "split_strategy": "label-stratified contiguous 70/15/15 recording-ID blocks",
        "known_limitation": "The public dataset does not publish signer identity; the split is video-ID-disjoint and class-balanced, not proven signer-disjoint.",
        "median_detection_rate": float(np.median([result["detection_rate"] for result in valid])),
        "median_motion_energy": float(np.median([result["motion_energy"] for result in valid])),
    }
    (args.output / "dynamic_audit.json").write_text(
        json.dumps({"summary": audit, "videos": serializable_results}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2), flush=True)
    print("DYNAMIC_V2_PREP_READY", args.output / "dynamic_positive_v2.npz", flush=True)


if __name__ == "__main__":
    main()
