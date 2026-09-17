from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from src.config import ARTIFACTS_DIR


@dataclass(frozen=True)
class HandCrop:
    rgb: np.ndarray
    box: tuple[int, int, int, int]
    hand_count: int
    confidence: float


def padded_square_box(
    points: np.ndarray,
    width: int,
    height: int,
    padding: float = 0.38,
    minimum_side_fraction: float = 0.18,
) -> tuple[int, int, int, int]:
    """Return a clamped square around one or two detected hands."""

    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) / 2.0
    side = max(float(maximum[0] - minimum[0]), float(maximum[1] - minimum[1]))
    side = max(
        side * (1.0 + 2.0 * padding),
        min(width, height) * minimum_side_fraction,
    )
    half = side / 2.0
    x1, y1 = center - half
    x2, y2 = center + half

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > width:
        x1 -= x2 - width
        x2 = width
    if y2 > height:
        y1 -= y2 - height
        y2 = height
    return (
        max(0, int(round(x1))),
        max(0, int(round(y1))),
        min(width, int(round(x2))),
        min(height, int(round(y2))),
    )


class MediaPipeHandCropper:
    """Detect up to two hands and normalize their joint region before classification."""

    def __init__(
        self,
        model_path: str | Path = ARTIFACTS_DIR / "hand_landmarker.task",
        min_confidence: float = 0.20,
    ) -> None:
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def close(self) -> None:
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def crop(self, bgr: np.ndarray) -> HandCrop | None:
        height, width = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        candidates = []
        for fraction in (1.0, 0.80, 0.64):
            view_width = int(round(width * fraction))
            view_height = int(round(height * fraction))
            x_offset = (width - view_width) // 2
            y_offset = (height - view_height) // 2
            view = np.ascontiguousarray(
                rgb[y_offset : y_offset + view_height, x_offset : x_offset + view_width]
            )
            result = self.landmarker.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=view)
            )
            scores = [
                float(categories[0].score)
                for categories in result.handedness
                if categories
            ]
            confidence = float(np.mean(scores)) if scores else 0.0
            candidates.append((len(result.hand_landmarks), confidence, result, fraction))
            if len(result.hand_landmarks) >= 2:
                break

        hand_count, confidence, result, fraction = max(
            candidates, key=lambda item: (item[0], item[1])
        )
        if not result.hand_landmarks:
            return None

        normalized = np.asarray(
            [[landmark.x, landmark.y] for hand in result.hand_landmarks for landmark in hand],
            dtype=np.float32,
        )
        view_width = width * fraction
        view_height = height * fraction
        offset = np.asarray(
            [(width - view_width) / 2.0, (height - view_height) / 2.0],
            dtype=np.float32,
        )
        pixels = normalized * np.asarray([view_width, view_height], dtype=np.float32) + offset
        minimum_side_fraction = 1.0 if hand_count == 1 else 0.32
        box = padded_square_box(
            pixels,
            width,
            height,
            minimum_side_fraction=minimum_side_fraction,
        )
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            return None
        return HandCrop(
            rgb=np.ascontiguousarray(rgb[y1:y2, x1:x2]),
            box=box,
            hand_count=hand_count,
            confidence=confidence,
        )
