from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

import numpy as np

from .schemas import HandObservation

if TYPE_CHECKING:
    from .hand_detector import HandDetector


def _box_iou(first: HandObservation, second: HandObservation) -> float:
    first_min = np.min(first.landmarks[:, :2], axis=0)
    first_max = np.max(first.landmarks[:, :2], axis=0)
    second_min = np.min(second.landmarks[:, :2], axis=0)
    second_max = np.max(second.landmarks[:, :2], axis=0)
    intersection_size = np.maximum(
        0.0, np.minimum(first_max, second_max) - np.maximum(first_min, second_min)
    )
    intersection = float(np.prod(intersection_size))
    first_area = float(np.prod(np.maximum(0.0, first_max - first_min)))
    second_area = float(np.prod(np.maximum(0.0, second_max - second_min)))
    union = first_area + second_area - intersection
    return intersection / union if union > 1e-9 else 0.0


def hands_are_duplicates(first: HandObservation, second: HandObservation) -> bool:
    """Detect two MediaPipe observations describing the same physical hand."""
    corresponding_distances = np.linalg.norm(
        first.landmarks[:, :2] - second.landmarks[:, :2], axis=1
    )
    median_distance = float(np.median(corresponding_distances))
    wrist_distance = float(
        np.linalg.norm(first.landmarks[0, :2] - second.landmarks[0, :2])
    )
    first_centered = first.landmarks[:, :2] - first.landmarks[0, :2]
    second_centered = second.landmarks[:, :2] - second.landmarks[0, :2]
    centered_shape_distance = float(
        np.median(np.linalg.norm(first_centered - second_centered, axis=1))
    )
    overlap = _box_iou(first, second)
    same_handedness = (
        first.handedness == second.handedness
        and first.handedness not in {"", "Unknown"}
    )
    return (
        (
            median_distance < 0.045
            and wrist_distance < 0.070
            and centered_shape_distance < 0.045
        )
        or (
            overlap > 0.70
            and median_distance < 0.10
            and centered_shape_distance < 0.045
        )
        or (
            same_handedness
            and wrist_distance < 0.10
            and centered_shape_distance < 0.055
        )
    )


def deduplicate_hands(hands: list[HandObservation]) -> list[HandObservation]:
    """Keep the strongest observation when MediaPipe emits a duplicate skeleton."""
    kept: list[HandObservation] = []
    for candidate in sorted(hands, key=lambda hand: hand.handedness_score, reverse=True):
        if not any(hands_are_duplicates(candidate, existing) for existing in kept):
            kept.append(candidate)
    return kept


def choose_orientation_probabilities(
    normal: np.ndarray,
    mirrored: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Return the more confident full distribution without changing its labels."""
    normal_values = np.asarray(normal, dtype=float)
    mirrored_values = np.asarray(mirrored, dtype=float)
    if normal_values.shape != mirrored_values.shape:
        raise ValueError("Normal and mirrored probability shapes must match")
    if float(np.max(mirrored_values)) > float(np.max(normal_values)):
        return mirrored_values, "ters el"
    return normal_values, "normal"


def mirror_hands(hands: list[HandObservation]) -> list[HandObservation]:
    """Mirror observations so a model can evaluate the opposite hand arrangement."""
    mirrored: list[HandObservation] = []
    for hand in hands:
        landmarks = hand.landmarks.copy()
        landmarks[:, 0] = 1.0 - landmarks[:, 0]
        world_landmarks = None
        if hand.world_landmarks is not None:
            world_landmarks = hand.world_landmarks.copy()
            world_landmarks[:, 0] *= -1.0
        handedness = {"Left": "Right", "Right": "Left"}.get(
            hand.handedness, hand.handedness
        )
        mirrored.append(
            HandObservation(
                landmarks=landmarks,
                world_landmarks=world_landmarks,
                handedness=handedness,
                handedness_score=hand.handedness_score,
            )
        )
    return mirrored


def remap_hand(
    hand: HandObservation,
    crop: tuple[int, int, int, int],
    frame_shape: tuple[int, ...],
) -> HandObservation:
    """Map landmarks detected in a crop back to full-frame coordinates."""
    x0, y0, x1, y1 = crop
    height, width = frame_shape[:2]
    landmarks = hand.landmarks.copy()
    landmarks[:, 0] = (x0 + landmarks[:, 0] * (x1 - x0)) / width
    landmarks[:, 1] = (y0 + landmarks[:, 1] * (y1 - y0)) / height
    return HandObservation(
        landmarks=landmarks,
        world_landmarks=hand.world_landmarks,
        handedness=hand.handedness,
        handedness_score=hand.handedness_score,
    )


def recover_two_hands(
    detector: HandDetector,
    frame: np.ndarray,
    full_hands: list[HandObservation] | None = None,
) -> list[HandObservation]:
    """Use overlapping crops when the full frame misses one of two hands."""
    full_hands = detector.detect_bgr(frame) if full_hands is None else list(full_hands)
    full_hands = deduplicate_hands(full_hands)
    if len(full_hands) == 2:
        return full_hands

    height, width = frame.shape[:2]
    crops = [
        (0, 0, width, round(height * 0.65)),
        (0, round(height * 0.35), width, height),
        (0, 0, round(width * 0.65), height),
        (round(width * 0.35), 0, width, height),
    ]
    candidates = list(full_hands)
    for crop in crops:
        x0, y0, x1, y1 = crop
        crop_frame = frame[y0:y1, x0:x1]
        for hand in detector.detect_bgr(crop_frame):
            remapped = remap_hand(hand, crop, frame.shape)
            wrist = remapped.landmarks[0, :2]
            duplicate = any(
                np.linalg.norm(wrist - item.landmarks[0, :2]) < 0.10
                or hands_are_duplicates(remapped, item)
                for item in candidates
            )
            if not duplicate:
                candidates.append(remapped)

    if len(candidates) <= 2:
        return candidates

    best_pair: tuple[HandObservation, HandObservation] | None = None
    best_score = -1.0
    for first_index, first in enumerate(candidates):
        for second in candidates[first_index + 1 :]:
            wrist_distance = float(
                np.linalg.norm(first.landmarks[0, :2] - second.landmarks[0, :2])
            )
            score = wrist_distance + 0.1 * (
                first.handedness_score + second.handedness_score
            )
            if score > best_score:
                best_pair = (first, second)
                best_score = score
    return list(best_pair) if best_pair else candidates[:2]


class TwoHandModeGate:
    """Require sustained direct detections before switching automatic hand modes."""

    def __init__(
        self,
        enter_frames: int = 2,
        exit_frames: int = 2,
    ) -> None:
        if enter_frames < 1 or exit_frames < 1:
            raise ValueError("Mode gate frame counts must be positive")
        self.enter_frames = enter_frames
        self.exit_frames = exit_frames
        self.active = False
        self._two_hand_frames = 0
        self._fewer_hand_frames = 0

    def reset(self) -> None:
        self.active = False
        self._two_hand_frames = 0
        self._fewer_hand_frames = 0

    def update(self, directly_detected_hands: int) -> bool:
        if directly_detected_hands >= 2:
            self._two_hand_frames += 1
            self._fewer_hand_frames = 0
            if self._two_hand_frames >= self.enter_frames:
                self.active = True
        else:
            self._two_hand_frames = 0
            self._fewer_hand_frames += 1
            if self._fewer_hand_frames >= self.exit_frames:
                self.active = False
        return self.active


class StableTwoHandTracker:
    """Bridge brief detector dropouts without permanently inventing a hand."""

    def __init__(self, grace_frames: int = 5, wrist_match_distance: float = 0.18) -> None:
        self.grace_frames = grace_frames
        self.wrist_match_distance = wrist_match_distance
        self._last_pair: list[HandObservation] = []
        self._missing_frames = 0
        self.hand_count_history: deque[int] = deque(maxlen=20)

    def reset(self) -> None:
        self._last_pair = []
        self._missing_frames = 0
        self.hand_count_history.clear()

    def update(self, hands: list[HandObservation]) -> list[HandObservation]:
        self.hand_count_history.append(len(hands))
        if len(hands) >= 2:
            self._last_pair = list(hands[:2])
            self._missing_frames = 0
            return self._last_pair

        if not self._last_pair or self._missing_frames >= self.grace_frames:
            self._last_pair = []
            self._missing_frames = 0
            return list(hands)

        self._missing_frames += 1
        if not hands:
            return list(self._last_pair)

        current = hands[0]
        distances = [
            float(np.linalg.norm(current.landmarks[0, :2] - old.landmarks[0, :2]))
            for old in self._last_pair
        ]
        matched_index = int(np.argmin(distances))
        if distances[matched_index] > self.wrist_match_distance:
            return list(self._last_pair)

        pair = list(self._last_pair)
        pair[matched_index] = current
        self._last_pair = pair
        return pair
