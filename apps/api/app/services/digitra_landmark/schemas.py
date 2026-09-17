from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _landmark_array(values: Any) -> np.ndarray:
    """Accept both the public array contract and MediaPipe's JSON object shape."""

    if isinstance(values, list) and values and isinstance(values[0], dict):
        values = [
            [point.get("x", 0.0), point.get("y", 0.0), point.get("z", 0.0)]
            for point in values
        ]
    return np.asarray(values, dtype=np.float32).reshape(21, 3)


@dataclass
class HandObservation:
    landmarks: np.ndarray
    world_landmarks: np.ndarray | None = None
    handedness: str = "Unknown"
    handedness_score: float = 0.0

    def __post_init__(self) -> None:
        self.landmarks = _landmark_array(self.landmarks)
        if self.world_landmarks is not None:
            self.world_landmarks = _landmark_array(self.world_landmarks)

    def as_dict(self) -> dict[str, Any]:
        return {
            "landmarks": self.landmarks.tolist(),
            "world_landmarks": (
                self.world_landmarks.tolist() if self.world_landmarks is not None else None
            ),
            "handedness": self.handedness,
            "handedness_score": float(self.handedness_score),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HandObservation:
        return cls(
            landmarks=data["landmarks"],
            world_landmarks=data.get("world_landmarks"),
            handedness=data.get("handedness", "Unknown"),
            handedness_score=data.get("handedness_score", 0.0),
        )
