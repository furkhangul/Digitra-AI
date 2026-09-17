from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schemas import HandObservation

EPS = 1e-8


@dataclass(frozen=True)
class NormalizedHand:
    local: np.ndarray
    image: np.ndarray
    wrist: np.ndarray
    palm_center: np.ndarray
    palm_normal: np.ndarray
    scale: float
    handedness: str
    handedness_score: float


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > EPS else np.zeros_like(vector)


def palm_scale(landmarks: np.ndarray) -> float:
    """Robust palm size from wrist/MCP distances in normalized image coordinates."""
    candidates = [
        np.linalg.norm(landmarks[9] - landmarks[0]),
        np.linalg.norm(landmarks[5] - landmarks[17]),
        np.linalg.norm(landmarks[5] - landmarks[0]),
        np.linalg.norm(landmarks[17] - landmarks[0]),
    ]
    valid = [float(value) for value in candidates if value > EPS]
    return float(np.median(valid)) if valid else 1.0


def normalize_hand(observation: HandObservation) -> NormalizedHand:
    """Create a wrist-centred, scale- and rotation-normalized local hand frame.

    The input image coordinates are retained separately so inter-hand geometry is not lost.
    Handedness is metadata only; canonical ordering never depends on it.
    """
    image = observation.landmarks.astype(np.float64, copy=True)
    origin = image[0]
    centered = image - origin
    scale = palm_scale(image)

    y_axis = _unit(centered[9])
    x_hint = centered[5] - centered[17]
    x_axis = _unit(x_hint - np.dot(x_hint, y_axis) * y_axis)
    z_axis = _unit(np.cross(x_axis, y_axis))
    if np.linalg.norm(x_axis) < EPS or np.linalg.norm(z_axis) < EPS:
        basis = np.eye(3)
    else:
        x_axis = _unit(np.cross(y_axis, z_axis))
        basis = np.stack([x_axis, y_axis, z_axis], axis=1)

    local = (centered @ basis) / scale
    palm_center = image[[0, 5, 9, 13, 17]].mean(axis=0)
    normal = _unit(np.cross(image[5] - image[0], image[17] - image[0]))
    return NormalizedHand(
        local=local.astype(np.float32),
        image=image.astype(np.float32),
        wrist=image[0].astype(np.float32),
        palm_center=palm_center.astype(np.float32),
        palm_normal=normal.astype(np.float32),
        scale=scale,
        handedness=observation.handedness,
        handedness_score=float(observation.handedness_score),
    )


def canonical_hand_order(hands: list[NormalizedHand]) -> list[NormalizedHand]:
    """Order by non-mirrored camera x position, not fallible handedness labels."""
    return sorted(hands, key=lambda hand: float(hand.wrist[0]))

