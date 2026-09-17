from __future__ import annotations

import numpy as np

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
]

ANGLE_TRIPLES = [
    (0, 1, 2), (1, 2, 3), (2, 3, 4),
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
]


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    u = a - b
    v = c - b
    denom = float(np.linalg.norm(u) * np.linalg.norm(v))
    if denom < 1e-8:
        return 0.0
    return float(np.clip(np.dot(u, v) / denom, -1.0, 1.0))


def _local_features(points: np.ndarray | None) -> np.ndarray:
    if points is None:
        return np.zeros(63 + len(HAND_CONNECTIONS) + len(ANGLE_TRIPLES), dtype=np.float32)
    centered = points - points[0]
    scale = float(np.max(np.linalg.norm(centered[:, :2], axis=1)))
    if scale < 1e-8:
        scale = 1.0
    norm = centered / scale
    bones = np.asarray([
        np.linalg.norm(norm[b] - norm[a]) for a, b in HAND_CONNECTIONS
    ], dtype=np.float32)
    angles = np.asarray([
        _angle(norm[a], norm[b], norm[c]) for a, b, c in ANGLE_TRIPLES
    ], dtype=np.float32)
    return np.concatenate([norm.reshape(-1), bones, angles]).astype(np.float32)


def build_two_hand_features(hands: list[np.ndarray]) -> np.ndarray:
    """Build a fixed vector while preserving inter-hand geometry.

    Hands are ordered from left to right in image coordinates, independent of
    MediaPipe handedness labels. This is stable for the mirrored webcam path.
    """
    ordered = sorted(hands[:2], key=lambda pts: float(pts[0, 0]))
    left = ordered[0] if ordered else None
    right = ordered[1] if len(ordered) > 1 else None

    local = np.concatenate([_local_features(left), _local_features(right)])

    raw = []
    for hand in (left, right):
        raw.append(hand if hand is not None else np.zeros((21, 3), dtype=np.float32))
    stacked = np.concatenate(raw, axis=0)

    present = np.asarray([left is not None, right is not None], dtype=np.float32)
    real_points = np.concatenate([h for h in (left, right) if h is not None], axis=0)
    center = np.mean(real_points[:, :2], axis=0)
    centered = stacked.copy()
    for hand_idx, hand in enumerate((left, right)):
        if hand is None:
            continue
        sl = slice(hand_idx * 21, (hand_idx + 1) * 21)
        centered[sl, :2] -= center
    scale = float(np.max(np.linalg.norm(centered[present.repeat(21).astype(bool), :2], axis=1)))
    if scale < 1e-8:
        scale = 1.0
    centered /= scale

    relation = np.zeros(4, dtype=np.float32)
    if left is not None and right is not None:
        delta = (right[0, :2] - left[0, :2]) / scale
        relation[:2] = delta
        relation[2] = float(np.linalg.norm(delta))
        relation[3] = float(right[0, 2] - left[0, 2]) / scale

    return np.concatenate([local, centered.reshape(-1), present, relation]).astype(np.float32)


N_FEATURES = 2 * (63 + len(HAND_CONNECTIONS) + len(ANGLE_TRIPLES)) + 126 + 2 + 4
