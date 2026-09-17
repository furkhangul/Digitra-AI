from __future__ import annotations

from itertools import pairwise

import numpy as np

# MediaPipe hand skeleton connections used by the existing 36-class dataset.
BONES = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
]
ADJ = {
    1: (0, 2),
    2: (1, 3),
    3: (2, 4),
    5: (0, 6),
    6: (5, 7),
    7: (6, 8),
    9: (5, 10),
    10: (9, 11),
    11: (10, 12),
    13: (9, 14),
    14: (13, 15),
    15: (14, 16),
    17: (13, 18),
    18: (17, 19),
    19: (18, 20),
}
JOINTS = sorted(ADJ)
TIPS = [4, 8, 12, 16, 20]
FINGER_DIRECTIONS = [(1, 4), (5, 8), (9, 12), (13, 16), (17, 20)]
ASL_FEATURE_COUNT = 281


def canonicalize(landmarks: np.ndarray) -> np.ndarray:
    """Make one hand translation, scale, rotation and mirror invariant."""
    points = np.asarray(landmarks, dtype=float).reshape(21, 3).copy()
    points -= points[0]
    scale = float(np.linalg.norm(points[9]))
    if scale < 1e-9:
        scale = float(np.linalg.norm(points[13])) or 1.0
    points /= scale

    angle = np.arctan2(points[9, 1], points[9, 0])
    cosine, sine = np.cos(-angle), np.sin(-angle)
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    points[:, :2] = points[:, :2] @ rotation.T

    # Canonical thumb side lets left- and right-handed users share one model.
    if float(np.mean(points[1:5, 0])) > 0:
        points[:, 0] *= -1
    return points


def _handedness_code(handedness: str) -> float:
    return {"Right": 1.0, "Left": 0.0}.get(handedness or "", 0.5)


def _angles(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cosines: list[float] = []
    radians: list[float] = []
    for joint in JOINTS:
        first, second = ADJ[joint]
        vector_a = points[first] - points[joint]
        vector_b = points[second] - points[joint]
        norm_a, norm_b = np.linalg.norm(vector_a), np.linalg.norm(vector_b)
        if norm_a < 1e-9 or norm_b < 1e-9:
            cosine = 0.0
        else:
            cosine = float(
                np.clip(np.dot(vector_a, vector_b) / (norm_a * norm_b), -1.0, 1.0)
            )
        cosines.append(cosine)
        radians.append(float(np.arccos(cosine)))
    return np.asarray(cosines), np.asarray(radians)


def _build_features(
    image_points: np.ndarray,
    world_points: np.ndarray | None,
    handedness: float,
) -> np.ndarray:
    features: list[float] = []

    def add(values: object) -> None:
        features.extend(np.asarray(values, dtype=float).ravel().tolist())

    add(image_points.ravel())
    bone_vectors = np.asarray([image_points[child] - image_points[parent] for parent, child in BONES])
    add(bone_vectors.ravel())
    add(np.linalg.norm(bone_vectors, axis=1))
    cosines, radians = _angles(image_points)
    add(cosines)
    add(radians)

    distances: list[float] = []
    for tip_index, tip in enumerate(TIPS):
        for other in TIPS[tip_index + 1 :]:
            distances.append(float(np.linalg.norm(image_points[tip] - image_points[other])))
        distances.append(float(np.linalg.norm(image_points[tip] - image_points[0])))
    add(distances)

    palm_width = float(np.linalg.norm(image_points[5] - image_points[17]))
    palm_length = float(np.linalg.norm(image_points[0] - image_points[9]))
    add(palm_width)
    add(palm_length)
    add(palm_width / palm_length if palm_length > 1e-9 else 0.0)
    add(np.asarray([image_points[tip] - image_points[0] for tip in TIPS]).ravel())

    directions = [image_points[end] - image_points[start] for start, end in FINGER_DIRECTIONS]
    adjacent_angles: list[float] = []
    for first, second in pairwise(directions):
        norm_first, norm_second = np.linalg.norm(first), np.linalg.norm(second)
        if norm_first < 1e-9 or norm_second < 1e-9:
            cosine = 0.0
        else:
            cosine = float(
                np.clip(np.dot(first, second) / (norm_first * norm_second), -1.0, 1.0)
            )
        adjacent_angles.append(float(np.arccos(cosine)))
    add(adjacent_angles)
    add(handedness)

    if world_points is None:
        add(np.zeros(len(BONES)))
        add(np.zeros(15))
        add(np.zeros(len(JOINTS)))
        add(np.zeros(len(JOINTS)))
    else:
        world_bones = np.asarray(
            [world_points[child] - world_points[parent] for parent, child in BONES]
        )
        add(np.linalg.norm(world_bones, axis=1))
        add(np.asarray([world_points[tip] - world_points[0] for tip in TIPS]).ravel())
        world_cosines, world_radians = _angles(world_points)
        add(world_cosines)
        add(world_radians)

    result = np.nan_to_num(
        np.asarray(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    if result.shape != (ASL_FEATURE_COUNT,):
        raise RuntimeError(f"Expected {ASL_FEATURE_COUNT} ASL features, got {result.shape}")
    return result


def compute_asl_features(
    landmarks: np.ndarray,
    world_landmarks: np.ndarray | None = None,
    handedness: str = "Unknown",
) -> np.ndarray:
    image_points = canonicalize(landmarks)
    world_points = None
    if world_landmarks is not None:
        candidate = np.asarray(world_landmarks, dtype=float).reshape(21, 3)
        if np.isfinite(candidate).all():
            world_points = canonicalize(candidate)
    return _build_features(image_points, world_points, _handedness_code(handedness))
