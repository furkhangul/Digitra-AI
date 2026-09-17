from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .landmark_normalizer import NormalizedHand, canonical_hand_order, normalize_hand
from .schemas import HandObservation

FINGERS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
TIPS = [4, 8, 12, 16, 20]
TIP_PAIRS = [(4, 8), (4, 12), (4, 16), (4, 20), (8, 12), (12, 16), (16, 20)]
EPS = 1e-8


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    first, second = a - b, c - b
    denom = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denom < EPS:
        return 0.0
    cosine = float(np.clip(np.dot(first, second) / denom, -1.0, 1.0))
    return float(np.arccos(cosine) / np.pi)


def _hand_features(hand: NormalizedHand) -> tuple[list[float], list[str]]:
    lm = hand.local
    values = lm.ravel().astype(float).tolist()
    names = [f"local_lm_{index}_{axis}" for index in range(21) for axis in "xyz"]

    for finger, joints in FINGERS.items():
        chain = (0, *joints) if finger == "thumb" else joints
        angles = [_angle(lm[chain[i]], lm[chain[i + 1]], lm[chain[i + 2]])
                  for i in range(len(chain) - 2)]
        while len(angles) < 3:
            angles.append(0.0)
        for index, value in enumerate(angles[:3]):
            values.append(value)
            names.append(f"{finger}_joint_angle_{index}")

        tip_distance = float(np.linalg.norm(lm[joints[-1]] - lm[0]))
        chain_length = sum(
            float(np.linalg.norm(lm[b] - lm[a]))
            for a, b in zip((0, *joints[:-1]), joints)
        )
        extension = tip_distance / max(chain_length, EPS)
        values.extend([tip_distance, extension])
        names.extend([f"{finger}_tip_wrist", f"{finger}_extension"])

    for first, second in TIP_PAIRS:
        values.append(float(np.linalg.norm(lm[first] - lm[second])))
        names.append(f"tip_distance_{first}_{second}")

    handed = hand.handedness.lower()
    values.extend([
        1.0,
        1.0 if handed == "left" else 0.0,
        1.0 if handed == "right" else 0.0,
        float(hand.handedness_score),
        *hand.palm_normal.astype(float).tolist(),
    ])
    names.extend([
        "present", "handed_left", "handed_right", "handedness_score",
        "palm_normal_x", "palm_normal_y", "palm_normal_z",
    ])
    return values, names


def _empty_hand_features() -> tuple[list[float], list[str]]:
    dummy = HandObservation(np.zeros((21, 3), dtype=np.float32))
    dummy.landmarks[9, 1] = 1.0
    dummy.landmarks[5, 0] = 0.4
    dummy.landmarks[17, 0] = -0.4
    values, names = _hand_features(normalize_hand(dummy))
    return [0.0] * len(values), names


def _inter_hand_features(
    first: NormalizedHand | None, second: NormalizedHand | None
) -> tuple[list[float], list[str]]:
    names = [
        "hand_count", "wrist_distance", "palm_center_distance",
        "relative_wrist_x", "relative_wrist_y", "relative_wrist_z",
        "relative_palm_x", "relative_palm_y", "relative_palm_z",
        "palm_normal_cosine",
    ]
    names += [f"cross_tip_distance_{a}_{b}" for a in TIPS for b in TIPS]
    names += [f"cross_tip_contact_{a}_{b}" for a in TIPS for b in TIPS]

    count = int(first is not None) + int(second is not None)
    if first is None or second is None:
        return [float(count)] + [0.0] * (len(names) - 1), names

    shared_scale = max((first.scale + second.scale) / 2.0, EPS)
    relative_wrist = (second.wrist - first.wrist) / shared_scale
    relative_palm = (second.palm_center - first.palm_center) / shared_scale
    wrist_distance = float(np.linalg.norm(relative_wrist))
    palm_distance = float(np.linalg.norm(relative_palm))
    normal_cosine = float(np.clip(np.dot(first.palm_normal, second.palm_normal), -1.0, 1.0))

    distances = [
        float(np.linalg.norm(first.image[a] - second.image[b]) / shared_scale)
        for a in TIPS for b in TIPS
    ]
    contacts = [1.0 if distance <= 0.18 else 0.0 for distance in distances]
    values = [
        2.0, wrist_distance, palm_distance,
        *relative_wrist.astype(float).tolist(),
        *relative_palm.astype(float).tolist(),
        normal_cosine,
        *distances,
        *contacts,
    ]
    return values, names


@dataclass(frozen=True)
class FeatureResult:
    vector: np.ndarray
    names: tuple[str, ...]
    hands_detected: int
    diagnostics: dict[str, float]


def extract_features(observations: list[HandObservation]) -> FeatureResult:
    if len(observations) > 2:
        raise ValueError("At most two hands are supported")

    normalized = canonical_hand_order([normalize_hand(item) for item in observations])
    empty_values, hand_names = _empty_hand_features()
    values: list[float] = []
    names: list[str] = []
    for slot in range(2):
        if slot < len(normalized):
            slot_values, slot_names = _hand_features(normalized[slot])
        else:
            slot_values, slot_names = empty_values, hand_names
        values.extend(slot_values)
        names.extend([f"hand_{slot}_{name}" for name in slot_names])

    first = normalized[0] if normalized else None
    second = normalized[1] if len(normalized) > 1 else None
    inter_values, inter_names = _inter_hand_features(first, second)
    values.extend(inter_values)
    names.extend(inter_names)
    vector = np.nan_to_num(np.asarray(values, dtype=np.float32))
    diagnostics = {
        "hands_detected": float(len(normalized)),
        "wrist_distance": float(inter_values[1]),
        "palm_center_distance": float(inter_values[2]),
        "minimum_cross_tip_distance": (
            float(min(inter_values[10:35])) if len(normalized) == 2 else 0.0
        ),
    }
    return FeatureResult(vector, tuple(names), len(normalized), diagnostics)


def feature_names() -> tuple[str, ...]:
    return extract_features([]).names

