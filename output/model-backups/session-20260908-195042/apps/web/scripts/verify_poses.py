# Verify the Eğitim section's letter poses against the trained DIGITRA
# landmark classifier: each pose's 21 rig joints are converted to the
# canonical_422_v1 feature and run through the extra-trees model. Letters the
# model misclassifies get their curls tuned until they read correctly.
#
#   python scripts/verify_poses.py [Right|Left]

import json
import sys
from pathlib import Path

import joblib
import numpy as np

MODEL_DIR = Path(r"C:\Users\Furkan\Desktop\Digitra AI\models\DIGITRA_V3_HYBRID_9415")
DUMP_DIR = Path(__file__).parent / "pose_dumps"

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15),
    (15, 16), (0, 17), (17, 18), (18, 19), (19, 20),
]
FINGER_CHAINS = [
    [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12],
    [0, 13, 14, 15, 16], [0, 17, 18, 19, 20],
]
TIP_IDS = [4, 8, 12, 16, 20]


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
    image_canonical = canonical_points(image_points)
    world_canonical = canonical_points(world_points)
    pairwise = [
        np.linalg.norm(world_canonical[left] - world_canonical[right])
        for left in range(21) for right in range(left + 1, 21)
    ]
    bones = []
    for left, right in HAND_EDGES:
        bones.extend(unit_vector(world_canonical[right] - world_canonical[left]).tolist())
    joint_cosines = []
    for chain in FINGER_CHAINS:
        for index in range(1, len(chain) - 1):
            first = unit_vector(world_canonical[chain[index - 1]] - world_canonical[chain[index]])
            second = unit_vector(world_canonical[chain[index + 1]] - world_canonical[chain[index]])
            joint_cosines.append(float(np.clip(np.dot(first, second), -1.0, 1.0)))
    tip_geometry = [float(np.linalg.norm(world_canonical[index])) for index in TIP_IDS]
    tip_geometry += [
        float(np.linalg.norm(world_canonical[4] - world_canonical[index]))
        for index in TIP_IDS[1:]
    ]
    handed = [
        1.0 if str(handedness).lower() == "left" else 0.0,
        float(handedness_score),
    ]
    feature = np.concatenate([
        image_canonical.reshape(-1), world_canonical.reshape(-1),
        np.asarray(pairwise, dtype=np.float32),
        np.asarray(bones, dtype=np.float32),
        np.asarray(joint_cosines, dtype=np.float32),
        np.asarray(tip_geometry, dtype=np.float32),
        np.asarray(handed, dtype=np.float32),
    ]).astype(np.float32)
    if feature.shape != (422,):
        raise RuntimeError(f"Expected 422 features, got {feature.shape}")
    return feature


def synthetic_landmarks(points):
    """Rebase the rig's finger chains onto an anatomically spread palm.

    The stylised rig has every finger root at one knuckle point, which
    collapses the classifier's canonical frame (it measures index-MCP →
    pinky-MCP). Finger shapes (curls) are preserved; only the bases move to
    real MCP positions.
    """
    points = np.asarray(points, dtype=np.float32)
    wrist = points[0]
    along = unit_vector(points[9] - wrist)          # knuckle centre → middle base
    side = unit_vector(points[20] - points[8])      # index tip → pinky tip
    hs = float(np.linalg.norm(points[9] - wrist))   # palm-length unit
    out = np.zeros((21, 3), dtype=np.float32)
    out[0] = wrist
    # thumb: low, well to the thumb side
    out[1:5] = points[1:5] + (wrist + along * 0.30 * hs - side * 0.55 * hs - points[1])
    chains = [(5, -0.38, 0.46), (9, 0.0, 0.50), (13, 0.30, 0.47), (17, 0.55, 0.40)]
    for rig_start, lateral, lift in chains:
        base = wrist + along * (lift * hs) + side * (lateral * hs)
        out[rig_start] = base
        out[rig_start + 1:rig_start + 4] = points[rig_start + 1:rig_start + 4] + (base - points[rig_start])
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    handedness = sys.argv[1] if len(sys.argv) > 1 else "Right"
    labels = json.loads((MODEL_DIR / "model_config.json").read_text(encoding="utf-8"))["labels"]
    scaler = joblib.load(MODEL_DIR / "landmark" / "feature_scaler.joblib")
    extra_trees = joblib.load(MODEL_DIR / "landmark" / "extra_trees.joblib")

    rows, correct = [], 0
    mirror = len(sys.argv) > 2 and sys.argv[2] == "mirror"
    for path in sorted(DUMP_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = np.asarray(data["points"], dtype=np.float32).copy()
        if mirror:
            raw[:, 0] *= -1.0
        points = synthetic_landmarks(raw)
        feature = landmark_feature(points, points.copy(), handedness, 0.95)
        scaled = scaler.transform(feature[None].astype(np.float32))
        proba = extra_trees.predict_proba(scaled)[0]
        order = proba.argsort()[::-1]
        predicted = labels[int(extra_trees.classes_[order[0]])]
        second = labels[int(extra_trees.classes_[order[1]])]
        ok = predicted == data["expected"]
        correct += ok
        mark = "OK  " if ok else "MISS"
        rows.append(
            f"{mark} {data['target']:>2} → beklenen {data['expected']:>2} | "
            f"tahmin {predicted} ({proba[order[0]]:.2f}) | ikinci {second} ({proba[order[1]]:.2f})"
        )

    print("\n".join(rows))
    print(f"\n{correct}/{len(rows)} doğru ({handedness})")


if __name__ == "__main__":
    main()
