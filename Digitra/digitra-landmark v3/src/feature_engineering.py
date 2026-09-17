"""Advanced feature engineering on top of MediaPipe hand landmarks.

Produces a fixed-length, label-order-consistent feature vector used by BOTH
training and real-time inference. Implements:
  - handedness canonicalization (label-independent, thumb-side flip)
  - wrist centering (translation invariance)
  - scale normalization (hand-span)
  - rotation normalization (wrist -> middle MCP aligned to +x)
  - raw canonical XYZ coords
  - bone vectors & lengths
  - joint (cosine + angle) features
  - fingertip pairwise distances & relative vectors
  - palm geometry
  - adjacent-finger direction angles
  - depth-aware features from world landmarks
  - train-only augmentation (jitter / rotation / scale noise)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import ARTIFACTS_DIR, setup_logging

# MediaPipe hand skeleton connections (parent -> child)
BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),                                  # palm base
]

# Joints at which we compute an interior angle (>=2 connections)
ADJ = {
    1: (0, 2), 2: (1, 3), 3: (2, 4),
    5: (0, 6), 6: (5, 7), 7: (6, 8),
    9: (5, 10), 10: (9, 11), 11: (10, 12),
    13: (9, 14), 14: (13, 15), 15: (14, 16),
    17: (13, 18), 18: (17, 19), 19: (18, 20),
}
JOINTS = sorted(ADJ.keys())
TIPS = [4, 8, 12, 16, 20]
FINGER_DIRS = {
    "thumb": (1, 4), "index": (5, 8), "middle": (9, 12),
    "ring": (13, 16), "pinky": (17, 20),
}
FINGER_ORDER = ["thumb", "index", "middle", "ring", "pinky"]


def canonicalize(L: np.ndarray) -> np.ndarray:
    L = np.asarray(L, dtype=float).reshape(-1, 3).copy()
    L = L - L[0]  # wrist centering -> translation invariance
    scale = float(np.linalg.norm(L[9]))
    if scale < 1e-9:
        scale = float(np.linalg.norm(L[13])) or 1.0
    L = L / scale  # scale normalization
    ang = np.arctan2(L[9, 1], L[9, 0])
    c, s = np.cos(-ang), np.sin(-ang)
    R = np.array([[c, -s], [s, c]])
    xy = L[:, :2] @ R.T  # rotation normalization
    L = np.concatenate([xy, L[:, 2:3]], axis=1)
    # handedness canonicalization (label-independent): flip so thumb is on -x
    thumb_x = float(np.mean(L[1:5, 0]))
    if thumb_x > 0:
        L[:, 0] = -L[:, 0]
    return L


def handed_code(handedness: str) -> float:
    return {"Right": 1.0, "Left": 0.0}.get(handedness or "", 0.5)


def _build(lm_c: np.ndarray, w_c, handed: float):
    feats, names = [], []

    def add(name, val):
        arr = np.asarray(val, dtype=float).ravel()
        feats.extend(arr.tolist())
        for i in range(arr.shape[0]):
            names.append(f"{name}_{i}")

    add("lm", lm_c.ravel())
    bv = np.array([lm_c[c] - lm_c[p] for p, c in BONES])
    add("bone_vec", bv.ravel())
    add("bone_len", np.linalg.norm(bv, axis=1))
    cosv, angv = [], []
    for j in JOINTS:
        a, b = ADJ[j]
        v1, v2 = lm_c[a] - lm_c[j], lm_c[b] - lm_c[j]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            cosv.append(0.0); angv.append(0.0)
        else:
            c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
            cosv.append(c); angv.append(float(np.arccos(c)))
    add("joint_cos", np.array(cosv))
    add("joint_ang", np.array(angv))
    pdist = []
    for i in range(len(TIPS)):
        for k in range(i + 1, len(TIPS)):
            pdist.append(np.linalg.norm(lm_c[TIPS[i]] - lm_c[TIPS[k]]))
        pdist.append(np.linalg.norm(lm_c[TIPS[i]] - lm_c[0]))
    add("pair_dist", np.array(pdist))
    pw = np.linalg.norm(lm_c[5] - lm_c[17])
    pl = np.linalg.norm(lm_c[0] - lm_c[9])
    add("palm_width", pw)
    add("palm_length", pl)
    add("palm_aspect", pw / pl if pl > 1e-9 else 0.0)
    ftip = np.array([lm_c[t] - lm_c[0] for t in TIPS])
    add("ftip_vec", ftip.ravel())
    dirs = {n: lm_c[b] - lm_c[a] for n, (a, b) in FINGER_DIRS.items()}
    adj = []
    for a, b in zip(FINGER_ORDER, FINGER_ORDER[1:]):
        v1, v2 = dirs[a], dirs[b]
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        c = 0.0 if (n1 < 1e-9 or n2 < 1e-9) else float(
            np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        adj.append(float(np.arccos(c)))
    add("adj_finger_ang", np.array(adj))
    add("handedness", np.array([handed]))
    if w_c is not None:
        wbv = np.array([w_c[c] - w_c[p] for p, c in BONES])
        add("w_bone_len", np.linalg.norm(wbv, axis=1))
        add("w_ftip_vec", np.array([w_c[t] - w_c[0] for t in TIPS]).ravel())
        wc, wa = [], []
        for j in JOINTS:
            a, b = ADJ[j]
            v1, v2 = w_c[a] - w_c[j], w_c[b] - w_c[j]
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1 < 1e-9 or n2 < 1e-9:
                wc.append(0.0); wa.append(0.0)
            else:
                c = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
                wc.append(c); wa.append(float(np.arccos(c)))
        add("w_joint_cos", np.array(wc))
        add("w_joint_ang", np.array(wa))
    else:
        add("w_bone_len", np.zeros(len(BONES)))
        add("w_ftip_vec", np.zeros(15))
        add("w_joint_cos", np.zeros(len(JOINTS)))
        add("w_joint_ang", np.zeros(len(JOINTS)))
    return np.array(feats, dtype=np.float32), names


def feature_names() -> list[str]:
    dummy = np.zeros((21, 3), dtype=float)
    dummy[9] = [1.0, 0.0, 0.0]
    dummy[4] = [0.5, 0.5, 0.0]
    _, names = _build(canonicalize(dummy), canonicalize(dummy), 0.5)
    return names


def compute_features(lm: np.ndarray, world, handedness: str) -> np.ndarray:
    lm_c = canonicalize(lm)
    w_c = None
    if world is not None and not (isinstance(world, float) and np.isnan(world)):
        w = np.asarray(world, dtype=float).reshape(-1, 3)
        if not np.isnan(w).any():
            w_c = canonicalize(w)
    feats, _ = _build(lm_c, w_c, handed_code(handedness))
    return feats


def augment_landmarks(lm: np.ndarray, world, rng: np.random.Generator):
    lm = lm.astype(float).copy()
    w = world.astype(float).copy() if world is not None else None
    ang = rng.uniform(-np.pi / 18, np.pi / 18)
    c, s = np.cos(ang), np.sin(ang)
    R = np.array([[c, -s], [s, c]])
    lm[:, :2] = lm[:, :2] @ R.T
    if w is not None:
        w[:, :2] = w[:, :2] @ R.T
    sc = rng.uniform(0.9, 1.1)
    lm *= sc
    if w is not None:
        w *= sc
    lm += rng.normal(0, 0.01, lm.shape)
    if w is not None:
        w += rng.normal(0, 0.01, w.shape)
    return lm, w


def build_feature_matrix(df: pd.DataFrame, augment: bool = False,
                         n_aug: int = 0, rng: np.random.Generator | None = None):
    """Return (X, labels) for the given dataframe."""
    logger = setup_logging()
    rng = rng or np.random.default_rng(42)
    X_rows, labels = [], []
    iterator = tqdm(df.iterrows(), total=len(df), desc="Features", unit="row",
                    mininterval=2.0)
    for _, r in iterator:
        lm = np.asarray(r["landmarks"], dtype=float).reshape(21, 3)
        world = r["world_landmarks"]
        if world is None or (isinstance(world, float) and np.isnan(world)):
            world = None
        else:
            world = np.asarray(world, dtype=float).reshape(21, 3)
            if np.isnan(world).any():
                world = None
        X_rows.append(compute_features(lm, world, r["handedness"]))
        labels.append(r["label"])
        if augment and n_aug > 0:
            for _ in range(n_aug):
                alm, aw = augment_landmarks(lm, world, rng)
                X_rows.append(compute_features(alm, aw, r["handedness"]))
                labels.append(r["label"])
    X = np.vstack(X_rows)
    X = np.where(np.isfinite(X), X, np.nan)
    n_nan = int(np.isnan(X).sum())
    n_inf = int(np.isinf(X).sum())
    if n_nan or n_inf:
        logger.warning("Feature matrix NaN=%d Inf=%d -> imputed with 0", n_nan, n_inf)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, labels


def save_metadata() -> dict:
    from utils import ensure_dirs
    ensure_dirs()
    names = feature_names()
    meta = {
        "n_features": len(names),
        "feature_names": names,
        "normalization": [
            "wrist centering (translation invariance)",
            "hand-span scale normalization",
            "rotation normalization (wrist->middle MCP to +x)",
            "handedness canonicalization (thumb-side flip)",
        ],
        "engineered_groups": {
            "raw_canonical_xyz": 63,
            "bone_vectors": 63,
            "bone_lengths": 21,
            "joint_angles_cos": 15,
            "joint_angles_rad": 15,
            "fingertip_pairwise_dist": 15,
            "palm_geometry": 3,
            "fingertip_relative_vectors": 15,
            "adjacent_finger_angles": 4,
            "handedness_code": 1,
            "world_bone_lengths": 21,
            "world_fingertip_vectors": 15,
            "world_joint_angles_cos": 15,
            "world_joint_angles_rad": 15,
        },
    }
    with open(ARTIFACTS_DIR / "feature_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    return meta


if __name__ == "__main__":
    m = save_metadata()
    print("Total features:", m["n_features"])
