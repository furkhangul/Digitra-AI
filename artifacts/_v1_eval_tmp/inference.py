from pathlib import Path
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
import mediapipe as mp

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
FINGER_CHAINS = [[0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [0, 9, 10, 11, 12], [0, 13, 14, 15, 16], [0, 17, 18, 19, 20]]
TIP_IDS = [4, 8, 12, 16, 20]

def unit_vector(v, eps=1e-7):
    return v / max(float(np.linalg.norm(v)), eps)

def canonical_points(points):
    p = np.asarray(points, dtype=np.float32) - np.asarray(points, dtype=np.float32)[0]
    ex = unit_vector(p[5] - p[17])
    y_hint = unit_vector(p[9])
    ez = unit_vector(np.cross(ex, y_hint))
    if np.linalg.norm(ez) < 1e-5:
        ez = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    ey = unit_vector(np.cross(ez, ex))
    coords = np.stack([p @ ex, p @ ey, p @ ez], axis=1)
    scale = np.mean([np.linalg.norm(p[i]) for i in [5, 9, 13, 17]])
    return coords / max(float(scale), 1e-6)

def landmark_feature_from_arrays(image_points, world_points, handedness, handedness_score):
    ci = canonical_points(image_points)
    cw = canonical_points(world_points)
    pairwise = []
    for i in range(21):
        for j in range(i + 1, 21):
            pairwise.append(np.linalg.norm(cw[i] - cw[j]))
    bone_vectors = []
    for a, b in HAND_EDGES:
        bone_vectors.extend(unit_vector(cw[b] - cw[a]).tolist())
    joint_cosines = []
    for chain in FINGER_CHAINS:
        for j in range(1, len(chain) - 1):
            v1 = unit_vector(cw[chain[j - 1]] - cw[chain[j]])
            v2 = unit_vector(cw[chain[j + 1]] - cw[chain[j]])
            joint_cosines.append(float(np.clip(np.dot(v1, v2), -1.0, 1.0)))
    tip_geometry = [float(np.linalg.norm(cw[t])) for t in TIP_IDS]
    tip_geometry += [float(np.linalg.norm(cw[4] - cw[t])) for t in TIP_IDS[1:]]
    handed = [1.0 if str(handedness).lower() == "left" else 0.0, float(handedness_score)]
    feature = np.concatenate([
        ci.reshape(-1), cw.reshape(-1),
        np.asarray(pairwise, dtype=np.float32),
        np.asarray(bone_vectors, dtype=np.float32),
        np.asarray(joint_cosines, dtype=np.float32),
        np.asarray(tip_geometry, dtype=np.float32),
        np.asarray(handed, dtype=np.float32),
    ]).astype(np.float32)
    if feature.shape != (422,):
        raise RuntimeError(f"Feature contract violated: expected 422, received {feature.shape}")
    return feature

class GeometryMLP(nn.Module):
    def __init__(self, n_in=422, n_out=28):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.18),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.16),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.12),
            nn.Linear(256, n_out),
        )
    def forward(self, x):
        return self.net(x)

_GRAPH_ADJ = np.eye(21, dtype=np.float32)
for _a, _b in HAND_EDGES:
    _GRAPH_ADJ[_a, _b] = _GRAPH_ADJ[_b, _a] = 1.0
_GRAPH_DEG = np.sum(_GRAPH_ADJ, axis=1)
_GRAPH_ADJ = _GRAPH_ADJ / np.sqrt(_GRAPH_DEG[:, None] * _GRAPH_DEG[None, :])

class GraphGeometryNet(nn.Module):
    def __init__(self, n_out=28):
        super().__init__()
        self.register_buffer("adj", torch.tensor(_GRAPH_ADJ))
        self.node_enc = nn.Sequential(nn.Linear(6, 128), nn.LayerNorm(128), nn.GELU())
        self.g1 = nn.Sequential(nn.Linear(128, 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(0.10))
        self.g2 = nn.Sequential(nn.Linear(192, 192), nn.LayerNorm(192), nn.GELU(), nn.Dropout(0.10))
        self.attn = nn.Linear(192, 1)
        self.global_branch = nn.Sequential(
            nn.Linear(296, 384), nn.BatchNorm1d(384), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(384, 256), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(192 * 3 + 256, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.17),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.10), nn.Linear(256, n_out),
        )
    def forward(self, x):
        ci = x[:, :63].reshape(-1, 21, 3)
        cw = x[:, 63:126].reshape(-1, 21, 3)
        h = self.node_enc(torch.cat([ci, cw], dim=2))
        h = self.g1(torch.einsum("ij,bjd->bid", self.adj, h))
        h = h + self.g2(torch.einsum("ij,bjd->bid", self.adj, h))
        w = torch.softmax(self.attn(h), dim=1)
        pooled = torch.cat([h.mean(1), h.amax(1), (w * h).sum(1)], dim=1)
        glob = self.global_branch(x[:, 126:])
        return self.head(torch.cat([pooled, glob], dim=1))

class DigitraRecognizer:
    def __init__(self, bundle_dir, device="cpu", initialize_detector=True):
        self.bundle_dir = Path(bundle_dir)
        self.device = torch.device(device)
        with open(self.bundle_dir / "metadata.json", encoding="utf-8") as handle:
            self.metadata = json.load(handle)
        neural = torch.load(self.bundle_dir / "neural_ensemble.pt", map_location=self.device, weights_only=False)
        self.classes = list(neural["classes"])
        if len(self.classes) != 28:
            raise RuntimeError(f"Expected 28 learned classes, got {len(self.classes)}")
        self.scaler = joblib.load(self.bundle_dir / "feature_scaler.joblib")
        self.extra_trees = joblib.load(self.bundle_dir / "extra_trees.joblib")
        expert_bundle = joblib.load(self.bundle_dir / "pair_experts.joblib")
        self.expert_pairs = [tuple(pair) for pair in expert_bundle["pairs"]]
        self.expert_specs = {tuple(pair): value for pair, value in expert_bundle["specs"].items()}
        self.geometry_models = []
        for state in neural["geometry_states"]:
            model = GeometryMLP(n_in=int(neural["input_dim"]), n_out=len(self.classes)).to(self.device)
            model.load_state_dict(state)
            model.eval()
            self.geometry_models.append(model)
        self.graph_model = GraphGeometryNet(n_out=len(self.classes)).to(self.device)
        self.graph_model.load_state_dict(neural["graph_state"])
        self.graph_model.eval()
        blend = self.metadata.get("blend", {})
        self.geometry_weight = float(blend.get("geometry", 0.8075))
        self.extra_trees_weight = float(blend.get("extra_trees", 0.0425))
        self.graph_weight = float(blend.get("graph", 0.15))
        if not np.isclose(self.geometry_weight + self.extra_trees_weight + self.graph_weight, 1.0):
            raise RuntimeError("Model blend weights do not sum to one")
        self.detector = None
        if initialize_detector:
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(self.bundle_dir / "hand_landmarker.task")),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_hands=1,
                min_hand_detection_confidence=0.30,
                min_hand_presence_confidence=0.30,
                min_tracking_confidence=0.30,
            )
            self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def predict_feature(self, feature):
        raw = np.asarray(feature, dtype=np.float32).reshape(1, -1)
        if raw.shape[1] != 422:
            raise ValueError(f"Expected 422 features, got {raw.shape[1]}")
        scaled = self.scaler.transform(raw).astype(np.float32)
        x = torch.from_numpy(scaled).to(self.device)
        with torch.inference_mode():
            geom_probability = np.mean(
                [torch.softmax(model(x), dim=1).cpu().numpy() for model in self.geometry_models], axis=0
            )
            graph_probability = torch.softmax(self.graph_model(x), dim=1).cpu().numpy()
        et_small = self.extra_trees.predict_proba(raw)
        et_probability = np.zeros((1, len(self.classes)), dtype=np.float64)
        et_probability[:, np.asarray(self.extra_trees.classes_, dtype=int)] = et_small
        probability = (
            self.geometry_weight * geom_probability
            + self.extra_trees_weight * et_probability
            + self.graph_weight * graph_probability
        )
        base_index = int(np.argmax(probability[0]))
        final_index = base_index
        expert_applied = None
        for pair in self.expert_pairs:
            if final_index not in pair:
                continue
            pair_spec = self.expert_specs[pair]
            if pair_spec["mode"] == "raw_subset":
                expert_input = raw[:, np.asarray(pair_spec["idx"], dtype=int)]
            elif pair_spec["mode"] == "scaled_all":
                expert_input = scaled
            else:
                raise RuntimeError(f"Unknown expert mode: {pair_spec['mode']}")
            final_index = int(pair_spec["model"].predict(expert_input)[0])
            expert_applied = [self.classes[pair[0]], self.classes[pair[1]]]
        return {
            "label": self.classes[final_index],
            "class_index": final_index,
            "base_label": self.classes[base_index],
            "confidence": float(probability[0, final_index]),
            "expert_applied": expert_applied,
            "probabilities": {label: float(score) for label, score in zip(self.classes, probability[0])},
        }

    @staticmethod
    def _to_rgb_array(image):
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                return np.asarray(ImageOps.exif_transpose(opened).convert("RGB"))
        if isinstance(image, Image.Image):
            return np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError("Expected an RGB/RGBA image array")
        if array.shape[2] == 4:
            array = array[:, :, :3]
        return np.ascontiguousarray(array.astype(np.uint8, copy=False))

    def extract_feature(self, image):
        if self.detector is None:
            raise RuntimeError("Detector was not initialized")
        rgb = self._to_rgb_array(image)
        result = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.hand_landmarks:
            return None, {"detected": False}
        image_points = np.asarray([[point.x, point.y, point.z] for point in result.hand_landmarks[0]], dtype=np.float32)
        world_points = np.asarray([[point.x, point.y, point.z] for point in result.hand_world_landmarks[0]], dtype=np.float32)
        category = result.handedness[0][0]
        feature = landmark_feature_from_arrays(
            image_points, world_points, category.category_name, float(category.score)
        )
        return feature, {
            "detected": True,
            "handedness": category.category_name,
            "handedness_score": float(category.score),
        }

    def predict_image(self, image):
        feature, detection = self.extract_feature(image)
        if feature is None:
            return {
                "label": "nothing", "class_index": None, "base_label": "nothing",
                "confidence": 1.0, "expert_applied": None, "probabilities": {}, **detection,
            }
        return {**self.predict_feature(feature), **detection}

    def close(self):
        if self.detector is not None:
            self.detector.close()
            self.detector = None
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
