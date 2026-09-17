"""Runtime inference for the self-contained Digitra V3 hybrid package."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import mediapipe as mp
import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import transforms


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


def normalized_adjacency() -> np.ndarray:
    adjacency = np.eye(21, dtype=np.float32)
    for left, right in HAND_EDGES:
        adjacency[left, right] = adjacency[right, left] = 1.0
    degree = adjacency.sum(axis=1)
    return adjacency / np.sqrt(degree[:, None] * degree[None, :])


class GeometryMLP(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 1024), nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.20),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.17),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.12),
            nn.Linear(256, n_out),
        )

    def forward(self, values):
        return self.net(values)


class GraphGeometryNet(nn.Module):
    def __init__(self, n_out: int):
        super().__init__()
        self.register_buffer("adj", torch.tensor(normalized_adjacency()))
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

    def forward(self, values):
        image_points = values[:, :63].reshape(-1, 21, 3)
        world_points = values[:, 63:126].reshape(-1, 21, 3)
        hidden = self.node_enc(torch.cat([image_points, world_points], dim=2))
        hidden = self.g1(torch.einsum("ij,bjd->bid", self.adj, hidden))
        hidden = hidden + self.g2(torch.einsum("ij,bjd->bid", self.adj, hidden))
        weights = torch.softmax(self.attn(hidden), dim=1)
        pooled = torch.cat([hidden.mean(1), hidden.amax(1), (weights * hidden).sum(1)], dim=1)
        return self.head(torch.cat([pooled, self.global_branch(values[:, 126:])], dim=1))


class DigitraV3Recognizer:
    def __init__(self, model_dir, device: str | None = None):
        self.root = Path(model_dir)
        self.config = json.loads((self.root / "model_config.json").read_text())
        self.labels = list(self.config["labels"])
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        rgb = self.config["rgb"]
        self.rgb_model = timm.create_model(
            rgb["architecture"], pretrained=False, num_classes=len(self.labels),
            img_size=int(rgb["image_size"]), drop_rate=float(rgb["drop_rate"]),
            drop_path_rate=float(rgb["drop_path_rate"]),
        ).to(self.device)
        self.rgb_model.load_state_dict(torch.load(
            self.root / "rgb" / "best_model_state.pt", map_location=self.device
        ))
        self.rgb_model.eval()
        data_config = timm.data.resolve_model_data_config(self.rgb_model)
        size = int(rgb["image_size"])
        self.transform = transforms.Compose([
            transforms.Resize(size + 32, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(size), transforms.ToTensor(),
            transforms.Normalize(mean=data_config["mean"], std=data_config["std"]),
        ])
        self.temperature = float(rgb["temperature"])
        self.scaler = joblib.load(self.root / "landmark" / "feature_scaler.joblib")
        self.extra_trees = joblib.load(self.root / "landmark" / "extra_trees.joblib")
        self.rbf_svc = joblib.load(self.root / "landmark" / "rbf_svc.joblib")
        self.pair_experts = joblib.load(self.root / "landmark" / "pair_experts.joblib")
        self.ru_specialist = joblib.load(self.root / "router" / "ru_specialist_p1_p7.joblib")
        self.router = json.loads((self.root / "router" / "router_config.json").read_text())
        self.blend = self.config["landmark"]["blend_weights"]
        neural = torch.load(self.root / "landmark" / "neural_ensemble.pt", map_location=self.device)
        self.geometry_models = []
        for state in neural["geometry_states"]:
            model = GeometryMLP(422, len(self.labels)).to(self.device)
            model.load_state_dict(state)
            model.eval()
            self.geometry_models.append(model)
        self.graph_model = GraphGeometryNet(len(self.labels)).to(self.device)
        self.graph_model.load_state_dict(neural["graph_state"])
        self.graph_model.eval()
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.root / "hand_landmarker.task")),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.22,
            min_hand_presence_confidence=0.22,
            min_tracking_confidence=0.22,
        )
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def _load_image(self, image) -> Image.Image:
        if isinstance(image, Image.Image):
            return ImageOps.exif_transpose(image).convert("RGB")
        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                return ImageOps.exif_transpose(opened).convert("RGB")
        return Image.fromarray(np.asarray(image).astype(np.uint8)).convert("RGB")

    def _detect(self, image: Image.Image):
        rgb = np.ascontiguousarray(np.asarray(image))
        result = self.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if not result.hand_landmarks:
            return None, image
        image_points = np.asarray([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float32)
        world_points = np.asarray([[p.x, p.y, p.z] for p in result.hand_world_landmarks[0]], dtype=np.float32)
        category = result.handedness[0][0]
        feature = landmark_feature(image_points, world_points, category.category_name, float(category.score))
        width, height = image.size
        x0, x1 = image_points[:, 0].min() * width, image_points[:, 0].max() * width
        y0, y1 = image_points[:, 1].min() * height, image_points[:, 1].max() * height
        side = max(x1 - x0, y1 - y0) * 1.45
        cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
        crop = image.crop((max(0, cx-side/2), max(0, cy-side/2), min(width, cx+side/2), min(height, cy+side/2)))
        return feature, crop

    def _rgb_probability(self, image: Image.Image) -> np.ndarray:
        values = self.transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = 0.5 * self.rgb_model(values) + 0.5 * self.rgb_model(torch.flip(values, dims=[3]))
            return torch.softmax(logits / self.temperature, dim=1)[0].cpu().numpy()

    def _landmark(self, feature: np.ndarray):
        raw = feature[None].astype(np.float32)
        scaled = self.scaler.transform(raw).astype(np.float32)
        tensor = torch.from_numpy(scaled).to(self.device)
        with torch.inference_mode():
            geometry = np.mean([
                torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
                for model in self.geometry_models
            ], axis=0)
            graph = torch.softmax(self.graph_model(tensor), dim=1).cpu().numpy()[0]
        extra = self.extra_trees.predict_proba(raw)[0]
        svc = self.rbf_svc.predict_proba(scaled)[0]
        probability = (
            float(self.blend["geometry"]) * geometry
            + float(self.blend["graph"]) * graph
            + float(self.blend["extra_trees"]) * extra
            + float(self.blend["rbf_svc"]) * svc
        )
        prediction = int(probability.argmax())
        for pair_name, expert in self.pair_experts.items():
            pair = [self.labels.index(name) for name in pair_name.split("_")]
            if prediction in pair:
                prediction = int(expert.predict(scaled)[0])
        return probability, prediction, raw

    def predict(self, image) -> dict:
        opened = self._load_image(image)
        feature, crop = self._detect(opened)
        rgb_probability = self._rgb_probability(crop)
        rgb_index = int(rgb_probability.argmax())
        final_index = rgb_index
        landmark_index = None
        if feature is not None:
            landmark_probability, landmark_index, raw = self._landmark(feature)
            rgb_name, landmark_name = self.labels[rgb_index], self.labels[landmark_index]
            if rgb_name in {"M", "N"} and landmark_name in {"A", "N", "S", "T"}:
                final_index = landmark_index
            elif rgb_name == "R" and landmark_name == "U":
                r_index, u_index = self.labels.index("R"), self.labels.index("U")
                rgb_r = rgb_probability[r_index] / max(rgb_probability[r_index] + rgb_probability[u_index], 1e-12)
                specialist = self.ru_specialist.predict_proba(raw)[0]
                specialist_r = specialist[list(self.ru_specialist.classes_).index(r_index)]
                final_index = r_index if 0.53 * rgb_r + 0.47 * specialist_r >= 0.5 else u_index
        return {
            "label": self.labels[final_index],
            "confidence": float(rgb_probability[final_index]),
            "rgb_label": self.labels[rgb_index],
            "landmark_label": None if landmark_index is None else self.labels[landmark_index],
            "hand_detected": feature is not None,
        }

    def close(self):
        self.detector.close()

