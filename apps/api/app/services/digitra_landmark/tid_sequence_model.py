from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def temporal_embedding(features: np.ndarray) -> np.ndarray:
    """Summarize pose and motion while preserving per-feature semantics."""
    sequence = np.asarray(features, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[0] < 2:
        raise ValueError("Expected a [time, features] sequence with at least two frames")
    velocity = np.diff(sequence, axis=0)
    parts = [
        np.mean(sequence, axis=0),
        np.std(sequence, axis=0),
        np.quantile(sequence, 0.10, axis=0),
        np.quantile(sequence, 0.90, axis=0),
        sequence[-1] - sequence[0],
        np.mean(np.abs(velocity), axis=0),
        np.std(velocity, axis=0),
    ]
    return np.nan_to_num(np.concatenate(parts).astype(np.float32))


def _knn_distances(projected: np.ndarray, neighbors: int = 3) -> np.ndarray:
    differences = projected[:, None, :] - projected[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    np.fill_diagonal(distances, np.inf)
    count = min(neighbors, len(projected) - 1)
    nearest = np.partition(distances, count - 1, axis=1)[:, :count]
    return np.mean(nearest, axis=1)


def fit_sequence_verifier(
    sequences: list[np.ndarray],
    label: str,
) -> dict[str, object]:
    if len(sequences) < 10:
        raise ValueError("At least 10 positive sequences are required")
    embeddings = np.stack([temporal_embedding(sequence) for sequence in sequences])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)
    component_count = min(8, len(sequences) - 1, scaled.shape[1])
    pca = PCA(n_components=component_count, random_state=42)
    projected = pca.fit_transform(scaled)
    leave_one_out_distances = _knn_distances(projected)
    threshold = max(float(np.quantile(leave_one_out_distances, 0.95) * 1.10), 1e-6)
    return {
        "model_type": "positive_knn_sequence_verifier",
        "label": label,
        "sequence_frames": int(sequences[0].shape[0]),
        "feature_count": int(sequences[0].shape[1]),
        "embedding_count": int(embeddings.shape[1]),
        "scaler": scaler,
        "pca": pca,
        "prototypes": projected,
        "neighbors": 3,
        "distance_threshold": threshold,
        "calibration": {
            "sequence_count": len(sequences),
            "median_leave_one_out_distance": float(np.median(leave_one_out_distances)),
            "maximum_leave_one_out_distance": float(np.max(leave_one_out_distances)),
            "threshold": threshold,
        },
    }


class TIDSequenceVerifier:
    def __init__(self, model: Path | str | dict[str, object]) -> None:
        payload = joblib.load(Path(model)) if isinstance(model, (Path, str)) else model
        self.label = str(payload["label"])
        self.sequence_frames = int(payload["sequence_frames"])
        self.feature_count = int(payload["feature_count"])
        self.scaler = payload["scaler"]
        self.pca = payload["pca"]
        self.prototypes = np.asarray(payload["prototypes"], dtype=np.float32)
        self.neighbors = int(payload["neighbors"])
        self.distance_threshold = float(payload["distance_threshold"])
        self.calibration = payload.get("calibration", {})

    def evaluate(self, features: np.ndarray) -> dict[str, float | bool | str]:
        sequence = np.asarray(features, dtype=np.float32)
        expected_shape = (self.sequence_frames, self.feature_count)
        if sequence.shape != expected_shape:
            raise ValueError(f"Expected sequence shape {expected_shape}, got {sequence.shape}")
        embedding = temporal_embedding(sequence).reshape(1, -1)
        projected = self.pca.transform(self.scaler.transform(embedding))[0]
        distances = np.linalg.norm(self.prototypes - projected, axis=1)
        count = min(self.neighbors, len(distances))
        distance = float(np.mean(np.partition(distances, count - 1)[:count]))
        similarity = float(2.0 ** (-distance / self.distance_threshold))
        return {
            "label": self.label,
            "accepted": distance <= self.distance_threshold,
            "similarity": similarity,
            "distance": distance,
            "threshold": self.distance_threshold,
        }


class TIDTemporalClassifier:
    """Multi-class classifier over fixed-length temporal landmark embeddings."""

    def __init__(self, model: Path | str | dict[str, object]) -> None:
        payload = joblib.load(Path(model)) if isinstance(model, (Path, str)) else model
        self.model = payload["model"]
        self.classes = np.asarray(payload["classes"])
        self.sequence_frames = int(payload["sequence_frames"])
        self.feature_count = int(payload["feature_count"])
        self.metrics = payload.get("metrics", {})

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        sequence = np.asarray(features, dtype=np.float32)
        expected_shape = (self.sequence_frames, self.feature_count)
        if sequence.shape != expected_shape:
            raise ValueError(f"Expected sequence shape {expected_shape}, got {sequence.shape}")
        embedding = temporal_embedding(sequence).reshape(1, -1)
        probabilities = np.asarray(self.model.predict_proba(embedding)[0], dtype=float)
        model_classes = np.asarray(self.model.classes_)
        if np.array_equal(model_classes, self.classes):
            return probabilities
        positions = {str(label): index for index, label in enumerate(model_classes)}
        return np.asarray([probabilities[positions[str(label)]] for label in self.classes])

    def predict(self, features: np.ndarray) -> dict[str, object]:
        probabilities = self.probabilities(features)
        order = np.argsort(probabilities)[::-1]
        best = int(order[0])
        return {
            "label": str(self.classes[best]),
            "confidence": float(probabilities[best]),
            "probabilities": probabilities,
            "candidates": [
                {
                    "label": str(self.classes[index]),
                    "confidence": float(probabilities[index]),
                }
                for index in order[:3]
            ],
        }
