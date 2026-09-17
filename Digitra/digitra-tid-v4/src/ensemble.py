from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.cnn import Prediction, TidImageClassifier


class TidEnsembleClassifier:
    def __init__(self, config_path: str | Path, device: str | None = None) -> None:
        path = Path(config_path)
        config = json.loads(path.read_text(encoding="utf-8"))
        self.recommended_confidence = float(config.get("recommended_confidence", 0.45))
        self.members = []
        for member in config["members"]:
            weight = float(member["weight"])
            if weight <= 0.0:
                continue
            checkpoint = Path(member["checkpoint"])
            if not checkpoint.is_absolute():
                checkpoint = path.parent / checkpoint
            classifier = TidImageClassifier(checkpoint, device=device)
            self.members.append((classifier, weight))
        if not self.members:
            raise ValueError("Ensemble en az bir model içermeli")
        self.labels = self.members[0][0].labels
        if any(classifier.labels != self.labels for classifier, _ in self.members):
            raise ValueError("Ensemble sınıf sıraları uyuşmuyor")

    def probability_vector(self, image: Image.Image | np.ndarray) -> np.ndarray:
        total = np.zeros(len(self.labels), dtype=np.float64)
        total_weight = 0.0
        for classifier, weight in self.members:
            total += weight * classifier.probability_vector(image)
            total_weight += weight
        if total_weight <= 0:
            raise ValueError("Ensemble toplam ağırlığı sıfır")
        return (total / total_weight).astype(np.float32)

    def predict(self, image: Image.Image | np.ndarray) -> Prediction:
        probabilities = self.probability_vector(image)
        index = int(np.argmax(probabilities))
        return Prediction(
            label=self.labels[index],
            confidence=float(probabilities[index]),
            probabilities={label: float(probabilities[i]) for i, label in enumerate(self.labels)},
        )
