from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image

from .image_model import DigitraTIDEnsemble


class SmoothedTIDImageRecognizer:
    """Smooth calibrated image probabilities before exposing a stable letter."""

    def __init__(
        self,
        classifier: DigitraTIDEnsemble,
        probability_window: int = 5,
        minimum_frames: int = 3,
        minimum_vote_ratio: float = 0.60,
        minimum_confidence: float | None = None,
        minimum_margin: float = 0.0,
    ) -> None:
        self.classifier = classifier
        self.history: deque[np.ndarray] = deque(maxlen=probability_window)
        self.minimum_frames = minimum_frames
        self.minimum_vote_ratio = minimum_vote_ratio
        self.minimum_confidence = max(classifier.recommended_confidence, minimum_confidence or 0.0)
        self.minimum_margin = minimum_margin

    def reset(self) -> None:
        self.history.clear()

    def update(self, image: Image.Image) -> dict[str, object]:
        probabilities = self.classifier.probability_vector(image)
        self.history.append(probabilities)
        mean_probabilities = np.mean(self.history, axis=0)
        order = np.argsort(mean_probabilities)[::-1]
        candidates = [
            {
                "label": self.classifier.labels[index],
                "confidence": float(mean_probabilities[index]),
            }
            for index in order[:3]
        ]
        confidence = float(candidates[0]["confidence"])
        second_confidence = float(candidates[1]["confidence"])
        margin = confidence - second_confidence
        winner = int(order[0])
        frame_winners = np.argmax(np.stack(self.history), axis=1)
        vote_ratio = float(np.mean(frame_winners == winner))
        ready = (
            len(self.history) >= self.minimum_frames
            and confidence >= self.minimum_confidence
            and margin >= self.minimum_margin
            and vote_ratio >= self.minimum_vote_ratio
        )
        return {
            "label": str(candidates[0]["label"]) if ready else "?",
            "candidate": str(candidates[0]["label"]),
            "confidence": confidence,
            "margin": margin,
            "vote_ratio": vote_ratio,
            "ready": ready,
            "frames": len(self.history),
            "candidates": candidates,
        }
