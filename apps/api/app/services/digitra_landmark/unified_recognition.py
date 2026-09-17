from __future__ import annotations

from collections import deque

import numpy as np

from .features import extract_features
from .hand_recovery import choose_orientation_probabilities, mirror_hands
from .schemas import HandObservation
from .tid_sequence_model import TIDTemporalClassifier


class SmoothedTIDTemporalRecognizer:
    """Build and smooth a two-hand temporal prediction in either orientation."""

    def __init__(
        self,
        classifier: TIDTemporalClassifier,
        confidence_threshold: float = 0.40,
        margin_threshold: float = 0.08,
        probability_window: int = 5,
        minimum_probability_frames: int = 3,
    ) -> None:
        self.classifier = classifier
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.minimum_probability_frames = minimum_probability_frames
        self.feature_history: deque[np.ndarray] = deque(maxlen=classifier.sequence_frames)
        self.mirrored_feature_history: deque[np.ndarray] = deque(
            maxlen=classifier.sequence_frames
        )
        self.probability_history: deque[np.ndarray] = deque(maxlen=probability_window)

    def reset(self) -> None:
        self.feature_history.clear()
        self.mirrored_feature_history.clear()
        self.probability_history.clear()

    def update(self, hands: list[HandObservation]) -> dict[str, object]:
        if len(hands) < 2:
            self.reset()
            return self._waiting_result()

        selected = hands[:2]
        self.feature_history.append(extract_features(selected).vector)
        self.mirrored_feature_history.append(
            extract_features(mirror_hands(selected)).vector
        )
        if len(self.feature_history) < self.classifier.sequence_frames:
            return self._waiting_result()

        normal = self.classifier.probabilities(np.stack(self.feature_history))
        mirrored = self.classifier.probabilities(np.stack(self.mirrored_feature_history))
        probabilities, orientation = choose_orientation_probabilities(normal, mirrored)
        self.probability_history.append(probabilities)
        mean_probabilities = np.mean(self.probability_history, axis=0)
        order = np.argsort(mean_probabilities)[::-1]
        candidates = [
            {
                "label": str(self.classifier.classes[index]),
                "confidence": float(mean_probabilities[index]),
            }
            for index in order[:3]
        ]
        confidence = float(candidates[0]["confidence"])
        second_confidence = float(candidates[1]["confidence"])
        margin = confidence - second_confidence
        ready = (
            confidence >= self.confidence_threshold
            and margin >= self.margin_threshold
            and len(self.probability_history) >= self.minimum_probability_frames
        )
        return {
            "label": str(candidates[0]["label"]) if ready else "?",
            "candidate": str(candidates[0]["label"]),
            "confidence": confidence,
            "margin": margin,
            "ready": ready,
            "frames": len(self.feature_history),
            "probability_frames": len(self.probability_history),
            "orientation": orientation,
            "candidates": candidates,
        }

    def _waiting_result(self) -> dict[str, object]:
        return {
            "label": "?",
            "candidate": "?",
            "confidence": 0.0,
            "margin": 0.0,
            "ready": False,
            "frames": len(self.feature_history),
            "probability_frames": 0,
            "orientation": "—",
            "candidates": [],
        }


class StableTextComposer:
    """Commit stable predictions while preventing held-sign repetitions."""

    def __init__(
        self,
        dwell_seconds: float = 0.90,
        release_seconds: float = 0.25,
        entry_guard_seconds: float = 0.60,
    ) -> None:
        self.dwell_seconds = dwell_seconds
        self.release_seconds = release_seconds
        self.entry_guard_seconds = entry_guard_seconds
        self.text = ""
        self.candidate = ""
        self.candidate_since: float | None = None
        self.locked_label = ""
        self.no_hands_since: float | None = None
        self.hands_visible_since: float | None = None

    def update(
        self,
        label: str,
        ready: bool,
        hands_visible: bool,
        now: float,
    ) -> str | None:
        if not hands_visible:
            self.candidate = ""
            self.candidate_since = None
            self.hands_visible_since = None
            if self.no_hands_since is None:
                self.no_hands_since = now
            elif now - self.no_hands_since >= self.release_seconds:
                self.locked_label = ""
            return None

        self.no_hands_since = None
        if self.hands_visible_since is None:
            self.hands_visible_since = now
            self.candidate = ""
            self.candidate_since = None
            if self.entry_guard_seconds > 0:
                return None
        if self.locked_label:
            self.candidate = ""
            self.candidate_since = None
            return None
        if now - self.hands_visible_since < self.entry_guard_seconds:
            self.candidate = ""
            self.candidate_since = None
            return None
        if not ready or not label or label == "?":
            self.candidate = ""
            self.candidate_since = None
            return None

        if label != self.candidate:
            self.candidate = label
            self.candidate_since = now
            return None
        if self.candidate_since is None:
            return None
        if now - self.candidate_since < self.dwell_seconds:
            return None

        self.text += label
        self.locked_label = label
        return label

    def progress(self, now: float) -> float:
        if not self.candidate or self.candidate_since is None:
            return 0.0
        return min(1.0, max(0.0, (now - self.candidate_since) / self.dwell_seconds))

    def append_space(self) -> None:
        if self.text and not self.text.endswith(" "):
            self.text += " "

    def append_prediction(self, label: str) -> None:
        if label and label != "?":
            self.text += label
            self.locked_label = label

    def backspace(self) -> None:
        self.text = self.text[:-1]

    def clear(self) -> None:
        self.text = ""
        self.candidate = ""
        self.candidate_since = None
        self.locked_label = ""
        self.no_hands_since = None
        self.hands_visible_since = None
