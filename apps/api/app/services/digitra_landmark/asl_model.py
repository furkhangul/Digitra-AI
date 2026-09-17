from __future__ import annotations

from collections import deque
from pathlib import Path

import joblib
import numpy as np

from .asl_features import ASL_FEATURE_COUNT, compute_asl_features
from .hand_recovery import choose_orientation_probabilities, mirror_hands
from .schemas import HandObservation

DEFAULT_ASL_MODEL_PATH = Path("models/digitra-landmark-personal-v1.0.0/asl_classifier_personal.joblib")


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials)


class ASLClassifier:
    def __init__(self, model_path: Path | str = DEFAULT_ASL_MODEL_PATH) -> None:
        payload = joblib.load(Path(model_path))
        self.model = payload["model"]
        self.scaler = payload.get("scaler")
        self.classes = np.asarray(payload["classes"])
        self.metrics = payload.get("metrics", {})
        if int(payload.get("feature_count", -1)) != ASL_FEATURE_COUNT:
            raise RuntimeError("ASL model feature version is incompatible")

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        row = np.asarray(features, dtype=np.float32).reshape(1, -1)
        if self.scaler is not None:
            row = self.scaler.transform(row)
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(row)[0], dtype=float)
        scores = np.asarray(self.model.decision_function(row)[0], dtype=float)
        return _softmax(scores)

    def predict_observation(self, hand: HandObservation) -> dict[str, object]:
        probabilities, orientation = self.probabilities_for_observation(hand)
        index = int(np.argmax(probabilities))
        return {
            "label": str(self.classes[index]),
            "confidence": float(probabilities[index]),
            "orientation": orientation,
            "probabilities": {
                str(label): float(probability)
                for label, probability in zip(self.classes, probabilities)
            },
        }

    def probabilities_for_observation(
        self, hand: HandObservation
    ) -> tuple[np.ndarray, str]:
        features = compute_asl_features(
            hand.landmarks,
            hand.world_landmarks,
            hand.handedness,
        )
        mirrored = mirror_hands([hand])[0]
        mirrored_features = compute_asl_features(
            mirrored.landmarks,
            mirrored.world_landmarks,
            mirrored.handedness,
        )
        return choose_orientation_probabilities(
            self.probabilities(features),
            self.probabilities(mirrored_features),
        )


class SmoothedASLRecognizer:
    def __init__(
        self,
        classifier: ASLClassifier,
        window_size: int = 8,
        confidence_threshold: float = 0.55,
        margin_threshold: float = 0.08,
        minimum_history_frames: int = 6,
        minimum_vote_ratio: float = 0.625,
    ) -> None:
        if not 1 <= minimum_history_frames <= window_size:
            raise ValueError("minimum_history_frames must be within the smoothing window")
        if not 0.0 <= minimum_vote_ratio <= 1.0:
            raise ValueError("minimum_vote_ratio must be between zero and one")
        self.classifier = classifier
        self.history: deque[np.ndarray] = deque(maxlen=window_size)
        self.motion_history: deque[tuple[np.ndarray, float]] = deque(
            maxlen=max(18, window_size * 2)
        )
        self._ij_motion_evidence_frames = 0
        self._j_motion_latch_frames = 0
        self._ij_static_frames = 0
        self._j_motion_armed = False
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold
        self.minimum_history_frames = minimum_history_frames
        self.minimum_vote_ratio = minimum_vote_ratio

    def reset(self) -> None:
        self.history.clear()
        self.motion_history.clear()
        self._ij_motion_evidence_frames = 0
        self._j_motion_latch_frames = 0
        self._ij_static_frames = 0
        self._j_motion_armed = False

    def _pinky_motion_score(self) -> float:
        if len(self.motion_history) < 6:
            return 0.0
        tips = np.stack([item[0] for item in self.motion_history])
        scales = np.asarray([item[1] for item in self.motion_history])
        smoothed = np.stack(
            [np.mean(tips[max(0, index - 2) : index + 1], axis=0) for index in range(len(tips))]
        )
        movement_range = float(np.linalg.norm(np.ptp(smoothed, axis=0)))
        scale = max(float(np.median(scales)), 1e-6)
        return movement_range / scale

    def _separate_static_i_from_moving_j(
        self, probabilities: np.ndarray
    ) -> tuple[np.ndarray, float]:
        class_positions = {
            str(label): index for index, label in enumerate(self.classifier.classes)
        }
        if "I" not in class_positions or "J" not in class_positions:
            return probabilities, 0.0
        i_index, j_index = class_positions["I"], class_positions["J"]
        combined = float(probabilities[i_index] + probabilities[j_index])
        other = np.delete(probabilities, [i_index, j_index])
        motion_score = self._pinky_motion_score()
        if combined < float(np.max(other)):
            self._ij_motion_evidence_frames = 0
            self._ij_static_frames = 0
            if self._j_motion_latch_frames > 0:
                self._j_motion_latch_frames -= 1
            return probabilities, motion_score
        if not self._j_motion_armed:
            if motion_score < 0.30:
                self._ij_static_frames += 1
            else:
                self._ij_static_frames = 0
            if self._ij_static_frames >= 6:
                self._j_motion_armed = True
        if self._j_motion_armed and motion_score >= 0.55:
            self._ij_motion_evidence_frames += 1
        else:
            self._ij_motion_evidence_frames = 0
        if self._ij_motion_evidence_frames >= 2:
            self._j_motion_latch_frames = 60
        moving_j = self._j_motion_latch_frames > 0
        if self._j_motion_latch_frames > 0:
            self._j_motion_latch_frames -= 1
        adjusted = probabilities.copy()
        adjusted[i_index] = 0.0 if moving_j else combined
        adjusted[j_index] = combined if moving_j else 0.0
        return adjusted, motion_score

    def update(self, hand: HandObservation | None) -> dict[str, object]:
        if hand is None:
            self.reset()
            return {"label": "—", "confidence": 0.0, "ready": False}
        palm_scale = float(np.linalg.norm(hand.landmarks[9, :2] - hand.landmarks[0, :2]))
        self.motion_history.append((hand.landmarks[20, :2].copy(), palm_scale))
        probabilities, orientation = self.classifier.probabilities_for_observation(hand)
        self.history.append(probabilities)
        mean_probability = np.mean(self.history, axis=0)
        mean_probability, motion_score = self._separate_static_i_from_moving_j(
            mean_probability
        )
        index = int(np.argmax(mean_probability))
        confidence = float(mean_probability[index])
        ranked_indices = np.argsort(mean_probability)[::-1][:3]
        second_confidence = (
            float(mean_probability[ranked_indices[1]])
            if len(ranked_indices) > 1
            else 0.0
        )
        margin = confidence - second_confidence
        frame_winners = np.argmax(np.stack(self.history), axis=1)
        class_positions = {
            str(label): position
            for position, label in enumerate(self.classifier.classes)
        }
        output_label = str(self.classifier.classes[index])
        if output_label in {"I", "J"} and {"I", "J"}.issubset(class_positions):
            accepted_winners = {
                class_positions["I"],
                class_positions["J"],
            }
            votes = int(np.isin(frame_winners, list(accepted_winners)).sum())
        else:
            votes = int(np.sum(frame_winners == index))
        vote_ratio = votes / len(self.history)
        ready = (
            len(self.history) >= self.minimum_history_frames
            and confidence >= self.confidence_threshold
            and margin >= self.margin_threshold
            and vote_ratio >= self.minimum_vote_ratio
        )
        return {
            "label": output_label,
            "candidate": output_label,
            "confidence": confidence,
            "margin": margin,
            "vote_ratio": vote_ratio,
            "votes": votes,
            "ready": ready,
            "frames": len(self.history),
            "orientation": orientation,
            "motion_score": motion_score,
            "j_motion_latched": self._j_motion_latch_frames > 0,
            "j_motion_armed": self._j_motion_armed,
            "candidates": [
                {
                    "label": str(self.classifier.classes[candidate_index]),
                    "confidence": float(mean_probability[candidate_index]),
                }
                for candidate_index in ranked_indices
            ],
        }
