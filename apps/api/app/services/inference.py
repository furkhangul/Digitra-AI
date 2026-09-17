"""Production bridge for Digitra's landmark and TİD image models.

The browser performs hand detection. For Robust V5 it sends only a bounded crop
around the detected hand(s) to the local API; the crop is decoded in memory and
is never persisted. Each WebSocket owns its smoothing and text-composition state.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path
from time import monotonic, perf_counter
from typing import Any

from app.core.config import get_settings
from app.models.schemas import ModelMode, PredictionResponse, TopKPrediction
from app.services.digitra_landmark.asl_model import ASLClassifier, SmoothedASLRecognizer
from app.services.digitra_landmark.hand_recovery import (
    StableTwoHandTracker,
    TwoHandModeGate,
    deduplicate_hands,
)
from app.services.digitra_landmark.schemas import HandObservation
from app.services.digitra_landmark.tid_sequence_model import TIDTemporalClassifier
from app.services.digitra_landmark.unified_recognition import (
    SmoothedTIDTemporalRecognizer,
    StableTextComposer,
)
from app.services.digitra_tid import (
    DigitraTIDEnsemble,
    SmoothedTIDImageRecognizer,
    decode_image_data,
)

MODEL_ID = "digitra-tid-robust"
MODEL_VERSION = "5.0.0"
MODEL_DIRECTORY = "digitra-tid-robust-v5"
LEGACY_MODEL_ID = "digitra-landmark-personal"
LEGACY_MODEL_VERSION = "1.0.0"
LEGACY_MODEL_DIRECTORY = "digitra-landmark-personal-v1.0.0"


def models_root() -> Path:
    settings = get_settings()
    api_root = Path(__file__).resolve().parents[2]
    root = Path(settings.models_dir)
    if not root.is_absolute():
        root = (api_root / root).resolve()
    return root


def model_directory() -> Path:
    return models_root() / MODEL_DIRECTORY


def legacy_model_directory() -> Path:
    return models_root() / LEGACY_MODEL_DIRECTORY


class LandmarkClassifier(ABC):
    @abstractmethod
    def predict(
        self,
        landmarks: list[list[float]],
        mode: ModelMode,
        *,
        image: str | None = None,
        world_landmarks: list[list[float]] | None = None,
        handedness: str = "Unknown",
        handedness_score: float = 0.0,
    ) -> PredictionResponse: ...


class DigitraLandmarkClassifier(LandmarkClassifier):
    """Load Robust V5 plus the backwards-compatible landmark models once."""

    def __init__(
        self,
        directory: Path | None = None,
        legacy_directory: Path | None = None,
    ) -> None:
        self.directory = directory or model_directory()
        self.legacy_directory = legacy_directory or legacy_model_directory()
        self.tid_image = DigitraTIDEnsemble(
            self.directory / "robust_cnn_ensemble.json"
        )
        self.asl = ASLClassifier(
            self.legacy_directory / "asl_classifier_personal.joblib"
        )
        self.tid = TIDTemporalClassifier(
            self.legacy_directory / "tid_specials_temporal.joblib"
        )

    def predict(
        self,
        landmarks: list[list[float]],
        mode: ModelMode,
        *,
        image: str | None = None,
        world_landmarks: list[list[float]] | None = None,
        handedness: str = "Unknown",
        handedness_score: float = 0.0,
    ) -> PredictionResponse:
        del mode  # This release currently exposes one measured fast model.
        started = perf_counter()
        if image:
            decoded = decode_image_data(image)
            result = self.tid_image.predict(decoded)
            ranked = sorted(
                result.probabilities.items(), key=lambda item: item[1], reverse=True
            )[:3]
            accepted = result.confidence >= self.tid_image.recommended_confidence
            return PredictionResponse(
                model=MODEL_ID,
                version=MODEL_VERSION,
                prediction=result.label if accepted else None,
                confidence=result.confidence,
                stable=False,
                top_k=[
                    TopKPrediction(label=label, probability=float(value))
                    for label, value in ranked
                ],
                latency_ms=round((perf_counter() - started) * 1000, 2),
                hand_detected=True,
                reject_reason=None if accepted else "LOW_CONFIDENCE",
            )
        if not landmarks:
            return PredictionResponse(
                model=LEGACY_MODEL_ID,
                version=LEGACY_MODEL_VERSION,
                hand_detected=False,
                reject_reason="NO_HAND",
                latency_ms=round((perf_counter() - started) * 1000, 2),
            )

        observation = HandObservation(
            landmarks=landmarks,
            world_landmarks=world_landmarks,
            handedness=handedness,
            handedness_score=handedness_score,
        )
        result = self.asl.predict_observation(observation)
        ranked = sorted(
            result["probabilities"].items(), key=lambda item: item[1], reverse=True
        )[:3]
        confidence = float(result["confidence"])
        margin = confidence - (float(ranked[1][1]) if len(ranked) > 1 else 0.0)
        accepted = confidence >= 0.55 and margin >= 0.08
        return PredictionResponse(
            model=LEGACY_MODEL_ID,
            version=LEGACY_MODEL_VERSION,
            prediction=str(result["label"]) if accepted else None,
            confidence=confidence,
            stable=False,
            top_k=[
                TopKPrediction(label=str(label), probability=float(value))
                for label, value in ranked
            ],
            latency_ms=round((perf_counter() - started) * 1000, 2),
            hand_detected=True,
            reject_reason=None if accepted else "LOW_CONFIDENCE",
        )

    def new_session(self) -> "RecognitionSession":
        return RecognitionSession(self)


class RecognitionSession:
    """Temporal recognizer and release-gated text buffer for one browser client."""

    def __init__(self, classifier: DigitraLandmarkClassifier) -> None:
        self.image = SmoothedTIDImageRecognizer(classifier.tid_image)
        self.asl = SmoothedASLRecognizer(classifier.asl)
        self.tid = SmoothedTIDTemporalRecognizer(classifier.tid)
        self.tracker = StableTwoHandTracker(grace_frames=3)
        self.mode_gate = TwoHandModeGate(enter_frames=2, exit_frames=3)
        self.composer = StableTextComposer()
        self.last_result = self._empty_result()
        self.last_mode = "waiting"
        self.direct_hand_count = 0

    @staticmethod
    def _empty_result() -> dict[str, object]:
        return {
            "label": "?",
            "candidate": "?",
            "confidence": 0.0,
            "margin": 0.0,
            "ready": False,
            "frames": 0,
            "candidates": [],
        }

    def update(self, payload: dict[str, Any]) -> dict[str, object]:
        started = perf_counter()
        now = monotonic()
        action = str(payload.get("action", ""))
        added: str | None = None

        if action:
            if action == "space":
                self.composer.append_space()
            elif action == "backspace":
                self.composer.backspace()
            elif action == "clear":
                self.composer.clear()
            elif action == "append" and bool(self.last_result.get("ready", False)):
                self.composer.append_prediction(str(self.last_result.get("label", "?")))
            else:
                raise ValueError(f"Desteklenmeyen işlem: {action}")
        else:
            raw_hands = payload.get("hands", [])
            if not isinstance(raw_hands, list):
                raise ValueError("hands bir liste olmalıdır")
            hands = [HandObservation.from_dict(item) for item in raw_hands[:2]]
            direct_hands = deduplicate_hands(hands)
            self.direct_hand_count = len(direct_hands)
            image_payload = payload.get("image")
            if image_payload is not None and not isinstance(image_payload, str):
                raise ValueError("image bir base64 metni veya null olmalıdır")

            if self.direct_hand_count == 0:
                self.image.reset()
                self.tracker.reset()
                self.asl.reset()
                self.tid.reset()
                self.mode_gate.update(0)
                self.last_mode = "waiting"
                self.last_result = self._empty_result()
            elif image_payload:
                self.tracker.reset()
                self.asl.reset()
                self.tid.reset()
                self.mode_gate.update(self.direct_hand_count)
                self.last_mode = "tid_image"
                self.last_result = self.image.update(decode_image_data(image_payload))
            elif self.mode_gate.update(self.direct_hand_count):
                self.image.reset()
                tracked_hands = self.tracker.update(direct_hands)
                self.asl.reset()
                if len(tracked_hands) >= 2:
                    self.last_mode = "two_hands"
                    self.last_result = self.tid.update(tracked_hands[:2])
                else:
                    self.last_mode = "waiting_two_hands"
                    self.tid.reset()
                    self.last_result = self._empty_result()
            elif self.direct_hand_count == 1:
                self.image.reset()
                self.tracker.reset()
                self.tid.reset()
                self.last_mode = "one_hand"
                self.last_result = self.asl.update(direct_hands[0])
            else:
                self.image.reset()
                self.tracker.reset()
                self.asl.reset()
                self.tid.reset()
                self.last_mode = "waiting"
                self.last_result = self._empty_result()

            added = self.composer.update(
                str(self.last_result.get("label", "?")),
                bool(self.last_result.get("ready", False)),
                self.direct_hand_count > 0,
                now,
            )

        ready = bool(self.last_result.get("ready", False))
        reject_reason = None
        if self.direct_hand_count == 0:
            reject_reason = "NO_HAND"
        elif not ready:
            reject_reason = "LOW_CONFIDENCE"

        return {
            "model_ready": True,
            "model": MODEL_ID if self.last_mode == "tid_image" else LEGACY_MODEL_ID,
            "version": (
                MODEL_VERSION
                if self.last_mode == "tid_image"
                else LEGACY_MODEL_VERSION
            ),
            "scope": (
                "tid_alphabet_image"
                if self.last_mode == "tid_image"
                else (
                    "tid_specials_two_hand"
                    if self.last_mode.startswith("two_hands")
                    else "asl_one_hand"
                )
            ),
            "mode": self.last_mode,
            "hands_detected": self.direct_hand_count,
            "prediction": str(self.last_result.get("label", "?")) if ready else None,
            "candidate": str(self.last_result.get("candidate", "?")),
            "confidence": float(self.last_result.get("confidence", 0.0)),
            "margin": float(self.last_result.get("margin", 0.0)),
            "ready": ready,
            "frames": int(self.last_result.get("frames", 0)),
            "candidates": self.last_result.get("candidates", []),
            "text": self.composer.text,
            "added": added,
            "locked": bool(self.composer.locked_label),
            "progress": self.composer.progress(now),
            "reject_reason": reject_reason,
            "latency_ms": round((perf_counter() - started) * 1000, 2),
        }


@lru_cache
def get_classifier() -> DigitraLandmarkClassifier:
    return DigitraLandmarkClassifier()


def classifier_health() -> dict[str, object]:
    try:
        classifier = get_classifier()
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        return {
            "ready": False,
            "model": MODEL_ID,
            "version": MODEL_VERSION,
            "error": str(error),
        }
    return {
        "ready": True,
        "model": MODEL_ID,
        "version": MODEL_VERSION,
        "device": str(classifier.tid_image.device),
        "tid_classes": list(classifier.tid_image.labels),
        "one_hand_classes": [str(item) for item in classifier.asl.classes],
        "two_hand_classes": [str(item) for item in classifier.tid.classes],
    }
