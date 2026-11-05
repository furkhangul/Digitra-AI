from typing import Literal

from pydantic import BaseModel, Field

ModelMode = Literal["fast", "balanced", "accurate"]

RejectReason = Literal[
    "NO_HAND",
    "LOW_LIGHT",
    "BLURRY",
    "MULTIPLE_HANDS",
    "LOW_CONFIDENCE",
    "UNKNOWN",
]


class TopKPrediction(BaseModel):
    label: str
    probability: float


class PredictionResponse(BaseModel):
    """
    Mirrors the standard inference output defined in
    DIGITRA_AI_MODEL_REHBERI.md #38. Null fields are only filled once a real
    model is wired in — no placeholder numbers are invented here.
    """

    model: str = "digitra"
    version: str
    prediction: str | None = None
    confidence: float | None = None
    stable: bool = False
    top_k: list[TopKPrediction] = Field(default_factory=list)
    latency_ms: float | None = None
    hand_detected: bool = False
    reject_reason: RejectReason | None = None


class PredictRequest(BaseModel):
    mode: ModelMode = "balanced"
    # Robust V5 accepts only a bounded crop around MediaPipe-detected hands.
    # The local API decodes this in memory and never persists it.
    image: str | None = Field(default=None, max_length=750_000)
    # Kept for legacy clients and the landmark fallback path.
    landmarks: list[list[float]] | None = None
    world_landmarks: list[list[float]] | None = None
    handedness: str = "Unknown"
    handedness_score: float = Field(default=0.0, ge=0.0, le=1.0)


class ModelInfo(BaseModel):
    id: str
    name: str
    version: str
    mode: ModelMode
    status: Literal["development", "candidate", "validated", "production", "archived"]
    description: str


class ModelMetrics(BaseModel):
    """
    Real metrics only. Every field is null until an actual evaluation run
    produces it — see DIGITRA_MASTER_REHBER.md #74, "TBD değerleri gerçek
    deneyler olmadan doldurulmaz."
    """

    model_version: str
    evaluation_scope: str | None = None
    internal_accuracy: float | None = None
    internal_macro_f1: float | None = None
    external_macro_f1: float | None = None
    expected_calibration_error: float | None = None
    avg_latency_ms: float | None = None
    fps: float | None = None
    model_size_mb: float | None = None
    measured_at: str | None = None


class STTRequest(BaseModel):
    language: str = "tr-TR"


class STTResponse(BaseModel):
    text: str
    provider: str
    confidence: float | None = None


class TTSRequest(BaseModel):
    text: str
    voice: str | None = None
    language: str = "tr-TR"
    speed: float = 1.0
    pitch: float = 1.0


class FeedbackRequest(BaseModel):
    predicted_label: str | None = Field(default=None, max_length=8)
    correct_label: str | None = Field(default=None, max_length=8)
    model_version: str = Field(max_length=64)
    consented_sample: bool = False
    source: str = Field(default="web", max_length=32)


class FeedbackResponse(BaseModel):
    received: bool = True
    queued_for_review: bool = False


class AssistantTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2_000)


class AssistantRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1_000)
    history: list[AssistantTurn] = Field(default_factory=list, max_length=16)
    section: str | None = Field(default=None, max_length=32)
    recognized_text: str | None = Field(default=None, max_length=500)


class AssistantAction(BaseModel):
    type: Literal["navigate", "spell", "camera_on", "camera_off", "quiz", "open_live_training"]
    target: str = ""
    text: str = ""


class AssistantResponse(BaseModel):
    reply: str
    action: AssistantAction | None = None
    provider: str
