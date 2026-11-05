import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter

from app.core.config import get_settings
from app.models.schemas import FeedbackRequest, FeedbackResponse

router = APIRouter(tags=["feedback"])

_write_lock = threading.Lock()


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(body: FeedbackRequest) -> FeedbackResponse:
    # Per DIGITRA_AI_MODEL_REHBERI.md #35 (Active Learning): a correction is
    # never auto-added as ground truth. It only enters a human review queue,
    # and only with explicit consent when a sample image is involved. Only the
    # labels are stored here; no image or landmark data ever reaches this file.
    queued = body.correct_label is not None and body.correct_label != body.predicted_label
    if queued:
        record = {
            "received_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "predicted_label": body.predicted_label,
            "correct_label": body.correct_label,
            "model_version": body.model_version,
            "source": body.source,
            "status": "pending_review",
        }
        path = Path(get_settings().feedback_queue_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with _write_lock, path.open("a", encoding="utf-8") as queue:
            queue.write(json.dumps(record, ensure_ascii=False) + "\n")
    return FeedbackResponse(received=True, queued_for_review=queued)
