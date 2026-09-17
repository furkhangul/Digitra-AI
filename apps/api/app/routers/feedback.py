from fastapi import APIRouter

from app.models.schemas import FeedbackRequest, FeedbackResponse

router = APIRouter(tags=["feedback"])


@router.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(body: FeedbackRequest) -> FeedbackResponse:
    # Per DIGITRA_AI_MODEL_REHBERI.md #35 (Active Learning): a correction is
    # never auto-added as ground truth. It only enters a human review queue,
    # and only with explicit consent when a sample image is involved.
    queued = body.consented_sample and body.correct_label is not None
    return FeedbackResponse(received=True, queued_for_review=queued)
