from fastapi import APIRouter, Depends, HTTPException

from app.models.schemas import PredictRequest, PredictionResponse
from app.services.inference import LandmarkClassifier, get_classifier

router = APIRouter(tags=["predict"])


@router.post("/predict", response_model=PredictionResponse)
def predict(
    body: PredictRequest,
    classifier: LandmarkClassifier = Depends(get_classifier),
) -> PredictionResponse:
    try:
        return classifier.predict(
            body.landmarks or [],
            body.mode,
            image=body.image,
            world_landmarks=body.world_landmarks,
            handedness=body.handedness,
            handedness_score=body.handedness_score,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
