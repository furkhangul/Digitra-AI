from fastapi import APIRouter

from app.services.inference import classifier_health

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, object]:
    model = classifier_health()
    return {"status": "ok" if model["ready"] else "degraded", "model": model}
