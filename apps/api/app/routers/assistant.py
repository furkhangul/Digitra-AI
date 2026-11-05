import asyncio

from fastapi import APIRouter, HTTPException

from app.models.schemas import AssistantRequest, AssistantResponse
from app.services.assistant import AssistantUnavailable, answer, assistant_status

router = APIRouter(tags=["assistant"])


@router.get("/assistant/status")
def status() -> dict[str, object]:
    return assistant_status()


@router.post("/assistant", response_model=AssistantResponse)
async def ask(body: AssistantRequest) -> AssistantResponse:
    try:
        return await asyncio.to_thread(answer, body)
    except AssistantUnavailable as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
