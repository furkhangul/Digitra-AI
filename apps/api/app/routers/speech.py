from fastapi import APIRouter, HTTPException, UploadFile

from app.core.config import get_settings
from app.models.schemas import STTResponse, TTSRequest

router = APIRouter(tags=["speech"])


@router.post("/stt", response_model=STTResponse)
async def speech_to_text(audio: UploadFile) -> STTResponse:
    settings = get_settings()
    if settings.stt_provider == "browser":
        raise HTTPException(
            status_code=400,
            detail=(
                "Bu ortam tarayıcı içi (Web Speech API) STT kullanacak şekilde "
                "yapılandırılmış; sunucu tarafı STT henüz bağlanmadı."
            ),
        )
    raise HTTPException(status_code=501, detail="Cloud/local STT sağlayıcısı henüz uygulanmadı.")


@router.post("/tts")
async def text_to_speech(body: TTSRequest):
    settings = get_settings()
    if settings.tts_provider == "browser":
        raise HTTPException(
            status_code=400,
            detail=(
                "Bu ortam tarayıcı içi (SpeechSynthesis API) TTS kullanacak şekilde "
                "yapılandırılmış; sunucu tarafı TTS henüz bağlanmadı."
            ),
        )
    raise HTTPException(status_code=501, detail="Cloud/local TTS sağlayıcısı henüz uygulanmadı.")
