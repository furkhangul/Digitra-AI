from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.routers import feedback, health, models, predict, recognition, speech

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="Digitra gerçek zamanlı işaret tanıma ve ses/metin köprüsü API'si.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


app.include_router(health.router)
app.include_router(models.router)
app.include_router(predict.router)
app.include_router(recognition.router)
app.include_router(speech.router)
app.include_router(feedback.router)
