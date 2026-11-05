"""
Free-form question answering for the Digitra voice assistant.

Navigation, spelling, camera and text commands are resolved in the browser so
they keep working offline. Only questions the browser's intent rules cannot
match reach this module, which asks Claude for a short spoken-style answer and
an optional site action.
"""

import json
import os
from functools import lru_cache
from pathlib import Path

import anthropic

from app.core.config import get_settings
from app.models.schemas import AssistantAction, AssistantRequest, AssistantResponse

MODEL = "claude-opus-5"

# Frozen so the prefix stays cacheable; volatile data only goes in messages.
SYSTEM_PROMPT = """Sen Digitra'nın sesli asistanısın. Digitra, Türk İşaret Dili (TİD) parmak alfabesini öğreten ve tanıyan bir web platformudur.

Sitenin bölümleri (action.target için kullan):
- ceviri: Canlı harf tanıma. Kamera açılır, MediaPipe eli bulur, TİD V6 modeli 29 harfi tanır ve metne ekler. "Ses → İşaret" sekmesi yazılan ya da söylenen kelimeyi 3D ellerle harf harf parmak alfabesiyle gösterir.
- egitim: 29 harfin 3D el gösterimleri, "Kendini dene" testi ve /egitim/canli canlı kelime pratiği.
- how-it-works: Digitra'nın neden ve nasıl çalıştığı.
- architecture: Teknoloji (MediaPipe el tespiti, V6 görüntü modeli, hareketli harfler için zaman serisi takibi).
- science: Doğrulama. Canlı tanımada kullanılan TİD V6 modeli 431 gerçek test görüntüsünde %90,26 doğruluk (%95 GA %87,2–%93,0) ve %90,23 macro F1 aldı; medyan model gecikmesi 20,4 ms. Test kişi bağımsız (signer-independent) değil; hareketli harflerin canlı başarısı ölçülmedi.
- datasets: Veri kaynakları.
- faq: Sık sorulan sorular.

Önemli gerçekler:
- TİD parmak alfabesi 29 harftir: A B C Ç D E F G Ğ H I İ J K L M N O Ö P R S Ş T U Ü V Y Z. Q, W, X yoktur.
- Ç, Ğ, İ, J, Ö, Ş, Ü hareketli ya da iki elli harflerdir; hareket tanıma deneme aşamasındadır.
- Digitra tam TİD cümle çevirisi yapmaz; parmak alfabesiyle harf harf heceler. TİD'in kendi dilbilgisi, yüz ifadeleri ve beden hareketleri vardır.
- Kamera görüntüsünün tamamı gönderilmez; yalnızca el çevresindeki kırpım yerel API'de bellekte işlenir ve kaydedilmez.
- 3D gösterimlerin TİD uzmanı doğrulaması henüz tamamlanmadı.

Yanıt kuralları:
- Yanıtın sesli okunacak: Türkçe, en fazla 2-3 kısa cümle, madde işareti, markdown ya da emoji yok.
- Bilmediğin bir şeyi uydurma; Digitra'nın yapamadığı bir şeyi yapabiliyormuş gibi söyleme.
- Kullanıcı bir bölüme gitmek, bir kelimeyi işaretle görmek, kamerayı açmak/kapatmak ya da testi başlatmak istiyorsa uygun action'ı doldur; yoksa action.type "none" olsun.
- action.type "spell" ise action.text yalnızca hecelenecek kelime(ler)dir."""

ACTION_TYPES = ["none", "navigate", "spell", "camera_on", "camera_off", "quiz", "open_live_training"]
SECTIONS = ["ceviri", "egitim", "how-it-works", "architecture", "science", "datasets", "faq", "top"]

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reply": {"type": "string"},
        "action": {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": ACTION_TYPES},
                "target": {"type": "string", "enum": SECTIONS},
                "text": {"type": "string"},
            },
            "required": ["type", "target", "text"],
            "additionalProperties": False,
        },
    },
    "required": ["reply", "action"],
    "additionalProperties": False,
}


class AssistantUnavailable(RuntimeError):
    """Raised when no Claude credentials are configured or the call fails."""


@lru_cache
def _client() -> anthropic.Anthropic:
    settings = get_settings()
    # LLM_API_KEY wins when set; otherwise the SDK resolves ANTHROPIC_API_KEY,
    # ANTHROPIC_AUTH_TOKEN or an `ant auth login` profile on its own.
    if settings.llm_api_key:
        return anthropic.Anthropic(api_key=settings.llm_api_key, timeout=30.0, max_retries=1)
    return anthropic.Anthropic(timeout=30.0, max_retries=1)


def _has_credentials() -> bool:
    # The SDK only discovers missing credentials when a request is built, so
    # check the same sources up front to report status and skip dead calls.
    if get_settings().llm_api_key:
        return True
    if any(os.environ.get(name) for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE")):
        return True
    config_dir = Path(os.environ.get("ANTHROPIC_CONFIG_DIR", Path.home() / ".config" / "anthropic"))
    return config_dir.is_dir() and any(config_dir.iterdir())


def assistant_status() -> dict[str, object]:
    if not _has_credentials():
        return {"llm": False, "model": None, "detail": "Claude kimlik bilgisi yapılandırılmadı."}
    return {"llm": True, "model": MODEL, "detail": None}


def answer(body: AssistantRequest) -> AssistantResponse:
    if not _has_credentials():
        raise AssistantUnavailable("Claude kimlik bilgisi yapılandırılmadı.")
    client = _client()

    messages: list[dict[str, str]] = [
        {"role": turn.role, "content": turn.content} for turn in body.history[-8:]
    ]
    context = f"Kullanıcının şu an baktığı bölüm: {body.section or 'bilinmiyor'}."
    if body.recognized_text:
        context += f" Canlı tanımada oluşan metin: {body.recognized_text!r}."
    messages.append({"role": "user", "content": f"{context}\n\n{body.message}"})

    try:
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            thinking={"type": "adaptive"},
            # Voice turn: keep the time to first answer short.
            output_config={
                "effort": "low",
                "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA},
            },
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
    except anthropic.AuthenticationError as error:
        raise AssistantUnavailable("Claude API anahtarı geçersiz.") from error
    except anthropic.RateLimitError as error:
        raise AssistantUnavailable("Claude istek sınırına ulaşıldı, biraz sonra tekrar dene.") from error
    except anthropic.APIStatusError as error:
        raise AssistantUnavailable(f"Claude API hatası ({error.status_code}).") from error
    except anthropic.APIConnectionError as error:
        raise AssistantUnavailable("Claude API'ye bağlanılamadı.") from error
    except TypeError as error:
        # Raised by the SDK when a profile exists but yields no usable credential.
        raise AssistantUnavailable("Claude kimlik bilgisi çözümlenemedi.") from error

    if response.stop_reason == "refusal":
        return AssistantResponse(
            reply="Bu konuda yardımcı olamıyorum. Digitra ile ilgili başka bir şey sorabilirsin.",
            provider="claude",
        )

    text = next((block.text for block in response.content if block.type == "text"), "")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as error:
        raise AssistantUnavailable("Claude yanıtı çözümlenemedi.") from error

    action = data.get("action") or {}
    return AssistantResponse(
        reply=data.get("reply", "").strip(),
        action=AssistantAction(**action) if action.get("type", "none") != "none" else None,
        provider="claude",
    )
