from fastapi import APIRouter

from app.models.schemas import ModelInfo, ModelMetrics

router = APIRouter(tags=["models"])

_REGISTERED_MODELS = [
    ModelInfo(
        id="digitra-tid-robust",
        name="Digitra TİD Robust",
        version="5.0.0",
        mode="accurate",
        status="candidate",
        description=(
            "29 harflik TİD parmak alfabesi için görüntü ensemble'ı. Sağ/sol el, "
            "ölçek, kadraj ve ışık dayanıklılığıyla eğitilmiş araştırma modelidir."
        ),
    ),
    ModelInfo(
        id="digitra-landmark-personal",
        name="Digitra Landmark Personal",
        version="1.0.0",
        mode="fast",
        status="candidate",
        description=(
            "Tek elde 0-9 ve A-Z teknik ASL parmak alfabesi. Dokuz sınıfta "
            "Furkan'ın kişisel verisiyle uyarlanmıştır; TİD modeli olarak sunulmaz."
        ),
    ),
    ModelInfo(
        id="digitra-tid-specials-temporal",
        name="Digitra TİD Özel Harfler Temporal",
        version="1.0.0",
        mode="fast",
        status="candidate",
        description=(
            "İki elde Ç, Ğ, İ, Ö, Ş ve Ü için zamansal landmark modeli. "
            "Tek kullanıcı/tek oturum verisi nedeniyle araştırma adayıdır."
        ),
    ),
]


@router.get("/models", response_model=list[ModelInfo])
def list_models() -> list[ModelInfo]:
    return _REGISTERED_MODELS


@router.get("/model-metrics", response_model=list[ModelMetrics])
def model_metrics() -> list[ModelMetrics]:
    return [
        ModelMetrics(
            model_version="digitra-tid-robust-v5.0.0",
            evaluation_scope=(
                "2.974 görüntü; 10 karelik pseudo-oturum split'i; kilitli test, "
                "kişi bağımsız değildir"
            ),
            internal_accuracy=0.8909512761020881,
            internal_macro_f1=0.8897662786728165,
            model_size_mb=33.62,
            measured_at="2026-09-09",
        ),
        ModelMetrics(
            model_version="digitra-landmark-personal-v1.0.0",
            evaluation_scope=(
                "Kişisel uyarlama: tek kullanıcı, aynı kayıt oturumunda kronolojik holdout"
            ),
            internal_accuracy=0.9943820224719101,
            internal_macro_f1=0.9944409700507263,
        ),
        ModelMetrics(
            model_version="digitra-tid-specials-temporal-v1.0.0",
            evaluation_scope=(
                "Ç/Ğ/İ/Ö/Ş/Ü: tek kullanıcı, tek oturum, kronolojik holdout; "
                "kişiler arası doğruluk değildir"
            ),
            internal_accuracy=1.0,
            internal_macro_f1=1.0,
        ),
    ]
