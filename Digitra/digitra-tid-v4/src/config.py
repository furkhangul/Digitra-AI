from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = ROOT / "artifacts"
FEATURES_DIR = ARTIFACTS_DIR / "features"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

KAGGLE_DATASET = "feronial/turkish-sign-languagefinger-spelling"
SEED = 20260909

LABEL_MAP = {
    "!": "İ",
    "+": "Ğ",
    ",": "Ç",
    ";": "Ş",
    "=": "Ü",
    "_": "Ö",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def ensure_dirs() -> None:
    for path in (RAW_DIR, FEATURES_DIR, MODELS_DIR, REPORTS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def fingerspelling_image_dir() -> Path:
    """Return the canonical 2,974-image alphabet directory."""
    candidates = [
        RAW_DIR / "tid_fingerspelling" / "tsl finger spelling" / "Images",
        RAW_DIR / "tid_fingerspelling" / "Images",
    ]
    ranked = []
    for candidate in candidates:
        count = sum(
            1 for p in candidate.glob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        ) if candidate.exists() else 0
        ranked.append((count, candidate))
    count, path = max(ranked, key=lambda item: item[0])
    if count == 0:
        raise FileNotFoundError(
            "TİD görüntüleri bulunamadı; önce scripts/download_kaggle.py çalıştırın."
        )
    return path
