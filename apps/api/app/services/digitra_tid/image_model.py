from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import nn
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MAX_ENCODED_IMAGE_CHARS = 750_000
MAX_DECODED_IMAGE_BYTES = 550_000
MAX_IMAGE_PIXELS = 1_500_000


class Letterbox:
    """Resize without changing the hand aspect ratio and pad to a square."""

    def __init__(self, size: int, fill: tuple[int, int, int] = (114, 114, 114)) -> None:
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageOps.pad(
            image.convert("RGB"),
            (self.size, self.size),
            method=Image.Resampling.BICUBIC,
            color=self.fill,
            centering=(0.5, 0.5),
        )


def decode_image_data(value: str) -> Image.Image:
    """Decode a bounded JPEG/PNG data URL sent by the local browser client."""

    if not isinstance(value, str) or not value:
        raise ValueError("El görüntüsü boş")
    encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    if len(encoded) > MAX_ENCODED_IMAGE_CHARS:
        raise ValueError("El görüntüsü izin verilen boyutu aşıyor")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("El görüntüsü geçerli base64 değil") from error
    if not raw or len(raw) > MAX_DECODED_IMAGE_BYTES:
        raise ValueError("El görüntüsü izin verilen byte boyutunu aşıyor")
    try:
        with Image.open(BytesIO(raw)) as source:
            if source.format not in {"JPEG", "PNG", "WEBP"}:
                raise ValueError("Yalnızca JPEG, PNG veya WEBP el görüntüsü kabul edilir")
            if source.width <= 0 or source.height <= 0:
                raise ValueError("El görüntüsünün boyutları geçersiz")
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError("El görüntüsünün çözünürlüğü çok yüksek")
            source.load()
            return source.convert("RGB").copy()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("El görüntüsü çözülemedi") from error


def _evaluation_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            Letterbox(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _build_model(architecture: str, num_classes: int) -> nn.Module:
    if architecture == "mobilenet_v3_large":
        model = models.mobilenet_v3_large(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if architecture == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"Desteklenmeyen TİD mimarisi: {architecture}")


def _scaled_view(batch: torch.Tensor, scale: float) -> torch.Tensor:
    if abs(scale - 1.0) < 1e-6:
        return batch
    return TF.affine(
        batch,
        angle=0.0,
        translate=[0, 0],
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0.0,
    )


def _tta_batch(
    batch: torch.Tensor,
    scales: Iterable[float],
    mirror: bool,
) -> torch.Tensor:
    views: list[torch.Tensor] = []
    for scale in scales:
        view = _scaled_view(batch, float(scale))
        views.append(view)
        if mirror:
            views.append(torch.flip(view, dims=(-1,)))
    if not views:
        raise ValueError("En az bir TTA ölçeği gerekli")
    return torch.cat(views, dim=0)


@dataclass(frozen=True)
class ImagePrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class _CheckpointClassifier:
    def __init__(self, checkpoint_path: Path, device: torch.device) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        self.labels = [str(label) for label in checkpoint["labels"]]
        self.image_size = int(checkpoint.get("image_size", IMAGE_SIZE))
        self.temperature = float(checkpoint.get("temperature", 1.0))
        self.scales = tuple(
            float(value) for value in checkpoint.get("tta_scales", (0.82, 1.0, 1.18))
        )
        self.mirror = bool(checkpoint.get("tta_mirror", True))
        self.device = device
        self.model = _build_model(str(checkpoint["architecture"]), len(self.labels))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(device).eval()
        if device.type == "cuda":
            self.model.to(memory_format=torch.channels_last)

    @torch.inference_mode()
    def probability_vector(self, batch: torch.Tensor) -> np.ndarray:
        views = _tta_batch(batch, self.scales, self.mirror)
        if self.device.type == "cuda":
            views = views.contiguous(memory_format=torch.channels_last)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda",
        ):
            logits = self.model(views).mean(dim=0, keepdim=True)
            probabilities = torch.softmax(logits / self.temperature, dim=1)[0]
        return probabilities.float().cpu().numpy()


class DigitraTIDEnsemble:
    """Validation-selected MobileNetV3/EfficientNet TİD ensemble."""

    def __init__(self, config_path: str | Path, device: str | None = None) -> None:
        path = Path(config_path)
        config = json.loads(path.read_text(encoding="utf-8"))
        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(selected_device)
        self.recommended_confidence = float(config.get("recommended_confidence", 0.28))
        self.members: list[tuple[_CheckpointClassifier, float]] = []
        for member in config["members"]:
            weight = float(member["weight"])
            if weight <= 0:
                continue
            checkpoint_path = Path(member["checkpoint"])
            if not checkpoint_path.is_absolute():
                checkpoint_path = path.parent / checkpoint_path
            self.members.append(
                (_CheckpointClassifier(checkpoint_path, self.device), weight)
            )
        if not self.members:
            raise ValueError("TİD ensemble en az bir model içermeli")

        self.labels = self.members[0][0].labels
        self.image_size = self.members[0][0].image_size
        if any(member.labels != self.labels for member, _ in self.members):
            raise ValueError("TİD ensemble sınıf sıraları uyuşmuyor")
        if any(member.image_size != self.image_size for member, _ in self.members):
            raise ValueError("TİD ensemble görüntü boyutları uyuşmuyor")
        self.transform = _evaluation_transform(self.image_size)
        self._lock = Lock()
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            self.probability_vector(
                Image.new("RGB", (self.image_size, self.image_size), (114, 114, 114))
            )
            torch.cuda.synchronize(self.device)

    def probability_vector(self, image: Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        batch = self.transform(image).unsqueeze(0).to(self.device)
        total = np.zeros(len(self.labels), dtype=np.float64)
        total_weight = 0.0
        with self._lock:
            for classifier, weight in self.members:
                total += weight * classifier.probability_vector(batch)
                total_weight += weight
        if total_weight <= 0:
            raise RuntimeError("TİD ensemble toplam ağırlığı sıfır")
        return (total / total_weight).astype(np.float32)

    def predict(self, image: Image.Image | np.ndarray) -> ImagePrediction:
        probabilities = self.probability_vector(image)
        index = int(np.argmax(probabilities))
        return ImagePrediction(
            label=self.labels[index],
            confidence=float(probabilities[index]),
            probabilities={
                label: float(probabilities[position])
                for position, label in enumerate(self.labels)
            },
        )
