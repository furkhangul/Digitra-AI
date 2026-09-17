from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SUPPORTED_ARCHITECTURES = ("mobilenet_v3_large", "efficientnet_b0")


class Letterbox:
    """Resize without distorting the hand and pad to a square canvas."""

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


def training_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Augment handedness, distance, framing, camera and lighting changes."""

    return transforms.Compose(
        [
            Letterbox(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=12,
                translate=(0.16, 0.16),
                scale=(0.55, 1.25),
                shear=(-4, 4, -3, 3),
                interpolation=InterpolationMode.BILINEAR,
                fill=(114, 114, 114),
            ),
            transforms.RandomApply(
                [transforms.RandomPerspective(distortion_scale=0.14, p=1.0)], p=0.20
            ),
            transforms.ColorJitter(
                brightness=0.28,
                contrast=0.25,
                saturation=0.18,
                hue=0.025,
            ),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.1))], p=0.12
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            transforms.RandomErasing(
                p=0.08,
                scale=(0.01, 0.05),
                ratio=(0.4, 2.5),
                value="random",
            ),
        ]
    )


def evaluation_transform(image_size: int = IMAGE_SIZE) -> transforms.Compose:
    return transforms.Compose(
        [
            Letterbox(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_model(
    architecture: str,
    num_classes: int,
    *,
    pretrained: bool,
) -> nn.Module:
    if architecture == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    if architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"Desteklenmeyen mimari: {architecture}; {SUPPORTED_ARCHITECTURES}")


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.features.parameters():
        parameter.requires_grad = trainable


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


def tta_logits(
    model: nn.Module,
    batch: torch.Tensor,
    scales: Iterable[float] = (0.82, 1.0, 1.18),
    mirror: bool = True,
) -> torch.Tensor:
    """Average scale views and their mirrors without inflating GPU batch size."""

    total: torch.Tensor | None = None
    count = 0
    for scale in scales:
        view = _scaled_view(batch, float(scale))
        logits = model(view)
        total = logits if total is None else total + logits
        count += 1
        if mirror:
            logits = model(torch.flip(view, dims=(-1,)))
            total = total + logits
            count += 1
    if total is None:
        raise ValueError("En az bir TTA ölçeği gerekli")
    return total / count


@dataclass(frozen=True)
class Prediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class TidImageClassifier:
    def __init__(self, checkpoint_path: str | Path, device: str | None = None) -> None:
        selected = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(selected)
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.labels = [str(label) for label in checkpoint["labels"]]
        self.image_size = int(checkpoint.get("image_size", IMAGE_SIZE))
        self.temperature = float(checkpoint.get("temperature", 1.0))
        self.recommended_confidence = float(checkpoint.get("recommended_confidence", 0.45))
        self.scales = tuple(float(v) for v in checkpoint.get("tta_scales", (0.82, 1.0, 1.18)))
        self.mirror = bool(checkpoint.get("tta_mirror", True))
        self.model = build_model(
            str(checkpoint["architecture"]), len(self.labels), pretrained=False
        )
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()
        self.transform = evaluation_transform(self.image_size)

    @torch.inference_mode()
    def probability_vector(self, image: Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        batch = self.transform(image).unsqueeze(0).to(self.device)
        logits = tta_logits(self.model, batch, self.scales, self.mirror)
        probabilities = torch.softmax(logits / self.temperature, dim=1)[0]
        return probabilities.detach().cpu().numpy()

    def predict(self, image: Image.Image | np.ndarray) -> Prediction:
        probabilities = self.probability_vector(image)
        index = int(np.argmax(probabilities))
        return Prediction(
            label=self.labels[index],
            confidence=float(probabilities[index]),
            probabilities={
                label: float(probabilities[i]) for i, label in enumerate(self.labels)
            },
        )
