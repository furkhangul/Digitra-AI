from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.cnn import Letterbox, TidImageClassifier, evaluation_transform, tta_logits
from src.config import ARTIFACTS_DIR, MODELS_DIR, REPORTS_DIR

HOG_SIZE = (96, 96)
HOG = cv2.HOGDescriptor(HOG_SIZE, (16, 16), (8, 8), (8, 8), 9)
VARIANTS = ("clean", "mirror", "far", "near", "shifted_far", "low_light")


def variant_image(image: Image.Image, variant: str, size: int = 224) -> Image.Image:
    if variant == "clean":
        return image.convert("RGB")
    if variant == "mirror":
        return ImageOps.mirror(image.convert("RGB"))
    base = Letterbox(size)(image)
    if variant == "far":
        small = base.resize((int(size * 0.55), int(size * 0.55)), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (size, size), (114, 114, 114))
        offset = ((size - small.width) // 2, (size - small.height) // 2)
        canvas.paste(small, offset)
        return canvas
    if variant == "near":
        large = base.resize((int(size * 1.32), int(size * 1.32)), Image.Resampling.BICUBIC)
        left = (large.width - size) // 2
        top = (large.height - size) // 2
        return large.crop((left, top, left + size, top + size))
    if variant == "shifted_far":
        small = base.resize((int(size * 0.66), int(size * 0.66)), Image.Resampling.BICUBIC)
        canvas = Image.new("RGB", (size, size), (114, 114, 114))
        canvas.paste(small, (int(size * 0.04), int(size * 0.27)))
        return canvas
    if variant == "low_light":
        return ImageEnhance.Contrast(ImageEnhance.Brightness(base).enhance(0.48)).enhance(1.20)
    raise ValueError(variant)


class VariantDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, labels: list[str], variant: str) -> None:
        self.frame = frame.reset_index(drop=True)
        self.to_index = {label: index for index, label in enumerate(labels)}
        self.variant = variant
        self.transform = evaluation_transform()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        with Image.open(row.path) as opened:
            image = variant_image(opened.convert("RGB"), self.variant)
        return self.transform(image), self.to_index[str(row.label)], str(row.path)


def hog_feature(image: Image.Image) -> np.ndarray:
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    scale = min(HOG_SIZE[0] / width, HOG_SIZE[1] / height)
    resized = cv2.resize(
        gray,
        (max(1, int(width * scale)), max(1, int(height * scale))),
    )
    canvas = np.zeros(HOG_SIZE, dtype=np.uint8)
    y = (HOG_SIZE[1] - resized.shape[0]) // 2
    x = (HOG_SIZE[0] - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(canvas)
    return HOG.compute(enhanced).reshape(-1)


@torch.inference_mode()
def evaluate_cnn(classifier: TidImageClassifier, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    predictions = []
    targets = []
    for images, truth, _ in tqdm(loader, desc="CNN", leave=False):
        images = images.to(classifier.device)
        logits = tta_logits(
            classifier.model, images, classifier.scales, classifier.mirror
        )
        predictions.extend(logits.argmax(1).cpu().numpy().tolist())
        targets.extend(truth.numpy().tolist())
    return np.asarray(targets), np.asarray(predictions)


def score(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=MODELS_DIR / "static_mobilenet_v3_large_robust.pt",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    manifest = pd.read_csv(ARTIFACTS_DIR / "manifest.csv", encoding="utf-8-sig")
    test = manifest[manifest.split == "test"].copy()
    classifier = TidImageClassifier(args.checkpoint)
    labels = classifier.labels
    hog_model = joblib.load(MODELS_DIR / "static_hog.joblib")
    label_to_index = {label: index for index, label in enumerate(labels)}

    rows = []
    for variant in VARIANTS:
        dataset = VariantDataset(test, labels, variant)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=classifier.device.type == "cuda",
            persistent_workers=False,
        )
        targets, cnn_predictions = evaluate_cnn(classifier, loader)
        rows.append({"model": "robust_cnn", "variant": variant, **score(targets, cnn_predictions)})

        hog_vectors = []
        for path in tqdm(test.path, desc=f"HOG {variant}", leave=False):
            with Image.open(path) as opened:
                hog_vectors.append(hog_feature(variant_image(opened.convert("RGB"), variant)))
        hog_labels = hog_model.predict(np.asarray(hog_vectors, dtype=np.float32))
        hog_predictions = np.asarray([label_to_index[str(label)] for label in hog_labels])
        rows.append({"model": "hog_v4", "variant": variant, **score(targets, hog_predictions)})

    frame = pd.DataFrame(rows)
    csv_path = REPORTS_DIR / "robustness_comparison.csv"
    frame.to_csv(csv_path, index=False)
    output = {
        "checkpoint": str(args.checkpoint.resolve()),
        "test_samples": len(test),
        "protocol": {
            "mirror": "horizontal reflection",
            "far": "55% centered scale",
            "near": "132% center zoom/crop",
            "shifted_far": "66% scale placed off-center",
            "low_light": "48% brightness with contrast shift",
        },
        "results": rows,
    }
    (REPORTS_DIR / "robustness_comparison.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
