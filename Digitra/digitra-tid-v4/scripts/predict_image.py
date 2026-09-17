from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

from src.cnn import TidImageClassifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    classifier = TidImageClassifier(args.checkpoint)
    with Image.open(args.image) as opened:
        prediction = classifier.predict(opened.convert("RGB"))
    ranked = sorted(prediction.probabilities.items(), key=lambda item: item[1], reverse=True)
    for label, probability in ranked[: args.top_k]:
        print(f"{label}: {probability:.4f}")


if __name__ == "__main__":
    main()
