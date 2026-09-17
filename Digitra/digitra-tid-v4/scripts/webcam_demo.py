from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from src.cnn import TidImageClassifier
from src.config import MODELS_DIR
from src.ensemble import TidEnsembleClassifier
from src.hand_crop import MediaPipeHandCropper


def main() -> None:
    parser = argparse.ArgumentParser()
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        "--checkpoint",
        type=Path,
        help="Tek bir CNN checkpoint'i kullan (varsayılan: doğrulamada seçilmiş ensemble).",
    )
    parser.add_argument("--camera", type=int, default=0)
    model_group.add_argument(
        "--ensemble-config",
        type=Path,
        default=MODELS_DIR / "robust_cnn_ensemble.json",
        help="CNN ensemble ayarı.",
    )
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--smoothing", type=int, default=5)
    parser.add_argument("--no-mirror", action="store_true")
    args = parser.parse_args()

    classifier = (
        TidImageClassifier(args.checkpoint)
        if args.checkpoint is not None
        else TidEnsembleClassifier(args.ensemble_config)
    )
    confidence_threshold = (
        args.confidence
        if args.confidence is not None
        else float(getattr(classifier, "recommended_confidence", 0.45))
    )
    history: deque[np.ndarray] = deque(maxlen=max(1, args.smoothing))
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Kamera açılamadı: {args.camera}")

    try:
        with MediaPipeHandCropper() as cropper:
            while True:
                ok, frame = camera.read()
                if not ok:
                    break
                if not args.no_mirror:
                    frame = cv2.flip(frame, 1)
                hand = cropper.crop(frame)
                title = "El bekleniyor"
                color = (0, 180, 255)
                if hand is None:
                    history.clear()
                else:
                    probabilities = classifier.probability_vector(hand.rgb)
                    history.append(probabilities)
                    smooth = np.mean(np.stack(history), axis=0)
                    index = int(np.argmax(smooth))
                    confidence = float(smooth[index])
                    label = classifier.labels[index]
                    title = f"{label}  {confidence:.0%}" if confidence >= confidence_threshold else f"?  {confidence:.0%}"
                    color = (40, 210, 80) if confidence >= confidence_threshold else (0, 180, 255)
                    x1, y1, x2, y2 = hand.box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    title,
                    (24, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    color,
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Digitra TID Robust V5 - Q: cikis", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
