"""Create a self-contained Digitra V3 model directory and ZIP archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def copy(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-run", type=Path, required=True)
    parser.add_argument("--landmark-run", type=Path, required=True)
    parser.add_argument("--router-run", type=Path, required=True)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--inference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite model package: {args.output}")
    args.output.mkdir(parents=True)

    copies = {
        args.rgb_run / "best_model_state.pt": args.output / "rgb" / "best_model_state.pt",
        args.rgb_run / "development_metrics.json": args.output / "rgb" / "development_metrics.json",
        args.landmark_run / "extra_trees.joblib": args.output / "landmark" / "extra_trees.joblib",
        args.landmark_run / "feature_scaler.joblib": args.output / "landmark" / "feature_scaler.joblib",
        args.landmark_run / "pair_experts.joblib": args.output / "landmark" / "pair_experts.joblib",
        args.landmark_run / "rbf_svc.joblib": args.output / "landmark" / "rbf_svc.joblib",
        args.landmark_run / "neural_ensemble.pt": args.output / "landmark" / "neural_ensemble.pt",
        args.landmark_run / "development_metrics.json": args.output / "landmark" / "development_metrics.json",
        args.router_run / "ru_specialist_p1_p7.joblib": args.output / "router" / "ru_specialist_p1_p7.joblib",
        args.router_run / "router_config.json": args.output / "router" / "router_config.json",
        args.router_run / "development_metrics.json": args.output / "router" / "development_metrics.json",
        args.task: args.output / "hand_landmarker.task",
        args.inference: args.output / "inference.py",
    }
    for source, destination in copies.items():
        copy(source, destination)

    labels = [chr(ord("A") + index) for index in range(26) if index not in (9, 25)]
    rgb_metrics = json.loads((args.rgb_run / "development_metrics.json").read_text())
    landmark_metrics = json.loads((args.landmark_run / "development_metrics.json").read_text())
    router_metrics = json.loads((args.router_run / "development_metrics.json").read_text())
    config = {
        "name": "DIGITRA_V3_HYBRID_9415",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "24 static ASL fingerspelling letters; J/Z are temporal and excluded",
        "labels": labels,
        "rgb": {
            "architecture": rgb_metrics["architecture"],
            "image_size": rgb_metrics["image_size"],
            "temperature": rgb_metrics["temperature"],
            "drop_rate": 0.06,
            "drop_path_rate": 0.14,
        },
        "landmark": {
            "feature_version": "canonical_422_v1",
            "blend_weights": landmark_metrics["blend_weights"],
            "kept_pair_experts": landmark_metrics["kept_pair_experts"],
        },
        "router_development_metrics": router_metrics,
        "historical_locked_test": {
            "accuracy": 0.9414583333333333,
            "macro_f1": 0.925553,
            "errors": 281,
            "samples": 4800,
            "note": "Historical one-shot P9/P10 result from the original frozen V3 run; not re-labeled as a new test.",
        },
    }
    (args.output / "model_config.json").write_text(json.dumps(config, indent=2))
    (args.output / "requirements.txt").write_text(
        "joblib\nmediapipe\nnumpy\nPillow\nscikit-learn\ntimm\ntorch\ntorchvision\n"
    )

    files = [path for path in args.output.rglob("*") if path.is_file()]
    manifest = {
        "name": config["name"],
        "files": {
            str(path.relative_to(args.output)).replace("\\", "/"): {
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in sorted(files)
        },
    }
    (args.output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    archive = shutil.make_archive(str(args.output), "zip", args.output.parent, args.output.name)
    print("DIGITRA_V3_PACKAGE", args.output, flush=True)
    print("DIGITRA_V3_ARCHIVE", archive, flush=True)


if __name__ == "__main__":
    main()
