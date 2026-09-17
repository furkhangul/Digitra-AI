"""Export Digitra V2's RGB branch to ONNX and prove prediction parity."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import timm
import torch
from PIL import Image
from torchvision import transforms


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_probe_images(manifest_path: Path, preprocessing: dict, limit: int) -> torch.Tensor:
    rows = []
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["split"] == "test"
                and row["status"] == "OK"
                and row["detected"].lower() == "true"
                and int(row["class_index"]) not in (9, 25)
            ):
                rows.append(row)
    # Deterministic coverage across the locked-test file order.  This parity probe
    # never changes model selection or metrics; it only compares two runtimes.
    indices = np.linspace(0, len(rows) - 1, min(limit, len(rows)), dtype=int)
    transform = transforms.Compose(
        [
            transforms.Resize(
                int(preprocessing["resize"]),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop(int(preprocessing["size"][0])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=tuple(preprocessing["mean"]), std=tuple(preprocessing["std"])
            ),
        ]
    )
    tensors = []
    for index in indices:
        with Image.open(rows[int(index)]["crop_path"]) as opened:
            tensors.append(transform(opened.convert("RGB")))
    return torch.stack(tensors)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=64)
    args = parser.parse_args()

    checkpoint = torch.load(args.bundle / "image_model_v2.pt", map_location="cpu")
    preprocessing = json.loads((args.bundle / "preprocessing.json").read_text())
    image_size = int(checkpoint["image_size"])
    model = timm.create_model(
        checkpoint["backbone"],
        pretrained=False,
        num_classes=len(checkpoint["classes"]),
        img_size=image_size,
        drop_rate=0.08,
        drop_path_rate=0.12,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    output_path = args.bundle / "image_model_v2.onnx"
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        dummy,
        output_path,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=18,
        do_constant_folding=True,
    )
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    probes = load_probe_images(args.manifest, preprocessing, args.samples)
    with torch.inference_mode():
        torch_logits = model(probes).numpy()
    providers = [
        provider for provider in ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if provider in ort.get_available_providers()
    ]
    session = ort.InferenceSession(str(output_path), providers=providers)
    ort_logits = session.run(["logits"], {"image": probes.numpy()})[0]
    difference = np.abs(torch_logits - ort_logits)
    parity = {
        "samples": int(len(probes)),
        "providers": session.get_providers(),
        "max_abs_logit_difference": float(difference.max()),
        "mean_abs_logit_difference": float(difference.mean()),
        "argmax_matches": int(
            np.sum(torch_logits.argmax(axis=1) == ort_logits.argmax(axis=1))
        ),
        "onnx_sha256": sha256(output_path),
        "onnx_bytes": output_path.stat().st_size,
    }
    if parity["argmax_matches"] != parity["samples"]:
        raise RuntimeError(f"ONNX prediction parity failed: {parity}")
    (args.bundle / "onnx_parity.json").write_text(
        json.dumps(parity, indent=2), encoding="utf-8"
    )

    file_manifest = {}
    for path in sorted(args.bundle.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            file_manifest[path.relative_to(args.bundle).as_posix()] = {
                "sha256": sha256(path), "bytes": path.stat().st_size
            }
    (args.bundle / "MANIFEST.json").write_text(
        json.dumps(file_manifest, indent=2), encoding="utf-8"
    )
    archive = Path(shutil.make_archive(str(args.bundle), "zip", args.bundle))
    archive_sha = sha256(archive)
    Path(str(archive) + ".sha256").write_text(
        f"{archive_sha} {archive.name}\n", encoding="utf-8"
    )
    print("ONNX_PARITY_OK", json.dumps(parity), flush=True)
    print("STATIC_FINAL_ARCHIVE", archive, archive_sha, flush=True)


if __name__ == "__main__":
    main()
