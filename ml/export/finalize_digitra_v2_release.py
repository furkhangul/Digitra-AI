"""Assemble static and temporal Digitra V2 artifacts into one audited release."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", type=Path, required=True)
    parser.add_argument("--dynamic", type=Path, required=True)
    parser.add_argument("--static-audit", type=Path, required=True)
    parser.add_argument("--dynamic-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sources", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite release: {args.output}")
    args.output.mkdir(parents=True)

    shutil.copytree(args.static, args.output / "static")
    shutil.copytree(args.dynamic, args.output / "dynamic")
    # Training-only state_dict duplicates are omitted from the delivery bundle;
    # metadata-rich `.pt` files and validated runtime exports remain.
    for redundant in [
        args.output / "static" / "best_image_model.pt",
        args.output / "dynamic" / "best_temporal_v2.pt",
    ]:
        if redundant.exists():
            redundant.unlink()
    for cache_dir in sorted(args.output.rglob("__pycache__"), reverse=True):
        shutil.rmtree(cache_dir)
    for bytecode in args.output.rglob("*.pyc"):
        bytecode.unlink()
    shutil.copy2(args.static_audit, args.output / "static_data_audit.json")
    shutil.copy2(args.dynamic_audit, args.output / "dynamic_data_audit.json")
    source_dir = args.output / "source"
    source_dir.mkdir()
    for path in args.sources:
        shutil.copy2(path, source_dir / path.name)

    static_metrics = json.loads((args.static / "metrics.json").read_text())
    dynamic_metrics = json.loads((args.dynamic / "metrics.json").read_text())
    strict_static = static_metrics.get("strict_reject", {}).get(
        "global_zero_error_plus_fixed_margin_0_005"
    )
    if strict_static:
        static_card_path = args.output / "static" / "model_card.md"
        static_card = static_card_path.read_text(encoding="utf-8")
        static_card += (
            "\n## Ultra Accurate reject mode\n"
            "Threshold is selected using calibration only: zero calibration errors "
            "plus a fixed 0.005 margin. See `strict_reject_report.json`. This metric "
            "covers accepted samples only and is not raw stream accuracy.\n"
        )
        static_card_path.write_text(static_card, encoding="utf-8")
    correct_dynamic_limitation = (
        "SigNN does not publish signer IDs; label-stratified contiguous video-ID "
        "blocks are disjoint, but a signer-disjoint positive split cannot be proven."
    )
    dynamic_metrics["positive_split_limitation"] = correct_dynamic_limitation
    (args.output / "dynamic" / "metrics.json").write_text(
        json.dumps(dynamic_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    dynamic_card_path = args.output / "dynamic" / "model_card.md"
    dynamic_card = dynamic_card_path.read_text(encoding="utf-8")
    dynamic_card = dynamic_card.replace(
        "pozitif J/Z split'i group-disjoint olsa da signer-disjoint olduğu kanıtlanamaz",
        "pozitif J/Z split'i video-ID-disjoint olsa da signer-disjoint olduğu kanıtlanamaz",
    )
    dynamic_card_path.write_text(dynamic_card, encoding="utf-8")

    # Rebuild nested manifests after provenance corrections.
    for subtree_name in ("static", "dynamic"):
        subtree = args.output / subtree_name
        subtree_manifest = {}
        for path in sorted(subtree.rglob("*")):
            if path.is_file() and path.name != "MANIFEST.json":
                subtree_manifest[path.relative_to(subtree).as_posix()] = {
                    "sha256": sha256(path), "bytes": path.stat().st_size
                }
        (subtree / "MANIFEST.json").write_text(
            json.dumps(subtree_manifest, indent=2), encoding="utf-8"
        )
    summary = {
        "release": "Digitra V2 Hybrid + Temporal",
        "static_scope": static_metrics["scope"],
        "dynamic_scope": dynamic_metrics["scope"],
        "static_locked_test": static_metrics["locked_test"],
        "static_selective_accuracy": static_metrics["selective_accuracy"],
        "static_ultra_accurate": strict_static,
        "dynamic_locked_test": dynamic_metrics["locked_test"],
        "dynamic_selective_accuracy": dynamic_metrics["selective_accuracy"],
        "claims": {
            "asl_not_tid": True,
            "universal_99_9_guarantee": False,
            "test_used_for_training_or_threshold": False,
        },
        "required_production_followup": (
            "Consented real Digitra webcam samples, especially non-sign/random motion, "
            "must be collected for device-domain robustness and external evaluation."
        ),
    }
    (args.output / "release_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    readme = """# Digitra V2 release\n\nThis release keeps V1 intact and adds a DINOv3 RGB + same-sample 422D landmark static hybrid, calibrated reject mode, and a separate J/Z/OTHER temporal gate.\n\n- `static/metrics.json`: raw, calibrated and selective signer-held-out results.\n- `dynamic/metrics.json`: J/Z/OTHER temporal results and dataset limitations.\n- `*_data_audit.json`: split, duplicate and detector audits.\n- `source/`: reproducible training/export sources.\n- `MANIFEST.json`: SHA-256 and byte size for every file.\n\nThis is ASL fingerspelling, not Turkish Sign Language. Selective 99.9% is accuracy among accepted high-confidence samples, not universal camera-stream accuracy.\n"""
    (args.output / "README.md").write_text(readme, encoding="utf-8")

    manifest = {}
    for path in sorted(args.output.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.json":
            manifest[path.relative_to(args.output).as_posix()] = {
                "sha256": sha256(path), "bytes": path.stat().st_size
            }
    (args.output / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    archive = Path(shutil.make_archive(str(args.output), "zip", args.output))
    archive_sha = sha256(archive)
    Path(str(archive) + ".sha256").write_text(
        f"{archive_sha} {archive.name}\n", encoding="utf-8"
    )
    print("DIGITRA_V2_RELEASE_READY", archive, flush=True)
    print("SHA256", archive_sha, flush=True)


if __name__ == "__main__":
    main()
