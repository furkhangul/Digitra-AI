"""Digitra Landmark V3 - single-command pipeline.

Dataset check -> landmark extraction -> feature engineering -> person split
-> model comparison + Optuna tuning -> final (unseen) test evaluation -> reports.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import json

import utils
from utils import ARTIFACTS_DIR, REPORTS_DIR, setup_logging, DATASET_PATH
from extract_landmarks import run as extract_run
from split_dataset import run as split_run
from feature_engineering import save_metadata
from train_models import run as train_run
from evaluate import write_reports
from inference import benchmark


def main():
    logger = setup_logging()
    utils.ensure_dirs()
    utils.set_seed()

    logger.info("=== STEP 1/5: Landmark extraction ===")
    lm_df = extract_run()

    logger.info("=== STEP 2/5: Person-based split ===")
    splits = split_run(lm_df)
    counts = splits.groupby("split").size().to_dict()

    logger.info("=== STEP 3/5: Feature metadata ===")
    meta = save_metadata()
    n_features = meta["n_features"]

    logger.info("=== STEP 4/5: Model comparison + tuning ===")
    res = train_run()

    logger.info("=== STEP 5/5: Reports + realtime benchmark ===")
    failed_path = ARTIFACTS_DIR / "failed_images.csv"
    failed = len(pd.read_csv(failed_path)) if failed_path.exists() else 0
    successful = len(lm_df)
    total = successful + failed

    try:
        rt = benchmark()
    except Exception as e:
        logger.warning("Benchmark skipped: %s", e)
        rt = {"avg_time_ms": "N/A", "fps": "N/A"}

    train_n = counts.get("train", 0) * (1 + 0)  # base train images
    # augmented train size: base train images * (1 + N_AUG)
    from train_models import N_AUG
    train_n = counts.get("train", 0) * (1 + N_AUG)

    overfit = "YES" if (res["train_val_gap"] > 0.03) else "no (within tolerance)"

    ctx = {
        "total_images": total,
        "classes": res["classes"],
        "persons": 10,
        "train_persons": "P1-P7", "val_persons": "P8", "test_persons": "P9,P10",
        "train_n": train_n, "val_n": counts.get("val", 0),
        "test_n": counts.get("test", 0),
        "successful": successful, "failed": failed,
        "success_rate": successful / max(1, total),
        "mediapipe_model": "hand_landmarker.task (MediaPipe 1.0.1, float16)",
        "n_features": n_features,
        "feature_groups": meta["engineered_groups"],
        "comparison": res["comparison"],
        "chosen": res["chosen"],
        "best_params": res["best_single"]["params"],
        "val_acc": res["val_acc"], "val_f1": res["val_f1"],
        "test_acc": res["test_acc"], "test_prec": res["test_prec"],
        "test_rec": res["test_rec"], "test_f1": res["test_f1"],
        "train_val_gap": res["train_val_gap"],
        "y_test": res["y_test"], "test_pred": res["test_pred"],
        "overfitting": overfit,
        "realtime": {
            "ready": "yes (inference.py)",
            "conf_threshold": 0.55,
            "smoothing": "probability averaging over last 8 frames",
            "avg_time": rt.get("avg_time_ms"), "fps": rt.get("fps"),
        },
        "files_created": [
            "artifacts/landmarks.parquet",
            "artifacts/failed_images.csv",
            "artifacts/splits.parquet",
            "artifacts/feature_metadata.json",
            "artifacts/features_cache.npz",
            "artifacts/models/best_model.pkl (or ensemble_*.pkl)",
            "artifacts/models/scaler.pkl",
            "artifacts/models/label_encoder.pkl",
            "artifacts/models/feature_config.json",
            "artifacts/models/pipeline_meta.json",
            "artifacts/reports/final_report.txt",
            "artifacts/reports/metrics.json",
            "artifacts/reports/classification_report.txt",
            "artifacts/reports/confusion_matrix.png",
            "artifacts/reports/model_comparison.csv",
        ],
        "reliable": "Evet, kisiler arasi genelleme test edildi (P9,P10 gorulmemis).",
        "expected": f"Test acc ~{res['test_acc']*100:.1f}% (gorulmemis kisilerde).",
        "target99": "Hayir" if res["test_acc"] < 0.99 else "Evet",
        "reason": "Kisiler arasi varyasyon, J/Z hareketli harflerin tek kare siniri, benzer harfler (M/N/T, U/V, A/S, G/H) sinif icigi.",
        "improvements": "1) Daha fazla kisi/otomatik augment, 2) sequence/LSTM ile J-Z hareketi, 3) feature ablation + model stacking.",
    }

    txt, metrics, pairs = write_reports(ctx)
    print("\n" + txt)
    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
