"""Person-based dataset splitting (no random image split, no person leakage).

Train: P1-P7 | Validation: P8 | Test: P9-P10
"""
from __future__ import annotations

import pandas as pd

from utils import ARTIFACTS_DIR, setup_logging

TRAIN_PERSONS = [f"P{i}" for i in range(1, 8)]   # P1..P7
VAL_PERSONS = ["P8"]
TEST_PERSONS = ["P9", "P10"]


def assign_split(person: str) -> str:
    if person in TRAIN_PERSONS:
        return "train"
    if person in VAL_PERSONS:
        return "val"
    if person in TEST_PERSONS:
        return "test"
    raise ValueError(f"Unknown person {person}")


def run(landmarks_df: pd.DataFrame | None = None,
        force: bool = False) -> pd.DataFrame:
    logger = setup_logging()
    out = ARTIFACTS_DIR / "splits.parquet"
    if out.exists() and not force:
        df = pd.read_parquet(out)
        logger.info("Loaded cached splits: %s", out)
        return df

    if landmarks_df is None:
        landmarks_df = pd.read_parquet(ARTIFACTS_DIR / "landmarks.parquet")

    df = landmarks_df.copy()
    df["split"] = df["person"].map(assign_split)
    df.to_parquet(out, index=False)

    counts = df.groupby("split").size().to_dict()
    logger.info("Split sizes (images): %s", counts)
    # leakage check
    for sp, persons in [("train", TRAIN_PERSONS), ("val", VAL_PERSONS),
                        ("test", TEST_PERSONS)]:
        present = sorted(df[df.split == sp].person.unique())
        assert present == sorted(persons), (sp, present)
    logger.info("Leakage check OK: persons strictly separated across splits.")
    return df


if __name__ == "__main__":
    run()
