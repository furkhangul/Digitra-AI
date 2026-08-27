import pandas as pd
import numpy as np

INPUT_CSV = "train_v7.csv"
OUTPUT_CSV = "train_augmented_v8.csv"

TARGET_COUNT = 70
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

df = pd.read_csv(INPUT_CSV, encoding="utf-8-sig")


# ==========================================================
# LANDMARK KOLONLARI
# ==========================================================

hand1_cols = []
hand2_cols = []

for i in range(21):

    hand1_cols.extend([
        f"hand1_x{i}",
        f"hand1_y{i}",
        f"hand1_z{i}"
    ])

    hand2_cols.extend([
        f"hand2_x{i}",
        f"hand2_y{i}",
        f"hand2_z{i}"
    ])


# ==========================================================
# AUGMENTATION
# ==========================================================

def augment_hand(values):

    points = np.array(
        values,
        dtype=np.float32
    ).reshape(21, 3)

    # -------------------------
    # Küçük rotasyon
    # -------------------------

    angle = np.deg2rad(
        np.random.uniform(-7, 7)
    )

    rotation = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle),  np.cos(angle)]
    ])

    points[:, :2] = (
        points[:, :2] @ rotation.T
    )


    # -------------------------
    # Scale
    # -------------------------

    scale = np.random.uniform(
        0.96,
        1.04
    )

    points *= scale


    # -------------------------
    # Çok küçük jitter
    # -------------------------

    noise = np.random.normal(
        0,
        0.004,
        points.shape
    )

    # Wrist'e noise verme
    noise[0] = 0

    points += noise

    return points.flatten()


# ==========================================================
# SATIR AUGMENTATION
# ==========================================================

def augment_row(row):

    new_row = row.copy()

    h1 = int(
        float(row["hand1_present"])
    )

    h2 = int(
        float(row["hand2_present"])
    )


    if h1 == 1:

        new_row[hand1_cols] = augment_hand(
            row[hand1_cols].values
        )


    if h2 == 1:

        new_row[hand2_cols] = augment_hand(
            row[hand2_cols].values
        )


    # İki el varsa göreli konuma
    # çok küçük değişiklik
    if h1 == 1 and h2 == 1:

        dx = (
            float(row["wrist_dx"])
            +
            np.random.normal(0, 0.005)
        )

        dy = (
            float(row["wrist_dy"])
            +
            np.random.normal(0, 0.005)
        )

        new_row["wrist_dx"] = dx
        new_row["wrist_dy"] = dy

        new_row["wrist_distance"] = (
            np.sqrt(dx**2 + dy**2)
        )


    return new_row


# ==========================================================
# TRAIN AUGMENT
# ==========================================================

synthetic_rows = []

counts = df["label"].value_counts()


print()
print("========================================")
print(" DIGITRA TRAIN AUGMENTATION V8")
print("========================================")
print()


for label in sorted(counts.index):

    count = counts[label]

    print(
        f"{label:3} -> {count}",
        end=""
    )


    if count >= TARGET_COUNT:

        print(" | yeterli")
        continue


    needed = TARGET_COUNT - count

    print(
        f" | +{needed} üretilecek"
    )


    class_df = df[
        df["label"] == label
    ]


    for _ in range(needed):

        source = class_df.sample(
            n=1
        ).iloc[0]

        synthetic = augment_row(
            source
        )

        synthetic_rows.append(
            synthetic
        )


# ==========================================================
# BİRLEŞTİR
# ==========================================================

synthetic_df = pd.DataFrame(
    synthetic_rows
)


final_df = pd.concat(
    [
        df,
        synthetic_df
    ],
    ignore_index=True
)


# ==========================================================
# KARIŞTIR
# ==========================================================

final_df = final_df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(drop=True)


# ==========================================================
# KAYDET
# ==========================================================

final_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ==========================================================
# RAPOR
# ==========================================================

print()
print("========================================")
print(" AUGMENTATION SONRASI TRAIN")
print("========================================")
print()


counts_after = (
    final_df["label"]
    .value_counts()
    .sort_index()
)


for label, count in counts_after.items():

    print(
        f"{label:3} -> {count}"
    )


print()
print(
    "Gerçek train:",
    len(df)
)

print(
    "Üretilen sentetik:",
    len(synthetic_df)
)

print(
    "Yeni train:",
    len(final_df)
)

print()
print(
    "Validation DEĞİŞMEDİ:",
    "val_v7.csv"
)

print(
    "Test DEĞİŞMEDİ:",
    "test_v7.csv"
)

print()
print(
    "Dosya:",
    OUTPUT_CSV
)