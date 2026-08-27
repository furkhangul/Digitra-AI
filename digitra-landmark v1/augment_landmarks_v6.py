import pandas as pd
import numpy as np


INPUT_CSV = "turkish_landmarks_final.csv"
OUTPUT_CSV = "turkish_landmarks_augmented_v6.csv"

TARGET_COUNT = 75

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)


# ==========================================================
# VERİYİ OKU
# ==========================================================

df = pd.read_csv(
    INPUT_CSV,
    encoding="utf-8-sig"
)


# ==========================================================
# FEATURE KOLONLARI
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
# LANDMARK AUGMENTATION
# ==========================================================

def augment_hand(values):

    points = np.array(
        values,
        dtype=np.float32
    ).reshape(21, 3)

    # ------------------------------------------------------
    # Küçük rotasyon
    # ------------------------------------------------------

    angle = np.deg2rad(
        np.random.uniform(
            -8,
            8
        )
    )

    rotation = np.array([
        [
            np.cos(angle),
            -np.sin(angle)
        ],
        [
            np.sin(angle),
            np.cos(angle)
        ]
    ])

    xy = points[:, :2]

    xy = xy @ rotation.T

    points[:, :2] = xy


    # ------------------------------------------------------
    # Küçük scale
    # ------------------------------------------------------

    scale = np.random.uniform(
        0.95,
        1.05
    )

    points *= scale


    # ------------------------------------------------------
    # Küçük jitter
    # ------------------------------------------------------

    noise = np.random.normal(
        loc=0.0,
        scale=0.006,
        size=points.shape
    )

    # Wrist'i mümkün olduğunca sabit tut
    noise[0] = 0

    points += noise


    return points.flatten()


# ==========================================================
# YENİ SATIR ÜRET
# ==========================================================

def augment_row(row):

    new_row = row.copy()

    hand1_present = int(
        row["hand1_present"]
    )

    hand2_present = int(
        row["hand2_present"]
    )


    if hand1_present == 1:

        new_row[
            hand1_cols
        ] = augment_hand(
            row[
                hand1_cols
            ].values
        )


    if hand2_present == 1:

        new_row[
            hand2_cols
        ] = augment_hand(
            row[
                hand2_cols
            ].values
        )


    # ------------------------------------------------------
    # İki el ilişkisine de küçük noise
    # ------------------------------------------------------

    if (
        hand1_present == 1
        and
        hand2_present == 1
    ):

        new_row["wrist_dx"] = (
            row["wrist_dx"]
            +
            np.random.normal(
                0,
                0.008
            )
        )

        new_row["wrist_dy"] = (
            row["wrist_dy"]
            +
            np.random.normal(
                0,
                0.008
            )
        )

        dx = new_row["wrist_dx"]
        dy = new_row["wrist_dy"]

        new_row[
            "wrist_distance"
        ] = np.sqrt(
            dx**2 +
            dy**2
        )


    return new_row


# ==========================================================
# AUGMENT
# ==========================================================

augmented_rows = []

counts = (
    df["label"]
    .value_counts()
)


print()
print("========================================")
print(" DIGITRA LANDMARK AUGMENTATION V6")
print("========================================")
print()


for label, count in counts.items():

    print(
        f"{label} -> başlangıç: {count}"
    )

    # Yeterli veri varsa dokunma
    if count >= TARGET_COUNT:
        continue

    class_data = df[
        df["label"] == label
    ]

    needed = (
        TARGET_COUNT -
        count
    )

    print(
        f"   + {needed} sentetik örnek üretilecek"
    )


    for _ in range(needed):

        random_index = np.random.choice(
            class_data.index
        )

        source_row = class_data.loc[
            random_index
        ]

        new_row = augment_row(
            source_row
        )

        augmented_rows.append(
            new_row
        )


# ==========================================================
# BİRLEŞTİR
# ==========================================================

if augmented_rows:

    augmented_df = pd.DataFrame(
        augmented_rows
    )

    final_df = pd.concat(
        [
            df,
            augmented_df
        ],
        ignore_index=True
    )

else:

    final_df = df.copy()


# ==========================================================
# KARIŞTIR
# ==========================================================

final_df = final_df.sample(
    frac=1,
    random_state=RANDOM_SEED
).reset_index(
    drop=True
)


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
print(" AUGMENTATION SONRASI")
print("========================================")
print()


new_counts = (
    final_df[
        "label"
    ]
    .value_counts()
    .sort_index()
)


for label, count in new_counts.items():

    print(
        f"{label:3} -> {count}"
    )


print()
print(
    "Toplam veri:",
    len(final_df)
)

print(
    "Sınıf sayısı:",
    final_df[
        "label"
    ].nunique()
)

print(
    "Dosya:",
    OUTPUT_CSV
)