import pandas as pd
from collections import Counter


INPUT_CSV = "turkish_landmarks_v4.csv"
OUTPUT_CSV = "turkish_landmarks_final.csv"

MIN_SAMPLES = 50


# ==========================================================
# VERİYİ OKU
# ==========================================================

df = pd.read_csv(
    INPUT_CSV,
    encoding="utf-8-sig"
)

print()
print("========================================")
print(" DIGITRA FINAL DATASET BUILDER V5")
print("========================================")
print()


# ==========================================================
# TEMEL KONTROL
# ==========================================================

print("Toplam ACCEPT veri:", len(df))

print()
print("Sınıf sayısı:", df["label"].nunique())

print()


# ==========================================================
# SINIF DAĞILIMI
# ==========================================================

counts = (
    df["label"]
    .value_counts()
    .sort_index()
)


print("========================================")
print(" SINIF DAĞILIMI")
print("========================================")
print()

for label, count in counts.items():

    print(
        f"{label:3} -> {count}"
    )


# ==========================================================
# AZ ÖRNEKLİ SINIFLAR
# ==========================================================

print()
print("========================================")
print(" AZ ÖRNEKLİ SINIFLAR")
print("========================================")
print()


low_sample_classes = (
    counts[
        counts < MIN_SAMPLES
    ]
)


if len(low_sample_classes) == 0:

    print(
        "MIN_SAMPLES altında sınıf yok."
    )

else:

    for label, count in low_sample_classes.items():

        print(
            f"{label:3} -> "
            f"{count} örnek"
        )


# ==========================================================
# ÇOK ÖRNEKLİ SINIFLAR
# ==========================================================

print()
print("========================================")
print(" YETERLİ SINIFLAR")
print("========================================")
print()


good_classes = (
    counts[
        counts >= MIN_SAMPLES
    ]
)


for label, count in good_classes.items():

    print(
        f"{label:3} -> "
        f"{count} örnek"
    )


# ==========================================================
# DENGESİZLİK
# ==========================================================

minimum = counts.min()
maximum = counts.max()

ratio = maximum / minimum


print()
print("========================================")
print(" DENGE ANALİZİ")
print("========================================")
print()

print(
    "En az örnek:",
    minimum
)

print(
    "En fazla örnek:",
    maximum
)

print(
    f"Max / Min oranı: {ratio:.2f}"
)


# ==========================================================
# FINAL CSV
# ==========================================================

df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


print()
print("========================================")
print(" FINAL")
print("========================================")
print()

print(
    "Final CSV oluşturuldu:",
    OUTPUT_CSV
)

print()

print(
    "Toplam satır:",
    len(df)
)

print(
    "Toplam sınıf:",
    df["label"].nunique()
)

print()