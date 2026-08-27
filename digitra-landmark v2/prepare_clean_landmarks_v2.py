import pandas as pd


INPUT_CSV = "asl_landmarks_full.csv"
OUTPUT_CSV = "asl_landmarks_26letters.csv"

REMOVE_CLASSES = [
    "del",
    "nothing",
    "space"
]


print()
print("========================================")
print(" CLEAN LANDMARK DATASET V2")
print("========================================")
print()


# ==========================================================
# VERİYİ OKU
# ==========================================================

df = pd.read_csv(
    INPUT_CSV,
    encoding="utf-8-sig"
)


print(
    "Başlangıç toplam satır:",
    len(df)
)

print(
    "Başlangıç sınıf sayısı:",
    df["label"].nunique()
)


# ==========================================================
# DEL / NOTHING / SPACE ÇIKAR
# ==========================================================

df = df[
    ~df["label"].isin(
        REMOVE_CLASSES
    )
].copy()


# ==========================================================
# METHOD KOLONUNU ŞİMDİLİK TUT
# ==========================================================
# method:
# original
# 2x
# 3x
# 2x_gamma
# ...
#
# Bu kolonu daha sonra feature olarak modele vermeyeceğiz.
# Sadece analiz amacıyla CSV'de kalıyor.
# ==========================================================


# ==========================================================
# SINIF DAĞILIMI
# ==========================================================

counts = (
    df["label"]
    .value_counts()
    .sort_index()
)


print()
print("========================================")
print(" 26 HARF SINIF DAĞILIMI")
print("========================================")
print()


for label, count in counts.items():

    print(
        f"{label:3} -> {count}"
    )


# ==========================================================
# DENGE ANALİZİ
# ==========================================================

minimum = counts.min()
maximum = counts.max()

ratio = (
    maximum / minimum
)


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
# NULL / NAN KONTROLÜ
# ==========================================================

nan_count = (
    df.isna()
    .sum()
    .sum()
)


print()
print("========================================")
print(" VERİ KALİTE KONTROLÜ")
print("========================================")
print()

print(
    "NaN toplam:",
    nan_count
)


# ==========================================================
# KAYDET
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
    "Toplam temiz örnek:",
    len(df)
)

print(
    "Sınıf sayısı:",
    df["label"].nunique()
)

print(
    "Dosya:",
    OUTPUT_CSV
)