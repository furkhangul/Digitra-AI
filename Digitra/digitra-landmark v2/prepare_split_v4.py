import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_CSV = "asl_landmarks_geometric_26letters.csv"

TRAIN_FILE = "train_v4.csv"
VAL_FILE = "val_v4.csv"
TEST_FILE = "test_v4.csv"

RANDOM_STATE = 42


print()
print("========================================")
print(" DIGITRA SPLIT V4")
print("========================================")
print()


# ==========================================================
# VERİYİ OKU
# ==========================================================

df = pd.read_csv(
    INPUT_CSV,
    encoding="utf-8-sig"
)

print("Toplam veri :", len(df))
print("Sınıf sayısı:", df["label"].nunique())
print("Feature     :", len(df.columns) - 1)


# ==========================================================
# 70 / 15 / 15
# ==========================================================
#
# Önce:
# %70 train
# %30 geçici
#
# Sonra geçicinin yarısı:
# %15 validation
# %15 test
#
# Stratify sayesinde sınıf oranları korunur.
# ==========================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=df["label"],
    shuffle=True
)


val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["label"],
    shuffle=True
)


# ==========================================================
# INDEX TEMİZLE
# ==========================================================

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)


# ==========================================================
# KAYDET
# ==========================================================

train_df.to_csv(
    TRAIN_FILE,
    index=False,
    encoding="utf-8-sig"
)

val_df.to_csv(
    VAL_FILE,
    index=False,
    encoding="utf-8-sig"
)

test_df.to_csv(
    TEST_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ==========================================================
# RAPOR
# ==========================================================

total = len(df)

print()
print("========================================")
print(" SPLIT SONUCU")
print("========================================")
print()

print(
    f"Train      : {len(train_df)} "
    f"(%{len(train_df) / total * 100:.2f})"
)

print(
    f"Validation : {len(val_df)} "
    f"(%{len(val_df) / total * 100:.2f})"
)

print(
    f"Test       : {len(test_df)} "
    f"(%{len(test_df) / total * 100:.2f})"
)


# ==========================================================
# SINIF DAĞILIMLARI
# ==========================================================

print()
print("========================================")
print(" TRAIN SINIF DAĞILIMI")
print("========================================")

print(
    train_df["label"]
    .value_counts()
    .sort_index()
)


print()
print("========================================")
print(" VALIDATION SINIF DAĞILIMI")
print("========================================")

print(
    val_df["label"]
    .value_counts()
    .sort_index()
)


print()
print("========================================")
print(" TEST SINIF DAĞILIMI")
print("========================================")

print(
    test_df["label"]
    .value_counts()
    .sort_index()
)


# ==========================================================
# KONTROLLER
# ==========================================================

print()
print("========================================")
print(" KONTROL")
print("========================================")
print()

print(
    "Train NaN:",
    train_df.isna().sum().sum()
)

print(
    "Val NaN:",
    val_df.isna().sum().sum()
)

print(
    "Test NaN:",
    test_df.isna().sum().sum()
)

print(
    "Train sınıf:",
    train_df["label"].nunique()
)

print(
    "Val sınıf:",
    val_df["label"].nunique()
)

print(
    "Test sınıf:",
    test_df["label"].nunique()
)


print()
print("Dosyalar:")
print(TRAIN_FILE)
print(VAL_FILE)
print(TEST_FILE)

print()
print("V4 tamamlandı.")