import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_CSV = "turkish_landmarks_final.csv"

TRAIN_CSV = "train_v7.csv"
VAL_CSV = "val_v7.csv"
TEST_CSV = "test_v7.csv"

RANDOM_STATE = 42


# ==========================================================
# VERİYİ OKU
# ==========================================================

df = pd.read_csv(
    INPUT_CSV,
    encoding="utf-8-sig"
)


print()
print("========================================")
print(" DIGITRA TRAIN / VAL / TEST SPLIT V7")
print("========================================")
print()

print("Toplam gerçek örnek:", len(df))
print("Toplam sınıf:", df["label"].nunique())


# ==========================================================
# X / y
# ==========================================================

X = df.drop(
    columns=["label"]
)

y = df["label"]


# ==========================================================
# 1. SPLIT
#
# %70 TRAIN
# %30 TEMP
# ==========================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)


# ==========================================================
# 2. TEMP -> VAL / TEST
#
# %15 VAL
# %15 TEST
# ==========================================================

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=temp_df["label"]
)


# ==========================================================
# KAYDET
# ==========================================================

train_df.to_csv(
    TRAIN_CSV,
    index=False,
    encoding="utf-8-sig"
)

val_df.to_csv(
    VAL_CSV,
    index=False,
    encoding="utf-8-sig"
)

test_df.to_csv(
    TEST_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ==========================================================
# RAPOR
# ==========================================================

print()
print("========================================")
print(" SPLIT SONUCU")
print("========================================")
print()

print(
    "Train:",
    len(train_df),
    f"(%{len(train_df)/len(df)*100:.2f})"
)

print(
    "Validation:",
    len(val_df),
    f"(%{len(val_df)/len(df)*100:.2f})"
)

print(
    "Test:",
    len(test_df),
    f"(%{len(test_df)/len(df)*100:.2f})"
)


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


print()
print("Dosyalar oluşturuldu:")

print(TRAIN_CSV)
print(VAL_CSV)
print(TEST_CSV)