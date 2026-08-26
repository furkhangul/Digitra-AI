import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from sklearn.neural_network import MLPClassifier


# ==========================================================
# DOSYALAR
# ==========================================================

TRAIN_CSV = "train_augmented_v8.csv"
VAL_CSV = "val_v7.csv"
TEST_CSV = "test_v7.csv"

MODEL_FILE = "digitra_mlp_v9.pkl"
SCALER_FILE = "digitra_scaler_v9.pkl"
ENCODER_FILE = "digitra_label_encoder_v9.pkl"


# ==========================================================
# VERİLERİ OKU
# ==========================================================

train_df = pd.read_csv(
    TRAIN_CSV,
    encoding="utf-8-sig"
)

val_df = pd.read_csv(
    VAL_CSV,
    encoding="utf-8-sig"
)

test_df = pd.read_csv(
    TEST_CSV,
    encoding="utf-8-sig"
)


print()
print("========================================")
print(" DIGITRA MLP TRAINING V9")
print("========================================")
print()

print("Train:", len(train_df))
print("Validation:", len(val_df))
print("Test:", len(test_df))


# ==========================================================
# KULLANILMAYACAK KOLONLAR
# ==========================================================
#
# quality_score tahminde gerçek zamanlı olarak
# güvenilir bir giriş özelliği değil.
# Bu yüzden modele vermiyoruz.
#
# ==========================================================

DROP_COLUMNS = [
    "label",
    "quality_score"
]


feature_columns = [
    col
    for col in train_df.columns
    if col not in DROP_COLUMNS
]


print()
print(
    "Feature sayısı:",
    len(feature_columns)
)


# ==========================================================
# X / Y
# ==========================================================

X_train = train_df[
    feature_columns
].values

X_val = val_df[
    feature_columns
].values

X_test = test_df[
    feature_columns
].values


y_train_text = train_df[
    "label"
].values

y_val_text = val_df[
    "label"
].values

y_test_text = test_df[
    "label"
].values


# ==========================================================
# LABEL ENCODER
# ==========================================================

encoder = LabelEncoder()

y_train = encoder.fit_transform(
    y_train_text
)

y_val = encoder.transform(
    y_val_text
)

y_test = encoder.transform(
    y_test_text
)


print()
print(
    "Sınıflar:"
)

print(
    list(encoder.classes_)
)


# ==========================================================
# STANDARD SCALER
# ==========================================================
#
# ÖNEMLİ:
# scaler sadece TRAIN üzerinde fit edilir.
#
# ==========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_val_scaled = scaler.transform(
    X_val
)

X_test_scaled = scaler.transform(
    X_test
)


# ==========================================================
# MLP
# ==========================================================

model = MLPClassifier(

    hidden_layer_sizes=(
        256,
        128,
        64
    ),

    activation="relu",

    solver="adam",

    alpha=0.001,

    batch_size=32,

    learning_rate_init=0.001,

    max_iter=500,

    early_stopping=True,

    validation_fraction=0.15,

    n_iter_no_change=30,

    random_state=42,

    verbose=True
)


# ==========================================================
# EĞİTİM
# ==========================================================

print()
print("========================================")
print(" EĞİTİM BAŞLIYOR")
print("========================================")
print()


model.fit(
    X_train_scaled,
    y_train
)


# ==========================================================
# TAHMİNLER
# ==========================================================

train_pred = model.predict(
    X_train_scaled
)

val_pred = model.predict(
    X_val_scaled
)

test_pred = model.predict(
    X_test_scaled
)


# ==========================================================
# ACCURACY
# ==========================================================

train_acc = accuracy_score(
    y_train,
    train_pred
)

val_acc = accuracy_score(
    y_val,
    val_pred
)

test_acc = accuracy_score(
    y_test,
    test_pred
)


# ==========================================================
# MACRO F1
# ==========================================================

train_f1 = f1_score(
    y_train,
    train_pred,
    average="macro"
)

val_f1 = f1_score(
    y_val,
    val_pred,
    average="macro"
)

test_f1 = f1_score(
    y_test,
    test_pred,
    average="macro"
)


# ==========================================================
# SONUÇ
# ==========================================================

print()
print("========================================")
print(" GENEL SONUÇ")
print("========================================")
print()

print(
    f"Train Accuracy : %{train_acc * 100:.2f}"
)

print(
    f"Val Accuracy   : %{val_acc * 100:.2f}"
)

print(
    f"Test Accuracy  : %{test_acc * 100:.2f}"
)

print()

print(
    f"Train Macro F1 : %{train_f1 * 100:.2f}"
)

print(
    f"Val Macro F1   : %{val_f1 * 100:.2f}"
)

print(
    f"Test Macro F1  : %{test_f1 * 100:.2f}"
)


# ==========================================================
# TEST CLASSIFICATION REPORT
# ==========================================================

print()
print("========================================")
print(" TEST CLASSIFICATION REPORT")
print("========================================")
print()


print(
    classification_report(
        y_test,
        test_pred,
        target_names=encoder.classes_,
        digits=4,
        zero_division=0
    )
)


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

cm = confusion_matrix(
    y_test,
    test_pred
)


cm_df = pd.DataFrame(
    cm,
    index=encoder.classes_,
    columns=encoder.classes_
)


cm_df.to_csv(
    "confusion_matrix_v9.csv",
    encoding="utf-8-sig"
)


print()
print(
    "Confusion matrix kaydedildi:"
)

print(
    "confusion_matrix_v9.csv"
)


# ==========================================================
# MODELİ KAYDET
# ==========================================================

joblib.dump(
    model,
    MODEL_FILE
)

joblib.dump(
    scaler,
    SCALER_FILE
)

joblib.dump(
    encoder,
    ENCODER_FILE
)


print()
print("========================================")
print(" MODEL DOSYALARI")
print("========================================")
print()

print(MODEL_FILE)
print(SCALER_FILE)
print(ENCODER_FILE)

print()
print("Eğitim tamamlandı.")