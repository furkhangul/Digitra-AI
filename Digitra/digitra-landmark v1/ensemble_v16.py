import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)


# ==========================================================
# DOSYALAR
# ==========================================================

TEST_RAW = "test_v7_tr.csv"
TEST_GEO = "test_geometric_v14.csv"

V9_MODEL = "digitra_mlp_v9.pkl"
V9_SCALER = "digitra_scaler_v9.pkl"
V9_ENCODER = "digitra_label_encoder_v9.pkl"

V15_MODEL = "digitra_mlp_v15.pkl"
V15_SCALER = "digitra_scaler_v15.pkl"
V15_ENCODER = "digitra_label_encoder_v15.pkl"


# ==========================================================
# ESKİ -> TÜRKÇE LABEL MAP
# ==========================================================

OLD_TO_TR = {
    "!": "İ",
    "+": "Ğ",
    ",": "Ç",
    ";": "Ş",
    "=": "Ü",
    "_": "Ö"
}


# ==========================================================
# LOAD
# ==========================================================

test_raw = pd.read_csv(
    TEST_RAW,
    encoding="utf-8-sig"
)

test_geo = pd.read_csv(
    TEST_GEO,
    encoding="utf-8-sig"
)

model9 = joblib.load(V9_MODEL)
scaler9 = joblib.load(V9_SCALER)
encoder9 = joblib.load(V9_ENCODER)

model15 = joblib.load(V15_MODEL)
scaler15 = joblib.load(V15_SCALER)
encoder15 = joblib.load(V15_ENCODER)


# ==========================================================
# FEATURE'LAR
# ==========================================================

raw_features = [
    c for c in test_raw.columns
    if c not in [
        "label",
        "quality_score"
    ]
]

geo_features = [
    c for c in test_geo.columns
    if c != "label"
]


X9 = test_raw[
    raw_features
].values

X15 = test_geo[
    geo_features
].values


X9 = scaler9.transform(
    X9
)

X15 = scaler15.transform(
    X15
)


# ==========================================================
# GERÇEK LABEL
# ==========================================================

y_text = test_raw[
    "label"
].values


# V15 encoder zaten Türkçe sınıfları biliyor
y_true = encoder15.transform(
    y_text
)


# ==========================================================
# PROBABILITY
# ==========================================================

prob9_old = model9.predict_proba(
    X9
)

prob15 = model15.predict_proba(
    X15
)


# ==========================================================
# V9 CLASS İSİMLERİNİ TÜRKÇEYE ÇEVİR
# ==========================================================

classes9_old = encoder9.classes_

classes9_tr = np.array([
    OLD_TO_TR.get(
        cls,
        cls
    )
    for cls in classes9_old
])


classes15 = encoder15.classes_


print()
print("V9 eski sınıflar:")
print(list(classes9_old))

print()
print("V9 Türkçe karşılık:")
print(list(classes9_tr))

print()
print("V15 sınıflar:")
print(list(classes15))


# ==========================================================
# V9 PROBABILITY SÜTUNLARINI
# V15 SIRASINA YENİDEN DİZ
# ==========================================================

prob9 = np.zeros_like(
    prob15
)


for old_index, tr_label in enumerate(
    classes9_tr
):

    new_index = np.where(
        classes15 == tr_label
    )[0]

    if len(new_index) != 1:

        raise ValueError(
            f"Sınıf eşleşmedi: {tr_label}"
        )

    new_index = new_index[0]

    prob9[
        :,
        new_index
    ] = prob9_old[
        :,
        old_index
    ]


# ==========================================================
# ENSEMBLE
# ==========================================================

ensemble_prob = (
    0.60 * prob9
    +
    0.40 * prob15
)


ensemble_pred = np.argmax(
    ensemble_prob,
    axis=1
)


# ==========================================================
# METRICS
# ==========================================================

acc = accuracy_score(
    y_true,
    ensemble_pred
)

f1 = f1_score(
    y_true,
    ensemble_pred,
    average="macro"
)


print()
print("========================================")
print(" DIGITRA ENSEMBLE V16")
print("========================================")
print()

print(
    f"Test Accuracy : %{acc * 100:.2f}"
)

print(
    f"Test Macro F1 : %{f1 * 100:.2f}"
)


print()
print(
    classification_report(
        y_true,
        ensemble_pred,
        target_names=classes15,
        digits=4,
        zero_division=0
    )
)