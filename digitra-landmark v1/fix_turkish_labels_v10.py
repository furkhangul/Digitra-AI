import pandas as pd
import os


LABEL_MAP = {
    "!": "İ",
    "+": "Ğ",
    ",": "Ç",
    ";": "Ş",
    "=": "Ü",
    "_": "Ö"
}


FILES = [
    "turkish_landmarks_final.csv",
    "train_v7.csv",
    "val_v7.csv",
    "test_v7.csv",
    "train_augmented_v8.csv"
]


print()
print("========================================")
print(" DIGITRA TURKISH LABEL FIX V10")
print("========================================")
print()


for filename in FILES:

    if not os.path.exists(filename):

        print(
            f"BULUNAMADI -> {filename}"
        )

        continue


    df = pd.read_csv(
        filename,
        encoding="utf-8-sig"
    )


    # --------------------------------------
    # Eski label sayıları
    # --------------------------------------

    print()
    print(
        "İşleniyor:",
        filename
    )


    for old, new in LABEL_MAP.items():

        count = (
            df["label"] == old
        ).sum()

        if count > 0:

            print(
                f"   {old} -> {new} : {count}"
            )


    # --------------------------------------
    # Değiştir
    # --------------------------------------

    df["label"] = df[
        "label"
    ].replace(
        LABEL_MAP
    )


    # --------------------------------------
    # Yeni dosya
    # --------------------------------------

    name, ext = os.path.splitext(
        filename
    )

    output = (
        name
        + "_tr"
        + ext
    )


    df.to_csv(
        output,
        index=False,
        encoding="utf-8-sig"
    )


    print(
        "   Kaydedildi ->",
        output
    )


print()
print("========================================")
print(" KONTROL")
print("========================================")
print()


df = pd.read_csv(
    "train_augmented_v8_tr.csv",
    encoding="utf-8-sig"
)


classes = sorted(
    df["label"].unique()
)


print(
    "Toplam sınıf:",
    len(classes)
)

print()

print(
    "Sınıflar:"
)

print(
    classes
)


print()
print("Bitti.")