import pandas as pd
import numpy as np


FILES = {
    "train": "train_augmented_v8_tr.csv",
    "val": "val_v7_tr.csv",
    "test": "test_v7_tr.csv"
}


# ==========================================================
# MEDIAPIPE LANDMARK INDEXLERİ
# ==========================================================

FINGERS = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20]
}

TIPS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20
}

MCPS = {
    "index": 5,
    "middle": 9,
    "ring": 13,
    "pinky": 17
}


# ==========================================================
# LANDMARK OKU
# ==========================================================

def get_hand(row, prefix):

    points = []

    for i in range(21):

        points.append([
            float(row[f"{prefix}_x{i}"]),
            float(row[f"{prefix}_y{i}"]),
            float(row[f"{prefix}_z{i}"])
        ])

    return np.array(
        points,
        dtype=np.float32
    )


# ==========================================================
# DISTANCE
# ==========================================================

def distance(a, b):

    return float(
        np.linalg.norm(a - b)
    )


# ==========================================================
# ANGLE
# ==========================================================

def angle(a, b, c):

    """
    a-b-c açısı
    merkez = b
    """

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cos_value = (
        np.dot(ba, bc)
        /
        (norm_ba * norm_bc)
    )

    cos_value = np.clip(
        cos_value,
        -1.0,
        1.0
    )

    return float(
        np.arccos(cos_value) / np.pi
    )


# ==========================================================
# TEK EL FEATURE
# ==========================================================

def hand_features(points, prefix):

    features = {}


    # ------------------------------------------------------
    # 1. PARMAK EKLEM AÇILARI
    # ------------------------------------------------------

    for finger, ids in FINGERS.items():

        if finger == "thumb":

            p1, p2, p3, p4 = ids

            features[
                f"{prefix}_{finger}_angle1"
            ] = angle(
                points[p1],
                points[p2],
                points[p3]
            )

            features[
                f"{prefix}_{finger}_angle2"
            ] = angle(
                points[p2],
                points[p3],
                points[p4]
            )

        else:

            mcp, pip, dip, tip = ids

            features[
                f"{prefix}_{finger}_angle_mcp"
            ] = angle(
                points[0],
                points[mcp],
                points[pip]
            )

            features[
                f"{prefix}_{finger}_angle_pip"
            ] = angle(
                points[mcp],
                points[pip],
                points[dip]
            )

            features[
                f"{prefix}_{finger}_angle_dip"
            ] = angle(
                points[pip],
                points[dip],
                points[tip]
            )


    # ------------------------------------------------------
    # 2. WRIST -> FINGERTIP MESAFESİ
    # ------------------------------------------------------

    wrist = points[0]

    for finger, tip in TIPS.items():

        features[
            f"{prefix}_wrist_{finger}_tip"
        ] = distance(
            wrist,
            points[tip]
        )


    # ------------------------------------------------------
    # 3. FINGERTIP ARASI MESAFELER
    # ------------------------------------------------------

    tip_names = list(
        TIPS.keys()
    )

    for i in range(len(tip_names)):

        for j in range(
            i + 1,
            len(tip_names)
        ):

            f1 = tip_names[i]
            f2 = tip_names[j]

            features[
                f"{prefix}_tip_{f1}_{f2}"
            ] = distance(
                points[TIPS[f1]],
                points[TIPS[f2]]
            )


    # ------------------------------------------------------
    # 4. BAŞPARMAK -> DİĞER PARMAKLAR
    #
    # Ç / G gibi sınıfların ayrımında
    # özellikle yararlı olabilir.
    # ------------------------------------------------------

    thumb_tip = points[
        TIPS["thumb"]
    ]

    for finger in [
        "index",
        "middle",
        "ring",
        "pinky"
    ]:

        features[
            f"{prefix}_thumb_to_{finger}"
        ] = distance(
            thumb_tip,
            points[TIPS[finger]]
        )


    # ------------------------------------------------------
    # 5. PALM GEOMETRİSİ
    # ------------------------------------------------------

    features[
        f"{prefix}_palm_width"
    ] = distance(
        points[5],
        points[17]
    )

    features[
        f"{prefix}_palm_height"
    ] = distance(
        points[0],
        points[9]
    )


    # ------------------------------------------------------
    # 6. PARMAK UZUNLUKLARI
    # ------------------------------------------------------

    for finger, ids in FINGERS.items():

        length = 0.0

        previous = 0 if finger == "thumb" else ids[0]

        if finger == "thumb":

            sequence = [
                0,
                1,
                2,
                3,
                4
            ]

        else:

            sequence = ids

        for i in range(
            len(sequence) - 1
        ):

            length += distance(
                points[sequence[i]],
                points[sequence[i + 1]]
            )

        features[
            f"{prefix}_{finger}_length"
        ] = length


    return features


# ==========================================================
# SATIR FEATURE
# ==========================================================

def create_features(row):

    features = {}


    h1_present = int(
        float(row["hand1_present"])
    )

    h2_present = int(
        float(row["hand2_present"])
    )


    # ------------------------------------------------------
    # HAND 1
    # ------------------------------------------------------

    if h1_present:

        hand1 = get_hand(
            row,
            "hand1"
        )

        features.update(
            hand_features(
                hand1,
                "h1"
            )
        )

    else:

        # Kolon yapısını sabit tutmak için
        # sıfır el oluştur
        hand1 = np.zeros(
            (21, 3),
            dtype=np.float32
        )

        temp = hand_features(
            hand1,
            "h1"
        )

        for key in temp:
            features[key] = 0.0


    # ------------------------------------------------------
    # HAND 2
    # ------------------------------------------------------

    if h2_present:

        hand2 = get_hand(
            row,
            "hand2"
        )

        features.update(
            hand_features(
                hand2,
                "h2"
            )
        )

    else:

        hand2 = np.zeros(
            (21, 3),
            dtype=np.float32
        )

        temp = hand_features(
            hand2,
            "h2"
        )

        for key in temp:
            features[key] = 0.0


    # ------------------------------------------------------
    # İKİ EL ARASI FEATURE
    # ------------------------------------------------------

    if h1_present and h2_present:

        # El merkezleri
        center1 = np.mean(
            hand1,
            axis=0
        )

        center2 = np.mean(
            hand2,
            axis=0
        )

        features[
            "hands_center_distance"
        ] = distance(
            center1,
            center2
        )


        # Aynı parmak uçlarının iki el arasındaki mesafeleri
        for finger, tip in TIPS.items():

            features[
                f"hands_{finger}_tip_distance"
            ] = distance(
                hand1[tip],
                hand2[tip]
            )


        # Bir elin başparmağı - diğer elin işaret parmağı
        features[
            "h1_thumb_h2_index"
        ] = distance(
            hand1[4],
            hand2[8]
        )

        features[
            "h2_thumb_h1_index"
        ] = distance(
            hand2[4],
            hand1[8]
        )

    else:

        features[
            "hands_center_distance"
        ] = 0.0

        for finger in TIPS:

            features[
                f"hands_{finger}_tip_distance"
            ] = 0.0

        features[
            "h1_thumb_h2_index"
        ] = 0.0

        features[
            "h2_thumb_h1_index"
        ] = 0.0


    return features


# ==========================================================
# DATASET İŞLE
# ==========================================================

for split_name, filename in FILES.items():

    print()
    print(
        "İşleniyor:",
        filename
    )

    df = pd.read_csv(
        filename,
        encoding="utf-8-sig"
    )

    geometric_rows = []


    for index, row in df.iterrows():

        geometric_rows.append(
            create_features(
                row
            )
        )


    geo_df = pd.DataFrame(
        geometric_rows
    )


    # quality_score'u modele sokmayacağız
    base_df = df.drop(
        columns=["quality_score"],
        errors="ignore"
    )


    label = base_df["label"].copy()

    base_df = base_df.drop(
        columns=["label"]
    )


    final_df = pd.concat(
        [
            base_df.reset_index(
                drop=True
            ),
            geo_df.reset_index(
                drop=True
            )
        ],
        axis=1
    )


    final_df["label"] = (
        label.values
    )


    output = (
        f"{split_name}_geometric_v14.csv"
    )


    final_df.to_csv(
        output,
        index=False,
        encoding="utf-8-sig"
    )


    print(
        "Satır:",
        len(final_df)
    )

    print(
        "Feature:",
        len(final_df.columns) - 1
    )

    print(
        "Kaydedildi:",
        output
    )


print()
print("========================================")
print(" V14 FEATURE ENGINEERING TAMAMLANDI")
print("========================================")