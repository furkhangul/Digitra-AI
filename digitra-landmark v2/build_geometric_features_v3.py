import pandas as pd
import numpy as np


INPUT_CSV = "asl_landmarks_26letters.csv"
OUTPUT_CSV = "asl_landmarks_geometric_26letters.csv"


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


# ==========================================================
# YARDIMCI FONKSİYONLAR
# ==========================================================

def distance(a, b):
    return float(np.linalg.norm(a - b))


def angle(a, b, c):
    """
    a-b-c açısı.
    Merkez nokta = b.
    Sonuç 0-1 aralığına normalize edilir.
    """

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cos_value = np.dot(ba, bc) / (norm_ba * norm_bc)

    cos_value = np.clip(
        cos_value,
        -1.0,
        1.0
    )

    return float(
        np.arccos(cos_value) / np.pi
    )


# ==========================================================
# LANDMARK OKU
# ==========================================================

def get_points(row):

    points = []

    for i in range(21):

        points.append([
            float(row[f"x{i}"]),
            float(row[f"y{i}"]),
            float(row[f"z{i}"])
        ])

    return np.array(
        points,
        dtype=np.float32
    )


# ==========================================================
# GEOMETRİK FEATURE
# ==========================================================

def create_geometric_features(points):

    features = {}


    # ======================================================
    # 1. PARMAK EKLEM AÇILARI
    # ======================================================

    # Başparmak
    features["thumb_angle_mcp"] = angle(
        points[1],
        points[2],
        points[3]
    )

    features["thumb_angle_ip"] = angle(
        points[2],
        points[3],
        points[4]
    )


    # Diğer parmaklar
    for finger, ids in FINGERS.items():

        if finger == "thumb":
            continue

        mcp, pip, dip, tip = ids

        features[
            f"{finger}_angle_mcp"
        ] = angle(
            points[0],
            points[mcp],
            points[pip]
        )

        features[
            f"{finger}_angle_pip"
        ] = angle(
            points[mcp],
            points[pip],
            points[dip]
        )

        features[
            f"{finger}_angle_dip"
        ] = angle(
            points[pip],
            points[dip],
            points[tip]
        )


    # ======================================================
    # 2. WRIST -> PARMAK UCU MESAFESİ
    # ======================================================

    wrist = points[0]

    for finger, tip in TIPS.items():

        features[
            f"wrist_to_{finger}_tip"
        ] = distance(
            wrist,
            points[tip]
        )


    # ======================================================
    # 3. PARMAK UÇLARI ARASI TÜM MESAFELER
    # ======================================================

    names = list(
        TIPS.keys()
    )

    for i in range(len(names)):

        for j in range(
            i + 1,
            len(names)
        ):

            f1 = names[i]
            f2 = names[j]

            features[
                f"tip_{f1}_{f2}"
            ] = distance(
                points[TIPS[f1]],
                points[TIPS[f2]]
            )


    # ======================================================
    # 4. BAŞPARMAK -> DİĞER PARMAKLAR
    # ======================================================

    thumb_tip = points[4]

    for finger in [
        "index",
        "middle",
        "ring",
        "pinky"
    ]:

        features[
            f"thumb_to_{finger}"
        ] = distance(
            thumb_tip,
            points[TIPS[finger]]
        )


    # ======================================================
    # 5. AVUÇ GEOMETRİSİ
    # ======================================================

    features["palm_width"] = distance(
        points[5],
        points[17]
    )

    features["palm_height"] = distance(
        points[0],
        points[9]
    )

    # Avuç oranı
    if features["palm_height"] > 0:

        features["palm_ratio"] = (
            features["palm_width"]
            /
            features["palm_height"]
        )

    else:
        features["palm_ratio"] = 0.0


    # ======================================================
    # 6. PARMAK UZUNLUKLARI
    # ======================================================

    sequences = {
        "thumb": [0, 1, 2, 3, 4],
        "index": [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring": [13, 14, 15, 16],
        "pinky": [17, 18, 19, 20]
    }


    for finger, sequence in sequences.items():

        length = 0.0

        for i in range(
            len(sequence) - 1
        ):

            length += distance(
                points[sequence[i]],
                points[sequence[i + 1]]
            )

        features[
            f"{finger}_length"
        ] = length


    # ======================================================
    # 7. PARMAK UZUNLUK ORANLARI
    # ======================================================

    middle_length = features[
        "middle_length"
    ]

    if middle_length > 0:

        for finger in [
            "thumb",
            "index",
            "ring",
            "pinky"
        ]:

            features[
                f"{finger}_middle_ratio"
            ] = (
                features[
                    f"{finger}_length"
                ]
                /
                middle_length
            )

    else:

        for finger in [
            "thumb",
            "index",
            "ring",
            "pinky"
        ]:

            features[
                f"{finger}_middle_ratio"
            ] = 0.0


    # ======================================================
    # 8. PARMAK AÇIKLIK / EXTENSION
    # ======================================================
    #
    # Wrist'e olan TIP mesafesini
    # MCP mesafesiyle karşılaştırıyoruz.
    #
    # Büyük değer → parmak daha açık.
    # ======================================================

    finger_mcp = {
        "index": 5,
        "middle": 9,
        "ring": 13,
        "pinky": 17
    }


    for finger, mcp in finger_mcp.items():

        tip = TIPS[finger]

        tip_distance = distance(
            points[0],
            points[tip]
        )

        mcp_distance = distance(
            points[0],
            points[mcp]
        )

        if mcp_distance > 0:

            features[
                f"{finger}_extension"
            ] = (
                tip_distance
                /
                mcp_distance
            )

        else:

            features[
                f"{finger}_extension"
            ] = 0.0


    # Başparmak extension
    base_distance = distance(
        points[0],
        points[2]
    )

    tip_distance = distance(
        points[0],
        points[4]
    )

    if base_distance > 0:

        features[
            "thumb_extension"
        ] = (
            tip_distance
            /
            base_distance
        )

    else:

        features[
            "thumb_extension"
        ] = 0.0


    # ======================================================
    # 9. AVUÇ YÖN / ORIENTATION
    # ======================================================

    # wrist -> middle MCP
    palm_vector = (
        points[9]
        -
        points[0]
    )

    features[
        "palm_orientation_x"
    ] = float(
        palm_vector[0]
    )

    features[
        "palm_orientation_y"
    ] = float(
        palm_vector[1]
    )

    features[
        "palm_orientation_z"
    ] = float(
        palm_vector[2]
    )


    return features


# ==========================================================
# DATASET
# ==========================================================

print()
print("========================================")
print(" GEOMETRIC FEATURE BUILDER V3")
print("========================================")
print()


df = pd.read_csv(
    INPUT_CSV,
    encoding="utf-8-sig"
)


print(
    "Girdi satır:",
    len(df)
)

print(
    "Sınıf:",
    df["label"].nunique()
)


# ==========================================================
# TÜM SATIRLARI İŞLE
# ==========================================================

geometric_rows = []


for index, row in df.iterrows():

    points = get_points(
        row
    )

    features = create_geometric_features(
        points
    )

    geometric_rows.append(
        features
    )


    if (
        (index + 1) % 5000
        == 0
    ):

        print(
            f"{index + 1}/{len(df)} işlendi..."
        )


# ==========================================================
# GEOMETRIC DATAFRAME
# ==========================================================

geo_df = pd.DataFrame(
    geometric_rows
)


# ==========================================================
# METHOD MODELE VERİLMEYECEK
# ==========================================================

base_df = df.drop(
    columns=[
        "label",
        "method"
    ],
    errors="ignore"
)


# ==========================================================
# HAM LANDMARK + GEOMETRİ
# ==========================================================

final_df = pd.concat(
    [
        base_df.reset_index(drop=True),
        geo_df.reset_index(drop=True)
    ],
    axis=1
)


final_df["label"] = (
    df["label"].values
)


# ==========================================================
# NaN / INF KONTROL
# ==========================================================

final_df.replace(
    [
        np.inf,
        -np.inf
    ],
    0,
    inplace=True
)

final_df.fillna(
    0,
    inplace=True
)


# ==========================================================
# KAYDET
# ==========================================================

final_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


print()
print("========================================")
print(" SONUÇ")
print("========================================")
print()

print(
    "Toplam örnek:",
    len(final_df)
)

print(
    "Toplam sınıf:",
    final_df[
        "label"
    ].nunique()
)

print(
    "Ham landmark feature:",
    63
)

print(
    "Geometrik feature:",
    len(geo_df.columns)
)

print(
    "Toplam feature:",
    len(final_df.columns) - 1
)

print(
    "NaN:",
    final_df.isna().sum().sum()
)

print()
print(
    "Dosya:",
    OUTPUT_CSV
)