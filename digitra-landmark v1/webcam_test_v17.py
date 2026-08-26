import cv2
import numpy as np
import joblib
import mediapipe as mp


# ==========================================================
# DOSYALAR
# ==========================================================

HAND_MODEL = r"C:\Users\Furkan\Desktop\digitra-landmark\hand_landmarker.task"

V9_MODEL = "digitra_mlp_v9.pkl"
V9_SCALER = "digitra_scaler_v9.pkl"
V9_ENCODER = "digitra_label_encoder_v9.pkl"

V15_MODEL = "digitra_mlp_v15.pkl"
V15_SCALER = "digitra_scaler_v15.pkl"
V15_ENCODER = "digitra_label_encoder_v15.pkl"


# ==========================================================
# AYARLAR
# ==========================================================

CONFIDENCE_THRESHOLD = 0.60

CAMERA_INDEX = 0


# ==========================================================
# ESKİ LABEL -> TÜRKÇE
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
# MODEL YÜKLE
# ==========================================================

model9 = joblib.load(V9_MODEL)
scaler9 = joblib.load(V9_SCALER)
encoder9 = joblib.load(V9_ENCODER)

model15 = joblib.load(V15_MODEL)
scaler15 = joblib.load(V15_SCALER)
encoder15 = joblib.load(V15_ENCODER)


# ==========================================================
# MEDIAPIPE
# ==========================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=HAND_MODEL
    ),

    running_mode=VisionRunningMode.IMAGE,

    num_hands=2,

    min_hand_detection_confidence=0.20,

    min_hand_presence_confidence=0.20
)


# ==========================================================
# LANDMARK CONNECTIONS
# ==========================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),

    (0, 5), (5, 6), (6, 7), (7, 8),

    (5, 9), (9, 10), (10, 11), (11, 12),

    (9, 13), (13, 14), (14, 15), (15, 16),

    (13, 17), (17, 18), (18, 19), (19, 20),

    (0, 17)
]


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
# NORMALIZE
# ==========================================================

def normalize_hand(landmarks):

    points = np.array(
        [
            [
                lm.x,
                lm.y,
                lm.z
            ]
            for lm in landmarks
        ],
        dtype=np.float32
    )

    wrist = points[0].copy()

    points = points - wrist


    distances = np.linalg.norm(
        points,
        axis=1
    )

    scale = np.max(
        distances
    )

    if scale > 0:

        points = (
            points / scale
        )


    return points


# ==========================================================
# GEOMETRI
# ==========================================================

def distance(a, b):

    return float(
        np.linalg.norm(
            a - b
        )
    )


def angle(a, b, c):

    ba = a - b
    bc = c - b


    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)


    if (
        norm_ba == 0
        or
        norm_bc == 0
    ):

        return 0.0


    cos_value = (

        np.dot(
            ba,
            bc
        )

        /

        (
            norm_ba
            *
            norm_bc
        )

    )


    cos_value = np.clip(
        cos_value,
        -1,
        1
    )


    return float(
        np.arccos(
            cos_value
        )
        /
        np.pi
    )


# ==========================================================
# HAND GEOMETRIC FEATURES
# ==========================================================

def hand_geo_features(points):

    features = []


    # ------------------------------------------------------
    # PARMAK AÇILARI
    # ------------------------------------------------------

    for finger, ids in FINGERS.items():

        if finger == "thumb":

            p1, p2, p3, p4 = ids

            features.append(
                angle(
                    points[p1],
                    points[p2],
                    points[p3]
                )
            )

            features.append(
                angle(
                    points[p2],
                    points[p3],
                    points[p4]
                )
            )


        else:

            mcp, pip, dip, tip = ids


            features.append(
                angle(
                    points[0],
                    points[mcp],
                    points[pip]
                )
            )

            features.append(
                angle(
                    points[mcp],
                    points[pip],
                    points[dip]
                )
            )

            features.append(
                angle(
                    points[pip],
                    points[dip],
                    points[tip]
                )
            )


    # ------------------------------------------------------
    # WRIST -> TIP
    # ------------------------------------------------------

    wrist = points[0]


    for finger, tip in TIPS.items():

        features.append(
            distance(
                wrist,
                points[tip]
            )
        )


    # ------------------------------------------------------
    # TIP -> TIP
    # ------------------------------------------------------

    names = list(
        TIPS.keys()
    )


    for i in range(
        len(names)
    ):

        for j in range(
            i + 1,
            len(names)
        ):

            f1 = names[i]
            f2 = names[j]

            features.append(

                distance(
                    points[TIPS[f1]],
                    points[TIPS[f2]]
                )

            )


    # ------------------------------------------------------
    # THUMB -> OTHER TIPS
    # ------------------------------------------------------

    thumb_tip = points[4]


    for finger in [
        "index",
        "middle",
        "ring",
        "pinky"
    ]:

        features.append(
            distance(
                thumb_tip,
                points[TIPS[finger]]
            )
        )


    # ------------------------------------------------------
    # PALM
    # ------------------------------------------------------

    features.append(
        distance(
            points[5],
            points[17]
        )
    )


    features.append(
        distance(
            points[0],
            points[9]
        )
    )


    # ------------------------------------------------------
    # PARMAK UZUNLUKLARI
    # ------------------------------------------------------

    for finger, ids in FINGERS.items():

        length = 0.0


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
                points[
                    sequence[i]
                ],
                points[
                    sequence[i + 1]
                ]
            )


        features.append(
            length
        )


    return features


# ==========================================================
# RAW FEATURE OLUŞTUR
# ==========================================================

def create_raw_features(
    detected_hands
):


    hand1 = np.zeros(
        (21, 3),
        dtype=np.float32
    )


    hand2 = np.zeros(
        (21, 3),
        dtype=np.float32
    )


    h1_present = 0
    h2_present = 0


    wrist1_original = None
    wrist2_original = None


    # ------------------------------------------------------
    # HAND 1
    # ------------------------------------------------------

    if len(
        detected_hands
    ) >= 1:

        landmarks = detected_hands[
            0
        ]["landmarks"]

        hand1 = normalize_hand(
            landmarks
        )

        h1_present = 1

        wrist1_original = np.array([
            landmarks[0].x,
            landmarks[0].y
        ])


    # ------------------------------------------------------
    # HAND 2
    # ------------------------------------------------------

    if len(
        detected_hands
    ) >= 2:

        landmarks = detected_hands[
            1
        ]["landmarks"]

        hand2 = normalize_hand(
            landmarks
        )

        h2_present = 1

        wrist2_original = np.array([
            landmarks[0].x,
            landmarks[0].y
        ])


    wrist_dx = 0.0
    wrist_dy = 0.0
    wrist_distance = 0.0


    if (
        wrist1_original is not None
        and
        wrist2_original is not None
    ):

        diff = (
            wrist2_original
            -
            wrist1_original
        )

        wrist_dx = float(
            diff[0]
        )

        wrist_dy = float(
            diff[1]
        )

        wrist_distance = float(
            np.linalg.norm(
                diff
            )
        )


    raw_features = (

        hand1.flatten().tolist()

        +

        hand2.flatten().tolist()

        +

        [
            h1_present,
            h2_present,
            wrist_dx,
            wrist_dy,
            wrist_distance
        ]

    )


    return (
        raw_features,
        hand1,
        hand2,
        h1_present,
        h2_present
    )


# ==========================================================
# GEOMETRIC FEATURE OLUŞTUR
# ==========================================================

def create_geo_features(
    raw_features,
    hand1,
    hand2,
    h1_present,
    h2_present
):


    geo = []


    # ------------------------------------------------------
    # HAND 1
    # ------------------------------------------------------

    if h1_present:

        geo.extend(
            hand_geo_features(
                hand1
            )
        )

    else:

        geo.extend(
            hand_geo_features(
                np.zeros(
                    (21, 3),
                    dtype=np.float32
                )
            )
        )


    # ------------------------------------------------------
    # HAND 2
    # ------------------------------------------------------

    if h2_present:

        geo.extend(
            hand_geo_features(
                hand2
            )
        )

    else:

        geo.extend(
            hand_geo_features(
                np.zeros(
                    (21, 3),
                    dtype=np.float32
                )
            )
        )


    # ------------------------------------------------------
    # İKİ EL ARASI
    # ------------------------------------------------------

    if (
        h1_present
        and
        h2_present
    ):

        center1 = np.mean(
            hand1,
            axis=0
        )

        center2 = np.mean(
            hand2,
            axis=0
        )


        geo.append(
            distance(
                center1,
                center2
            )
        )


        for finger, tip in TIPS.items():

            geo.append(
                distance(
                    hand1[tip],
                    hand2[tip]
                )
            )


        geo.append(
            distance(
                hand1[4],
                hand2[8]
            )
        )


        geo.append(
            distance(
                hand2[4],
                hand1[8]
            )
        )


    else:

        # center distance
        geo.append(0.0)

        # 5 fingertip distance
        geo.extend(
            [0.0] * 5
        )

        # cross thumb/index
        geo.extend(
            [0.0] * 2
        )


    return (
        raw_features
        +
        geo
    )


# ==========================================================
# DRAW LANDMARKS
# ==========================================================

def draw_landmarks(
    frame,
    detected_hands
):

    h, w, _ = frame.shape


    for hand in detected_hands:

        landmarks = hand[
            "landmarks"
        ]

        points = []


        for lm in landmarks:

            x = int(
                lm.x * w
            )

            y = int(
                lm.y * h
            )

            points.append(
                (x, y)
            )


        for start, end in HAND_CONNECTIONS:

            cv2.line(
                frame,
                points[start],
                points[end],
                (255, 255, 255),
                2
            )


        for point in points:

            cv2.circle(
                frame,
                point,
                4,
                (0, 255, 0),
                -1
            )


# ==========================================================
# ENSEMBLE PREDICTION
# ==========================================================

def predict(
    raw_features,
    geo_features
):


    raw = np.array(
        raw_features,
        dtype=np.float32
    ).reshape(1, -1)


    geo = np.array(
        geo_features,
        dtype=np.float32
    ).reshape(1, -1)


    # ------------------------------------------------------
    # FEATURE SAYISI KONTROL
    # ------------------------------------------------------

    if raw.shape[1] != scaler9.n_features_in_:

        raise ValueError(
            f"V9 feature mismatch: "
            f"{raw.shape[1]} != "
            f"{scaler9.n_features_in_}"
        )


    if geo.shape[1] != scaler15.n_features_in_:

        raise ValueError(
            f"V15 feature mismatch: "
            f"{geo.shape[1]} != "
            f"{scaler15.n_features_in_}"
        )


    raw_scaled = scaler9.transform(
        raw
    )


    geo_scaled = scaler15.transform(
        geo
    )


    prob9_old = model9.predict_proba(
        raw_scaled
    )[0]


    prob15 = model15.predict_proba(
        geo_scaled
    )[0]


    # ------------------------------------------------------
    # V9 CLASS MAP
    # ------------------------------------------------------

    classes15 = encoder15.classes_


    prob9 = np.zeros(
        len(classes15),
        dtype=np.float32
    )


    for i, old_class in enumerate(
        encoder9.classes_
    ):

        tr_class = OLD_TO_TR.get(
            old_class,
            old_class
        )


        new_index = np.where(
            classes15 == tr_class
        )[0]


        if len(new_index) == 1:

            prob9[
                new_index[0]
            ] = prob9_old[i]


    # ------------------------------------------------------
    # ENSEMBLE
    # ------------------------------------------------------

    final_prob = (
        0.60 * prob9
        +
        0.40 * prob15
    )


    prediction_index = np.argmax(
        final_prob
    )


    predicted_label = classes15[
        prediction_index
    ]


    confidence = final_prob[
        prediction_index
    ]


    return (
        predicted_label,
        float(confidence)
    )


# ==========================================================
# CAMERA
# ==========================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX
)


cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


if not cap.isOpened():

    print(
        "Kamera açılamadı."
    )

    exit()


print()
print("========================================")
print(" DIGITRA WEBCAM TEST V17")
print("========================================")
print()

print(
    "Çıkmak için Q'ya bas."
)


# ==========================================================
# MEDIAPIPE
# ==========================================================

with HandLandmarker.create_from_options(
    options
) as landmarker:


    while True:


        success, frame = cap.read()


        if not success:

            break


        # Mirror görüntü
        frame = cv2.flip(
            frame,
            1
        )


        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(

            image_format=
            mp.ImageFormat.SRGB,

            data=rgb
        )


        result = landmarker.detect(
            mp_image
        )


        detected_hands = []


        # ==================================================
        # ELLER
        # ==================================================

        for landmarks in result.hand_landmarks:

            wrist_x = landmarks[
                0
            ].x


            detected_hands.append({

                "landmarks":
                    landmarks,

                "wrist_x":
                    wrist_x

            })


        # Görüntü sol -> sağ
        detected_hands.sort(
            key=lambda x:
                x["wrist_x"]
        )


        # ==================================================
        # PREDICT
        # ==================================================

        prediction = "EL YOK"
        confidence = 0.0


        if len(
            detected_hands
        ) > 0:


            (
                raw_features,
                hand1,
                hand2,
                h1_present,
                h2_present
            ) = create_raw_features(
                detected_hands
            )


            geo_features = create_geo_features(
                raw_features,
                hand1,
                hand2,
                h1_present,
                h2_present
            )


            try:

                (
                    predicted,
                    confidence
                ) = predict(
                    raw_features,
                    geo_features
                )


                if confidence >= CONFIDENCE_THRESHOLD:

                    prediction = predicted

                else:

                    prediction = "BELIRSIZ"


            except Exception as e:

                prediction = "FEATURE ERROR"

                print(
                    "Prediction error:",
                    e
                )


        # ==================================================
        # LANDMARK ÇİZ
        # ==================================================

        draw_landmarks(
            frame,
            detected_hands
        )


        # ==================================================
        # UI
        # ==================================================

        cv2.rectangle(
            frame,
            (20, 20),
            (520, 150),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            frame,
            "DIGITRA",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Tahmin: {prediction}",
            (40, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Confidence: %{confidence * 100:.1f}",
            (40, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"El sayisi: {len(detected_hands)}",
            (frame.shape[1] - 200, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


        cv2.imshow(
            "Digitra - V17",
            frame
        )


        key = cv2.waitKey(
            1
        ) & 0xFF


        if key == ord("q"):

            break


# ==========================================================
# KAPAT
# ==========================================================

cap.release()

cv2.destroyAllWindows()

print()
print("Digitra kapatıldı.")