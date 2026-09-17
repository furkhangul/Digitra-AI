import cv2
import numpy as np
import joblib
import mediapipe as mp
from collections import deque, Counter


# ==========================================================
# DOSYALAR
# ==========================================================

MODEL_PATH = r"C:\Users\Furkan\Desktop\Digitra\digitra-landmark v2\hand_landmarker.task"

CLASSIFIER_FILE = "digitra_asl_mlp_v5.pkl"
SCALER_FILE = "digitra_asl_scaler_v5.pkl"
ENCODER_FILE = "digitra_asl_encoder_v5.pkl"


# ==========================================================
# AYARLAR
# ==========================================================

CAMERA_INDEX = 0

CONFIDENCE_THRESHOLD = 0.85

# Son kaç frame'e bakacağız
HISTORY_LENGTH = 10

# Aynı harf en az kaç frame tekrar etmeli
MIN_STABLE_COUNT = 7


# ==========================================================
# MODEL YÜKLE
# ==========================================================

model = joblib.load(CLASSIFIER_FILE)
scaler = joblib.load(SCALER_FILE)
encoder = joblib.load(ENCODER_FILE)


print()
print("Model yüklendi.")
print("Sınıflar:", list(encoder.classes_))
print("Model feature:", scaler.n_features_in_)


# ==========================================================
# MEDIAPIPE
# ==========================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=VisionRunningMode.IMAGE,

    num_hands=1,

    min_hand_detection_confidence=0.20,

    min_hand_presence_confidence=0.20
)


# ==========================================================
# LANDMARK CONNECTIONS
# ==========================================================

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
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
# NORMALIZATION
# ==========================================================

def normalize_landmarks(landmarks):

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
        points = points / scale

    return points


# ==========================================================
# GEOMETRIC FUNCTIONS
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
        np.arccos(cos_value)
        /
        np.pi
    )


# ==========================================================
# GEOMETRIC FEATURES
# ==========================================================

def create_geometric_features(points):

    features = []


    # ======================================================
    # 1. EKLEM AÇILARI
    # ======================================================

    features.append(
        angle(
            points[1],
            points[2],
            points[3]
        )
    )

    features.append(
        angle(
            points[2],
            points[3],
            points[4]
        )
    )


    for finger, ids in FINGERS.items():

        if finger == "thumb":
            continue

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


    # ======================================================
    # 2. WRIST -> TIP
    # ======================================================

    wrist = points[0]

    for finger, tip in TIPS.items():

        features.append(
            distance(
                wrist,
                points[tip]
            )
        )


    # ======================================================
    # 3. TIP -> TIP
    # ======================================================

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


    # ======================================================
    # 4. THUMB -> DİĞER PARMAKLAR
    # ======================================================

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


    # ======================================================
    # 5. PALM GEOMETRY
    # ======================================================

    palm_width = distance(
        points[5],
        points[17]
    )

    palm_height = distance(
        points[0],
        points[9]
    )

    features.append(
        palm_width
    )

    features.append(
        palm_height
    )

    if palm_height > 0:

        features.append(
            palm_width / palm_height
        )

    else:

        features.append(
            0.0
        )


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


    lengths = {}


    for finger, sequence in sequences.items():

        length = 0.0

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

        lengths[
            finger
        ] = length

        features.append(
            length
        )


    # ======================================================
    # 7. LENGTH RATIOS
    # ======================================================

    middle_length = lengths[
        "middle"
    ]


    for finger in [
        "thumb",
        "index",
        "ring",
        "pinky"
    ]:

        if middle_length > 0:

            features.append(
                lengths[finger]
                /
                middle_length
            )

        else:

            features.append(
                0.0
            )


    # ======================================================
    # 8. EXTENSION
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

            features.append(
                tip_distance
                /
                mcp_distance
            )

        else:

            features.append(
                0.0
            )


    base_distance = distance(
        points[0],
        points[2]
    )

    thumb_distance = distance(
        points[0],
        points[4]
    )

    if base_distance > 0:

        features.append(
            thumb_distance
            /
            base_distance
        )

    else:

        features.append(
            0.0
        )


    # ======================================================
    # 9. PALM ORIENTATION
    # ======================================================

    palm_vector = (
        points[9]
        -
        points[0]
    )

    features.extend([
        float(palm_vector[0]),
        float(palm_vector[1]),
        float(palm_vector[2])
    ])


    return features


# ==========================================================
# FULL 116 FEATURES
# ==========================================================

def create_model_features(landmarks):

    points = normalize_landmarks(
        landmarks
    )

    raw_features = (
        points
        .flatten()
        .tolist()
    )

    geo_features = (
        create_geometric_features(
            points
        )
    )

    features = (
        raw_features
        +
        geo_features
    )

    return np.array(
        features,
        dtype=np.float32
    )


# ==========================================================
# LANDMARK DRAW
# ==========================================================

def draw_hand(
    frame,
    landmarks
):

    h, w = frame.shape[:2]

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
            5,
            (0, 255, 0),
            -1
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

    raise RuntimeError(
        "Kamera açılamadı."
    )


history = deque(
    maxlen=HISTORY_LENGTH
)


print()
print("========================================")
print(" DIGITRA WEBCAM TEST V6")
print("========================================")
print()

print("Çıkış: Q")
print()


# ==========================================================
# LOOP
# ==========================================================

with HandLandmarker.create_from_options(
    options
) as landmarker:


    while True:


        success, frame = cap.read()


        if not success:
            break


        # --------------------------------------------------
        # ŞİMDİLİK MIRROR YAPMIYORUZ
        #
        # Eğitim görüntülerimiz flip edilmediği için
        # coordinate distribution'ı değiştirmiyoruz.
        # --------------------------------------------------


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


        display_label = "EL YOK"
        confidence = 0.0


        # ==================================================
        # EL BULUNDU
        # ==================================================

        if len(
            result.hand_landmarks
        ) > 0:


            landmarks = (
                result.hand_landmarks[0]
            )


            draw_hand(
                frame,
                landmarks
            )


            try:


                features = (
                    create_model_features(
                        landmarks
                    )
                )


                if len(features) != 116:

                    raise ValueError(
                        f"Feature sayısı {len(features)}"
                    )


                X = features.reshape(
                    1,
                    -1
                )


                X_scaled = scaler.transform(
                    X
                )


                probabilities = (
                    model.predict_proba(
                        X_scaled
                    )[0]
                )


                best_index = np.argmax(
                    probabilities
                )


                confidence = float(
                    probabilities[
                        best_index
                    ]
                )


                predicted_label = (
                    encoder.classes_[
                        best_index
                    ]
                )


                # ==========================================
                # CONFIDENCE FILTER
                # ==========================================

                if (
                    confidence
                    >=
                    CONFIDENCE_THRESHOLD
                ):

                    history.append(
                        predicted_label
                    )

                else:

                    history.append(
                        "UNKNOWN"
                    )


                # ==========================================
                # TEMPORAL STABILIZATION
                # ==========================================

                counts = Counter(
                    history
                )


                most_common_label, count = (
                    counts.most_common(1)[0]
                )


                if (
                    count
                    >=
                    MIN_STABLE_COUNT
                    and
                    most_common_label
                    !=
                    "UNKNOWN"
                ):

                    display_label = (
                        most_common_label
                    )

                else:

                    display_label = (
                        "BELIRSIZ"
                    )


            except Exception as e:


                print(
                    "Prediction error:",
                    e
                )

                display_label = (
                    "HATA"
                )


        else:

            history.clear()


        # ==================================================
        # UI
        # ==================================================

        cv2.rectangle(
            frame,
            (20, 20),
            (520, 170),
            (0, 0, 0),
            -1
        )


        cv2.putText(
            frame,
            "DIGITRA V6",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )


        cv2.putText(
            frame,
            f"Tahmin: {display_label}",
            (40, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"Confidence: %{confidence*100:.1f}",
            (40, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


        cv2.imshow(
            "Digitra Webcam V6",
            frame
        )


        key = (
            cv2.waitKey(1)
            &
            0xFF
        )


        if key == ord("q"):

            break


# ==========================================================
# CLEANUP
# ==========================================================

cap.release()

cv2.destroyAllWindows()

print()
print("Kamera kapatıldı.")