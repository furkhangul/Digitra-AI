import cv2
import time
import numpy as np
import joblib
import mediapipe as mp

from pathlib import Path
from collections import deque, Counter


# ==========================================================
# DOSYA YOLLARI
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = str(
    BASE_DIR / "hand_landmarker.task"
)

CLASSIFIER_FILE = str(
    BASE_DIR / "digitra_asl_mlp_v5.pkl"
)

SCALER_FILE = str(
    BASE_DIR / "digitra_asl_scaler_v5.pkl"
)

ENCODER_FILE = str(
    BASE_DIR / "digitra_asl_encoder_v5.pkl"
)


# ==========================================================
# AYARLAR
# ==========================================================

CAMERA_INDEX = 0

CONFIDENCE_THRESHOLD = 0.85

HISTORY_LENGTH = 10
MIN_STABLE_COUNT = 7

# Landmark kalite filtresi
MIN_BBOX_AREA = 0.010
MIN_SPREAD = 0.08
MAX_BORDER_POINTS = 8


# ==========================================================
# DOSYA KONTROLÜ
# ==========================================================

required_files = [
    MODEL_PATH,
    CLASSIFIER_FILE,
    SCALER_FILE,
    ENCODER_FILE
]

for file_path in required_files:

    if not Path(file_path).exists():

        raise FileNotFoundError(
            f"Gerekli dosya bulunamadı:\n{file_path}"
        )


# ==========================================================
# MODEL YÜKLE
# ==========================================================

model = joblib.load(
    CLASSIFIER_FILE
)

scaler = joblib.load(
    SCALER_FILE
)

encoder = joblib.load(
    ENCODER_FILE
)


print()
print("Model yüklendi.")

print(
    "Sınıflar:",
    list(encoder.classes_)
)

print(
    "Feature:",
    scaler.n_features_in_
)


# ==========================================================
# MEDIAPIPE VIDEO MODE
# ==========================================================

BaseOptions = mp.tasks.BaseOptions

HandLandmarker = (
    mp.tasks.vision.HandLandmarker
)

HandLandmarkerOptions = (
    mp.tasks.vision.HandLandmarkerOptions
)

VisionRunningMode = (
    mp.tasks.vision.RunningMode
)


options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=
        VisionRunningMode.VIDEO,

    num_hands=1,

    min_hand_detection_confidence=0.35,

    min_hand_presence_confidence=0.35,

    min_tracking_confidence=0.40
)


# ==========================================================
# LANDMARK CONNECTIONS
# ==========================================================

HAND_CONNECTIONS = [

    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17)
]


FINGERS = {

    "thumb": [
        1,
        2,
        3,
        4
    ],

    "index": [
        5,
        6,
        7,
        8
    ],

    "middle": [
        9,
        10,
        11,
        12
    ],

    "ring": [
        13,
        14,
        15,
        16
    ],

    "pinky": [
        17,
        18,
        19,
        20
    ]
}


TIPS = {

    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20

}


# ==========================================================
# SOL EL -> KANONİK SAĞ EL
# ==========================================================

def canonicalize_left_hand(
    points,
    handedness
):

    points = points.copy()

    if handedness == "Left":

        points[:, 0] *= -1.0

    return points


# ==========================================================
# NORMALIZATION
# ==========================================================

def normalize_landmarks(
    landmarks,
    handedness
):

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


    # Wrist merkezleme
    wrist = points[0].copy()

    points = (
        points
        -
        wrist
    )


    # Scale normalize
    distances = np.linalg.norm(

        points,

        axis=1

    )


    scale = np.max(
        distances
    )


    if scale > 0:

        points = (
            points
            /
            scale
        )


    # Sol eli kanonik hale getir
    points = canonicalize_left_hand(

        points,

        handedness

    )


    return points


# ==========================================================
# LANDMARK QUALITY CHECK
# ==========================================================

def check_landmark_quality(
    landmarks
):

    xy = np.array(

        [
            [
                lm.x,
                lm.y
            ]

            for lm in landmarks
        ],

        dtype=np.float32

    )


    min_x = np.min(
        xy[:, 0]
    )

    max_x = np.max(
        xy[:, 0]
    )

    min_y = np.min(
        xy[:, 1]
    )

    max_y = np.max(
        xy[:, 1]
    )


    width = (
        max_x
        -
        min_x
    )

    height = (
        max_y
        -
        min_y
    )


    bbox_area = (
        width
        *
        height
    )


    spread = (

        np.std(
            xy[:, 0]
        )

        +

        np.std(
            xy[:, 1]
        )

    )


    border_points = np.sum(

        (
            (xy[:, 0] < 0.01)

            |

            (xy[:, 0] > 0.99)

            |

            (xy[:, 1] < 0.01)

            |

            (xy[:, 1] > 0.99)
        )

    )


    if bbox_area < MIN_BBOX_AREA:

        return (
            False,
            "KUCUK LANDMARK"
        )


    if spread < MIN_SPREAD:

        return (
            False,
            "LANDMARK YIGILMIS"
        )


    if border_points > MAX_BORDER_POINTS:

        return (
            False,
            "SINIRA TASMIS"
        )


    return (
        True,
        "OK"
    )


# ==========================================================
# GEOMETRIC HELPERS
# ==========================================================

def distance(
    a,
    b
):

    return float(

        np.linalg.norm(
            a - b
        )

    )


def angle(
    a,
    b,
    c
):

    ba = (
        a
        -
        b
    )

    bc = (
        c
        -
        b
    )


    norm_ba = np.linalg.norm(
        ba
    )

    norm_bc = np.linalg.norm(
        bc
    )


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

        -1.0,

        1.0

    )


    return float(

        np.arccos(
            cos_value
        )

        /

        np.pi

    )


# ==========================================================
# GEOMETRIC FEATURES
# ==========================================================

def create_geometric_features(
    points
):

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

                    points[
                        TIPS[f1]
                    ],

                    points[
                        TIPS[f2]
                    ]

                )

            )


    # ======================================================
    # 4. THUMB -> DIGER TIPS
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

                points[
                    TIPS[finger]
                ]

            )

        )


    # ======================================================
    # 5. PALM
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

            palm_width
            /
            palm_height

        )

    else:

        features.append(
            0.0
        )


    # ======================================================
    # 6. FINGER LENGTH
    # ======================================================

    sequences = {

        "thumb": [
            0,
            1,
            2,
            3,
            4
        ],

        "index": [
            5,
            6,
            7,
            8
        ],

        "middle": [
            9,
            10,
            11,
            12
        ],

        "ring": [
            13,
            14,
            15,
            16
        ],

        "pinky": [
            17,
            18,
            19,
            20
        ]
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

    middle_length = (
        lengths["middle"]
    )


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

        tip = TIPS[
            finger
        ]


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
    # 9. ORIENTATION
    # ======================================================

    palm_vector = (

        points[9]

        -

        points[0]

    )


    features.extend(

        [

            float(
                palm_vector[0]
            ),

            float(
                palm_vector[1]
            ),

            float(
                palm_vector[2]
            )

        ]

    )


    return features


# ==========================================================
# FULL FEATURE
# ==========================================================

def create_model_features(
    landmarks,
    handedness
):

    points = normalize_landmarks(

        landmarks,

        handedness

    )


    raw = (

        points
        .flatten()
        .tolist()

    )


    geo = create_geometric_features(
        points
    )


    features = (

        raw

        +

        geo

    )


    return np.array(

        features,

        dtype=np.float32

    )


# ==========================================================
# DRAW HAND
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


start_time = time.time()


print()
print("========================================")
print(" DIGITRA WEBCAM V7")
print("========================================")
print()

print(
    "Q = çıkış"
)

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


        # ==================================================
        # VIDEO MODE TIMESTAMP
        # ==================================================

        timestamp_ms = int(

            (
                time.time()
                -
                start_time
            )

            *

            1000

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


        result = (
            landmarker.detect_for_video(

                mp_image,

                timestamp_ms

            )
        )


        display_label = (
            "EL YOK"
        )

        confidence = 0.0

        handedness_text = "-"

        quality_text = "-"


        # ==================================================
        # EL BULUNDU
        # ==================================================

        if len(
            result.hand_landmarks
        ) > 0:


            landmarks = (
                result.hand_landmarks[0]
            )


            # ==================================================
            # HANDEDNESS
            # ==================================================

            handedness = "Right"


            if len(
                result.handedness
            ) > 0:


                handedness = (

                    result
                    .handedness[0][0]
                    .category_name

                )


            handedness_text = (
                handedness
            )


            # Landmark çiz
            draw_hand(

                frame,

                landmarks

            )


            # ==================================================
            # LANDMARK QUALITY
            # ==================================================

            (
                quality_ok,
                quality_text
            ) = check_landmark_quality(
                landmarks
            )


            if quality_ok:


                try:


                    features = create_model_features(

                        landmarks,

                        handedness

                    )


                    if len(
                        features
                    ) != 116:

                        raise ValueError(

                            "Feature sayısı yanlış: "
                            f"{len(features)}"

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


                    best_index = int(
                        np.argmax(
                            probabilities
                        )
                    )


                    confidence = float(

                        probabilities[
                            best_index
                        ]

                    )


                    predicted_label = (

                        encoder
                        .classes_[
                            best_index
                        ]

                    )


                    # ==================================================
                    # CONFIDENCE FILTER
                    # ==================================================

                    if confidence >= CONFIDENCE_THRESHOLD:

                        history.append(
                            predicted_label
                        )

                    else:

                        history.append(
                            "UNKNOWN"
                        )


                    # ==================================================
                    # TEMPORAL SMOOTHING
                    # ==================================================

                    if len(
                        history
                    ) > 0:


                        counts = Counter(
                            history
                        )


                        (
                            most_common_label,
                            stable_count
                        ) = (
                            counts.most_common(
                                1
                            )[0]
                        )


                        if (

                            stable_count
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

                display_label = (
                    "LANDMARK HATALI"
                )


        else:

            history.clear()


        # ==================================================
        # UI
        # ==================================================

        cv2.rectangle(

            frame,

            (20, 20),

            (600, 220),

            (0, 0, 0),

            -1

        )


        cv2.putText(

            frame,

            "DIGITRA V7",

            (40, 60),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            f"Tahmin: {display_label}",

            (40, 105),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0, 255, 0),

            2

        )


        cv2.putText(

            frame,

            f"Confidence: %{confidence * 100:.1f}",

            (40, 140),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            f"El: {handedness_text}",

            (40, 175),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            f"Landmark: {quality_text}",

            (40, 205),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.60,

            (255, 255, 255),

            2

        )


        cv2.imshow(

            "Digitra Webcam V7",

            frame

        )


        key = (

            cv2.waitKey(
                1
            )

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
print(
    "Digitra V7 kapatıldı."
)