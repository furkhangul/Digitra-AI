import cv2
import time
import csv
import numpy as np
import joblib
import mediapipe as mp

from pathlib import Path
from collections import Counter


# ==========================================================
# DOSYA YOLLARI
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

HAND_MODEL_PATH = str(
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

OUTPUT_CSV = str(
    BASE_DIR / "webcam_evaluation_results_v8.csv"
)


# ==========================================================
# AYARLAR
# ==========================================================

CAMERA_INDEX = 0

TEST_DURATION = 5.0

CONFIDENCE_THRESHOLD = 0.85

MIN_BBOX_AREA = 0.010
MIN_SPREAD = 0.08
MAX_BORDER_POINTS = 8

VALID_KEYS = list(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
)


# ==========================================================
# DOSYA KONTROL
# ==========================================================

required_files = [
    HAND_MODEL_PATH,
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
# MODELLER
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
print("Sınıflar:", list(encoder.classes_))
print("Feature:", scaler.n_features_in_)


# ==========================================================
# MEDIAPIPE
# ==========================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=HAND_MODEL_PATH
    ),

    running_mode=VisionRunningMode.VIDEO,

    num_hands=1,

    min_hand_detection_confidence=0.35,

    min_hand_presence_confidence=0.35,

    min_tracking_confidence=0.40
)


# ==========================================================
# LANDMARK SABİTLERİ
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
    "thumb": [1,2,3,4],
    "index": [5,6,7,8],
    "middle": [9,10,11,12],
    "ring": [13,14,15,16],
    "pinky": [17,18,19,20]
}


TIPS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20
}


# ==========================================================
# GEOMETRİ
# ==========================================================

def distance(a, b):
    return float(
        np.linalg.norm(a - b)
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
# NORMALIZATION
# ==========================================================

def normalize_landmarks(
    landmarks,
    handedness
):

    points = np.array(
        [
            [lm.x, lm.y, lm.z]
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

    # Sol eli kanonik sağ el yönüne çevir
    if handedness == "Left":
        points[:, 0] *= -1.0

    return points


# ==========================================================
# QUALITY CHECK
# ==========================================================

def check_landmark_quality(
    landmarks
):

    xy = np.array(
        [
            [lm.x, lm.y]
            for lm in landmarks
        ],
        dtype=np.float32
    )

    min_x = np.min(xy[:, 0])
    max_x = np.max(xy[:, 0])

    min_y = np.min(xy[:, 1])
    max_y = np.max(xy[:, 1])

    width = max_x - min_x
    height = max_y - min_y

    bbox_area = width * height

    spread = (
        np.std(xy[:, 0])
        +
        np.std(xy[:, 1])
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
        return False, "KUCUK"

    if spread < MIN_SPREAD:
        return False, "YIGILMIS"

    if border_points > MAX_BORDER_POINTS:
        return False, "SINIR"

    return True, "OK"


# ==========================================================
# GEOMETRIC FEATURES
# ==========================================================

def create_geometric_features(points):

    features = []

    # Başparmak açıları
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

    # Diğer parmak açıları
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

    # Wrist -> tip
    wrist = points[0]

    for finger, tip in TIPS.items():

        features.append(
            distance(
                wrist,
                points[tip]
            )
        )

    # Tip -> tip
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

            features.append(
                distance(
                    points[TIPS[f1]],
                    points[TIPS[f2]]
                )
            )

    # Thumb -> other tips
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

    # Palm geometry
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

    # Finger lengths
    sequences = {
        "thumb": [0,1,2,3,4],
        "index": [5,6,7,8],
        "middle": [9,10,11,12],
        "ring": [13,14,15,16],
        "pinky": [17,18,19,20]
    }

    lengths = {}

    for finger, sequence in sequences.items():

        length = 0.0

        for i in range(
            len(sequence) - 1
        ):

            length += distance(
                points[sequence[i]],
                points[sequence[i + 1]]
            )

        lengths[finger] = length

        features.append(
            length
        )

    # Length ratios
    middle_length = lengths["middle"]

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

    # Extension
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

    # Orientation
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
# LANDMARK ÇİZ
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
            (255,255,255),
            2
        )

    for point in points:

        cv2.circle(
            frame,
            point,
            4,
            (0,255,0),
            -1
        )


# ==========================================================
# CSV HAZIRLA
# ==========================================================

if not Path(OUTPUT_CSV).exists():

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "target",
            "total_frames",
            "hand_detected",
            "landmark_ok",
            "landmark_fail",
            "correct",
            "wrong",
            "low_confidence",
            "accuracy_percent",
            "avg_confidence",
            "most_wrong_prediction"
        ])


# ==========================================================
# KAMERA
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


print()
print("========================================")
print(" DIGITRA WEBCAM EVALUATION V8")
print("========================================")
print()
print("A-Z: test başlat")
print("ESC: çıkış")
print()
print(
    f"Her test {TEST_DURATION:.0f} saniye."
)


start_time = time.time()


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


        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )


        cv2.rectangle(
            frame,
            (20,20),
            (670,160),
            (0,0,0),
            -1
        )


        cv2.putText(
            frame,
            "DIGITRA V8 EVALUATION",
            (40,60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2
        )


        cv2.putText(
            frame,
            "Bir harfe bas -> 5 saniye testi baslat",
            (40,105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255,255,255),
            2
        )


        cv2.putText(
            frame,
            "ESC -> cikis",
            (40,140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255,255,255),
            2
        )


        if len(
            result.hand_landmarks
        ) > 0:

            draw_hand(
                frame,
                result.hand_landmarks[0]
            )


        cv2.imshow(
            "Digitra Evaluation V8",
            frame
        )


        key = cv2.waitKey(
            1
        ) & 0xFF


        if key == 27:
            break


        if key == 255:
            continue


        pressed = chr(
            key
        ).upper()


        if pressed not in VALID_KEYS:
            continue


        target = pressed


        print()
        print(
            f"{target} testi başlıyor..."
        )


        # ==================================================
        # 3-2-1 COUNTDOWN
        # ==================================================

        for countdown in [
            3,
            2,
            1
        ]:

            countdown_start = (
                time.time()
            )


            while (
                time.time()
                -
                countdown_start
                <
                1.0
            ):


                success, countdown_frame = (
                    cap.read()
                )


                if not success:
                    break


                cv2.putText(
                    countdown_frame,
                    f"{target} HAZIRLAN: {countdown}",
                    (150,200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0,255,255),
                    4
                )


                cv2.imshow(
                    "Digitra Evaluation V8",
                    countdown_frame
                )


                cv2.waitKey(1)


        # ==================================================
        # TEST COUNTERS
        # ==================================================

        total_frames = 0
        hand_detected = 0
        landmark_ok_count = 0
        landmark_fail = 0

        correct = 0
        wrong = 0
        low_confidence = 0

        confidence_values = []

        wrong_predictions = []


        test_start = time.time()


        while (
            time.time()
            -
            test_start
            <
            TEST_DURATION
        ):


            success, test_frame = (
                cap.read()
            )


            if not success:
                break


            total_frames += 1


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
                test_frame,
                cv2.COLOR_BGR2RGB
            )


            mp_image = mp.Image(
                image_format=
                    mp.ImageFormat.SRGB,
                data=rgb
            )


            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )


            current_prediction = "EL YOK"
            current_confidence = 0.0
            current_quality = "-"


            if len(
                result.hand_landmarks
            ) > 0:


                hand_detected += 1

                landmarks = (
                    result.hand_landmarks[0]
                )


                draw_hand(
                    test_frame,
                    landmarks
                )


                handedness = "Right"

                if len(
                    result.handedness
                ) > 0:

                    handedness = (
                        result
                        .handedness[0][0]
                        .category_name
                    )


                quality_ok, current_quality = (
                    check_landmark_quality(
                        landmarks
                    )
                )


                if quality_ok:


                    landmark_ok_count += 1


                    features = (
                        create_model_features(
                            landmarks,
                            handedness
                        )
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


                    current_prediction = (
                        encoder.classes_[
                            best_index
                        ]
                    )


                    current_confidence = float(
                        probabilities[
                            best_index
                        ]
                    )


                    confidence_values.append(
                        current_confidence
                    )


                    if (
                        current_confidence
                        <
                        CONFIDENCE_THRESHOLD
                    ):

                        low_confidence += 1

                    else:

                        if (
                            current_prediction
                            ==
                            target
                        ):

                            correct += 1

                        else:

                            wrong += 1

                            wrong_predictions.append(
                                current_prediction
                            )


                else:

                    landmark_fail += 1


            elapsed = (
                time.time()
                -
                test_start
            )

            remaining = max(
                0,
                TEST_DURATION
                -
                elapsed
            )


            cv2.rectangle(
                test_frame,
                (20,20),
                (650,220),
                (0,0,0),
                -1
            )


            cv2.putText(
                test_frame,
                f"TARGET: {target}",
                (40,65),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,255),
                2
            )


            cv2.putText(
                test_frame,
                f"Tahmin: {current_prediction}",
                (40,110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0,255,0),
                2
            )


            cv2.putText(
                test_frame,
                f"Confidence: %{current_confidence*100:.1f}",
                (40,150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255,255,255),
                2
            )


            cv2.putText(
                test_frame,
                f"Landmark: {current_quality}",
                (40,185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255,255,255),
                2
            )


            cv2.putText(
                test_frame,
                f"Kalan: {remaining:.1f}s",
                (400,65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255,255,255),
                2
            )


            cv2.imshow(
                "Digitra Evaluation V8",
                test_frame
            )


            cv2.waitKey(1)


        # ==================================================
        # TEST SONUCU
        # ==================================================

        valid_predictions = (
            correct
            +
            wrong
        )


        accuracy = (
            correct
            /
            valid_predictions
            *
            100
            if valid_predictions > 0
            else 0
        )


        avg_confidence = (
            np.mean(
                confidence_values
            )
            *
            100
            if confidence_values
            else 0
        )


        if wrong_predictions:

            most_wrong = (
                Counter(
                    wrong_predictions
                )
                .most_common(1)[0][0]
            )

        else:

            most_wrong = "-"


        print()
        print(
            f"===== {target} SONUÇ ====="
        )

        print(
            "Toplam frame      :",
            total_frames
        )

        print(
            "El bulundu        :",
            hand_detected
        )

        print(
            "Landmark OK       :",
            landmark_ok_count
        )

        print(
            "Landmark fail     :",
            landmark_fail
        )

        print(
            "Doğru             :",
            correct
        )

        print(
            "Yanlış            :",
            wrong
        )

        print(
            "Low confidence    :",
            low_confidence
        )

        print(
            f"Accuracy          : %{accuracy:.2f}"
        )

        print(
            f"Ort. confidence   : %{avg_confidence:.2f}"
        )

        print(
            "En çok yanlış     :",
            most_wrong
        )


        # ==================================================
        # CSV
        # ==================================================

        with open(
            OUTPUT_CSV,
            "a",
            newline="",
            encoding="utf-8-sig"
        ) as file:


            writer = csv.writer(
                file
            )


            writer.writerow([
                target,
                total_frames,
                hand_detected,
                landmark_ok_count,
                landmark_fail,
                correct,
                wrong,
                low_confidence,
                round(
                    accuracy,
                    2
                ),
                round(
                    avg_confidence,
                    2
                ),
                most_wrong
            ])


# ==========================================================
# KAPAT
# ==========================================================

cap.release()

cv2.destroyAllWindows()

print()
print(
    "Sonuç dosyası:"
)

print(
    OUTPUT_CSV
)