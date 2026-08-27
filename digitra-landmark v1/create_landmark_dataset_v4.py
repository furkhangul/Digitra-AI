import os
import csv
import cv2
import time
import numpy as np
import mediapipe as mp
from collections import defaultdict


# ============================================================
# AYARLAR
# ============================================================

DATASET_DIR = r"C:\Users\Furkan\Desktop\digitra-landmark\archive (1)\tsl finger spelling\Images"
MODEL_PATH = r"C:\Users\Furkan\Desktop\digitra-landmark\hand_landmarker.task"

OUTPUT_CSV = "turkish_landmarks_v4.csv"
QUALITY_REPORT = "landmark_quality_report_v4.csv"

REVIEW_DIR = "quality_review"
REJECT_DIR = "quality_reject"

os.makedirs(REVIEW_DIR, exist_ok=True)
os.makedirs(REJECT_DIR, exist_ok=True)


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.20,
    min_hand_presence_confidence=0.20
)


# ============================================================
# BAĞLANTILAR
# ============================================================

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


# ============================================================
# LABEL
# ============================================================

def get_label(filename):

    if " (" not in filename:
        return None

    return filename.split(" (")[0].strip()


# ============================================================
# PREPROCESS
# ============================================================

def upscale(image, scale):

    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )


def apply_clahe(image):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    return cv2.cvtColor(
        cv2.merge((l, a, b)),
        cv2.COLOR_LAB2BGR
    )


def sharpen(image):

    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])

    return cv2.filter2D(
        image,
        -1,
        kernel
    )


def create_variants(image):

    x2 = upscale(image, 2)
    x3 = upscale(image, 3)

    clahe = apply_clahe(x2)
    sharp = sharpen(clahe)

    return [
        ("original", image),
        ("2x", x2),
        ("3x", x3),
        ("2x_clahe", clahe),
        ("2x_clahe_sharp", sharp)
    ]


# ============================================================
# DETECTION
# ============================================================

def detect(landmarker, image):

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    return landmarker.detect(mp_image)


def result_confidence(result):

    scores = []

    for handedness in result.handedness:

        if handedness:
            scores.append(
                handedness[0].score
            )

    if not scores:
        return 0.0

    return float(np.mean(scores))


def find_best_detection(
    landmarker,
    original_image
):

    best = None
    best_name = None
    best_variant = None

    best_hand_count = 0
    best_confidence = 0

    for name, variant in create_variants(
        original_image
    ):

        result = detect(
            landmarker,
            variant
        )

        hand_count = len(
            result.hand_landmarks
        )

        confidence = result_confidence(
            result
        )

        if (
            hand_count > best_hand_count
            or
            (
                hand_count == best_hand_count
                and confidence > best_confidence
            )
        ):

            best = result
            best_name = name
            best_variant = variant
            best_hand_count = hand_count
            best_confidence = confidence

        if hand_count == 2:
            break

    return (
        best,
        best_name,
        best_variant,
        best_hand_count,
        best_confidence
    )


# ============================================================
# QUALITY CHECK
# ============================================================

def check_hand_quality(landmarks):

    points = np.array([
        [lm.x, lm.y, lm.z]
        for lm in landmarks
    ], dtype=np.float32)

    xy = points[:, :2]

    min_x = np.min(xy[:, 0])
    max_x = np.max(xy[:, 0])

    min_y = np.min(xy[:, 1])
    max_y = np.max(xy[:, 1])

    bbox_width = max_x - min_x
    bbox_height = max_y - min_y

    # --------------------------------------------------------
    # El çok küçük/yığılmış mı?
    # --------------------------------------------------------

    bbox_area = (
        bbox_width *
        bbox_height
    )

    # --------------------------------------------------------
    # Noktalar görüntünün aşırı kenarında mı?
    # --------------------------------------------------------

    near_border = np.sum(
        (
            (xy[:, 0] < 0.015) |
            (xy[:, 0] > 0.985) |
            (xy[:, 1] < 0.015) |
            (xy[:, 1] > 0.985)
        )
    )

    # --------------------------------------------------------
    # Nokta varyansı
    # Landmarklar tek noktaya yığılmışsa düşük çıkar
    # --------------------------------------------------------

    spread_x = np.std(
        xy[:, 0]
    )

    spread_y = np.std(
        xy[:, 1]
    )

    spread = (
        spread_x +
        spread_y
    )

    # --------------------------------------------------------
    # Palm size
    # Bilek 0 -> orta parmak MCP 9
    # --------------------------------------------------------

    palm_size = np.linalg.norm(
        xy[9] - xy[0]
    )

    # --------------------------------------------------------
    # Bağlantı uzunlukları
    # --------------------------------------------------------

    bone_lengths = []

    for start, end in HAND_CONNECTIONS:

        length = np.linalg.norm(
            xy[start] -
            xy[end]
        )

        bone_lengths.append(
            length
        )

    bone_lengths = np.array(
        bone_lengths
    )

    tiny_bones = np.sum(
        bone_lengths < 0.003
    )

    # ========================================================
    # PUAN
    # ========================================================

    score = 100
    reasons = []

    # Çok küçük/yığılmış landmark
    if bbox_area < 0.008:

        score -= 45
        reasons.append(
            "landmark_yigilmis"
        )

    elif bbox_area < 0.015:

        score -= 20
        reasons.append(
            "landmark_alani_kucuk"
        )

    # Spread
    if spread < 0.07:

        score -= 35
        reasons.append(
            "dusuk_spread"
        )

    elif spread < 0.10:

        score -= 15
        reasons.append(
            "spread_supheli"
        )

    # Palm
    if palm_size < 0.04:

        score -= 30
        reasons.append(
            "palm_cok_kucuk"
        )

    # Border
    if near_border >= 8:

        score -= 35
        reasons.append(
            "cok_fazla_border"
        )

    elif near_border >= 4:

        score -= 15
        reasons.append(
            "border_supheli"
        )

    # Çok kısa bağlantılar
    if tiny_bones >= 5:

        score -= 30
        reasons.append(
            "eklemler_yigilmis"
        )

    elif tiny_bones >= 3:

        score -= 15
        reasons.append(
            "kisa_bone"
        )

    # ========================================================
    # KARAR
    # ========================================================

    if score >= 75:

        status = "ACCEPT"

    elif score >= 50:

        status = "REVIEW"

    else:

        status = "REJECT"

    return {
        "score": max(score, 0),
        "status": status,
        "bbox_area": float(bbox_area),
        "spread": float(spread),
        "palm_size": float(palm_size),
        "border_points": int(near_border),
        "tiny_bones": int(tiny_bones),
        "reasons": "|".join(reasons)
    }


# ============================================================
# NORMALIZE
# ============================================================

def normalize_hand(landmarks):

    points = np.array([
        [
            lm.x,
            lm.y,
            lm.z
        ]
        for lm in landmarks
    ], dtype=np.float32)

    wrist = points[0].copy()

    points -= wrist

    distances = np.linalg.norm(
        points,
        axis=1
    )

    scale = np.max(
        distances
    )

    if scale > 0:
        points /= scale

    return points.flatten().tolist()


# ============================================================
# LANDMARK ÇİZ
# ============================================================

def draw_result(image, result):

    output = image.copy()

    height, width, _ = output.shape

    for landmarks in result.hand_landmarks:

        points = []

        for landmark in landmarks:

            x = int(
                landmark.x * width
            )

            y = int(
                landmark.y * height
            )

            points.append((x, y))

        for start, end in HAND_CONNECTIONS:

            cv2.line(
                output,
                points[start],
                points[end],
                (255, 255, 255),
                2
            )

        for point in points:

            cv2.circle(
                output,
                point,
                4,
                (0, 255, 0),
                -1
            )

    return output


# ============================================================
# CSV HEADER
# ============================================================

header = []

for hand in ["hand1", "hand2"]:

    for i in range(21):

        header.extend([
            f"{hand}_x{i}",
            f"{hand}_y{i}",
            f"{hand}_z{i}"
        ])

header.extend([
    "hand1_present",
    "hand2_present",
    "wrist_dx",
    "wrist_dy",
    "wrist_distance",
    "quality_score",
    "label"
])


# ============================================================
# DOSYALAR
# ============================================================

files = [
    f for f in os.listdir(DATASET_DIR)
    if f.lower().endswith(
        (".png", ".jpg", ".jpeg")
    )
]

files.sort()

total = len(files)


# ============================================================
# İSTATİSTİK
# ============================================================

class_stats = defaultdict(
    lambda: {
        "total": 0,
        "detected": 0,
        "accept": 0,
        "review": 0,
        "reject": 0,
        "zero": 0
    }
)

overall = {
    "accept": 0,
    "review": 0,
    "reject": 0,
    "zero": 0
}


# ============================================================
# BAŞLA
# ============================================================

print()
print("============================================")
print(" DIGITRA LANDMARK QUALITY FILTER V4")
print("============================================")
print()

start_time = time.time()


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as output_file, open(
    QUALITY_REPORT,
    "w",
    newline="",
    encoding="utf-8-sig"
) as report_file:

    dataset_writer = csv.writer(
        output_file
    )

    dataset_writer.writerow(
        header
    )

    report_writer = csv.writer(
        report_file
    )

    report_writer.writerow([
        "filename",
        "label",
        "hands",
        "preprocess",
        "confidence",
        "quality_score",
        "status",
        "reasons"
    ])


    with HandLandmarker.create_from_options(
        options
    ) as landmarker:


        for index, filename in enumerate(
            files,
            start=1
        ):

            label = get_label(
                filename
            )

            if label is None:
                continue

            class_stats[label][
                "total"
            ] += 1

            path = os.path.join(
                DATASET_DIR,
                filename
            )

            image = cv2.imread(
                path
            )

            if image is None:
                continue


            (
                result,
                preprocess_name,
                variant_image,
                hand_count,
                confidence
            ) = find_best_detection(
                landmarker,
                image
            )


            # =================================================
            # 0 EL
            # =================================================

            if (
                result is None
                or
                hand_count == 0
            ):

                class_stats[label][
                    "zero"
                ] += 1

                overall[
                    "zero"
                ] += 1

                report_writer.writerow([
                    filename,
                    label,
                    0,
                    "",
                    0,
                    0,
                    "REJECT",
                    "el_bulunamadi"
                ])

                print(
                    f"[{index}/{total}] "
                    f"{label} | "
                    f"EL YOK"
                )

                continue


            class_stats[label][
                "detected"
            ] += 1


            # =================================================
            # HER ELİN KALİTESİ
            # =================================================

            quality_results = []

            for landmarks in result.hand_landmarks:

                quality_results.append(
                    check_hand_quality(
                        landmarks
                    )
                )


            # En kötü elin skorunu kullan
            quality_score = min(
                q["score"]
                for q in quality_results
            )

            statuses = [
                q["status"]
                for q in quality_results
            ]

            reasons = "|".join([
                q["reasons"]
                for q in quality_results
                if q["reasons"]
            ])


            if "REJECT" in statuses:

                final_status = "REJECT"

            elif "REVIEW" in statuses:

                final_status = "REVIEW"

            else:

                final_status = "ACCEPT"


            class_stats[label][
                final_status.lower()
            ] += 1

            overall[
                final_status.lower()
            ] += 1


            # =================================================
            # RAPOR
            # =================================================

            report_writer.writerow([
                filename,
                label,
                hand_count,
                preprocess_name,
                confidence,
                quality_score,
                final_status,
                reasons
            ])


            # =================================================
            # REVIEW / REJECT GÖRSEL
            # =================================================

            if final_status in [
                "REVIEW",
                "REJECT"
            ]:

                drawn = draw_result(
                    variant_image,
                    result
                )

                cv2.putText(
                    drawn,
                    f"{final_status} SCORE:{quality_score}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

                save_dir = (
                    REVIEW_DIR
                    if final_status == "REVIEW"
                    else REJECT_DIR
                )

                cv2.imwrite(
                    os.path.join(
                        save_dir,
                        filename
                    ),
                    drawn
                )


            # =================================================
            # SADECE ACCEPT OLANLARI CSV'YE EKLE
            # =================================================

            if final_status != "ACCEPT":

                print(
                    f"[{index}/{total}] "
                    f"{label} | "
                    f"{final_status} | "
                    f"{quality_score}"
                )

                continue


            detected_hands = []

            for landmarks in result.hand_landmarks:

                wrist = landmarks[0]

                detected_hands.append({
                    "landmarks":
                        landmarks,
                    "wrist_x":
                        float(wrist.x),
                    "wrist_y":
                        float(wrist.y)
                })


            detected_hands.sort(
                key=lambda h:
                    h["wrist_x"]
            )


            hand1_features = [0.0] * 63
            hand2_features = [0.0] * 63

            hand1_present = 0
            hand2_present = 0

            hand1_wrist = None
            hand2_wrist = None


            if len(detected_hands) >= 1:

                h = detected_hands[0]

                hand1_features = normalize_hand(
                    h["landmarks"]
                )

                hand1_present = 1

                hand1_wrist = np.array([
                    h["wrist_x"],
                    h["wrist_y"]
                ])


            if len(detected_hands) >= 2:

                h = detected_hands[1]

                hand2_features = normalize_hand(
                    h["landmarks"]
                )

                hand2_present = 1

                hand2_wrist = np.array([
                    h["wrist_x"],
                    h["wrist_y"]
                ])


            wrist_dx = 0
            wrist_dy = 0
            wrist_distance = 0


            if (
                hand1_wrist is not None
                and
                hand2_wrist is not None
            ):

                diff = (
                    hand2_wrist -
                    hand1_wrist
                )

                wrist_dx = float(
                    diff[0]
                )

                wrist_dy = float(
                    diff[1]
                )

                wrist_distance = float(
                    np.linalg.norm(diff)
                )


            dataset_writer.writerow(
                hand1_features
                +
                hand2_features
                +
                [
                    hand1_present,
                    hand2_present,
                    wrist_dx,
                    wrist_dy,
                    wrist_distance,
                    quality_score,
                    label
                ]
            )


            print(
                f"[{index}/{total}] "
                f"{label} | "
                f"ACCEPT | "
                f"{quality_score}"
            )


# ============================================================
# SONUÇ
# ============================================================

elapsed = (
    time.time() -
    start_time
)

print()
print("============================================")
print("               V4 SONUÇ")
print("============================================")

print(
    "ACCEPT :",
    overall["accept"]
)

print(
    "REVIEW :",
    overall["review"]
)

print(
    "REJECT :",
    overall["reject"]
)

print(
    "EL YOK :",
    overall["zero"]
)

print()
print(
    f"Süre: {elapsed:.2f} saniye"
)

print()
print("============================================")
print("          SINIF BAZLI SONUÇ")
print("============================================")

for label in sorted(
    class_stats.keys()
):

    s = class_stats[label]

    print(
        f"{label:3} "
        f"| Toplam:{s['total']:3} "
        f"| Accept:{s['accept']:3} "
        f"| Review:{s['review']:3} "
        f"| Reject:{s['reject']:3} "
        f"| ElYok:{s['zero']:3}"
    )