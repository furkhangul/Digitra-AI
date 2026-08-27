import os
import csv
import cv2
import time
import numpy as np
import mediapipe as mp


# ==========================================================
# AYARLAR
# ==========================================================

DATASET_DIR = r"C:\Users\Furkan\Desktop\digitra-landmark\archive (1)\tsl finger spelling\Images"

MODEL_PATH = r"C:\Users\Furkan\Desktop\digitra-landmark\hand_landmarker.task"

OUTPUT_CSV = "turkish_landmarks_v2.csv"
FAILED_FILE = "failed_images_v2.txt"

# Hangi preprocessing yönteminin kaç kere işe yaradığını görmek için
PREPROCESS_REPORT = "preprocess_report_v2.txt"


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

    num_hands=2,

    # V1'de 0.3 idi
    min_hand_detection_confidence=0.20,

    min_hand_presence_confidence=0.20
)


# ==========================================================
# LABEL
# ==========================================================

def get_label(filename):

    if " (" not in filename:
        return None

    return filename.split(" (")[0].strip()


# ==========================================================
# PREPROCESSING
# ==========================================================

def upscale(image, scale):

    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )


def apply_clahe(image):

    # BGR -> LAB
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

    merged = cv2.merge(
        (l, a, b)
    )

    result = cv2.cvtColor(
        merged,
        cv2.COLOR_LAB2BGR
    )

    return result


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

    variants = []

    # 1 - Orijinal
    variants.append(
        ("original", image)
    )

    # 2 - 2x büyüt
    img_2x = upscale(
        image,
        2
    )

    variants.append(
        ("2x", img_2x)
    )

    # 3 - 3x büyüt
    img_3x = upscale(
        image,
        3
    )

    variants.append(
        ("3x", img_3x)
    )

    # 4 - 2x + kontrast
    img_clahe = apply_clahe(
        img_2x
    )

    variants.append(
        ("2x_clahe", img_clahe)
    )

    # 5 - 2x + kontrast + keskinlik
    img_sharp = sharpen(
        img_clahe
    )

    variants.append(
        ("2x_clahe_sharp", img_sharp)
    )

    return variants


# ==========================================================
# MEDIAPIPE DETECTION
# ==========================================================

def detect_hands(landmarker, image):

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    return landmarker.detect(
        mp_image
    )


def detection_score(result):

    """
    Aynı sayıda el bulan iki preprocessing varsa
    daha yüksek güven skoruna sahip olanı seç.
    """

    scores = []

    for handedness in result.handedness:

        if len(handedness) > 0:

            scores.append(
                handedness[0].score
            )

    if len(scores) == 0:
        return 0.0

    return float(
        np.mean(scores)
    )


def find_best_detection(
    landmarker,
    image
):

    variants = create_variants(
        image
    )

    best_result = None
    best_name = None

    best_hand_count = 0
    best_score = 0.0

    for name, variant in variants:

        result = detect_hands(
            landmarker,
            variant
        )

        hand_count = len(
            result.hand_landmarks
        )

        score = detection_score(
            result
        )

        # Daha fazla el bulduysa direkt daha iyi
        if hand_count > best_hand_count:

            best_result = result
            best_name = name

            best_hand_count = hand_count
            best_score = score

        # El sayısı aynıysa güven skoruna bak
        elif (
            hand_count == best_hand_count
            and
            score > best_score
        ):

            best_result = result
            best_name = name

            best_score = score

        # 2 el bulduysak daha fazla aramaya gerek yok
        if hand_count == 2:
            break

    return (
        best_result,
        best_name,
        best_hand_count
    )


# ==========================================================
# LANDMARK NORMALIZATION
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

    # 0 = bilek
    wrist = points[0].copy()

    # Eli bileğe göre merkezle
    points = points - wrist

    # El büyüklüğü
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

    return points.flatten().tolist()


# ==========================================================
# CSV HEADER
# ==========================================================

header = []

for hand in [
    "hand1",
    "hand2"
]:

    for i in range(21):

        header.extend([
            f"{hand}_x{i}",
            f"{hand}_y{i}",
            f"{hand}_z{i}"
        ])


header.extend([

    "hand1_present",
    "hand2_present",

    # İki el arasındaki konum
    "wrist_dx",
    "wrist_dy",
    "wrist_distance",

    "label"
])


# ==========================================================
# DOSYALAR
# ==========================================================

files = [

    f for f in os.listdir(
        DATASET_DIR
    )

    if f.lower().endswith(
        (
            ".png",
            ".jpg",
            ".jpeg"
        )
    )

]

files.sort()

total = len(files)


print()
print("==========================================")
print(" DIGITRA LANDMARK DATASET GENERATOR V2")
print("==========================================")
print()

print(
    "Toplam görüntü:",
    total
)

print()


# ==========================================================
# İSTATİSTİKLER
# ==========================================================

processed = 0

two_hands = 0
one_hand = 0
zero_hands = 0

failed_images = []


preprocess_stats = {

    "original": 0,
    "2x": 0,
    "3x": 0,
    "2x_clahe": 0,
    "2x_clahe_sharp": 0

}


# ==========================================================
# CSV
# ==========================================================

start_time = time.time()


with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as csv_file:

    writer = csv.writer(
        csv_file
    )

    writer.writerow(
        header
    )


    with HandLandmarker.create_from_options(
        options
    ) as landmarker:


        for index, filename in enumerate(
            files,
            start=1
        ):


            # =================================================
            # LABEL
            # =================================================

            label = get_label(
                filename
            )

            if label is None:
                continue


            # =================================================
            # IMAGE
            # =================================================

            path = os.path.join(
                DATASET_DIR,
                filename
            )

            image = cv2.imread(
                path
            )


            if image is None:

                failed_images.append(
                    filename
                )

                zero_hands += 1

                continue


            # =================================================
            # EN İYİ DETECTION'I BUL
            # =================================================

            (
                result,
                used_preprocess,
                hand_count
            ) = find_best_detection(
                landmarker,
                image
            )


            # =================================================
            # EL BULUNAMADI
            # =================================================

            if (
                result is None
                or
                hand_count == 0
            ):

                zero_hands += 1

                failed_images.append(
                    filename
                )

                print(
                    f"[{index}/{total}] "
                    f"EL BULUNAMADI -> {filename}"
                )

                continue


            # =================================================
            # PREPROCESS SAYACI
            # =================================================

            if used_preprocess in preprocess_stats:

                preprocess_stats[
                    used_preprocess
                ] += 1


            # =================================================
            # ELLERİ TOPLA
            # =================================================

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


            # =================================================
            # SAĞ / SOL YERİNE X'E GÖRE SIRALA
            # =================================================
            #
            # Görüntünün sol tarafındaki el = hand1
            # Görüntünün sağ tarafındaki el = hand2
            #
            # Böylece handedness hatalarından etkilenmiyoruz.
            # =================================================

            detected_hands.sort(
                key=lambda hand:
                    hand["wrist_x"]
            )


            # =================================================
            # DEFAULT
            # =================================================

            hand1_features = [
                0.0
            ] * 63

            hand2_features = [
                0.0
            ] * 63


            hand1_present = 0
            hand2_present = 0


            hand1_wrist = None
            hand2_wrist = None


            # =================================================
            # HAND 1
            # =================================================

            if len(
                detected_hands
            ) >= 1:

                hand = detected_hands[
                    0
                ]

                hand1_features = normalize_hand(
                    hand["landmarks"]
                )

                hand1_present = 1

                hand1_wrist = np.array(
                    [
                        hand["wrist_x"],
                        hand["wrist_y"]
                    ],
                    dtype=np.float32
                )


            # =================================================
            # HAND 2
            # =================================================

            if len(
                detected_hands
            ) >= 2:

                hand = detected_hands[
                    1
                ]

                hand2_features = normalize_hand(
                    hand["landmarks"]
                )

                hand2_present = 1

                hand2_wrist = np.array(
                    [
                        hand["wrist_x"],
                        hand["wrist_y"]
                    ],
                    dtype=np.float32
                )


            # =================================================
            # İKİ EL ARASINDAKİ İLİŞKİ
            # =================================================

            wrist_dx = 0.0
            wrist_dy = 0.0
            wrist_distance = 0.0


            if (
                hand1_wrist is not None
                and
                hand2_wrist is not None
            ):

                difference = (
                    hand2_wrist
                    -
                    hand1_wrist
                )

                wrist_dx = float(
                    difference[0]
                )

                wrist_dy = float(
                    difference[1]
                )

                wrist_distance = float(
                    np.linalg.norm(
                        difference
                    )
                )


            # =================================================
            # CSV
            # =================================================

            row = (

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

                    label
                ]
            )


            writer.writerow(
                row
            )


            processed += 1


            # =================================================
            # İSTATİSTİK
            # =================================================

            if hand_count >= 2:

                two_hands += 1

            else:

                one_hand += 1


            # =================================================
            # TERMINAL
            # =================================================

            print(

                f"[{index}/{total}] "

                f"{label} | "

                f"El: {hand_count} | "

                f"{used_preprocess}"

            )


# ==========================================================
# FAILED IMAGES
# ==========================================================

with open(
    FAILED_FILE,
    "w",
    encoding="utf-8"
) as file:

    for filename in failed_images:

        file.write(
            filename + "\n"
        )


# ==========================================================
# PREPROCESS REPORT
# ==========================================================

with open(
    PREPROCESS_REPORT,
    "w",
    encoding="utf-8"
) as file:

    for name, count in preprocess_stats.items():

        file.write(
            f"{name}: {count}\n"
        )


# ==========================================================
# SONUÇ
# ==========================================================

elapsed = (
    time.time()
    -
    start_time
)


success_rate = (
    processed
    /
    total
    *
    100
)


failure_rate = (
    zero_hands
    /
    total
    *
    100
)


print()
print("==========================================")
print("                 SONUÇ V2")
print("==========================================")
print()

print(
    "Toplam görüntü :",
    total
)

print(
    "İşlenen        :",
    processed
)

print(
    "2 el bulunan   :",
    two_hands
)

print(
    "1 el bulunan   :",
    one_hand
)

print(
    "El bulunamayan :",
    zero_hands
)

print()

print(
    f"Başarı oranı   : %{success_rate:.2f}"
)

print(
    f"Başarısızlık   : %{failure_rate:.2f}"
)

print()

print(
    f"Süre           : {elapsed:.2f} saniye"
)

print()

print(
    "Preprocessing kullanımları:"
)

for name, count in preprocess_stats.items():

    print(
        f"{name:18}: {count}"
    )

print()

print(
    "CSV:",
    OUTPUT_CSV
)

print(
    "Başarısız:",
    FAILED_FILE
)

print(
    "Preprocess raporu:",
    PREPROCESS_REPORT
)

print()
print("==========================================")