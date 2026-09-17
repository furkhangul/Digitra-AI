import os
import csv
import cv2
import time
import numpy as np
import mediapipe as mp


# ============================================================
# AYARLAR
# ============================================================

DATASET_DIR = r"C:\Users\Furkan\Desktop\digitra-landmark\archive (1)\tsl finger spelling\Images"

MODEL_PATH = "hand_landmarker.task"

OUTPUT_CSV = "turkish_landmarks.csv"

FAILED_FILE = "failed_images.txt"


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

    # İki eli de istiyoruz
    num_hands=2,

    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3
)


# ============================================================
# LABEL
# ============================================================

def get_label(filename):
    """
    A (1).png -> A
    B (15).png -> B
    ! (1).png -> !
    """

    if " (" not in filename:
        return None

    return filename.split(" (")[0].strip()


# ============================================================
# LANDMARK NORMALIZATION
# ============================================================

def normalize_hand(landmarks):
    """
    Tek eli bileğe göre normalize eder.

    Landmark 0 = wrist
    """

    points = np.array(
        [
            [lm.x, lm.y, lm.z]
            for lm in landmarks
        ],
        dtype=np.float32
    )

    # Bileği orijin yap
    wrist = points[0].copy()

    points = points - wrist

    # El büyüklüğünü normalize et
    distances = np.linalg.norm(points, axis=1)

    scale = np.max(distances)

    if scale > 0:
        points = points / scale

    return points.flatten().tolist()


# ============================================================
# CSV HEADER
# ============================================================

header = []

for hand in ["left", "right"]:

    for i in range(21):

        header.extend([
            f"{hand}_x{i}",
            f"{hand}_y{i}",
            f"{hand}_z{i}"
        ])

header.extend([
    "left_present",
    "right_present",
    "wrist_dx",
    "wrist_dy",
    "wrist_distance",
    "label"
])


# ============================================================
# DOSYALARI BUL
# ============================================================

files = [
    f for f in os.listdir(DATASET_DIR)
    if f.lower().endswith(
        (".png", ".jpg", ".jpeg")
    )
]

files.sort()

total = len(files)

print()
print("====================================")
print(" DIGITRA LANDMARK DATASET GENERATOR")
print("====================================")
print()
print("Toplam görüntü:", total)
print()


# ============================================================
# İSTATİSTİK
# ============================================================

two_hands = 0
one_hand = 0
zero_hands = 0
processed = 0

failed_images = []


# ============================================================
# CSV
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8-sig"
) as csv_file:

    writer = csv.writer(csv_file)

    writer.writerow(header)

    # ========================================================
    # MEDIAPIPE MODEL
    # ========================================================

    with HandLandmarker.create_from_options(
        options
    ) as landmarker:

        start_time = time.time()

        # ====================================================
        # TÜM GÖRÜNTÜLER
        # ====================================================

        for index, filename in enumerate(
            files,
            start=1
        ):

            path = os.path.join(
                DATASET_DIR,
                filename
            )

            label = get_label(filename)

            if label is None:
                continue

            # --------------------------------------------
            # OpenCV
            # --------------------------------------------

            image = cv2.imread(path)

            if image is None:

                failed_images.append(filename)

                zero_hands += 1

                continue

            # --------------------------------------------
            # RGB
            # --------------------------------------------

            rgb = cv2.cvtColor(
                image,
                cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            # --------------------------------------------
            # LANDMARK
            # --------------------------------------------

            result = landmarker.detect(mp_image)

            hand_count = len(
                result.hand_landmarks
            )

            # --------------------------------------------
            # EL BULUNAMADI
            # --------------------------------------------

            if hand_count == 0:

                zero_hands += 1

                failed_images.append(
                    filename
                )

                print(
                    f"[{index}/{total}] "
                    f"EL BULUNAMADI -> {filename}"
                )

                continue

            # --------------------------------------------
            # DEFAULT
            # --------------------------------------------

            left_features = [0.0] * 63
            right_features = [0.0] * 63

            left_present = 0
            right_present = 0

            left_wrist = None
            right_wrist = None

            # --------------------------------------------
            # HER ELİ İŞLE
            # --------------------------------------------

            for hand_index, landmarks in enumerate(
                result.hand_landmarks
            ):

                handedness = result.handedness[
                    hand_index
                ][0].category_name

                features = normalize_hand(
                    landmarks
                )

                wrist = landmarks[0]

                wrist_position = np.array(
                    [
                        wrist.x,
                        wrist.y
                    ]
                )

                if handedness == "Left":

                    left_features = features

                    left_present = 1

                    left_wrist = wrist_position

                elif handedness == "Right":

                    right_features = features

                    right_present = 1

                    right_wrist = wrist_position

            # --------------------------------------------
            # İKİ EL ARASINDAKİ İLİŞKİ
            # --------------------------------------------

            wrist_dx = 0.0
            wrist_dy = 0.0
            wrist_distance = 0.0

            if (
                left_wrist is not None
                and
                right_wrist is not None
            ):

                difference = (
                    right_wrist -
                    left_wrist
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

            # --------------------------------------------
            # CSV SATIRI
            # --------------------------------------------

            row = (
                left_features
                +
                right_features
                +
                [
                    left_present,
                    right_present,
                    wrist_dx,
                    wrist_dy,
                    wrist_distance,
                    label
                ]
            )

            writer.writerow(row)

            processed += 1

            # --------------------------------------------
            # İSTATİSTİK
            # --------------------------------------------

            if hand_count >= 2:

                two_hands += 1

            else:

                one_hand += 1

            # --------------------------------------------
            # TERMINAL
            # --------------------------------------------

            print(
                f"[{index}/{total}] "
                f"{label} | "
                f"El: {hand_count}"
            )


# ============================================================
# FAILED DOSYALAR
# ============================================================

with open(
    FAILED_FILE,
    "w",
    encoding="utf-8"
) as file:

    for filename in failed_images:

        file.write(
            filename + "\n"
        )


# ============================================================
# SONUÇ
# ============================================================

elapsed = time.time() - start_time


print()
print("====================================")
print("             SONUÇ")
print("====================================")
print()

print("Toplam görüntü :", total)
print("İşlenen        :", processed)
print("2 el bulunan   :", two_hands)
print("1 el bulunan   :", one_hand)
print("El bulunamayan :", zero_hands)

print()

print(
    f"Süre           : {elapsed:.2f} saniye"
)

print()

print(
    "CSV oluşturuldu:",
    OUTPUT_CSV
)

print(
    "Başarısızlar:",
    FAILED_FILE
)

print()
print("====================================")
