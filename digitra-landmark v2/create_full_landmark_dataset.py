import os
import csv
import cv2
import numpy as np
import mediapipe as mp
from collections import defaultdict


# ==========================================================
# AYARLAR
# ==========================================================

DATASET_DIR = r"C:\Users\Furkan\Desktop\Digitra\digitra-landmark v2\archive\asl_alphabet_train\asl_alphabet_train"

MODEL_PATH = r"C:\Users\Furkan\Desktop\Digitra\digitra-landmark v2\hand_landmarker.task"

OUTPUT_CSV = "asl_landmarks_full.csv"
FAILED_FILE = "failed_landmarks_full.txt"


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
# PREPROCESS
# ==========================================================

def upscale(image, scale):

    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )


def clahe_bgr(image):

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

    return cv2.cvtColor(
        merged,
        cv2.COLOR_LAB2BGR
    )


def gamma_correct(image, gamma=1.25):

    inv_gamma = 1.0 / gamma

    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255
        for i in np.arange(256)
    ]).astype("uint8")

    return cv2.LUT(
        image,
        table
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

    x2_clahe = clahe_bgr(x2)

    x2_gamma = gamma_correct(
        x2,
        gamma=1.25
    )

    x2_clahe_sharp = sharpen(
        x2_clahe
    )

    return [
        ("original", image),
        ("2x", x2),
        ("3x", x3),
        ("2x_clahe", x2_clahe),
        ("2x_gamma", x2_gamma),
        ("2x_clahe_sharp", x2_clahe_sharp)
    ]


# ==========================================================
# DETECT
# ==========================================================

def detect(landmarker, image):

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


# ==========================================================
# NORMALIZE
# ==========================================================

def normalize_landmarks(landmarks):

    points = np.array([
        [
            lm.x,
            lm.y,
            lm.z
        ]
        for lm in landmarks
    ], dtype=np.float32)

    # Wrist'i orijin yap
    wrist = points[0].copy()

    points = points - wrist

    # Ölçek normalize
    distances = np.linalg.norm(
        points,
        axis=1
    )

    scale = np.max(
        distances
    )

    if scale > 0:
        points = points / scale

    return points.flatten().tolist()


# ==========================================================
# CSV HEADER
# ==========================================================

header = []

for i in range(21):

    header.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])

header.extend([
    "method",
    "label"
])


# ==========================================================
# SINIFLAR
# ==========================================================

classes = sorted([
    d
    for d in os.listdir(DATASET_DIR)
    if os.path.isdir(
        os.path.join(DATASET_DIR, d)
    )
])


print()
print("========================================")
print(" FULL ASL LANDMARK DATASET")
print("========================================")
print()

print(
    "Toplam sınıf:",
    len(classes)
)

print()


# ==========================================================
# İSTATİSTİK
# ==========================================================

class_stats = defaultdict(
    lambda: {
        "total": 0,
        "success": 0,
        "failed": 0
    }
)

method_stats = defaultdict(int)

failed_files = []

global_total = 0
global_success = 0
global_failed = 0


# ==========================================================
# CSV
# ==========================================================

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

        # ==================================================
        # HER SINIF
        # ==================================================

        for class_index, class_name in enumerate(
            classes,
            start=1
        ):

            class_path = os.path.join(
                DATASET_DIR,
                class_name
            )

            files = sorted([
                f
                for f in os.listdir(class_path)
                if f.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                )
            ])

            print()
            print(
                f"[{class_index}/{len(classes)}] "
                f"Sınıf: {class_name}"
            )

            # ==============================================
            # HER GÖRÜNTÜ
            # ==============================================

            for index, filename in enumerate(
                files,
                start=1
            ):

                global_total += 1

                class_stats[
                    class_name
                ]["total"] += 1

                path = os.path.join(
                    class_path,
                    filename
                )

                image = cv2.imread(
                    path
                )

                if image is None:

                    global_failed += 1

                    class_stats[
                        class_name
                    ]["failed"] += 1

                    failed_files.append(
                        path
                    )

                    continue


                found = False


                for method_name, variant in create_variants(
                    image
                ):

                    result = detect(
                        landmarker,
                        variant
                    )


                    if len(
                        result.hand_landmarks
                    ) > 0:

                        landmarks = (
                            result.hand_landmarks[0]
                        )

                        features = normalize_landmarks(
                            landmarks
                        )

                        row = (
                            features
                            +
                            [
                                method_name,
                                class_name
                            ]
                        )

                        writer.writerow(
                            row
                        )

                        global_success += 1

                        class_stats[
                            class_name
                        ]["success"] += 1

                        method_stats[
                            method_name
                        ] += 1

                        found = True

                        break


                if not found:

                    global_failed += 1

                    class_stats[
                        class_name
                    ]["failed"] += 1

                    failed_files.append(
                        path
                    )


                if index % 250 == 0:

                    success = class_stats[
                        class_name
                    ]["success"]

                    print(
                        f"   {index}/{len(files)} "
                        f"| success: {success}"
                    )


            # ==============================================
            # SINIF SONUCU
            # ==============================================

            s = class_stats[
                class_name
            ]

            rate = (
                s["success"]
                /
                s["total"]
                *
                100
            )

            print(
                f"{class_name} bitti -> "
                f"{s['success']}/{s['total']} "
                f"(%{rate:.2f})"
            )


# ==========================================================
# FAILED DOSYALAR
# ==========================================================

with open(
    FAILED_FILE,
    "w",
    encoding="utf-8"
) as file:

    for path in failed_files:

        file.write(
            path + "\n"
        )


# ==========================================================
# FINAL RAPOR
# ==========================================================

print()
print("========================================")
print(" GENEL SONUÇ")
print("========================================")
print()

print(
    "Toplam görüntü :",
    global_total
)

print(
    "Başarılı       :",
    global_success
)

print(
    "Başarısız      :",
    global_failed
)

print(
    f"Başarı oranı   : "
    f"%{global_success/global_total*100:.2f}"
)

print()

print(
    "Yöntem dağılımı:"
)

for method, count in method_stats.items():

    print(
        f"{method:18} -> {count}"
    )


print()
print("========================================")
print(" SINIF BAZLI")
print("========================================")
print()


for class_name in classes:

    s = class_stats[
        class_name
    ]

    rate = (
        s["success"]
        /
        s["total"]
        *
        100
    )

    print(
        f"{class_name:8} "
        f"| {s['success']:4}/{s['total']:4} "
        f"| %{rate:.2f}"
    )


print()
print(
    "CSV oluşturuldu:",
    OUTPUT_CSV
)

print(
    "Başarısız listesi:",
    FAILED_FILE
)