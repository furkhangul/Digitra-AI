import os
import cv2
import numpy as np
import mediapipe as mp


# ==========================================================
# AYARLAR
# ==========================================================

DATASET_DIR = r"C:\Users\Furkan\Desktop\Digitra\digitra-landmark v2\archive\asl_alphabet_train\asl_alphabet_train\A"

MODEL_PATH = r"C:\Users\Furkan\Desktop\Digitra\digitra-landmark v2\hand_landmarker.task"

OUTPUT_FILE = "A_landmark_retry_check.jpg"

MAX_SAMPLES = 12


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
# CONNECTIONS
# ==========================================================

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]


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

    x2 = upscale(
        image,
        2
    )

    x3 = upscale(
        image,
        3
    )

    x2_clahe = clahe_bgr(
        x2
    )

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
# DETECTION
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
# DRAW
# ==========================================================

def draw_landmarks(image, result, method_name, filename):

    output = image.copy()

    h, w = output.shape[:2]

    landmarks = result.hand_landmarks[0]

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
            output,
            points[start],
            points[end],
            (255, 255, 255),
            2
        )

    for i, point in enumerate(points):

        cv2.circle(
            output,
            point,
            4,
            (0, 255, 0),
            -1
        )

        cv2.putText(
            output,
            str(i),
            point,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 0, 255),
            1
        )

    output = cv2.resize(
        output,
        (250, 250)
    )

    cv2.putText(
        output,
        filename,
        (5, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1
    )

    cv2.putText(
        output,
        method_name,
        (5, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (0, 255, 255),
        1
    )

    return output


# ==========================================================
# DOSYALAR
# ==========================================================

files = sorted([
    f for f in os.listdir(DATASET_DIR)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
])

total = len(files)


# ==========================================================
# İSTATİSTİK
# ==========================================================

stats = {
    "original": 0,
    "2x": 0,
    "3x": 0,
    "2x_clahe": 0,
    "2x_gamma": 0,
    "2x_clahe_sharp": 0
}

detected = 0
failed = 0

samples = []


# ==========================================================
# BAŞLA
# ==========================================================

with HandLandmarker.create_from_options(
    options
) as landmarker:

    for index, filename in enumerate(
        files,
        start=1
    ):

        path = os.path.join(
            DATASET_DIR,
            filename
        )

        image = cv2.imread(
            path
        )

        if image is None:

            failed += 1
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

                detected += 1

                stats[
                    method_name
                ] += 1

                found = True


                # Sadece retry ile kurtulanlardan örnek göster
                if (
                    method_name != "original"
                    and
                    len(samples) < MAX_SAMPLES
                ):

                    samples.append(
                        draw_landmarks(
                            variant,
                            result,
                            method_name,
                            filename
                        )
                    )


                break


        if not found:

            failed += 1


        if index % 250 == 0:

            print(
                f"{index}/{total} işlendi..."
            )


# ==========================================================
# COLLAGE
# ==========================================================

while len(samples) < MAX_SAMPLES:

    samples.append(
        np.zeros(
            (250, 250, 3),
            dtype=np.uint8
        )
    )


row1 = np.hstack(
    samples[0:4]
)

row2 = np.hstack(
    samples[4:8]
)

row3 = np.hstack(
    samples[8:12]
)

collage = np.vstack([
    row1,
    row2,
    row3
])

cv2.imwrite(
    OUTPUT_FILE,
    collage
)


# ==========================================================
# SONUÇ
# ==========================================================

success_rate = (
    detected / total * 100
    if total > 0
    else 0
)


print()
print("========================================")
print(" A LANDMARK RETRY TEST V2")
print("========================================")
print()

print(
    "Toplam A görüntüsü :",
    total
)

print(
    "Landmark bulundu   :",
    detected
)

print(
    "Bulunamadı         :",
    failed
)

print(
    f"Başarı oranı       : %{success_rate:.2f}"
)

print()

print(
    "Hangi yöntem kaç görüntü kurtardı:"
)

for name, count in stats.items():

    print(
        f"{name:18} -> {count}"
    )

print()
print(
    "Retry kontrol görseli:"
)

print(
    OUTPUT_FILE
)
