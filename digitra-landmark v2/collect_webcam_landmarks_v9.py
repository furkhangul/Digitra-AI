import cv2
import time
import csv
import numpy as np
import mediapipe as mp

from pathlib import Path


# ==========================================================
# DIGITRA WEBCAM LANDMARK COLLECTOR V9
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

HAND_MODEL_PATH = str(
    BASE_DIR / "hand_landmarker.task"
)

OUTPUT_FILE = str(
    BASE_DIR / "webcam_landmarks_v9.csv"
)


# ==========================================================
# AYARLAR
# ==========================================================

CAMERA_INDEX = 0

TARGET_CLASSES = [
    "G",
    "H",
    "K",
    "M",
    "P",
    "Q",
    "R",
    "S",
    "T"
]

# Her harften kaç temiz örnek toplanacak
TARGET_SAMPLES = 500

# Aynı pozu aşırı tekrar kaydetmemek için
SAVE_INTERVAL = 0.05

# Landmark kalite kontrolü
MIN_BBOX_AREA = 0.010
MIN_SPREAD = 0.08
MAX_BORDER_POINTS = 8


# ==========================================================
# MODEL DOSYASI KONTROL
# ==========================================================

if not Path(HAND_MODEL_PATH).exists():

    raise FileNotFoundError(
        f"hand_landmarker.task bulunamadı:\n"
        f"{HAND_MODEL_PATH}"
    )


# ==========================================================
# MEDIAPIPE
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
        model_asset_path=HAND_MODEL_PATH
    ),

    running_mode=
        VisionRunningMode.VIDEO,

    num_hands=1,

    min_hand_detection_confidence=0.30,

    min_hand_presence_confidence=0.30,

    min_tracking_confidence=0.35
)


# ==========================================================
# HAND CONNECTIONS
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


# ==========================================================
# CSV HEADER
# ==========================================================

def create_header():

    header = []

    for i in range(21):

        header.append(
            f"x{i}"
        )

        header.append(
            f"y{i}"
        )

        header.append(
            f"z{i}"
        )

    header.extend([
        "handedness",
        "label"
    ])

    return header


# ==========================================================
# CSV DOSYASINI HAZIRLA
# ==========================================================

if not Path(OUTPUT_FILE).exists():

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            create_header()
        )


# ==========================================================
# LANDMARK QUALITY
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

        return False, "EL COK KUCUK"


    if spread < MIN_SPREAD:

        return False, "LANDMARK YIGILMIS"


    if border_points > MAX_BORDER_POINTS:

        return False, "EL KADRAJ DISINDA"


    return True, "OK"


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


    for i, point in enumerate(
        points
    ):

        cv2.circle(

            frame,

            point,

            5,

            (0, 255, 0),

            -1
        )


        cv2.circle(

            frame,

            point,

            7,

            (0, 0, 0),

            1
        )


# ==========================================================
# CSV'YE KAYDET
# ==========================================================

def save_landmarks(
    landmarks,
    handedness,
    label
):

    row = []


    for lm in landmarks:

        row.extend([
            float(lm.x),
            float(lm.y),
            float(lm.z)
        ])


    row.extend([
        handedness,
        label
    ])


    with open(

        OUTPUT_FILE,

        "a",

        newline="",

        encoding="utf-8-sig"

    ) as file:

        writer = csv.writer(
            file
        )

        writer.writerow(
            row
        )


# ==========================================================
# MEVCUT ÖRNEK SAYILARINI OKU
# ==========================================================

def load_existing_counts():

    counts = {

        label: 0

        for label in TARGET_CLASSES

    }


    if not Path(
        OUTPUT_FILE
    ).exists():

        return counts


    try:

        with open(

            OUTPUT_FILE,

            "r",

            encoding="utf-8-sig"

        ) as file:

            reader = csv.DictReader(
                file
            )


            for row in reader:

                label = row.get(
                    "label"
                )


                if label in counts:

                    counts[label] += 1


    except Exception:

        pass


    return counts


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


# ==========================================================
# DURUM
# ==========================================================

counts = load_existing_counts()

current_label = None

recording = False

last_save_time = 0.0

program_start = time.time()


print()
print("========================================")
print(" DIGITRA LANDMARK COLLECTOR V9")
print("========================================")
print()

print(
    "Test edilecek harfler:"
)

print(
    TARGET_CLASSES
)

print()

print(
    "HARF TUŞU = sınıf seç"
)

print(
    "SPACE      = kayıt başlat/durdur"
)

print(
    "R          = R harfini seçer"
)

print(
    "ESC        = çıkış"
)

print()

print(
    f"Hedef: her sınıf için {TARGET_SAMPLES} örnek"
)

print()


# ==========================================================
# MEDIAPIPE LOOP
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
                program_start
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


        hand_found = False

        quality_ok = False

        quality_text = "EL YOK"

        handedness = "-"


        # ==================================================
        # EL
        # ==================================================

        if len(
            result.hand_landmarks
        ) > 0:


            hand_found = True

            landmarks = (
                result.hand_landmarks[0]
            )


            draw_hand(
                frame,
                landmarks
            )


            # ==============================================
            # HANDEDNESS
            # ==============================================

            if len(
                result.handedness
            ) > 0:


                handedness = (

                    result
                    .handedness[0][0]
                    .category_name

                )


            # ==============================================
            # QUALITY
            # ==============================================

            (
                quality_ok,
                quality_text

            ) = check_landmark_quality(
                landmarks
            )


            # ==============================================
            # KAYIT
            # ==============================================

            if (

                recording

                and

                current_label is not None

                and

                quality_ok

                and

                counts[
                    current_label
                ] < TARGET_SAMPLES

            ):


                current_time = (
                    time.time()
                )


                if (

                    current_time
                    -
                    last_save_time

                    >=

                    SAVE_INTERVAL

                ):


                    save_landmarks(

                        landmarks,

                        handedness,

                        current_label
                    )


                    counts[
                        current_label
                    ] += 1


                    last_save_time = (
                        current_time
                    )


                    if (

                        counts[
                            current_label
                        ]

                        >=

                        TARGET_SAMPLES

                    ):


                        recording = False


                        print(

                            f"{current_label} tamamlandı "
                            f"-> {counts[current_label]}"

                        )


        # ==================================================
        # UI
        # ==================================================

        overlay = frame.copy()


        cv2.rectangle(

            overlay,

            (20, 20),

            (720, 260),

            (0, 0, 0),

            -1
        )


        frame = cv2.addWeighted(

            overlay,
            0.75,

            frame,
            0.25,

            0
        )


        cv2.putText(

            frame,

            "DIGITRA DATA COLLECTOR V9",

            (40, 60),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.9,

            (255, 255, 255),

            2
        )


        selected_text = (

            current_label

            if current_label is not None

            else "-"

        )


        cv2.putText(

            frame,

            f"Harf: {selected_text}",

            (40, 105),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0, 255, 255),

            2
        )


        if current_label is not None:

            progress = (

                f"{counts[current_label]}"
                f"/"
                f"{TARGET_SAMPLES}"

            )

        else:

            progress = "-"


        cv2.putText(

            frame,

            f"Ornek: {progress}",

            (40, 145),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2
        )


        status = (

            "KAYIT"

            if recording

            else "BEKLIYOR"

        )


        cv2.putText(

            frame,

            f"Durum: {status}",

            (40, 185),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (
                (0, 255, 0)

                if recording

                else

                (255, 255, 255)
            ),

            2
        )


        cv2.putText(

            frame,

            f"Landmark: {quality_text}",

            (40, 225),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (
                (0, 255, 0)

                if quality_ok

                else

                (0, 0, 255)
            ),

            2
        )


        cv2.putText(

            frame,

            f"El: {handedness}",

            (430, 105),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2
        )


        # ==================================================
        # ALT BİLGİ
        # ==================================================

        cv2.putText(

            frame,

            "Harf sec -> SPACE -> eli hafif hareket ettir",

            (30, frame.shape[0] - 55),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2
        )


        cv2.putText(

            frame,

            "SPACE: Kayit   ESC: Cikis",

            (30, frame.shape[0] - 20),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.65,

            (255, 255, 255),

            2
        )


        cv2.imshow(

            "Digitra Collector V9",

            frame
        )


        # ==================================================
        # KEYBOARD
        # ==================================================

        key = (

            cv2.waitKey(1)

            &

            0xFF
        )


        if key == 27:

            break


        # SPACE
        if key == 32:


            if current_label is None:

                print(
                    "Önce bir harf seç."
                )


            elif (

                counts[
                    current_label
                ]

                >=

                TARGET_SAMPLES

            ):

                print(

                    current_label,

                    "zaten tamamlandı."

                )


            else:

                recording = (
                    not recording
                )


                print(

                    current_label,

                    "KAYIT"

                    if recording

                    else

                    "DURDU"

                )


            continue


        # ==================================================
        # HARF SEÇ
        # ==================================================

        if key != 255:


            try:

                pressed = chr(
                    key
                ).upper()


                if pressed in TARGET_CLASSES:


                    current_label = (
                        pressed
                    )


                    recording = False


                    print()

                    print(
                        "Seçildi:",
                        current_label
                    )

                    print(

                        "Mevcut:",

                        counts[
                            current_label
                        ],

                        "/",

                        TARGET_SAMPLES
                    )


            except ValueError:

                pass


# ==========================================================
# CLEANUP
# ==========================================================

cap.release()

cv2.destroyAllWindows()


# ==========================================================
# FINAL
# ==========================================================

print()
print("========================================")
print(" TOPLANAN VERİ")
print("========================================")
print()


total = 0


for label in TARGET_CLASSES:

    count = counts[
        label
    ]

    total += count

    print(
        f"{label} -> {count}"
    )


print()

print(
    "Toplam:",
    total
)

print()

print(
    "Dosya:"
)

print(
    OUTPUT_FILE
)