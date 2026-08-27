import os
import cv2
import numpy as np
import mediapipe as mp


DATASET_DIR = r"C:\Users\Furkan\Desktop\digitra-landmark\archive (1)\tsl finger spelling\Images"
MODEL_PATH = r"C:\Users\Furkan\Desktop\digitra-landmark\hand_landmarker.task"

TARGET_LABEL = "A"
OUTPUT_FILE = "A_landmark_check.jpg"


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


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


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


def draw_landmarks(image, result):

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

            points.append(
                (x, y)
            )

        # bağlantıları çiz
        for start, end in HAND_CONNECTIONS:

            cv2.line(
                output,
                points[start],
                points[end],
                (255, 255, 255),
                2
            )

        # noktaları çiz
        for i, point in enumerate(points):

            cv2.circle(
                output,
                point,
                5,
                (0, 255, 0),
                -1
            )

            cv2.putText(
                output,
                str(i),
                point,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 255),
                1
            )

    return output


files = [
    f for f in os.listdir(DATASET_DIR)
    if f.startswith(TARGET_LABEL + " ")
    and f.lower().endswith(".png")
]

files.sort()

successful_images = []


with HandLandmarker.create_from_options(
    options
) as landmarker:

    for filename in files:

        path = os.path.join(
            DATASET_DIR,
            filename
        )

        image = cv2.imread(
            path
        )

        if image is None:
            continue

        result = detect(
            landmarker,
            image
        )

        if len(
            result.hand_landmarks
        ) == 0:

            continue

        drawn = draw_landmarks(
            image,
            result
        )

        drawn = cv2.resize(
            drawn,
            (300, 300)
        )

        cv2.putText(
            drawn,
            filename,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        successful_images.append(
            drawn
        )

        if len(successful_images) == 9:
            break


# 9 adet çıkmadıysa boş kare ekle
while len(successful_images) < 9:

    successful_images.append(
        np.zeros(
            (300, 300, 3),
            dtype=np.uint8
        )
    )


row1 = np.hstack(
    successful_images[0:3]
)

row2 = np.hstack(
    successful_images[3:6]
)

row3 = np.hstack(
    successful_images[6:9]
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

print()
print("Kontrol görseli oluşturuldu:")
print(OUTPUT_FILE)