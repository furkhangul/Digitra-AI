import os
import cv2
import numpy as np
import mediapipe as mp

DATASET_DIR = r"C:\Users\Furkan\Desktop\Digitra\digitra-landmark v2\archive\asl_alphabet_train\asl_alphabet_train\A"
MODEL_PATH = r"C:\Users\Furkan\Desktop\Digitra\digitra-landmark v2\hand_landmarker.task"

OUTPUT_FILE = "A_landmark_check.jpg"

MAX_SAMPLES = 12

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
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3
)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

files = sorted([
    f for f in os.listdir(DATASET_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

total = len(files)
detected = 0
failed = 0
sample_images = []

with HandLandmarker.create_from_options(options) as landmarker:

    for index, filename in enumerate(files, start=1):

        path = os.path.join(DATASET_DIR, filename)

        image = cv2.imread(path)

        if image is None:
            failed += 1
            continue

        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect(mp_image)

        if len(result.hand_landmarks) == 0:
            failed += 1
            continue

        detected += 1

        # İlk 12 başarılı örneği görselleştir
        if len(sample_images) < MAX_SAMPLES:

            output = image.copy()

            h, w = output.shape[:2]

            landmarks = result.hand_landmarks[0]

            points = []

            for lm in landmarks:

                x = int(lm.x * w)
                y = int(lm.y * h)

                points.append((x, y))

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
                (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1
            )

            sample_images.append(output)


# Boş kare ekle
while len(sample_images) < MAX_SAMPLES:

    sample_images.append(
        np.zeros(
            (250, 250, 3),
            dtype=np.uint8
        )
    )


# 4 x 3 collage
row1 = np.hstack(sample_images[0:4])
row2 = np.hstack(sample_images[4:8])
row3 = np.hstack(sample_images[8:12])

collage = np.vstack([
    row1,
    row2,
    row3
])

cv2.imwrite(
    OUTPUT_FILE,
    collage
)


success_rate = (
    detected / total * 100
    if total > 0
    else 0
)


print()
print("====================================")
print(" A LANDMARK TEST")
print("====================================")
print()

print("Toplam A görüntüsü :", total)
print("Landmark bulundu   :", detected)
print("Bulunamadı         :", failed)
print(f"Başarı oranı       : %{success_rate:.2f}")

print()
print("Kontrol görseli:")
print(OUTPUT_FILE)