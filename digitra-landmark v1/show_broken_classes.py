import os
import cv2
import numpy as np

DATASET_DIR = r"C:\Users\Furkan\Desktop\digitra-landmark\archive (1)\tsl finger spelling\Images"

BROKEN_CLASSES = ["!", ",", ";", "_", "+", "="]

OUTPUT_DIR = "broken_class_samples"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for label in BROKEN_CLASSES:

    files = [
        f for f in os.listdir(DATASET_DIR)
        if f.startswith(label + " ") and f.lower().endswith(".png")
    ]

    files = files[:9]

    images = []

    for filename in files:
        path = os.path.join(DATASET_DIR, filename)

        img = cv2.imread(path)

        if img is None:
            continue

        img = cv2.resize(img, (250, 250))

        cv2.putText(
            img,
            filename,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        images.append(img)

    # 3x3 kolaj
    while len(images) < 9:
        images.append(
            np.zeros((250, 250, 3), dtype=np.uint8)
        )

    row1 = np.hstack(images[0:3])
    row2 = np.hstack(images[3:6])
    row3 = np.hstack(images[6:9])

    collage = np.vstack([row1, row2, row3])

    output_path = os.path.join(
        OUTPUT_DIR,
        f"class_{ord(label)}.jpg"
    )

    cv2.imwrite(output_path, collage)

    print(f"{label} -> {output_path}")

print("Bitti.")