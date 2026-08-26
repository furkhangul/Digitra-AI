import os
import cv2
import numpy as np


DATASET_DIR = r"C:\Users\Furkan\Desktop\digitra-landmark\archive (1)\tsl finger spelling\Images"

TARGET_LABELS = [
    "!",
    "+",
    ",",
    ";",
    "=",
    "_"
]

OUTPUT_FILE = "turkish_special_labels.jpg"

SAMPLES_PER_CLASS = 5
CELL_WIDTH = 220
CELL_HEIGHT = 220


def get_label(filename):
    if " (" not in filename:
        return None

    return filename.split(" (")[0].strip()


rows = []


for label in TARGET_LABELS:

    files = []

    for filename in os.listdir(DATASET_DIR):

        if not filename.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):
            continue

        current_label = get_label(filename)

        if current_label == label:
            files.append(filename)


    files.sort()
    files = files[:SAMPLES_PER_CLASS]

    cells = []


    for filename in files:

        path = os.path.join(
            DATASET_DIR,
            filename
        )

        image = cv2.imread(path)

        if image is None:
            continue


        # Oranı bozmadan küçült
        h, w = image.shape[:2]

        scale = min(
            (CELL_WIDTH - 20) / w,
            (CELL_HEIGHT - 45) / h
        )

        new_w = int(w * scale)
        new_h = int(h * scale)

        resized = cv2.resize(
            image,
            (new_w, new_h)
        )


        canvas = np.zeros(
            (
                CELL_HEIGHT,
                CELL_WIDTH,
                3
            ),
            dtype=np.uint8
        )


        x = (
            CELL_WIDTH - new_w
        ) // 2

        y = (
            CELL_HEIGHT - new_h
        ) // 2 + 15


        canvas[
            y:y + new_h,
            x:x + new_w
        ] = resized


        cv2.putText(
            canvas,
            filename,
            (5, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )


        cells.append(canvas)


    # Eksik varsa siyah kutu
    while len(cells) < SAMPLES_PER_CLASS:

        cells.append(
            np.zeros(
                (
                    CELL_HEIGHT,
                    CELL_WIDTH,
                    3
                ),
                dtype=np.uint8
            )
        )


    row = np.hstack(cells)


    # Sol tarafa label yazmak için alan
    label_area = np.zeros(
        (
            CELL_HEIGHT,
            100,
            3
        ),
        dtype=np.uint8
    )


    cv2.putText(
        label_area,
        label,
        (30, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (255, 255, 255),
        3
    )


    row = np.hstack([
        label_area,
        row
    ])

    rows.append(row)


final_image = np.vstack(rows)


cv2.imwrite(
    OUTPUT_FILE,
    final_image
)


print()
print("==============================")
print(" DIGITRA LABEL CHECK")
print("==============================")
print()

for label in TARGET_LABELS:

    count = 0

    for filename in os.listdir(DATASET_DIR):

        if get_label(filename) == label:
            count += 1

    print(
        f"{label} -> {count} görüntü"
    )


print()
print(
    "Görsel oluşturuldu:",
    OUTPUT_FILE
)