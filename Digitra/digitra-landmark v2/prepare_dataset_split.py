import os
import shutil
import random

SOURCE_DIR = r"C:\Users\Furkan\Desktop\Digitra\digitra-landmark v2\archive\asl_alphabet_train\asl_alphabet_train"
OUTPUT_DIR = r"C:\Users\Furkan\Desktop\Digitra\digitra-landmark v2\dataset_split"

TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

SEED = 42

random.seed(SEED)


for split in ["train", "val", "test"]:
    os.makedirs(
        os.path.join(OUTPUT_DIR, split),
        exist_ok=True
    )


classes = sorted([
    d for d in os.listdir(SOURCE_DIR)
    if os.path.isdir(
        os.path.join(SOURCE_DIR, d)
    )
])


print("Toplam sınıf:", len(classes))
print()


for class_name in classes:

    class_path = os.path.join(
        SOURCE_DIR,
        class_name
    )

    images = [
        f for f in os.listdir(class_path)
        if f.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    random.shuffle(images)

    total = len(images)

    train_end = int(
        total * TRAIN_RATIO
    )

    val_end = train_end + int(
        total * VAL_RATIO
    )

    train_files = images[:train_end]
    val_files = images[
        train_end:val_end
    ]
    test_files = images[
        val_end:
    ]


    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }


    for split_name, split_files in splits.items():

        destination = os.path.join(
            OUTPUT_DIR,
            split_name,
            class_name
        )

        os.makedirs(
            destination,
            exist_ok=True
        )

        for filename in split_files:

            source_file = os.path.join(
                class_path,
                filename
            )

            destination_file = os.path.join(
                destination,
                filename
            )

            shutil.copy2(
                source_file,
                destination_file
            )


    print(
        f"{class_name:8} | "
        f"Train: {len(train_files):4} | "
        f"Val: {len(val_files):3} | "
        f"Test: {len(test_files):3}"
    )


print()
print("Bitti.")