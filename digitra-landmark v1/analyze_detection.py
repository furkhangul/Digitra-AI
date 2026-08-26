import os
import csv
from collections import defaultdict


CSV_FILE = "turkish_landmarks_v2.csv"
FAILED_FILE = "failed_images_v2.txt"


# -----------------------------------------
# Başarılı örnekleri oku
# -----------------------------------------

stats = defaultdict(lambda: {
    "success": 0,
    "one_hand": 0,
    "two_hands": 0,
    "failed": 0,
    "total": 0
})


with open(
    CSV_FILE,
    "r",
    encoding="utf-8-sig"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        label = row["label"]

        hand1 = int(
            float(row["hand1_present"])
        )

        hand2 = int(
            float(row["hand2_present"])
        )

        stats[label]["success"] += 1
        stats[label]["total"] += 1

        if hand1 == 1 and hand2 == 1:

            stats[label]["two_hands"] += 1

        else:

            stats[label]["one_hand"] += 1


# -----------------------------------------
# Başarısız örnekleri oku
# -----------------------------------------

with open(
    FAILED_FILE,
    "r",
    encoding="utf-8"
) as file:

    for line in file:

        filename = line.strip()

        if not filename:
            continue

        # A (1).png -> A
        if " (" in filename:

            label = filename.split(
                " ("
            )[0].strip()

            stats[label]["failed"] += 1
            stats[label]["total"] += 1


# -----------------------------------------
# Rapor
# -----------------------------------------

print()
print("==============================================================")
print("        DIGITRA - SINIF BAZLI LANDMARK ANALIZI")
print("==============================================================")
print()

print(
    f"{'Sınıf':<8}"
    f"{'Toplam':<10}"
    f"{'Başarılı':<12}"
    f"{'1 El':<10}"
    f"{'2 El':<10}"
    f"{'Başarısız':<12}"
    f"{'Başarı %':<10}"
)

print("-" * 72)


sorted_labels = sorted(
    stats.keys()
)


for label in sorted_labels:

    s = stats[label]

    total = s["total"]

    success_rate = (
        s["success"] /
        total *
        100
        if total > 0
        else 0
    )

    print(
        f"{label:<8}"
        f"{total:<10}"
        f"{s['success']:<12}"
        f"{s['one_hand']:<10}"
        f"{s['two_hands']:<10}"
        f"{s['failed']:<12}"
        f"{success_rate:<10.2f}"
    )


# -----------------------------------------
# Problemli sınıflar
# -----------------------------------------

print()
print("==============================================================")
print("              PROBLEMLI SINIFLAR")
print("==============================================================")
print()

problematic = []

for label, s in stats.items():

    total = s["total"]

    if total == 0:
        continue

    success_rate = (
        s["success"] /
        total *
        100
    )

    if success_rate < 70:

        problematic.append(
            (
                label,
                success_rate,
                s["failed"]
            )
        )


problematic.sort(
    key=lambda x: x[1]
)


for label, rate, failed in problematic:

    print(
        f"{label} -> "
        f"Başarı: %{rate:.2f} | "
        f"Başarısız: {failed}"
    )


print()
print("Bitti.")
