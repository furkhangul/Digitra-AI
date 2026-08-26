import pandas as pd

FILE = "confusion_matrix_v9.csv"

cm = pd.read_csv(
    FILE,
    index_col=0,
    encoding="utf-8-sig"
)

print()
print("========================================")
print(" DIGITRA CONFUSION ANALYSIS V13")
print("========================================")
print()

pairs = []

for true_label in cm.index:

    row = cm.loc[true_label]

    total = row.sum()

    correct = row[true_label]

    errors = row.drop(true_label)

    if errors.sum() == 0:
        continue

    for pred_label, count in errors.items():

        if count > 0:

            pairs.append(
                (
                    true_label,
                    pred_label,
                    int(count),
                    int(total),
                    float(count / total)
                )
            )


pairs.sort(
    key=lambda x: x[2],
    reverse=True
)


print("En çok karışan sınıflar:")
print()

for (
    true_label,
    pred_label,
    count,
    total,
    ratio
) in pairs[:30]:

    print(
        f"{true_label:3} -> {pred_label:3} "
        f"| {count}/{total} "
        f"| %{ratio*100:.1f}"
    )


print()
print("========================================")
print(" SINIF BAZLI EN BÜYÜK HATA")
print("========================================")
print()


for true_label in cm.index:

    row = cm.loc[true_label]

    errors = row.drop(true_label)

    if errors.sum() == 0:

        print(
            f"{true_label:3} -> hata yok"
        )

        continue

    pred_label = errors.idxmax()

    count = errors.max()

    total = row.sum()

    print(
        f"{true_label:3} "
        f"en çok -> {pred_label:3} "
        f"| {count}/{total}"
    )