# Classify every dumped letter pose with the trained model and — for poses
# the model misreads — suggest curl adjustments based on the extended-finger
# signature difference between the predicted and target letter.
#
#   python scripts/classify_suggest.py [handedness]
# Writes pose_dumps/suggestions.json + prints a report.

import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
MODEL_DIR = Path(r"C:\Users\Furkan\Desktop\Digitra AI\models\DIGITRA_V3_HYBRID_9415")
src = Path(__file__).parent.joinpath("verify_poses.py").read_text(encoding="utf-8")
ns = {"__file__": str(Path(__file__).resolve())}
exec(src.split("def main()")[0], ns)

PASS_CONF = 0.45
MAP = {"Ç": "C", "Ğ": "G", "İ": "I", "Ö": "O", "Ş": "S", "Ü": "U"}
SIGNATURES = {
    "A": set(), "B": {"index", "middle", "ring", "pinky"}, "C": set(), "D": {"index"},
    "E": set(), "F": {"middle", "ring", "pinky"}, "G": {"index"}, "H": {"index", "middle"},
    "I": {"pinky"}, "K": {"index", "middle"}, "L": {"index"}, "M": set(), "N": set(),
    "O": set(), "P": {"index", "middle"}, "Q": {"index"}, "R": {"index", "middle"},
    "S": set(), "T": set(), "U": {"index", "middle"}, "V": {"index", "middle"},
    "W": {"index", "middle", "ring"}, "X": {"index"}, "Y": {"pinky"},
}
THUMB_EXT = {"L", "Y"}


def main():
    handedness = sys.argv[1] if len(sys.argv) > 1 else "Right"
    labels = json.loads((MODEL_DIR / "model_config.json").read_text(encoding="utf-8"))["labels"]
    scaler = joblib.load(MODEL_DIR / "landmark" / "feature_scaler.joblib")
    extra_trees = joblib.load(MODEL_DIR / "landmark" / "extra_trees.joblib")

    all_data = json.loads(
        (Path(__file__).parent / "pose_dumps" / "all.json").read_text(encoding="utf-8")
    )
    suggestions, rows, passed = {}, [], 0
    for ch, points in all_data.items():
        if ch in ("J", "Z"):
            continue
        expected = MAP.get(ch, ch)
        pts = ns["synthetic_landmarks"](np.asarray(points, dtype=np.float32))
        feature = ns["landmark_feature"](pts, pts.copy(), handedness, 0.95)
        scaled = scaler.transform(feature[None].astype(np.float32))
        proba = extra_trees.predict_proba(scaled)[0]
        order = proba.argsort()[::-1]
        predicted = labels[int(extra_trees.classes_[order[0]])]
        conf = float(proba[order[0]])
        ok = predicted == expected and conf >= PASS_CONF
        passed += ok
        rows.append(
            f"{'OK  ' if ok else 'MISS'} {ch:>2} → {expected:>2} | "
            f"tahmin {predicted} ({conf:.2f})"
        )
        if not ok:
            target_sig = SIGNATURES[expected]
            pred_sig = SIGNATURES.get(predicted, set())
            deltas = {}
            for f in ("index", "middle", "ring", "pinky"):
                te, pe = f in target_sig, f in pred_sig
                if te and not pe:
                    deltas[f] = -0.3
                elif (not te) and pe:
                    deltas[f] = 0.3
                else:
                    deltas[f] = -0.12 if te else 0.12
            te, pe = expected in THUMB_EXT, predicted in THUMB_EXT
            deltas["thumb"] = (-0.25 if te and not pe else 0.25 if pe and not te else (-0.1 if te else 0.1))
            suggestions[ch] = deltas

    out = Path(__file__).parent / "pose_dumps" / "suggestions.json"
    out.write_text(json.dumps(suggestions, indent=1), encoding="utf-8")
    print("\n".join(rows))
    print(f"\n{passed}/{len(rows)} geçti — suggestions.json yazıldı ({len(suggestions)} düzeltme)")


if __name__ == "__main__":
    main()
