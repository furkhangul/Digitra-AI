# Feed a hand-crafted, anatomically sensible "D" hand to the classifier.
# If this predicts D, the feature pipeline is fine and the rig dump is the
# problem; if not, the feature replication is broken.
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

labels = json.loads((MODEL_DIR / "model_config.json").read_text(encoding="utf-8"))["labels"]
scaler = joblib.load(MODEL_DIR / "landmark" / "feature_scaler.joblib")
extra_trees = joblib.load(MODEL_DIR / "landmark" / "extra_trees.joblib")

# Hand-crafted D: index extended, middle/ring/pinky folded to the palm,
# thumb out to the side. Palm faces +Z, fingers +Y.
lm = np.zeros((21, 3), dtype=np.float32)
lm[0] = [0, 0, 0]
# thumb
lm[1], lm[2], lm[3], lm[4] = [-0.30, 0.12, 0.05], [-0.45, 0.28, 0.08], [-0.55, 0.42, 0.10], [-0.62, 0.55, 0.12]
# index (straight up)
lm[5], lm[6], lm[7], lm[8] = [-0.08, 0.45, 0], [-0.10, 0.68, 0], [-0.11, 0.90, 0], [-0.12, 1.10, 0]
# middle (folded to palm)
lm[9], lm[10], lm[11], lm[12] = [0.0, 0.48, 0], [0.02, 0.70, 0.06], [0.05, 0.72, 0.14], [0.04, 0.62, 0.20]
# ring (folded)
lm[13], lm[14], lm[15], lm[16] = [0.08, 0.45, 0], [0.10, 0.65, 0.06], [0.12, 0.66, 0.13], [0.11, 0.56, 0.19]
# pinky (folded)
lm[17], lm[18], lm[19], lm[20] = [0.16, 0.38, 0], [0.18, 0.56, 0.05], [0.19, 0.56, 0.11], [0.18, 0.47, 0.16]

feature = ns["landmark_feature"](lm, lm.copy(), "Right", 0.95)
scaled = scaler.transform(feature[None].astype(np.float32))
proba = extra_trees.predict_proba(scaled)[0]
order = proba.argsort()[::-1]
for idx in order[:5]:
    print(labels[int(idx)], round(float(proba[idx]), 3))
