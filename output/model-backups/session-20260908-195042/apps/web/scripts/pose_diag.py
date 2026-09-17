# Diagnostic: inspect the synthetic landmarks for one letter.
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
src = Path(__file__).parent.joinpath("verify_poses.py").read_text(encoding="utf-8")
ns = {"__file__": str(Path(__file__).resolve())}
exec(src.split("def main()")[0], ns)

letter = sys.argv[1] if len(sys.argv) > 1 else "D"
data = json.loads((Path(__file__).parent / "pose_dumps" / f"{letter}.json").read_text(encoding="utf-8"))
raw = np.asarray(data["points"], dtype=np.float32)
lm = ns["synthetic_landmarks"](raw)

chains = {
    "thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8], "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16], "pinky": [17, 18, 19, 20],
}
uv = ns["unit_vector"]
for name, ch in chains.items():
    cosines = []
    for i in range(1, len(ch) - 1):
        a = uv(lm[ch[i - 1]] - lm[ch[i]])
        b = uv(lm[ch[i + 1]] - lm[ch[i]])
        cosines.append(round(float(np.dot(a, b)), 2))
    print(f"{name:>6} joint cos: {cosines}  tip: {[round(v, 2) for v in lm[ch[3]]]}")
print("wrist:", [round(v, 2) for v in lm[0]])
print("mcp5:", [round(v, 2) for v in lm[5]], " mcp17:", [round(v, 2) for v in lm[17]])
