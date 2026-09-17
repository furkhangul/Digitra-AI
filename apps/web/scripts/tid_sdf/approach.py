"""Per-letter release/approach clearance, measured on the real surface.

A single fixed separation is not enough for every sign: some letters interlock,
so a straight move from "apart" to the final pose drives one hand through the
other. For each sign this finds the smallest separation, along the line the two
hands actually part on, that leaves a clear margin between the surfaces.

Writes apps/web/src/components/tid/approach-offsets.json.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import letters as LT

ROOT = r"C:/Users/Furkan/Desktop/Digitra AI"
TARGET = os.path.join(ROOT, "apps/web/src/components/tid/approach-offsets.json")
CLEAR = 0.16          # wanted gap between the two surfaces while apart
MIN_PART = 1.05       # the parting must also survive the blend between two signs
MAX = 2.2

def centroid(h):
    return np.array([a for segs in h.caps for (a, b, r1, r2) in segs]).mean(0)

def main():
    requested = set(sys.argv[1:])
    out = json.load(open(TARGET, encoding="utf-8")) if requested else {}
    for ch, L in LT.letters():
        if requested and ch not in requested:
            continue
        R, Lh = LT.pair(L)
        if not (R.visible and Lh.visible):
            out[ch] = [0.0, 0.0, 0.0]; continue
        d = centroid(R) - centroid(Lh)
        n = np.linalg.norm(d)
        d = d / n if n > 1e-6 else np.array([1.0, 0.0, 0.0])
        sr = R.surface_points(n_per_seg=70, seed=301)
        sl = Lh.surface_points(n_per_seg=70, seed=302)
        gap = lambda t: float(min(Lh.sdf(sr + d * t).min(), R.sdf(sl - d * t).min()))
        lo, hi = 0.0, MAX
        if gap(hi) < CLEAR:
            t = hi
        else:
            for _ in range(24):
                mid = (lo + hi) / 2
                if gap(mid) >= CLEAR: hi = mid
                else: lo = mid
            t = hi
        # stored as the RIGHT hand's offset; the left mirrors it
        out[ch] = [round(float(v), 4) for v in d * max(t, MIN_PART) / 2]
        print(f"{ch:3s} clearance={t:.3f} gap={gap(t):+.3f} used={max(t,MIN_PART):.2f} offset={np.round(d*max(t,MIN_PART)/2,3).tolist()}")
    json.dump(out, open(TARGET, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("wrote", TARGET)

if __name__ == "__main__":
    main()
