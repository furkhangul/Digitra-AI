"""Solve each sign's INTENDED contact on real surfaces, then report anything else that collides.

Stage 1 brings the two anchors the sign actually joins (index tip to index tip,
thumb pad to middle pad, ...) to surface contact.
Stage 2 measures, separately, the deepest collision anywhere else on the two
hands. Those two numbers are reported apart from each other: a solved contact is
not evidence that nothing else interpenetrates.
"""
import numpy as np, json, sys
import letters as LT
import joined as J

ROOT = r"C:/Users/Furkan/Desktop/Digitra AI"
OFFSETS = ROOT + "/apps/web/src/components/tid/surface-offsets.json"
WANT_GAP = {"Ö": 0.11, "Ç": 0.09}   # the source holds these two apart
MAX_SHIFT = 0.60          # never distort a sign further than this to fix contact

def anchor_world(hand, shape, rotation, side, name):
    """Anchor position in world space for an already-built Hand."""
    p = J.local_anchor(shape, name).copy()
    return hand.M @ p + hand.T

def near(points, centre, radius):
    m = np.linalg.norm(points - centre, axis=1) < radius
    return points[m]

def solve(ch, L):
    R, Lh = LT.pair(L)
    if not (R.visible and Lh.visible): return None
    con = L.get("contact")
    centroid = lambda h: np.array([a for segs in h.caps for (a, b, ra, rb) in segs]).mean(0)
    d = centroid(R) - centroid(Lh)
    d = d / max(np.linalg.norm(d), 1e-9)

    sr_all = R.surface_points(n_per_seg=80, seed=5)
    sl_all = Lh.surface_points(n_per_seg=80, seed=6)

    if con:
        ar = anchor_world(R, L["authored"]["right"]["shape"], None, "right", con["right"])
        al = anchor_world(Lh, L["authored"]["left"]["shape"], None, "left", con["left"])
        rad = 0.34
        sr = near(sr_all, ar, rad); sl = near(sl_all, al, rad)
        if len(sr) < 20 or len(sl) < 20:
            sr, sl = sr_all, sl_all
    else:
        sr, sl = sr_all, sl_all

    want = WANT_GAP.get(ch, 0.0)
    def gap(t):      # gap between the intended contact regions
        off = d * t
        return float(min(Lh.sdf(sr + off).min(), R.sdf(sl - off).min()))
    def worst(t):    # deepest collision anywhere on the two hands
        off = d * t
        return float(min(Lh.sdf(sr_all + off).min(), R.sdf(sl_all - off).min()))

    lo, hi = -MAX_SHIFT, MAX_SHIFT
    g_lo, g_hi = gap(lo), gap(hi)
    if (g_lo - want) * (g_hi - want) > 0:
        t = lo if abs(g_lo - want) < abs(g_hi - want) else hi
        capped = True
    else:
        capped = False
        for _ in range(30):
            mid = (lo + hi) / 2
            g = gap(mid)
            if (g - want) * (g_lo - want) <= 0: hi = mid
            else: lo, g_lo = mid, g
        t = (lo + hi) / 2
    return t, d, gap(t), worst(t), capped

def main():
    requested = set(sys.argv[1:])
    old = json.load(open(OFFSETS, encoding="utf-8"))
    new = dict(old); rows = []
    for ch, L in LT.letters():
        if requested and ch not in requested:
            continue
        r = solve(ch, L)
        if r is None:
            continue
        t, d, g, w, capped = r
        new[ch] = [round(float(v), 6) for v in (np.array(old.get(ch, [0, 0, 0]), float) + d * t)]
        rows.append((ch, t, g, w, capped))
        print(f"{ch:3s} shift={t:+.4f} contact={g:+.4f} worstCollision={w:+.4f} {'CAPPED' if capped else ''}")
    json.dump(new, open(OFFSETS, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    bad = [r for r in rows if r[3] < -0.03]
    print("\nunintended collisions deeper than 0.03:",
          ", ".join(f"{r[0]}({r[3]:+.3f})" for r in bad) or "none")
    print("capped:", ", ".join(r[0] for r in rows if r[4]) or "none")

if __name__ == "__main__":
    main()
