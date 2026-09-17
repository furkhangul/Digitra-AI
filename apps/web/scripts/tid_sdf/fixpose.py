"""Search a small pose adjustment that makes a sign's intended contact happen
without any other part of the two hands going through each other.

Only the two wrist orientations and the approach offset move; the hand shapes and
the contact anchors stay exactly as authored, so the sign is not redefined.
"""
import numpy as np, json, sys
import letters as LT, joined as J

def cheap_points(hand, n=26, seed=0):
    """Capsule/palm shell samples without the Newton projection: fast, and close
    enough to rank candidates. Winners are re-checked with the full sampler."""
    rng = np.random.default_rng(seed); pts = []
    for segs in hand.caps:
        for a, b, ra, rb in segs:
            t = rng.random(n)
            c = a + (b - a) * t[:, None]; r = ra + (rb - ra) * t
            v = rng.normal(size=(n, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
            pts.append(c + v * r[:, None])
    for k in LT.PALM_KEYS:
        ctr = np.array(LT.SPEC[k][0]); rad = np.array(LT.SPEC[k][1])
        v = rng.normal(size=(150, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True)
        pts.append((ctr + v * rad) @ hand.M.T + hand.T)
    return np.concatenate(pts, 0)

def evaluate(shapes, rr, lr, ra, la, gap, cheap=True, seed=0):
    p = J.joined(shapes[0], shapes[1], rr, lr, ra, la, gap)
    R, L = J.hands_from(p)
    if cheap:
        sr, sl = cheap_points(R, seed=seed), cheap_points(L, seed=seed + 1)
    else:
        sr, sl = R.surface_points(n_per_seg=80, seed=7), L.surface_points(n_per_seg=80, seed=8)
    ar = R.M @ J.local_anchor(shapes[0], ra) + R.T
    al = L.M @ J.local_anchor(shapes[1], la) + L.T
    nr = sr[np.linalg.norm(sr - ar, axis=1) < .34]
    nl = sl[np.linalg.norm(sl - al, axis=1) < .34]
    if len(nr) < 10 or len(nl) < 10: nr, nl = sr, sl
    contact = float(min(L.sdf(nr).min(), R.sdf(nl).min()))
    worst = float(min(L.sdf(sr).min(), R.sdf(sl).min()))
    return contact, worst, p

def solve_gap(shapes, rr, lr, ra, la, base_gap, seed=0):
    """Slide along the line between the hands until the intended anchors touch."""
    d = None
    lo, hi = -0.7, 0.7
    def f(t):
        nonlocal d
        g = list(base_gap)
        if d is None:
            _, _, p = evaluate(shapes, rr, lr, ra, la, g, seed=seed)
            R, L = J.hands_from(p)
            c = lambda h: np.array([a for segs in h.caps for (a, b, r1, r2) in segs]).mean(0)
            v = c(R) - c(L); d = v / max(np.linalg.norm(v), 1e-9)
        gg = list(np.array(base_gap) + d * t)
        return evaluate(shapes, rr, lr, ra, la, gg, seed=seed)
    c_lo, _, _ = f(lo); c_hi, _, _ = f(hi)
    if c_lo * c_hi > 0:
        t = lo if abs(c_lo) < abs(c_hi) else hi
    else:
        for _ in range(24):
            mid = (lo + hi) / 2
            c, _, _ = f(mid)
            if c * c_lo <= 0: hi = mid
            else: lo, c_lo = mid, c
        t = (lo + hi) / 2
    gg = list(np.array(base_gap) + d * t)
    con, worst, p = f(t)
    return gg, con, worst

def search(ch, rr0, lr0, ra, la, gap0, span=26, steps=3, seed=0):
    shapes = (J.shape_of(ch, "right"), J.shape_of(ch, "left"))
    best = None
    deltas = np.linspace(-span, span, steps)
    for dry in deltas:
        for drz in deltas:
            for dly in deltas:
              for dlz in deltas:
                rr = [rr0[0], rr0[1] + dry, rr0[2] + drz]
                lr = [lr0[0], lr0[1] + dly, lr0[2] + dlz]
                gg, con, worst = solve_gap(shapes, rr, lr, ra, la, gap0, seed=seed)
                if abs(con) > .05: continue
                penalty = abs(dry) + abs(drz) + abs(dly) + abs(dlz)
                score = min(worst, 0.0) * 100 - penalty * .012
                if best is None or score > best[0]:
                    best = (score, rr, lr, gg, con, worst)
    return best
