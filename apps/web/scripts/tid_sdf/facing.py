"""Bring a sign into the natural viewing convention, without changing its shapes.

The convention, from the signs that already read correctly (B, Ç, D): the right
hand sits on the right and the left hand on the left, and each palm is turned
INWARD — the learner sees another person's hands, not their own. Only the two
wrist orientations and the approach offset move; the hand shapes and the contact
anchors are untouched, so the sign itself is not redefined.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, joined as J, fixpose as F

def centroid(h):
    return np.array([a for segs in h.caps for (a, b, r1, r2) in segs]).mean(0)

def palm_x(h):
    n = h.M @ np.array([0, 0, 1.0])
    return float(n[0] / np.linalg.norm(n))

def evaluate(ch, rr, lr, ra, la, gap, swap=False):
    sr = J.shape_of(ch, "left" if swap else "right")
    sl = J.shape_of(ch, "right" if swap else "left")
    gg, con, worst = F.solve_gap((sr, sl), rr, lr, ra, la, gap)
    p = J.joined(sr, sl, rr, lr, ra, la, gg)
    R, L = J.hands_from(p)
    return dict(gap=gg, contact=con, worst=worst, R=R, L=L,
                sideOK=centroid(R)[0] > centroid(L)[0],
                rx=palm_x(R), lx=palm_x(L))

def search(ch, rr0, lr0, ra, la, gap0, swap=False,
           ry_choices=(-60, -45, -30, 0), lry_choices=(0, 30, 45, 60),
           gapx_signs=(1, -1)):
    """Try inward-facing orientations; keep the one that satisfies every rule
    with the most collision margin, preferring the smallest change."""
    best = None
    for dry in ry_choices:
        for dly in lry_choices:
            for sx in gapx_signs:
                rr = [rr0[0], dry, rr0[2]]
                lr = [lr0[0], dly, lr0[2]]
                g = [gap0[0] * sx, gap0[1], gap0[2]]
                try:
                    m = evaluate(ch, rr, lr, ra, la, g, swap)
                except Exception:
                    continue
                if abs(m["contact"]) > .03 or m["worst"] < -.02: continue
                if not m["sideOK"]: continue
                if m["rx"] > .15 or m["lx"] < -.15: continue
                score = m["worst"] - .0015 * (abs(dry) + abs(dly))
                if best is None or score > best[0]:
                    best = (score, rr, lr, m)
    return best
