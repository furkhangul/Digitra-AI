"""Python mirror of poses.ts joined()/centre() so two-hand layouts can be swept offline."""
import numpy as np, math, json
from sdfhand import RIG, FINGERS
import letters as LT

def euler_quat(rx, ry, rz):
    """THREE.Euler order 'XYZ' -> quaternion (x,y,z,w)."""
    hx, hy, hz = [math.radians(v)/2 for v in (rx, ry, rz)]
    c1, c2, c3 = math.cos(hx), math.cos(hy), math.cos(hz)
    s1, s2, s3 = math.sin(hx), math.sin(hy), math.sin(hz)
    return np.array([s1*c2*c3 + c1*s2*s3,
                     c1*s2*c3 - s1*c2*s3,
                     c1*c2*s3 + s1*s2*c3,
                     c1*c2*c3 - s1*s2*s3])

def local_anchor(shape, anchor):
    if anchor == "wrist": return np.zeros(3)
    if anchor == "palm":  return np.array([0, .42, .17])
    name, seg = anchor.split("_")
    spec = RIG[name]
    d = np.array(spec["direction"], float)
    axis = np.cross(d, [0, 0, 1.0]); axis /= np.linalg.norm(axis)
    f = shape[name]
    q = LT.qmul(LT.axis_quat([0, 0, 1], f["spread"]), LT.axis_quat([0, 1, 0], f["oppose"]))
    p = np.array(spec["base"], float)
    n = 3 if seg == "tip" else int(seg)
    for i in range(n):
        q = LT.qmul(q, LT.axis_quat(axis, f["bend"][i]))
        p = p + LT.qrot(q, d*spec["lengths"][i])
    return p

def anchor_offset(shape, rotation, side, name):
    p = local_anchor(shape, name).copy()
    if side == "left": p[0] *= -1
    return LT.qrot(euler_quat(*rotation), p)

ALL_ANCHORS = ["wrist"] + [f"{f}_{s}" for f in FINGERS for s in ("0", "1", "2", "tip")]

def centre(poses):
    pts = []
    for side, (shape, rot, pos, vis) in poses.items():
        if not vis: continue
        for a in ALL_ANCHORS:
            pts.append(anchor_offset(shape, rot, side, a) + pos)
        pts.append(LT.qrot(euler_quat(*rot), np.array([0, .035, 0])) + pos)
    P = np.array(pts); c = (P.min(0)+P.max(0))/2
    return {s: (sh, r, p - c, v) for s, (sh, r, p, v) in poses.items()}

def joined(right_shape, left_shape, rr, lr, ra, la, gap=(0, 0, .09)):
    r_pos = np.array([-.6, -.7, 0.]); l_pos = np.array([.6, -.7, -.1])
    rp = anchor_offset(right_shape, rr, "right", ra)
    lp = anchor_offset(left_shape, lr, "left", la)
    r_pos = lp + l_pos - rp + np.array(gap, float)
    return centre({"right": (right_shape, rr, r_pos, True),
                   "left":  (left_shape,  lr, l_pos, True)})

def hands_from(poses):
    out = []
    for side in ("right", "left"):
        shape, rot, pos, vis = poses[side]
        angles = []
        for f in FINGERS:
            angles += list(shape[f]["bend"]) + [shape[f]["spread"], shape[f]["oppose"]]
        out.append(LT.Hand({"position": list(pos), "rotation": list(euler_quat(*rot)),
                            "angles": angles, "presence": 1.0 if vis else 0.0}, side))
    return out

def shape_of(ch, side):
    for c, L in LT.letters():
        if c == ch: return L["authored"][side]["shape"]
    raise KeyError(ch)
