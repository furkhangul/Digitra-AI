"""Offline mirror of the web hand SDF. Numpy raymarcher for fast shape iteration."""
import numpy as np, json, math

# ---------- math helpers (mirror GLSL) ----------
def smooth_union(a, b, k):
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * k * 0.25

def ellipsoid(p, c, r):
    q = (p - c) / r
    k0 = np.linalg.norm(q, axis=-1)
    k1 = np.linalg.norm((p - c) / (r * r), axis=-1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-5)

def capsule(p, a, b, ra, rb):
    seg = b - a
    d2 = max(float(seg @ seg), 1e-9)
    t = np.clip(((p - a) @ seg) / d2, 0.0, 1.0)
    proj = a + t[..., None] * seg
    return np.linalg.norm(p - proj, axis=-1) - (ra + (rb - ra) * t)

# ---------- rig ----------
ROOT = r"C:/Users/Furkan/Desktop/Digitra AI"
RIG = json.load(open(ROOT + "/apps/web/src/components/tid/rig-spec.json"))
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

def quat(axis, deg):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    h = math.radians(deg) / 2.0
    return np.array([*(axis * math.sin(h)), math.cos(h)])

def qmul(a, b):
    ax, ay, az, aw = a; bx, by, bz, bw = b
    return np.array([aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx,
                     aw*bz+ax*by-ay*bx+az*bw, aw*bw-ax*bx-ay*by-az*bz])

def qrot(q, v):
    x, y, z, w = q; u = np.array([x, y, z])
    return v + 2.0 * np.cross(u, np.cross(u, v) + w * v)

def catmull(pts, t):
    """centripetal catmull-rom through 4 points, matching THREE.CatmullRomCurve3."""
    P = np.asarray(pts, float); n = len(P) - 1
    f = t * n; i = min(int(f), n - 1); u = f - i
    p0 = P[i - 1] if i > 0 else P[0] * 2 - P[1]
    p1, p2 = P[i], P[i + 1]
    p3 = P[i + 2] if i + 2 <= n else P[n] * 2 - P[n - 1]
    def tj(ti, a, b):
        d = np.linalg.norm(b - a)
        return ti + (d ** 0.25 if d > 1e-9 else 1e-4)
    t0 = 0.0; t1 = tj(t0, p0, p1); t2 = tj(t1, p1, p2); t3 = tj(t2, p2, p3)
    tt = t1 + (t2 - t1) * u
    A1 = (t1-tt)/(t1-t0)*p0 + (tt-t0)/(t1-t0)*p1
    A2 = (t2-tt)/(t2-t1)*p1 + (tt-t1)/(t2-t1)*p2
    A3 = (t3-tt)/(t3-t2)*p2 + (tt-t2)/(t3-t2)*p3
    B1 = (t2-tt)/(t2-t0)*A1 + (tt-t0)/(t2-t0)*A2
    B2 = (t3-tt)/(t3-t1)*A2 + (tt-t1)/(t3-t1)*A3
    return (t2-tt)/(t2-t1)*B1 + (tt-t1)/(t2-t1)*B2

def finger_points(name, bend, spread, oppose):
    spec = RIG[name]
    d = np.array(spec["direction"], float)
    axis = np.cross(d, [0, 0, 1.0]); axis /= np.linalg.norm(axis)
    q = qmul(quat([0, 0, 1], spread), quat([0, 1, 0], oppose))
    p = np.array(spec["base"], float); pts = [p.copy()]
    for j in range(3):
        q = qmul(q, quat(axis, bend[j]))
        p = p + qrot(q, d * spec["lengths"][j])
        pts.append(p.copy())
    return pts
