"""Hand shape spec -> SDF. Single source of truth, ported verbatim to GLSL."""
import numpy as np
from sdfhand import smooth_union, ellipsoid, capsule, finger_points, catmull, FINGERS, RIG

SEGMENTS = 8

# ---- shape parameters (tunable) ----
SPEC = dict(
    palm_core   = ((0.005, 0.470, -0.020), (0.400, 0.500, 0.185)),
    knuckle_pad = ((0.005, 0.800, -0.005), (0.465, 0.215, 0.160)),
    heel        = ((0.020, 0.185, -0.015), (0.345, 0.245, 0.180)),
    thenar      = ((-0.235, 0.315,  0.100), (0.268, 0.380, 0.232)),
    thumb_web   = ((-0.315, 0.560,  0.045), (0.150, 0.235, 0.150)),
    hypothenar  = ((0.325, 0.420, -0.005), (0.185, 0.325, 0.170)),
    k_palm      = 0.26,
    k_finger    = 0.105,
    k_thumb     = 0.255,
    radius_scale = dict(thumb=1.0, index=1.0, middle=1.0, ring=1.0, pinky=1.0),
    profile      = (1.10, 1.03, 0.985, 0.975, 1.005, 0.955, 0.985, 0.945, 0.90),
    profile_thumb= (1.00, 1.02, 1.03, 1.03, 1.02, 1.01, 1.00, 0.99, 0.96),
)

def profile_at(u, prof):
    f = u * (len(prof) - 1); i = min(int(f), len(prof) - 2); t = f - i
    return prof[i] * (1 - t) + prof[i + 1] * t

def build_capsules(shape, spec=SPEC):
    """shape: {finger: (bend3, spread, oppose)} -> list of (a,b,ra,rb) in local space."""
    caps = {}
    for name in FINGERS:
        bend, spread, oppose = shape[name]
        pts = finger_points(name, bend, spread, oppose)
        r0 = RIG[name]["radius"] * spec["radius_scale"][name]
        prof = spec["profile_thumb"] if name == "thumb" else spec["profile"]
        segs = []
        prev = catmull(pts, 0.0); prev_r = r0 * profile_at(0.0, prof)
        for s in range(SEGMENTS):
            u = (s + 1) / SEGMENTS
            cur = catmull(pts, u); cur_r = r0 * profile_at(u, prof)
            segs.append((prev, cur, prev_r, cur_r))
            prev, prev_r = cur, cur_r
        caps[name] = segs
    return caps

def hand_sdf(p, caps, spec=SPEC):
    """p: (...,3) local-space points -> signed distance."""
    E = lambda key: ellipsoid(p, np.array(spec[key][0]), np.array(spec[key][1]))
    palm = E("palm_core")
    for key in ("knuckle_pad", "heel", "thenar", "hypothenar", "thumb_web"):
        palm = smooth_union(palm, E(key), spec["k_palm"])
    # Each finger blends into the PALM only. Fingers are combined with a hard
    # min, so touching fingers keep a crease instead of melting together.
    d = palm
    for name in FINGERS:
        segs = caps[name]
        f = capsule(p, segs[0][0], segs[0][1], segs[0][2], segs[0][3])
        for a, b, ra, rb in segs[1:]:
            f = np.minimum(f, capsule(p, a, b, ra, rb))
        k = spec["k_thumb"] if name == "thumb" else spec["k_finger"]
        d = np.minimum(d, smooth_union(palm, f, k))
    return d
