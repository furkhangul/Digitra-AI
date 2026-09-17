"""Two-hand TID letter SDF, matching apps/web/src/components/tid/rounded-hand.ts."""
import numpy as np, json, math
from sdfhand import smooth_union, ellipsoid, capsule, catmull, FINGERS, RIG
from shape import SPEC, profile_at, SEGMENTS

ROOT = r"C:/Users/Furkan/Desktop/Digitra AI"
FRAMES = json.load(open(ROOT + "/tmp/tid/letter-frames.json", encoding="utf-8"))

PALM_KEYS = ("palm_core", "knuckle_pad", "heel", "thenar", "hypothenar", "thumb_web")

def quat_matrix(q):
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])

def qmul(a, b):
    ax,ay,az,aw=a; bx,by,bz,bw=b
    return np.array([aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx,
                     aw*bz+ax*by-ay*bx+az*bw, aw*bw-ax*bx-ay*by-az*bz])

def axis_quat(axis, deg):
    a=np.asarray(axis,float); a=a/np.linalg.norm(a); h=math.radians(deg)/2
    return np.array([*(a*math.sin(h)), math.cos(h)])

def qrot(q,v):
    x,y,z,w=q; u=np.array([x,y,z])
    return v + 2.0*np.cross(u, np.cross(u,v)+w*v)

class Hand:
    """One posed hand: world-space capsules plus a local-space palm."""
    def __init__(self, hand_frame, side):
        self.presence = max(1e-3, hand_frame["presence"])
        self.visible = hand_frame["presence"] > .008
        pos = np.array(hand_frame["position"], float)
        rot = np.array(hand_frame["rotation"], float)
        sgn = -1.0 if side == "left" else 1.0
        R = quat_matrix(rot)
        S = np.diag([sgn*self.presence, self.presence, self.presence])
        self.M = R @ S                    # local -> world linear part
        self.T = pos
        self.Minv = np.linalg.inv(self.M)
        angles = hand_frame["angles"]
        self.caps = []
        for fi, name in enumerate(FINGERS):
            spec = RIG[name]; off = fi*5
            d = np.array(spec["direction"], float)
            axis = np.cross(d, [0,0,1.0]); axis /= np.linalg.norm(axis)
            q = qmul(axis_quat([0,0,1], angles[off+3]), axis_quat([0,1,0], angles[off+4]))
            p = np.array(spec["base"], float); pts=[p.copy()]
            for j in range(3):
                q = qmul(q, axis_quat(axis, angles[off+j]))
                p = p + qrot(q, d*spec["lengths"][j]); pts.append(p.copy())
            prof = SPEC["profile_thumb"] if name=="thumb" else SPEC["profile"]
            r0 = spec["radius"]
            prev = self.to_world(catmull(pts,0.0)); prev_r = r0*profile_at(0.0,prof)*self.presence
            segs=[]
            for s in range(SEGMENTS):
                u=(s+1)/SEGMENTS
                cur=self.to_world(catmull(pts,u)); cur_r=r0*profile_at(u,prof)*self.presence
                segs.append((prev,cur,prev_r,cur_r)); prev,prev_r=cur,cur_r
            self.caps.append(segs)

    def to_world(self, p):  return self.M @ np.asarray(p,float) + self.T
    def to_local(self, p):  return (np.asarray(p,float) - self.T) @ self.Minv.T

    def sdf(self, P):
        P = np.atleast_2d(P)
        if not self.visible: return np.full(len(P), 100.0)
        L = (P - self.T) @ self.Minv.T
        d = ellipsoid(L, np.array(SPEC["palm_core"][0]), np.array(SPEC["palm_core"][1]))
        for k in PALM_KEYS[1:]:
            d = smooth_union(d, ellipsoid(L, np.array(SPEC[k][0]), np.array(SPEC[k][1])), SPEC["k_palm"])
        d = d * self.presence
        for fi, segs in enumerate(self.caps):
            f = capsule(P, *segs[0])
            for s in segs[1:]: f = np.minimum(f, capsule(P, *s))
            k = (SPEC["k_thumb"] if FINGERS[fi]=="thumb" else SPEC["k_finger"]) * self.presence
            d = smooth_union(d, f, k)
        return d

    def surface_points(self, n_per_seg=110, seed=0):
        """Points near the surface: seeded on capsule/palm shells, then projected
        onto the true isosurface with Newton steps on the SDF."""
        rng = np.random.default_rng(seed)
        pts=[]
        for segs in self.caps:
            for a,b,ra,rb in segs:
                t = rng.random(n_per_seg)
                c = a + (b-a)*t[:,None]; r = ra + (rb-ra)*t
                v = rng.normal(size=(n_per_seg,3)); v/=np.linalg.norm(v,axis=1,keepdims=True)
                pts.append(c + v*r[:,None]*1.05)
        for k in PALM_KEYS:
            ctr=np.array(SPEC[k][0]); rad=np.array(SPEC[k][1])
            v=rng.normal(size=(420,3)); v/=np.linalg.norm(v,axis=1,keepdims=True)
            local = ctr + v*rad*1.05
            pts.append(local @ self.M.T + self.T)
        P=np.concatenate(pts,0)
        for _ in range(6):                       # project onto the isosurface
            d=self.sdf(P)
            e=1e-3
            g=np.stack([self.sdf(P+[e,0,0])-self.sdf(P-[e,0,0]),
                        self.sdf(P+[0,e,0])-self.sdf(P-[0,e,0]),
                        self.sdf(P+[0,0,e])-self.sdf(P-[0,0,e])],-1)/(2*e)
            g/=np.maximum(np.linalg.norm(g,axis=1,keepdims=True),1e-9)
            P = P - g*d[:,None]
        keep = np.abs(self.sdf(P)) < 5e-3
        return P[keep]

def letters():
    for L in FRAMES["letters"]:
        yield L["ch"], L

def pair(letter):
    f = letter["frame"]
    return Hand(f["right"], "right"), Hand(f["left"], "left")
