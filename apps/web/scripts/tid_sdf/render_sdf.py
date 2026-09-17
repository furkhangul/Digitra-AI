"""Raymarch the hand SDF offline (ray-compacted) and save a PNG."""
import numpy as np, math
from PIL import Image
from shape import build_capsules, hand_sdf, SPEC

def rot_matrix(rx, ry, rz):
    rx, ry, rz = map(math.radians, (rx, ry, rz))
    Rx = np.array([[1,0,0],[0,math.cos(rx),-math.sin(rx)],[0,math.sin(rx),math.cos(rx)]])
    Ry = np.array([[math.cos(ry),0,math.sin(ry)],[0,1,0],[-math.sin(ry),0,math.cos(ry)]])
    Rz = np.array([[math.cos(rz),-math.sin(rz),0],[math.sin(rz),math.cos(rz),0],[0,0,1]])
    return Rz @ Ry @ Rx

def render(shape, path, W=380, H=380, scale=1.45, center=(0.0,0.62,0.0),
           view=(0,0,0), spec=SPEC, bg=(0.10,0.09,0.16), base=(0.96,0.69,0.37),
           caps=None):
    if caps is None: caps = build_capsules(shape, spec)
    R = rot_matrix(*view)
    xs = (np.arange(W)+.5)/W*2-1; ys = 1-(np.arange(H)+.5)/H*2
    gx, gy = np.meshgrid(xs*scale+center[0], ys*scale+center[1])
    ro = np.stack([gx, gy, np.full_like(gx, 2.2)], -1).reshape(-1,3) @ R
    rd = R.T @ np.array([0.0,0.0,-1.0])
    N = ro.shape[0]
    t = np.zeros(N); hit = np.zeros(N, bool)
    idx = np.arange(N)                      # alive ray indices
    for _ in range(110):
        if idx.size == 0: break
        p = ro[idx] + t[idx,None]*rd
        d = hand_sdf(p, caps, spec)
        h = d < 1.2e-3
        if h.any():
            hit[idx[h]] = True
        t[idx] += np.maximum(d*0.85, 8e-4)
        keep = (~h) & (t[idx] < 5.0)
        idx = idx[keep]
    col = np.tile(np.array(bg,float), (N,1))
    hi = np.where(hit)[0]
    if hi.size:
        p = ro[hi] + t[hi,None]*rd
        e = 1.6e-3
        def dv(o): return hand_sdf(p+o, caps, spec)
        n = np.stack([dv(np.array([e,0,0]))-dv(np.array([-e,0,0])),
                      dv(np.array([0,e,0]))-dv(np.array([0,-e,0])),
                      dv(np.array([0,0,e]))-dv(np.array([0,0,-e]))],-1)
        n /= np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-9)
        ao = np.ones(hi.size)
        for i in range(1,4):
            hh = i*0.055
            ao -= np.maximum(hh - hand_sdf(p+n*hh, caps, spec), 0)*1.25
        ao = np.clip(ao,0.45,1.0)
        L = np.array([-0.55,0.95,1.35]); L/=np.linalg.norm(L); Lw = R.T@L
        Hh = Lw - rd; Hh/=np.linalg.norm(Hh)
        diff = np.clip(n@Lw,0,1); sp = np.clip(n@Hh,0,1)**34
        rim = (1-np.clip(n@(-rd),0,1))**3
        c = np.array(base)*(0.50+diff*0.58)[:,None]*ao[:,None]
        c += np.array([1.0,0.92,0.80])*(sp*0.30)[:,None]
        c += np.array([0.55,0.34,1.0])*(rim*0.22)[:,None]
        col[hi] = c
    img = np.clip(col.reshape(H,W,3),0,1)**(1/2.2)
    Image.fromarray((img*255).astype(np.uint8)).save(path)
    return int(hit.sum())
