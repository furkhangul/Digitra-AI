"""Render a posed two-hand letter with the same SDF and lighting as the site."""
import numpy as np, math
from PIL import Image, ImageDraw, ImageFont
import letters as LT

RIGHT = np.array([.96, .69, .37])
LEFT  = np.array([.32, .12, .72])
BG    = np.array([.145, .118, .239])

def rot_matrix(rx, ry, rz):
    rx, ry, rz = map(math.radians, (rx, ry, rz))
    Rx=np.array([[1,0,0],[0,math.cos(rx),-math.sin(rx)],[0,math.sin(rx),math.cos(rx)]])
    Ry=np.array([[math.cos(ry),0,math.sin(ry)],[0,1,0],[-math.sin(ry),0,math.cos(ry)]])
    Rz=np.array([[math.cos(rz),-math.sin(rz),0],[math.sin(rz),math.cos(rz),0],[0,0,1]])
    return Rz@Ry@Rx

def frame_bounds(hands, pad=.30):
    pts=[]
    for h in hands:
        if not h.visible: continue
        for segs in h.caps:
            for a,b,ra,rb in segs: pts += [a-ra, a+ra, b-rb, b+rb]
        for k in LT.PALM_KEYS:
            c=np.array(LT.SPEC[k][0]); r=np.array(LT.SPEC[k][1])
            for s in [-1,1]:
                for ax in range(3):
                    v=np.zeros(3); v[ax]=s*r[ax]
                    pts.append(h.to_world(c+v))
    P=np.array(pts); lo=P.min(0)-pad; hi=P.max(0)+pad
    return lo, hi

def render(hands, path, W=360, H=360, view=(0,0,0), scale=None, centre=None):
    lo, hi = frame_bounds(hands)
    c = (lo+hi)/2 if centre is None else np.array(centre)
    s = float(max(hi[0]-lo[0], hi[1]-lo[1]))/2 if scale is None else scale
    R = rot_matrix(*view)
    xs=(np.arange(W)+.5)/W*2-1; ys=1-(np.arange(H)+.5)/H*2
    gx,gy=np.meshgrid(xs*s+c[0], ys*s+c[1])
    depth = float(hi[2]-lo[2])+2.0
    ro=np.stack([gx,gy,np.full_like(gx,c[2]+depth)],-1).reshape(-1,3)
    ro=(ro-c)@R+c
    rd=R.T@np.array([0.,0.,-1.])
    N=ro.shape[0]; t=np.zeros(N); hit=np.zeros(N,bool); which=np.zeros(N,int)
    idx=np.arange(N)
    for _ in range(150):
        if idx.size==0: break
        p=ro[idx]+t[idx,None]*rd
        da=hands[0].sdf(p); db=hands[1].sdf(p)
        d=np.minimum(da,db)
        h=d<1.2e-3
        if h.any():
            hit[idx[h]]=True; which[idx[h]]=np.where(da[h]<=db[h],0,1)
        t[idx]+=np.maximum(d*.85,8e-4)
        keep=(~h)&(t[idx]<2*depth+4)
        idx=idx[keep]
    col=np.tile(BG,(N,1))
    hi_i=np.where(hit)[0]
    if hi_i.size:
        p=ro[hi_i]+t[hi_i,None]*rd; w=which[hi_i]
        def sdf_sel(q):
            a=hands[0].sdf(q); b=hands[1].sdf(q)
            return np.where(w==0,a,b)
        e=1.6e-3
        n=np.stack([sdf_sel(p+[e,0,0])-sdf_sel(p-[e,0,0]),
                    sdf_sel(p+[0,e,0])-sdf_sel(p-[0,e,0]),
                    sdf_sel(p+[0,0,e])-sdf_sel(p-[0,0,e])],-1)
        n/=np.maximum(np.linalg.norm(n,axis=-1,keepdims=True),1e-9)
        ao=np.ones(hi_i.size)
        for i in range(1,5):
            hh=i*.055
            ao-=np.maximum(hh-np.minimum(hands[0].sdf(p+n*hh),hands[1].sdf(p+n*hh)),0)*1.25
        ao=np.clip(ao,.45,1.0)
        L=np.array([-.55,.95,1.35]); L/=np.linalg.norm(L); Lw=R.T@L
        Hh=Lw-rd; Hh/=np.linalg.norm(Hh)
        diff=np.clip(n@Lw,0,1); sp=np.clip(n@Hh,0,1)**34; sh=np.clip(n@Hh,0,1)**6
        rim=(1-np.clip(n@(-rd),0,1))**3
        base=np.where(w[:,None]==0,RIGHT,LEFT)
        c2=base*(.50+diff*.58)[:,None]*ao[:,None]
        c2+=np.array([1.,.92,.80])*(sp*.30)[:,None]+np.array([1.,.80,.58])*(sh*.05)[:,None]
        c2+=np.array([.55,.34,1.])*(rim*.22)[:,None]
        col[hi_i]=c2
    img=np.clip(col.reshape(H,W,3),0,1)**(1/2.2)
    im=Image.fromarray((img*255).astype(np.uint8))
    im.save(path)
    return im
