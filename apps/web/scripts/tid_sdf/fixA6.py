import sys, copy; sys.path.insert(0,'.')
import numpy as np, joined as J, letters as LT
from scipy.optimize import minimize
FORK=J.shape_of('A','left'); POINT=J.shape_of('A','right')
# legs = right hand, bar = left hand pointing right; the bar hand's roll is free
def ROTS(x): return [0,0,180+x[4]], [x[5],x[6],-90+x[7]]
def seg_seg(p1,q1,p2,q2):
    d1,d2=q1-p1,q2-p2; r=p1-p2
    a,e,f=d1@d1,d2@d2,d2@r; c=d1@r; b=d1@d2
    den=a*e-b*b
    s=np.clip((b*f-c*e)/den,0,1) if den>1e-9 else 0.0
    t=np.clip((b*s+f)/e,0,1); s=np.clip((b*t-c)/a,0,1)
    A=p1+d1*s; B=p2+d2*t
    return np.linalg.norm(A-B), s, t, A, B
def pt_seg(q,p1,q1):
    d=q1-p1; t=np.clip((q-p1)@d/max(d@d,1e-9),0,1)
    return np.linalg.norm(q-(p1+d*t))
RB=LT.RIG['index']['radius']; RI=RB; RM=LT.RIG['middle']['radius']
def setup(spread,gap,x=None):
    fk=copy.deepcopy(FORK); fk['index']['spread']=spread; fk['middle']['spread']=-spread
    rr,lr = ROTS(x) if x is not None else ([0,0,180],[0,0,-90])
    p=J.joined(fk,POINT,rr,lr,'index_1','index_2',list(gap))
    R,L=J.hands_from(p)
    seg=lambda h,s,f:(h.M@J.local_anchor(s,f+'_0')+h.T, h.M@J.local_anchor(s,f+'_tip')+h.T)
    return fk,R,L,seg(L,POINT,'index'),seg(R,fk,'index'),seg(R,fk,'middle')
def measure(spread,gap,x=None):
    fk,R,L,BAR,A1,A2=setup(spread,gap,x)
    bardir=BAR[1]-BAR[0]
    # "far" = the leg the bar reaches later along its own direction
    proj=lambda LEG: float(((LEG[0]+LEG[1])/2-BAR[0])@bardir)
    near,far=(A1,A2) if proj(A1)<proj(A2) else (A2,A1)
    dN,sN,tN,AN,BN=seg_seg(*BAR,*near); dF,sF,tF,AF,BF=seg_seg(*BAR,*far)
    palmL=L.M@np.array([0,.42,0.])+L.T; palmR=R.M@np.array([0,.42,0.])+R.T
    return dict(gN=dN-(RB+RI), gF=dF-(RB+RM), tN=tN, tF=tF, sF=sF,
                frontN=AN[2]-BN[2], frontF=AF[2]-BF[2],
                fist=min(pt_seg(palmL,*near),pt_seg(palmL,*far)),
                fists=float(np.linalg.norm(palmL-palmR)),
                tipSep=float(np.linalg.norm((R.M@J.local_anchor(fk,'index_tip')+R.T)
                                           -(R.M@J.local_anchor(fk,'middle_tip')+R.T))),
                hands=(fk,R,L))
def cost(x):
    spread=x[0]; m=measure(spread,x[1:4],x)
    c  = (m['gN']-0.005)**2*500 + (m['gF']-0.005)**2*500
    c += (m['tN']-0.45)**2*8 + (m['tF']-0.45)**2*8
    c += max(0.0,0.07-m['frontN'])**2*300 + max(0.0,0.07-m['frontF'])**2*300
    c += (m['sF']-0.80)**2*40
    c += max(0.0,0.62-m['fist'])**2*60 + max(0.0,1.30-m['fists'])**2*120
    c += (m['tipSep']-0.50)**2*3
    c += max(0.0,4.0-spread)**2*2          # keep the legs spread in the image plane
    return c
best=None
for s in range(14):
    rng=np.random.default_rng(s)
    x0=np.concatenate([[rng.uniform(5,12)],rng.uniform(-.5,.5,3),rng.uniform(-25,25,1),rng.uniform(-40,40,3)])
    r=minimize(cost,x0,method='Nelder-Mead',options={'xatol':2e-4,'fatol':1e-10,'maxiter':2500})
    if best is None or r.fun<best.fun: best=r
spread=round(float(best.x[0]),1); gap=[round(float(v),3) for v in best.x[1:4]]
m=measure(spread,gap,best.x); fk,R,L=m['hands']
rr,lr=ROTS(best.x); print('right rot',[round(v,1) for v in rr],' left rot',[round(v,1) for v in lr])
def cen(h): return np.array([a for segs in h.caps for (a,b,r1,r2) in segs]).mean(0)
sr=R.surface_points(n_per_seg=80,seed=11); sl=L.surface_points(n_per_seg=80,seed=12)
print('spread +/-%.1f  gap %s'%(spread,gap))
print('touch near %+.4f far %+.4f ; crosses %.2f/%.2f ; front %.2f/%.2f'%(m['gN'],m['gF'],m['tN'],m['tF'],m['frontN'],m['frontF']))
print('leg tip sep %.2f ; bar reaches %.2f ; fist-fist %.2f'%(m['tipSep'],m['sF'],m['fists']))
print('right x %.2f  left x %.2f -> %s'%(cen(R)[0],cen(L)[0],'R on RIGHT ok' if cen(R)[0]>cen(L)[0] else 'WRONG'))
print('overlap %+.4f'%float(min(L.sdf(sr).min(), R.sdf(sl).min())))
import render_pair as RP
from PIL import Image
ims=[]
for v in [(0,0,0),(0,28,0)]:
    f=f'sheet/A7-{v[1]}.png'; RP.render((R,L),f,W=340,H=340,view=v)
    im=Image.open(f); im.load(); ims.append(im.convert('RGB'))
sh=Image.new('RGB',(ims[0].width*len(ims),ims[0].height))
for i,im in enumerate(ims): sh.paste(im,(i*im.width,0))
sh.save('sheet/A-roles.png'); print('rendered')
