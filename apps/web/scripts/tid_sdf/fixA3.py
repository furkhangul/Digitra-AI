import sys, copy; sys.path.insert(0,'.')
import numpy as np, joined as J, letters as LT
from scipy.optimize import minimize
sr=J.shape_of('A','right'); base=J.shape_of('A','left')
LROT=[0,0,204]; RROT=[0,0,-90]
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
RP_=LT.RIG['index']['radius']; RI=RP_; RM=LT.RIG['middle']['radius']
def setup(spread,gap):
    sl=copy.deepcopy(base); sl['index']['spread']=spread; sl['middle']['spread']=-spread
    p=J.joined(sr,sl,RROT,LROT,'index_2','index_1',list(gap))
    R,L=J.hands_from(p)
    seg=lambda h,s,f:(h.M@J.local_anchor(s,f+'_0')+h.T, h.M@J.local_anchor(s,f+'_tip')+h.T)
    return sl,R,L,seg(R,sr,'index'),seg(L,sl,'index'),seg(L,sl,'middle')
def measure(spread,gap):
    sl,R,L,P,FI,FM=setup(spread,gap)
    dI,sI,tI,AI,BI=seg_seg(*P,*FI); dM,sM,tM,AM,BM=seg_seg(*P,*FM)
    palm=R.M@np.array([0,.42,.0])+R.T          # centre of the pointing hand's fist
    clearI=pt_seg(palm,*FI); clearM=pt_seg(palm,*FM)
    return dict(gI=dI-(RP_+RI), gM=dM-(RP_+RM), tI=tI, tM=tM, sI=sI, sM=sM,
                frontI=AI[2]-BI[2], frontM=AM[2]-BM[2],
                fist=min(clearI,clearM), tipSep=float(np.linalg.norm(
                    (L.M@J.local_anchor(sl,'index_tip')+L.T)-(L.M@J.local_anchor(sl,'middle_tip')+L.T))),
                hands=(sl,R,L))
def cost(x):
    m=measure(x[0],x[1:4])
    c  = (m['gI']-0.005)**2*500 + (m['gM']-0.005)**2*500
    c += (m['tI']-0.45)**2*8 + (m['tM']-0.45)**2*8
    c += max(0.0,0.07-m['frontI'])**2*300 + max(0.0,0.07-m['frontM'])**2*300
    c += (m['sM']-0.80)**2*40                 # let the bar's tip emerge past the far leg
    c += max(0.0,0.62-m['fist'])**2*60        # fist stays clear of the legs
    c += (m['tipSep']-0.50)**2*3              # legs a readable, not splayed, distance apart
    return c
best=None
for s in range(12):
    rng=np.random.default_rng(s)
    x0=np.concatenate([[rng.uniform(6,16)],rng.uniform(-.5,.5,3)])
    r=minimize(cost,x0,method='Nelder-Mead',options={'xatol':2e-4,'fatol':1e-10,'maxiter':2000})
    if best is None or r.fun<best.fun: best=r
spread=round(float(best.x[0]),1); gap=[round(float(v),3) for v in best.x[1:4]]
m=measure(spread,gap); sl,R,L=m['hands']
print('spread +/-%.1f  gap %s'%(spread,gap))
print('touch: left leg %+.4f  right leg %+.4f'%(m['gI'],m['gM']))
print('crosses %.2f/%.2f down the legs; in front by %.2f/%.2f'%(m['tI'],m['tM'],m['frontI'],m['frontM']))
print('leg tip separation %.2f ; fist clearance %.2f ; bar reaches %.2f'%(m['tipSep'],m['fist'],m['sM']))
srf=R.surface_points(n_per_seg=85,seed=11); slf=L.surface_points(n_per_seg=85,seed=12)
print('worst surface overlap %+.4f'%float(min(L.sdf(srf).min(), R.sdf(slf).min())))
import render_pair as RP
from PIL import Image, ImageDraw
ims=[]
for v in [(0,0,0),(0,28,0),(0,-28,0)]:
    f=f'sheet/A5-{v[1]}.png'; RP.render((R,L),f,W=340,H=340,view=v)
    im=Image.open(f); im.load(); im=im.convert('RGB')
    ImageDraw.Draw(im).text((6,4),f'view {v[1]}',fill=(255,230,150)); ims.append(im)
sh=Image.new('RGB',(ims[0].width*len(ims),ims[0].height))
for i,im in enumerate(ims): sh.paste(im,(i*im.width,0))
sh.save('sheet/A-fix5.png'); print('rendered')
