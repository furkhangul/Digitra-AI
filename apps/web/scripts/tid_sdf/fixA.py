import sys, copy; sys.path.insert(0,'.')
import numpy as np, joined as J, letters as LT
from scipy.optimize import minimize
sr=J.shape_of('A','right'); base=J.shape_of('A','left')
def seg_seg(p1,q1,p2,q2):
    d1,d2=q1-p1,q2-p2; r=p1-p2
    a,e,f=d1@d1,d2@d2,d2@r; c=d1@r; b=d1@d2
    den=a*e-b*b
    s=np.clip((b*f-c*e)/den,0,1) if den>1e-9 else 0.0
    t=np.clip((b*s+f)/e,0,1); s=np.clip((b*t-c)/a,0,1)
    return np.linalg.norm((p1+d1*s)-(p2+d2*t)), s, t
def setup(spread,gap):
    sl=copy.deepcopy(base); sl['index']['spread']=spread; sl['middle']['spread']=-spread
    p=J.joined(sr,sl,[0,0,-90],[0,0,180],'index_2','index_1',list(gap))
    R,L=J.hands_from(p)
    seg=lambda h,s,f:(h.M@J.local_anchor(s,f+'_0')+h.T, h.M@J.local_anchor(s,f+'_tip')+h.T)
    return sl,R,L,seg(R,sr,'index'),seg(L,sl,'index'),seg(L,sl,'middle')
RP_=LT.RIG['index']['radius']; RI=RP_; RM=LT.RIG['middle']['radius']
def measure(spread,gap):
    sl,R,L,P,FI,FM=setup(spread,gap)
    dI,_,tI=seg_seg(*P,*FI); dM,_,tM=seg_seg(*P,*FM)
    return dI-(RP_+RI), dM-(RP_+RM), tI, tM, (sl,R,L)
def cost(x):
    gI,gM,tI,tM,_=measure(x[0],x[1:4])
    return (gI-0.01)**2*300+(gM-0.01)**2*300+(tI-0.5)**2*4+(tM-0.5)**2*4
best=None
for s in range(8):
    rng=np.random.default_rng(s)
    x0=np.concatenate([[rng.uniform(12,26)],rng.uniform(-.5,.5,3)])
    r=minimize(cost,x0,method='Nelder-Mead',options={'xatol':2e-4,'fatol':1e-9,'maxiter':1200})
    if best is None or r.fun<best.fun: best=r
spread=round(float(best.x[0]),1); gap=[round(float(v),3) for v in best.x[1:4]]
gI,gM,tI,tM,(sl,R,L)=measure(spread,gap)
print('spread +/-%.1f  gap %s'%(spread,gap))
print('clearance fork-index %+.4f  fork-middle %+.4f'%(gI,gM))
print('crossing at %.2f / %.2f along the fingers'%(tI,tM))
srf=R.surface_points(n_per_seg=85,seed=11); slf=L.surface_points(n_per_seg=85,seed=12)
print('worst surface overlap %+.4f'%float(min(L.sdf(srf).min(), R.sdf(slf).min())))
import render_pair as RP
from PIL import Image, ImageDraw
ims=[]
for v in [(0,0,0),(0,30,0),(0,-30,0)]:
    f=f'sheet/A2-{v[1]}.png'; RP.render((R,L),f,W=340,H=340,view=v)
    im=Image.open(f); im.load(); im=im.convert('RGB')
    ImageDraw.Draw(im).text((6,4),f'view {v[1]}',fill=(255,230,150)); ims.append(im)
sh=Image.new('RGB',(ims[0].width*len(ims),ims[0].height))
for i,im in enumerate(ims): sh.paste(im,(i*im.width,0))
sh.save('sheet/A-fix2.png'); print('rendered')
