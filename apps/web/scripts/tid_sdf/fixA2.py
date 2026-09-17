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
    A=p1+d1*s; B=p2+d2*t
    return np.linalg.norm(A-B), s, t, A, B
SPREAD=18.0
def setup(gap):
    sl=copy.deepcopy(base); sl['index']['spread']=SPREAD; sl['middle']['spread']=-SPREAD
    p=J.joined(sr,sl,[0,0,-90],[0,0,180],'index_2','index_1',list(gap))
    R,L=J.hands_from(p)
    seg=lambda h,s,f:(h.M@J.local_anchor(s,f+'_0')+h.T, h.M@J.local_anchor(s,f+'_tip')+h.T)
    return sl,R,L,seg(R,sr,'index'),seg(L,sl,'index'),seg(L,sl,'middle')
RP_=LT.RIG['index']['radius']; RI=RP_; RM=LT.RIG['middle']['radius']
def measure(gap):
    sl,R,L,P,FI,FM=setup(gap)
    dI,sI,tI,AI,BI=seg_seg(*P,*FI); dM,sM,tM,AM,BM=seg_seg(*P,*FM)
    frontI=AI[2]-BI[2]; frontM=AM[2]-BM[2]      # crossbar should sit in front (+z)
    return dict(gI=dI-(RP_+RI), gM=dM-(RP_+RM), tI=tI, tM=tM,
                frontI=frontI, frontM=frontM, sI=sI, sM=sM, hands=(sl,R,L))
def cost(g):
    m=measure(g)
    c  = (m['gI']-0.005)**2*400 + (m['gM']-0.005)**2*400          # rest against both legs
    c += (m['tI']-0.46)**2*6 + (m['tM']-0.46)**2*6                # cross about mid-way down
    c += max(0.0, 0.06-m['frontI'])**2*200 + max(0.0, 0.06-m['frontM'])**2*200  # stay in front
    c += max(0.0, 0.55-m['sM'])**2*8                               # tip clears the far leg
    return c
best=None
for s in range(10):
    rng=np.random.default_rng(s)
    r=minimize(cost,rng.uniform(-.5,.5,3),method='Nelder-Mead',options={'xatol':2e-4,'fatol':1e-10,'maxiter':1500})
    if best is None or r.fun<best.fun: best=r
gap=[round(float(v),3) for v in best.x]
m=measure(gap); sl,R,L=m['hands']
print('gap',gap)
print('clearance  left leg %+.4f   right leg %+.4f'%(m['gI'],m['gM']))
print('crosses at %.2f / %.2f down the legs; in front by %.3f / %.3f'%(m['tI'],m['tM'],m['frontI'],m['frontM']))
print('crossbar reaches %.2f of its own length at the far leg'%m['sM'])
srf=R.surface_points(n_per_seg=85,seed=11); slf=L.surface_points(n_per_seg=85,seed=12)
print('worst surface overlap %+.4f'%float(min(L.sdf(srf).min(), R.sdf(slf).min())))
import render_pair as RP
from PIL import Image, ImageDraw
ims=[]
for v in [(0,0,0),(0,28,0),(0,-28,0)]:
    f=f'sheet/A3-{v[1]}.png'; RP.render((R,L),f,W=340,H=340,view=v)
    im=Image.open(f); im.load(); im=im.convert('RGB')
    ImageDraw.Draw(im).text((6,4),f'view {v[1]}',fill=(255,230,150)); ims.append(im)
sh=Image.new('RGB',(ims[0].width*len(ims),ims[0].height))
for i,im in enumerate(ims): sh.paste(im,(i*im.width,0))
sh.save('sheet/A-fix3.png'); print('rendered')
