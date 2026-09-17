import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, letters as LT
data=json.load(open(r"C:/Users/Furkan/Desktop/Digitra AI/tmp/tid/motion-frames.json",encoding='utf-8'))
requested=set(sys.argv[1:])
rows=[]
for L in data['letters']:
    if requested and L['ch'] not in requested: continue
    worst, at = 1e9, None
    for f in L['frames'][::4]:
        R=LT.Hand(f['frame']['right'],'right'); Lh=LT.Hand(f['frame']['left'],'left')
        if not (R.visible and Lh.visible): continue
        sr=R.surface_points(n_per_seg=35,seed=401); sl=Lh.surface_points(n_per_seg=35,seed=402)
        w=float(min(Lh.sdf(sr).min(), R.sdf(sl).min()))
        if w<worst: worst, at = w, f['t']
    rows.append({'letter':L['ch'],'from':L['from'],'worstCollision':round(worst,4),'atSeconds':at})
    print(f"{L['ch']:3s} worst={worst:+.4f} at {at}s", flush=True)
bad=[r for r in rows if r['worstCollision']<-0.02]
print('\ncollision <-0.02:', ', '.join(f"{r['letter']}({r['worstCollision']:+.3f}@{r['atSeconds']}s)" for r in bad) or 'NONE')
print('worst overall %.4f'%min(r['worstCollision'] for r in rows))
json.dump(rows, open('motion-check.json','w',encoding='utf-8'), indent=1, ensure_ascii=False)
