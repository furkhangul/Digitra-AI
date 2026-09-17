import sys; sys.path.insert(0,'.')
import numpy as np, facing as FA
# letter: (right rot, left rot, right anchor, left anchor, gap)  as authored today
JOBS = {
 'İ': ([0,0,180],[0,0,0],'index_tip','index_tip',[0,.22,.04]),
 'M': ([0,0,180],[0,0,180],'index_tip','index_tip',[-.191,0,0]),
 'N': ([0,0,180],[0,0,154],'index_tip','index_tip',[-.196,.023,-.015]),
 'S': ([0,65,-10],[0,65,170],'thumb_tip','index_tip',[0,0,.08]),
 'Ş': ([0,65,-10],[0,0,20],'thumb_tip','index_tip',[0,0,.08]),
 'Ü': ([0,-65,85],[0,-20,0],'thumb_tip','index_tip',[0,.08,.05]),
 'Z': ([0,0,-90],[0,0,90],'index_tip','middle_tip',[0,0,.09]),
}
for ch,(rr,lr,ra,la,g) in JOBS.items():
    b=FA.search(ch,rr,lr,ra,la,g)
    if b is None:
        print(f"{ch}: no orientation satisfied every rule"); continue
    score,RR,LR,m=b
    print(f"{ch}: rr={RR} lr={LR} gap={np.round(m['gap'],3).tolist()} "
          f"contact={m['contact']:+.4f} worst={m['worst']:+.4f} palm.x R{m['rx']:+.2f} L{m['lx']:+.2f}")
