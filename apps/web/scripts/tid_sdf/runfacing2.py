import sys; sys.path.insert(0,'.')
import numpy as np, facing as FA
JOBS = {
 'M': ([0,0,180],[0,0,180],'index_tip','index_tip',[-.191,0,0]),
 'N': ([0,0,180],[0,0,154],'index_tip','index_tip',[-.196,.023,-.015]),
 'S': ([0,65,-10],[0,65,170],'index_tip','thumb_tip',[0,0,.08]),
 'Z': ([0,0,-90],[0,0,90],'middle_tip','index_tip',[0,0,.09]),
}
for ch,(rr,lr,ra,la,g) in JOBS.items():
    b=FA.search(ch,rr,lr,ra,la,g,swap=True,
                ry_choices=(-60,-45,-30,0),lry_choices=(0,30,45,60))
    if b is None:
        print(f"{ch}: still none with roles swapped"); continue
    score,RR,LR,m=b
    print(f"{ch}: SWAP rr={RR} lr={LR} gap={np.round(m['gap'],3).tolist()} "
          f"contact={m['contact']:+.4f} worst={m['worst']:+.4f} palm.x R{m['rx']:+.2f} L{m['lx']:+.2f}")
