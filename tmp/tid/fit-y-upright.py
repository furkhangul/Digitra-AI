import sys, copy, json
import numpy as np
sys.path.insert(0, 'apps/web/scripts/tid_sdf')
import letters as lt

letter = next(a for ch, a in lt.letters() if ch == 'Y')
frame = letter['frame']
right = lt.Hand(frame['right'], 'right')
rp = right.surface_points(n_per_seg=65, seed=353)
options = []
for pitch in [0, -15, -25]:
    f = copy.deepcopy(frame['left'])
    f['rotation'] = lt.axis_quat([1,0,0], pitch).tolist()
    f['position'] = [0,0,0]
    offset = lt.Hand(f, 'left').caps[1][-1][1]
    for x in [-.30, -.2275]:
        for z in [.72, .76, .80, .84, .88]:
            f['position'] = (np.array(frame['right']['position']) + [x, .94, z] - offset).tolist()
            left = lt.Hand(f, 'left')
            gap = float(left.sdf(rp).min())
            if True:
                score = abs(gap-.004) + abs(pitch)*.0004 + abs(x+.2275)*.04
                options.append((score,pitch,x,z,gap,copy.deepcopy(f)))
options.sort(key=lambda v:v[0])
for _,pitch,x,z,gap,f in options[:4]:
    left=lt.Hand(f,'left')
    sp=left.surface_points(n_per_seg=65,seed=354)
    print(json.dumps(dict(pitch=pitch,x=x,y=.94,z=z,gap=gap,other=float(right.sdf(sp).min()))))
