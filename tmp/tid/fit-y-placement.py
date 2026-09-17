import sys, copy, json
import numpy as np
sys.path.insert(0, 'apps/web/scripts/tid_sdf')
import letters as lt

letter = next(a for ch, a in lt.letters() if ch == 'Y')
frame = letter['frame']
right = lt.Hand(frame['right'], 'right')
rp = right.surface_points(n_per_seg=70, seed=353)
options = []
for pitch in [-25, -10, 0]:
    for roll in [-35, -45, -55, -65]:
        f = copy.deepcopy(frame['left'])
        f['rotation'] = lt.qmul(lt.axis_quat([1,0,0], pitch), lt.axis_quat([0,0,1], roll)).tolist()
        f['position'] = [0,0,0]
        offset = lt.Hand(f, 'left').caps[1][-1][1]
        for x in [-.30, -.25, -.22]:
            for z in [.22, .25, .28]:
                target = np.array(frame['right']['position']) + [x, .94, z]
                f['position'] = (target - offset).tolist()
                left = lt.Hand(f, 'left')
                gap = float(left.sdf(rp).min())
                if -.005 < gap < .04:
                    score = abs(gap-.004) + abs(roll+45)*.0001 + abs(x+.2275)*.01
                    options.append((score,pitch,roll,x,z,gap,copy.deepcopy(f)))
options.sort(key=lambda v: v[0])
for _,pitch,roll,x,z,gap,f in options[:6]:
    left=lt.Hand(f,'left')
    sp=left.surface_points(n_per_seg=70,seed=354)
    other=float(right.sdf(sp).min())
    print(json.dumps(dict(pitch=pitch,roll=roll,x=x,y=.94,z=z,gap=gap,other=other)))
