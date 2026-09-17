"""Measure inter-hand triangle intersections and minimally release penetration.

This validates authored poses, not linguistic accuracy or every animation frame.
"""
import bpy,json,math
from pathlib import Path
from mathutils import Vector,Quaternion
from mathutils.bvhtree import BVHTree
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'output/models'
bpy.ops.wm.open_mainfile(filepath=str(OUT/'digitra-tid-hands.blend'))
letters=json.loads((ROOT/'tmp/tid/poses.json').read_text(encoding='utf-8'))
spec=json.loads((ROOT/'apps/web/src/components/tid/rig-spec.json').read_text())
offset_path=ROOT/'apps/web/src/components/tid/surface-offsets.json'
offsets=json.loads(offset_path.read_text())
def xyz(v):return Vector((v[0],-v[2],v[1]))
rigs={side:bpy.data.objects[name] for side,name in [('right','Right_hand'),('left','Left_hand')]}
def pose(rig,h):
    rig.animation_data_clear()
    x,y,z=[math.radians(v)/2 for v in h['rotation']]
    c1,c2,c3=math.cos(x),math.cos(y),math.cos(z)
    s1,s2,s3=math.sin(x),math.sin(y),math.sin(z)
    q=(s1*c2*c3+c1*s2*s3,c1*s2*c3-s1*c2*s3,c1*c2*s3+s1*s2*c3,c1*c2*c3-s1*s2*s3)
    rig.rotation_mode='QUATERNION';rig.rotation_quaternion=Quaternion((q[3],q[0],-q[2],q[1]))
    rig.location=xyz(h['position'])
    for f in spec:
        shape=h['shape'][f]
        axis=Vector(spec[f]['direction']).cross(Vector((0,0,1))).normalized()
        for j in range(3):
            pb=rig.pose.bones[f'{f}_{j}'];rest=pb.bone.matrix_local.to_quaternion()
            rot=Quaternion(xyz(axis),math.radians(shape['bend'][j]))
            if j==0:rot=Quaternion(xyz((0,0,1)),math.radians(shape['spread']))@Quaternion(xyz((0,1,0)),math.radians(shape['oppose']))@rot
            pb.rotation_mode='QUATERNION';pb.rotation_quaternion=rest.inverted()@rot@rest
def surface(side):
    ob=next(c for c in rigs[side].children if c.type=='MESH')
    ev=ob.evaluated_get(bpy.context.evaluated_depsgraph_get());mesh=ev.to_mesh()
    mesh.calc_loop_triangles()
    vertices=[ev.matrix_world@v.co for v in mesh.vertices]
    faces=[tuple(t.vertices) for t in mesh.loop_triangles]
    ev.to_mesh_clear()
    return vertices,faces
report=[]
for letter in letters:
    if not letter['pose']['left']['visible']:continue
    for side,rig in rigs.items():pose(rig,letter['pose'][side])
    bpy.context.view_layer.update()
    rv,rf=surface('right');lv,lf=surface('left')
    left=BVHTree.FromPolygons(lv,lf,all_triangles=True)
    def hits(delta):return len(BVHTree.FromPolygons([v+delta for v in rv],rf,all_triangles=True).overlap(left))
    before=hits(Vector())
    distance=0.
    direction=xyz((0,1,0) if letter['ch'] in ['B','G','Ğ','Ö','İ','Ç'] else (0,0,1))
    if before:
        best=None
        for axis in [direction]+[xyz(v) for v in [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]]:
            hi=.025
            while hits(axis*hi) and hi<2.:hi*=1.5
            lo=0.
            for _ in range(8):
                mid=(lo+hi)/2
                if hits(axis*mid):lo=mid
                else:hi=mid
            if best is None or hi<best[0]-.005:best=(hi,axis)
        distance=best[0]+.008
        direction=best[1]
    delta=direction*distance
    after=hits(delta)
    assert after==0,(letter['ch'],after)
    gltf=[delta.x,delta.z,-delta.y]
    old=offsets.get(letter['ch'],[0,0,0])
    offsets[letter['ch']]=[round(a+b,6) for a,b in zip(old,gltf)]
    report.append({'letter':letter['ch'],'intersectingTrianglePairsBefore':before,'after':after,'releaseDistance':round(distance,4)})
    print(report[-1],flush=True)
offset_path.write_text(json.dumps(offsets,indent=2,ensure_ascii=False),encoding='utf-8')
(OUT/'surface-contact-report.json').write_text(json.dumps({'scope':'Static authored poses, inter-hand surface crossings only','poses':report},indent=2,ensure_ascii=False),encoding='utf-8')
