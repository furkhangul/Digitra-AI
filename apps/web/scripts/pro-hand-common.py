import bpy, bmesh, json, math
from pathlib import Path
from mathutils import Vector, Quaternion
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'output/hand-professional'
OUT.mkdir(exist_ok=True,parents=True)
SPEC=json.loads((ROOT/'apps/web/src/components/tid/rig-spec.json').read_text())
def xyz(v): return Vector((v[0],-v[2],v[1]))
def pose(rig,h):
    rig.animation_data_clear()
    x,y,z=[math.radians(v)/2 for v in h.get('rotation',[0,0,0])]
    c1,c2,c3=math.cos(x),math.cos(y),math.cos(z);s1,s2,s3=math.sin(x),math.sin(y),math.sin(z)
    q=(s1*c2*c3+c1*s2*s3,c1*s2*c3-s1*c2*s3,c1*c2*s3+s1*s2*c3,c1*c2*c3-s1*s2*s3)
    rig.rotation_mode='QUATERNION';rig.rotation_quaternion=Quaternion((q[3],q[0],-q[2],q[1]))
    rig.location=xyz(h.get('position',[0,0,0]))
    rig.scale=(-1,1,1) if rig.name.startswith('Left') else (1,1,1)
    for f in SPEC:
        shape=h['shape'][f]; axis=Vector(SPEC[f]['direction']).cross(Vector((0,0,1))).normalized()
        for j in range(3):
            pb=rig.pose.bones[f'{f}_{j}'];rest=pb.bone.matrix_local.to_quaternion()
            rot=Quaternion(xyz(axis),math.radians(shape['bend'][j]))
            if j==0:rot=Quaternion(xyz((0,0,1)),math.radians(shape['spread']))@Quaternion(xyz((0,1,0)),math.radians(shape['oppose']))@rot
            pb.rotation_mode='QUATERNION';pb.rotation_quaternion=rest.inverted()@rot@rest
    bpy.context.view_layer.update()
def setup_render(width=600,height=600):
    scene=bpy.context.scene
    for ob in list(scene.objects):
        if ob.type in ['LIGHT','CAMERA']:bpy.data.objects.remove(ob,do_unlink=True)
    scene.render.engine='CYCLES';scene.cycles.samples=16;scene.cycles.use_denoising=True
    scene.world.use_nodes=True
    scene.world.node_tree.nodes['Background'].inputs[0].default_value=(.16,.12,.24,1)
    scene.world.node_tree.nodes['Background'].inputs[1].default_value=.5
    scene.view_settings.view_transform='AgX'
    for location,power,size in [((-3,5,6),550,5),((4,2,4),250,4),((1,4,-4),400,3)]:
        bpy.ops.object.light_add(type='AREA',location=xyz(location));ob=bpy.context.object
        ob.data.energy=power;ob.data.shape='DISK';ob.data.size=size
        ob.rotation_euler=(xyz((0,.7,0))-ob.location).to_track_quat('-Z','Y').to_euler()
    bpy.ops.object.camera_add(location=xyz((0,.8,6)));cam=bpy.context.object
    cam.data.type='ORTHO';cam.data.ortho_scale=2.8;scene.camera=cam
    scene.render.resolution_x=width;scene.render.resolution_y=height;scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG';scene.render.film_transparent=True
    return scene,cam
def camera(cam,angle=0,center=(0,.8,0),scale=2.8):
    rad=math.radians(angle);c=xyz(center)
    cam.location=c+xyz((6*math.sin(rad),.2,6*math.cos(rad)))
    cam.rotation_euler=(c-cam.location).to_track_quat('-Z','Y').to_euler();cam.data.ortho_scale=scale
if __name__=='__main__':
    bpy.ops.wm.open_mainfile(filepath=str(ROOT/'output/models/digitra-tid-hands.blend'))
    report={'objects':[],'actions':[a.name for a in bpy.data.actions]}
    for ob in bpy.context.scene.objects:
        if ob.type=='MESH':
            bm=bmesh.new();bm.from_mesh(ob.data)
            report['objects'].append({'name':ob.name,'vertices':len(bm.verts),'faces':len(bm.faces),'boundaryEdges':sum(e.is_boundary for e in bm.edges),'nonManifoldEdges':sum(not e.is_manifold for e in bm.edges),'weightSumRange':[min(sum(g.weight for g in v.groups) for v in ob.data.vertices),max(sum(g.weight for g in v.groups) for v in ob.data.vertices)],'modifiers':[(m.name,m.type) for m in ob.modifiers]});bm.free()
    (OUT/'baseline-audit.json').write_text(json.dumps(report,indent=2))
    right=bpy.data.objects['Right_hand']
    for ob in bpy.context.scene.objects:
        if ob.type=='MESH' and ob.parent!=right:ob.hide_render=True
    scene,cam=setup_render()
    letters=json.loads((ROOT/'tmp/tid/poses.json').read_text(encoding='utf-8'))
    shapes={'open':{f:{'bend':[0,0,0],'spread':0,'oppose':0} for f in SPEC},'half':{f:{'bend':[35,45,25],'spread':0,'oppose':0} for f in SPEC},'fist':letters[7]['pose']['left']['shape'],'ring':letters[17]['pose']['right']['shape'],'hook':letters[7]['pose']['right']['shape']}
    for name,shape in shapes.items():
        pose(right,{'shape':shape})
        for angle in [0,65,180]:
            camera(cam,angle)
            scene.render.filepath=str(OUT/f'baseline-{name}-{angle}.png');bpy.ops.render.render(write_still=True)
    print(json.dumps(report['objects']),flush=True)
