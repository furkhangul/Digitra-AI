"""Render all authored poses and store editable TID actions in the .blend."""
import bpy, json, math, sys
from pathlib import Path
from mathutils import Vector, Quaternion, Euler
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'output/models'
bpy.ops.wm.open_mainfile(filepath=str(OUT/'digitra-tid-hands.blend'))
letters=json.loads((ROOT/'tmp/tid/poses.json').read_text(encoding='utf-8'))
spec=json.loads((ROOT/'apps/web/src/components/tid/rig-spec.json').read_text())
def xyz(v): return Vector((v[0],-v[2],v[1]))
right=bpy.data.objects['Right_hand']
left=bpy.data.objects['Left_hand']
right.animation_data_clear()
left.animation_data_clear()
for action in list(bpy.data.actions):
    if action.name.startswith('TID_'): bpy.data.actions.remove(action)
for rig in [right,left]:
    rig.rotation_mode='QUATERNION'
for mat,color in [(bpy.data.materials['Warm porcelain'],(.870,.727,.567,1))]:
    mat.diffuse_color=color
left_mat=bpy.data.materials['Warm porcelain'].copy()
left_mat.name='Supporting hand violet'
left_mat.diffuse_color=(.115,.034,.564,1)
left_mat.node_tree.nodes.get('Principled BSDF').inputs['Base Color'].default_value=left_mat.diffuse_color
bpy.data.objects['Left_hand_surface'].data.materials[0]=left_mat

def pose(rig,h):
    # glTF Euler XYZ matches Three's intrinsic XYZ quaternion.
    x,y,z=[math.radians(v)/2 for v in h['rotation']]
    c1,c2,c3=math.cos(x),math.cos(y),math.cos(z)
    s1,s2,s3=math.sin(x),math.sin(y),math.sin(z)
    q=(s1*c2*c3+c1*s2*s3,c1*s2*c3-s1*c2*s3,c1*c2*s3+s1*s2*c3,c1*c2*c3-s1*s2*s3)
    rig.rotation_quaternion=Quaternion((q[3],q[0],-q[2],q[1]))
    rig.location=xyz(h['position'])
    rig.hide_render=not h['visible']
    for child in rig.children: child.hide_render=not h['visible']
    for f in spec:
        shape=h['shape'][f]
        axis=Vector(spec[f]['direction']).cross(Vector((0,0,1))).normalized()
        for j in range(3):
            pb=rig.pose.bones[f'{f}_{j}']
            rest=pb.bone.matrix_local.to_quaternion()
            rotation=Quaternion(xyz(axis),math.radians(shape['bend'][j]))
            if j==0:
                rotation=Quaternion(xyz((0,0,1)),math.radians(shape['spread'])) @ Quaternion(xyz((0,1,0)),math.radians(shape['oppose'])) @ rotation
            pb.rotation_mode='QUATERNION'
            pb.rotation_quaternion=rest.inverted() @ rotation @ rest

scene=bpy.context.scene
scene.render.engine='CYCLES'
scene.cycles.samples=16
scene.cycles.use_denoising=True
scene.world.use_nodes=True
scene.world.node_tree.nodes.get('Background').inputs[0].default_value=(.12,.15,.23,1)
scene.world.node_tree.nodes.get('Background').inputs[1].default_value=.45
for location,energy,size,color in [((-3,5,6),650,5,(1,1,1)),((4,1,2),350,4,(.72,.66,1)),((0,-3,3),90,3,(.8,.9,1))]:
    bpy.ops.object.light_add(type='AREA',location=xyz(location))
    light=bpy.context.object
    light.data.energy=energy
    light.data.shape='DISK'
    light.data.size=size
    light.data.color=color
    light.rotation_euler=(xyz((0,.5,0))-light.location).to_track_quat('-Z','Y').to_euler()
scene.display.shading.light='STUDIO'
scene.display.shading.studiolight_rotate_z=.4
scene.display.shading.color_type='MATERIAL'
scene.display.shading.show_shadows=True
scene.display.shading.show_cavity=True
scene.display.shading.cavity_type='BOTH'
scene.display.shading.curvature_ridge_factor=1.2
scene.display.shading.curvature_valley_factor=.7
scene.display.shading.background_type='WORLD'
scene.world.color=(.055,.05,.075)
scene.render.resolution_x=480
scene.render.resolution_y=400
scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG'
bpy.ops.object.camera_add(location=xyz((0,.1,7.3)))
camera=bpy.context.object
camera.name='TID Review Camera'
camera.rotation_euler=(xyz((0,.1,0))-camera.location).to_track_quat('-Z','Y').to_euler()
camera.data.type='ORTHO'
camera.data.ortho_scale=5.2
scene.camera=camera
(ROOT/'tmp/tid/renders').mkdir(exist_ok=True)
for i,l in enumerate(letters):
    for rig,side in [(right,'right'),(left,'left')]:
        rig.animation_data_clear()
        pose(rig,l['pose'][side])
        rig.keyframe_insert(data_path='location',frame=1)
        rig.keyframe_insert(data_path='rotation_quaternion',frame=1)
        for pb in rig.pose.bones:
            if pb.name.endswith(('_0','_1','_2')):
                pb.keyframe_insert(data_path='rotation_quaternion',frame=1)
        action=rig.animation_data.action
        action.name=f'TID_{i+1:02d}_{l["ch"]}_{side}'
        action.use_fake_user=True
    scene.render.filepath=str(ROOT/f'tmp/tid/renders/{i+1:02d}.png')
    if '--quick' not in sys.argv or i in [1,2,7,8,14,17,18,25]:
        bpy.ops.render.render(write_still=True)
for rig,side in [(right,'right'),(left,'left')]:
    rig.animation_data_clear()
    pose(rig,letters[0]['pose'][side])
scene.frame_end=65
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'digitra-tid-hands.blend'))
print('Rendered all 29 signs; saved per-hand TID pose actions.')
