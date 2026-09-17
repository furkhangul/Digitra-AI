"""Bake the exact web player's two-hand animation into the editable Blender asset."""
import bpy,json,math
from pathlib import Path
from mathutils import Vector,Quaternion
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'output/models'
bpy.ops.wm.open_mainfile(filepath=str(OUT/'digitra-tid-hands.blend'))
timeline=json.loads((ROOT/'tmp/tid/timeline.json').read_text())
spec=json.loads((ROOT/'apps/web/src/components/tid/rig-spec.json').read_text())
fingers=['thumb','index','middle','ring','pinky']
def xyz(v):return Vector((v[0],-v[2],v[1]))
rigs={'right':bpy.data.objects['Right_hand'],'left':bpy.data.objects['Left_hand']}
for action in list(bpy.data.actions):
    if action.name.startswith('TID_Alphabet_'):bpy.data.actions.remove(action)
bindings={}
for side,rig in rigs.items():
    rig.animation_data_clear()
    rig.hide_render=False
    for ob in rig.children:ob.hide_render=False
    bindings[side]=[]
    for f in fingers:
        axis=Vector(spec[f]['direction']).cross(Vector((0,0,1))).normalized()
        for j in range(3):
            pb=rig.pose.bones[f'{f}_{j}']
            rest=pb.bone.matrix_local.to_quaternion()
            bindings[side].append((pb,rest,xyz(axis),fingers.index(f)*5,j))
scene=bpy.context.scene
scene.render.fps=timeline['fps']
scene.frame_start=1
scene.frame_end=len(timeline['samples'])
scene.timeline_markers.clear()
previous=None
for i,sample in enumerate(timeline['samples'],1):
    if previous!=sample['letter']:
        scene.timeline_markers.new(sample['letter'],frame=i)
        previous=sample['letter']
    for side,rig in rigs.items():
        h=sample[side]
        x,y,z,w=h['rotation']
        rig.location=xyz(h['position'])
        rig.rotation_mode='QUATERNION'
        rig.rotation_quaternion=Quaternion((w,x,-z,y))
        presence=max(.0001,h['presence'])
        rig.scale=((-presence if side=='left' else presence),presence,presence)
        for field in ['location','rotation_quaternion','scale']:rig.keyframe_insert(data_path=field,frame=i)
        for pb,rest,axis,offset,j in bindings[side]:
            rotation=Quaternion(axis,math.radians(h['angles'][offset+j]))
            if j==0:
                rotation=Quaternion(xyz((0,0,1)),math.radians(h['angles'][offset+3])) @ Quaternion(xyz((0,1,0)),math.radians(h['angles'][offset+4])) @ rotation
            pb.rotation_mode='QUATERNION'
            pb.rotation_quaternion=rest.inverted() @ rotation @ rest
            pb.keyframe_insert(data_path='rotation_quaternion',frame=i)
for side,rig in rigs.items():
    action=rig.animation_data.action
    action.name=f'TID_Alphabet_{side}'
    action.use_fake_user=True
    # Samples already contain the easing. Linear keys avoid extra overshoot.
    for layer in action.layers:
        for strip in layer.strips:
            for bag in strip.channelbags:
                for curve in bag.fcurves:
                    for key in curve.keyframe_points:key.interpolation='LINEAR'
scene.frame_set(1)
scene['TID_source']='Dikyuva, Makaroglu, Arik (2015), TID Dilbilgisi Kitabi p.91; linguistic review pending.'
scene['Playback']='Space: play two-hand alphabet. Timeline markers select letters. Separate TID_XX actions store editable static poses.'
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'digitra-tid-hands.blend'),compress=True)
bpy.ops.object.select_all(action='DESELECT')
for rig in rigs.values():
    rig.select_set(True)
    for ob in rig.children:ob.select_set(True)
bpy.context.view_layer.objects.active=rigs['right']
bpy.ops.export_scene.gltf(filepath=str(OUT/'tid-alphabet-animated.glb'),export_format='GLB',use_selection=True,
    export_animations=True,export_animation_mode='SCENE',export_frame_range=True,export_frame_step=2,
    export_force_sampling=True,export_yup=True,export_extras=True)
import runpy
runpy.run_path(str(Path(__file__).with_name('merge-tid-clips.py')),run_name='__main__')
print(json.dumps({'frames':scene.frame_end,'fps':scene.render.fps,'actions':len(bpy.data.actions),'animated_glb_bytes':(OUT/'tid-alphabet-animated.glb').stat().st_size}))
