import bpy,json
from pathlib import Path
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
root=Path(__file__).resolve().parents[3]
bpy.ops.import_scene.gltf(filepath=str(root/'tmp/tid/source/webxr-right.glb'))
for ob in bpy.context.scene.objects:
    print('OBJECT',ob.name,ob.type,list(ob.location),list(ob.scale))
    if ob.type=='ARMATURE':
        for bone in ob.data.bones:print('BONE',bone.name,list(bone.head_local),list(bone.tail_local))
    if ob.type=='MESH':print('MESH',len(ob.data.vertices),len(ob.data.polygons),[g.name for g in ob.vertex_groups])
