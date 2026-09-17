"""Adapt the MIT WebXR hand mesh and authored skin weights to Digitra's rig.

The previous procedural builder is retained separately. Joint locations and
pose semantics remain defined by rig-spec.json; no voxel remeshing is used.
"""
import bpy, json, math, bmesh
from pathlib import Path
from mathutils import Vector

ROOT=Path(__file__).resolve().parents[3]
WEB=ROOT/'apps/web'
OUT=ROOT/'output/models'
spec=json.loads((WEB/'src/components/tid/rig-spec.json').read_text())
def xyz(p):return Vector((p[0],-p[2],p[1]))
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
bpy.ops.import_scene.gltf(filepath=str(WEB/'assets/hand-source/right.glb'))
source=next(o for o in bpy.context.scene.objects if o.type=='ARMATURE')
skin=bpy.data.objects['r_handMeshNode']
origin=source.data.bones['wrist'].head_local.copy()
def canonical(p):
    p=p-origin
    return Vector((-p.y*12-.085,-p.z*12-.20,p.x*12))
heads={b.name:canonical(b.head_local) for b in source.data.bones}
mapping={'wrist':'wrist'}
transforms={}
for finger,f in spec.items():
    prefix=finger if finger=='thumb' else finger+'-finger'
    names=([prefix+'-metacarpal',prefix+'-phalanx-proximal',prefix+'-phalanx-distal',prefix+'-tip']
           if finger=='thumb' else [prefix+'-phalanx-proximal',prefix+'-phalanx-intermediate',prefix+'-phalanx-distal',prefix+'-tip'])
    target=[Vector(p) for p in f['points']]
    for j in range(3):
        start,end=heads[names[j]],heads[names[j+1]]
        direction=(end-start).normalized()
        destdir=(target[j+1]-target[j]).normalized()
        rotation=direction.rotation_difference(destdir)
        ratio=(target[j+1]-target[j]).length/(end-start).length
        transforms[names[j]]=(start,target[j],direction,rotation,ratio)
        mapping[names[j]]=f'{finger}_{j}'
    transforms[names[3]]=transforms[names[2]]
    mapping[names[3]]=f'{finger}_2'
    if finger!='thumb':
        meta=prefix+'-metacarpal'
        start=heads[meta]
        end=heads[names[0]]
        direction=(end-start).normalized()
        dest=target[0]-start
        transforms[meta]=(start,start,direction,direction.rotation_difference(dest.normalized()),dest.length/(end-start).length)
        mapping[meta]='wrist'
group_names={g.index:g.name for g in skin.vertex_groups}
rows=[]
for vertex in skin.data.vertices:
    p=canonical(vertex.co)
    position=Vector()
    row={}
    total=sum(g.weight for g in vertex.groups)
    for g in vertex.groups:
        name=group_names[g.group]
        w=g.weight/total
        moved=p.copy()
        if name in transforms:
            start,dest,axis,rotation,ratio=transforms[name]
            offset=p-start
            along=offset.dot(axis)
            # A modest transverse expansion gives the illustrative model
            # softer finger pads while keeping joint and fingertip anchors.
            width=1.38 if not name.endswith('-metacarpal') else 1.06
            offset = axis*along*ratio + (offset-axis*along)*width
            moved=dest+rotation@offset
        position+=moved*w
        joint=mapping.get(name,'wrist')
        row[joint]=row.get(joint,0)+w
    position.z=position.z*1.16
    vertex.co=xyz(position)
    rows.append(row)
skin.parent=None
skin.matrix_world.identity()
skin.modifiers.clear()
skin.vertex_groups.clear()
for name in dict.fromkeys(mapping.values()):skin.vertex_groups.new(name=name)
for i,row in enumerate(rows):
    for name,weight in row.items():skin.vertex_groups[name].add([i],weight,'REPLACE')
for ob in list(bpy.context.scene.objects):
    if ob!=skin:bpy.data.objects.remove(ob,do_unlink=True)

# Weld UV-seam duplicates before subdivision, preserving the artist's topology.
bpy.context.view_layer.objects.active=skin
skin.select_set(True)
bm=bmesh.new()
bm.from_mesh(skin.data)
bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=.00001)
# Remove the wrist tube geometrically, so it stays absent from every angle.
bmesh.ops.bisect_plane(bm,geom=list(bm.verts)+list(bm.edges)+list(bm.faces),
    dist=.000001,plane_co=(0,0,.035),plane_no=(0,0,1),clear_inner=True)
boundary=[e for e in bm.edges if e.is_boundary]
if boundary:bmesh.ops.holes_fill(bm,edges=boundary,sides=0)
bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
bm.to_mesh(skin.data)
bm.free()
if skin.data.has_custom_normals:
    skin.data.normals_split_custom_set([(0,0,0)]*len(skin.data.loops))
# Low resolution, closed surface for real-time inter-hand collision queries.
skin.data.calc_loop_triangles()
proxy={'positions':[[v.co.x,v.co.z,-v.co.y] for v in skin.data.vertices],
       'triangles':[list(t.vertices) for t in skin.data.loop_triangles],
       'weights':[[[skin.vertex_groups[g.group].name,g.weight] for g in sorted(v.groups,key=lambda g:g.weight,reverse=True)[:4]] for v in skin.data.vertices]}
(WEB/'src/components/tid/collision-surface.json').write_text(json.dumps(proxy,separators=(',',':')))
sub=skin.modifiers.new('Smooth anatomical contours','SUBSURF')
sub.levels=2
bpy.ops.object.modifier_apply(modifier=sub.name)
# Match the four-influence glTF renderer in the editable Blender source too.
rows=[sorted([(g.group,g.weight) for g in v.groups],key=lambda p:p[1],reverse=True)[:4] for v in skin.data.vertices]
for group in skin.vertex_groups:group.remove(list(range(len(skin.data.vertices))))
for i,row in enumerate(rows):
    total=sum(w for _,w in row)
    for g,w in row:skin.vertex_groups[g].add([i],w/total,'REPLACE')
for polygon in skin.data.polygons:polygon.use_smooth=True
skin.name='Digitra_AnatomicalHand'
mat=bpy.data.materials.new('Warm porcelain')
mat.use_nodes=True
mat.diffuse_color=(.871,.738,.565,1)
bsdf=mat.node_tree.nodes.get('Principled BSDF')
bsdf.inputs['Base Color'].default_value=mat.diffuse_color
bsdf.inputs['Roughness'].default_value=.32
bsdf.inputs['Coat Weight'].default_value=.22
skin.data.materials.clear()
skin.data.materials.append(mat)
arm=bpy.data.armatures.new('Digitra_TID_Rig')
rig=bpy.data.objects.new('Digitra_TID_Rig',arm)
bpy.context.collection.objects.link(rig)
bpy.context.view_layer.objects.active=rig
bpy.ops.object.mode_set(mode='EDIT')
wrist=arm.edit_bones.new('wrist')
wrist.head=xyz((0,0,0));wrist.tail=xyz((0,.5,0))
for f,data in spec.items():
    parent=wrist
    for j in range(3):
        b=arm.edit_bones.new(f'{f}_{j}')
        b.head=xyz(data['points'][j]);b.tail=xyz(data['points'][j+1])
        b.parent=parent;b.use_connect=j>0
        parent=b
    b=arm.edit_bones.new(f'{f}_tip')
    b.head=xyz(data['points'][3]);b.tail=b.head+xyz(data['direction'])*.045
    b.parent=parent;b.use_connect=True;b.use_deform=False
bpy.ops.object.mode_set(mode='OBJECT')
mod=skin.modifiers.new('Articulated skin','ARMATURE');mod.object=rig
skin.parent=rig
rig.show_in_front=True
rig['source']='WebXR Input Profiles generic-hand, MIT. Adapted geometry and rig for Digitra.'
bpy.ops.object.select_all(action='DESELECT')
skin.select_set(True);rig.select_set(True)
path=WEB/'public/models/tid-hand.glb'
bpy.ops.export_scene.gltf(filepath=str(path),export_format='GLB',use_selection=True,export_animations=False,export_yup=True,export_extras=True,
    export_copyright='WebXR Input Profiles assets, MIT License, Copyright (c) 2019 Amazon. Adapted for Digitra; see THIRD_PARTY_HAND_LICENSE.md.')
(OUT/'tid-hand.glb').write_bytes(path.read_bytes())
rig.name='Right_hand';rig.location.x=-.85
left=rig.copy();left.data=rig.data.copy();left.name='Left_hand'
bpy.context.collection.objects.link(left)
left.location.x=.85;left.scale.x=-1
left_skin=skin.copy();left_skin.data=skin.data.copy()
bpy.context.collection.objects.link(left_skin)
left_skin.parent=left;left_skin.modifiers['Articulated skin'].object=left
left_skin.name='Left_hand_surface'
bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'digitra-tid-hands.blend'))
license=(WEB/'assets/hand-source/LICENSE.md').read_text()
for folder in [OUT,WEB/'public/models']:
    (folder/'THIRD_PARTY_HAND_LICENSE.md').write_text('Source: https://github.com/immersive-web/webxr-input-profiles/tree/main/packages/assets/profiles/generic-hand\n\nModified for Digitra: rest pose adaptation, subdivision, TID rig and animations.\n\n'+license)
print(json.dumps({'vertices':len(skin.data.vertices),'bones':len(arm.bones),'bytes':path.stat().st_size}))
