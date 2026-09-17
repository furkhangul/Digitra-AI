"""Create Digitra's original, independently articulated teaching hand in Blender.

Run with blender --background --python scripts/build-tid-hand.py.
Coordinates in this builder are expressed as glTF x/right, y/up, z/palm-front.
The hand ends at the wrist: no forearm, no cuff. Shapes are lofted from
anatomical proportions (palm length ~1.2 palm widths, middle finger ~0.9 palm
length, four metacarpal heads on a descending arc, scalloped finger webs,
thenar and hypothenar pads, dorsal knuckles, jointed phalanges with a fleshier
palm side than back). No third party mesh, texture or reference illustration is
embedded, and no fingerspelling.xyz asset is used.
"""
import bpy
import json
import math
from pathlib import Path
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[3]
WEB = ROOT / "apps/web"
OUT = ROOT / "output/models"
OUT.mkdir(parents=True, exist_ok=True)

RING = 34
Z = Vector((0, 0, 1))


def xyz(p):
    return Vector((p[0], -p[2], p[1]))


def ramp(table, x):
    """Piecewise linear read of a sorted (position, value) table."""
    if x <= table[0][0]:
        return table[0][1]
    for (x0, v0), (x1, v1) in zip(table, table[1:]):
        if x <= x1:
            return v0 + (v1 - v0) * (x - x0) / (x1 - x0)
    return table[-1][1]


def outline(back_x, front_x, back_z, front_z, power):
    """One closed cross-section: a superellipse with four independent halves."""
    points = []
    for i in range(RING):
        a = 2 * math.pi * i / RING
        u = math.copysign(abs(math.cos(a)) ** (2 / power), math.cos(a))
        v = math.copysign(abs(math.sin(a)) ** (2 / power), math.sin(a))
        points.append((u * (front_x if u >= 0 else back_x), v * (front_z if v >= 0 else back_z)))
    return points


parts = []


def surface(name, rings):
    """Loft closed rings into a capped shell."""
    vertices, faces = [], []
    n = len(rings[0])
    for ring in rings:
        vertices.extend(xyz(p) for p in ring)
    for r in range(len(rings) - 1):
        for i in range(n):
            a, b = r * n + i, r * n + (i + 1) % n
            faces.append((a, a + n, b + n, b))
    faces.append(tuple(reversed(range(n))))
    faces.append(tuple(range((len(rings) - 1) * n, len(rings) * n)))
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    ob = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(ob)
    parts.append(ob)
    return ob


def blob(name, center, scale, tilt=0.):
    """A soft ellipsoid pad, optionally tilted inside the palm plane."""
    bpy.ops.mesh.primitive_uv_sphere_add(segments=28, ring_count=18, location=xyz(center))
    ob = bpy.context.object
    ob.name = name
    ob.scale = (scale[0], scale[2], scale[1])
    ob.rotation_euler = (0, -math.radians(tilt), 0)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
    parts.append(ob)
    return ob


bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

spec = {
    "index":  {"base": [-.345, .905, .005], "lengths": [.415, .275, .215], "radius": .101},
    "middle": {"base": [-.110, .975, .010], "lengths": [.455, .310, .230], "radius": .106},
    "ring":   {"base": [.128, .935, .004], "lengths": [.425, .290, .215], "radius": .100},
    "pinky":  {"base": [.348, .830, -.008], "lengths": [.360, .240, .195], "radius": .087},
    "thumb":  {"base": [-.205, .095, .045], "lengths": [.365, .270, .215], "radius": .127},
}
# Fingers splay very slightly in rest, the way an open hand actually sits.
DIRECTIONS = {"index": (-.048, .999, 0), "ring": (.042, .999, 0),
              "pinky": (.092, .996, 0), "thumb": (-.633, .724, .222)}

# The four metacarpal heads sit on a descending arc, and the web between two
# fingers rides higher than the knuckles themselves.
KNUCKLES = [(-.62, .795), (-.45, .862), (-.345, .905), (-.110, .975),
            (.128, .935), (.348, .815), (.45, .772), (.62, .715)]
WEBS = [(-.62, .018), (-.45, .034), (-.345, .052), (-.228, .150), (-.110, .052),
        (.009, .152), (.128, .050), (.238, .142), (.348, .046), (.45, .030), (.62, .018)]

# s runs 0 at the rounded wrist end to 1 at the knuckle arc.
PALM_BOTTOM, PALM_SPAN = -.30, 1.20
PALM = [
    # s,    -x,   +x,   -z,   +z,    cx,    cz,  power
    (.000, .068, .068, .042, .042, .000, .007, 2.0),
    (.030, .170, .170, .090, .092, .000, .006, 2.1),
    (.070, .232, .230, .117, .122, .000, .005, 2.2),
    (.125, .281, .277, .138, .146, -.004, .003, 2.4),
    (.205, .314, .304, .151, .162, -.006, .000, 2.5),
    (.305, .345, .330, .163, .174, -.008, .000, 2.6),
    (.415, .372, .353, .172, .180, -.010, .000, 2.7),
    (.525, .393, .375, .178, .182, -.010, -.001, 2.7),
    (.635, .410, .393, .180, .178, -.010, -.004, 2.8),
    (.745, .424, .409, .178, .171, -.011, -.007, 2.8),
    (.845, .434, .421, .173, .161, -.013, -.010, 2.9),
    (.925, .438, .427, .166, .150, -.015, -.013, 2.9),
    (1.00, .434, .419, .155, .137, -.016, -.015, 2.9),
]

rings = []
for s, bx, fx, bz, fz, cx, cz, power in PALM:
    arch = min(1., s) ** 2.4
    ring = []
    for x, z in outline(bx, fx, bz, fz, power):
        px = x + cx
        ring.append((px, PALM_BOTTOM + s * PALM_SPAN + arch * (ramp(KNUCKLES, px) - .90), z + cz))
    rings.append(ring)
# Roll the rim over into the scalloped web line instead of capping it flat.
rim = rings[-1]
for u in [.20, .40, .58, .74, .87, .96, 1.]:
    a = u * math.pi / 2
    rings.append([(x * (1 - .17 * u * u), y + ramp(WEBS, x) * math.sin(a),
                   z * max(.05, math.cos(a))) for x, y, z in rim])
surface("Palm", rings)

# Broad, shallow muscle pads: they widen the palm and swell its front a little
# without reading as separate lumps once the shell is fused.
blob("Thenar", (-.230, .285, .020), (.215, .380, .168), tilt=38.)
blob("Hypothenar", (.288, .292, .004), (.146, .375, .158), tilt=-7.)
blob("Wrist_pad", (0, -.12, .008), (.262, .150, .128))

for name, f in spec.items():
    thumb = name == "thumb"
    direction = Vector(DIRECTIONS.get(name, (0, 1, 0))).normalized()
    f["direction"] = list(direction)
    base = Vector(f["base"])
    points = [base.copy()]
    for length in f["lengths"]:
        points.append(points[-1] + direction * length)
    f["points"] = [list(p) for p in points]
    if not thumb:
        blob(f"Knuckle_{name}", (base.x, base.y - .045, base.z - .062),
             (f["radius"] * .90, f["radius"] * 1.02, f["radius"] * .60))

    across = direction.cross(Z).normalized()
    front = across.cross(direction).normalized()
    a, b, c = f["lengths"]
    total = a + b + c
    # Phalanges: a knuckle at each joint, a slimmer shaft between them, and a
    # pad that is fuller on the palm side than on the back of the finger.
    profile = [(-.34 if thumb else -.22, .95 if thumb else .60),
               (-.26 if thumb else -.15, .97 if thumb else .84),
               (-.14 if thumb else -.07, .98), (-.02, 1.04), (.05, 1.03),
               (a * .42, .945), (a - .06, .955), (a + .01, .99), (a + .07, .930),
               (a + b * .5, .893), (a + b - .05, .905), (a + b + .01, .940), (a + b + .06, .875),
               (total - .11, .850), (total - .04, .822), (total + .012, .762),
               (total + .042, .620), (total + .064, .390), (total + .078, .055)]
    tube = []
    for t, r in profile:
        centre = base + direction * t
        radius = f["radius"] * r
        round_tip = min(1., max(0., (t - total + .10) / .14))
        depth = radius * (.82 + .16 * round_tip)
        tube.append([tuple(centre + across * u + front * v)
                     for u, v in outline(radius, radius, depth, radius * (.98 + .02 * round_tip), 2.5)])
    surface(name, tube)

bpy.ops.object.select_all(action='DESELECT')
for ob in parts:
    bpy.context.view_layer.objects.active = ob
    ob.select_set(True)
bpy.context.view_layer.objects.active = parts[0]
bpy.ops.object.join()
skin = bpy.context.object
skin.name = "Digitra_SoftHand"
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
# Fuse the overlapping pieces into one continuous shell. The voxel grid is fine
# enough to keep the slots between neighbouring fingers open, and only a light
# relax follows so knuckles and pads are not melted away.
fusion = skin.modifiers.new("Continuous sculpted surface", 'REMESH')
fusion.mode = 'VOXEL'
fusion.voxel_size = .0072
fusion.use_smooth_shade = True
bpy.ops.object.modifier_apply(modifier=fusion.name)
polish = skin.modifiers.new("Relax voxel steps", 'SMOOTH')
polish.factor = .25
polish.iterations = 1
bpy.ops.object.modifier_apply(modifier=polish.name)
reduce = skin.modifiers.new("Web surface budget", 'DECIMATE')
reduce.ratio = min(1., 15000 / max(1, len(skin.data.polygons)))
bpy.ops.object.modifier_apply(modifier=reduce.name)
for p in skin.data.polygons:
    p.use_smooth = True

mat = bpy.data.materials.new("Warm porcelain")
mat.diffuse_color = (.870, .727, .567, 1)
mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
bsdf.inputs["Base Color"].default_value = mat.diffuse_color
bsdf.inputs["Roughness"].default_value = .34
bsdf.inputs["Metallic"].default_value = 0
bsdf.inputs["Coat Weight"].default_value = .2
bsdf.inputs["Coat Roughness"].default_value = .3
skin.data.materials.clear()
skin.data.materials.append(mat)

arm = bpy.data.armatures.new("Digitra_TID_Rig")
rig = bpy.data.objects.new("Digitra_TID_Rig", arm)
bpy.context.collection.objects.link(rig)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.mode_set(mode='EDIT')
wrist = arm.edit_bones.new("wrist")
wrist.head = xyz((0, -.16, 0))
wrist.tail = xyz((0, .45, 0))
for name, f in spec.items():
    parent = wrist
    for i in range(3):
        bone = arm.edit_bones.new(f"{name}_{i}")
        bone.head = xyz(f["points"][i])
        bone.tail = xyz(f["points"][i + 1])
        bone.parent = parent
        bone.use_connect = i > 0
        parent = bone
    tip = arm.edit_bones.new(f"{name}_tip")
    tip.head = xyz(f["points"][3])
    tip.tail = tip.head + xyz(f["direction"]) * .045
    tip.parent = parent
    tip.use_connect = True
    tip.use_deform = False
bpy.ops.object.mode_set(mode='OBJECT')

groups = {b.name: skin.vertex_groups.new(name=b.name) for b in arm.bones if b.use_deform}


def clamp(v):
    return max(0., min(1., v))


for vertex in skin.data.vertices:
    p = Vector((vertex.co.x, vertex.co.z, -vertex.co.y))
    best, score = None, 1e10
    for name, f in spec.items():
        base, direction = Vector(f["base"]), Vector(f["direction"])
        along = (p - base).dot(direction)
        if along < -.16:
            continue
        near = base + direction * max(0., min(sum(f["lengths"]), along))
        distance = (p - near).length / f["radius"]
        if distance < score:
            best, score = (name, f, along), distance
    weights = {"wrist": 1.}
    if best:
        name, f, along = best
        # Fade in along the bone chain and out across the finger, so the palm
        # keeps the root of every knuckle and nothing snaps at the boundary.
        share = clamp((along + .16) / .30) * clamp((2.15 - score) / .95)
        a, b = f["lengths"][0], f["lengths"][0] + f["lengths"][1]
        half = .075
        split = {0: 0., 1: 0., 2: 0.}
        if along < a - half:
            split[0] = 1.
        elif along < a + half:
            t = (along - a + half) / (2 * half)
            split[0], split[1] = 1 - t, t
        elif along < b - half:
            split[1] = 1.
        elif along < b + half:
            t = (along - b + half) / (2 * half)
            split[1], split[2] = 1 - t, t
        else:
            split[2] = 1.
        weights = {"wrist": 1 - share}
        for j, value in split.items():
            if value:
                weights[f"{name}_{j}"] = share * value
    for name, weight in weights.items():
        if weight > .00001:
            groups[name].add([vertex.index], weight, 'REPLACE')

# Diffuse joint influences along the fused topology so the sculpted knuckles
# retain a smooth surface when fingers fold or oppose the palm.
neighbors = [[] for _ in skin.data.vertices]
for edge in skin.data.edges:
    a, b = edge.vertices
    neighbors[a].append(b)
    neighbors[b].append(a)
weights = [{g.group: g.weight for g in vertex.groups} for vertex in skin.data.vertices]
for _ in range(12):
    updated = []
    for i, near in enumerate(neighbors):
        row = {k: v * .45 for k, v in weights[i].items()}
        for j in near:
            for k, v in weights[j].items():
                row[k] = row.get(k, 0) + v * .55 / len(near)
        updated.append(row)
    weights = updated
for group in skin.vertex_groups:
    group.remove(list(range(len(skin.data.vertices))))
for i, row in enumerate(weights):
    top = sorted(row.items(), key=lambda item: item[1], reverse=True)[:4]
    total = sum(v for _, v in top)
    for k, v in top:
        if v > .000001:
            skin.vertex_groups[k].add([i], v / total, 'REPLACE')

mod = skin.modifiers.new("Articulated skin", 'ARMATURE')
mod.object = rig
mod.use_deform_preserve_volume = False  # Match glTF linear skinning.
skin.parent = rig
rig.show_in_front = True
rig["description"] = "Original Digitra rig. Three independent joints per finger, fingertip anchors, wrist. TID poses authored separately."
rig["coordinate_system"] = "glTF: +Y fingers, +Z palm, thumb toward -X"

# Diagnostic motion lives in the editable source; the web runtime owns TID poses.
rig.animation_data_create()
for frame, curl in [(1, 0), (35, 1), (65, 0)]:
    for name in spec:
        for i, angle in enumerate([1.15, 1.4, .9]):
            pb = rig.pose.bones[f"{name}_{i}"]
            pb.rotation_mode = 'XYZ'
            pb.rotation_euler.x = angle * curl * (.55 if name == 'thumb' else 1)
            pb.keyframe_insert(data_path="rotation_euler", frame=frame)
rig.animation_data.action.name = "Rig_Flexion_Test"
bpy.context.scene.frame_set(1)
bpy.context.scene.frame_end = 65

bpy.ops.object.select_all(action='DESELECT')
rig.select_set(True)
skin.select_set(True)
bpy.context.view_layer.objects.active = rig
web_path = WEB / "public/models/tid-hand.glb"
bpy.ops.export_scene.gltf(filepath=str(web_path), export_format='GLB', use_selection=True,
    export_animations=False, export_yup=True, export_extras=True,
    export_copyright="Original Digitra teaching hand, generated for this project, 2026.")
(OUT / "tid-hand.glb").write_bytes(web_path.read_bytes())
(WEB / "src/components/tid/rig-spec.json").parent.mkdir(parents=True, exist_ok=True)
(WEB / "src/components/tid/rig-spec.json").write_text(json.dumps(spec, indent=2), encoding='utf-8')

# Keep a second, separately editable rig in the Blender deliverable.
rig.location.x = -.85
rig.name = "Right_hand"
left = rig.copy()
left.data = rig.data.copy()
left.name = "Left_hand"
bpy.context.collection.objects.link(left)
left.animation_data_clear()
left.location.x = .85
left.scale.x = -1
left_skin = skin.copy()
left_skin.data = skin.data.copy()
bpy.context.collection.objects.link(left_skin)
left_skin.parent = left
left_skin.modifiers["Articulated skin"].object = left
left_skin.name = "Left_hand_surface"
for screen in bpy.data.screens:
    for area in screen.areas:
        if area.type == 'VIEW_3D':
            area.spaces.active.region_3d.view_distance = 5
            area.spaces.active.region_3d.view_location = xyz((0, .6, 0))
bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "digitra-tid-hands.blend"))
print(json.dumps({"vertices": len(skin.data.vertices), "faces": len(skin.data.polygons),
                  "bones": len(arm.bones), "glb_bytes": web_path.stat().st_size}))
