"""Close-up deformation checks on the built hand: open, half curl, fist, ring, hook.

Renders each pose from several angles, including from behind the thumb, so the
thumb root, the finger webs and the wrist cap can be inspected for creases,
holes or collapsed volume.

Run: blender -b --factory-startup --python apps/web/scripts/render-tid-check.py
"""
import bpy, json, math, sys
from pathlib import Path
from mathutils import Vector, Quaternion

ROOT = Path(__file__).resolve().parents[3]
WEB = ROOT / "apps/web"
OUT = ROOT / "output/hand-check"
OUT.mkdir(parents=True, exist_ok=True)
SPEC = json.loads((WEB / "src/components/tid/rig-spec.json").read_text(encoding="utf-8"))
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

def yup(v): return Vector((v[0], -v[2], v[1]))

bpy.ops.wm.open_mainfile(filepath=str(ROOT / "output/models/digitra-tid-hands.blend"))
rig = bpy.data.objects["Digitra_TID_Rig"]
ob = bpy.data.objects["Digitra_TID_Hand"]

def pose(angles):
    rig.animation_data_clear()
    for fi, f in enumerate(FINGERS):
        off = fi * 5
        d = Vector(SPEC[f]["direction"]).normalized()
        axis = yup(d.cross(Vector((0, 0, 1))).normalized())
        q = Quaternion(yup((0, 0, 1)), math.radians(angles[off+3])) @ \
            Quaternion(yup((0, 1, 0)), math.radians(angles[off+4]))
        for j in range(3):
            pb = rig.pose.bones[f"{f}_{j}"]
            rest = pb.bone.matrix_local.to_quaternion()
            loc = Quaternion(axis, math.radians(angles[off+j]))
            if j == 0: loc = q @ loc
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = rest.inverted() @ loc @ rest
    bpy.context.view_layer.update()

def angles(per_finger):
    out = []
    for f in FINGERS: out += list(per_finger[f])
    return out

def uniform(b, s=0, o=0): return {f: [*b, s, o] for f in FINGERS}

CLOSED_THUMB = [25, 35, 20, -50, 20]
POSES = {
  "open":  angles(uniform([0, 0, 0])),
  "half":  angles({**uniform([38, 48, 26]), "thumb": [14, 20, 10, 6, 16]}),
  "fist":  angles({**uniform([88, 100, 55]), "thumb": CLOSED_THUMB}),
  "ring":  angles({**uniform([0, 0, 0]), "index": [25, 42, 45, 0, 0], "thumb": [3, 0, 0, -43, 10]}),
  "hook":  angles({**uniform([88, 100, 55]), "index": [8, 95, 40, 0, 0], "thumb": CLOSED_THUMB}),
  "spread": angles({"thumb": [0, 0, 0, 12, 0], "index": [0, 0, 0, 16, 0], "middle": [0, 0, 0, 5, 0],
                    "ring": [0, 0, 0, -8, 0], "pinky": [0, 0, 0, -18, 0]}),
}

sc = bpy.context.scene
for o in list(sc.objects):
    if o.type in ("LIGHT", "CAMERA"): bpy.data.objects.remove(o, do_unlink=True)
sc.render.engine = "CYCLES"; sc.cycles.samples = 40; sc.cycles.use_denoising = True
sc.render.resolution_x = sc.render.resolution_y = 560
sc.view_settings.view_transform = "AgX"
sc.world = bpy.data.worlds.new("W"); sc.world.use_nodes = True
sc.world.node_tree.nodes["Background"].inputs[0].default_value = (.07, .06, .11, 1)
sc.world.node_tree.nodes["Background"].inputs[1].default_value = .7
target = yup((0, .78, 0))
for loc, e, size in [((-2.4, 3.4, 3.6), 900, 3.2), ((3.2, 1.6, 2.4), 320, 2.6), ((0, 2.2, -3.6), 420, 3.0)]:
    bpy.ops.object.light_add(type="AREA", location=yup(loc)); L = bpy.context.object
    L.data.energy = e; L.data.size = size
    L.rotation_euler = (target - L.location).to_track_quat("-Z", "Y").to_euler()
cd = bpy.data.cameras.new("C"); cam = bpy.data.objects.new("C", cd)
sc.collection.objects.link(cam); sc.camera = cam; cd.type = "ORTHO"; cd.ortho_scale = 2.9

# 0 front (palm), 90 little-finger side, 180 back, 250 behind the thumb root
VIEWS = [0, 60, 180, 250]
for name, a in POSES.items():
    pose(a)
    for ang in VIEWS:
        r = math.radians(ang)
        cam.location = target + yup((3.6*math.sin(r), .35, 3.6*math.cos(r)))
        cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
        sc.render.filepath = str(OUT / f"{name}-{ang:03d}.png")
        bpy.ops.render.render(write_still=True)
print("CHECK renders in", OUT)
