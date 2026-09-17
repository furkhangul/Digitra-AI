"""Build the Digitra TID hand in Blender from the analytic SDF surface.

The rest mesh comes from marching cubes over exactly the same distance field the
site raymarches (see components/tid/rounded-hand.ts), so the editable model, the
GLB and the web view share one shape definition. Skin weights are computed from
the rig geometry rather than left to automatic weighting: each vertex is bound to
the finger chain it actually belongs to, which keeps neighbouring fingers from
dragging each other and preserves knuckle volume when a fist closes.

Run:  blender -b --factory-startup --python apps/web/scripts/build-tid-sdf-hand.py -- <rest.obj>
"""
import bpy, bmesh, json, math, sys
import numpy as np
from pathlib import Path
from mathutils import Vector, Quaternion, Matrix

ROOT = Path(__file__).resolve().parents[3]
WEB = ROOT / "apps/web"
OUT = ROOT / "output/models"
OUT.mkdir(parents=True, exist_ok=True)
SPEC = json.loads((WEB / "src/components/tid/rig-spec.json").read_text(encoding="utf-8"))
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
REST_OBJ = argv[0] if argv else str(OUT / "hand_rest.obj")
FRAMES = json.loads((ROOT / "tmp/tid/letter-frames.json").read_text(encoding="utf-8"))

# The site works in Y-up; Blender is Z-up.
def yup(v):
    return Vector((v[0], -v[2], v[1]))

def chain_points(finger):
    """Rest-pose joint positions of one finger, in Blender space."""
    s = SPEC[finger]
    d = Vector(s["direction"]).normalized()
    p = Vector(s["base"])
    pts = [p.copy()]
    for L in s["lengths"]:
        p = p + d * L
        pts.append(p.copy())
    return [yup(q) for q in pts]

# ---------------------------------------------------------------- mesh
def load_mesh():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    # The OBJ is in the site's Y-up frame; these axes apply exactly the same
    # mapping as yup(), so mesh and rig land in one space.
    bpy.ops.wm.obj_import(filepath=REST_OBJ, forward_axis="NEGATIVE_Z", up_axis="Y")
    ob = bpy.context.selected_objects[0]
    ob.name = "Digitra_TID_Hand"
    ob.data.name = "Digitra_TID_Hand"
    # The importer expresses the axis conversion as an object transform; bake it
    # so vertex coordinates and the rig live in the same space.
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    # marching cubes emits axis-aligned steps; merge, smooth and thin it down
    bpy.context.view_layer.objects.active = ob
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=1e-5)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    m = ob.modifiers.new("Relax", "SMOOTH"); m.factor = 0.6; m.iterations = 6
    bpy.ops.object.modifier_apply(modifier=m.name)
    d = ob.modifiers.new("Budget", "DECIMATE"); d.ratio = 0.075
    bpy.ops.object.modifier_apply(modifier=d.name)
    m = ob.modifiers.new("Polish", "SMOOTH"); m.factor = 0.45; m.iterations = 3
    bpy.ops.object.modifier_apply(modifier=m.name)
    for p in ob.data.polygons: p.use_smooth = True
    return ob

# ---------------------------------------------------------------- rig
def build_rig():
    arm = bpy.data.armatures.new("Digitra_TID_Rig")
    rig = bpy.data.objects.new("Digitra_TID_Rig", arm)
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    root = arm.edit_bones.new("wrist")
    root.head = Vector((0, 0, 0)); root.tail = yup((0, .42, 0))
    for f in FINGERS:
        pts = chain_points(f)
        parent = root
        for j in range(3):
            b = arm.edit_bones.new(f"{f}_{j}")
            b.head, b.tail = pts[j], pts[j + 1]
            b.parent = parent
            b.use_connect = j > 0
            parent = b
        # Leaf marker at the fingertip: carries no weight, but gives anything
        # consuming the GLB an explicit end effector per finger.
        tip = arm.edit_bones.new(f"{f}_tip")
        d = (pts[3] - pts[2]).normalized()
        tip.head, tip.tail = pts[3], pts[3] + d * (SPEC[f]["radius"] * .9)
        tip.parent = parent
        tip.use_connect = True
        tip.use_deform = False
    bpy.ops.object.mode_set(mode="OBJECT")
    return rig

# ---------------------------------------------------------------- weights
def segment_distance(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-9), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1), t

def bind(ob, rig):
    """Analytic skinning: pick the finger a vertex belongs to, then split its
    weight across that finger's three joints along the chain, leaving the rest
    on the wrist. Nothing is inherited from a neighbouring finger."""
    P = np.array([list(v.co) for v in ob.data.vertices], float)
    n = len(P)
    chains = {f: [np.array(list(p)) for p in chain_points(f)] for f in FINGERS}
    radius = {f: SPEC[f]["radius"] for f in FINGERS}

    # distance to each finger's whole polyline, plus arclength along it
    best_d = np.full(n, 1e9); best_f = np.zeros(n, int); best_s = np.zeros(n)
    lengths = {}
    for fi, f in enumerate(FINGERS):
        pts = chains[f]
        segs = [(pts[j], pts[j + 1]) for j in range(3)]
        acc = 0.0; Ls = []
        d_f = np.full(n, 1e9); s_f = np.zeros(n)
        for (a, b) in segs:
            d, t = segment_distance(P, a, b)
            L = float(np.linalg.norm(b - a))
            closer = d < d_f
            s_f[closer] = acc + t[closer] * L
            d_f[closer] = d[closer]
            acc += L; Ls.append(L)
        lengths[f] = Ls
        take = d_f < best_d
        best_d[take] = d_f[take]; best_f[take] = fi; best_s[take] = s_f[take]

    names = ["wrist"] + [f"{f}_{j}" for f in FINGERS for j in range(3)]
    groups = {n: ob.vertex_groups.new(name=n) for n in names}
    W = np.zeros((n, len(names)))

    def smoothstep(x):
        x = np.clip(x, 0, 1); return x * x * (3 - 2 * x)

    for fi, f in enumerate(FINGERS):
        mask = best_f == fi
        if not mask.any(): continue
        idx = np.where(mask)[0]
        s = best_s[idx]; Ls = lengths[f]; r = radius[f]
        # how much this vertex belongs to the finger at all: fades in over the
        # first part of the proximal phalanx so the palm keeps its own volume.
        # The thumb's first bone is a metacarpal buried in the thenar, so it
        # takes over later and more gently.
        # Wide, overlapping bands: an abrupt hand-off between two bones tears the
        # surface open where a finger folds hard.
        onset = (0.55 * Ls[0], 1.5 * r) if f == "thumb" else (0.22 * Ls[0], 1.4 * r)
        belong = smoothstep((s - onset[0]) / max(onset[1], 1e-6))
        j0, j1 = Ls[0], Ls[0] + Ls[1]
        band = 1.45 * r
        w1 = smoothstep((s - j0) / band + .5)     # into the middle phalanx
        w2 = smoothstep((s - j1) / band + .5)     # into the distal phalanx
        wb = np.stack([(1 - w1), w1 * (1 - w2), w1 * w2], 1) * belong[:, None]
        W[np.ix_(idx, [1 + 3*fi, 2 + 3*fi, 3 + 3*fi])] = wb
        W[idx, 0] = 1.0 - wb.sum(1)

    # Relax the weight field over the mesh edges. This is what stops the surface
    # from splitting where two bones hand over, and it costs nothing at runtime.
    nb = [[] for _ in range(n)]
    for e in ob.data.edges:
        a, b = e.vertices; nb[a].append(b); nb[b].append(a)
    flat = np.concatenate([np.array(v, int) for v in nb]) if n else np.array([], int)
    counts = np.array([len(v) for v in nb])
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    for _ in range(8):
        acc = np.add.reduceat(W[flat], starts, axis=0)
        acc[counts == 0] = W[counts == 0]
        avg = acc / np.maximum(counts, 1)[:, None]
        W = 0.45 * W + 0.55 * avg
    W = np.clip(W, 0, None)
    W /= np.maximum(W.sum(1, keepdims=True), 1e-9)
    for gi, gname in enumerate(names):
        g = groups[gname]
        for v in np.where(W[:, gi] > 1e-4)[0]:
            g.add([int(v)], float(W[v, gi]), "REPLACE")

    ob.parent = rig
    mod = ob.modifiers.new("Articulated skin", "ARMATURE")
    mod.object = rig
    mod.use_vertex_groups = True
    return ob

# ---------------------------------------------------------------- posing
def pose_from_angles(rig, angles, side):
    for fi, f in enumerate(FINGERS):
        off = fi * 5
        d = Vector(SPEC[f]["direction"]).normalized()
        axis = yup(d.cross(Vector((0, 0, 1))).normalized())
        spread, oppose = angles[off + 3], angles[off + 4]
        q = Quaternion(yup((0, 0, 1)), math.radians(spread)) @ \
            Quaternion(yup((0, 1, 0)), math.radians(oppose))
        for j in range(3):
            pb = rig.pose.bones[f"{f}_{j}"]
            rest = pb.bone.matrix_local.to_quaternion()
            local = Quaternion(axis, math.radians(angles[off + j]))
            if j == 0: local = q @ local
            pb.rotation_mode = "QUATERNION"
            pb.rotation_quaternion = rest.inverted() @ local @ rest

def export_collision_proxy(ob, budget=0.055):
    """A low-resolution skinned copy for the runtime contact guard.

    This is an APPROXIMATION used only to keep playback cheap; the numbers in
    the verification report come from the full surface, never from this proxy.
    """
    copy = ob.copy(); copy.data = ob.data.copy()
    bpy.context.scene.collection.objects.link(copy)
    bpy.context.view_layer.objects.active = copy
    d = copy.modifiers.new("Proxy", "DECIMATE"); d.ratio = budget
    bpy.ops.object.modifier_apply(modifier=d.name)
    me = copy.data
    me.calc_loop_triangles()
    names = [g.name for g in copy.vertex_groups]
    out = {
        "note": "Approximate collision proxy for playback; not a verification surface.",
        "positions": [[round(c, 5) for c in v.co] for v in me.vertices],
        "triangles": [list(t.vertices) for t in me.loop_triangles],
        "weights": [[[names[g.group], round(g.weight, 4)]
                     for g in sorted(v.groups, key=lambda g: -g.weight)[:4] if g.weight > .02]
                    for v in me.vertices],
    }
    # every vertex needs at least one bone or the guard divides by zero
    for i, w in enumerate(out["weights"]):
        if not w: out["weights"][i] = [["wrist", 1.0]]
    path = WEB / "src/components/tid/collision-surface.json"
    path.write_text(json.dumps(out), encoding="utf-8")
    bpy.data.objects.remove(copy, do_unlink=True)
    print("PROXY", len(out["positions"]), "verts", len(out["triangles"]), "tris",
          round(path.stat().st_size / 1024), "KB")

def main():
    ob = load_mesh()
    rig = build_rig()
    bind(ob, rig)
    bpy.context.view_layer.update()

    # one action per letter per hand, plus a neutral rest action
    for L in FRAMES["letters"]:
        for side in ("right", "left"):
            hand = L["frame"][side]
            if hand["presence"] <= .008: continue
            name = f"TID_{L['ch']}_{side}"
            act = bpy.data.actions.new(name)
            rig.animation_data_create()
            rig.animation_data.action = act
            pose_from_angles(rig, hand["angles"], side)
            for f in FINGERS:
                for j in range(3):
                    rig.pose.bones[f"{f}_{j}"].keyframe_insert("rotation_quaternion", frame=1)
            act.use_fake_user = True
    rig.animation_data.action = None
    for pb in rig.pose.bones:
        pb.rotation_mode = "QUATERNION"
        pb.rotation_quaternion = Quaternion()

    mat = bpy.data.materials.new("Digitra_Skin")
    mat.use_nodes = True
    b = mat.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (.96, .69, .37, 1)
    b.inputs["Roughness"].default_value = .48
    ob.data.materials.append(mat)

    export_collision_proxy(ob)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "digitra-tid-hands.blend"))
    bpy.ops.object.select_all(action="DESELECT")
    ob.select_set(True); rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.export_scene.gltf(filepath=str(OUT / "tid-hand.glb"), export_format="GLB",
                              use_selection=True, export_animations=True,
                              export_yup=True, export_apply=False)
    me = ob.data
    bm = bmesh.new(); bm.from_mesh(me)
    report = {
        "vertices": len(bm.verts), "faces": len(bm.faces),
        "boundaryEdges": sum(e.is_boundary for e in bm.edges),
        "nonManifoldEdges": sum(not e.is_manifold for e in bm.edges),
        "bones": len(rig.data.bones), "actions": len(bpy.data.actions),
        "vertexGroups": len(ob.vertex_groups),
        "weightSumRange": [min(sum(g.weight for g in v.groups) for v in me.vertices),
                           max(sum(g.weight for g in v.groups) for v in me.vertices)],
        # Guards against the mesh and rig drifting into different spaces: if that
        # happens almost everything ends up bound to the wrist and nothing bends.
        "fingerBoundFraction": round(sum(
            1 for v in me.vertices
            if any(ob.vertex_groups[g.group].name != "wrist" and g.weight > .5 for g in v.groups)
        ) / len(me.vertices), 4),
    }
    bm.free()
    (OUT / "sdf-build-report.json").write_text(json.dumps(report, indent=2))
    print("BUILD", json.dumps(report))

main()
