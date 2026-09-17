"""Mesh the rest-pose hand from the same SDF the site renders, via marching cubes."""
import numpy as np, sys
from skimage import measure
from shape import build_capsules, hand_sdf, SPEC
from sdfhand import FINGERS

REST = {n: ([0, 0, 0], 0, 0) for n in FINGERS}
VOXEL = float(sys.argv[1]) if len(sys.argv) > 1 else 0.008
OUT = sys.argv[2] if len(sys.argv) > 2 else "hand_rest.obj"

caps = build_capsules(REST, SPEC)
pts = np.array([p for segs in caps.values() for (a, b, ra, rb) in segs for p in (a, b)])
rmax = max(max(ra, rb) for segs in caps.values() for (a, b, ra, rb) in segs)
palm = np.array([SPEC[k][0] for k in ("palm_core","knuckle_pad","heel","thenar","hypothenar","thumb_web")])
prad = np.array([SPEC[k][1] for k in ("palm_core","knuckle_pad","heel","thenar","hypothenar","thumb_web")])
lo = np.minimum(pts.min(0) - rmax, (palm - prad).min(0)) - 0.10
hi = np.maximum(pts.max(0) + rmax, (palm + prad).max(0)) + 0.10
dims = np.ceil((hi - lo) / VOXEL).astype(int) + 1
print("bounds", np.round(lo, 3), np.round(hi, 3), "grid", dims, "voxels", int(np.prod(dims)))

xs = lo[0] + np.arange(dims[0]) * VOXEL
ys = lo[1] + np.arange(dims[1]) * VOXEL
zs = lo[2] + np.arange(dims[2]) * VOXEL
field = np.empty(tuple(dims), np.float32)
gy, gz = np.meshgrid(ys, zs, indexing="ij")
flat = np.stack([np.zeros_like(gy), gy, gz], -1).reshape(-1, 3)
for i, x in enumerate(xs):
    flat[:, 0] = x
    field[i] = hand_sdf(flat, caps, SPEC).reshape(dims[1], dims[2])
    if i % 40 == 0: print(f"  slab {i}/{dims[0]}", flush=True)

verts, faces, normals, _ = measure.marching_cubes(field, level=0.0, spacing=(VOXEL,)*3)
verts = verts + lo
print("mesh", len(verts), "verts", len(faces), "tris")
with open(OUT, "w") as f:
    f.write("# Digitra TID hand, marching cubes over the analytic hand SDF\n")
    for v in verts: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
    for t in faces: f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")
print("wrote", OUT)
