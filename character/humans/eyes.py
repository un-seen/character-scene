"""Each eye's centre (world, rest) of the cast member in the open .blend, for server/eyes.json:
the page aims gaze from there.   blender -b humans/ID.blend --python humans/eyes.py -- ID"""
import sys, json, bpy
from mathutils import Vector
cid = sys.argv[sys.argv.index("--") + 1]
rig = next(o for o in bpy.data.objects if o.type == "ARMATURE")
rig.data.pose_position = "REST"; bpy.context.view_layer.update()
eyes = next(o for o in bpy.data.objects if o.type == "MESH" and o.name.endswith("high-poly") and not o.hide_render)
dg = bpy.context.evaluated_depsgraph_get(); me = eyes.evaluated_get(dg).to_mesh()
P = [eyes.matrix_world @ v.co for v in me.vertices]
out = {}
for s, f in (("L", lambda p: p.x > 0), ("R", lambda p: p.x < 0)):
    q = [p for p in P if f(p)]; c = sum(q, Vector()) / len(q); out[s] = [round(c.x, 4), round(c.y, 4), round(c.z, 4)]
print("EYES", json.dumps({cid: out}))
