"""Close-ups of the face keys: rest, then each key at full strength.
  blender -b humans/ID.blend --python humans/face.py -- OUT_PREFIX"""
import sys

import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1:]
out, near = args[0], len(args) > 1     # near: the eyes only
sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE"
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 360, 420, 100
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.82, 0.84, 0.88, 1)
w.node_tree.nodes["Background"].inputs[1].default_value = 0.7
eyes = [o for o in bpy.data.objects if o.name.endswith("high-poly")][0]
c = sum((eyes.matrix_world @ v.co for v in eyes.data.vertices), Vector()) / len(eyes.data.vertices) - Vector((0, 0, 0.0 if len(sys.argv[sys.argv.index("--") + 1:]) > 1 else 0.03))
for name, loc, energy in (("key", (0.8, -1.4, 0.5), 120), ("fill", (-1.0, -1.0, 0.2), 60)):
    L = bpy.data.lights.new(name, "AREA"); L.energy = energy; L.size = 1.0
    o = bpy.data.objects.new(name, L); sc.collection.objects.link(o); o.location = c + Vector(loc)
    o.rotation_euler = (c - o.location).to_track_quat("-Z", "Y").to_euler()
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.data.lens = 85; cam.data.sensor_fit = "AUTO"; cam.data.sensor_width = 36
cam.location = c + (Vector((0.05, -0.3, 0.05)) if near else Vector((0.12, -0.75, 0.02))); cam.rotation_euler = (c - cam.location).to_track_quat("-Z", "Y").to_euler()
keyed = [o for o in bpy.data.objects if o.type == "MESH" and o.data.shape_keys and not o.hide_render]
for k in (("rest", "blink") if near else ("rest", "blink", "smile", "brows_up", "frown", "jaw_open")):
    for o in keyed:
        for b in o.data.shape_keys.key_blocks[1:]: b.value = 1.0 if b.name == k else 0.0
    sc.render.filepath = f"{out}_{k}.png"
    bpy.ops.render.render(write_still=True)
