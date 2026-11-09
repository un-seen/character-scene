"""A head-and-shoulders portrait of the cast member in the open .blend, for the page's look picker.
  blender -b humans/ID.blend --python humans/portrait.py -- OUT.png [HIDE_NAME ...]"""
import sys, bpy
from mathutils import Vector
args = sys.argv[sys.argv.index("--") + 1:]
out, hide = args[0], set(args[1:])
sc = bpy.context.scene
for o in bpy.data.objects:
    if o.name in hide: o.hide_render = True
sc.render.engine = "BLENDER_EEVEE"
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 336, 400, 100
sc.view_settings.view_transform = "AgX"; sc.view_settings.look = "AgX - Base Contrast"
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.86, 0.85, 0.82, 1)
w.node_tree.nodes["Background"].inputs[1].default_value = 0.5
eyes = [o for o in bpy.data.objects if o.name.endswith("high-poly") and not o.hide_render][0]
c = sum((eyes.matrix_world @ v.co for v in eyes.data.vertices), Vector()) / len(eyes.data.vertices) - Vector((0, 0, 0.09))
for name, loc, energy in (("key", (0.9, -1.5, 0.6), 55), ("fill", (-1.2, -1.2, 0.2), 22), ("rim", (0.0, 1.4, 0.8), 30)):
    L = bpy.data.lights.new(name, "AREA"); L.energy = energy; L.size = 1.5
    o = bpy.data.objects.new(name, L); sc.collection.objects.link(o); o.location = c + Vector(loc)
    o.rotation_euler = (c - o.location).to_track_quat("-Z", "Y").to_euler()
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.data.lens = 85; cam.data.sensor_fit = "AUTO"; cam.data.sensor_width = 36
cam.location = c + Vector((0.18, -1.15, 0.03)); cam.rotation_euler = (c - cam.location).to_track_quat("-Z", "Y").to_euler()
sc.render.filepath = out
bpy.ops.render.render(write_still=True)
print("PORTRAIT", out)
