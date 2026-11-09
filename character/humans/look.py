"""A look at whatever is in the open .blend: front and three-quarter renders, soft key and fill light.
  blender -b FILE.blend --python humans/look.py -- OUT_PREFIX [HIDE_NAME ...]"""
import math
import sys

import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1:]
out, hide = args[0], set(args[1:])
sc = bpy.context.scene
for o in bpy.data.objects:
    if o.name in hide: o.hide_render = True
sc.render.engine = "BLENDER_EEVEE" if "BLENDER_EEVEE" in [e.identifier for e in bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items] else "BLENDER_EEVEE_NEXT"
sc.render.resolution_x, sc.render.resolution_y = 640, 900
sc.render.resolution_percentage = 100
sc.render.film_transparent = False
w = bpy.data.worlds.new("w") if not sc.world else sc.world
sc.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.82, 0.84, 0.88, 1)
w.node_tree.nodes["Background"].inputs[1].default_value = 0.6
for name, loc, energy in (("key", (2.0, -3.0, 3.0), 600), ("fill", (-3.0, -2.0, 2.0), 250), ("rim", (0.0, 3.0, 3.0), 300)):
    L = bpy.data.lights.new(name, "AREA"); L.energy = energy; L.size = 2.5
    o = bpy.data.objects.new(name, L); sc.collection.objects.link(o); o.location = loc
    o.rotation_euler = (Vector((0, 0, 1.0)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.data.lens = 50; cam.data.sensor_fit = "AUTO"; cam.data.sensor_width = 36; cam.data.shift_x = cam.data.shift_y = 0
for tag, ang in (("front", 0), ("q", 35)):
    a = math.radians(ang); d = 4.6
    cam.location = (d * math.sin(a), -d * math.cos(a), 1.0)
    cam.rotation_euler = (Vector((0, 0, 0.88)) - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = f"{out}_{tag}.png"
    bpy.ops.render.render(write_still=True)
