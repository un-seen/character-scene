"""Stills of an animated scene: a three-quarter view that follows the hips, soft light.
  [FRAMING=close] blender -b scene.blend --python humans/frames.py -- OUT_PREFIX FRAME [FRAME ...]"""
import math
import os
import sys

import bpy
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1:]
out, frames = args[0], [int(f) for f in args[1:]]
sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE"
sc.render.resolution_x, sc.render.resolution_y, sc.render.resolution_percentage = 560, 760, 100
w = bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
w.node_tree.nodes["Background"].inputs[0].default_value = (0.82, 0.84, 0.88, 1)
w.node_tree.nodes["Background"].inputs[1].default_value = 0.6
lights = []
for name, loc, energy in (("key", (2.0, -3.0, 3.0), 600), ("fill", (-3.0, -2.0, 2.0), 250), ("rim", (0.0, 3.0, 3.0), 300)):
    L = bpy.data.lights.new(name, "AREA"); L.energy = energy; L.size = 2.5
    o = bpy.data.objects.new(name, L); sc.collection.objects.link(o); lights.append((o, Vector(loc)))
cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
cam.data.lens = 45; cam.data.sensor_fit = "AUTO"; cam.data.sensor_width = 36
rig = bpy.data.objects["Rig"]
for f in frames:
    sc.frame_set(f)
    hips = rig.matrix_world @ rig.pose.bones["DEF-HIPS"].head
    close = os.environ.get("FRAMING") == "close"       # the chest and shoulders, from the front
    c = Vector((hips.x, hips.y, 1.3 if close else 0.85))
    a, d = math.radians(8 if close else 30), (1.9 if close else 4.4)
    cam.location = c + Vector((d * math.sin(a), -d * math.cos(a), 0.15))
    cam.rotation_euler = (c - cam.location).to_track_quat("-Z", "Y").to_euler()
    for o, loc in lights:
        o.location = c + loc; o.rotation_euler = (c + Vector((0, 0, 0.2)) - o.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.filepath = f"{out}_{f:03d}.png"
    bpy.ops.render.render(write_still=True)
