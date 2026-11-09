"""Preview fixtures: a floor, a sun and a camera that follows the walker at 3/4. These live
outside the program, so a program cannot move the floor or the camera."""
import math
import os
import shutil
import subprocess

import bpy
from mathutils import Vector

from . import registry as R


def preview(out_mp4, frames, fps, engine="BLENDER_EEVEE", size=(960, 540), close_s=2.5):
    sc = bpy.context.scene
    rig = bpy.data.objects[R.RIG]; rig.hide_render = True
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, 0))
    floor = bpy.context.object
    mat = bpy.data.materials.new("floor"); mat.use_nodes = True
    nt = mat.node_tree; bsdf = nt.nodes["Principled BSDF"]
    chk = nt.nodes.new("ShaderNodeTexChecker"); chk.inputs["Scale"].default_value = 200
    chk.inputs["Color1"].default_value = (0.42, 0.40, 0.37, 1); chk.inputs["Color2"].default_value = (0.30, 0.29, 0.27, 1)
    nt.links.new(chk.outputs["Color"], bsdf.inputs["Base Color"]); floor.data.materials.append(mat)
    sun = bpy.data.objects.new("sun", bpy.data.lights.new("sun", "SUN")); sc.collection.objects.link(sun)
    sun.data.energy = 3.2; sun.rotation_euler = (math.radians(50), math.radians(10), math.radians(35))
    w = sc.world or bpy.data.worlds.new("w"); sc.world = w; w.use_nodes = True
    w.node_tree.nodes["Background"].inputs["Color"].default_value = (0.55, 0.6, 0.68, 1)
    aim = bpy.data.objects.new("aim", None); sc.collection.objects.link(aim)
    cam = bpy.data.objects.new("cam", bpy.data.cameras.new("cam")); sc.collection.objects.link(cam); sc.camera = cam
    tr = cam.constraints.new("TRACK_TO"); tr.target = aim; tr.track_axis = "TRACK_NEGATIVE_Z"; tr.up_axis = "UP_Y"
    # Full body from a 3/4 front view that follows the walker; for the last `close_s` seconds the
    # camera eases into a head-and-shoulders close-up so the face reads.
    root, head = [], []
    P = R.Pose(rig)
    for f in range(1, frames + 1):
        sc.frame_set(f)
        root.append((rig.matrix_world @ P["root"].matrix).translation.copy())
        head.append(rig.matrix_world @ P["DEF-spine.006"].matrix @ Vector((0, 0.06, 0)))
    def smooth(xs, f, w):
        near = xs[max(0, f - 1 - w):f + w]; return sum(near, Vector()) / len(near)
    t_close = max(1, frames - int(close_s * fps))
    for f in range(1, frames + 1):
        c, h = smooth(root, f, 12), smooth(head, f, 6)
        k = 0.0 if f < t_close else min(1.0, (f - t_close) / (0.8 * fps))
        k = k * k * (3 - 2 * k)
        body = Vector((c.x, c.y, 0.92))
        aim.location = body.lerp(h - Vector((0, 0, 0.05)), k); aim.keyframe_insert("location", frame=f)
        cam.location = Vector((c.x + 2.9, c.y - 2.6, 1.25)).lerp(Vector((c.x + 2.9, c.y - 2.6, 1.6)), k)
        cam.keyframe_insert("location", frame=f)
        cam.data.lens = 38 + k * (125 - 38); cam.data.keyframe_insert("lens", frame=f)
    r = sc.render
    r.engine = engine; r.resolution_x, r.resolution_y = size; r.fps = fps
    r.image_settings.file_format = "PNG"         # frames, then ffmpeg: Blender 5 moved video out of image formats
    frames_dir = os.path.splitext(out_mp4)[0] + "_frames"
    os.makedirs(frames_dir, exist_ok=True)
    r.filepath = os.path.join(frames_dir, "f_")
    if engine.startswith("BLENDER_EEVEE"): sc.eevee.taa_render_samples = 16
    sc.frame_start, sc.frame_end = 1, frames
    bpy.ops.render.render(animation=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", os.path.join(frames_dir, "f_%04d.png"),
                    "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_mp4], check=True)
    shutil.rmtree(frames_dir)
