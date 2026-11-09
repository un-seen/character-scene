"""A cast member as the page loads it (2026-11-02): its body, clothes, hair and eyes on the reference rig's
deform bones at rest, textures as JPEG at most MAX_TEX px, no animation. The page plays every clip
(the clips the server compiles on the reference rig body) on this model by bone name: the deform bones and
their rest are the reference rig's, so the clips fit it as they are.
  blender -b humans/ID.blend --python humans/export_model.py -- OUT.glb [MAX_TEX]"""
import sys

import bpy

args = sys.argv[sys.argv.index("--") + 1:]
out, max_tex = args[0], int(args[1]) if len(args) > 1 else 1024
rig = bpy.data.objects["Rig"]
parts = [o for o in rig.children if o.type == "MESH" and not o.hide_render]
for im in bpy.data.images:                               # smaller textures: the page loads two people
    if im.size[0] > max_tex or im.size[1] > max_tex:
        s = max_tex / max(im.size)
        im.scale(max(1, int(im.size[0] * s)), max(1, int(im.size[1] * s)))
rig.data.pose_position = "REST"
for o in bpy.data.objects:
    o.select_set(o is rig or o in parts)
bpy.context.view_layer.objects.active = rig
bpy.ops.export_scene.gltf(
    filepath=out, export_format="GLB", use_selection=True,
    export_def_bones=True, export_hierarchy_flatten_bones=False,
    export_animations=False, export_skins=True, export_morph=True, export_apply=False, export_yup=True,
    export_image_format="JPEG", export_jpeg_quality=85)
print("MODEL", out, [o.name for o in parts])
