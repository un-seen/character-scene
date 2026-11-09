"""Export a compiled clip as a single .glb for the browser: the body mesh, the deform bones
only (Rigify's control and mechanism bones stay behind), the animation baked from the
evaluated pose, and the face shape keys as morph-target animation."""
import bpy

from . import registry as R


def glb(path, frames):
    sc = bpy.context.scene
    sc.frame_start, sc.frame_end = 1, frames
    rig, body = bpy.data.objects[R.RIG], bpy.data.objects[R.BODY]
    parts = {rig, body} | {o for o in bpy.data.objects if o.type == "MESH" and o.parent is rig}   # eyes ride the head bone
    for o in bpy.data.objects:
        o.select_set(o in parts)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.export_scene.gltf(
        filepath=path, export_format="GLB", use_selection=True,
        export_def_bones=True, export_hierarchy_flatten_bones=False,
        export_animations=True, export_animation_mode="ACTIVE_ACTIONS", export_force_sampling=True,
        export_frame_range=True, export_anim_single_armature=True, export_bake_animation=True,
        export_morph=True, export_morph_animation=True, export_skins=True,
        export_optimize_animation_size=False, export_apply=False, export_yup=True,
        export_image_format="NONE")     # textures ship once, separately (4K PNGs are ~45 MB)
