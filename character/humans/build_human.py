"""A cast member built with MPFB (MakeHuman for Blender, CC0 assets), 2026-11-02: its body from the
phenotype in cast.json, skin, eyes, brows, lashes, hair and clothes, on MPFB's default rig with its
weights (fit_human.py uses them to bend the body onto the reference rig skeleton).
  blender -b --python humans/build_human.py -- ID   -> humans/ID_mh.blend"""
import json
import os
import sys

import bpy
from bl_ext.user_default.mpfb.services import HumanService

HERE = os.path.dirname(os.path.abspath(__file__))
cid = sys.argv[sys.argv.index("--") + 1]
spec = json.load(open(os.path.join(HERE, "cast.json")))[cid]

for o in list(bpy.data.objects): bpy.data.objects.remove(o)
info = HumanService._create_default_human_info_dict()
info.update(phenotype=spec["phenotype"], rig="default", skin_mhmat=spec["skin"], skin_material_type="GAMEENGINE",
            eyes="high-poly/high-poly.mhclo", eyes_material_type="MAKESKIN", eyebrows=spec["eyebrows"],
            eyelashes=spec["eyelashes"], hair=spec["hair"], teeth="teeth_base/teeth_base.mhclo",
            tongue="tongue01/tongue01.mhclo", clothes=spec["clothes"], clothes_material_type="MAKESKIN",
            alternative_materials={})
st = HumanService.get_default_deserialization_settings()
st.update(subdiv_levels=0)
HumanService.deserialize_from_dict(info, st)
for o in bpy.data.objects:
    print("OBJ", o.type, o.name, o.parent and o.parent.name, len(o.data.vertices) if o.type == "MESH" else len(o.data.bones),
          [s.material.name for s in o.material_slots if s.material] if o.type == "MESH" else "",
          [m.type for m in o.modifiers] if o.type == "MESH" else "")
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, f"{cid}_mh.blend"))
