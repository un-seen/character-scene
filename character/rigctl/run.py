"""Command line, run inside Blender:

  blender -b out/rig.blend --python rigctl/run.py -- registry OUT.json
  blender -b out/rig.blend --python rigctl/run.py -- compile PROGRAM.json OUT_DIR [--preview] [--glb]
  blender -b ANIMATED.blend      --python rigctl/run.py -- check OUT_DIR

compile writes OUT_DIR/scene.blend and OUT_DIR/report.json (validation, compile and check
results). A program that fails validation is not compiled; its report lists every error.
"""
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bpy  # noqa: E402
from rigctl import registry, program, checker  # noqa: E402


def main(argv):
    cmd = argv[0]
    rig = bpy.data.objects[registry.RIG]
    if cmd == "registry":
        registry.save(registry.build(rig), argv[1]); print("REGISTRY", argv[1]); return 0
    if cmd == "check":
        out = argv[1]; os.makedirs(out, exist_ok=True)
        res = checker.check()
        json.dump(res, open(os.path.join(out, "report.json"), "w"), indent=1)
        print("CHECK", json.dumps({"ok": res["ok"], "reward": res["reward"]})); return 0
    if cmd == "compile":
        from rigctl.compiler import Compiler
        src, out = argv[1], argv[2]; os.makedirs(out, exist_ok=True)
        reg = registry.build(rig)
        rep = {"program": src}
        try:
            raw = json.load(open(src))
            prog, warn = program.validate(json.loads(json.dumps(raw)), reg)
            if any(k.get("spec") for b in raw["beats"] if b["verb"] == "pose" for k in b["args"]["keys"]):
                from rigctl import poseprog                # keys in body words: solved to angles, then checked as such
                t = time.time(); raw = poseprog.lower(raw, rep); rep["pose_solve_s"] = round(time.time() - t, 1)
                prog, warn = program.validate(raw, reg)
        except program.ProgramError as e:
            rep.update({"valid": False, "errors": e.args[0], "reward": 0.0})
            json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1)
            print("INVALID", json.dumps(e.args[0])); return 1
        except (ValueError, KeyError) as e:
            rep.update({"valid": False, "errors": [f"unreadable program: {e}"], "reward": 0.0})
            json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1)
            print("INVALID", e); return 1
        rep.update({"valid": True, "warnings": warn})
        t = time.time()
        try:
            crep = Compiler(prog, reg).compile()
        except Exception:
            rep.update({"compiled": False, "errors": [traceback.format_exc()], "reward": 0.0})
            json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1)
            print("COMPILE-FAILED"); traceback.print_exc(); return 1
        rep["compile"] = crep; rep["compile_s"] = round(time.time() - t, 1)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "scene.blend"))
        t = time.time()
        res = checker.check(prog, crep, (1, prog["clip"]["frames"]))
        rep["check"] = res; rep["check_s"] = round(time.time() - t, 1); rep["reward"] = res["reward"]
        json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1, default=str)
        print("RESULT", json.dumps({"ok": res["ok"], "reward": res["reward"],
                                    "failed": [k for k, c in res["checks"].items() if not c["pass"]]}))
        if "--glb" in argv:
            from rigctl import export
            export.glb(os.path.join(out, "clip.glb"), prog["clip"]["frames"])
            print("GLB", os.path.join(out, "clip.glb"))
        if "--preview" in argv:
            from rigctl import scene
            scene.preview(os.path.join(out, "preview.mp4"), prog["clip"]["frames"], prog["clip"]["fps"])
        return 0
    print(__doc__); return 2


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    sys.exit(main(args))
