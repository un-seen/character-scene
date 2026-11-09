"""A Blender that stays running and takes compile jobs, one JSON line at a time on stdin:

  blender -b --python rigctl/worker.py

  {"op": "frames", "blend": ..., "program": PATH, "frames": [f, ...], "face": [f, ...], "out": PATH}
      pose those frames (0-based, ascending) and write their values (Compiler.snapshot) to out,
      then the face's alone on the frames `face` (Compiler.face_snapshot)
  {"op": "finish", "blend": ..., "program": PATH, "chunks": [PATH, ...], "out_dir": DIR, "glb": true}
      key every frame from the chunks, check the clip and export it: out_dir/report.json (as
      run.py writes it) and out_dir/clip.glb

Each job answers with one line, "@@DONE" or "@@FAIL <error>" ("finish" says "@@PLAYABLE" first,
once the .glb is written and before the checks run); everything else Blender prints is noise. The character's .blend is opened afresh for a job unless the last job, a "frames" one,
used the same file (posing frames keys nothing, and the compiler resets the pose it starts from).
"""
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bpy  # noqa: E402
from rigctl import checker, program, registry  # noqa: E402
from rigctl.compiler import Compiler  # noqa: E402

loaded = {"blend": None, "clean": False}


def open_blend(path):
    if loaded["blend"] != path or not loaded["clean"]:
        bpy.ops.wm.open_mainfile(filepath=path)
        loaded["blend"], loaded["clean"] = path, True


def frames(job):
    open_blend(job["blend"])
    rig = bpy.data.objects[registry.RIG]
    reg = registry.build(rig)
    prog, _ = program.validate(json.load(open(job["program"])), reg)
    c = Compiler(prog, reg)
    face = job.get("face", [])
    snaps = c.compile(job["frames"], face)
    json.dump({"frames": job["frames"] + face, "snaps": snaps, "report": c.report}, open(job["out"], "w"))


def _merge(reports):
    """One compile report from the chunks' reports: clamps summed, notes and pins joined."""
    rep = dict(reports[0]); rep["clamps"], rep["notes"], rep["pins"] = {}, [], []
    for r in reports:
        for k, v in r.get("clamps", {}).items():
            m = rep["clamps"].setdefault(k, {"frames": 0, "max": 0.0, "first": v["first"]})
            m["frames"] += v["frames"]; m["max"] = max(m["max"], v["max"]); m["first"] = min(m["first"], v["first"])
        rep["notes"] += [n for n in r.get("notes", []) if n not in rep["notes"]]
        rep["pins"] += r.get("pins", [])
        if "reach_targets" in r: rep.setdefault("reach_targets", {}).update(r["reach_targets"])
        if "reach_trunk" in r: rep.setdefault("reach_trunk", {}).update(r["reach_trunk"])
    return rep


def finish(job):
    open_blend(job["blend"]); loaded["clean"] = False
    out = job["out_dir"]; os.makedirs(out, exist_ok=True)
    rig = bpy.data.objects[registry.RIG]
    reg = registry.build(rig)
    prog, warn = program.validate(json.load(open(job["program"])), reg)
    t = time.time()
    c = Compiler(prog, reg)
    sc = bpy.context.scene
    sc.render.fps = c.fps; sc.frame_start = 1; sc.frame_end = c.frames
    parts = sorted((json.load(open(p)) for p in job["chunks"]), key=lambda x: min(x["frames"]))
    for part in parts:
        for f, snap in zip(part["frames"], part["snaps"]):
            c.insert(f + 1, snap)
    crep = _merge([p["report"] for p in parts])
    rep = {"program": job["program"], "valid": True, "warnings": warn, "compile": crep,
           "compile_s": round(time.time() - t + sum(job.get("frame_s", [0])), 1)}
    c.update()
    # The clip is playable once it is exported: say so, then check it (the checks take a couple of
    # seconds and only fill in the scores).
    if job.get("glb"):
        from rigctl import export
        export.glb(os.path.join(out, "clip.glb"), prog["clip"]["frames"])
        print("@@PLAYABLE", flush=True)
    t = time.time()
    res = checker.check(prog, crep, (1, prog["clip"]["frames"]))
    rep["check"] = res; rep["check_s"] = round(time.time() - t, 1); rep["reward"] = res["reward"]
    json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1, default=str)
    if job.get("dump"):                                   # every keyframe, for comparing compilers
        act = rig.animation_data.action
        keys = {f"{fc.data_path}[{fc.array_index}]": [round(k.co[1], 6) for k in fc.keyframe_points]
                for layer in act.layers for strip in layer.strips for cb in strip.channelbags for fc in cb.fcurves}
        json.dump({"keys": keys}, open(job["dump"], "w"))


def main():
    for line in sys.stdin:
        if not line.strip(): continue
        try:
            job = json.loads(line)
            {"frames": frames, "finish": finish}[job["op"]](job)
            print("@@DONE", flush=True)
        except Exception:
            loaded["clean"] = False
            print("@@FAIL " + json.dumps(traceback.format_exc()[-1500:]), flush=True)


main()
