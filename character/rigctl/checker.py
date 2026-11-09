"""Score an animated character scene. Works on any animation of this rig, compiled or not.

Everything is measured on the evaluated mesh and armature in world space; nothing the
program says about itself is trusted. The floor is z = 0 by definition, so moving a floor
object cannot help. Errors fail closed: a frame that cannot be evaluated scores zero.

Foot slide is measured on sole vertices that touch the floor in two consecutive frames, so a
foot rolling heel to toe (whose contact point legitimately moves) does not count as sliding,
while a planted foot dragging along the floor does.
"""
import math
from itertools import groupby

import bpy
from mathutils import Vector

from . import anatomy as A
from . import registry as R

GROUND = 0.004        # a sole vertex this close to z = 0 counts as touching the floor
LIMITS = {            # pass thresholds
    "slide_cm": 1.5, "penetration_mm": -5.0, "clip_cm": 4.0,
    # per frame jump and acceleration; a swinging foot legitimately moves much faster than the rest
    "jump_m": {"foot": 0.25, "other": 0.12}, "accel_ms2": {"foot": 150.0, "other": 60.0},
}
# Running, both feet leave the floor every step, and feet and hands move as fast as the reference rig's
# run moves them (in place: 0.44 m a frame and 318 m/s² at the feet, 0.40 and 289 at the
# hands); its frames are held to that and a third more, and the body's travel (0.15 m a frame).
RUN_LIMITS = {"jump_m": 0.6, "accel_ms2": 420.0}
ANATOMY_SLACK = 5.0   # degrees past a human range before a joint counts as broken


def check(prog=None, compile_report=None, frames=None, stride=1):
    sc = bpy.context.scene
    rig, body = bpy.data.objects[R.RIG], bpy.data.objects[R.BODY]
    frames = frames or (sc.frame_start, sc.frame_end)
    fs = list(range(frames[0], frames[1] + 1, stride))
    fps = sc.render.fps / stride
    rest = [v.co.copy() for v in body.data.vertices]
    mw = body.matrix_world
    sole = {"L": [i for i, c in enumerate(rest) if c.z < 0.03 and c.x > 0.02],
            "R": [i for i, c in enumerate(rest) if c.z < 0.03 and c.x < -0.02]}
    feet = {s: [i for i, c in enumerate(rest) if c.z < 0.12 and (c.x > 0.02 if s == "L" else c.x < -0.02)][::4]
            for s in "LR"}
    sample = list(range(0, len(rest), 7))
    track = ["DEF-hand.L", "DEF-hand.R", "DEF-foot.L", "DEF-foot.R", "DEF-spine.006", "DEF-spine", "DEF-spine.003"]
    out = {"frames": len(fs), "errors": []}
    per = []
    angles = A.Angles(rig)
    # Where the head looks: the axis of the head bone that points the way the body faces at rest
    # (-y). It is not the bone's own z: on a rig whose head bone follows the neck's slope, z points
    # 15-22 degrees below where the face looks.
    gaze = (R.Rest(rig)["DEF-spine.006"].matrix_local.to_3x3().inverted()
            @ (rig.matrix_world.to_3x3().inverted() @ Vector((0, -1, 0)))).normalized()
    for f in fs:
        sc.frame_set(f)
        try:
            dg = bpy.context.evaluated_depsgraph_get()
            ev = body.evaluated_get(dg); me = ev.to_mesh()
            co = [mw @ v.co for v in me.vertices]
            ev.to_mesh_clear()
        except Exception as e:            # fail closed
            out["errors"].append(f"frame {f}: {e}"); continue
        M = rig.matrix_world; P = R.Pose(rig)
        root = P["root"].matrix
        yaw = _yaw(root)
        per.append({
            "f": f, "co_sole": {s: [co[i] for i in sole[s]] for s in "LR"},
            "co_feet": {s: [co[i] for i in feet[s]] for s in "LR"},
            "com": sum((co[i] for i in sample), Vector()) / len(sample),
            "bones": {n: M @ P[n].head for n in track},
            "neck_base": M @ P["DEF-spine.004"].head, "hip": M @ P["DEF-spine"].head,
            "head_rel": (P["DEF-spine.003"].matrix.inverted() @ P["DEF-spine.006"].matrix).to_euler(),
            "head_fwd": (M.to_3x3() @ P["DEF-spine.006"].matrix.to_3x3()) @ gaze,
            "root": (M @ root.translation), "yaw": yaw, "joints": angles.read(),
        })
    if not per:
        out.update({"ok": False, "reward": 0.0}); return out
    m = {"drivers": len(rig.animation_data.drivers) if rig.animation_data else 0, "drivers_built": int(rig.get("rigctl_drivers", 1))}
    run = _run_frames(prog)
    m.update(_feet(per, fps, run))
    m.update(_posture(per, _bent_frames(prog, compile_report) | run))   # a run leans as the reference rig's sprint does
    m.update(_smooth(per, fps, run))
    m.update(_balance(per, fps, run))
    m.update(_anatomy(per))
    if prog: m.update(_task(per, prog, fps, compile_report or {}))
    if compile_report:
        m["clamp_frames"] = sum(c["frames"] for c in compile_report.get("clamps", {}).values())
        m["pins"] = _pins(compile_report, rig)
    out["metrics"] = m
    out["checks"] = _verdicts(m, prog, compile_report, frames, out["errors"])
    out["ok"] = all(c["pass"] for c in out["checks"].values())
    out["reward"] = _reward(m, out["checks"])
    return out


def _yaw(root):
    """The walker's heading: the root bone points +Y at rest, so yaw turns that axis to (−sin, cos)."""
    y = root.to_3x3() @ Vector((0, 1, 0))
    return math.atan2(-y.x, y.y)


def _anatomy(per):
    """Frames where a joint goes past its human range (anatomy.LIMITS, ANATOMY_SLACK degrees
    allowed for the flesh), and the worst of them in words."""
    bad, worst = 0, None
    for r in per:
        v = A.violations(r["joints"], ANATOMY_SLACK)
        if v: bad += 1
        for j, i, val, lim, what in v:
            if worst is None or abs(val - lim) > worst[0]:
                base, _, s = j.partition(".")
                worst = (abs(val - lim), f"{A.SIDE[s or '-']} {base} {what} {abs(val):.0f} deg (limit {abs(lim):.0f}) at frame {r['f']}".strip())
    return {"anatomy_frames": bad, "anatomy_worst": worst[1] if worst else None}


def _run_frames(prog):
    """The scene frames of the program's runs, jumps and postures: they leave the floor, move fast or
    rest on more than the soles (a scene frame is the program's plus one)."""
    from .program import RUNS, POSTURES
    out = {f + 1 for b in (prog or {}).get("beats", []) if b["verb"] in RUNS + ("jump",) + tuple(POSTURES) for f in range(b["start"], b["end"] + 1)}
    for b in (prog or {}).get("beats", []):             # a pose beat that sits, kneels, crouches or lies
        if b["verb"] == "pose" and _pose_sets(b, lambda c, e: c == "torso" and e.get("loc", [0, 0, 0])[2] < -0.10):
            out |= {f + 1 for f in range(b["start"], b["end"] + 1)}
    return out


def _pose_sets(b, test):
    """Whether any key of a pose beat sets a control as `test(control, {"rot", "loc"})` says."""
    return any(test(c, e) for k in b.get("args", {}).get("keys", []) for c, e in k.get("set", {}).items())


def _feet(per, fps, run=()):  # run: frames of runs, jumps and postures
    """Slide is the net drift of sole points while they stay on the floor. A foot that rolls
    puts each point down and lifts it again without moving it; a foot that skates drags its
    points along. For each vertex, each stretch of at least 3 frames on the floor gives one
    drift (first to last position); the score is the 95th percentile, so one stray vertex in
    the mesh cannot decide it, and the maximum is reported beside it."""
    drift = {"L": [], "R": []}
    pen, clip, both_air = 0.0, 9.9, 0
    for s in "LR":
        n = len(per[0]["co_sole"][s])
        for i in range(n):
            start = None
            for k, r in enumerate(per + [None]):
                on = r is not None and r["co_sole"][s][i].z < GROUND and r["f"] not in run   # getting down moves the feet on purpose
                if on and start is None: start = k
                if not on and start is not None:
                    if k - start >= 3:
                        a, b = per[start]["co_sole"][s][i], per[k - 1]["co_sole"][s][i]
                        drift[s].append(math.hypot(b.x - a.x, b.y - a.y))
                    start = None
    for r in per:
        lows = {s: min(p.z for p in r["co_sole"][s]) for s in "LR"}
        pen = min(pen, *lows.values())
        if min(lows.values()) > 0.03 and r["f"] not in run: both_air += 1   # a run flies every step, a body lying down rests on its back
        fl, fr = r["co_feet"]["L"], r["co_feet"]["R"]
        clip = min(clip, min(math.hypot(p.x - q.x, p.y - q.y) for p in fl[::3] for q in fr[::3]))
    allv = sorted(drift["L"] + drift["R"]) or [0.0]
    p95 = allv[min(len(allv) - 1, int(0.95 * len(allv)))]
    return {"slide_cm": round(100 * p95, 2), "slide_max_cm": round(100 * allv[-1], 2),
            "penetration_mm": round(1000 * pen, 1), "both_feet_air_frames": both_air,
            "foot_clip_cm": round(100 * clip, 1)}


def _bent_frames(prog, rep):
    """Scene frames where a reach bends the trunk on purpose (the compiler's reach_trunk), from
    its start until the trunk is back (the compiler's ret 12 + TRUNK_LAG 6 frames after the hold),
    and a pose beat's frames when its keys turn the trunk."""
    out = set()
    for b in (prog or {}).get("beats", []):
        if b["verb"] == "reach" and b["id"] in (rep or {}).get("reach_trunk", {}):
            out |= set(range(b["start"] + 1, b["end"] + b["args"].get("hold", 12) + 18 + 2))
        # a pose beat that turns the trunk (the solver writes chest, torso or hips only when it moves them)
        if b["verb"] == "pose" and _pose_sets(b, lambda c, e: c in ("chest", "torso", "hips") and "rot" in e):
            out |= set(range(b["start"] + 1, b["end"] + 2))
    return out


def _posture(per, bent=()):
    """The lean is the body's posture, so frames where a reach bends it are left out (no lean if
    every frame is bent: there is no posture to judge); the head's range counts every frame."""
    leans, nods = [], {"X": [], "Y": [], "Z": []}
    for r in per:
        d = r["neck_base"] - r["hip"]
        fwd = Vector((math.sin(r["yaw"]), -math.cos(r["yaw"]), 0))
        if r["f"] not in bent: leans.append(100 * d.dot(fwd))
        for i, ax in enumerate("XYZ"): nods[ax].append(math.degrees(r["head_rel"][i]))
    lean = {"lean_fwd_cm_mean": round(sum(leans) / len(leans), 1), "lean_fwd_cm_min": round(min(leans), 1),
            "lean_fwd_cm_max": round(max(leans), 1)} if leans else {}
    return {**lean,
            "neck_range_deg": {ax: round(max(v) - min(v), 1) for ax, v in nods.items()}}


def _smooth(per, fps, run=()):
    worst = {"foot": [0.0, 0.0], "other": [0.0, 0.0]}      # [jump m, accel m/s²], outside runs
    ran = [0.0, 0.0]                                      # the same in runs, every bone
    jerk_sum, n = 0.0, 0
    for name in per[0]["bones"]:
        grp = "foot" if "foot" in name else "other"
        xs = [r["bones"][name] for r in per]
        for i in range(1, len(xs)):
            w = ran if per[i]["f"] in run else worst[grp]
            w[0] = max(w[0], (xs[i] - xs[i - 1]).length)
        for i in range(2, len(xs)):
            w = ran if per[i]["f"] in run else worst[grp]
            w[1] = max(w[1], (xs[i] - 2 * xs[i - 1] + xs[i - 2]).length * fps * fps)
        for i in range(3, len(xs)):
            jerk_sum += (xs[i] - 3 * xs[i - 1] + 3 * xs[i - 2] - xs[i - 3]).length * fps ** 3; n += 1
    energy = 0.0
    for name in ("DEF-hand.L", "DEF-hand.R", "DEF-spine.006"):
        xs = [r["bones"][name] - r["root"] for r in per]
        energy += sum((xs[i] - xs[i - 1]).length for i in range(1, len(xs))) * fps / max(len(xs) - 1, 1)
    return {"max_jump_m": {g: round(v[0], 3) for g, v in worst.items()},
            "max_accel_ms2": {g: round(v[1], 1) for g, v in worst.items()},
            **({"run_max_jump_m": round(ran[0], 3), "run_max_accel_ms2": round(ran[1], 1)} if run else {}),
            "mean_jerk_ms3": round(jerk_sum / max(n, 1), 1), "body_energy_ms": round(energy / 3, 3)}


def _balance(per, fps, run=()):
    """Static balance: on frames where the body stands (neither travelling nor turning), the
    centre of mass is over the feet on the floor. Walking and quick turning steps balance
    dynamically, so their frames are left out, and so are a run's and a jump's (a jump's crouch
    and absorb are slow but loaded, not standing)."""
    inside, slow = 0, 0
    for a, b in zip(per, per[1:]):
        if b["f"] in run: continue
        if (b["root"] - a["root"]).length * fps > 0.05: continue
        if abs((b["yaw"] - a["yaw"] + math.pi) % (2 * math.pi) - math.pi) * fps > math.radians(15): continue
        pts = [p for s in "LR" for p in b["co_sole"][s] if p.z < 0.02]
        if len(pts) < 3: continue
        slow += 1
        if _in_hull((b["com"].x, b["com"].y), _hull([(p.x, p.y) for p in pts])): inside += 1
    return {"static_frames": slow, "balance_frac": round(inside / slow, 3) if slow else None}


def _task(per, prog, fps, rep):
    out = {}
    byf = {r["f"]: r for r in per}
    for b in prog["beats"]:
        f_end = b["end"] + 1
        r = byf.get(min(f_end, per[-1]["f"]))
        if b["verb"] in ("walk_to", "run_to") and r:
            tx, ty = b["args"]["target"]
            out[f"{b['id']}:walk_to_err_cm"] = round(100 * math.hypot(r["root"].x - tx, r["root"].y - ty), 1)
        if b["verb"] == "look_at":
            mid = byf.get((b["start"] + b["end"]) // 2 + 1)
            if mid:
                d = (Vector(b["args"]["target"]) - mid["bones"]["DEF-spine.006"]).normalized()
                ang = math.degrees(math.acos(max(-1, min(1, mid["head_fwd"].normalized().dot(d)))))
                out[f"{b['id']}:look_err_deg"] = round(ang, 1)
        if b["verb"] == "reach":
            fr = rep.get("reach_targets", {}).get(b["id"], {}).get("frames", [b["start"], b["end"]])
            side = b["args"]["side"]
            ds = [(byf[f + 1]["bones"][f"DEF-hand.{side}"] - Vector(b["args"]["target"])).length
                  for f in range(fr[0], fr[1] + 1) if f + 1 in byf]
            if ds: out[f"{b['id']}:reach_err_cm"] = round(100 * min(ds), 1)
    return out


def _pins(rep, rig):
    errs = []
    for p in rep.get("pins", []):
        bpy.context.scene.frame_set(p["f"] + 1)
        if p["ctrl"] in R.Pose(rig) and p["ch"] in ("loc", "rot") and p["mode"] == "abs":
            b = R.Pose(rig)[p["ctrl"]]
            if p["ch"] == "loc":
                errs.append(100 * (Vector(b.location) - Vector(p["v"])).length)
            else:
                q = b.rotation_quaternion if b.rotation_mode == "QUATERNION" else b.rotation_euler.to_quaternion()
                from mathutils import Euler
                want = Euler([math.radians(x) for x in p["v"]], "XYZ").to_quaternion()
                errs.append(math.degrees(q.rotation_difference(want).angle))
    return {"n": len(errs), "max_err": round(max(errs), 3) if errs else 0.0}


def _verdicts(m, prog, rep, frames, errors):
    c = {}
    def v(name, ok, val, need):
        c[name] = {"pass": bool(ok), "value": val, "need": need}
    v("evaluated", not errors, len(errors), "no frame errors")
    v("rig_intact", m.get("drivers", 0) >= m.get("drivers_built", 1), m.get("drivers"), f">= {m.get('drivers_built')} rig drivers present (as the rig file carries them)")
    v("foot_slide", m["slide_cm"] <= LIMITS["slide_cm"], m["slide_cm"], f"<= {LIMITS['slide_cm']} cm drift while on the floor (p95)")
    v("penetration", m["penetration_mm"] >= LIMITS["penetration_mm"], m["penetration_mm"], ">= -5 mm")
    v("airborne", m["both_feet_air_frames"] == 0, m["both_feet_air_frames"], "0 frames with both feet up (but running)")
    v("foot_clip", m["foot_clip_cm"] >= LIMITS["clip_cm"], m["foot_clip_cm"], f">= {LIMITS['clip_cm']} cm")
    if "lean_fwd_cm_mean" in m:
        v("lean", -8.0 <= m["lean_fwd_cm_mean"] <= 20.0, m["lean_fwd_cm_mean"], "-8..20 cm neck ahead of hips (proud leans back, a hunch forward)")
    ok = all(m["max_jump_m"][g] <= LIMITS["jump_m"][g] and m["max_accel_ms2"][g] <= LIMITS["accel_ms2"][g] for g in ("foot", "other"))
    v("no_pops", ok, [m["max_jump_m"], m["max_accel_ms2"]], f"jump <= {LIMITS['jump_m']} m, accel <= {LIMITS['accel_ms2']} m/s²")
    if "run_max_jump_m" in m:
        v("run_no_pops", m["run_max_jump_m"] <= RUN_LIMITS["jump_m"] and m["run_max_accel_ms2"] <= RUN_LIMITS["accel_ms2"],
          [m["run_max_jump_m"], m["run_max_accel_ms2"]], f"running: jump <= {RUN_LIMITS['jump_m']} m, accel <= {RUN_LIMITS['accel_ms2']} m/s²")
    v("anatomy", m["anatomy_frames"] == 0, m["anatomy_worst"] or 0, "every joint within its human range (anatomy.py)")
    v("moving", m["body_energy_ms"] >= 0.01, m["body_energy_ms"], ">= 0.01 m/s body motion")
    if m.get("balance_frac") is not None:
        v("balance", m["balance_frac"] >= 0.9, m["balance_frac"], ">= 90% of still frames")
    if prog:
        v("frame_range", frames[1] - frames[0] + 1 == prog["clip"]["frames"], frames, "clip.frames")
        for k, val in m.items():
            if k.endswith("walk_to_err_cm"): v(k, val <= 25, val, "<= 25 cm")
            if k.endswith("reach_err_cm"): v(k, val <= 5, val, "<= 5 cm")
            if k.endswith("look_err_deg"): v(k, val <= 25, val, "<= 25 deg")
    if rep and rep.get("pins"):
        v("pins", m["pins"]["max_err"] <= 0.5, m["pins"]["max_err"], "<= 0.5 (cm or deg)")
    return c


def _reward(m, checks):
    """Gate on evaluation, then a smooth score anchored to the thresholds (exp(-err/scale))."""
    if not checks["evaluated"]["pass"]: return 0.0
    terms = [math.exp(-m["slide_cm"] / 1.5), math.exp(-max(0, -m["penetration_mm"] - 5) / 5),
             math.exp(-m["both_feet_air_frames"] / 3), min(1.0, m["foot_clip_cm"] / 4.0),
             # the reference rig's walk carries the neck over the hips (-0.1 cm), its hero stand 3 cm behind
             math.exp(-max(0, abs(m["lean_fwd_cm_mean"]) - 6) / 4) if "lean_fwd_cm_mean" in m else 1.0,
             math.exp(-max(0, m["max_accel_ms2"]["other"] - 30) / 30),
             math.exp(-max(0, m["max_accel_ms2"]["foot"] - 100) / 50), min(1.0, m["body_energy_ms"] / 0.05)]
    if "run_max_accel_ms2" in m: terms.append(math.exp(-max(0, m["run_max_accel_ms2"] - 320) / 100))
    for k, val in m.items():
        if k.endswith("walk_to_err_cm"): terms.append(math.exp(-val / 25))
        if k.endswith("reach_err_cm"): terms.append(math.exp(-val / 5))
        if k.endswith("look_err_deg"): terms.append(math.exp(-val / 25))
    if "clamp_frames" in m: terms.append(math.exp(-m["clamp_frames"] / 50))
    return round(sum(terms) / len(terms), 4)


def _hull(pts):
    pts = sorted(set(pts))
    if len(pts) < 3: return pts
    def cross(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lo, hi = [], []
    for p in pts:
        while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0: lo.pop()
        lo.append(p)
    for p in reversed(pts):
        while len(hi) >= 2 and cross(hi[-2], hi[-1], p) <= 0: hi.pop()
        hi.append(p)
    return lo[:-1] + hi[:-1]


def _in_hull(p, h):
    if len(h) < 3: return False
    for a, b in zip(h, h[1:] + h[:1]):
        if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) < 0: return False
    return True
