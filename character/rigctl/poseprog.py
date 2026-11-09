"""The pose program: a pose beat's keys written as what each body part does (2026-11-02), the way
qwen-animate writes them from v13 on, and lowered to rig control values by the solver
(posesolve.py) before the compiler sees them. reports/Constrained LLM pose writing.md, stage 3.

A pose beat in the compact program, one key a line (its frame from the beat's start, then clauses
for the parts it sets, "; " between parts and ", " within one):

    pose +b0
     at 14 arm.R upper out, forearm up, level head
     at 20 arm.R forearm up_across
     at 26 arm.R forearm up_out
     at 32 head turn left; trunk bend forward

Parts and their words:
    arm.L / arm.R   dir D (where the hand is from the shoulder), upper D (the upper arm, where the
                    elbow points), forearm D (elbow to wrist), elbow straight|bent|tight,
                    hand LANDMARK (what the palm touches), level low|waist|chest|head|above
    hands           together|apart
    head            turn left|right|ahead, pitch up|down|level, tilt left|right|level
    trunk           bend forward|back|upright, side left|right|none, twist left|right|none
    shoulders       up|level
    posture         stand|crouch|kneel|sit|lie
D is up, down, forward, back, out (the arm's own side), across, or two of them joined by "_" (up_out,
down_forward, forward_across, ...): halfway between. Landmarks are posecat's on the skin (hip.L,
chest, face, ear.R, ...) and the other arm (upper_arm.L, elbow.R).

A key carries forward what earlier keys set: a part keeps its last words until a key changes them
(a key that gives both arms new words also lets go of "hands together"), and each key is solved
from the one before, so the body moves only as the words change.

  parse_key(line) / render_key(key)      the text of one key line <-> {"at", "spec"}
  ebnf_rules()                           the grammar's rules for the key lines (compact.ebnf)
  check(keys) -> [errors]                the words, before lowering
  lower(prog) -> prog                    every spec key solved to {"at", "set"} rig values
"""
import hashlib
import json
import os

from . import posecat

HERE = os.path.dirname(__file__)
PRIMARY = ["up", "down", "forward", "back", "out", "across"]
COMBOS = [f"{a}_{b}" for a in ("up", "down") for b in ("forward", "back", "out", "across")] + ["forward_out", "forward_across", "back_out", "back_across"]
DIRS = PRIMARY + COMBOS
ELBOWS = ["straight", "bent", "tight"]
LEVELS = ["low", "waist", "chest", "head", "above"]
LANDMARKS = sorted(posecat.LM) + ["upper_arm.L", "upper_arm.R", "elbow.L", "elbow.R"]
ARM = {"dir": DIRS, "upper": DIRS, "forearm": DIRS, "elbow": ELBOWS, "hand": LANDMARKS, "level": LEVELS}
HEAD = {"turn": ["left", "right", "ahead"], "pitch": ["up", "down", "level"], "tilt": ["left", "right", "level"]}
TRUNK = {"bend": ["forward", "back", "upright"], "side": ["left", "right", "none"], "twist": ["left", "right", "none"]}
POSTURES = ["stand", "crouch", "kneel", "sit", "lie"]
PARTS = ["arm.L", "arm.R", "hands", "head", "trunk", "shoulders", "posture"]
CACHE = os.path.join(HERE, "data", "pose_cache")


def parse_key(line):
    """' at 14 arm.R upper out, forearm up; head turn left' -> {"at": 14, "spec": {...}}"""
    w = line.strip().split(None, 2)
    assert w[0] == "at", line
    spec = {}
    for clause in (w[2] if len(w) > 2 else "").split("; "):
        part, _, rest = clause.partition(" ")
        if part in ("hands", "shoulders", "posture"): spec[part] = rest.strip(); continue
        spec[part] = {}
        for item in rest.split(", "):
            k, _, v = item.strip().partition(" ")
            spec[part][k] = v
    return {"at": int(w[1]), "spec": spec}


def render_key(k):
    out = []
    for part in PARTS:
        if part not in k["spec"]: continue
        v = k["spec"][part]
        out.append(f"{part} {v}" if isinstance(v, str) else f"{part} " + ", ".join(f"{a} {b}" for a, b in v.items()))
    return f' at {k["at"]} ' + "; ".join(out)


def check(keys, where="pose"):
    e = []
    for i, k in enumerate(keys):
        for part, v in k.get("spec", {}).items():
            w = f"{where}.keys[{i}].{part}"
            if part == "hands":
                if v not in ("together", "apart"): e.append(f"{w} must be together or apart")
            elif part == "shoulders":
                if v not in ("up", "level"): e.append(f"{w} must be up or level")
            elif part == "posture":
                if v not in POSTURES: e.append(f"{w} must be one of {POSTURES}")
            elif part in ("arm.L", "arm.R", "head", "trunk"):
                table = ARM if part.startswith("arm") else HEAD if part == "head" else TRUNK
                if not isinstance(v, dict) or not v: e.append(f"{w} says nothing"); continue
                for a, b in v.items():
                    if a not in table: e.append(f"{w}.{a} is not a word for {part}")
                    elif b not in table[a]: e.append(f"{w}.{a}={b} is not one of its words")
            else:
                e.append(f"{w} is not a part")
    return e


def _spec_text(spec):
    """The spec as posesolve reads it (its own line format)."""
    lines = []
    for part, v in spec.items():
        if isinstance(v, str): lines.append(f"{part}: {v}")
        else: lines.append(f"{part}: " + "; ".join(f"{a} {b}" for a, b in v.items()))
    return "\n".join(lines)


def _merge(acc, spec):
    """What the parts say now: a part's words replace its earlier ones (an arm's words merge)."""
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in acc.items()}
    if "arm.L" in spec and "arm.R" in spec and "hands" not in spec: out.pop("hands", None)
    for part, v in spec.items():
        if isinstance(v, dict) and isinstance(out.get(part), dict):
            merged = dict(out[part]); merged.update(v)
            # a new place for the hand or a new direction drops the words it would fight
            if "hand" in v: merged.pop("dir", None); merged.pop("level", None)
            if "dir" in v and "hand" not in v: merged.pop("hand", None)
            out[part] = merged
        else:
            out[part] = v
    return out


def solve_key(spec, base=None, posture="stand"):
    """The controls for one key's spec, solved from `base` (the key before); cached on disk. `posture`
    is the beat's (a posture beat before it), which a key's own posture word overrides."""
    from . import posesolve
    key = hashlib.sha1(json.dumps([spec, base, posesolve.VERSION] + ([posture] if posture != "stand" else []), sort_keys=True).encode()).hexdigest()[:20]
    path = os.path.join(CACHE, key + ".json")
    if os.path.exists(path): return json.load(open(path))
    S = posesolve.Solver(posesolve.parse(_spec_text(spec)), base); S.beat_posture = posture
    r = S.solve()
    out = {"controls": r["controls"], "met": r["met"], "violations": [list(v[:3]) for v in r["violations"]]}
    os.makedirs(CACHE, exist_ok=True)
    json.dump(out, open(path, "w"))
    return out


_REG = {}


def _ranges():
    """Each control's allowed rot_deg and loc_m (rigctl/data/registry.json), as validation reads them."""
    if not _REG: _REG.update(json.load(open(os.path.join(HERE, "data", "registry.json")))["controls"])
    return _REG


# controls a pose key may set (the root is the walker's; the rest are what posesolve moves or the
# posture library sets)
_SKIP = {"root"}


def lower(prog, report=None):
    """Every pose beat's spec keys solved, in order, each from the one before: {"at", "set"} keys in
    the values a pose writes (program.rig_pose: an upper arm's X raised from hanging)."""
    from .program import rig_pose, POSTURES, pose_back, pose_frames
    from . import posetime
    posture = "stand"
    fps = (prog.get("clip") or {}).get("fps", 24)
    for b in prog.get("beats", []):
        if b.get("verb") in POSTURES: posture = b["verb"]
        if b.get("verb") == "stand": posture = "stand"
        if b.get("verb") != "pose": continue
        keys = b.get("args", {}).get("keys", [])
        if not keys or "spec" not in keys[0]: continue
        acc, solved = {}, []
        base = None
        for k in keys:
            acc = _merge(acc, k["spec"])
            solved.append(solve_key(acc, base, posture)); base = solved[-1]["controls"]
        # standing, the beat is timed so no hand jumps past the checker's limit in a frame (posetime);
        # the poses stay as solved, the keys come later in proportion, and what follows moves with them
        if posture == "stand" and not any(isinstance(s.get("posture"), str) and s["posture"] != "stand" for s in (k["spec"] for k in keys)) \
                and all(isinstance(k.get("at"), int) for k in keys):
            ats = [k["at"] for k in keys]
            new = posetime.retime(solve_key({}, None)["controls"], [r["controls"] for r in solved], ats, fps,
                                  lambda a: pose_back({"keys": [{"at": a}]}, fps))
            if new != ats:
                old_len = pose_frames(b["args"], fps)
                for k, a in zip(keys, new): k["at"] = a
                delta = pose_frames(b["args"], fps) - old_len
                start = b.get("start") if isinstance(b.get("start"), int) else None
                if start is not None:
                    for o in prog["beats"]:
                        if o is not b and isinstance(o.get("start"), int) and o["start"] >= start + old_len: o["start"] += delta
                if isinstance(prog.get("clip", {}).get("frames"), int): prog["clip"]["frames"] += delta
                if report is not None:
                    report.setdefault("pose_retimed", []).append({"beat": b.get("id"), "from": ats, "to": new})
        acc, base, out = {}, None, []
        for k, r in zip(keys, solved):
            acc = _merge(acc, k["spec"])
            base = r["controls"]
            st = {}
            for c, v in r["controls"].items():
                if c in _SKIP: continue
                e = {}
                lim = _ranges().get(c, {})              # the solver's few tenths past a control's range: its edge
                clip = lambda xs, rng, d: [round(min(max(x, lo), hi), d) for x, (lo, hi) in zip(xs, rng)] if rng else [round(x, d) for x in xs]
                if "rot" in v: e["rot"] = rig_pose(c, "rot", clip(v["rot"], lim.get("rot_deg"), 1))   # the range is the rig's (a pose writes the upper arm raised from hanging)
                if "loc" in v: e["loc"] = clip(v["loc"], lim.get("loc_m"), 3)
                if e: st[c] = e
            out.append({"at": k["at"], "set": st})
            if report is not None:
                report.setdefault("pose_solve", []).append({"beat": b.get("id"), "at": k["at"], "met": r["met"], "violations": r["violations"]})
        # the standing body the solver starts from (posesolve.STAND_BODY: feet, torso, chest) is the
        # compiler's own, so it is left out of the keys, unless some key of the beat moves it (a lean's
        # hips, a bend's chest): then every key says where it is, and none leaves it held from another
        from .posesolve import STAND_BODY
        stand = {c: {a: [round(x, 3) for x in (rig_pose(c, a, v) if a == "rot" else v)] for a, v in e.items()} for c, e in STAND_BODY.items()}
        same = lambda c, e: all(max(abs(x - y) for x, y in zip(e.get(a, w), w)) < 0.05 for a, w in stand[c].items())
        moved = {c for k in out for c, e in k["set"].items() if c in stand and not same(c, e)}
        for k in out:
            for c in list(k["set"]):
                if c in stand and c not in moved: del k["set"][c]
        b["args"] = dict(b["args"], keys=out)
    return prog


def ebnf_rules():
    """The key lines as grammar rules (compact.ebnf takes them for "poselines")."""
    w = lambda xs: "(" + " | ".join('"' + x + '"' for x in xs) + ")"
    armitem = " | ".join(f'"{k} " {w(v)}' for k, v in ARM.items())
    return {
        "poselines": f'("\\n at " int " " poseclause ("; " poseclause){{0,6}}){{1,8}}',
        "poseclause": ('"arm." ("L" | "R") " " armitem (", " armitem){0,5} | "hands " ("together" | "apart") | "head " headitem (", " headitem){0,2}'
                       ' | "trunk " trunkitem (", " trunkitem){0,2} | "shoulders " ("up" | "level") | "posture " ' + w(POSTURES)),
        "armitem": armitem,
        "headitem": " | ".join(f'"{k} " {w(v)}' for k, v in HEAD.items()),
        "trunkitem": " | ".join(f'"{k} " {w(v)}' for k, v in TRUNK.items()),
    }
