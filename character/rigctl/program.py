"""The program format and its validator.

A program has three layers, all optional except the clip:

  beats       verbs on a timeline       {"id", "verb", "start" | "after", "dur"?, "args"}
  primitives  overrides for a beat's     {"beat", "type", "params"}
              generator parameters
  keys        sparse keys on named        {"ctrl", "ch", "f", "v", "mode", "ease", "pin", "blend", "hold"}
              controls, digits and face   (eased in/out over `blend` frames; `hold` keeps the last value)
  props       limb switches               {"prop": "upper_arm_parent.R:IK_FK", "f", "v"}
  style       the character's attitude    {param: value} for the whole clip (see GAIT);
              beats' moods and primitives override it where they apply

Validation never guesses: an unknown name, a value outside its range or a malformed field
is an error and the program does not compile. The compiler only clamps what the validator
let through on purpose (keys are range-checked here, so clamping there is a safety net).
"""
import math

from . import world

VERBS = {"walk", "walk_to", "run", "run_to", "jump", "crouch", "kneel", "sit", "lie", "pose", "turn", "idle", "look_at", "express", "hand", "reach"}
WALKS = ("walk", "walk_to", "run", "run_to")      # the verbs that travel
RUNS = ("run", "run_to")
# A jump, as an animator blocks one (the reference rig has none on two feet: its hop is a
# one-legged hero spring into a superhero landing): a crouch, the push up onto the toes, a
# ballistic flight (the hips rise `height` above standing, g 9.81), a landing on the toes that the
# legs absorb, and the stand again. Seconds: the crouch and the absorb deepen with the height.
JUMP_HEIGHT, JUMP_TOES, G = 0.3, 0.06, 9.81
# Postures on the floor (data/postures.json): the poses a body passes through going down, each
# reached in POSTURE_STEP seconds; it holds the last for the beat's dur less the way down and back up
# (POSTURE_HOLD unless given), and comes up the same way.
POSTURES = {"crouch": ["crouch"], "kneel": ["crouch", "kneel"], "sit": ["sit"], "lie": ["sit", "lie"]}   # through a crouch, a sit overbent the hips (155 deg)
POSTURE_STEP, POSTURE_HOLD = 0.6, 1.6


# A pose beat: key poses the program writes itself, each {control: {"rot": deg XYZ, "loc": m}} at a
# frame from the beat's start. The body goes from its own pose through them as a posture goes
# through the library's (each control eased from key to key; one a key leaves out keeps the body's
# own), holds the last POSE_HOLD s and comes back as slowly as it went to the first key (POSE_BACK s
# at least: an arm raised overhead in 0.9 s and dropped in 0.6 s whipped). The root is the walker's, so a
# pose leaves it out. A joint is turned no faster than a body turns one (POSE_DEG_S, POSE_M_S
# between two keys); how far it may go is the checker's anatomy (anatomy.py).
# An upper arm's X in a pose is how far the arm is raised from hanging at the side, as people say it
# (0 hangs, 90 is level, 180 overhead); the rig's own X is that less POSE_ARM_X (its rest is a T).
POSE_HOLD, POSE_BACK, POSE_KEYS = 0.4, 0.6, 8
POSE_ARM_X = 90.0
POSE_DEG_S = {"arm": 720.0, "hand": 720.0, "body": 300.0, "leg": 400.0}
POSE_M_S = 2.0


def _pose_group(ctrl):
    base = ctrl.split(".")[0]
    if base in ("shoulder", "upper_arm_fk", "forearm_fk", "hand_ik", "upper_arm_ik_target"): return "arm"
    if base in ("hand_fk", "palm"): return "hand"
    if base in ("torso", "hips", "chest", "neck", "head"): return "body"
    return "leg"


def pose_rig(ctrl, n, v):
    """A pose's value for a control's channel as the rig has it."""
    if n == "rot" and ctrl.split(".")[0] == "upper_arm_fk": return [v[0] - POSE_ARM_X, v[1], v[2]]
    return v


def rig_pose(ctrl, n, v):
    """The rig's value for a control's channel as a pose writes it."""
    if n == "rot" and ctrl.split(".")[0] == "upper_arm_fk": return [round(v[0] + POSE_ARM_X, 1), v[1], v[2]]
    return v


def pose_back(a, fps):
    return max(int(round(POSE_BACK * fps)), a["keys"][0]["at"])


def pose_frames(a, fps):
    return a["keys"][-1]["at"] + int(round(POSE_HOLD * fps)) + pose_back(a, fps)


def posture_frames(verb, fps):
    return int(round((2 * len(POSTURES[verb]) * POSTURE_STEP + POSTURE_HOLD) * fps))
JUMP_CROUCH, JUMP_PUSH, JUMP_LAND, JUMP_RECOVER = 0.32, 0.12, 0.22, 0.4


def jump_times(h):
    """(crouch, push, flight, land, recover) seconds for a jump that raises the hips h metres."""
    rise = max(0.02, h - JUMP_TOES)                   # the toes' push gives the first JUMP_TOES
    return JUMP_CROUCH + 0.3 * h, JUMP_PUSH, 2 * math.sqrt(2 * rise / G), JUMP_LAND + 0.2 * h, JUMP_RECOVER


def jump_frames(h, fps):
    return int(round(sum(jump_times(h)) * fps))
SHAPES = {"relaxed", "fist", "point", "open", "grip"}
MOODS = {"neutral", "tired", "confident", "sneaky", "happy", "angry", "sad"}
EASES = {"auto_clamped", "linear", "constant"}

# Gait parameters: (low, high, neutral).
GAIT = {
    "speed_ms": (0.3, 1.8, 0.98), "cadence_spm": (70, 140, 90), "hip_bob_cm": (0, 5, 1.6),
    "sway_cm": (0, 5, 2.8), "hip_twist_deg": (0, 15, 7), "chest_twist_deg": (0, 20, 9),
    "lean_deg": (-10, 20, 5), "arm_swing": (0, 1.5, 1.0), "arm_lag_f": (0, 8, 2),
    "neck_overlap_f": (0, 8, 3), "finger_curl": (0, 1, 0.3), "shoulder_fwd_deg": (-10, 30, 16),
    "lift_cm": (2, 15, 5), "heel_up_deg": (0, 30, 20), "toe_off_deg": (0, 50, 42),
    # posture and face: these shape the body standing or walking
    "spine_curve_deg": (-10, 35, 0), "shoulder_drop_deg": (-10, 25, 0), "head_drop_deg": (-20, 35, 0),
    "elbow_bend_deg": (0, 60, 16), "lid_droop": (0, 0.7, 0), "brow_knit": (0, 1, 0), "jaw_slack": (0, 0.5, 0),
    "blink_rate_pm": (4, 40, 15), "blink_len": (0.6, 3.0, 1.0), "breath_depth": (0, 3, 1),
}
# Style parameters that apply to a whole stretch of the clip, walking or not.
STYLE = ["lean_deg", "shoulder_fwd_deg", "spine_curve_deg", "shoulder_drop_deg", "head_drop_deg",
         "elbow_bend_deg", "lid_droop", "brow_knit", "jaw_slack", "blink_rate_pm", "blink_len",
         "breath_depth", "finger_curl"]
MOOD_GAIT = {
    "neutral": {},
    "tired": {"speed_ms": 0.55, "cadence_spm": 80, "hip_bob_cm": 0.8, "sway_cm": 3.4, "lean_deg": 6,
              "arm_swing": 0.35, "shoulder_fwd_deg": 22, "lift_cm": 3, "heel_up_deg": 10, "toe_off_deg": 22,
              "spine_curve_deg": 20, "shoulder_drop_deg": 14, "head_drop_deg": 18, "elbow_bend_deg": 8,
              "lid_droop": 0.42, "brow_knit": 0.35, "jaw_slack": 0.15, "blink_rate_pm": 9, "blink_len": 2.2,
              "breath_depth": 2.0, "finger_curl": 0.2, "neck_overlap_f": 5},
    "confident": {"speed_ms": 1.05, "cadence_spm": 104, "chest_twist_deg": 12, "lean_deg": 2,
                  "arm_swing": 1.2, "shoulder_fwd_deg": 4, "lift_cm": 6},
    "sneaky": {"speed_ms": 0.45, "cadence_spm": 78, "hip_bob_cm": 0.6, "lean_deg": 14, "arm_swing": 0.3,
               "lift_cm": 8, "heel_up_deg": 6, "finger_curl": 0.5},
    "happy": {"speed_ms": 1.0, "cadence_spm": 110, "hip_bob_cm": 2.6, "arm_swing": 1.25, "lift_cm": 7},
    "angry": {"speed_ms": 1.2, "cadence_spm": 112, "lean_deg": 10, "arm_swing": 0.9, "finger_curl": 0.85,
              "shoulder_fwd_deg": 20, "chest_twist_deg": 6},
    "sad": {"speed_ms": 0.55, "cadence_spm": 80, "hip_bob_cm": 0.8, "lean_deg": 11, "arm_swing": 0.35,
            "shoulder_fwd_deg": 26, "lift_cm": 3},
}
# A run is a walk at the reference rig's run: its cadence, its speed unless the beat gives one,
# and the run flag that tells the planner and compiler to run it (moods shape it no further).
RUN_GAIT = {"speed_ms": 4.5, "cadence_spm": 150, "run": 1.0}
RUN_SPEED = (2.5, 6.0)
DEFAULT_DUR = {"turn": 20, "idle": 48, "look_at": 36, "express": 36, "hand": 12, "reach": 30}
# What each verb does and its args' ranges, as _check_args holds them (for people: the page's
# motion catalog).
VERB_DOC = {
    "walk": ("walks straight ahead", {"distance": "0.2–50 m", "speed": "0.3–1.8 m/s"}),
    "walk_to": ("walks to a spot on the floor, turning to it first", {"target": "x, y m from where it stands", "speed": "0.3–1.8 m/s"}),
    "run": ("runs straight ahead, as the reference rig's run", {"distance": "0.2–50 m", "speed": "2.5–6 m/s, 4.5 unless given"}),
    "run_to": ("runs to a spot on the floor, turning to it first", {"target": "x, y m from where it stands", "speed": "2.5–6 m/s, 4.5 unless given"}),
    "jump": ("jumps: a crouch, a push, the flight and a landing it absorbs", {"height": "0.1–0.6 m the hips rise, 0.3 unless given", "distance": "0–1.5 m forward, 0 in place"}),
    "crouch": ("crouches down, holds and stands up", {}),
    "kneel": ("kneels on both knees (through a crouch), holds and stands up", {}),
    "sit": ("sits on the floor, holds and stands up", {}),
    "lie": ("lies down on its back (through a sit), holds and gets up", {}),
    "pose": ("moves through key poses it sets on named controls, holds the last and comes back", {
        "keys": f"up to {POSE_KEYS}: {{at: frames from the beat's start, set: {{control: {{rot: deg XYZ, loc: m}}}}}}; "
                "an upper arm's X is degrees raised from hanging (0 hangs, 90 level, 180 overhead)"}),
    "turn": ("turns on the spot", {"deg": "−180–180, + to the left"}),
    "idle": ("stands: breathing, weight shifts, blinks", {}),
    "look_at": ("turns eyes, head and chest to a point", {"target": "x, y, z m", "amount": "0–1, how much of the body joins in"}),
    "express": ("a face, rising to its apex and fading", {"keys": "blink, smile, brows_up, frown, jaw_open: 0–1 each",
                                                         "onset": "frames", "apex": "frames", "offset": "frames"}),
    "hand": ("shapes a hand", {"side": "L, R or both", "shape": "relaxed, fist, point, open or grip", "amount": "0–1"}),
    "reach": ("reaches a hand to a point and holds it there", {"side": "L or R", "target": "x, y, z m", "hold": "0–240 frames"}),
}


class ProgramError(Exception):
    pass


def validate(prog, reg):
    """Return (normalised program, warnings) or raise ProgramError listing every problem."""
    err, warn = [], []
    if not isinstance(prog, dict):
        raise ProgramError(["program must be a JSON object"])
    clip = prog.get("clip") or {}
    fps = clip.get("fps", 24); frames = clip.get("frames")
    if fps not in (24, 25, 30): err.append("clip.fps must be 24, 25 or 30")
    if not isinstance(frames, int) or not 1 <= frames <= 1440: err.append("clip.frames must be an int in 1..1440")
    var = clip.get("variation", 0.3)
    if not _num(var) or not 0 <= var <= 1: err.append("clip.variation must be in 0..1")
    seed = clip.get("seed", 0)
    if not isinstance(seed, int): err.append("clip.seed must be an int")

    beats, ids = [], {}
    for i, b in enumerate(prog.get("beats", [])):
        where = f"beats[{i}]"
        if not isinstance(b, dict): err.append(f"{where} must be an object"); continue
        verb = b.get("verb")
        if verb not in VERBS: err.append(f"{where}.verb {verb!r} is not one of {sorted(VERBS)}"); continue
        bid = b.get("id", f"b{i}")
        if bid in ids: err.append(f"{where}.id {bid!r} repeats")
        args = b.get("args", {}) or {}
        err += _check_args(where, verb, args, reg, fps)
        for act in [verb] + ([f"hand:{args.get('shape')}"] if verb == "hand" else []):
            if not world.allows(act): err.append(f"{where}: {act} needs {', '.join(world.lacks(act))}, and the world has none")
        if "start" in b and "after" in b: err.append(f"{where}: give start or after, not both")
        if "start" in b and (not isinstance(b["start"], int) or b["start"] < 0):
            err.append(f"{where}.start must be a frame >= 0")
        if "after" in b and b["after"] not in ids: err.append(f"{where}.after {b['after']!r} names no earlier beat")
        if "dur" in b and (not isinstance(b["dur"], int) or b["dur"] < 1):
            err.append(f"{where}.dur must be a positive int")
        nb = {"id": bid, "verb": verb, "args": args, "start": b.get("start"), "after": b.get("after"),
              "dur": b.get("dur")}
        ids[bid] = nb; beats.append(nb)

    prims = {}
    for i, p in enumerate(prog.get("primitives", [])):
        where = f"primitives[{i}]"
        if p.get("beat") not in ids: err.append(f"{where}.beat names no beat"); continue
        if p.get("type") not in ("gait", "style"): err.append(f"{where}.type must be 'gait' or 'style'"); continue
        for k, v in (p.get("params") or {}).items():
            if k not in GAIT: err.append(f"{where}.params.{k} is not a gait parameter"); continue
            lo, hi, _ = GAIT[k]
            if not _num(v) or not lo <= v <= hi: err.append(f"{where}.params.{k}={v} outside {lo}..{hi}")
        prims[p["beat"]] = dict(p.get("params") or {})

    keys = []
    for i, k in enumerate(prog.get("keys", [])):
        e = _check_key(f"keys[{i}]", k, reg, frames if isinstance(frames, int) else 10 ** 6)
        if e: err += e
        else: keys.append(_norm_key(k))
    props = []
    for i, p in enumerate(prog.get("props", [])):
        where = f"props[{i}]"
        if p.get("prop") not in reg["props"]: err.append(f"{where}.prop {p.get('prop')!r} unknown"); continue
        if not isinstance(p.get("f"), int) or not _num(p.get("v")) or not 0 <= p["v"] <= 1:
            err.append(f"{where} needs an int f and v in 0..1"); continue
        props.append(dict(p))
    if err:
        raise ProgramError(err)

    style = {}
    for k, v in (prog.get("style") or {}).items():
        if k not in GAIT: err.append(f"style.{k} is not a style or gait parameter"); continue
        lo, hi, _ = GAIT[k]
        if not _num(v) or not lo <= v <= hi: err.append(f"style.{k}={v} outside {lo}..{hi}")
        else: style[k] = v
    if err:
        raise ProgramError(err)
    _schedule(beats, fps, prims, warn, style)
    end = max([b["end"] for b in beats] + [k["f"] for k in keys] + [0])
    if end > frames: warn.append(f"timeline reaches frame {end} but clip.frames is {frames}; the tail is cut")
    return {"plan": prog.get("plan", ""), "clip": {"fps": fps, "frames": frames, "variation": var, "seed": seed},
            "beats": beats, "prims": prims, "keys": keys, "props": props, "style": style}, warn


def base_style(style=None):
    """Neutral defaults with the clip's top-level style applied: the character's attitude."""
    g = {k: v[2] for k, v in GAIT.items()}
    g.update(style or {})
    g["run"] = 0.0
    return g


def gait_params(beat, prims, style=None):
    g = base_style(style)
    g.update(MOOD_GAIT.get(beat["args"].get("mood", "neutral"), {}))
    if beat["verb"] in RUNS: g.update(RUN_GAIT)
    if "speed" in beat["args"]: g["speed_ms"] = beat["args"]["speed"]
    g.update(prims.get(beat["id"], {}))
    return g


RUN_RAMP_STEPS = 2     # a run speeds up over two steps and slows over two (a walk: one)
TURN_FRAMES = 20
TURN_STEP_DEG = 45     # a turn re-squares the feet in shuffle-step pairs of at most this much
STAY_M = 0.15          # a walk_to target this close is where the walker already stands


def turn_frames(deg):
    """Frames a turn takes when the program does not say: 20 for a quarter turn or less, then
    14 per pair of shuffle steps (a half turn is four pairs, 2.3 s at 24 fps)."""
    return max(TURN_FRAMES, 14 * math.ceil(abs(deg) / TURN_STEP_DEG))


def _schedule(beats, fps, prims, warn, style=None):
    """Give every beat a start and end frame. Locomotion beats (walk, walk_to, turn) move the
    root one after another; the root's position and heading are tracked through them so that
    walk_to knows how far it is and which way to face. Yaw 0 faces −Y; forward is (sin, −cos)."""
    x = y = yaw = 0.0
    busy_until = 0
    by_id = {}
    for b in beats:
        if b["after"]: b["start"] = by_id[b["after"]]["end"]
        if b["start"] is None: b["start"] = 0
        a = b["args"]
        v = b["verb"]
        if v in WALKS + ("turn", "jump"):
            if b["start"] < busy_until:
                warn.append(f"beat {b['id']} starts at {b['start']} while the previous locomotion beat runs "
                            f"to {busy_until}; it is moved to {busy_until}")
                b["start"] = busy_until
            b["from"] = (x, y, yaw)
            if v == "jump":
                b["turn_deg"], b["distance"], b["jump_h"] = 0.0, a.get("distance", 0.0), a.get("height", JUMP_HEIGHT)
                b["dur"] = b["dur"] or jump_frames(b["jump_h"], fps)
                x += math.sin(yaw) * b["distance"]; y += -math.cos(yaw) * b["distance"]
            elif v == "turn":
                b["turn_deg"] = a["deg"]; b["distance"] = 0.0
                b["dur"] = b["dur"] or turn_frames(a["deg"])
                yaw += math.radians(a["deg"])
            else:
                g = gait_params(b, prims, style); b["gait"] = g
                if v in ("walk_to", "run_to"):
                    dx, dy = a["target"][0] - x, a["target"][1] - y
                    dist = math.hypot(dx, dy)
                    if dist < STAY_M:                     # no turn to face a target underfoot
                        dist, dx, dy = 0.0, 0.0, 0.0
                    want = math.atan2(dx, -dy) if dist > 1e-6 else yaw
                    turn = math.degrees(_wrap(want - yaw))
                    b["turn_deg"] = turn if abs(turn) > 10 else 0.0
                    yaw = want                            # a smaller turn is taken setting off (planner)
                else:
                    dist = a.get("distance", 3.0); b["turn_deg"] = 0.0
                b["distance"] = dist
                turn_f = turn_frames(b["turn_deg"]) if b["turn_deg"] else 0
                b["dur"] = b["dur"] or (turn_f + _walk_frames(dist, g, fps))
                x += math.sin(yaw) * dist; y += -math.cos(yaw) * dist
            b["to"] = (x, y, yaw)
            busy_until = b["start"] + b["dur"]
        elif v == "pose":                                  # the body stays put, as in a posture
            if b["start"] < busy_until:
                warn.append(f"beat {b['id']} starts at {b['start']} while the previous locomotion beat runs "
                            f"to {busy_until}; it is moved to {busy_until}")
                b["start"] = busy_until
            b["dur"] = max(b["dur"] or 0, pose_frames(a, fps))
            busy_until = b["start"] + b["dur"]
        elif v in POSTURES:                                # the body stays put, so nothing travels meanwhile
            if b["start"] < busy_until:
                warn.append(f"beat {b['id']} starts at {b['start']} while the previous locomotion beat runs "
                            f"to {busy_until}; it is moved to {busy_until}")
                b["start"] = busy_until
            b["dur"] = max(b["dur"] or 0, posture_frames(v, fps) - int(round(POSTURE_HOLD * fps)) + 1) if b["dur"] else posture_frames(v, fps)
            busy_until = b["start"] + b["dur"]
        elif b["dur"] is None:
            b["dur"] = a.get("dur", DEFAULT_DUR[v])
        b["end"] = b["start"] + b["dur"]
        by_id[b["id"]] = b


def _walk_frames(dist, g, fps):
    # cruise distance plus about one step each to start and stop
    ramp = RUN_RAMP_STEPS if g.get("run") else 1                   # steps it takes to reach speed, and to stop
    close = 2 if g.get("run") else 1                               # the steps that square the feet (slower after a run)
    return int(round((0.25 + dist / g["speed_ms"] + (ramp + close) * 60.0 / g["cadence_spm"]) * fps))   # 0.25 s weight-shift lead-in


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def beats_by_id(beats):
    return {b["id"]: b for b in beats}


def _check_args(where, verb, a, reg=None, fps=24):
    e = []
    def rng(name, lo, hi, req=False):
        if name not in a:
            if req: e.append(f"{where}.args.{name} is required")
            return
        if not _num(a[name]) or not lo <= a[name] <= hi: e.append(f"{where}.args.{name}={a[name]} outside {lo}..{hi}")
    def vec(name, n, req=False, lim=50.0):
        if name not in a:
            if req: e.append(f"{where}.args.{name} is required")
            return
        v = a[name]
        if not isinstance(v, list) or len(v) != n or not all(_num(x) and abs(x) <= lim for x in v):
            e.append(f"{where}.args.{name} must be {n} numbers within ±{lim}")
    if "mood" in a and a["mood"] not in MOODS: e.append(f"{where}.args.mood not in {sorted(MOODS)}")
    if verb == "walk": rng("distance", 0.2, 50); rng("speed", 0.3, 1.8)
    elif verb == "walk_to": vec("target", 2, req=True); rng("speed", 0.3, 1.8)
    elif verb == "run": rng("distance", 0.2, 50); rng("speed", *RUN_SPEED)
    elif verb == "run_to": vec("target", 2, req=True); rng("speed", *RUN_SPEED)
    elif verb == "jump": rng("height", 0.1, 0.6); rng("distance", 0.0, 1.5)
    elif verb == "turn": rng("deg", -180, 180, req=True)
    elif verb == "look_at": vec("target", 3, req=True); rng("amount", 0, 1)
    elif verb == "express":
        ks = a.get("keys")
        if not isinstance(ks, dict) or not ks: e.append(f"{where}.args.keys must map face keys to 0..1")
        else:
            for k, v in ks.items():
                if k not in ("blink", "smile", "brows_up", "frown", "jaw_open"): e.append(f"{where}.args.keys.{k} unknown")
                elif not _num(v) or not 0 <= v <= 1: e.append(f"{where}.args.keys.{k} outside 0..1")
        for t in ("onset", "apex", "offset"):
            if t in a and (not isinstance(a[t], int) or a[t] < 0): e.append(f"{where}.args.{t} must be a frame offset >= 0")
    elif verb == "hand":
        if a.get("side", "both") not in ("L", "R", "both"): e.append(f"{where}.args.side must be L, R or both")
        if a.get("shape") not in SHAPES: e.append(f"{where}.args.shape must be one of {sorted(SHAPES)}")
        rng("amount", 0, 1); vec("target", 3)        # what a point points at (set by the server's aim)
    elif verb == "reach":
        if a.get("side") not in ("L", "R"): e.append(f"{where}.args.side must be L or R")
        vec("target", 3, req=True); rng("hold", 0, 240)
    elif verb == "pose":
        e += _check_pose(where, a, reg, fps)
    return e


def _check_pose(where, a, reg, fps):
    e = []
    ks = a.get("keys")
    if not isinstance(ks, list) or not 1 <= len(ks) <= POSE_KEYS:
        return [f"{where}.args.keys must be 1..{POSE_KEYS} key poses"]
    if all(isinstance(k, dict) and "spec" in k for k in ks):   # in body words: the words now, the angles once solved
        from . import poseprog
        last = 0
        for i, k in enumerate(ks):
            if not isinstance(k.get("at"), int) or k["at"] <= last: e.append(f"{where}.args.keys[{i}].at must be a frame after the last key's ({last})")
            else: last = k["at"]
        return e + poseprog.check(ks, f"{where}.args")
    last, prev = 0, {}
    for i, k in enumerate(ks):
        w = f"{where}.args.keys[{i}]"
        at, st = k.get("at") if isinstance(k, dict) else None, k.get("set") if isinstance(k, dict) else None
        if not isinstance(at, int) or at <= last: e.append(f"{w}.at must be a frame after the last key's ({last})"); continue
        if not isinstance(st, dict) or not st: e.append(f"{w}.set must set at least one control"); continue
        for ctrl, ch in st.items():
            c = (reg or {}).get("controls", {}).get(ctrl)
            if ctrl == "root" or c is None: e.append(f"{w}.set.{ctrl} is not a control a pose sets"); continue
            if not isinstance(ch, dict) or not ch or set(ch) - set(c["channels"]):
                e.append(f"{w}.set.{ctrl} must give {' and/or '.join(c['channels'])}"); continue
            for n, v in ch.items():
                lim = c["rot_deg"] if n == "rot" else c["loc_m"]
                if not isinstance(v, list) or len(v) != 3 or not all(_num(x) for x in v):
                    e.append(f"{w}.set.{ctrl}.{n} must be 3 numbers"); continue
                v = pose_rig(ctrl, n, v)
                for j, (x, (lo, hi)) in enumerate(zip(v, lim)):
                    if not lo <= x <= hi: e.append(f"{w}.set.{ctrl}.{n}[{j}]={x} outside {lo}..{hi}")
                if (ctrl, n) in prev:                      # as fast as a body moves it, no faster
                    pat, pv = prev[(ctrl, n)]; dt = (at - pat) / fps
                    if n == "rot":
                        deg = _rot_between(pv, v); top = POSE_DEG_S[_pose_group(ctrl)]
                        if deg / dt > top: e.append(f"{w}.set.{ctrl}: turns {deg:.0f} deg in {dt:.2f} s, faster than {top:.0f} deg/s")
                    else:
                        m = math.dist(pv, v)
                        if m / dt > POSE_M_S: e.append(f"{w}.set.{ctrl}: moves {m:.2f} m in {dt:.2f} s, faster than {POSE_M_S} m/s")
                prev[(ctrl, n)] = (at, v)
        last = at if isinstance(at, int) else last
    return e


def _rot_between(a, b):
    """Degrees between two XYZ Euler rotations (degrees)."""
    def q(v):
        x, y, z = (math.radians(t) / 2 for t in v)
        cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
        return (cx * cy * cz + sx * sy * sz, sx * cy * cz - cx * sy * sz, cx * sy * cz + sx * cy * sz, cx * cy * sz - sx * sy * cz)
    d = abs(sum(s * t for s, t in zip(q(a), q(b))))
    return math.degrees(2 * math.acos(min(1.0, d)))


def _check_key(where, k, reg, frames):
    e = []
    ctrl, ch, f, v = k.get("ctrl"), k.get("ch"), k.get("f"), k.get("v")
    if not isinstance(f, int) or not 0 <= f <= frames: e.append(f"{where}.f must be a frame in 0..{frames}")
    if k.get("mode", "abs") not in ("abs", "add"): e.append(f"{where}.mode must be abs or add")
    if k.get("ease", "auto_clamped") not in EASES: e.append(f"{where}.ease must be one of {sorted(EASES)}")
    if not isinstance(k.get("blend", 6), int) or not 0 <= k.get("blend", 6) <= 48: e.append(f"{where}.blend must be 0..48 frames")
    if not isinstance(k.get("hold", False), bool): e.append(f"{where}.hold must be true or false")
    if ctrl in reg["face"]:
        if ch not in (None, "value"): e.append(f"{where}.ch for a face key is 'value'")
        if not _num(v) or not 0 <= v <= 1: e.append(f"{where}.v outside 0..1")
    elif ctrl in reg["digits"]:
        d = reg["digits"][ctrl]
        if ch == "curl":
            if not _num(v) or not d["curl"][0] <= v <= d["curl"][1]: e.append(f"{where}.v curl outside 0..1")
        elif ch == "spread":
            if not _num(v) or abs(v) > d["spread_deg"][1]: e.append(f"{where}.v spread outside ±{d['spread_deg'][1]}")
        else: e.append(f"{where}.ch for a digit is curl or spread")
    elif ctrl in reg["controls"]:
        c = reg["controls"][ctrl]
        if ch not in c["channels"]: e.append(f"{where}.ch {ch!r} not in {c['channels']} for {ctrl}")
        elif not isinstance(v, list) or len(v) != 3 or not all(_num(x) for x in v): e.append(f"{where}.v must be 3 numbers")
        else:
            lim = c["rot_deg"] if ch == "rot" else c["loc_m"]
            for i, (x, (lo, hi)) in enumerate(zip(v, lim)):
                if not lo <= x <= hi: e.append(f"{where}.v[{i}]={x} outside {lo}..{hi} for {ctrl}.{ch}")
    else:
        e.append(f"{where}.ctrl {ctrl!r} is not a control, digit or face key")
    return e


def _norm_key(k):
    return {"ctrl": k["ctrl"], "ch": k.get("ch", "value"), "f": k["f"], "v": k["v"],
            "mode": k.get("mode", "abs"), "ease": k.get("ease", "auto_clamped"), "pin": bool(k.get("pin", False)),
            "blend": k.get("blend", 6), "hold": k.get("hold", False)}


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
