"""Compile a validated program onto the character's rig, one frame at a time.

Order each frame: root, hips/torso, chest/neck/head, feet (from the footstep plan), FK arms,
fingers and hand shapes, face, look_at, reach, then the program's own keys and props, then
clamping to the registry limits. Every control the compiler drives is keyed on every frame,
so what the checker measures is exactly what was compiled.

The compiler returns a correction report: every clamp it applied, every key it could not
honour, and the pins the checker must verify.
"""
import json
import math
import os
import random
import time
from bisect import bisect_right
from types import SimpleNamespace

import bpy
from mathutils import Matrix, Quaternion, Vector, Euler

from . import anatomy
from . import registry as R
from .planner import Plan, STANCE, ease, fwd, lerp
from .program import WALKS, base_style


HERE = os.path.dirname(__file__)
# The foot leaves the floor as the reference rig's walk has it: the heel peels over the last 31%
# of the cycle before the toes leave, the foot rolls over its toes another TOE_ROLL_DEG for the
# first TOE_ROLL of the swing, then lifts at once, still heel up (AIR_PITCH_DEG more): the ankle is
# highest a fifth of the way through the swing (28 cm above the floor in the reference rig, three times
# its height standing), not halfway, and the lift is LIFT_GAIN times the style's lift_cm.
PEEL = 0.31
TOE_ROLL_DEG, TOE_ROLL = 12.0, 0.15
AIR_PITCH_DEG = 2.0
# The heel peels only as far as the leg trails: a foot the body has passed leans the shin forward
# and the heel goes up by that lean and PEEL_POINT_DEG of the ankle pointing (a push-off points it
# about 20 deg); a foot the body has not passed (the step that stops a walk) peels no more than the
# ankle points. At a fixed 54 deg, a stopping step pointed the ankle 60 deg (anatomy allows 50).
PEEL_POINT_DEG = 30.0
# A foot in the air is pitched within the ankle's range against its shin (anatomy.py's, less a
# margin): the reference rig's run swings the foot toes up 40 deg in the world, which on our more bent
# knee flexed the ankle up 53 deg. The shin is found as the leg's IK will put it (two bones, the
# knee bending forward).
SWING_ANKLE = (-44.0, 34.0)   # degrees from rest: pointed .. flexed up
LIFT_GAIN, LIFT_EARLY = 1.8, 0.45     # lift shape sin(pi * v**LIFT_EARLY): highest at v = 0.5**(1/LIFT_EARLY)
_CREASES = {}                                         # (.blend, modified, vertex count) -> crease table, kept by a warm worker
CREASE_CACHE = os.path.join(HERE, "data", "crease_cache.json")   # the same, kept across workers
_AFFECTS = {}                                         # (.blend, modified) -> what each control moves, kept by a warm worker
AFFECTS_CACHE = os.path.join(HERE, "data", "affects_cache.json")   # the same, kept across workers
FINGERS = ["f_index", "f_middle", "f_ring", "f_pinky"]
SHAPES = {   # per digit: curl 0..1 (thumb last), and spread toward the thumb side
    "relaxed": None,
    "fist": ([1, 1, 1, 1], 1.0, [3, 0, 3, 7]),
    "grip": ([0.62, 0.66, 0.7, 0.74], 0.6, [2, 0, 2, 5]),
    "point": ([0.03, 0.95, 1, 1], 0.85, [0, 0, 3, 7]),
    "open": ([0.02, 0.02, 0.03, 0.04], 0.0, [-8, 0, 6, 12]),
}


# Key-pose transitions as an animator times them (research_notes/Two character Disney quality
# animation/principles_procedural.md): succession, the head leading and the arm following from the
# shoulder out (Neff & Fiume 2003: each joint further down the chain a little later; frames at 24 fps),
# and a big move taking a small anticipation and settling past its pose (easeInOutBack, its strength
# growing with the turn: none under ANTIC_MIN_DEG, POSE_BACK_S at ANTIC_FULL_DEG and over).
# controls the rig bends on one axis and twists along the bone; a pose beat blends their angles
HINGES = ("forearm_fk",)
POSE_LAG = {"head": -2, "neck": -1, "chest": 0, "shoulder": 1, "upper_arm_fk": 2, "forearm_fk": 4, "hand_fk": 6}
ANTIC_MIN_DEG, ANTIC_FULL_DEG, POSE_BACK_S = 30.0, 90.0, 0.45
# an anticipation takes time of its own (Williams): squeezed into a quick move it only makes the move
# quicker, so it fades out as the move's mean speed rises from ANTIC_SLOW to ANTIC_FAST deg/s
ANTIC_SLOW, ANTIC_FAST = 60.0, 120.0
ANTIC_CTRLS = ("upper_arm_fk", "forearm_fk", "hand_fk", "chest", "head", "neck", "shoulder")


def pose_curve(u, deg, ctrl, span=1.0):
    """Progress 0..1 between two key poses `span` seconds apart (it may dip below 0 or pass 1 a little)."""
    u = min(max(u, 0.0), 1.0)
    s = 0.0
    if ctrl.split(".")[0] in ANTIC_CTRLS and deg > ANTIC_MIN_DEG:
        s = POSE_BACK_S * min(1.0, (deg - ANTIC_MIN_DEG) / (ANTIC_FULL_DEG - ANTIC_MIN_DEG))
        s *= min(1.0, max(0.0, (ANTIC_FAST - deg / max(span, 1e-3)) / (ANTIC_FAST - ANTIC_SLOW)))
    smooth = u * u * (3 - 2 * u)
    if s <= 0: return smooth
    # the full back-ease (easeInOutBack at POSE_BACK_S) blended in by the anticipation's strength, so a
    # faint anticipation is a faint change: the cubic alone peaks at 3x the mean speed at any strength
    c2 = 1.70158 * POSE_BACK_S * 1.525
    if u < 0.5: back = (2 * u) ** 2 * ((c2 + 1) * 2 * u - c2) / 2
    else: back = ((2 * u - 2) ** 2 * ((c2 + 1) * (2 * u - 2) + c2) + 2) / 2
    a = s / POSE_BACK_S
    return smooth + (back - smooth) * a


def pitch(deg):
    return Matrix.Rotation(math.radians(-deg), 4, "X")       # + lifts the toes


def rz(yaw):
    return Matrix.Rotation(yaw, 4, "Z")


def _blend(Ma, Mb, w):
    """Matrix w of the way from Ma to Mb (translation lerp, rotation slerp)."""
    la, ra, sa = Ma.decompose(); lb, rb, sb = Mb.decompose()
    return Matrix.LocRotScale(la.lerp(lb, w), ra.slerp(rb, w), sa.lerp(sb, w))


def env(t, a, b, c, d):
    if t <= a or t >= d: return 0.0
    if t < b: return ease((t - a) / max(b - a, 1e-6))
    if t <= c: return 1.0
    return 1 - ease((t - c) / max(d - c, 1e-6))


SHIFT_MARGIN = 0.035   # the centre of mass stops this far inside a standing sole's centre
LEAD_S, LEAD_MAX = 0.35, 30.0   # the head turns this far ahead of the body (seconds), at most this much (degrees)
TRUNK_LAG = 6          # frames the trunk keeps moving after the hand arrives, and returns after it
TRUNK_MAX = 75         # the deepest a reach bends the trunk (degrees)
REACH_FRAC = 0.95      # a reach keeps the arm this much short of straight
BACK_K = 0.45          # hips go back this share of how far the bend takes the shoulder forward
HIP_SHARE = 0.45       # of a reach's bend, the share taken at the hips (torso); the chest takes the rest
# The body as a reference walk carries it (measured on the same rig): walking, the
# pelvis rides low enough that the knees bend as the reference rig's do (5-65 deg, 32 on average) and bobs
# 7.4 cm, lowest just after each heel strike and highest over the planted foot, where the knee is
# all but straight; standing, the knees stay soft. The trunk is held upright (hips to neck at 0 deg; the
# moods' lean tipped it 5 deg forward of that). The arms hang 8 deg out from the body.
WALK_SINK = 0.038      # metres the pelvis drops at a full walk, besides the bob's own
BOB_GAIN = 2.3         # the bob is this many times the style's hip_bob_cm
BOB_LOW = 0.13         # stride phase of the bob's low (0 = left heel strike): the body lands on a straight leg, then sinks
STAND_SINK = 0.012     # metres it drops standing
UPRIGHT_DEG = -5.0     # degrees the trunk tips forward of the moods' lean, walking or not
ARM_DOWN_DEG = 82.0    # degrees the arm is lowered from the T-pose (90 would hang it straight down)
ARM_PITCH_DEG = -14.0  # the arm's pitch about which it swings (the reference rig swings it about 2 deg forward of hanging)
# The arm swing as the reference rig has it: 53 deg from back to front at arm_swing 1, the elbow
# all but straight at the back (13 deg) and bent most at the front (40), and the arm coming in at
# either end of the swing and out from the body as it passes it (1 to 14 deg).
ARM_SWING_DEG = 25.0   # the upper arm's swing either way of its pitch, at arm_swing 1
ELBOW_BACK, ELBOW_SWING = -11.0, 27.0   # walking, the elbow's bend at the back of the swing and its rise to the front
ARM_IN_DEG, ARM_PASS_DEG = 3.0, 8.0     # walking, how far the arm comes in at the ends and out as it passes the body
# The trunk's motion in the reference rig's walk, as shares of what the style's numbers gave: the pelvis
# twists 12 deg either way and the chest counter-twists to 12 the other (47 deg between them), the
# pelvis drops 5 deg to the swinging side and the shoulders tilt 3.4, the hips sway 3.4 cm; the
# wrist flexes 3-17 deg (10 on average), and the feet turn out 3 deg.
HIP_TWIST_K, CHEST_TWIST_K = 0.85, 0.66
HIP_DROP_K, SHOULDER_TILT_K, SWAY_K = 0.62, 0.64, 0.62
WRIST_FLEX_DEG = 10.0  # walking, the wrist's mean flex beyond its swing
TOE_OUT_DEG = 3.0
# Standing empty-handed: the world holds nothing (rigctl/world.py), so the arms hang at the sides,
# elbows and wrists soft, the fingers at their relaxed curl, the chest a little out and the head a
# little down. (The reference rig's hero stand flares the arms 26 deg and
# bends the elbows 51 deg with the hands in loose fists: without a prop it reads as about to take
# hold of something.) A mood that slumps or leans takes less of it (hero(): 1 neutral, 0 tired).
STAND_ARM_OUT, STAND_ARM_FWD = 2.0, 0.0
STAND_ELBOW, STAND_WRIST, STAND_CHEST, STAND_HEAD = 4.0, 2.0, 2.0, 2.0
STAND_CURL = 0.0       # on top of the style's finger_curl (0.3 is relaxed, 1 a full fist)
# The run is the reference rig's own (data/run_cycle.json): the
# body above the legs and the arms take its keys, sampled at the run's phase, and a foot in the
# air takes its path (the heel kicked up behind, the knee driven through) on its way between the
# plants the planner put down, which keep it from sliding. The reference rig's foot is down 3.5 frames
# of its 24, ours as long as the body takes to pass over it (planner.stance_of), so the cycle is
# read faster in the air and slower on the ground: its strike and lift fall on ours.
RUN = json.load(open(os.path.join(HERE, "data", "run_cycle.json")))
RUN_HEEL_DEG, RUN_TOE_OFF_DEG = 28.0, 45.0
# The body and arms read the reference rig's cycle 40% of the way from its own pace to the one that puts
# its strikes on ours: all the way, its 8.5 frames in the air pass in our 4.5 at 4.5 m/s, and the
# arms pump at twice its speed (13 m/s hands).
RUN_WARP = 0.4      # the reference rig lands toes 28 deg up and leaves heel ~50 up
# The jump's body (program.jump_times for its timing): the hips dip JUMP_DIP + JUMP_DIP_K * height
# in the crouch and the absorb, the trunk tips forward into the crouch and the landing, and the
# arms swing back in the crouch, drive forward and up in the push and settle in the air. The heels
# peel JUMP_PEEL_DEG in the push (the hips rise program.JUMP_TOES on the toes) and the body lands
# toes first, JUMP_LAND_DEG, the knees tucking JUMP_TUCK in the air.
JUMP_DIP, JUMP_DIP_K, JUMP_ABSORB_K = 0.10, 0.25, 0.25
JUMP_LEAN_CROUCH, JUMP_LEAN_LAND, JUMP_ARM_BACK, JUMP_ARM_UP = 30.0, 22.0, 45.0, 70.0
JUMP_BACK_K = 0.33      # crouching, the hips go back a third as far as they drop: the body stays over the feet
JUMP_PEEL_DEG, JUMP_LAND_DEG, JUMP_TUCK = 35.0, 20.0, 0.12
from .program import JUMP_TOES as JUMP_TOES_M
POSTURE_LIB = json.load(open(os.path.join(HERE, "data", "postures.json")))
ANKLE_LEAD = 0.06       # metres ahead of the hip joint the ankle plants (the reference rig lands 0.39 m ahead of the hips and leaves 0.45 behind)

class Compiler:
    def __init__(self, prog, reg, face_json=os.path.join(HERE, "data", "face_keys.json")):
        self.p, self.reg = prog, reg
        self.rig = bpy.data.objects[R.RIG]; self.body = bpy.data.objects[R.BODY]
        self.pb = R.Pose(self.rig)                       # bones by the program's names (registry.BONE)
        self.dirty = set()                                # bones posed since the last depsgraph update
        self.posed = {}                                   # the pose matrix last set on each dirty bone
        self.stale_all = False                            # something no dependency covers changed
        self.fps = prog["clip"]["fps"]; self.frames = prog["clip"]["frames"]
        self.var = prog["clip"]["variation"]
        self.rng = random.Random(prog["clip"]["seed"])
        self.plan = Plan(prog["beats"], self.fps, prog["clip"]["seed"], self.var)
        self.report = {"clamps": {}, "notes": [], "pins": [], "end_s": self.plan.end_time()}
        self._reset_all()
        self.ctrl = sorted(reg["controls"])
        self.digit_bones = [f"{d}.0{i}.{s}" for d in R.DIGITS for i in (1, 2, 3) for s in "LR"]
        self.rest = {n: self.pb[n].matrix.copy() for n in self.ctrl if n in self.pb}
        self.moves = self._affects()
        self.ankle = {s: self.pb[f"foot_ik.{s}"].head.copy() for s in "LR"}
        # feet land at a natural step width (the rig body's own, ±8.8 cm), not wherever the rest pose
        # put them: an A-pose body stands with its feet far apart
        self.step_x = {s: (1 if s == "L" else -1) * min(abs(self.ankle[s].x), 0.095) - self.ankle[s].x for s in "LR"}
        # the ankle lands under the hip: the ankle is the deform foot's head (the reference rig's IK foot
        # control lies mid-foot on the floor, 12 cm ahead of it, and planting that under the hip left
        # every stance behind the body)
        self.under_hip = (self.pb["thigh_ik.L"].head - self.pb["DEF-foot.L"].head).y - ANKLE_LEAD
        self._pivots()
        # each character's sole sits a little below its ankle bone's floor; lift it to z = 0
        soles = [(self.body.matrix_world @ v.co).z for v in self.body.data.vertices
                 if (self.body.matrix_world @ v.co).z < 0.05]
        self.sole = Vector((0, 0, 0.001 - min(soles)))
        # how far each sole's centre sits outside its ankle (feet turned out): the hips must
        # carry the body over the sole, not the ankle, before the other foot lifts
        self.sole_out = {}
        floor = [self.body.matrix_world @ v.co for v in self.body.data.vertices]
        floor = [c for c in floor if c.z < min(soles) + 0.02]
        for s, side in (("L", 1), ("R", -1)):
            xs = [side * c.x for c in floor if side * c.x > 0]
            self.sole_out[s] = max(0.0, sum(xs) / len(xs) - abs(self.ankle[s].x)) if xs else 0.0
        self.report["sole_out_cm"] = {s: round(100 * v, 1) for s, v in self.sole_out.items()}
        self.crease = self._crease_table()
        # the head bone's axis that looks where the face looks: the body faces -y at rest
        self.gaze = (R.Rest(self.rig)["DEF-spine.006"].matrix_local.to_3x3().inverted()
                     @ (self.rig.matrix_world.to_3x3().inverted() @ Vector((0, -1, 0)))).normalized()
        # arms are lowered relative to where they rest (T-pose for the rig body, A-pose for others)
        self.arm_rest_down = {}
        for s in "LR":
            b = R.Rest(self.rig)[f"DEF-upper_arm.{s}"]
            d = (b.tail_local - b.head_local).normalized()
            self.arm_rest_down[s] = math.degrees(math.asin(max(-1.0, min(1.0, -d.z))))
        self.axes = self._digit_axes()
        self.face = self._face_keys(face_json)
        self.tracks = self._tracks(prog["keys"])
        self.props = self._prop_tracks(prog["props"])
        self.blinks = self._blink_times()
        self.reach_start = {}
        self.reach_bend = {}
        self.gesture_start = {}
        B = R.Rest(self.rig)
        self.arm_len = (B["DEF-hand.L"].head_local - B["DEF-upper_arm.L"].head_local).length
        self.arm_phase = {"L": -0.25, "R": 0.25}
        self.hips_local = R.Rest(self.rig)["DEF-spine"].head_local.copy()     # the hips in the root's frame at rest
        self.foot_rest = {f"foot_ik.{s}": R.Rest(self.rig)[f"foot_ik.{s}"].head_local.copy() for s in "LR"}
        from .program import RUNS
        runs = any(b["verb"] in RUNS for b in prog["beats"])
        self.run_bones = [b for b in RUN["body"] + RUN["arm"] if b in self.pb and b not in self.ctrl] if runs else []
        self.head_follow = 1.0                            # the rig's HEAD-FOLLOW as this frame has it

    # -- setup ------------------------------------------------------------------------------
    def _reset_all(self):
        # Drop any old action but keep animation_data itself: it holds Rigify's 100+ drivers
        # (IK/FK switches, follow and stretch), and clearing it would silently cut them.
        ad = self.rig.animation_data
        if ad:
            ad.action = None
            for tr in list(ad.nla_tracks): ad.nla_tracks.remove(tr)
        for b in self.pb:
            b.location = (0, 0, 0); b.rotation_quaternion = (1, 0, 0, 0)
            b.rotation_euler = (0, 0, 0); b.scale = (1, 1, 1)
        for s in "LR":
            self.pb[f"thigh_parent.{s}"]["IK_FK"] = 0.0
            self.pb[f"upper_arm_parent.{s}"]["IK_FK"] = 1.0
        bpy.context.view_layer.update()

    def _pivots(self):
        """Rolling-contact tables from this mesh's own sole. As the foot pitches, it turns about
        whichever sole vertex is lowest at that moment, the way a wheel turns about its contact
        point, so nothing touching the floor slides. The heel table covers toes-up angles over
        the whole sole; the ball table covers heel-up angles over the sole behind the toe joint
        (the toes themselves are held flat on the floor separately)."""
        mw = self.body.matrix_world
        a = self.ankle["L"]; joint = self.pb["toe_ik.L"].head
        sole = [mw @ v.co for v in self.body.data.vertices if (mw @ v.co).x > 0.02 and (mw @ v.co).z < 0.05]
        q_all = [Vector((p.y - a.y, p.z - a.z)) for p in sole]
        q_ball = [q for q, p in zip(q_all, sole) if p.y > joint.y]
        self.roll = {1: self._roll_table(q_all, 1), -1: self._roll_table(q_ball, -1)}
        self.report["roll"] = {"heel_at_20": [round(x, 3) for x in self._roll(20)],
                               "ball_at_-38": [round(x, 3) for x in self._roll(-38)]}

    def _crease_table(self, step=10, top=80):
        """How far below the floor this mesh's sole sinks with the heel up by each angle and the
        toes flat: skinning pinches the crease at the toe joint, and the sharper the bend the
        lower it goes (on a short-toed foot it passes 5 mm at a push-off). The foot is posed on
        a plant at the origin and the left sole's lowest vertex is read from the evaluated mesh,
        against the same foot flat; _planted lifts the ankle by the dip."""
        path = bpy.data.filepath
        key = f"{path}|{os.path.getmtime(path) if path and os.path.exists(path) else 0}|{len(self.body.data.vertices)}"
        if key in _CREASES: return _CREASES[key]
        try: disk = json.load(open(CREASE_CACHE))
        except (OSError, ValueError): disk = {}
        if key in disk:
            _CREASES[key] = (disk[key], step)
            self.report["crease_mm"] = [round(1000 * x, 1) for x in disk[key]]
            return _CREASES[key]
        t0 = time.time()
        mw = self.body.matrix_world
        sole = [i for i, v in enumerate(self.body.data.vertices) if (mw @ v.co).x > 0.02 and (mw @ v.co).z < 0.05]
        plant = SimpleNamespace(x=0.0, y=0.0, yaw=0.0, wide=0.0)
        A = self.ankle["L"]
        flat = self._planted("L", plant, 0.0)
        lows = []
        for ang in range(0, top + 1, step):
            ankle = self._planted("L", plant, -ang, crease=False)
            self.pb["foot_ik.L"].matrix = Matrix.Translation(ankle) @ pitch(-ang) @ Matrix.Translation(-A) @ self.rest["foot_ik.L"]
            bpy.context.view_layer.update()
            self.pb["toe_ik.L"].matrix = Matrix.Translation(flat) @ Matrix.Translation(-A) @ self.rest["toe_ik.L"]
            bpy.context.view_layer.update()
            ev = self.body.evaluated_get(bpy.context.evaluated_depsgraph_get()); me = ev.to_mesh()
            lows.append(min((mw @ me.vertices[i].co).z for i in sole))
            ev.to_mesh_clear()
        table = [0.0]                                     # dip below the flat foot's sole, never less for a sharper bend
        for low in lows[1:]: table.append(max(table[-1], lows[0] - low))
        for n in ("foot_ik.L", "toe_ik.L"):
            b = self.pb[n]; b.location = (0, 0, 0); b.rotation_quaternion = (1, 0, 0, 0); b.rotation_euler = (0, 0, 0)
        bpy.context.view_layer.update()
        _CREASES[key] = (table, step)
        self.report["crease_mm"] = [round(1000 * x, 1) for x in table]
        self.report["crease_s"] = round(time.time() - t0, 2)
        if path:
            try:                                          # another worker may be writing it too: last one wins, same values
                disk = {**disk, key: table}
                tmp = f"{CREASE_CACHE}.{os.getpid()}"
                json.dump(disk, open(tmp, "w")); os.replace(tmp, CREASE_CACHE)
            except OSError:
                pass
        return _CREASES[key]

    def _crease(self, bend):
        if bend <= 0: return 0.0
        table, step = self.crease
        i = min(bend / step, len(table) - 1.001)
        k = int(i); u = i - k
        return lerp(table[k], table[k + 1], u)

    @staticmethod
    def _roll_table(pts, sign, step=0.5, top=60.0):
        def rot(v, deg):                       # pitch(deg) in the (y, z) plane; + lifts the toes
            r = math.radians(-deg); c, s = math.cos(r), math.sin(r)
            return Vector((c * v.x - s * v.y, s * v.x + c * v.y))
        o, table, ang = Vector((0.0, 0.0)), [Vector((0.0, 0.0))], 0.0
        while ang < top:
            contact = min((o + rot(q, sign * ang) for q in pts), key=lambda p: p.y)
            d = sign * step
            o = contact + rot(o - contact, d)
            ang += step; table.append(o.copy())
        return table, step

    def _roll(self, ang):
        """Ankle offset (forward y, up z) for a foot pitched by `ang` degrees in rolling contact."""
        if abs(ang) < 1e-6: return Vector((0, 0, 0))
        table, step = self.roll[1 if ang > 0 else -1]
        i = min(abs(ang) / step, len(table) - 1.001)
        k = int(i); u = i - k
        o = table[k].lerp(table[k + 1], u)
        return Vector((0, o.x, o.y))

    def _digit_axes(self):
        out, bones = {}, R.Rest(self.rig)
        for s in "LR":
            palm = Vector(self.rig.get(f"palm_normal_{s}", (0, 0, -1)))      # palms down in the rig body's T-pose
            mid = (bones[f"f_middle.03.{s}"].tail_local - bones[f"f_middle.01.{s}"].head_local).normalized()
            idx = (bones[f"f_index.03.{s}"].tail_local - bones[f"f_index.01.{s}"].head_local).normalized()
            for d in R.DIGITS:
                for i in (1, 2, 3):
                    b = bones[f"{d}.0{i}.{s}"]
                    v = (b.tail_local - b.head_local).normalized()
                    inv = b.matrix_local.to_3x3().inverted()
                    close = v.cross(idx if d == "thumb" else mid)
                    out[f"{d}.0{i}.{s}"] = ((inv @ v.cross(palm)).normalized(),
                                            (inv @ close).normalized() if close.length > 1e-3 else None)
        return out

    def _face_keys(self, path):
        offsets = json.load(open(path))
        if max(int(i) for d in offsets.values() for i in d) >= len(self.body.data.vertices) or self.rig.get("rigctl_character"):
            self.report["notes"].append("no face shape keys for this character; face channels are not animated")
            return None
        me = self.body.data
        if not me.shape_keys: self.body.shape_key_add(name="Basis", from_mix=False)
        basis = me.shape_keys.key_blocks["Basis"]
        for name, d in offsets.items():
            k = me.shape_keys.key_blocks.get(name) or self.body.shape_key_add(name=name, from_mix=False)
            for i, c in d.items():
                k.data[int(i)].co = basis.data[int(i)].co + Vector(c)
            k.slider_min, k.slider_max, k.value = 0.0, 1.0, 0.0
        if me.shape_keys.animation_data: me.shape_keys.animation_data.action = None
        return me.shape_keys.key_blocks

    def _blink_times(self):
        """Blinks at the style's rate (per minute), jittered by the seed."""
        t, out, end = 0.8 + self.rng.uniform(0, 1.2), [], self.frames / self.fps
        while t < end:
            g = self._posture_gait(t)
            mean = 60.0 / g["blink_rate_pm"]
            out.append(t); t += self.rng.uniform(0.6 * mean, 1.4 * mean)
        return out

    def _tracks(self, keys):
        tr = {}
        for k in keys:
            tr.setdefault((k["ctrl"], k["ch"], k["mode"]), []).append(k)
        for v in tr.values(): v.sort(key=lambda k: k["f"])
        return tr

    def _prop_tracks(self, props):
        tr = {}
        for p in props: tr.setdefault(p["prop"], []).append((p["f"], p["v"]))
        for v in tr.values(): v.sort()
        return tr

    # -- helpers ----------------------------------------------------------------------------
    # A depsgraph update evaluates the whole rig (about 4.5 ms), so it runs only when a pose is
    # about to be read or set from a stale one: a bone's evaluated pose is stale when a control
    # posed since the last update moves it. What each control moves is measured on the rig
    # (_affects), not read from its constraints: the reference rig also carries bones by drivers and
    # bendy-bone handles, and a dependency missed there reads a pose a step behind.
    def update(self):
        bpy.context.view_layer.update(); self.dirty.clear(); self.posed.clear(); self.stale_all = False

    def _affects(self):
        """{control's bone: the bones a change to it moves}, for every control and digit the
        compiler poses, measured with the arms in FK and in IK (a hand's IK control moves nothing
        while its arm is in FK), cached per .blend."""
        path = bpy.data.filepath
        key = f"{path}|{os.path.getmtime(path) if path and os.path.exists(path) else 0}"
        if key in _AFFECTS: return _AFFECTS[key]
        try: disk = json.load(open(AFFECTS_CACHE))
        except (OSError, ValueError): disk = {}
        if key in disk:
            _AFFECTS[key] = {k: set(v) for k, v in disk[key].items()}
            return _AFFECTS[key]
        t0 = time.time()
        rig, P = self.rig, self.rig.pose.bones
        names = sorted({R.BONE.get(n, n) for n in self.ctrl + self.digit_bones})
        out = {n: set() for n in names}
        nudge = Quaternion((1, 0.5, 0.25), math.radians(6))
        for arms in (1.0, 0.0):                       # Rigify's IK_FK: 1 FK, 0 IK
            for s in "LR": self.pb[f"upper_arm_parent.{s}"]["IK_FK"] = arms
            bpy.context.view_layer.update()
            before = {b.name: b.matrix.copy() for b in P}
            for n in names:
                b = P[n]; loc, q = b.location.copy(), b.rotation_quaternion.copy()
                b.location = loc + Vector((0.02, 0.015, 0.01)); b.rotation_quaternion = nudge @ q
                bpy.context.view_layer.update()
                for o in P:
                    M0, M1 = before[o.name], o.matrix
                    if any(abs(M0[r][c] - M1[r][c]) > 1e-6 for r in range(3) for c in range(4)): out[n].add(o.name)
                b.location, b.rotation_quaternion = loc, q
                bpy.context.view_layer.update()
        for s in "LR": self.pb[f"upper_arm_parent.{s}"]["IK_FK"] = 1.0
        bpy.context.view_layer.update()
        _AFFECTS[key] = out
        self.report["affects_s"] = round(time.time() - t0, 2)
        if path:
            try:                                      # another worker may be writing it too: last one wins, same values
                disk = {**disk, key: {k: sorted(v) for k, v in out.items()}}
                tmp = f"{AFFECTS_CACHE}.{os.getpid()}"
                json.dump(disk, open(tmp, "w")); os.replace(tmp, AFFECTS_CACHE)
            except OSError:
                pass
        return out

    def _moved(self, name, but=None):
        """Whether a control posed since the last update (other than `but`) moves bone `name`."""
        b = R.BONE.get(name, name)
        return any(b in self.moves.get(R.BONE.get(d, d), ()) for d in self.dirty if d != but)

    def fresh(self, name, own=True):
        """Update first if the pose of `name` (or, own=False, only what it hangs from) is stale."""
        if self.stale_all: self.update(); return
        if not self.dirty: return
        if own: stale = self._moved(name)
        else:
            par = self.pb[name].parent
            stale = par is not None and self._moved(par.name)
        if stale: self.update()

    def place(self, name, M):
        self.fresh(name, own=False); self.pb[name].matrix = M; self._posed(name, M)

    def _posed(self, name, M):
        self.dirty.add(name)
        # A bone with no constraints evaluates to exactly the matrix set on it, so a second turn
        # of it can start from that instead of waiting for an update.
        if not any(c.enabled and c.influence for c in self.pb[name].constraints): self.posed[name] = M.copy()
        else: self.posed.pop(name, None)

    def pose_of(self, name):
        """The bone's pose matrix: the one just set on it if that is still exact, else evaluated."""
        if not self.stale_all and name in self.posed and not self._moved(name, but=name): return self.posed[name]
        self.fresh(name); return self.pb[name].matrix

    def turn(self, name, axis, deg, about=None, frame=None):
        """Rotate a control about an axis given in the walker's frame (yaw applied), or in `frame`."""
        if not deg: return
        M = self.pose_of(name)
        h = M.to_translation() if about is None else about
        ax = ((self.yawm.to_3x3() if frame is None else frame) @ Vector({"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[axis]))
        Rm = Matrix.Translation(h) @ Matrix.Rotation(math.radians(deg), 4, ax) @ Matrix.Translation(-h)
        M = Rm @ M
        self.pb[name].matrix = M; self._posed(name, M)

    def curl(self, name, deg, close=0.0):
        flex, toward = self.axes[name]
        q = Quaternion(flex, math.radians(deg))
        if toward is not None and close: q = Quaternion(toward, math.radians(close)) @ q
        self.pb[name].rotation_quaternion = q

    # -- the frame --------------------------------------------------------------------------
    def _calibrate_sole(self):
        """Pose the standing first frame, measure the lowest sole vertex and lift the feet so it
        sits 1 mm above the floor. The rest mesh is not enough: placing the feet at step width
        tilts the shins, and sole vertices weighted partly to the shin dip with them."""
        self.frame(0)
        dg = bpy.context.evaluated_depsgraph_get(); ev = self.body.evaluated_get(dg); me = ev.to_mesh()
        mw = self.body.matrix_world
        low = min((mw @ v.co).z for v, r in zip(me.vertices, self.body.data.vertices) if (mw @ r.co).z < 0.05)
        ev.to_mesh_clear()
        self.sole = self.sole + Vector((0, 0, 0.001 - low))
        self.report["sole_lift_mm"] = round(1000 * self.sole.z, 1)

    def compile(self, frames=None, face=()):
        """Pose and key every frame; or, given a range of frame indices, pose only those and return
        their values (snapshots) for `insert` to key elsewhere, then the face's values alone on
        the frame indices `face` (face_snapshot: blinks are too quick to leave to the curves
        between posed frames, and the face needs no posing). Frames do not depend on each other
        (each starts from the reset pose), so a clip can be posed in parallel ranges. The one
        state that crosses frames, a reach's starting hand pose, is seeded from the reach's
        first frame when a range starts inside a reach."""
        sc = bpy.context.scene
        sc.render.fps = self.fps; sc.frame_start = 1; sc.frame_end = self.frames
        sc.frame_set(1); self._calibrate_sole()
        # Posing needs only the armature, but every update would also deform the meshes, so
        # they are left out of the depsgraph while the frames are posed.
        meshes = [o for o in sc.objects if o.type == "MESH" and not o.hide_viewport]
        for o in meshes: o.hide_viewport = True
        out = []
        try:
            todo = range(self.frames) if frames is None else frames
            if frames is not None and len(frames):
                for f0 in sorted(b["start"] for b in self.p["beats"] if b["verb"] == "reach" and b["start"] < frames[0]):
                    sc.frame_set(f0 + 1); self.frame(f0)
            for f in todo:
                sc.frame_set(f + 1)
                self.frame(f)
                if frames is None: self.key(f + 1)
                else: out.append(self.snapshot())
        finally:
            for o in meshes: o.hide_viewport = False
            self.update()
        self.report["frames"] = self.frames
        return self.report if frames is None else out + [self.face_snapshot(f) for f in face]

    def frame(self, f):
        t = f / self.fps
        for n in self.ctrl + self.digit_bones + self.run_bones:
            b = self.pb[n]; b.location = (0, 0, 0); b.rotation_quaternion = (1, 0, 0, 0); b.rotation_euler = (0, 0, 0)
            b.scale = (1, 1, 1)
        for s in "LR":
            self.pb[f"thigh_parent.{s}"]["IK_FK"] = 0.0
            self.pb[f"upper_arm_parent.{s}"]["IK_FK"] = 1.0
        self.update()
        x, y, yaw = self.plan.root_at(t)
        self.yawm = rz(yaw); base = Matrix.Translation((x, y, 0)) @ self.yawm
        self.place("root", base @ self.rest["root"])
        a, p, g, seg = self.plan.gait_at(t)
        g = g or self._posture_gait(t)
        k = self._cycle_jitter(t, seg)
        self.jz, self.jlean, self.jarm = self.jump_pose(t)
        self.body_pose(t, a, p, g, k, base)
        run = a if g.get("run") else 0.0                  # how much of the reference rig's run the body takes
        if self.run_bones:
            self.head_follow = 1.0 - run                  # the reference rig runs with HEAD-FOLLOW 0
            self.pb["PROPERTIES"]["HEAD-FOLLOW"] = self.head_follow
        if run: self.reference_pose(RUN["body"], self.run_frame(p, seg.stance), run)
        self.lead(t)
        self.reach_trunk(f)
        feet = [self.foot(s, t) for s in "LR"]
        for ft in feet: next(ft)
        for ft in feet: next(ft, None)
        self.arms(t, a, p, g, k)
        self.poles()
        self.hands(t, a, p, g)
        if run: self.reference_pose(RUN["arm"], self.run_frame(p, seg.stance), run)
        self.face_frame(t)
        self.look(t)
        self.posture(f)                                   # before the reach and gesture, which go from the arm it leaves
        self.poles()
        self.reach(f, t)
        self.gesture(f, t)
        self.apply_keys(f)
        self.apply_props(f)
        self.clamp(f)
        self.limit_joints()        # last: the human ranges, past the control ranges clamp() keeps programs to

    def _styled_at(self, t):
        """A non-walking beat that names a mood or carries style overrides sets the attitude for
        its span (eased over 0.4 s at each end); the walk's gait applies otherwise."""
        from .program import gait_params, STYLE
        f = t * self.fps
        for b in self.p["beats"]:
            if b["verb"] in WALKS or not (b["args"].get("mood") or b["id"] in self.p["prims"]): continue
            w = env(f, b["start"] - 0.4 * self.fps, b["start"], b["end"], b["end"] + 0.4 * self.fps)
            if not w: continue
            base = self._walk_posture(t)
            st = gait_params(b, self.p["prims"], self.p.get("style"))
            return {k: (lerp(base[k], st[k], w) if k in STYLE else base[k]) for k in base}
        return None

    def _walk_posture(self, t):
        from .program import GAIT
        walks = [s for s in self.plan.segs if s.kind == "walk"]
        if not walks: return base_style(self.p.get("style"))
        prev = [s for s in walks if s.t0 <= t]
        return (prev[-1] if prev else walks[0]).gait

    def _posture_gait(self, t):
        """Attitude outside a walk: the nearest walk's, so the body is already in the mood
        before it sets off; between two walks it blends from one to the next over 0.5 s."""
        from .program import GAIT
        walks = [s for s in self.plan.segs if s.kind == "walk"]
        styled = self._styled_at(t)
        if styled is not None: return styled
        if not walks: return base_style(self.p.get("style"))
        prev = [s for s in walks if s.t0 <= t]
        nxt = [s for s in walks if s.t0 > t]
        if not prev: return nxt[0].gait
        if not nxt: return prev[-1].gait
        u = ease((t - (nxt[0].t0 - 0.5)) / 0.5)
        a, b = prev[-1].gait, nxt[0].gait
        return {k: lerp(a[k], b[k], u) for k in a}

    def _cycle_jitter(self, t, seg):
        """Per-stride variation so cycles do not repeat exactly (Bodenheimer et al. 1999)."""
        if not seg or not self.var: return {"bob": 1, "swing": 1, "lag": 0}
        n = int((t - seg.t0) / (2 * seg.step_t))
        r = random.Random(hash((self.p["clip"]["seed"], n)) & 0xffffffff)
        v = self.var
        return {"bob": 1 + 0.25 * v * r.uniform(-1, 1), "swing": 1 + 0.25 * v * r.uniform(-1, 1),
                "lag": 0.03 * v * r.uniform(-1, 1)}

    # -- body -------------------------------------------------------------------------------
    def body_pose(self, t, a, p, g, k, base):
        w = lambda lag=0.0, kk=1: math.sin(2 * math.pi * kk * (p - lag))
        idle = 1 - a
        hb = BOB_GAIN * 0.01 * g["hip_bob_cm"]
        bob = a * (-hb * 0.6 - hb * k["bob"] * math.cos(4 * math.pi * (p - BOB_LOW)))
        sway = a * SWAY_K * 0.01 * g["sway_cm"] * w(0.05)
        # standing: a slow weight shift and breathing, never perfectly still
        sway += idle * 0.012 * math.sin(2 * math.pi * t / 7.7) * (0.5 + self.var)
        sway += idle * self._support_shift(t)
        breathe = math.sin(2 * math.pi * t / 4.2)
        sink = a * WALK_SINK + idle * STAND_SINK
        back = JUMP_BACK_K * max(0.0, -self.jz)          # a jump's crouch sits the hips back (+Y is behind)
        self.place("torso", base @ Matrix.Translation((sway, back, bob - sink - idle * 0.002 * (1 + breathe) + self.jz)) @ self.rest["torso"])
        chest_out = idle * self.hero(g) * STAND_CHEST * max(0.0, min(1.0, g["lean_deg"] / 5))   # not on a body already leaning back
        self.turn("torso", "X", UPRIGHT_DEG - chest_out + g["lean_deg"] * (0.8 + 0.2 * a) + self.jlean)   # the mesh stands swaybacked; correct it standing too
        self.turn("hips", "Z", a * HIP_TWIST_K * g["hip_twist_deg"] * w(0.25))
        self.turn("hips", "Y", -a * HIP_DROP_K * 4.0 * w(0.0) + idle * 1.5 * math.sin(2 * math.pi * t / 7.7))
        self.turn("chest", "Z", -a * CHEST_TWIST_K * g["chest_twist_deg"] * w(0.27))
        self.turn("chest", "Y", a * SHOULDER_TILT_K * 2.5 * w(0.02))
        self.turn("hips", "X", -0.25 * g["spine_curve_deg"])                       # pelvis tucks under a rounded back
        self.turn("chest", "X", 6.0 + 0.75 * g["spine_curve_deg"] + (idle + 0.4 * a) * 1.2 * g["breath_depth"] * breathe)
        look = 6.0 * math.sin(2 * math.pi * t / 5.3) + 2.5 * math.sin(2 * math.pi * t / 2.1 + 1.0)
        lag = g["neck_overlap_f"] / self.fps / (2 * 60 / g["cadence_spm"]) if a else 0
        self.turn("neck", "Z", a * 5.0 * w(0.33 + lag) + 0.4 * look * (0.5 + self.var))
        self.turn("neck", "X", a * 2.5 * math.cos(4 * math.pi * (p - 0.18 - lag)))
        self.turn("neck", "Y", a * 2.0 * w(0.10))
        self.turn("neck", "X", 0.35 * g["spine_curve_deg"] + 0.4 * g["head_drop_deg"] + idle * self.hero(g) * STAND_HEAD)
        self.turn("head", "X", -UPRIGHT_DEG - g["lean_deg"] * 1.6 * (0.8 + 0.2 * a) - 0.9 * g["spine_curve_deg"]
                  + 0.6 * g["head_drop_deg"] + a * 2.5 * math.cos(4 * math.pi * (p - 0.24 - lag)))
        self.turn("head", "Z", a * 3.0 * w(0.40) + 0.6 * look * (0.5 + self.var))
        self.turn("head", "Y", a * 2.5 * w(0.15))

    def lead(self, t):
        """The head turns before the body: it points where the body will face LEAD_S from now, up
        to LEAD_MAX degrees ahead, and the chest follows part of the way. People start a turn with
        the head 0.2-0.6 s before the trunk and lead the path by up to ~25° (Grasso & Berthoz
        1998; Imai et al. 2001), so a turn starts at the head and ends at the hips and feet."""
        d = math.degrees(self.plan.root_at(t + LEAD_S)[2] - self.plan.root_at(t)[2])
        if abs(d) < 0.05: return
        d = LEAD_MAX * math.tanh(d / LEAD_MAX)
        # closed loop, as look() is: the head does not follow the chest on this rig (head_follow
        # 0), so the head's own turn is measured and topped up until it leads by d
        y0 = self._gaze_yaw()
        self.turn("chest", "Z", 0.2 * d)
        for _ in range(2):
            r = d - (self._gaze_yaw() - y0)
            self.turn("neck", "Z", 0.4 * r); self.turn("head", "Z", 0.6 * r)

    def _gaze_yaw(self):
        """Where the face looks, in degrees from the body's facing."""
        self.fresh("DEF-spine.006")
        g = self.yawm.to_3x3().inverted() @ (self.rig.matrix_world.to_3x3() @ self.pb["DEF-spine.006"].matrix.to_3x3() @ self.gaze)
        return math.degrees(math.atan2(g.x, -g.y))

    def reach_trunk(self, f):
        """Beyond arm's length the trunk joins a reach: it bends toward the target and turns the
        reaching shoulder forward, by what the arm alone falls short. It starts with the arm,
        finishes TRUNK_LAG frames after the hand arrives and returns last (Kaminski, Bock &
        Gentile 1995). Within reach it stays out of it."""
        for b in self.p["beats"]:
            if b["verb"] != "reach": continue
            s = b["args"]["side"]; hold = b["args"].get("hold", 12); ret = 12
            tau, D = f - b["start"], b["dur"]
            if tau < 0 or tau > D + hold + ret + TRUNK_LAG: continue
            if b["id"] not in self.reach_bend:              # from where the body is when the reach starts
                self.reach_bend[b["id"]] = self._trunk_for(b["id"], s, Vector(b["args"]["target"]))
            bend, twist, back = self.reach_bend[b["id"]]
            if not (bend or twist): continue
            w = ease(tau / (D + TRUNK_LAG)) * (1 - ease((tau - D - hold) / (ret + TRUNK_LAG)))
            # the hips go back (and a little down, knees giving) as the trunk goes forward, so
            # the centre of mass stays over the feet
            self.place("torso", Matrix.Translation(self.yawm.to_3x3() @ Vector((0, back * w, -0.3 * back * w))) @ self.pose_of("torso"))
            self.turn("torso", "X", HIP_SHARE * bend * w)
            self.turn("chest", "X", (1 - HIP_SHARE) * bend * w)
            self.turn("chest", "Z", twist * w)
            # the eyes go to what the hand reaches for: the head does not follow the chest on this
            # rig, so left alone it stays level while the trunk bends (the neck tipped back 116 deg)
            self.look_toward(Vector(b["args"]["target"]), w)

    def _trunk_for(self, bid, s, tgt):
        """The least forward bend (with the reaching shoulder turned forward) that brings the
        target within REACH_FRAC of the arm's length, found by moving the shoulder through the
        same turns reach_trunk makes; capped at TRUNK_MAX. Returns (bend, twist, hips back)."""
        M = self.rig.matrix_world
        self.fresh(f"DEF-upper_arm.{s}")
        S0 = M @ self.pb[f"DEF-upper_arm.{s}"].head
        Pt, Pc = M @ self.pose_of("torso").to_translation(), M @ self.pose_of("chest").to_translation()
        reach = REACH_FRAC * self.arm_len
        short = (tgt - S0).length - reach
        if short <= 0: return 0.0, 0.0, 0.0
        Y3 = self.yawm.to_3x3()
        X, Z, back_dir = Y3 @ Vector((1, 0, 0)), Vector((0, 0, 1)), Y3 @ Vector((0, 1, -0.3))
        u = Y3.inverted() @ (tgt - S0)
        bearing = math.degrees(math.atan2(u.x, -u.y))
        twist = max(-25.0, min(25.0, (0.35 * bearing + (12.0 if s == "R" else -12.0)) * min(1.0, short / 0.15)))
        rot = lambda p, c, deg, ax: c + Matrix.Rotation(math.radians(deg), 3, ax) @ (p - c)
        for bend in range(0, TRUNK_MAX + 1):
            S, C = rot(S0, Pt, HIP_SHARE * bend, X), rot(Pc, Pt, HIP_SHARE * bend, X)
            S = rot(rot(S, C, (1 - HIP_SHARE) * bend, X), C, twist, Z)
            back = BACK_K * max(0.0, (Y3 @ Vector((0, -1, 0))).dot(S - S0))      # of how far the shoulder went forward
            S = S + back * back_dir
            if (tgt - S).length <= reach: break
        self.report.setdefault("reach_trunk", {})[bid] = {"short_cm": round(100 * short, 1), "bend_deg": bend,
                                                         "twist_deg": round(twist, 1), "hips_back_cm": round(100 * back, 1),
                                                         "still_short_cm": round(100 * max(0.0, (tgt - S).length - reach), 1)}
        return float(bend), twist, back

    def _support_shift(self, t):
        """Sideways hip shift toward the standing foot around each step taken while slow:
        it starts a quarter second before the lift and returns after the landing."""
        x = 0.0
        for s, side in (("L", 1.0), ("R", -1.0)):
            o = "R" if s == "L" else "L"
            for p0, p1 in zip(self.plan.plants[s], self.plan.plants[s][1:]):
                w = env(t, p0.lift - 0.25, p0.lift, p1.land, p1.land + 0.2)
                if w: x += -side * self._shift_to(o, p0.lift) * w   # toward the other foot
        return x

    def _shift_to(self, s, t):
        """How far the hips move sideways to stand on foot s: enough to bring the centre of mass
        within SHIFT_MARGIN of that sole's centre (it follows the hips about 71%, measured), never
        less than the 10 cm every rig used before and never more than 16 cm."""
        x, y, yaw, kind, d = self.plan.foot_at(s, t)
        wide = d[2].wide if kind == "stance" else 0.0
        lateral = min(abs(self.ankle[s].x), 0.095) + wide + self.sole_out[s]
        return max(0.10, min(0.16, (lateral - SHIFT_MARGIN) / 0.71))

    def foot(self, s, t):
        pl = self.plan
        x, y, yaw, kind, d = pl.foot_at(s, t)
        if kind == "stance":
            since, to_lift, plant = d
            ankle, ang, toe = self._stance(s, plant, since, to_lift)
            yaw = plant.yaw
            toes_on, toe_w = (plant if ang < 0 else None), 1.0
        else:
            u, p0, p1 = d
            a1, ang1, toe1 = self._stance(s, p1, 0.0, p1.lift - p1.land)
            # A foot that peeled off keeps rolling over its ball for the first part of the swing,
            # toes still on the floor, before it lifts; then it travels and lands heel first with
            # no horizontal speed left (the lift reaches zero exactly at landing).
            g0 = self._plant_gait(p0)
            U0 = TOE_ROLL if p0.peel_out and not g0.get("run") and not p1.jump_in else 0.0
            toes_on, toe_w = None, 1.0
            T = p1.land - p0.lift
            if u < U0:
                ang = -min(g0["toe_off_deg"] + TOE_ROLL_DEG * ease(u / U0), self._peel_max(p0, t))
                ankle = self._planted(s, p0, ang)
                toe, yaw, toes_on = -ang, p0.yaw, p0
            else:
                if U0:
                    ang0 = -min(g0["toe_off_deg"] + TOE_ROLL_DEG, self._peel_max(p0, p0.lift + U0 * T))
                    a0, toe0 = self._planted(s, p0, ang0), -ang0
                else:
                    a0, ang0, toe0 = self._stance(s, p0, p0.lift - p0.land, 0.0)
                v = (u - U0) / (1 - U0)
                # up before across, across, then down: travel waits for the first and last 5%
                e = ease((v - 0.05) / 0.9)
                run = g0.get("run") and self._plant_gait(p1).get("run")
                ankle = a0.lerp(a1, e)
                ankle.z += LIFT_GAIN * max(p1.lift_h, 0.035) * math.sin(math.pi * v ** LIFT_EARLY) ** 0.8
                if run:
                    # the reference rig's path, less the straight line from its lift to its strike, on
                    # top of ours from plant to plant: its shape with our ends
                    q0, c0 = self.run_swing(s, 0.0); q1, c1 = self.run_swing(s, 1.0); qv, cv = self.run_swing(s, v)
                    y = lerp(p0.yaw, p1.yaw, v)
                    ankle = a0.lerp(a1, v) + rz(y).to_3x3() @ (qv - q0.lerp(q1, v))
                    # height: the reference rig's own, with ours at each end fading out fast (a toe-off
                    # lower than the reference rig's no longer lowers the whole heel kick)
                    ankle.z = qv.z + (a0.z - q0.z) * (1 - v) ** 3 + (a1.z - q1.z) * v ** 3
                    ang = cv - lerp(c0, c1, v) + lerp(ang0, ang1, v)
                elif p1.jump_in:
                    # in the air: the feet go where the hips go, the knees tucking a little, from
                    # the push's pose to the landing's
                    ankle = a0.lerp(a1, v)
                    ankle.z += self.jz - JUMP_TOES_M + JUMP_TUCK * math.sin(math.pi * v)
                    ang = lerp(ang0, ang1, ease(min(1.0, v / 0.6)))   # set for the landing before it comes
                elif p0.peel_out:                              # heel up into the air, toes trailing down, then level for the landing
                    up, dip = ang0 - AIR_PITCH_DEG, -15.0
                    if v < 0.2: ang = lerp(ang0, up, ease(v / 0.2))
                    elif v < 0.5: ang = lerp(up, dip, ease((v - 0.2) / 0.3))
                    else: ang = lerp(dip, ang1, ease((v - 0.5) / 0.5))
                elif v < 0.35:
                    ang = ang0
                else:
                    ang = lerp(ang0, ang1, ease((v - 0.35) / 0.65))
                toe = toe0 * (1 - ease(v / 0.45))            # the push-off bend straightens in the air
                if p1.jump_in: toe = lerp(toe0, toe1, ease(min(1.0, v / 0.6)))   # bent for a landing on them
                if p0.peel_out and v < 0.3 and not p1.jump_in:
                    toes_on, toe_w = p0, 1 - ease(v / 0.3)   # toes leave the floor smoothly
                yaw = lerp(p0.yaw, p1.yaw, e)
        A = self.ankle[s]
        out = math.radians(TOE_OUT_DEG) * (1 if s == "L" else -1)
        if kind != "stance" and toes_on is None: ang = self._swing_ankle(s, ankle, yaw, out, ang)
        M = Matrix.Translation(ankle) @ rz(yaw + out) @ pitch(ang) @ Matrix.Translation(-A) @ self.rest[f"foot_ik.{s}"]
        self.place(f"foot_ik.{s}", M)
        yield                                             # the other foot is placed before the toes read their pose
        self.fresh(f"toe_ik.{s}")
        b = self.pb[f"toe_ik.{s}"]
        follow = b.matrix.copy()
        if toe:
            # toes keep some push-off bend relative to the foot (about their own joint) while in the air
            h = b.head.copy(); side_ax = rz(yaw).to_3x3() @ Vector((1, 0, 0))
            follow = Matrix.Translation(h) @ Matrix.Rotation(math.radians(-toe), 4, side_ax) @ Matrix.Translation(-h) @ follow
        if toes_on is not None:
            # toes on the floor: held where they lie when the foot is flat on that plant; as the
            # foot lifts away they blend into following it, still bent, so the tips never dig in
            flat = self._planted(s, toes_on, 0.0)
            flat.z += self._crease(toe)                   # the foot is lifted off its pinched crease; the toes rise with it
            M_flat = Matrix.Translation(flat) @ rz(toes_on.yaw + out) @ Matrix.Translation(-A) @ self.rest[f"toe_ik.{s}"]
            self.place(f"toe_ik.{s}", _blend(follow, M_flat, toe_w))
        elif toe:
            self.place(f"toe_ik.{s}", follow)

    def posture(self, f):
        """A posture beat (program.POSTURES): the body goes down through its poses, each in
        POSTURE_STEP s, holds the last and comes back up. A pose beat: the body goes through the
        program's own key poses at their frames, holds the last POSE_HOLD s and comes back
        (program.pose_back). Each control the poses set is moved from what the body was doing to the pose;
        one a pose leaves out keeps the body's own; the root's pose is applied on top of where the
        body stands."""
        from .program import POSTURES, POSTURE_STEP, pose_back, pose_rig
        for b in self.p["beats"]:
            if (b["verb"] not in POSTURES and b["verb"] != "pose") or not b["start"] <= f <= b["end"]: continue
            tau, dur = (f - b["start"]) / self.fps, (b["end"] - b["start"]) / self.fps
            if b["verb"] == "pose":
                keys = b["args"]["keys"]
                # the program's key poses as the rig has them (an upper arm's X is written from hanging);
                # the posture library is written on the rig's values already
                sets = [{c: {n: pose_rig(c, n, v) for n, v in ch.items()} for c, ch in k["set"].items()} for k in keys]
                path = [None] + sets + [sets[-1], None]
                times = [0.0] + [k["at"] / self.fps for k in keys] + [dur - pose_back(b["args"], self.fps) / self.fps, dur]
                k = max(i for i in range(len(times) - 1) if times[i] <= tau)
                span = times[k + 1] - times[k]
                w = ease(min(1.0, max(0.0, (tau - times[k]) / span))) if span > 0 else 1.0
                pose_t = (times, tau, dur)                  # each control then keeps its own clock (succession)
            else:
                pose_t = None
                path = [None] + [POSTURE_LIB[n] for n in POSTURES[b["verb"]]]
                n = len(path) - 1
                u = min(tau, dur - tau, n * POSTURE_STEP) / POSTURE_STEP      # 0 standing .. n the last pose
                k = min(int(u), n - 1); w = ease(u - k)
            ctrls = sorted({c for p in path[1:] if p for c in p}, key=lambda c: (c != "torso", c != "root"))   # torso, root, then the rest
            tip = None                                    # the root's two poses and its pose now, while it tips
            for c in ctrls:
                pb = self.pb[c]
                cur_q = pb.rotation_quaternion.copy() if pb.rotation_mode == "QUATERNION" else pb.rotation_euler.to_quaternion()
                cur_l = Vector(pb.location)
                def val(p):
                    v = p.get(c) if p else None
                    q = Euler([math.radians(x) for x in v["rot"]], "XYZ").to_quaternion() if v and "rot" in v else None
                    l = Vector(v["loc"]) if v and "loc" in v else None
                    if c == "root":                        # on top of where the body stands
                        return (cur_l + cur_q @ (l or Vector()), cur_q @ (q or Quaternion()))
                    return (l if l is not None else cur_l, q if q is not None else cur_q)
                if pose_t:
                    kc, u, span_c = self._pose_clock(c, *pose_t)
                    (la, qa), (lb, qb) = val(path[kc]), val(path[kc + 1])
                    if qa.dot(qb) < 0: qb.negate()
                    d = qa.inverted() @ qb
                    wc = pose_curve(u, math.degrees(d.angle), c, span_c)
                    loc, rot = la.lerp(lb, wc), qa @ Quaternion(d.axis, d.angle * wc) if d.angle > 1e-6 else qa.copy()
                    if c.split(".")[0] in HINGES:
                        # a hinge with a twist: blend its angles, so it stays a bend plus a twist all the
                        # way (a slerp between two such turns swings the forearm sideways on the way)
                        ea, eb = qa.to_euler("XYZ"), qb.to_euler("XYZ", qa.to_euler("XYZ"))
                        rot = Euler([x + (y - x) * wc for x, y in zip(ea, eb)], "XYZ").to_quaternion()
                else:
                    (la, qa), (lb, qb) = val(path[k]), val(path[k + 1])
                    if qa.dot(qb) < 0: qb.negate()
                    loc, rot = la.lerp(lb, w), qa.slerp(qb, w)
                if c == "root" and (qa.rotation_difference(qb).angle > 1e-3):
                    # tipping over (sitting back to lying): about the hips, which slide straight
                    # along the floor from one pose's place to the next instead of swinging up
                    tor = lambda p: Vector(p["torso"]["loc"]) if p and "torso" in p and "loc" in p["torso"] else Vector()
                    ha, hb = self.hips_local + tor(path[k]), self.hips_local + tor(path[k + 1])
                    h = ha.lerp(hb, w)
                    P = (la + qa @ ha).lerp(lb + qb @ hb, w)
                    loc = P - rot @ h
                    tip = (la, qa, lb, qb, loc, rot)
                if tip and c.startswith("foot_ik"):
                    # the feet go their own way across the floor, not swung round with the root
                    rla, rqa, rlb, rqb, rl, rq = tip
                    r0 = self.foot_rest[c]
                    Fa, Fb = rla + rqa @ (r0 + la), rlb + rqb @ (r0 + lb)
                    Wa, Wb = rqa @ qa, rqb @ qb
                    if Wa.dot(Wb) < 0: Wb.negate()
                    loc = rq.inverted() @ (Fa.lerp(Fb, w) - rl) - r0
                    rot = rq.inverted() @ Wa.slerp(Wb, w)
                if c.startswith("foot_ik") and (la - lb).length > 0.05 and "root" not in (path[k] or {}) and "root" not in (path[k + 1] or {}):
                    loc = loc + Vector((0, 0, 0.06 * math.sin(math.pi * w)))   # a foot moving across lifts, not drags
                pb.location = loc
                if pb.rotation_mode == "QUATERNION": pb.rotation_quaternion = rot
                else: pb.rotation_euler = rot.to_euler(pb.rotation_mode)
            self.update()
            return

    def _pose_clock(self, c, times, tau, dur):
        """A pose beat's segment and progress for one control: its clock offset by POSE_LAG (the head
        leads, the arm follows from the shoulder to the hand)."""
        lag = POSE_LAG.get(c.split(".")[0], 0) / self.fps
        tc = min(max(tau - lag, 0.0), dur)
        k = max(i for i in range(len(times) - 1) if times[i] <= tc)
        span = times[k + 1] - times[k]
        return k, (min(1.0, max(0.0, (tc - times[k]) / span)) if span > 0 else 1.0), span

    def jump_pose(self, t):
        """(hips' height offset m, trunk's forward tip deg, the arms' forward swing deg) at t."""
        from .program import G, JUMP_TOES
        j = self.plan.jump_at(t)
        if not j: return 0.0, 0.0, 0.0
        s, tau = j
        h, c, pu, fl, la, re = s.jump
        dip, absorb = JUMP_DIP + JUMP_DIP_K * h, JUMP_DIP + JUMP_ABSORB_K * h - 0.04
        vb = math.sqrt(2 * G * max(0.02, h - JUMP_TOES))
        # the arms swing back over the crouch's first 60%, then drive forward and up from there
        # through the push (about 0.28 s: in the push alone they whipped 115 deg in 3 frames)
        back = 0.6 * c
        arm = (-JUMP_ARM_BACK * ease(tau / back) if tau < back else
               lerp(-JUMP_ARM_BACK, JUMP_ARM_UP, ease(min(1.0, (tau - back) / (c + pu - back)))))
        if tau < c:
            u = ease(tau / c); return -dip * u, JUMP_LEAN_CROUCH * u, arm
        tau -= c
        if tau < pu:
            u = ease(tau / pu); return lerp(-dip, JUMP_TOES, u), lerp(JUMP_LEAN_CROUCH, 4.0, u), arm
        tau -= pu
        if tau < fl:
            return JUMP_TOES + vb * tau - 0.5 * G * tau * tau, 4.0, lerp(JUMP_ARM_UP, 20.0, tau / fl)
        tau -= fl
        if tau < la:
            u = math.sin(0.5 * math.pi * tau / la)     # the landing's speed is taken out at once, then eased
            return lerp(JUMP_TOES, -absorb, u), lerp(4.0, JUMP_LEAN_LAND, u), lerp(20.0, 30.0, u)
        u = ease((tau - la) / re)
        return lerp(-absorb, 0.0, u), lerp(JUMP_LEAN_LAND, 0.0, u), lerp(30.0, 0.0, u)

    def _stance(self, s, plant, since, to_lift):
        """World ankle, foot pitch and toe bend for a foot on `plant` (rolling contact). Running,
        the heel comes down over the first quarter of the stance and peels up over its last half,
        as the reference rig's foot does."""
        c = self._cycle(plant)
        ang = 0.0
        if plant.jump_out or plant.jump_in:               # a jump's: heels up in the push, toes first at the landing
            from .program import JUMP_PUSH
            if plant.jump_out and to_lift < JUMP_PUSH: ang = -JUMP_PEEL_DEG * ease(1 - to_lift / JUMP_PUSH)
            if plant.jump_in and since < 0.08: ang = -JUMP_LAND_DEG * (1 - ease(since / 0.08))
            if ang: return self._planted(s, plant, ang), ang, -ang
            if plant.jump_out: return self._planted(s, plant, 0.0), 0.0, 0.0   # flat through the crouch, until the push
        g = self._plant_gait(plant)
        if g.get("run"):
            down = self._seg_of(plant).stance * c
            heel, toe, roll, peel = RUN_HEEL_DEG, RUN_TOE_OFF_DEG, 0.25 * down, 0.5 * down
        else:
            heel, toe, roll, peel = g["heel_up_deg"], g["toe_off_deg"], 0.12 * c, PEEL * c
        if plant.roll_in and since < roll:
            ang = heel * (1 - ease(since / roll))
        if plant.peel_out and to_lift < peel:
            ang = -min(toe * ease(1 - to_lift / peel), self._peel_max(plant, plant.lift - to_lift))
        return self._planted(s, plant, ang), ang, (-ang if ang < 0 else 0.0)

    def _swing_ankle(self, s, ankle, yaw, out, ang):
        """The foot's pitch moved into SWING_ANKLE: the ankle's angle (shin to foot, from rest) at
        a pitch, with the knee where a two-bone IK puts it for this hip and ankle. The foot control
        pivots mid-foot, so a pitch moves the ankle too (and the knee with it): a few Newton steps."""
        B = R.Rest(self.rig)
        H0, K0, A0, T0 = tuple(B[n].head_local for n in (f"DEF-LEG-01.{s}", f"DEF-SHIN-01.{s}", f"DEF-FOOT.{s}")) + (B[f"DEF-FOOT.{s}"].tail_local,)
        l1, l2 = (K0 - H0).length, (A0 - K0).length
        rest = math.degrees((A0 - K0).angle(T0 - A0))
        self.fresh(f"DEF-LEG-01.{s}", own=False)
        H = self.pb[f"DEF-LEG-01.{s}"].head
        ahead = rz(yaw).to_3x3() @ Vector((0, -1, 0))
        def flex(g):
            Rm = (rz(yaw + out) @ pitch(g)).to_3x3()
            Aw = ankle + Rm @ (A0 - self.ankle[s]); foot = Rm @ (T0 - A0)
            d = Aw - H; n = d.length
            if n >= l1 + l2 - 1e-4: K = H + d * (l1 / n)
            else:
                a = (l1 * l1 - l2 * l2 + n * n) / (2 * n); h = math.sqrt(max(l1 * l1 - a * a, 0.0))
                perp = ahead - d * (ahead.dot(d) / (n * n))
                K = H + d * (a / n) + (perp.normalized() * h if perp.length > 1e-6 else Vector())
            return math.degrees((Aw - K).angle(foot)) - rest
        lo, hi = SWING_ANKLE
        for _ in range(4):
            f0 = flex(ang)
            if lo <= f0 <= hi: break
            k = flex(ang + 1.0) - f0
            if abs(k) < 1e-3: break
            ang += ((hi if f0 > hi else lo) - f0) / k
        return ang

    def _peel_max(self, plant, t):
        """How far (deg) the heel of a foot on `plant` may be up at t: the leg's lean behind the
        hips, and the ankle pointed PEEL_POINT_DEG."""
        x, y, _ = self.plan.root_at(t)
        fx, fy = fwd(plant.yaw)
        ahead = (x - plant.x) * fx + (y - plant.y) * fy - ANKLE_LEAD
        return math.degrees(math.atan2(max(ahead, 0.0), self.hips_local.z)) + PEEL_POINT_DEG

    def _planted(self, s, plant, ang, crease=True):
        local = self.ankle[s] + Vector((self.step_x[s] + (1 if s == "L" else -1) * plant.wide, self.under_hip, 0)) + self._roll(ang) + self.sole
        if crease and ang < 0: local.z += self._crease(-ang)
        return Vector((plant.x, plant.y, 0)) + rz(plant.yaw).to_3x3() @ local

    def _seg_of(self, plant):
        for sg in self.plan.segs:
            if sg.kind == "walk" and sg.t0 - 1e-6 <= plant.land <= sg.t1 + 1e-6: return sg
        return None

    # -- the reference rig's run ------------------------------------------------------------------
    @staticmethod
    def run_frame(p, S, warp=None):
        """The reference rig's frame (0..24) at stride phase p (0 = left heel strike) when a foot is down
        S of the stride: its left foot is down 14 to 17.5, its right 2 to 5.5. warp 1 puts its
        strikes and lifts on ours; 0 reads it at its own pace from the strike on."""
        warp = RUN_WARP if warp is None else warp
        lo, hi = RUN["strike_L"], RUN["lift_L"]
        down, up = hi - lo, 12.0 - (hi - lo)
        q = p % 0.5
        f = lo + (down * q / S if q < S else down + up * (q - S) / (0.5 - S))
        return (lo + 24.0 * p + warp * (f - lo - 24.0 * q)) % 24.0

    @staticmethod
    def _run_at(f):
        """The reference rig's cycle at frame f (0..24, between frames linearly; rotations slerped)."""
        i = int(math.floor(f)) % 24; w = f - math.floor(f)
        return RUN["cycle"][i], RUN["cycle"][i + 1], w

    def reference_pose(self, bones, f, r):
        """The reference rig's run's local transforms on these bones at its frame f, blended in by r over
        what the walk and the stand made of them."""
        a, b, w = self._run_at(f)
        for n in bones:
            if n not in self.pb: continue
            pb = self.pb[n]
            (la, qa, sa), (lb, qb, sb) = a["bones"][n], b["bones"][n]
            qa, qb = Quaternion(qa), Quaternion(qb)
            if qa.dot(qb) < 0: qb.negate()                # the short way: the reference rig keys some as q, some as -q
            loc = Vector(la).lerp(Vector(lb), w); rot = qa.slerp(qb, w); scl = Vector(sa).lerp(Vector(sb), w)
            cur = pb.rotation_quaternion.copy() if pb.rotation_mode == "QUATERNION" else pb.rotation_euler.to_quaternion()
            if cur.dot(rot) < 0: rot.negate()
            pb.location = Vector(pb.location).lerp(loc, r)
            q = cur.slerp(rot, r)
            if pb.rotation_mode == "QUATERNION": pb.rotation_quaternion = q
            else: pb.rotation_euler = q.to_euler(pb.rotation_mode)
            pb.scale = Vector(pb.scale).lerp(scl, r)
        self.update()                                     # set locally: everything below them moved

    def run_swing(self, s, v):
        """The reference rig's swing for foot s at v (0 lift .. 1 strike): its ankle in the floor's frame
        (facing -Y) and its pitch (+ toes up)."""
        lo, hi = RUN["lift_L"], RUN["strike_L"] + 24.0
        f = (lo + v * (hi - lo)) % 24.0                 # the swing is the foots own: either side reads the lefts
        a, b, w = self._run_at(f)
        ank = Vector(a["ankle_L"]).lerp(Vector(b["ankle_L"]), w)
        if s == "R": ank.x = -ank.x
        return ank, lerp(a["pitch_L"], b["pitch_L"], w)

    def _cycle(self, plant):
        for sg in self.plan.segs:
            if sg.kind == "walk" and sg.t0 - 1e-6 <= plant.land <= sg.t1 + 1e-6: return 2 * sg.step_t
        return 1.25

    def _plant_gait(self, plant):
        for sg in self.plan.segs:
            if sg.kind == "walk" and sg.t0 - 1e-6 <= plant.land <= sg.t1 + 1e-6: return sg.gait
        from .program import GAIT
        return base_style(self.p.get("style"))

    @staticmethod
    def hero(g):
        """How much of the reference rig's hero stand a style takes: none slumped, less leaning forward."""
        slump = max(0.0, 1 - g["spine_curve_deg"] / 20 - max(0.0, g["head_drop_deg"]) / 36)
        return slump * max(0.0, min(1.0, 1 - max(0.0, g["lean_deg"] - 5) / 10))

    def standing(self, t, a, g):
        """The weight of the hero stand at time t: standing, not walking, and not turning on the
        spot (the arms come in for a turn, and a flared arm swings its hand fast round)."""
        turning = max([env(t, sg.t0 - 0.25, sg.t0, sg.t1, sg.t1 + 0.25) for sg in self.plan.segs if sg.kind == "turn"] + [0.0])
        return (1 - a) * (1 - turning) * self.hero(g)

    def arms(self, t, a, p, g, k):
        # Each link of the chain is turned on both sides before the next link reads its pose, so
        # one depsgraph update serves both arms (the two chains do not depend on each other).
        sides = (("L", 1, -0.25), ("R", -1, 0.25))
        for s, sign, off in sides:
            self.turn(f"shoulder.{s}", "Z", -g["shoulder_fwd_deg"] * sign)
            self.turn(f"shoulder.{s}", "Y", (3 + g["shoulder_drop_deg"]) * sign)
        for s, sign, off in sides:
            q = math.sin(2 * math.pi * (p - 0.08 + off - k["lag"]))
            sw = q * a * g["arm_swing"] * k["swing"]
            stand = self.standing(t, a, g)
            out = a * min(1.0, g["arm_swing"]) * (ARM_PASS_DEG * (1 - q * q) - ARM_IN_DEG) + stand * STAND_ARM_OUT
            self.turn(f"upper_arm_fk.{s}", "Y", (ARM_DOWN_DEG - self.arm_rest_down[s] - out) * sign)
            self.turn(f"upper_arm_fk.{s}", "Z", -6 * sign)
            # standing, the arms are never still: they breathe with the chest and drift with
            # the weight shift, each side a little out of step with the other
            idle = 1 - a
            drift = idle * (2.2 * math.sin(2 * math.pi * t / 4.2 + (0.6 if s == "L" else 0.0))
                            + 2.5 * math.sin(2 * math.pi * t / 7.7 + (0.0 if s == "L" else 1.3)))
            # arms hang with gravity: undo the forward pitch of torso and chest so a leaning or
            # slumped body does not carry the hands out in front
            hang = UPRIGHT_DEG + g["lean_deg"] * (0.8 + 0.2 * a) + 6.0 + 0.75 * g["spine_curve_deg"]
            self.turn(f"upper_arm_fk.{s}", "X", -ARM_SWING_DEG * sw + ARM_PITCH_DEG - stand * STAND_ARM_FWD + drift + 0.9 * hang - self.jarm)
            self.turn(f"upper_arm_fk.{s}", "Y", idle * 1.5 * sign * math.sin(2 * math.pi * t / 5.1 + (0 if s == "L" else 2)))
        for s, sign, off in sides:
            u = 0.5 * (1 + math.sin(2 * math.pi * (p - 0.14 + off)))     # 0 at the back of the swing, 1 at the front
            w = a * min(1.0, g["arm_swing"])
            self.turn(f"forearm_fk.{s}", "X", -(g["elbow_bend_deg"] + w * (ELBOW_BACK + ELBOW_SWING * u * g["arm_swing"] / max(1.0, g["arm_swing"]))
                                             + self.standing(t, a, g) * STAND_ELBOW))
        for s, sign, off in sides:
            self.turn(f"hand_fk.{s}", "X", 8 * a * math.sin(2 * math.pi * (p - 0.22 + off)) + a * WRIST_FLEX_DEG
                      + self.standing(t, a, g) * STAND_WRIST)

    def poles(self):
        """Each arm's IK pole, set where its FK elbow points (out from the line from shoulder to
        hand): an arm that goes into IK for a reach or a gesture keeps its elbow where it was
        instead of swinging it to the pole the rest pose left (a pop of 13 cm in a frame once the
        standing elbows bent 51 deg)."""
        for s in "LR": self.fresh(f"forearm_fk.{s}")
        for s in "LR":
            sh, el, hd = (self.pb[f"{n}.{s}"].head for n in ("upper_arm_fk", "forearm_fk", "hand_fk"))
            d = el - (sh + hd) / 2
            if d.length < 1e-3: d = self.yawm.to_3x3() @ Vector((0, 1, 0))     # a straight arm: the elbow points back
            M = self.pb[f"upper_arm_ik_target.{s}"].matrix.copy(); M.translation = el + d.normalized() * 0.35
            self.place(f"upper_arm_ik_target.{s}", M)

    def hands(self, t, a, p, g):
        f = t * self.fps
        for s in "LR":
            off = self.arm_phase[s]
            # the relaxed cascade, then any hand-shape beats blended over it
            curls = []
            for kidx, n in enumerate(FINGERS):
                sway = 4 * a * math.sin(2 * math.pi * (p - 0.30 - 0.04 * kidx + off))
                scale = (g["finger_curl"] + STAND_CURL * self.standing(t, a, g)) / 0.3
                curls.append(([((14 + 5 * kidx) + sway) * scale, ((24 + 6 * kidx) + 1.2 * sway) * scale,
                               ((10 + 3 * kidx) + 0.5 * sway) * scale], [3, 0, 3, 7][kidx]))
            thumb = ([14, 12, 12], 32)
            for b in self.p["beats"]:
                if b["verb"] != "hand" or b["args"].get("side", "both") not in (s, "both"): continue
                wgt = env(f, b["start"], b["start"] + 4, b["end"] - 4, b["end"]) * b["args"].get("amount", 1.0)
                shape = SHAPES[b["args"]["shape"]]
                if not wgt or shape is None: continue
                fc, tc, spread = shape
                for kidx, n in enumerate(FINGERS):
                    full = R.CURL_FULL[n]
                    tgt = [fc[kidx] * full[i] for i in range(3)]
                    curls[kidx] = ([lerp(curls[kidx][0][i], tgt[i], wgt) for i in range(3)],
                                   lerp(curls[kidx][1], spread[kidx], wgt))
                full = R.CURL_FULL["thumb"]
                thumb = ([lerp(thumb[0][i], tc * full[i], wgt) for i in range(3)], lerp(thumb[1], 32 + 30 * tc, wgt))
            for kidx, n in enumerate(FINGERS):
                degs, close = curls[kidx]
                for i in (1, 2, 3):
                    self.curl(f"{n}.0{i}.{s}", degs[i - 1], close if i == 1 else 0.0)
            self.curl(f"thumb.01.{s}", thumb[0][0], thumb[1])
            self.curl(f"thumb.02.{s}", thumb[0][1]); self.curl(f"thumb.03.{s}", thumb[0][2])
        self.dirty.update(self.digit_bones)               # posed directly; read only after an update

    def face_frame(self, t):
        f = t * self.fps
        g = self.plan.gait_at(t)[2] or self._posture_gait(t)
        vals = {k: 0.0 for k in R.FACE_KEYS}
        vals["frown"] = 0.6 * g["brow_knit"]; vals["jaw_open"] = g["jaw_slack"]
        for b in self.p["beats"]:
            if b["verb"] != "express": continue
            a = b["args"]; s0 = b["start"]
            on = a.get("onset", 6); off = a.get("offset", 8)
            apex = a.get("apex", max(0, b["dur"] - on - off))
            w = env(f, s0, s0 + on, s0 + on + apex, s0 + on + apex + off)
            for k, v in a["keys"].items(): vals[k] = max(vals[k], v * w)
        L = g["blink_len"]
        pulse = max([env(t, b, b + 0.06 * L, b + 0.1 * L, b + 0.22 * L) for b in self.blinks] + [0.0])
        lid = g["lid_droop"]
        vals["blink"] = max(vals["blink"], lid + (1 - lid) * pulse)
        self.face_vals = vals

    def look(self, t):
        f = t * self.fps
        for b in self.p["beats"]:
            if b["verb"] != "look_at": continue
            w = env(f, b["start"], b["start"] + 8, b["end"] - 10, b["end"]) * b["args"].get("amount", 1.0)
            if not w: continue
            self.look_toward(Vector(b["args"]["target"]), w)

    def limit_joints(self):
        """The neck and wrists held within their human ranges (see limit_joint)."""
        self.limit_joint("neck", ["head"])
        for s in "LR":
            for _ in range(2):                              # the forearm's twist bones follow the hand
                self.limit_joint(f"wrist.{s}", [f"hand_ik.{s}", f"hand_fk.{s}"])

    def limit_joint(self, joint, ctrls, margin=2.0):
        """A joint held within its human range (anatomy.LIMITS, less a margin), read as the checker
        reads it (parent deform bone to child): what the layers above ask past it, the child does
        not turn; `ctrls` (the controls that turn the child) are turned back by the excess. The
        head does not follow the chest on this rig, so a trunk bent 80 deg into a reach left the
        neck tipped back 80 deg; a hand returning from a gesture bent its wrist 50 deg."""
        if not hasattr(self, "anat"): self.anat = anatomy.Angles(self.rig)
        p, c = self.anat.j[joint]
        self.fresh(c)
        Pm, Cm = self.pb[p].matrix, self.pb[c].matrix
        rest = self.anat.rest[joint]
        q = rest.inverted() @ (Pm.inverted() @ Cm).to_quaternion()
        sx, sz, tw = anatomy.swing_twist(q)
        lim = {i: (lo + margin, hi - margin) for i, lo, hi, _, _ in anatomy.limits(joint)}
        cx, cz, ct = (max(lim[i][0], min(lim[i][1], v)) for i, v in enumerate((sx, sz, tw)))
        if (cx, cz, ct) == (sx, sz, tw): return
        sw = Vector((cx, 0.0, cz))
        q2 = (Quaternion(sw.normalized(), math.radians(sw.length)) if sw.length > 1e-6 else Quaternion()) @ Quaternion((0, 1, 0), math.radians(ct))
        want = Pm.to_quaternion() @ rest @ q2                  # the child's turn as the range allows
        D = (want @ Cm.to_quaternion().inverted()).to_matrix().to_4x4()
        h = Cm.to_translation()
        for n in ctrls:
            M = Matrix.Translation(h) @ D @ Matrix.Translation(-h) @ self.pose_of(n)
            self.pb[n].matrix = M; self._posed(n, M.copy())

    def look_toward(self, target, w):
        """Closed loop: measure where the head looks now (posture, breathing and gait have
        already moved it), turn neck and head by what is left to the target, then once more for
        what that first turn missed. Limits are on the gaze relative to the chest, the neck's own
        base: relative to the body's heading, a body lying on its back or bent into a reach tipped
        the neck back 117-128 deg to reach a target an upright body sees within its range."""
        M = self.rig.matrix_world
        self.fresh("DEF-CHEST-02")
        rest = M.to_3x3() @ R.Rest(self.rig)["DEF-CHEST-02"].matrix_local.to_3x3()
        chest = (M.to_3x3() @ self.pb["DEF-CHEST-02"].matrix.to_3x3() @ rest.inverted()).to_quaternion()   # the chest's turn since rest
        F = chest.to_matrix(); to_body = F.inverted()     # measured and turned in the same frame
        for it in range(3):
            self.fresh("DEF-spine.006")
            head = M @ self.pb["DEF-spine.006"].head
            local = to_body @ (target - head)
            yaw = max(-80.0, min(80.0, math.degrees(math.atan2(local.x, -local.y))))
            pit = max(-40.0, min(40.0, math.degrees(math.atan2(-local.z, math.hypot(local.x, local.y)))))
            g = to_body @ (M.to_3x3() @ self.pb["DEF-spine.006"].matrix.to_3x3() @ self.gaze)
            dyaw = yaw - math.degrees(math.atan2(g.x, -g.y))
            dpit = pit - math.degrees(math.atan2(-g.z, math.hypot(g.x, g.y)))
            k = w if it == 0 else w * w      # a partial look (w < 1) is not refined to the full target
            self.turn("neck", "Z", 0.4 * dyaw * k, frame=F); self.turn("head", "Z", 0.6 * dyaw * k, frame=F)
            self.turn("neck", "X", 0.3 * dpit * k, frame=F); self.turn("head", "X", 0.7 * dpit * k, frame=F)

    def reach(self, f, t):
        for b in self.p["beats"]:
            if b["verb"] != "reach": continue
            s = b["args"]["side"]
            hold = b["args"].get("hold", 12); ret = 12
            t0, D = b["start"], b["dur"]
            tau = f - t0
            if tau < 0 or tau > D + hold + ret: continue
            self.fresh(f"hand_fk.{s}")
            fk = self.pb[f"hand_fk.{s}"].matrix.copy()
            if b["id"] not in self.reach_start: self.reach_start[b["id"]] = fk.copy()
            M0 = self.reach_start[b["id"]]
            p0, tgt = M0.to_translation(), Vector(b["args"]["target"])
            dirv = (tgt - p0)
            if tau < 4:
                pos = p0 - 0.03 * dirv.normalized() * ease(tau / 4)
            elif tau < D - 6:
                u = ease((tau - 4) / max(D - 10, 1))
                pos = (p0 - 0.03 * dirv.normalized()).lerp(tgt + 0.08 * dirv, u)
            elif tau <= D + hold:
                pos = (tgt + 0.08 * dirv).lerp(tgt, ease((tau - (D - 6)) / 6))
            else:
                pos = tgt.lerp(fk.to_translation(), ease((tau - D - hold) / ret))
            ik_fk = 1 - ease(tau / 4) if tau < 4 else (ease((tau - D - hold) / ret) if tau > D + hold else 0.0)
            Mh = M0.copy(); Mh.translation = pos
            self.place(f"hand_ik.{s}", Mh)
            self.pb[f"upper_arm_parent.{s}"]["IK_FK"] = ik_fk
            self.wrist_straight(s, pos)
            self.report.setdefault("reach_targets", {})[b["id"]] = {"side": s, "target": list(tgt), "frames": [t0 + D - 6, t0 + D + hold]}
        self.stale_all = True                             # an IK/FK switch moves bones no dependency lists

    def wrist_straight(self, s, pos):
        """Turn the IK hand at pos as its forearm is turned since rest, once the arm has reached
        for it: the wrist stays straight (anatomy.py: a hand kept as it hung, its arm raised,
        bent 70-150 deg toward the thumb). The hand's turn does not move the arm's IK."""
        self.update()
        fa, ik = "DEF-FOREARM-04." + s, R.BONE[f"hand_ik.{s}"]
        B = self.rig.data.bones
        Mh = self.rig.pose.bones[fa].matrix @ B[fa].matrix_local.inverted() @ B[ik].matrix_local
        Mh.translation = pos
        self.place(f"hand_ik.{s}", Mh)
        return Mh

    # A hand beat is more than its finger shape: the arm makes the gesture. A point extends the arm
    # toward the beat's target (the member it points at) or ahead; clenched fists come up, tense,
    # in front of the belly; open hands come forward and out, palms open. The hand moves there by
    # IK over the first ~10 frames, holds for the beat, and eases back to the arm's own motion over
    # 10 frames after it (as a reach does). Goals are set from where the shoulder is when the beat
    # starts, in the body's frame (yaw), so a gesture made while turning stays the body's.
    GESTURES = {"point", "fist", "open"}

    def gesture(self, f, t):
        # per arm, the gesture beats in order; at frame f only the latest that has started and has
        # not finished returning moves the arm, and one that follows another starts from that one's
        # goal, in IK the whole way, so the hand travels from gesture to gesture without a pop
        plan = {"L": [], "R": []}
        for b in self.p["beats"]:
            if b["verb"] != "hand" or b["args"].get("shape") not in self.GESTURES: continue
            shape, side = b["args"]["shape"], b["args"].get("side", "both")
            for s in (["R"] if shape == "point" and side == "both" else (["L", "R"] if side == "both" else [side])):
                plan[s].append(b)
        ret = 10
        for s, beats in plan.items():
            beats.sort(key=lambda b: b["start"])
            live = [k for k, b in enumerate(beats) if b["start"] <= f <= b["start"] + b["dur"] + ret]
            if not live: continue
            k = live[-1]; b = beats[k]
            if any(r["verb"] == "reach" and r["args"]["side"] == s and r["start"] <= f <= r["start"] + r["dur"] + 24
                   for r in self.p["beats"]): continue          # a reach has this arm
            t0, D = b["start"], b["dur"]
            tau, app = f - t0, min(10, max(4, D // 2))
            nxt = beats[k + 1]["start"] if k + 1 < len(beats) else None
            self.fresh(f"hand_fk.{s}")
            fk = self.pb[f"hand_fk.{s}"].matrix.copy()
            key = (b["id"], s)
            if key not in self.gesture_start:
                prev = self.gesture_start.get((beats[k - 1]["id"], s)) if k else None
                chained = prev is not None and t0 <= beats[k - 1]["start"] + beats[k - 1]["dur"] + ret
                S = self.pb[f"DEF-upper_arm.{s}"].head.copy()
                Y = self.yawm.to_3x3()
                fwd, left, up = Y @ Vector((0, -1, 0)), Y @ Vector((1, 0, 0)), Vector((0, 0, 1))
                out = 1.0 if s == "L" else -1.0           # away from the body's middle
                shape = b["args"]["shape"]
                if shape == "point":
                    tg = b["args"].get("target")
                    dirv = (Vector(tg) - S).normalized() if tg else (fwd - 0.15 * up).normalized()
                    goal = S + dirv * 0.92 * self.arm_len
                elif shape == "fist":
                    goal = S + fwd * 0.28 - left * out * 0.06 - up * 0.38
                else:                                      # open
                    goal = S + fwd * 0.32 + left * out * 0.18 - up * 0.22
                M0 = prev[3] if chained else fk.copy()      # the hand as the previous gesture left it
                self.gesture_start[key] = (M0, goal, S, None, chained)
            M0, goal, S, _, chained = self.gesture_start[key]
            p0 = M0.to_translation()
            ends = nxt is not None and nxt <= t0 + D + ret   # the next gesture takes over: no return
            if tau < app:
                pos, ik_fk = p0.lerp(goal, ease(tau / app)), (0.0 if chained else 1 - ease(min(1.0, tau / 3)))
            elif tau <= D or ends:
                pos, ik_fk = goal, 0.0
            else:
                # back in IK to where the FK hand is (the pole is on the FK elbow), over 70% of the
                # return, and only then over to FK: blending IK into FK bone by bone from a raised arm bent the elbow 49 deg sideways
                v = (tau - D) / ret; pos, ik_fk = goal.lerp(fk.to_translation(), ease(min(1.0, v / 0.7))), ease(max(0.0, (v - 0.7) / 0.3))
            Mh = M0.copy(); Mh.translation = pos
            self.place(f"hand_ik.{s}", Mh)
            self.pb[f"upper_arm_parent.{s}"]["IK_FK"] = ik_fk
            Mh = self.wrist_straight(s, pos)                # the hand lies along its forearm
            g = self.gesture_start[key]; self.gesture_start[key] = (g[0], g[1], g[2], Mh.copy(), g[4])
        self.stale_all = True

    # -- the program's own layer ------------------------------------------------------------
    def apply_keys(self, f):
        for (ctrl, ch, mode), ks in self.tracks.items():
            v, w = self._sample(ks, f)
            if w <= 0: continue
            if ctrl in self.reg["face"]:
                self.face_vals[ctrl] = lerp(self.face_vals[ctrl], v, w) if mode == "abs" else self.face_vals[ctrl] + v * w
            elif ctrl in self.reg["digits"]:
                d, s = ctrl.split(".")
                for i in (1, 2, 3):
                    n = f"{d}.0{i}.{s}"
                    flex, toward = self.axes[n]
                    q = self.pb[n].rotation_quaternion.copy()
                    if ch == "curl":
                        tq = Quaternion(flex, math.radians(v * R.CURL_FULL[d][i - 1]))
                        self.pb[n].rotation_quaternion = q.slerp(tq, w) if mode == "abs" else q @ Quaternion(flex, math.radians(v * R.CURL_FULL[d][i - 1] * w))
                    elif i == 1 and toward is not None:
                        self.pb[n].rotation_quaternion = Quaternion(toward, math.radians(v * w)) @ q
            else:
                b = self.pb[ctrl]
                if ch == "loc":
                    vv = Vector(v)
                    b.location = b.location.lerp(vv, w) if mode == "abs" else b.location + vv * w
                else:
                    e = Euler([math.radians(x) for x in v], "XYZ").to_quaternion()
                    q0 = self._q(b)
                    q = q0.slerp(e, w) if mode == "abs" else q0 @ Quaternion().slerp(e, w)
                    self._setq(b, q)
            for k in ks:
                if k["pin"] and k["f"] == f:
                    self.report["pins"].append({"ctrl": ctrl, "ch": ch, "f": f, "v": k["v"], "mode": mode})
        self.stale_all = True                             # keys set bones directly

    def _sample(self, ks, f):
        """Value and weight of a key track at frame f: monotone between keys (no overshoot),
        eased in and out over each end's blend frames, held to the end if the last key says so."""
        fs = [k["f"] for k in ks]
        first, last = ks[0], ks[-1]
        bi, bo = first.get("blend", 6), last.get("blend", 6)
        if f < first["f"]:
            return first["v"], ease(1 - (first["f"] - f) / bi) if bi and f > first["f"] - bi else 0.0
        if f > last["f"]:
            if last.get("hold"): return last["v"], 1.0
            return last["v"], ease(1 - (f - last["f"]) / bo) if bo and f < last["f"] + bo else 0.0
        i = max(0, bisect_right(fs, f) - 1)
        if i >= len(ks) - 1 or ks[i]["f"] == f: return ks[i]["v"], 1.0
        a, b = ks[i], ks[i + 1]
        u = (f - a["f"]) / (b["f"] - a["f"])
        if a["ease"] == "constant": u = 0.0
        elif a["ease"] == "auto_clamped": u = ease(u)
        if isinstance(a["v"], list): return [lerp(x, y, u) for x, y in zip(a["v"], b["v"])], 1.0
        return lerp(a["v"], b["v"], u), 1.0

    def apply_props(self, f):
        for prop, ks in self.props.items():
            bone, name = prop.split(":")
            if f < ks[0][0]: continue
            i = max(0, bisect_right([k[0] for k in ks], f) - 1)
            if i < len(ks) - 1:
                (f0, v0), (f1, v1) = ks[i], ks[i + 1]
                v = lerp(v0, v1, (f - f0) / (f1 - f0))
            else:
                v = ks[i][1]
            self.pb[bone][name] = v
        self.stale_all = True

    def clamp(self, f):
        for name, c in self.reg["controls"].items():
            b = self.pb[name]
            if "rot_deg" in c:
                q = self._q(b); e = q.to_euler("XYZ")
                new = [max(lo, min(hi, math.degrees(a))) for a, (lo, hi) in zip(e, c["rot_deg"])]
                over = max(abs(math.degrees(a) - n) for a, n in zip(e, new))
                if over > 0.05:
                    self._setq(b, Euler([math.radians(x) for x in new], "XYZ").to_quaternion())
                    self._note_clamp(name, "rot", f, over)
            if "loc_m" in c:
                new = [max(lo, min(hi, x)) for x, (lo, hi) in zip(b.location, c["loc_m"])]
                over = max(abs(x - n) for x, n in zip(b.location, new))
                if over > 1e-4:
                    b.location = new; self._note_clamp(name, "loc", f, over)
        for k in R.FACE_KEYS:
            self.face_vals[k] = max(0.0, min(1.0, self.face_vals[k]))
        # Keyframes are taken from the values set on the bones, so nothing is evaluated here; the
        # next frame starts with an update.
        self.stale_all = True

    def _note_clamp(self, name, ch, f, over):
        r = self.report["clamps"].setdefault(f"{name}.{ch}", {"frames": 0, "max": 0.0, "first": f})
        r["frames"] += 1; r["max"] = round(max(r["max"], over), 4)

    def _q(self, b):
        return b.rotation_quaternion.copy() if b.rotation_mode == "QUATERNION" else b.rotation_euler.to_quaternion()

    def _setq(self, b, q):
        if b.rotation_mode == "QUATERNION": b.rotation_quaternion = q
        else: b.rotation_euler = q.to_euler(b.rotation_mode)

    def key(self, frame):
        self.insert(frame, self.snapshot())

    def snapshot(self):
        """The values this frame keys: each control's and digit's location and rotation, the
        IK/FK switches and the face keys."""
        bones = {}
        for n in self.ctrl + self.digit_bones:
            b = self.pb[n]
            rot = b.rotation_quaternion if b.rotation_mode == "QUATERNION" else b.rotation_euler
            bones[n] = [list(b.location), list(rot)]
        props = {f"{base}.{s}": self.pb[f"{base}.{s}"]["IK_FK"] for s in "LR" for base in ("thigh_parent", "upper_arm_parent")}
        run = {}
        for n in self.run_bones:                          # the reference rig's bones a run poses (scale too)
            b = self.pb[n]
            run[n] = [list(b.location), list(b.rotation_quaternion if b.rotation_mode == "QUATERNION" else b.rotation_euler), list(b.scale)]
        snap = {"bones": bones, "props": props, "face": dict(self.face_vals) if self.face else {}}
        if self.run_bones: snap.update(run=run, head_follow=self.head_follow)
        return snap

    def face_snapshot(self, f):
        """The face's values on frame index f, as frame() and apply_keys leave them."""
        self.face_frame(f / self.fps)
        for (ctrl, ch, mode), ks in self.tracks.items():
            if ctrl not in self.reg["face"]: continue
            v, w = self._sample(ks, f)
            if w > 0: self.face_vals[ctrl] = lerp(self.face_vals[ctrl], v, w) if mode == "abs" else self.face_vals[ctrl] + v * w
        return {"face": dict(self.face_vals) if self.face else {}}

    def insert(self, frame, snap):
        """Key a snapshot on a frame; one with the face alone keys only the face."""
        if "bones" not in snap:
            for k, v in snap["face"].items():
                self.face[k].value = v; self.face[k].keyframe_insert("value", frame=frame)
            return
        for n in self.ctrl + self.digit_bones:
            b = self.pb[n]; loc, rot = snap["bones"][n]
            b.location = loc
            if b.rotation_mode == "QUATERNION": b.rotation_quaternion = rot
            else: b.rotation_euler = rot
            b.keyframe_insert("location", frame=frame)
            b.keyframe_insert("rotation_quaternion" if b.rotation_mode == "QUATERNION" else "rotation_euler", frame=frame)
        for name, v in snap["props"].items():
            self.pb[name]["IK_FK"] = v; self.pb[name].keyframe_insert('["IK_FK"]', frame=frame)
        for n, (loc, rot, scl) in snap.get("run", {}).items():
            b = self.pb[n]; b.location = loc; b.scale = scl
            if b.rotation_mode == "QUATERNION": b.rotation_quaternion = rot
            else: b.rotation_euler = rot
            b.keyframe_insert("location", frame=frame); b.keyframe_insert("scale", frame=frame)
            b.keyframe_insert("rotation_quaternion" if b.rotation_mode == "QUATERNION" else "rotation_euler", frame=frame)
        if "head_follow" in snap:
            P = self.pb["PROPERTIES"]; P["HEAD-FOLLOW"] = snap["head_follow"]; P.keyframe_insert('["HEAD-FOLLOW"]', frame=frame)
        if self.face:
            for k, v in snap["face"].items():
                self.face[k].value = v; self.face[k].keyframe_insert("value", frame=frame)
