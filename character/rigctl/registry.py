"""The control registry: every name a program may use, with its channels, units and limits, and
where each one is on the rig.

The rig is out/rig.blend ("Rig", its body "Body"; not published: see the repository README). Programs, the compiler and the checker keep the names they were written with,
which came from the first rig, a generated Rigify one: "torso", "hand_ik.L", "f_index.02.L",
"DEF-spine.006" for the head. BONE says which bone of this rig each one is, and Pose and Bones
look them up through it; a name it does not hold is the rig's own.

The rig defines no joint limits, so the limits here are written by hand (degrees per local Euler
XYZ axis, metres from rest). They are deliberately the true range of the control, not a
dataset's percentiles: every legal pose must stay reachable.
"""
import json

RIG = "Rig"
BODY = "Body"
FACE_KEYS = ["blink", "smile", "brows_up", "frown", "jaw_open"]
DIGITS = ["thumb", "f_index", "f_middle", "f_ring", "f_pinky"]

# (control, channels, rot limit deg per axis, loc limit m per axis)
_TABLE = [
    ("root",            ("loc", "rot"), (90, 0, 360),  (100, 100, 5)),     # pitched, it lies the body down
    ("torso",           ("loc", "rot"), (75, 75, 75),  (0.6, 0.6, 1.0)),    # the reference rig's run tips it 67; sitting drops it 0.87
    ("hips",            ("rot",),       (45, 45, 45),  None),
    ("chest",           ("rot",),       (45, 45, 45),  None),
    ("neck",            ("rot",),       (45, 60, 45),  None),
    ("head",            ("rot",),       (70, 80, 70),  None),
]
_SIDED = [
    ("shoulder",            ("rot",),       (40, 40, 40),    None),
    ("upper_arm_fk",        ("rot",),       (160, 160, 160), None),
    ("forearm_fk",          ("rot",),       (150, 150, 150), None),
    ("hand_fk",             ("rot",),       (80, 80, 80),    None),
    ("hand_ik",             ("loc", "rot"), (180, 180, 180), (1.2, 1.2, 1.2)),
    ("upper_arm_ik_target", ("loc",),       None,            (1.0, 1.0, 1.0)),
    ("palm",                ("rot",),       (30, 30, 30),    None),
    ("thigh_fk",            ("rot",),       (120, 120, 120), None),
    ("shin_fk",             ("rot",),       (150, 150, 150), None),
    ("foot_fk",             ("rot",),       (60, 60, 60),    None),
    ("toe_fk",              ("rot",),       (60, 60, 60),    None),
    ("foot_ik",             ("loc", "rot"), (180, 180, 180), (1.2, 1.2, 1.2)),
    ("foot_heel_ik",        ("rot",),       (60, 60, 60),    None),
    ("toe_ik",              ("rot",),       (70, 70, 70),    None),
    ("thigh_ik_target",     ("loc",),       None,            (1.0, 1.0, 1.0)),
]
# The program's name -> this rig's bone. Controls first, then the bones the compiler and checker
# only read: the hip joint, and the deform bones of the head, neck, chest, hips, upper arms,
# hands and feet. The reference rig's palm is four bones, one per finger; "palm" is the middle
# finger's. Its fingers are lettered from the little finger (A) to the index (D).
BONE = {"root": "ROOT", "torso": "TORSO", "hips": "HIPS", "chest": "chest", "neck": "NECK", "head": "HEAD",
        "DEF-spine": "DEF-HIPS", "DEF-spine.003": "DEF-CHEST-02", "DEF-spine.004": "DEF-NECK-01",
        "DEF-spine.005": "DEF-NECK-02", "DEF-spine.006": "DEF-HEAD"}
_SIDED_BONE = {"shoulder": "SHOULDER", "upper_arm_fk": "ARM-FK", "forearm_fk": "FOREARM-FK", "hand_fk": "HAND-FK",
               "hand_ik": "HAND-IK", "upper_arm_ik_target": "ARM-POLE-IK", "palm": "PALM-C",
               "thigh_fk": "LEG-FK", "shin_fk": "SHIN-FK", "foot_fk": "FOOT-FK", "toe_fk": "TOES-FK",
               "foot_ik": "FOOT-IK-MASTER", "foot_heel_ik": "FOOT-ROLE-IK", "toe_ik": "TOES-IK",
               "thigh_ik_target": "LEG-POLE-IK", "thigh_ik": "ORG-LEG-01",
               "DEF-upper_arm": "DEF-ARM-01", "DEF-hand": "DEF-HAND", "DEF-foot": "DEF-FOOT"}
_DIGIT_BONE = {"thumb": "THUMB", "f_index": "FINGER-D", "f_middle": "FINGER-C", "f_ring": "FINGER-B", "f_pinky": "FINGER-A"}
for _s in "LR":
    BONE.update({f"{k}.{_s}": f"{v}.{_s}" for k, v in _SIDED_BONE.items()})
    BONE.update({f"{d}.0{i}.{_s}": f"{v}-0{i}.{_s}" for d, v in _DIGIT_BONE.items() for i in (1, 2, 3)})

# Limb switches: a program's "upper_arm_parent.R:IK_FK" (as Rigify had it: 0 IK, 1 FK) is the
# influence of the rig's "FK-IK" constraints on the limb's MCH-SWITCH bones (0 FK, 1 IK).
_SWITCHED = {"upper_arm_parent": ("ARM", "FOREARM", "HAND"), "thigh_parent": ("LEG", "SHIN", "FOOT", "TOES", "TOES-BIG")}
_PROPS = {"upper_arm_parent": ("IK_FK",), "thigh_parent": ("IK_FK",)}
# Digit curl: 0 is straight, 1 a full fist; degrees reached at 1 for each segment.
CURL_FULL = {"thumb": (58, 50, 50), "f_index": (75, 92, 50), "f_middle": (75, 92, 50),
             "f_ring": (75, 92, 50), "f_pinky": (75, 92, 50)}
SPREAD_LIMIT = 25.0


class Switch:
    """A limb's switch, read and keyed as Rigify's IK_FK property was (0 IK, 1 FK)."""

    def __init__(self, rig, base, side):
        self.cons = [rig.pose.bones[f"MCH-SWITCH-{p}.{side}"].constraints["FK-IK"] for p in _SWITCHED[base]]
        self.paths = [f'pose.bones["MCH-SWITCH-{p}.{side}"].constraints["FK-IK"].influence' for p in _SWITCHED[base]]
        self.rig = rig

    def __contains__(self, key): return key == "IK_FK"
    def __getitem__(self, key): return 1.0 - self.cons[0].influence

    def __setitem__(self, key, v):
        if key != "IK_FK": raise KeyError(key)
        for c in self.cons: c.influence = 1.0 - v

    def keyframe_insert(self, path, frame):
        for p in self.paths: self.rig.keyframe_insert(p, frame=frame)


class Bones:
    """A rig's bones (rig.pose.bones or rig.data.bones) by the program's names."""

    def __init__(self, rig, bones, switches=False):
        self.rig, self.bones = rig, bones
        self.switch = {f"{b}.{s}": Switch(rig, b, s) for b in _SWITCHED for s in "LR"} if switches else {}

    def __getitem__(self, name):
        return self.switch[name] if name in self.switch else self.bones[BONE.get(name, name)]

    def __contains__(self, name): return name in self.switch or BONE.get(name, name) in self.bones
    def __iter__(self): return iter(self.bones)
    def get(self, name, default=None): return self[name] if name in self else default


def Pose(rig): return Bones(rig, rig.pose.bones, switches=True)


def Rest(rig): return Bones(rig, rig.data.bones)


def _axes(rig, name):
    """Which world direction each local axis points at rest (rounded), as a hint for writers."""
    m = Rest(rig)[name].matrix_local.to_3x3()
    return {a: [round(v, 2) for v in m.col[i]] for i, a in enumerate("xyz")}


def build(rig):
    pb = Pose(rig)
    controls = {}
    for name, ch, rl, ll in _TABLE:
        controls[name] = _entry(rig, name, ch, rl, ll, None, "body")
    for base, ch, rl, ll in _SIDED:
        for s, o in (("L", "R"), ("R", "L")):
            n = f"{base}.{s}"
            if n in pb:
                controls[n] = _entry(rig, n, ch, rl, ll, f"{base}.{o}", base)
    digits = {f"{d}.{s}": {"channels": ["curl", "spread"], "curl": [0.0, 1.0],
                           "spread_deg": [-SPREAD_LIMIT, SPREAD_LIMIT],
                           "mirror": f"{d}.{'R' if s == 'L' else 'L'}", "group": "digit"}
              for d in DIGITS for s in "LR"}
    props = {}
    for base, names in _PROPS.items():
        for s in "LR":
            for p in names:
                if p in pb[f"{base}.{s}"]:
                    props[f"{base}.{s}:{p}"] = {"range": [0.0, 1.0],
                                                "rest": float(pb[f"{base}.{s}"][p])}
    face = {k: {"range": [0.0, 1.0]} for k in FACE_KEYS}
    return {"rig": RIG, "units": {"rot": "deg, local Euler XYZ from rest", "loc": "m, bone-local from rest",
                                  "curl": "0 straight .. 1 fist", "spread": "deg toward the thumb side"},
            "controls": controls, "digits": digits, "props": props, "face": face}


def _entry(rig, name, ch, rl, ll, mirror, group):
    e = {"channels": list(ch), "group": group, "axes_at_rest": _axes(rig, name)}
    if rl: e["rot_deg"] = [[-a, a] for a in rl]
    if ll: e["loc_m"] = [[-a, a] for a in ll]
    if mirror: e["mirror"] = mirror
    return e


def save(reg, path):
    json.dump(reg, open(path, "w"), indent=1)
