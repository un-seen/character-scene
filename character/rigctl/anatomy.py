"""Joint angles as anatomy names them, read from the deform bones, whatever moved them.

Each joint's rotation is the child bone's turn relative to its parent since rest, in the child's
rest frame, split into swing (the bone's direction: two signed angles about its rest X and Z
axes) and twist (about the bone, its Y). The angles are degrees.
"""
import math

try:
    from mathutils import Quaternion, Vector
except ImportError:                   # outside Blender (npcheck.py): the limits and violations only
    Quaternion = Vector = None

# joint: (parent deform bone, child deform bone); sided ones are named with {s}
JOINTS = {
    "spine": ("DEF-HIPS", "DEF-CHEST-02"),
    "neck": ("DEF-CHEST-02", "DEF-HEAD"),
    "shoulder.{s}": ("DEF-CHEST-02", "DEF-ARM-01.{s}"),
    "elbow.{s}": ("DEF-ARM-04.{s}", "DEF-FOREARM-01.{s}"),
    "wrist.{s}": ("DEF-FOREARM-04.{s}", "DEF-HAND.{s}"),
    "hip.{s}": ("DEF-HIPS", "DEF-LEG-01.{s}"),
    "knee.{s}": ("DEF-LEG-04.{s}", "DEF-SHIN-01.{s}"),
    "ankle.{s}": ("DEF-SHIN-04.{s}", "DEF-FOOT.{s}"),
}


def joints():
    out = {}
    for j, (p, c) in JOINTS.items():
        for s in ("LR" if "{s}" in j else "-"):
            out[j.format(s=s)] = (p.format(s=s), c.format(s=s))
    return out


def swing_twist(q):
    """(swing about X, swing about Z, twist about Y) in degrees, q = swing @ twist."""
    t = Quaternion((q.w, 0, q.y, 0))
    if t.magnitude < 1e-9: t = Quaternion()
    t.normalize()
    sw = q @ t.inverted()
    if sw.w < 0: sw.negate()
    tw = (2 * math.degrees(math.atan2(t.y, t.w)) + 180) % 360 - 180
    a = sw.angle
    ax = sw.axis if a > 1e-9 else Vector((1, 0, 0))
    return math.degrees(a * ax.x), math.degrees(a * ax.z), tw


class Angles:
    def __init__(self, rig):
        self.rig = rig
        self.j = joints()
        B = rig.data.bones
        self.rest = {j: (B[p].matrix_local.inverted() @ B[c].matrix_local).to_quaternion() for j, (p, c) in self.j.items()}

    def read(self):
        P = self.rig.pose.bones
        out = {}
        for j, (p, c) in self.j.items():
            cur = (P[p].matrix.inverted() @ P[c].matrix).to_quaternion()
            out[j] = tuple(round(x, 1) for x in swing_twist(self.rest[j].inverted() @ cur))   # the turn since rest, in the child's rest frame
        return out


# Human ranges of motion (degrees, AAOS and Kapandji's figures, a few degrees wider for the rig's
# flesh), as (component, low, high) on this rig's angles, for the left side; the right's swing about
# Z and twist mirror. Components: 0 swing about X, 1 swing about Z, 2 twist. What each is, measured on
# the rig (anaprobe.py): the elbow's hinge is Z (L flexes -Z), its X is a sideways bend no elbow
# makes; the knee flexes +Z (L); the wrist extends +X and deviates on Z (L ulnar +Z); the ankle
# flexes up +X (toes to the shin; measured on renders 2026-11-01: a heel peeled up reads -X) and points -X; the hip flexes -Z (L) and abducts +X; the shoulder raises the arm +X from the
# rig's T and swings it forward -Z (L); the spine flexes +X and bends to its left -Z; the neck flexes
# +X, bends to the side on Z and turns on twist.
LIMITS = {
    "spine":    [(0, -35, 90, "extends back", "bends forward"), (1, -40, 40, "bends to its left", "bends to its right"), (2, -50, 50, "turns", "turns")],
    "neck":     [(0, -70, 70, "tips back", "drops forward"), (1, -45, 45, "tilts", "tilts"), (2, -85, 85, "turns", "turns")],
    "shoulder": [(0, -100, 95, "swings down into the body", "raises past overhead"), (1, -135, 50, "swings across the body", "swings back"), (2, -90, 90, "rolls", "rolls")],
    "elbow":    [(0, -12, 12, "bends sideways", "bends sideways"), (1, -150, 5, "bends", "bends backwards"), (2, -85, 85, "twists", "twists")],
    "wrist":    [(0, -80, 70, "flexes", "extends"), (1, -25, 40, "bends to the thumb", "bends to the little finger"), (2, -15, 15, "twists", "twists")],
    "hip":      [(0, -35, 50, "crosses in", "spreads out"), (1, -130, 30, "bends", "extends back"), (2, -45, 45, "rolls", "rolls")],
    "knee":     [(0, -8, 8, "bends sideways", "bends sideways"), (1, -5, 155, "bends backwards", "bends"), (2, -30, 30, "twists", "twists")],
    "ankle":    [(0, -50, 40, "points", "flexes up"), (1, -35, 35, "rolls", "rolls"), (2, -30, 30, "twists", "twists")],
}
SIDE = {"L": "left", "R": "right", "-": ""}


def limits(joint):
    """(component, low, high, word below, word above) for a joint as read, the right side mirrored."""
    base, _, s = joint.partition(".")
    out = []
    for i, lo, hi, wlo, whi in LIMITS[base]:
        if s == "R" and i in (1, 2): lo, hi, wlo, whi = -hi, -lo, whi, wlo
        out.append((i, lo, hi, wlo, whi))
    return out


def violations(angles, slack=0.0):
    """Every angle outside its human range: [(joint, component, value, limit, what it does)]."""
    out = []
    for j, v in angles.items():
        for i, lo, hi, wlo, whi in limits(j):
            if v[i] < lo - slack: out.append((j, i, v[i], lo, wlo))
            elif v[i] > hi + slack: out.append((j, i, v[i], hi, whi))
    return out
