"""A cast member (humans/ID_mh.blend, build_human.py) put on the reference rig skeleton, 2026-11-02.

The rig stays out/rig.blend's ("Rig": the solver, the compiler and every clip are built on it);
only the body it carries changes. Two steps:

1. Fit. Every MakeHuman bone gets one affine transform that carries its rest pose onto the reference rig's:
   a limb segment (shoulder to elbow, elbow to wrist, hip to knee, knee to ankle, clavicle, neck) is
   swung from MakeHuman's A pose to the reference rig's T pose and stretched along its length to the reference rig's
   length (across it, scaled by the height ratio); a hand or a foot is turned as a frame (wrist, middle
   knuckle, index to little knuckle; ankle, ball, left to right); the torso is scaled per axis between
   hip and shoulder widths and heights; the head is moved and scaled whole. Each vertex moves by its
   MakeHuman weights (linear blend skinning of those transforms), clothes and hair alike.
2. Bind. The full fitted skin (nothing deleted under the clothes) takes the reference rig body's deform
   weights at the nearest point of the reference rig's skin (barycentric, so its twist and corrective bones
   carry over), except what MakeHuman weights to the head (face, eyes, hair, teeth): that rides
   DEF-HEAD. Around the shoulders and hips those weights are smoothed over the skin. Every mesh
   (body, clothes, hair, eyes) then reads its weights from that one skin, so cloth moves with the
   skin under it.

  blender -b out/rig.blend --python humans/fit_human.py -- ID [SMOOTH_ITERS [OUT]]
      -> humans/ID.blend (the reference rig, this body on it; the reference rig body kept, hidden, for the checker)"""
import json
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

HERE = os.path.dirname(os.path.abspath(__file__))
ARGS = sys.argv[sys.argv.index("--") + 1:]
cid = ARGS[0]
SMOOTH_ITERS = int(ARGS[1]) if len(ARGS) > 1 else 40
OUT = cid if len(ARGS) < 3 else ARGS[2]
WEIGHTS = os.environ.get("WEIGHTS", "mh")              # mh: MakeHuman's own, renamed (human shoulders); rig: the rig body's
RIG, BODY = "Rig", "Body"
rig, body = bpy.data.objects[RIG], bpy.data.objects[BODY]

# ---- the MakeHuman character, appended -------------------------------------------------------
src = os.path.join(HERE, f"{cid}_mh.blend")
with bpy.data.libraries.load(src) as (fr, to):
    to.objects = list(fr.objects)
mh = [o for o in to.objects if o is not None]
for o in mh: bpy.context.scene.collection.objects.link(o)
mrig = next(o for o in mh if o.type == "ARMATURE")
meshes = [o for o in mh if o.type == "MESH"]
dg = bpy.context.evaluated_depsgraph_get()


def T(name):                                            # the reference rig's rest bone (world)
    b = rig.data.bones[name]; M = rig.matrix_world
    return M @ b.head_local, M @ b.tail_local


def S(name):                                            # MakeHuman's rest bone (world)
    b = mrig.data.bones[name]; M = mrig.matrix_world
    return M @ b.head_local, M @ b.tail_local


def mean(*v): return sum(v, Vector()) / len(v)


H_t = T("DEF-HEAD")[1].z                                # heights to the top of the head bone
H_s = S("head")[1].z
k = H_t / H_s                                           # across a limb: scaled with the body's height


def mat(A3, a_src, a_dst):
    """x -> A3 (x - a_src) + a_dst, as a 4x4."""
    M = Matrix.Identity(4)
    for i in range(3):
        for j in range(3): M[i][j] = A3[i][j]
    t = a_dst - A3 @ a_src
    M[0][3], M[1][3], M[2][3] = t
    return M


def segment(a_s, b_s, a_d, b_d, across=None):
    """The swing that turns a_s->b_s onto a_d->b_d, with the length stretched to fit and the girth
    scaled by `across` (k by default)."""
    across = k if across is None else across
    u = (b_s - a_s); L_s = u.length; u.normalize()
    v = (b_d - a_d); L_d = v.length; v.normalize()
    R = u.rotation_difference(v).to_matrix()
    U = Matrix(((u.x * u.x, u.x * u.y, u.x * u.z), (u.y * u.x, u.y * u.y, u.y * u.z), (u.z * u.x, u.z * u.y, u.z * u.z)))
    Sc = Matrix.Identity(3) * across + U * (L_d / L_s - across)
    return mat(R @ Sc, a_s, a_d)


def frame(o, x_to, y_hint):
    x = (x_to - o).normalized(); y = (y_hint - x * y_hint.dot(x)).normalized(); z = x.cross(y)
    return Matrix((x, y, z)).transposed()


def turned(o_s, x_s, y_s, o_d, x_d, y_d):
    """A hand or a foot as a whole: its frame turned onto the reference rig's, scaled by its length."""
    F_s, F_d = frame(o_s, x_s, y_s), frame(o_d, x_d, y_d)
    s = (x_d - o_d).length / (x_s - o_s).length
    return mat(F_d @ F_s.transposed() * s, o_s, o_d)


X = {}                                                   # MakeHuman bone -> its transform
for side in ("L", "R"):
    sh_s, el_s, wr_s = S(f"upperarm01.{side}")[0], S(f"lowerarm01.{side}")[0], S(f"wrist.{side}")[0]
    sh_d, el_d, wr_d = T(f"DEF-ARM-01.{side}")[0], T(f"DEF-FOREARM-01.{side}")[0], T(f"DEF-HAND.{side}")[0]
    up = segment(sh_s, el_s, sh_d, el_d); fore = segment(el_s, wr_s, el_d, wr_d)
    for b in ("upperarm01", "upperarm02", "shoulder01"): X[f"{b}.{side}"] = up
    for b in ("lowerarm01", "lowerarm02"): X[f"{b}.{side}"] = fore
    cl_s = S(f"clavicle.{side}")[0]; cl_d = T(f"DEF-SHOULDER.{side}")[0]
    X[f"clavicle.{side}"] = segment(cl_s, sh_s, cl_d, sh_d)
    # the hand: wrist, middle knuckle, index to little knuckle (the reference rig's fingers run little A .. index D)
    hand = turned(wr_s, S(f"finger3-1.{side}")[0], S(f"finger2-1.{side}")[0] - S(f"finger5-1.{side}")[0] + wr_s,
                  wr_d, T(f"DEF-FINGER-C-01.{side}")[0], T(f"DEF-FINGER-D-01.{side}")[0] - T(f"DEF-FINGER-A-01.{side}")[0] + wr_d)
    for b in mrig.data.bones:
        if b.name.endswith("." + side) and (b.name.startswith(("wrist", "finger", "metacarpal"))): X[b.name] = hand
    # the twist the hand asks of the forearm, for the record
    hip_s, kn_s, an_s = S(f"upperleg01.{side}")[0], S(f"lowerleg01.{side}")[0], S(f"foot.{side}")[0]
    hip_d, kn_d, an_d = T(f"DEF-LEG-01.{side}")[0], T(f"DEF-SHIN-01.{side}")[0], T(f"DEF-FOOT.{side}")[0]
    thigh = segment(hip_s, kn_s, hip_d, kn_d); shin = segment(kn_s, an_s, kn_d, an_d)
    for b in ("upperleg01", "upperleg02"): X[f"{b}.{side}"] = thigh
    for b in ("lowerleg01", "lowerleg02"): X[f"{b}.{side}"] = shin
    ball_s, ball_d = S(f"toe3-1.{side}")[0], T(f"DEF-TOES.{side}")[0]
    lat = Vector((1, 0, 0)) if side == "L" else Vector((-1, 0, 0))
    foot = turned(an_s, ball_s, an_s + lat, an_d, ball_d, an_d + lat)
    for b in mrig.data.bones:
        if b.name.endswith("." + side) and b.name.startswith(("foot", "toe")): X[b.name] = foot
    print("FIT", side, "hand twist vs forearm (deg):",
          round(np.degrees((hand.to_3x3().normalized() @ fore.to_3x3().normalized().inverted()).to_quaternion().angle), 1))

# the torso: per-axis scale, hip width at the hips to shoulder width at the shoulders
hc_s = mean(S("upperleg01.L")[0], S("upperleg01.R")[0]); hc_d = mean(T("DEF-LEG-01.L")[0], T("DEF-LEG-01.R")[0])
nk_s, nk_d = S("neck01")[0], T("DEF-NECK-01")[0]
hw = (T("DEF-LEG-01.L")[0].x - T("DEF-LEG-01.R")[0].x) / (S("upperleg01.L")[0].x - S("upperleg01.R")[0].x)
sw = (T("DEF-ARM-01.L")[0].x - T("DEF-ARM-01.R")[0].x) / (S("upperarm01.L")[0].x - S("upperarm01.R")[0].x)
hw = (hw + k) / 2                                       # the pelvis keeps half its own width
sh_z = S("upperarm01.L")[0].z
az = (T("DEF-ARM-01.L")[0].z - hc_d.z) / (sh_z - hc_s.z)   # hip joints to shoulder joints, as the reference rig's
for b in mrig.data.bones:
    if b.name in X: continue
    chain = [b] + list(b.parent_recursive)
    names = {c.name for c in chain}
    if "head" in names or "neck01" in names: continue
    t = min(max((b.head_local.z * mrig.matrix_world[2][2] + mrig.matrix_world[2][3] - hc_s.z) / (sh_z - hc_s.z), 0), 1)
    A3 = Matrix(((hw + (sw - hw) * t, 0, 0), (0, k, 0), (0, 0, az)))
    ys, yd = hc_s.y + (nk_s.y - hc_s.y) * t, hc_d.y + (nk_d.y - hc_d.y) * t
    X[b.name] = mat(A3, Vector((0, ys, hc_s.z)), Vector((0, yd, hc_d.z)))
# the neck and head keep their own proportions (scaled by k) from where the torso puts the neck's base:
# the head then sits a little off the reference rig's head pivot, which only a turn or a nod shows
hb_s = S("head")[0]
nk_at = Vector((0, nk_d.y, hc_d.z + az * (nk_s.z - hc_s.z)))
hb_at = nk_at + (hb_s - nk_s) * k
neck = mat(Matrix.Identity(3) * k, nk_s, nk_at)
for n in ("neck01", "neck02", "neck03"): X[n] = neck
head = mat(Matrix.Identity(3) * k, hb_s, hb_at)
for b in mrig.data.bones:
    if b.name not in X: X[b.name] = head
print("FIT head base off the reference rig pivot by", [round(x, 3) for x in (hb_at - T("DEF-HEAD")[0])], "top at", round((hb_at + (S("head")[1] - hb_s) * k).z, 3))
print("FIT k", round(k, 3), "hips", round(hw, 3), "shoulders", round(sw, 3), "torso height", round(az, 3))

HEADISH = {b.name for b in mrig.data.bones if b.name == "head" or "head" in {p.name for p in b.parent_recursive}}

# ---- weights: the reference rig body's, carried onto one full skin, smoothed, shared by every mesh --------
defs = [b.name for b in rig.data.bones if b.use_deform]
di = {n: i for i, n in enumerate(defs)}
HEAD_I = di["DEF-HEAD"]


class Surface:
    """Per-vertex weights on a triangle mesh, read at the nearest point of it (barycentric)."""
    def __init__(self, P, tris, W):
        self.P, self.tris, self.W = np.asarray(P), np.asarray(tris), W
        self.bvh = BVHTree.FromPolygons([tuple(p) for p in self.P], [tuple(t) for t in self.tris])

    def __call__(self, p):
        loc, _, fi, _ = self.bvh.find_nearest(Vector(p))
        a, b, c = (self.P[i] for i in self.tris[fi])
        v0, v1, v2 = b - a, c - a, np.array(loc) - a
        d00, d01, d11, d20, d21 = v0 @ v0, v0 @ v1, v1 @ v1, v2 @ v0, v2 @ v1
        den = max(d00 * d11 - d01 * d01, 1e-12)
        w1 = (d11 * d20 - d01 * d21) / den; w2 = (d00 * d21 - d01 * d20) / den
        w = np.clip(np.array([1 - w1 - w2, w1, w2]), 0, 1); w /= w.sum()
        return w @ self.W[self.tris[fi]]


def triangles(me):
    return [(p.vertices[0], p.vertices[i], p.vertices[i + 1]) for p in me.polygons for i in range(1, len(p.vertices) - 1)]


gi = {g.index: g.name for g in body.vertex_groups}
W_t = np.zeros((len(body.data.vertices), len(defs)), np.float32)
for v in body.data.vertices:
    for g in v.groups:
        n = gi[g.group]
        if n in di: W_t[v.index, di[n]] = g.weight
W_t /= np.maximum(W_t.sum(1, keepdims=True), 1e-8)
rig_weights = Surface([body.matrix_world @ v.co for v in body.data.vertices], triangles(body.data), W_t)


def fitted(o, full=False):
    """The mesh of `o` at rest, every vertex moved by the fit (MakeHuman weights over X), and the share
    of each vertex's MakeHuman weight on the head. full: the body with nothing deleted under clothes."""
    saved = {}
    for m in o.modifiers:
        saved[m.name] = m.show_viewport
        if m.type == "ARMATURE" or (full and m.type == "MASK" and m.name.startswith("Delete.")): m.show_viewport = False
    dg = bpy.context.evaluated_depsgraph_get()
    me = bpy.data.meshes.new_from_object(o.evaluated_get(dg), preserve_all_data_layers=True, depsgraph=dg)
    for m in o.modifiers: m.show_viewport = saved[m.name]
    names = {g.index: g.name for g in o.vertex_groups}
    me.transform(o.matrix_world)
    n = len(me.vertices)
    out = np.zeros((n, 3)); h = np.ones(n); own = []
    pre = np.array([v.co for v in me.vertices])
    for v in me.vertices:
        acc, tot, hh, mine = Vector(), 0.0, 0.0, {}
        for g in v.groups:
            b = names.get(g.group)
            if b not in X or g.weight <= 0: continue
            acc += (X[b] @ v.co) * g.weight; tot += g.weight; mine[b] = g.weight
            if b in HEADISH: hh += g.weight
        own.append(mine)
        out[v.index] = (acc / tot) if tot > 0 else (head @ v.co)
        if tot > 0: h[v.index] = hh / tot
    me.vertices.foreach_set("co", out.astype(np.float32).ravel())
    return me, out, h, own, pre


# MakeHuman's own weights, renamed: a bone that the reference rig splits into a chain (an upper arm into four
# twist segments, the spine into hips, spine and chest) shares its weight along that chain by where the
# vertex lies along it; the reference rig's fingers run little (A) to index (D)
CHAINS = {}
for s_ in "LR":
    for mhb, ch in ((("upperarm01", "upperarm02"), [f"DEF-ARM-0{i}.{s_}" for i in range(1, 5)]),
                    (("lowerarm01", "lowerarm02"), [f"DEF-FOREARM-0{i}.{s_}" for i in range(1, 5)]),
                    (("upperleg01", "upperleg02"), [f"DEF-LEG-0{i}.{s_}" for i in range(1, 5)]),
                    (("lowerleg01", "lowerleg02"), [f"DEF-SHIN-0{i}.{s_}" for i in range(1, 5)])):
        for b in mhb: CHAINS[f"{b}.{s_}"] = ch
TRUNK = ["DEF-HIPS", "DEF-SPINE-01", "DEF-SPINE-02", "DEF-CHEST-01", "DEF-CHEST-02"]
for b in ("spine01", "spine02", "spine03", "spine04", "spine05"): CHAINS[b] = TRUNK
for b in ("neck01", "neck02", "neck03"): CHAINS[b] = ["DEF-NECK-01", "DEF-NECK-02"]
RENAME = {"root": "DEF-HIPS"}
for s_ in "LR":
    RENAME.update({f"pelvis.{s_}": "DEF-HIPS", f"clavicle.{s_}": f"DEF-SHOULDER.{s_}", f"shoulder01.{s_}": f"DEF-SHOULDER.{s_}",
                   f"breast.{s_}": "DEF-CHEST-02", f"wrist.{s_}": f"DEF-HAND.{s_}", f"foot.{s_}": f"DEF-FOOT.{s_}"})
    for i, f in enumerate("DCBA"):                      # index, middle, ring, little
        RENAME[f"metacarpal{i + 1}.{s_}"] = f"DEF-PALM-{f}.{s_}"
        for k in (1, 2, 3): RENAME[f"finger{i + 2}-{k}.{s_}"] = f"DEF-FINGER-{f}-0{k}.{s_}"
    for k in (1, 2, 3): RENAME[f"finger1-{k}.{s_}"] = f"DEF-THUMB-0{k}.{s_}"
    for b in mrig.data.bones:
        if b.name.startswith("toe") and b.name.endswith("." + s_):
            RENAME[b.name] = f"DEF-TOES-BIG.{s_}" if b.name.startswith("toe1") else f"DEF-TOES.{s_}"


def along(chain, p):
    """How a point shares a chain: by its place between the chain's segment middles."""
    mids = np.array([np.array(mean(*T(c))) for c in chain])
    a, b = mids[0], mids[-1]; ab = b - a
    t = np.clip((p - a) @ ab / (ab @ ab), 0, 1) * (len(chain) - 1)
    i = min(int(t), len(chain) - 2); f = t - i
    return {chain[i]: 1 - f, chain[i + 1]: f}


def own_weights(p, mine):
    w = np.zeros(len(defs), np.float32)
    for b, x in mine.items():
        if b in HEADISH: continue
        if b in CHAINS:
            for c, y in along(CHAINS[b], p).items(): w[di[c]] += x * y
        else: w[di[RENAME[b]]] += x
    return w / max(w.sum(), 1e-8)


# the full skin: the reference rig's weights at each point, the head's to DEF-HEAD
base = next(o for o in meshes if o.name == "Human")
skin_me, skin_P, skin_h, skin_own, _ = fitted(base, full=True)
Ws = np.zeros((len(skin_P), len(defs)), np.float32)
for i, p in enumerate(skin_P):
    if skin_h[i] < 0.999: Ws[i] = (1 - skin_h[i]) * (own_weights(p, skin_own[i]) if WEIGHTS == "mh" else rig_weights(p))
    Ws[i, HEAD_I] += skin_h[i]
# smoothed where a limb meets the trunk: the reference rig's armour weights end sharply at the shoulder and
# the hip, which skin shows as a ridge once the arm hangs from its T pose; inside these balls each
# vertex's weights relax toward its neighbours' (the rest of the skin holds them in place)
E = np.array([e.vertices[:] for e in skin_me.edges])
deg = np.bincount(E.ravel(), minlength=len(skin_P)).astype(np.float32)
zone = np.zeros(len(skin_P), bool)
for jn, r in (("DEF-ARM-01.L", 0.15), ("DEF-ARM-01.R", 0.15), ("DEF-LEG-01.L", 0.13), ("DEF-LEG-01.R", 0.13)):
    zone |= np.linalg.norm(skin_P - np.array(T(jn)[0]), axis=1) < r
zone &= skin_h < 0.5
for _ in range(SMOOTH_ITERS):
    nb = np.zeros_like(Ws)
    np.add.at(nb, E[:, 0], Ws[E[:, 1]]); np.add.at(nb, E[:, 1], Ws[E[:, 0]])
    nb /= np.maximum(deg, 1)[:, None]
    Ws[zone] = 0.5 * Ws[zone] + 0.5 * nb[zone]
print("SKIN", len(skin_P), "verts;", int(zone.sum()), "smoothed")
skin = Surface(skin_P, triangles(skin_me), Ws)

# ---- the face keys the clips animate (the reference rig body's: blink, smile, brows_up, frown, jaw_open),
# made from MakeHuman's expression units for this person's ancestry mix, on the raw body (helpers and
# all, so the teeth and tongue helpers open with the jaw); every other mesh takes the change of its
# nearest body vertex (lashes close with the lids), eyes and hair excepted
from bl_ext.user_default.mpfb.services import TargetService       # noqa: E402
BLINK = float(os.environ.get("BLINK", "0.4"))   # the closure unit at full drops the lid past the eye
FACE = {"blink": {"eye-left-closure": BLINK, "eye-right-closure": BLINK},
        "smile": {"mouth-corner-puller": 1.0, "mouth-upward-retraction": 0.3, "eye-left-slit": 0.25, "eye-right-slit": 0.25},
        "brows_up": {"eyebrows-left-up": 1.0, "eyebrows-right-up": 1.0, "eyebrows-left-inner-up": 0.5, "eyebrows-right-inner-up": 0.5},
        "frown": {"eyebrows-left-down": 1.0, "eyebrows-right-down": 1.0, "mouth-depression": 0.6},
        "jaw_open": {"mouth-open": 0.45}}                 # the unit at full opens past a human jaw
UNITS = os.path.expanduser("~/.config/blender/5.1/extensions/user_default/mpfb/data/targets/expression/units")
race = json.load(open(os.path.join(HERE, "cast.json")))[cid]["phenotype"]["race"]


def raw(o):
    keep = {m.name: m.show_viewport for m in o.modifiers}
    for m in o.modifiers: m.show_viewport = False
    dg = bpy.context.evaluated_depsgraph_get()
    ev = o.evaluated_get(dg); P = np.array([v.co for v in ev.to_mesh().vertices])
    ev.to_mesh_clear()
    for m in o.modifiers: m.show_viewport = keep[m.name]
    return P @ np.array(o.matrix_world.to_3x3()).T + np.array(o.matrix_world.translation)


R0 = raw(base)
keys = {}
for unit in sorted({u for f in FACE.values() for u in f}):
    for rc, w in race.items():
        if w > 0: keys[(unit, rc)] = TargetService.load_target(base, os.path.join(UNITS, rc, unit + ".target.gz"), weight=0.0, name=f"fx-{rc}-{unit}")
D = {}
for name, units in FACE.items():
    for (unit, rc), k_ in keys.items(): k_.value = units.get(unit, 0.0) * race[rc] / sum(race.values())
    D[name] = raw(base) - R0
    for k_ in keys.values(): k_.value = 0.0
from mathutils.kdtree import KDTree                  # noqa: E402
gb = base.vertex_groups["body"].index                # the skin; the rest are MakeHuman's helpers
on_skin = np.array([any(g.group == gb and g.weight > 0.5 for g in v.groups) for v in base.data.vertices])


def tree(sel):
    t = KDTree(int(sel.sum()))
    for i in np.flatnonzero(sel): t.insert(R0[i], int(i))
    t.balance(); return t


kd_skin, kd_helper = tree(on_skin), tree(~on_skin)  # lashes and brows ride the skin; teeth and tongue the mouth helpers
NO_FACE = ("high-poly", "afro", "ponytail", "short0", "bob0", "braid", "long0")
print("FACE", {n: round(float(np.abs(d).max()), 4) for n, d in D.items()})


made = []
for o in meshes:
    me, P, h, _, pre = fitted(o)
    n = len(P)
    W = np.zeros((n, len(defs)), np.float32)
    for i in range(n):
        if h[i] < 0.999: W[i] = (1 - h[i]) * skin(P[i])
        W[i, HEAD_I] += h[i]
    # four strongest influences (what glTF carries), renormalised
    keep = np.argsort(-W, 1)[:, :4]
    W4 = np.take_along_axis(W, keep, 1); W4 /= np.maximum(W4.sum(1, keepdims=True), 1e-8)
    new = bpy.data.objects.new(o.name.replace("Human.", f"{cid}.").replace("Human", f"{cid}.body"), me)
    bpy.context.scene.collection.objects.link(new)
    new.vertex_groups.clear()                          # MakeHuman's own weights came along with the mesh
    for d in defs: new.vertex_groups.new(name=d)
    for i in range(n):
        for j in range(4):
            if W4[i, j] > 1e-4: new.vertex_groups[int(keep[i, j])].add([i], float(W4[i, j]), "REPLACE")
    if not any(t in o.name for t in NO_FACE) and (h > 0.5).any():
        kd = kd_helper if any(t in o.name for t in ("teeth", "tongue")) else kd_skin
        near = [kd.find(Vector(p)) for p in pre]
        idx = np.array([j for _, j, _ in near]); ok = np.array([d < 0.02 for _, _, d in near]) & (h > 0.5)
        new.shape_key_add(name="Basis")
        for name, d in D.items():                       # the head's own scale carries the change over
            dd = d[idx]
            if name == "blink" and "eyelash" in o.name:  # lashes fold onto the closed lid, not hang off it
                dd = dd - 0.7 * (pre - R0[idx])
            sk = new.shape_key_add(name=name, from_mix=False)
            sk.data.foreach_set("co", (P + k * dd * ok[:, None]).astype(np.float32).ravel())
            sk.value = 0.0                              # a new key starts at full
    new.parent = rig
    mod = new.modifiers.new("Armature", "ARMATURE"); mod.object = rig
    made.append(new)
    print("BOUND", new.name, n, "verts; head-weighted", int((h > 0.5).sum()))

# ---- MakeHuman's own rig goes; the reference rig body stays, hidden (the compiler and checker read it) ----
for o in [body] + [c for c in rig.children if c.type == "MESH" and c not in made]:
    o.hide_render = True                              # still evaluated: the checker reads its soles
gone = {o.name: o for o in mh}
for o in gone.values():
    bpy.data.objects.remove(o, do_unlink=True)
# where the eyes are at rest (the page's gaze finds the eye vertex nearest each)
eyes = next(o for o in made if o.name.endswith("high-poly"))
E = np.array([v.co for v in eyes.data.vertices])
fit = {"k": k, "eyes": {"L": E[E[:, 0] > 0].mean(0).round(4).tolist(), "R": E[E[:, 0] < 0].mean(0).round(4).tolist()}}
json.dump(fit, open(os.path.join(HERE, f"{OUT}_fit.json"), "w"))
print("EYES", fit["eyes"])
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(HERE, f"{OUT}.blend"))
print("SAVED", OUT)
