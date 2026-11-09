"""A pose beat's hands frame by frame, as the compiler moves them (compiler.Compiler's pose branch:
each control on its own clock, POSE_LAG; pose_curve's ease, with its anticipation; slerp, a hinge's
angles blended), on the fast rig (fk), without Blender; and the beat timed so no hand jumps more than
the checker allows in a frame (2026-11-08).

The checker (checker._smooth) fails a clip whose hand moves over 0.12 m in one frame. Keys the model
times well enough for an average speed can still pass it: a big raise's ease and the arm's lag (the hand
6 frames behind the shoulder) put most of the travel in a few frames. retime() stretches a beat's key
times, in proportion, until this simulation keeps every frame under JUMP_SAFE; the poses do not change.
The constants below are the compiler's (it imports bpy, so they are copied; keep them in step)."""
import math

import numpy as np

from . import fk

# --- copied from compiler.py ----------------------------------------------------------------------
HINGES = ("forearm_fk",)
POSE_LAG = {"head": -2, "neck": -1, "chest": 0, "shoulder": 1, "upper_arm_fk": 2, "forearm_fk": 4, "hand_fk": 6}
ANTIC_MIN_DEG, ANTIC_FULL_DEG, POSE_BACK_S = 30.0, 90.0, 0.45
ANTIC_SLOW, ANTIC_FAST = 60.0, 120.0
ANTIC_CTRLS = ("upper_arm_fk", "forearm_fk", "hand_fk", "chest", "head", "neck", "shoulder")


def pose_curve(u, deg, ctrl, span=1.0):
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
# ---------------------------------------------------------------------------------------------------

JUMP_SAFE = 0.10       # m a frame: the checker's 0.12 less a margin (this copy reads up to 0.015 low: the idle's sway)
HANDS = ("DEF-HAND.L", "DEF-HAND.R")


def _q(rot):
    return np.array(fk.m3_to_quat(fk.euler_to_m3([math.radians(x) for x in rot], "XYZ")), float)


def _rot(q):
    return [math.degrees(x) for x in fk.m3_to_euler(fk.quat_to_m3(list(q)), "XYZ")]


def _qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _qinv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _blend(c, a, b, u, span):
    """Control c's value u of the way (its own clock) from a to b, as the compiler blends it."""
    ra, rb = a.get("rot", [0, 0, 0]), b.get("rot", [0, 0, 0])
    qa, qb = _q(ra), _q(rb)
    if np.dot(qa, qb) < 0: qb = -qb
    d = _qmul(_qinv(qa), qb); ang = 2 * math.acos(min(1.0, abs(d[0])))
    if d[0] < 0: d = -d
    w = pose_curve(u, math.degrees(ang), c, span)
    if c.split(".")[0] in HINGES:
        ea = np.array(ra, float); eb = np.array(_rot(qb), float)
        eb = ea + (eb - ea + 180) % 360 - 180               # the turn's nearest equivalent, as to_euler(compat) picks
        rot = list(ea + (eb - ea) * w)
    elif ang > 1e-6:
        axis = d[1:] / max(np.linalg.norm(d[1:]), 1e-12)
        h = ang * w / 2
        rot = _rot(_qmul(qa, np.array([math.cos(h), *(axis * math.sin(h))])))
    else:
        rot = list(ra)
    out = {"rot": rot}
    la, lb = a.get("loc"), b.get("loc")
    if la is not None or lb is not None:
        la, lb = np.array(la or [0, 0, 0], float), np.array(lb or [0, 0, 0], float)
        out["loc"] = list(la + (lb - la) * w)
    return out


def hands(rest, sets, ats, fps, back):
    """[(frame, {hand: xyz})] through a pose beat: the body from `rest` to each key's controls (`sets`, the
    solver's) at frames `ats`, held, and back to `rest` over `back` frames."""
    from .posesolve import basis_of
    dur = ats[-1] + int(round(0.4 * fps)) + back            # program.pose_frames: last key + POSE_HOLD + the way back
    times = [0.0] + [a / fps for a in ats] + [(dur - back) / fps, dur / fps]
    path = [rest] + list(sets) + [sets[-1], rest]
    ctrls = sorted({c for p in path for c in p})
    out = []
    for f in range(dur + 1):
        tau = f / fps; now = {}
        for c in ctrls:
            tc = min(max(tau - POSE_LAG.get(c.split(".")[0], 0) / fps, 0.0), dur / fps)
            k = max(i for i in range(len(times) - 1) if times[i] <= tc)
            span = times[k + 1] - times[k]
            u = min(1.0, max(0.0, (tc - times[k]) / span)) if span > 0 else 1.0
            a, b = path[k].get(c, rest.get(c, {"rot": [0, 0, 0]})), path[k + 1].get(c, rest.get(c, {"rot": [0, 0, 0]}))
            now[c] = _blend(c, a, b, u, span)
        M = fk.pose(basis_of(now))
        out.append((f, {h: np.array(M[h])[:3, 3] for h in HANDS}))
    return out


def max_jump(rest, sets, ats, fps, back):
    hs = hands(rest, sets, ats, fps, back)
    best = (0.0, 0)
    for (f0, a), (f1, b) in zip(hs, hs[1:]):
        j = max(float(np.linalg.norm(b[h] - a[h])) for h in HANDS)
        if j > best[0]: best = (j, f1)
    return best


def retime(rest, sets, ats, fps, back_of):
    """Key frames, stretched in proportion from `ats` until no hand jumps over JUMP_SAFE a frame (at most
    three tries); back_of(first_key) is the way back's length (program.pose_back)."""
    ats = list(ats)
    for _ in range(3):
        j, _f = max_jump(rest, sets, ats, fps, back_of(ats[0]))
        if j <= JUMP_SAFE: break
        k = j / JUMP_SAFE * 1.04
        ats = [max(a + 1, int(math.ceil(x * k))) if i else int(math.ceil(x * k)) for i, (x, a) in enumerate(zip(ats, [0] + ats))]
        for i in range(1, len(ats)): ats[i] = max(ats[i], ats[i - 1] + 1)
    return ats
