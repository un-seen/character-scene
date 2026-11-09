"""What a body pose is, in the words people use for it: per body part, a category read from the
posed bones (world matrices), for the free-text pose evaluation (pose_eval.py) and, later, as the
fidelity score of the pose solver. NumPy only, so the Blender checker and a NumPy one read alike.

Categories (the character's own left and right; its frame is the root's: +x its left, -y ahead):
  arm.S      where the hand is from the shoulder: up, down, forward, back, out (its own side),
             across (the other side); folded when the hand is within 0.25 m of the shoulder
  elbow.S    straight (< 35 deg), bent (35-115), tight (> 115)
  hand.S     the landmark on the skin the palm is at (posecat.json, within HAND_AT m), the other
             arm (upper_arm.O, elbow.O), above_head, or free
  level.S    the hand's height: low (below the hips), waist, chest, head, above (the head's top)
  hand_high  yes when either hand is at the head's height or above
  hands      together (palms within 0.15 m) or apart
  head.turn  left / right (> 20 deg), ahead;  head.pitch  down / up (> 15 deg), level;
  head.tilt  left / right (> 12 deg), level  (relative to the chest)
  trunk.bend forward (> 20 deg) / back (> 10 deg) / upright;  trunk.side  left / right (> 12 deg), none;
  trunk.twist left / right (> 20 deg), none   (the chest relative to the root)
  shoulders  up (both raised > 2.5 cm from rest, relative to the chest), level
  posture    stand, crouch, kneel, sit, lie
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(__file__)
DATA = json.load(open(os.path.join(HERE, "data", "posecat.json")))
LM = DATA["landmarks"]
REST = {k: np.array(v) for k, v in DATA["rest"].items()}
HAND_AT = 0.14
OTHER = {"L": "R", "R": "L"}


def _p(M, local): return (M @ np.append(np.asarray(local, float), 1.0))[:3]


def _rot(M): return M[:3, :3] / np.linalg.norm(M[:3, :3], axis=0)


def _since_rest(world, name):
    """The bone's turn since rest, in the world (a 3x3)."""
    return _rot(world(name)) @ _rot(REST[name]).T


def _deg(a): return math.degrees(a)


def read(world):
    """Body measures from world(bone) -> 4x4 world matrix of the posed bone."""
    root = _since_rest(world, "ROOT")
    to_body = root.T                                     # world -> the body's frame (+x left, -y ahead, +z up)
    head = lambda n: world(n)[:3, 3]
    m = {"to_body": to_body}
    m["lm"] = {k: _p(world(v["bone"]), v["local"]) for k, v in LM.items()}
    for s in "LR":
        m[f"shoulder.{s}"] = head(f"DEF-ARM-01.{s}")
        m[f"elbow.{s}"] = head(f"DEF-FOREARM-01.{s}")
        m[f"wrist.{s}"] = head(f"DEF-HAND.{s}")
        # the palm's middle: halfway from the middle metacarpals' bases to their knuckles
        m[f"palm.{s}"] = sum(head(n) for n in (f"DEF-PALM-B.{s}", f"DEF-PALM-C.{s}", f"DEF-FINGER-B-01.{s}", f"DEF-FINGER-C-01.{s}")) / 4
        m[f"knee.{s}"] = head(f"DEF-SHIN-01.{s}")
    m["hips"] = head("DEF-HIPS")
    m["chest_rot"] = _since_rest(world, "DEF-CHEST-02")
    m["head_rot"] = _since_rest(world, "DEF-HEAD")
    m["root_rot"] = root
    return m


def _ang(u, v):
    c = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
    return _deg(math.acos(max(-1.0, min(1.0, c))))


def categories(world):
    m = read(world); B = m["to_body"]; lm = m["lm"]
    out = {}
    for s in "LR":
        o = OTHER[s]; side = 1.0 if s == "L" else -1.0
        sh, el, wr, pa = m[f"shoulder.{s}"], m[f"elbow.{s}"], m[f"wrist.{s}"], m[f"palm.{s}"]
        v = B @ (pa - sh)
        if np.linalg.norm(v) < 0.25: out[f"arm.{s}"] = "folded"
        else:
            u = v / np.linalg.norm(v)
            dirs = {"up": u[2], "down": -u[2], "forward": -u[1], "back": u[1], "out": side * u[0], "across": -side * u[0]}
            out[f"arm.{s}"] = max(dirs, key=dirs.get)
        bend = _ang(el - sh, wr - el)
        out[f"elbow.{s}"] = "straight" if bend < 35 else "bent" if bend < 115 else "tight"
        cands = {k: p for k, p in lm.items()}
        cands[f"upper_arm.{o}"] = (m[f"shoulder.{o}"] + m[f"elbow.{o}"]) / 2
        cands[f"elbow.{o}"] = m[f"elbow.{o}"]
        near = min(cands, key=lambda k: np.linalg.norm(cands[k] - pa))
        z = (B @ pa)[2]; ztop = (B @ lm["head_top"])[2]
        if np.linalg.norm(cands[near] - pa) < HAND_AT: out[f"hand.{s}"] = near
        elif z > ztop: out[f"hand.{s}"] = "above_head"
        else: out[f"hand.{s}"] = "free"
        zh, zc, zm = (B @ m["hips"])[2], (B @ lm["chest"])[2], (B @ lm["mouth"])[2]
        out[f"level.{s}"] = ("above" if z > ztop else "head" if z > zm - 0.05 else "chest" if z > zc - 0.12
                             else "waist" if z > zh - 0.08 else "low")
    out["hand_high"] = "yes" if out["level.L"] in ("head", "above") or out["level.R"] in ("head", "above") else "no"
    out["hands"] = "together" if np.linalg.norm(m["palm.L"] - m["palm.R"]) < 0.15 else "apart"
    # the head relative to the chest: where its face points (rest: -y) and its top (rest: +z)
    rel = m["chest_rot"].T @ m["head_rot"]
    f, up = rel @ np.array([0, -1.0, 0]), rel @ np.array([0, 0, 1.0])
    turn = _deg(math.atan2(f[0], -f[1])); pitch = _deg(math.asin(max(-1, min(1, -f[2])))); tilt = _deg(math.atan2(up[0], up[2]))
    out["head.turn"] = "left" if turn > 20 else "right" if turn < -20 else "ahead"
    out["head.pitch"] = "down" if pitch > 15 else "up" if pitch < -15 else "level"
    out["head.tilt"] = "left" if tilt > 12 else "right" if tilt < -12 else "level"
    # the chest relative to the root
    c = m["root_rot"].T @ m["chest_rot"]
    cu, cf = c @ np.array([0, 0, 1.0]), c @ np.array([0, -1.0, 0])
    fwd = _deg(math.atan2(-cu[1], cu[2])); sidebend = _deg(math.atan2(cu[0], cu[2])); twist = _deg(math.atan2(cf[0], -cf[1]))
    out["trunk.bend"] = "forward" if fwd > 20 else "back" if fwd < -10 else "upright"
    out["trunk.side"] = "left" if sidebend > 12 else "right" if sidebend < -12 else "none"
    out["trunk.twist"] = "left" if twist > 20 else "right" if twist < -20 else "none"
    # shoulders: each shoulder joint's height over the chest, against its rest height
    rise = []
    for s in "LR":
        rest_h = (REST[f"DEF-ARM-01.{s}"][:3, 3] - REST["DEF-CHEST-02"][:3, 3])[2]
        now = (m["chest_rot"].T @ (m[f"shoulder.{s}"] - world("DEF-CHEST-02")[:3, 3]))[2]
        rise.append(now - rest_h)
    out["shoulders"] = "up" if min(rise) > 0.025 else "level"
    # posture, from the world: the trunk's lie, the hips' and knees' heights
    up_w = m["chest_rot"] @ np.array([0, 0, 1.0])
    hz, kz = m["hips"][2], min(m["knee.L"][2], m["knee.R"][2])
    # knees on the floor are a kneel whatever the hips' height; a seat on the floor, a sit; a flat trunk
    # with the hips up is a bend from standing, not a lie (2026-11-06)
    out["posture"] = ("lie" if up_w[2] < 0.5 and hz < 0.5 else "kneel" if kz < 0.15 and hz > 0.25 else "sit" if hz < 0.4
                      else "crouch" if hz < 0.75 else "stand")
    return out


def score(cats, expect):
    """{key: hit} for the expected keys (each a list of allowed categories)."""
    return {k: cats.get(k) in v for k, v in expect.items()}
