"""A static pose checked without Blender (stage 0 of reports/Constrained LLM pose writing.md): the
rig posed by rigctl/fk.py, then what checker.py reads of one standing frame: the joints' angles and
their breaks (anatomy.py's swing-twist and LIMITS), the centre of mass over the soles (the mean of
every 7th skinned vertex, as checker.py takes it), and what the body does in words (posecat.py).
A pose takes a few milliseconds, so many can be tried and solved for.

  check(basis) -> {"joints", "violations", "categories", "com", "balanced", "mats"}
"""
import math

import numpy as np

from . import anatomy, fk, posecat


def qmul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def qinv(q): return np.array([q[0], -q[1], -q[2], -q[3]])


def swing_twist(q):
    """anatomy.swing_twist on a (w, x, y, z) array: (swing about X, swing about Z, twist about Y) deg."""
    t = np.array([q[0], 0.0, q[2], 0.0]); n = np.linalg.norm(t)
    t = np.array([1.0, 0, 0, 0]) if n < 1e-9 else t / n
    sw = qmul(q, qinv(t))
    if sw[0] < 0: sw = -sw
    tw = (2 * math.degrees(math.atan2(t[2], t[0])) + 180) % 360 - 180
    a = 2 * math.acos(max(-1.0, min(1.0, sw[0])))
    s = math.sin(a / 2)
    ax = sw[1:] / s if s > 1e-9 else np.array([1.0, 0, 0])
    return math.degrees(a * ax[0]), math.degrees(a * ax[2]), tw


def _rot(M): return M[:3, :3] / np.linalg.norm(M[:3, :3], axis=0)


_REST_Q = None


def rest_q():
    global _REST_Q
    if _REST_Q is None:
        by = fk.data()["by"]
        _REST_Q = {j: fk.m3_to_quat(_rot(by[p]["rest"]).T @ _rot(by[c]["rest"])) for j, (p, c) in anatomy.joints().items()}
    return _REST_Q


def joints(M, only=None):
    out = {}
    for j, (p, c) in anatomy.joints().items():
        if only is not None and j not in only: continue
        cur = fk.m3_to_quat(_rot(M[p]).T @ _rot(M[c]))
        out[j] = tuple(round(x, 1) for x in swing_twist(qmul(qinv(rest_q()[j]), cur)))
    return out


_SKIN = {}


def skinned(M, idx):
    """Linear blend skinning (the body's armature modifier) of the checker's mesh points `idx`."""
    key = tuple(idx)
    if key not in _SKIN:
        sk = fk.data()["skin"]["points"]
        names = sorted({n for i in idx for n in sk[str(i)]["w"]})
        W = np.zeros((len(idx), len(names))); CO = np.ones((len(idx), 4))
        for r, i in enumerate(idx):
            p = sk[str(i)]; CO[r, :3] = p["co"]
            for n, w in p["w"].items(): W[r, names.index(n)] = w
        W /= np.where(W.sum(axis=1, keepdims=True) > 0, W.sum(axis=1, keepdims=True), 1)
        _SKIN[key] = (names, W, CO)
    names, W, CO = _SKIN[key]
    by = fk.data()["by"]
    T = np.stack([M[n] @ by[n]["irest"] for n in names])          # (bones, 4, 4)
    moved = np.einsum("bij,pj->pbi", T, CO)[:, :, :3]                # (points, bones, 3)
    return np.einsum("pb,pbi->pi", W, moved)


def _hull(pts):
    pts = sorted(set(map(tuple, pts)))
    if len(pts) < 3: return pts
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
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
    for i in range(len(h)):
        a, b = h[i], h[(i + 1) % len(h)]
        if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) < 0: return False
    return True


def check(basis, slack=None):
    M = fk.pose(basis)
    W = fk.data()["world"]
    world = lambda n: W @ M[n]
    J = joints(M)
    v = anatomy.violations(J, anatomy_slack() if slack is None else slack)
    sk = fk.data()["skin"]
    com = skinned(M, sk["sample"]).mean(axis=0)
    soles = skinned(M, sk["sole"]["L"] + sk["sole"]["R"])
    on = soles[soles[:, 2] < 0.02]
    bal = bool(len(on) >= 3 and _in_hull((com[0], com[1]), _hull([(p[0], p[1]) for p in on])))
    return {"joints": J, "violations": v, "categories": posecat.categories(world), "com": com, "balanced": bal, "mats": M}


def anatomy_slack():
    return 5.0      # checker.ANATOMY_SLACK (checker.py needs Blender to import)
