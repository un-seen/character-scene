"""The rig in NumPy: the bones rigctl/posecat.py and rigctl/anatomy.py read, posed from control
values the way Blender poses them (data/fk.json, written by tools/export_fk.py), so that a pose can be
checked or solved without Blender. Blender's evaluation, per bone in dependency order: its pose is
its parent's pose, its rest offset from the parent, then its own location, rotation and scale; then
its constraints, each in world space unless noted, blended with the result before it by influence.

  pose(basis) -> {bone: 4x4 armature-space matrix}
      basis: {bone: {"loc": [3], "quat": [w, x, y, z] or "euler": [3] (radians, the bone's mode), "scale": [3]}}

Checked against Blender with tools/fk_check.py (random control values; positions and rotations).
"""
import json
import math
import os

import numpy as np

HERE = os.path.dirname(__file__)
_D = None


def data():
    global _D
    if _D is None:
        d = json.load(open(os.path.join(HERE, "data", "fk.json")))
        for b in d["bones"]: b["rest"] = np.array(b["rest"])
        d["world"] = np.array(d["world"]); d["iworld"] = np.linalg.inv(d["world"])
        d["by"] = {b["name"]: b for b in d["bones"]}
        for b in d["bones"]:                             # the rest offset from the parent, and the rest's inverse
            p = d["by"].get(b["parent"]) if b["parent"] else None
            b["offs"] = (np.linalg.inv(p["rest"]) @ b["rest"]) if p is not None else b["rest"]
            b["irest"] = np.linalg.inv(b["rest"])
        _D = d
    return _D


# -- small rotation helpers --------------------------------------------------------------------------
def quat_to_m3(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def m3_to_quat(m):
    t = np.trace(m)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2; return np.array([0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s])
    i = int(np.argmax(np.diag(m)))
    if i == 0:
        s = math.sqrt(1 + m[0, 0] - m[1, 1] - m[2, 2]) * 2; return np.array([(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s])
    if i == 1:
        s = math.sqrt(1 + m[1, 1] - m[0, 0] - m[2, 2]) * 2; return np.array([(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s])
    s = math.sqrt(1 + m[2, 2] - m[0, 0] - m[1, 1]) * 2; return np.array([(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s])


def _axis(i, a):
    c, s = math.cos(a), math.sin(a)
    if i == 0: return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if i == 1: return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def euler_to_m3(e, order="XYZ"):
    """Blender's Euler: the axes applied in `order` (XYZ: X first), i.e. R = Rz Ry Rx for XYZ."""
    R = np.eye(3)
    for ch in order: R = _axis("XYZ".index(ch), e["XYZ".index(ch)]) @ R
    return R


def m3_to_euler(m, order="XYZ"):
    """Euler angles (radians, by axis X, Y, Z) of a rotation for an order, one of Blender's solutions."""
    i, j, k = ("XYZ".index(order[0]), "XYZ".index(order[1]), "XYZ".index(order[2]))
    par = 1.0 if (i, j, k) in ((0, 1, 2), (1, 2, 0), (2, 0, 1)) else -1.0
    # R = R_k R_j R_i ; the general extraction (Shoemake), for Tait-Bryan orders
    cy = math.hypot(m[i, i], m[j, i])
    if cy > 1e-9:
        a = math.atan2(m[k, j], m[k, k]) ; b = math.atan2(-m[k, i], cy); c = math.atan2(m[j, i], m[i, i])
    else:
        a = math.atan2(-m[j, k], m[j, j]); b = math.atan2(-m[k, i], cy); c = 0.0
    if par < 0: a, b, c = -a, -b, -c
    e = [0.0, 0.0, 0.0]; e[i], e[j], e[k] = a, b, c
    return np.array(e)


def decompose(M):
    loc = M[:3, 3].copy(); S = np.linalg.norm(M[:3, :3], axis=0)
    R = M[:3, :3] / np.where(S == 0, 1, S)
    if np.linalg.det(R) < 0: S[0] = -S[0]; R[:, 0] = -R[:, 0]
    return loc, R, S


def compose(loc, R, S):
    M = np.eye(4); M[:3, :3] = R * S; M[:3, 3] = loc
    return M


def slerp(q0, q1, t):
    if np.dot(q0, q1) < 0: q1 = -q1
    d = min(1.0, float(np.dot(q0, q1)))
    if d > 0.9995: q = q0 + t * (q1 - q0); return q / np.linalg.norm(q)
    th = math.acos(d); return (math.sin((1 - t) * th) * q0 + math.sin(t * th) * q1) / math.sin(th)


def blend(A, B, t):
    """Blender's blend_m4_m4m4: location and scale lerped, rotation slerped."""
    if t >= 1.0: return B.copy()
    if t <= 0.0: return A.copy()
    la, Ra, Sa = decompose(A); lb, Rb, Sb = decompose(B)
    q = slerp(m3_to_quat(Ra), m3_to_quat(Rb), t)
    return compose(la + (lb - la) * t, quat_to_m3(q), Sa + (Sb - Sa) * t)


# -- the evaluation ----------------------------------------------------------------------------------
class Pose:
    def __init__(self, basis=None):
        self.d = data(); self.basis = basis or {}; self.M = {}

    def chan(self, b):
        v = self.basis.get(b["name"], {})
        loc = np.array(v.get("loc", [0.0, 0.0, 0.0])); S = np.array(v.get("scale", [1.0, 1.0, 1.0]))
        if "quat" in v: R = quat_to_m3(np.array(v["quat"], float))
        elif "euler" in v: R = euler_to_m3(v["euler"], b["mode"] if b["mode"] in ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX") else "XYZ")
        else: R = np.eye(3)
        return compose(loc, R, S)

    def offs(self, b):
        return b["offs"]

    def parent_mat(self, b):
        return self.M[b["parent"]] if b["parent"] else np.eye(4)

    def target(self, c):
        T = self.M[c["target"]]; t = self.d["by"][c["target"]]
        ht = c.get("head_tail", 0.0)
        if ht:
            T = T.copy(); T[:3, 3] = T[:3, 3] + T[:3, :3] @ np.array([0, t["length"], 0]) * ht
        return T

    def local_of(self, name, M):
        b = self.d["by"][name]
        return np.linalg.inv(self.parent_mat(b) @ self.offs(b)) @ M

    def from_local(self, name, L):
        b = self.d["by"][name]
        return self.parent_mat(b) @ self.offs(b) @ L

    def evaluate(self):
        W, iW = self.d["world"], self.d["iworld"]
        for b in self.d["bones"]: self.eval_bone(b, W, iW)
        return self.M

    def eval_bone(self, b, W, iW):
        n = b["name"]
        M = self.parent_mat(b) @ self.offs(b) @ self.chan(b)
        for c in b["constraints"]:
            if not c["enabled"] or c["influence"] <= 0: continue
            if c["type"] == "IK":
                self.ik(b, c, M); M = self.M[n]; continue
            old = M
            if c["ospace"] == "LOCAL" or c["tspace"] == "LOCAL":
                M = self.local_constraint(n, c, M)
            else:
                M = iW @ self.world_constraint(n, c, W @ M, W @ self.target(c) if c["target"] else None) @ W
            M = blend(old, M, c["influence"])
            self.M[n] = M
        self.M[n] = M

    def world_constraint(self, n, c, O, T):
        t = c["type"]
        if t == "COPY_TRANSFORMS": return T.copy()
        if t == "COPY_LOCATION":
            O = O.copy()
            for i in range(3):
                if c["use"][i]: O[i, 3] = T[i, 3] + (O[i, 3] if c["offset"] else 0.0)
            return O
        if t == "COPY_SCALE":
            lo, Ro, So = decompose(O); lt, Rt, St = decompose(T)
            S = np.array([St[i] ** c["power"] if c["use"][i] else So[i] for i in range(3)])
            return compose(lo, Ro, S)
        if t == "COPY_ROTATION":
            lo, Ro, So = decompose(O); lt, Rt, St = decompose(T)
            if all(c["use"]): R = Rt
            else:
                order = c["order"] if c["order"] != "AUTO" else (self.d["by"][n]["mode"] if self.d["by"][n]["mode"] in ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX") else "XYZ")
                eo, et = m3_to_euler(Ro, order), m3_to_euler(Rt, order)
                et = [-x if inv else x for x, inv in zip(et, c.get("invert", [False] * 3))]
                R = euler_to_m3([et[i] if c["use"][i] else eo[i] for i in range(3)], order)
            return compose(lo, R, So)
        if t == "DAMPED_TRACK":
            v = T[:3, 3] - O[:3, 3]; nv = np.linalg.norm(v)
            if nv < 1e-9: return O
            v = v / nv; axis = {"TRACK_X": 0, "TRACK_Y": 1, "TRACK_Z": 2, "TRACK_NEGATIVE_X": 0, "TRACK_NEGATIVE_Y": 1, "TRACK_NEGATIVE_Z": 2}[c["axis"]]
            ob = O[:3, axis] * (-1 if "NEGATIVE" in c["axis"] else 1); ob = ob / np.linalg.norm(ob)
            r = np.cross(ob, v); s = np.linalg.norm(r); co = float(np.clip(np.dot(ob, v), -1, 1))
            if s < 1e-7:
                if co > 0: return O
                r = np.cross(ob, [1.0, 0, 0]) if abs(ob[0]) < 0.9 else np.cross(ob, [0, 1.0, 0]); r /= np.linalg.norm(r); s = 0.0
            else: r = r / s
            ang = math.atan2(s, co)
            K = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
            Rm = np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K
            out = O.copy(); out[:3, :3] = Rm @ O[:3, :3]
            return out
        if t == "STRETCH_TO":
            S = np.linalg.norm(O[:3, :3], axis=0)
            xx, zz = O[:3, 0] / S[0], O[:3, 2] / S[2]
            v = T[:3, 3] - O[:3, 3]; dist = np.linalg.norm(v); v = v / dist
            dist /= S[1]
            org = c["rest_length"] or dist
            sy = dist / org; bulge = (org / dist) ** c["bulge"]
            vm = c["volume"]
            sx, sz = ((math.sqrt(bulge), math.sqrt(bulge)) if vm == "VOLUME_XZX" else (bulge, 1.0) if vm == "VOLUME_X"
                      else (1.0, bulge) if vm == "VOLUME_Z" else (1.0, 1.0))
            if c["keep_axis"] == "PLANE_X":
                z = np.cross(xx, v); z /= np.linalg.norm(z); x = np.cross(v, z); x /= np.linalg.norm(x)
            else:
                x = np.cross(v, zz); x /= np.linalg.norm(x); z = np.cross(x, v); z /= np.linalg.norm(z)
            out = O.copy(); out[:3, 0], out[:3, 1], out[:3, 2] = x * S[0] * sx, v * S[1] * sy, z * S[2] * sz
            return out
        if t == "LIMIT_ROTATION": return O
        raise NotImplementedError(t)

    def local_constraint(self, n, c, M):
        t = c["type"]; b = self.d["by"][n]
        L = self.local_of(n, M)
        if t == "LIMIT_ROTATION":
            lo, R, S = decompose(L)
            order = c["order"] if c["order"] != "AUTO" else "XYZ"
            e = m3_to_euler(R, order)
            for i, (use, mn, mx) in enumerate(c["lim"]):
                if use: e[i] = min(max(e[i], mn), mx)
            return self.from_local(n, compose(lo, euler_to_m3(e, order), S))
        Lt = self.local_of(c["target"], self.M[c["target"]])
        if t == "COPY_TRANSFORMS": return self.from_local(n, Lt)
        if t == "COPY_ROTATION":
            lo, Ro, So = decompose(L); lt, Rt, St = decompose(Lt)
            if all(c["use"]): R = Rt
            else:
                order = c["order"] if c["order"] != "AUTO" else (b["mode"] if b["mode"] in ("XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX") else "XYZ")
                eo, et = m3_to_euler(Ro, order), m3_to_euler(Rt, order)
                et = [-x if inv else x for x, inv in zip(et, c.get("invert", [False] * 3))]
                R = euler_to_m3([et[i] if c["use"][i] else eo[i] for i in range(3)], order)
            return self.from_local(n, compose(lo, R, So))
        raise NotImplementedError(("local", t))

    def ik(self, b, c, M):
        """A two-bone chain (the owner and its parent) reaching for the target with the owner's tail,
        as Blender's solver leaves it (measured with tools/fk_check.py, limbs that keep their
        length): the owner turns only about its own Z from its rest relation to the root (its
        lock_ik), by the angle that puts the tip at the target's distance; the chain is turned onto
        the root-to-target line, the root's X axis (perpendicular to that line) is the pole's
        direction turned by pole_angle, and of the two bends the one with the knee on the pole's
        side is taken. Past its reach the chain is held straight."""
        n = b["name"]; self.M[n] = M
        p = self.d["by"][b["parent"]]
        A = self.M[p["name"]]; Bm = M
        S1, S2 = np.linalg.norm(A[:3, :3], axis=0), np.linalg.norm(Bm[:3, :3], axis=0)
        l1, l2 = p["length"] * S1[1], b["length"] * S2[1]
        H = A[:3, 3]; Tgt = self.target(c)[:3, 3]
        d = Tgt - H; D = np.linalg.norm(d); u = d / D
        P = self.M[c["pole"]][:3, 3]
        w = (P - H) - np.dot(P - H, u) * u; w /= np.linalg.norm(w)
        ang = c["pole_angle"]
        K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
        xr = (np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K) @ w
        R1r, R2r = p["rest"][:3, :3], b["rest"][:3, :3]
        R1r = R1r / np.linalg.norm(R1r, axis=0); R2r = R2r / np.linalg.norm(R2r, axis=0)
        Rrel = R1r.T @ R2r
        tip = lambda th: np.array([0, l1, 0]) + Rrel @ _axis(2, th) @ np.array([0, l2, 0])
        # |tip|^2 = l1^2 + l2^2 + 2 l1 l2 (Rrel Rz(th) y)_y, and (Rrel Rz(th) y)_y = A cos(th + phi)
        A_, phi = math.hypot(Rrel[1, 0], Rrel[1, 1]), math.atan2(Rrel[1, 0], Rrel[1, 1])
        cth = (D * D - l1 * l1 - l2 * l2) / (2 * l1 * l2 * A_)
        if cth >= 1.0: roots = [-phi]                     # out of reach: the straightest
        else:
            g = math.acos(max(-1.0, cth)); roots = [-phi + g, -phi - g]
        def rot(axis, s, co):
            Kv = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
            return np.eye(3) + s * Kv + (1 - co) * Kv @ Kv
        def onto(a_, b_):
            v = np.cross(a_, b_); s = np.linalg.norm(v); co = float(np.clip(np.dot(a_, b_), -1, 1))
            return np.eye(3) if s < 1e-9 else rot(v / s, s, co)
        # the straight chain (rest relation) pointed at the target, the root's X (perpendicular to u) along xr
        t0 = tip(0.0); Ra = onto(t0 / np.linalg.norm(t0), u)
        x0 = Ra @ np.array([1.0, 0, 0]); x0 = x0 - np.dot(x0, u) * u; x0 /= np.linalg.norm(x0)
        phi = math.atan2(np.dot(np.cross(x0, xr), u), np.dot(x0, xr))
        R0 = rot(u, math.sin(phi), math.cos(phi)) @ Ra
        hinge = R0 @ Rrel @ np.array([0, 0, 1.0])               # the knee's axis, in the world
        best = None
        for th in roots:
            # then bent: the root turns about the knee's axis until the tip is back on the line
            tl = R0 @ tip(th); tl /= np.linalg.norm(tl)
            alpha = math.atan2(np.dot(np.cross(tl - np.dot(tl, hinge) * hinge, u - np.dot(u, hinge) * hinge), hinge),
                               np.dot(tl - np.dot(tl, hinge) * hinge, u - np.dot(u, hinge) * hinge))
            F1 = rot(hinge, math.sin(alpha), math.cos(alpha)) @ R0
            knee = H + F1 @ np.array([0, l1, 0])
            side = np.dot(knee - H - np.dot(knee - H, u) * u, w)
            if best is None or side > best[0]: best = (side, F1, th, knee)
        _, F1, th, knee = best
        self.M[p["name"]] = compose(H, F1, S1)
        self.M[n] = compose(knee, F1 @ Rrel @ _axis(2, th), S2)


def pose(basis):
    return Pose(basis).evaluate()


class Rig(Pose):
    """A pose kept between calls: set(changes) re-evaluates only the bones that depend on the
    changed ones (an arm's control moves about 40 of the 302), for the solver's many tries."""

    def __init__(self, basis=None):
        super().__init__(dict(basis or {}))
        self.evaluate()

    def dependents(self, names):
        d = self.d
        if "rdeps" not in d:
            r = {b["name"]: set() for b in d["bones"]}
            for b in d["bones"]:
                for m in ([b["parent"]] if b["parent"] else []) + [c["target"] for c in b["constraints"] if c.get("target")] \
                         + [c["pole"] for c in b["constraints"] if c["type"] == "IK" and c.get("pole")]:
                    if m in r: r[m].add(b["name"])
            # an IK owner moves its chain's root: whatever reads the root depends on the owner
            for b in d["bones"]:
                for c in b["constraints"]:
                    if c["type"] == "IK" and b["parent"] in r:
                        r[b["name"]] |= r[b["parent"]] | {b["parent"]}
            d["rdeps"] = r; d["dcache"] = {}
        key = frozenset(names)
        if key not in d["dcache"]:
            out, stack = set(names), list(names)
            while stack:
                for m in d["rdeps"].get(stack.pop(), ()):
                    if m not in out: out.add(m); stack.append(m)
            d["dcache"][key] = [b for b in d["bones"] if b["name"] in out]
        return d["dcache"][key]

    def trial(self, changes):
        """set(changes), keeping what it replaces: undo(token) puts it back without evaluating."""
        token = ({n: self.basis.get(n) for n in changes}, {b["name"]: self.M[b["name"]] for b in self.dependents(changes.keys())})
        self.set(changes)
        return token

    def undo(self, token):
        basis, mats = token
        for n, v in basis.items():
            if v is None: self.basis.pop(n, None)
            else: self.basis[n] = v
        self.M.update(mats)

    def set(self, changes):
        """changes: {bone: basis entry}; returns the matrices (all bones)."""
        self.basis.update(changes)
        W, iW = self.d["world"], self.d["iworld"]
        for b in self.dependents(changes.keys()):
            self.eval_bone(b, W, iW)
        return self.M
