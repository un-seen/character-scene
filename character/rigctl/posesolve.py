"""A pose from a spec in body words (stage 2 of reports/Constrained LLM pose writing.md): the model
says what each part does, in posecat.py's words; this finds rig control values that do it, within
the joints' human ranges, without the arms passing through the body, close to where the body
already is. The text's targets are hard (heavily weighted), the anatomy soft.

A spec, one part a line (any part left out keeps the base pose):

  intent: hands rest on the hips
  arm.L: hand hip.L; elbow bent
  arm.R: dir up; elbow straight; level above
  hands: together
  head: turn left; pitch down; tilt left
  trunk: bend forward; side left; twist left
  shoulders: up
  posture: sit

The words are posecat's categories (arm directions, elbow, hand landmarks, levels, ...); the solved
pose is read back with posecat.categories and must say the same.

  solve(spec_text) -> {"controls": {control: {"rot": [deg XYZ], "loc": [m]}}, "categories", "met", "violations", ...}
"""
import json
import math
import os

import numpy as np

from . import fk, npcheck, posecat, registry as R

HERE = os.path.dirname(__file__)
POSTURES = json.load(open(os.path.join(HERE, "data", "postures.json")))
STAND = {"upper_arm_fk.L": [-80, 0, 0], "upper_arm_fk.R": [-80, 0, 0], "forearm_fk.L": [0, 0, -4], "forearm_fk.R": [0, 0, 4]}
# the compiler's standing body (read off a compiled idle frame, ~/cf/stance_read.py; the fast rig then puts
# hips and ankles where Blender does to the mm): feet under the hips, chest a little out. Without it the
# legs stood at the rig's rest, ankles 21 cm behind the hips, and every lean looked balanced.
STAND_BODY = {"torso": {"rot": [-3.0, 0.0, 0.0], "loc": [0.0, 0.0, -0.014]}, "chest": {"rot": [6.0, 0.0, 0.0]},
              "foot_ik.L": {"rot": [0.0, 0.0, 3.0], "loc": [0.0, -0.2118, 0.0079]},
              "foot_ik.R": {"rot": [0.0, 0.0, -3.0], "loc": [0.0, -0.2118, 0.0079]}}
HEAD_MARKS = {"head_top", "forehead", "face", "mouth", "back_of_head", "ear.L", "ear.R"}
HEAD_REACH_FLEX = 70   # deg the elbow bends at least for a hand on the head, unless an elbow word says otherwise
CONTACT_SLACK = 0.04   # m past a contact's own target radius that still counts as met (skin 0.03 -> 7 cm; the other arm's 0.08-0.09 -> 12-13 cm)
BALANCE_IN = 0.06      # the centre of mass at least this far inside the feet (m), else the hips move over them
DIRS = {"up": (0, 0, 1), "down": (0, 0, -1), "forward": (0, -1, 0), "back": (0, 1, 0)}   # out/across depend on the side
ELBOW = {"straight": (0, 20), "bent": (55, 100), "tight": (125, 145)}
HEAD = {"turn": {"left": 40, "right": -40, "ahead": 0}, "pitch": {"down": 25, "up": -25, "level": 0}, "tilt": {"left": 20, "right": -20, "level": 0}}
# a forward bend with a hand that must reach the knees or lower bends at the hips too (2026-11-06):
# to the knees 75 deg with the knees bent (the hips 0.25 m lower), lower than the knees 80 deg, legs straight
DEEP_BEND = {"knee": (75, 0.25), "low": (80, 0.0)}
LOW_MARKS = ("shin", "foot", "ankle", "toe")
TRUNK = {"bend": {"forward": 35, "back": -18, "upright": 0}, "side": {"left": 20, "right": -20, "none": 0}, "twist": {"left": 30, "right": -30, "none": 0}}
SKIN_GAP = 0.035          # the palm's centre this far out from the skin it touches
BODY_R = {"torso": 0.115, "head": 0.105}
PATH_W = 0.05          # a start's cost per degree its way there spends past a joint's range (2026-11-02)
VERSION = 8            # poseprog's cache key: 2 = starts chosen by the way there too; 3 = the compiler's stance, balanced;
                       # 4 = head contacts with a bent elbow, contacts met only at 7 cm; 5 = met at each target's own radius + 4 cm;
                       # 6 = a touching arm's forearm may lie against the body; 7 = reach-first starts for missed contacts;
                       # 8 = a forward bend with a hand at the knees or low bends at the hips too (DEEP_BEND)


def axis(word, side):
    """A direction word as a unit vector in the body's frame (+x its left, -y ahead, +z up): one of
    up, down, forward, back, out (the arm's own side), across, or two joined by "_" (halfway)."""
    prim = dict(DIRS); prim["out"] = (side, 0, 0); prim["across"] = (-side, 0, 0)
    v = sum(np.array(prim[w], float) for w in word.split("_"))
    return v / np.linalg.norm(v)


def parse(text):
    spec = {}
    for line in text.strip().splitlines():
        if ":" not in line: continue
        part, rest = (x.strip() for x in line.split(":", 1))
        if part == "intent": spec["intent"] = rest; continue
        if part in ("hands", "shoulders", "posture"): spec[part] = rest.split()[0]; continue
        items = {}
        for it in rest.replace(",", ";").split(";"):
            w = it.split()
            if len(w) >= 2: items[w[0]] = w[1]
        spec[part] = items
    return spec


def bone_of(ctrl):
    return R.BONE.get(ctrl) or (f"{R._SIDED_BONE[ctrl.split('.')[0]]}.{ctrl.split('.')[1]}" if "." in ctrl else ctrl)


def basis_of(controls):
    """{control: {"rot": deg XYZ, "loc": m}} -> fk basis {bone: {...}} in each bone's rotation mode."""
    out = {}
    by = fk.data()["by"]
    for c, v in controls.items():
        b = bone_of(c); e = {}
        if b not in by: continue
        if "rot" in v:
            Rm = fk.euler_to_m3([math.radians(x) for x in v["rot"]], "XYZ")
            mode = by[b]["mode"]
            if mode == "QUATERNION": e["quat"] = list(fk.m3_to_quat(Rm))
            else: e["euler"] = list(fk.m3_to_euler(Rm, mode))
        if "loc" in v: e["loc"] = list(v["loc"])
        out[b] = e
    return out


class Solver:
    def __init__(self, spec, base=None):
        self.spec = spec
        posture = spec.get("posture", "stand")
        ctl = {k: {"rot": list(v)} for k, v in STAND.items()}
        ctl.update({k: {a: list(x) for a, x in v.items()} for k, v in STAND_BODY.items()})
        # lying down, the arms are solved on the standing body (the body's frame is the same) and the
        # compiler lays it down: it tips the root about the hips, which this rig copy does not
        self.lying = posture == "lie"
        for name in ({"stand": [], "crouch": ["crouch"], "kneel": ["crouch", "kneel"], "sit": ["sit"], "lie": []}[posture]):
            for c, v in POSTURES[name].items():
                if isinstance(v, dict): ctl[c] = {k: list(x) for k, x in v.items()}
        ctl = {c: (v if isinstance(v, dict) else {"rot": v}) for c, v in ctl.items()}
        for c, v in (base or {}).items(): ctl[c] = dict(v)
        self.ctl = ctl
        self.x0 = {c: list(v.get("rot", [0, 0, 0])) for c, v in ctl.items()}
        self.rig = fk.Rig(basis_of(ctl))
        self.W = fk.data()["world"]
        self.lm = posecat.LM

    # -- reading the posed body -------------------------------------------------------------------
    def world(self, n): return self.W @ self.rig.M[n]

    def measure(self):
        return posecat.read(self.world)

    def set_rot(self, ctrl, rot):
        self.ctl.setdefault(ctrl, {})["rot"] = list(rot)
        self.rig.set(basis_of({ctrl: self.ctl[ctrl]}))

    # -- the objective, as residuals (the cost is their sum of squares) ----------------------------
    def landmark(self, m, name, s):
        """A point the palm should be at for `hand NAME`, and how close counts."""
        if name in self.lm:
            v = self.lm[name]; M = self.world(v["bone"])
            n = M[:3, :3] @ np.array(v["nlocal"]); n /= np.linalg.norm(n)
            return m["lm"][name] + n * SKIN_GAP, 0.03
        o = name.split(".")[1]
        if name.startswith("upper_arm."): return (m[f"shoulder.{o}"] + m[f"elbow.{o}"]) / 2, 0.09
        if name.startswith("elbow."): return m[f"elbow.{o}"], 0.08
        raise KeyError(name)

    def arm_res(self, s, m, B, r):
        sp = self.spec.get(f"arm.{s}", {})
        side = 1.0 if s == "L" else -1.0
        sh, el, wr, pa = m[f"shoulder.{s}"], m[f"elbow.{s}"], m[f"wrist.{s}"], m[f"palm.{s}"]
        axes = dict(DIRS); axes["out"] = (side, 0, 0); axes["across"] = (-side, 0, 0)
        if "dir" in sp:
            v = B @ (pa - sh); u = v / (np.linalg.norm(v) + 1e-9)
            c = float(u @ axis(sp["dir"], side))
            r.append(2.0 * (1 - c))
            if "_" not in sp["dir"]:                       # a plain direction must also read as that word
                best_other = max(float(u @ np.array(a, float)) for k, a in axes.items() if k != sp["dir"])
                r.append(3.2 * max(0.0, best_other - c + 0.25))
        if "upper" in sp:                                  # the upper arm's own direction (where the elbow points)
            v = B @ (el - sh); u = v / (np.linalg.norm(v) + 1e-9)
            # within ~18 deg for a joined word, ~30 for a plain one; loose (60) when the hand has a place to be
            thr = 0.95 if "_" in sp["upper"] else 0.5 if "hand" in sp else 0.87
            r.append(1.7 * max(0.0, thr - float(u @ axis(sp["upper"], side))) / 0.1)
        if "forearm" in sp:                                # elbow to wrist
            v = B @ (wr - el); u = v / (np.linalg.norm(v) + 1e-9)
            thr = 0.95 if "_" in sp["forearm"] else 0.87
            r.append(1.7 * max(0.0, thr - float(u @ axis(sp["forearm"], side))) / 0.1)
        if "elbow" in sp:
            a = posecat._ang(el - sh, wr - el); lo, hi = ELBOW[sp["elbow"]]
            r += [max(0.0, lo - a) / 15, max(0.0, a - hi) / 15]
        elif sp.get("hand") in HEAD_MARKS:                 # a hand to the head comes up in front, the elbow bent
            r.append(max(0.0, HEAD_REACH_FLEX - posecat._ang(el - sh, wr - el)) / 15)
        if "hand" in sp and sp["hand"] not in ("free", "above_head"):
            p, rad = self.landmark(m, sp["hand"], s)
            r.append(5.5 * max(0.0, np.linalg.norm(pa - p) - rad) / 0.05)
        if sp.get("hand") == "above_head" or sp.get("level") == "above":
            z, top = (B @ pa)[2], (B @ m["lm"]["head_top"])[2]
            r.append(3.2 * max(0.0, top + 0.08 - z) / 0.05)
        if "level" in sp and sp["level"] != "above":
            z = (B @ pa)[2]
            zh, zc, zm, zt = ((B @ m["hips"])[2], (B @ m["lm"]["chest"])[2], (B @ m["lm"]["mouth"])[2], (B @ m["lm"]["head_top"])[2])
            band = {"low": (-9, zh - 0.15), "waist": (zh - 0.03, zc - 0.17), "chest": (zc - 0.07, zm - 0.1), "head": (zm, zt - 0.02)}[sp["level"]]
            r += [3.2 * max(0.0, band[0] - z) / 0.05, 3.2 * max(0.0, z - band[1]) / 0.05]

    def collision_res(self, m, s, r):
        """The arm's bones (and the hand, unless it is to touch the body) out of the torso and head."""
        hips, neck = m["hips"], (m["shoulder.L"] + m["shoulder.R"]) / 2
        head = (m["lm"]["head_top"] + m["lm"]["mouth"]) / 2
        sp = self.spec.get(f"arm.{s}", {})
        touching = "hand" in sp and sp["hand"] not in ("free", "above_head")
        # a hand that touches the body brings its forearm against it: then only the forearm by the elbow
        # keeps out, and only of the body itself (no margin); a free arm keeps a hand's breadth clear
        pts = [(m[f"shoulder.{s}"] * (1 - t) + m[f"elbow.{s}"] * t, 0.04, 0.03) for t in (0.5, 0.75, 1.0)]
        if touching: pts += [(m[f"elbow.{s}"] * 0.67 + m[f"wrist.{s}"] * 0.33, 0.0, 0.0)]
        else: pts += [(m[f"elbow.{s}"] * (1 - t) + m[f"wrist.{s}"] * t, 0.04, 0.03) for t in (0.33, 0.66, 1.0)] + [(m[f"palm.{s}"], 0.04, 0.03)]
        seg = neck - hips; ss = float(seg @ seg)
        for p, mt, mh in pts:
            t = min(1.0, max(0.0, float((p - hips) @ seg) / ss))
            d = np.linalg.norm(p - (hips + t * seg))
            r += [max(0.0, BODY_R["torso"] + mt - d) / 0.03, max(0.0, BODY_R["head"] + mh - np.linalg.norm(p - head)) / 0.03]

    def anatomy_res(self, joints, r, weight=1.0, margin=3.0):
        J = npcheck.joints(self.rig.M, only=set(joints))
        for j in joints:
            if j not in J: continue
            for i, lo, hi, _, _ in npcheck.anatomy.limits(j):
                v = J[j][i]
                r += [weight * max(0.0, lo + margin - v) / 10, weight * max(0.0, v - hi + margin) / 10]

    def path_excess(self, sides, ctrls, cand, steps=(0.25, 0.5, 0.75)):
        """Degrees past the human ranges of the arms' joints on the way to `cand` from where this key
        starts (self.x0), each control's rotation slerped as the compiler blends keys: an end pose a
        body can make may still be reached only by swinging the arm behind it."""
        joints = [f"{j}.{s}" for s in sides for j in ("shoulder", "elbow", "wrist")]
        q = lambda rot: fk.m3_to_quat(fk.euler_to_m3([math.radians(x) for x in rot], "XYZ"))
        ends = {c: (q(self.x0.get(c, [0, 0, 0])), q(cand[c])) for c in ctrls}
        out = 0.0
        for t in steps:
            mid = {c: {"rot": [math.degrees(x) for x in fk.m3_to_euler(fk.quat_to_m3(fk.slerp(a, b, t)), "XYZ")]}
                   for c, (a, b) in ends.items()}
            self.rig.set(basis_of({**{c: self.ctl[c] for c in ctrls}, **mid}))
            J = npcheck.joints(self.rig.M, only=set(joints))
            for j in joints:
                if j not in J: continue
                for i, lo, hi, _, _ in npcheck.anatomy.limits(j):
                    out += max(0.0, lo - J[j][i], J[j][i] - hi)
        self.rig.set(basis_of({c: self.ctl[c] for c in ctrls}))
        return out

    def stay_res(self, ctrls, r, weight):
        for c in ctrls:
            for a, b in zip(self.ctl[c]["rot"], self.x0.get(c, [0, 0, 0])): r.append(weight * (a - b) / 90.0)

    def head_trunk_res(self, m, what, r, deep=False):
        sp = self.spec.get(what, {})
        if not sp: return
        if what == "head":
            rel = m["chest_rot"].T @ m["head_rot"]
            f, up = rel @ np.array([0, -1.0, 0]), rel @ np.array([0, 0, 1.0])
            got = {"turn": math.degrees(math.atan2(f[0], -f[1])), "pitch": math.degrees(math.asin(max(-1, min(1, -f[2])))),
                   "tilt": math.degrees(math.atan2(up[0], up[2]))}
            table = HEAD
        else:
            c = m["root_rot"].T @ m["chest_rot"]
            cu, cf = c @ np.array([0, 0, 1.0]), c @ np.array([0, -1.0, 0])
            got = {"bend": math.degrees(math.atan2(-cu[1], cu[2])), "side": math.degrees(math.atan2(cu[0], cu[2])),
                   "twist": math.degrees(math.atan2(cf[0], -cf[1]))}
            table = TRUNK
        for k in table:
            want = table[k][sp[k]] if k in sp else 0.0
            if deep and k == "bend": want = DEEP_BEND[deep][0]
            r.append((got[k] - want) / 10 * (1.7 if k in sp else 0.55))

    # -- solving ------------------------------------------------------------------------------------
    def levmar(self, ctrls, residuals, dims, iters=25, eps=0.5):
        """Levenberg-Marquardt on the controls' Euler degrees; the Jacobian by forward differences,
        each try undone from the rig's saved matrices instead of evaluated back."""
        keys = [(c, i) for c in ctrls for i in dims[c]]
        def apply(c, rot):
            self.ctl[c]["rot"] = list(rot); return self.rig.trial(basis_of({c: self.ctl[c]}))
        r0 = np.array(residuals()); cost = float(r0 @ r0); lam = 1e-2
        for _ in range(iters):
            if cost < 1e-5: break
            J = np.zeros((len(r0), len(keys)))
            for k, (c, i) in enumerate(keys):
                old = list(self.ctl[c]["rot"]); rot = list(old); rot[i] += eps
                tok = apply(c, rot); J[:, k] = (np.array(residuals()) - r0) / eps
                self.rig.undo(tok); self.ctl[c]["rot"] = old
            g, H = J.T @ r0, J.T @ J
            for _ in range(6):                               # damping up until a step lowers the cost
                step = -np.linalg.solve(H + lam * (np.diag(np.diag(H)) + np.eye(len(keys)) * 1e-3), g)
                step = np.clip(step, -25, 25)
                saved = {c: list(self.ctl[c]["rot"]) for c in ctrls}
                for k, (c, i) in enumerate(keys): self.ctl[c]["rot"][i] += float(step[k])
                self.rig.set(basis_of({c: self.ctl[c] for c in ctrls}))
                r1 = np.array(residuals()); c1 = float(r1 @ r1)
                if c1 < cost:
                    lam = max(lam / 3, 1e-6); improved = cost - c1; r0, cost = r1, c1; break
                for c in ctrls: self.ctl[c]["rot"] = saved[c]
                self.rig.set(basis_of({c: self.ctl[c] for c in ctrls})); lam *= 4
            else:
                break
            if improved < 1e-6 * max(cost, 1e-9): break
        return cost

    def solve(self):
        sp = self.spec
        def ensure(ctrls):
            for c in ctrls: self.ctl.setdefault(c, {"rot": [0, 0, 0]}); self.ctl[c].setdefault("rot", [0, 0, 0])
        mark = lambda a: str(a.get("hand", "")).split(".")[0]
        arms_sp = [sp.get(f"arm.{s}") or {} for s in "LR"]
        deep = None
        if (sp.get("trunk") or {}).get("bend") == "forward" and sp.get("posture", getattr(self, "beat_posture", "stand")) == "stand":
            if any(mark(a) in LOW_MARKS or a.get("level") == "low" for a in arms_sp): deep = "low"
            elif any(mark(a) == "knee" for a in arms_sp): deep = "knee"
        if deep and DEEP_BEND[deep][1]:
            t = self.ctl.setdefault("torso", {"rot": [0, 0, 0]}); t.setdefault("loc", [0.0, 0.0, 0.0])
            t["loc"] = [t["loc"][0], t["loc"][1], t["loc"][2] - DEEP_BEND[deep][1]]
            self.rig.set(basis_of({"torso": t}))
        if sp.get("trunk") or sp.get("shoulders"):
            ctrls = (["torso"] if deep else []) + ["chest"] + (["shoulder.L", "shoulder.R"] if sp.get("shoulders") else []); ensure(ctrls)
            def rt():
                m = self.measure(); r = []
                self.head_trunk_res(m, "trunk", r, deep); self.anatomy_res(["spine"], r)
                if sp.get("shoulders") == "up":
                    for s in "LR":
                        rise = ((m["chest_rot"].T @ (m[f"shoulder.{s}"] - self.world("DEF-CHEST-02")[:3, 3]))[2]
                                - (posecat.REST[f"DEF-ARM-01.{s}"][:3, 3] - posecat.REST["DEF-CHEST-02"][:3, 3])[2])
                        r.append(max(0.0, 0.04 - rise) / 0.01)
                self.stay_res([c for c in ctrls if c != "torso"], r, 0.45); self.stay_res(["torso"] if deep else [], r, 0.08); return r
            self.levmar(ctrls, rt, {c: [0, 1, 2] for c in ctrls})
        if deep and not sp.get("head"):
            # a deep bend carries the head down with it: the neck kept in its range, not the head held level
            ctrls = ["neck", "head"]; ensure(ctrls)
            def rn():
                m = self.measure(); r = []; self.anatomy_res(["neck"], r, margin=10.0); self.stay_res(ctrls, r, 0.3); return r
            self.levmar(ctrls, rn, {c: [0, 1, 2] for c in ctrls})
        if sp.get("head"):
            ctrls = ["neck", "head"]; ensure(ctrls)
            def rh():
                m = self.measure(); r = []
                self.head_trunk_res(m, "head", r); self.anatomy_res(["neck"], r); self.stay_res(ctrls, r, 0.45); return r
            self.levmar(ctrls, rh, {c: [0, 1, 2] for c in ctrls})
        both = sp.get("hands") == "together"
        arms = ["L", "R"] if both else [s for s in "LR" if sp.get(f"arm.{s}")]
        crossing = any(str(sp.get(f"arm.{s}", {}).get("hand", "")).split(".")[0] in ("upper_arm", "elbow", "shoulder") for s in arms)
        rounds = 2 if both or crossing else 1
        dims = lambda s: {f"upper_arm_fk.{s}": [0, 1, 2], f"forearm_fk.{s}": [1, 2], f"hand_fk.{s}": [0, 2]}
        if both:
            # two hands that meet are one problem: both arms solved together, from symmetric starts
            ctrls = list(dims("L")) + list(dims("R")); ensure(ctrls)
            dd = {**dims("L"), **dims("R")}
            def rb():
                m = self.measure(); r = []
                for s in "LR":
                    self.arm_res(s, m, m["to_body"], r); self.collision_res(m, s, r)
                    self.anatomy_res([f"shoulder.{s}", f"elbow.{s}", f"wrist.{s}"], r, weight=4.5)
                r.append(5.5 * max(0.0, np.linalg.norm(m["palm.L"] - m["palm.R"]) - 0.07) / 0.05)
                self.stay_res(ctrls, r, 0.22); return r
            starts = [{c: list(self.ctl[c]["rot"]) for c in ctrls}]
            for up in (-40, 0, 60):                       # hands low in front, at the chest, overhead
                st = {}
                for s, sg in (("L", 1), ("R", -1)):
                    st[f"upper_arm_fk.{s}"] = [up, 0, -60 * sg]; st[f"forearm_fk.{s}"] = [0, 0, -70 * sg]; st[f"hand_fk.{s}"] = [0, 0, 0]
                starts.append(st)
            tried = []
            for st in starts:
                for c, rot in st.items(): self.ctl[c]["rot"] = list(rot)
                self.rig.set(basis_of({c: self.ctl[c] for c in ctrls}))
                f = self.levmar(ctrls, rb, dd, iters=6); cand = {c: list(self.ctl[c]["rot"]) for c in ctrls}
                f += PATH_W * self.path_excess("LR", ctrls, cand)      # reached without leaving human range
                tried.append((f, cand))
                if f < 0.02: break
            for c, rot in min(tried, key=lambda x: x[0])[1].items(): self.ctl[c]["rot"] = list(rot)
            self.rig.set(basis_of({c: self.ctl[c] for c in ctrls}))
            self.levmar(ctrls, rb, dd, iters=30)
            return self.result()
        for rnd in range(rounds):
            for s in arms:
                ctrls = list(dims(s)); ensure(ctrls)
                def ra(s=s, ctrls=ctrls):
                    m = self.measure(); r = []
                    self.arm_res(s, m, m["to_body"], r); self.collision_res(m, s, r)
                    if both: r.append(5.5 * max(0.0, np.linalg.norm(m["palm.L"] - m["palm.R"]) - 0.07) / 0.05)
                    if sp.get("hands") == "apart": r.append(5.5 * max(0.0, 0.25 - np.linalg.norm(m["palm.L"] - m["palm.R"])) / 0.05)
                    # the joints' ranges outweigh every target: a pose a body cannot make is no answer
                    self.anatomy_res([f"shoulder.{s}", f"elbow.{s}", f"wrist.{s}"], r, weight=4.5)
                    self.stay_res(ctrls, r, 0.22); return r
                if rnd == 0:
                    # a few starts (hanging, forward with the elbow bent, rolled with the elbow bent, up):
                    # each gets a few steps, and the best one is finished
                    sg = 1 if s == "L" else -1
                    starts = [{c: list(self.ctl[c]["rot"]) for c in ctrls},
                              {f"upper_arm_fk.{s}": [0, 0, -90 * sg], f"forearm_fk.{s}": [0, 0, -90 * sg], f"hand_fk.{s}": [0, 0, 0]},
                              {f"upper_arm_fk.{s}": [-30, -90 * sg, -20 * sg], f"forearm_fk.{s}": [0, 0, -100 * sg], f"hand_fk.{s}": [0, 0, 0]},
                              {f"upper_arm_fk.{s}": [60, 0, -10 * sg], f"forearm_fk.{s}": [0, 0, -10 * sg], f"hand_fk.{s}": [0, 0, 0]}]
                    tried = []
                    for st in starts:
                        for c, rot in st.items(): self.ctl[c]["rot"] = list(rot)
                        self.rig.set(basis_of({c: self.ctl[c] for c in ctrls}))
                        f = self.levmar(ctrls, ra, dims(s), iters=5); cand = {c: list(self.ctl[c]["rot"]) for c in ctrls}
                        f += PATH_W * self.path_excess(s, ctrls, cand)   # reached without leaving human range
                        tried.append((f, cand))
                        if f < 0.02: break
                    hand = sp.get(f"arm.{s}", {}).get("hand")
                    if min(t[0] for t in tried) >= 0.02 and hand and hand not in ("free", "above_head"):
                        # a contact none of the starts reached: reach for it first (the contact and the
                        # joints' ranges only, from each start), then finish that with every term
                        def reach(s=s):
                            m = self.measure(); r = []
                            p, rad = self.landmark(m, hand, s)
                            r.append(5.5 * max(0.0, np.linalg.norm(m[f"palm.{s}"] - p) - rad) / 0.05)
                            self.anatomy_res([f"shoulder.{s}", f"elbow.{s}", f"wrist.{s}"], r, weight=1.0)
                            return r
                        for st in starts:
                            for c, rot in st.items(): self.ctl[c]["rot"] = list(rot)
                            self.rig.set(basis_of({c: self.ctl[c] for c in ctrls}))
                            self.levmar(ctrls, reach, dims(s), iters=12)
                            f = self.levmar(ctrls, ra, dims(s), iters=5); cand = {c: list(self.ctl[c]["rot"]) for c in ctrls}
                            f += PATH_W * self.path_excess(s, ctrls, cand)
                            tried.append((f, cand))
                            if f < 0.02: break
                    f, best = min(tried, key=lambda x: x[0])
                    for c, rot in best.items(): self.ctl[c]["rot"] = list(rot)
                    self.rig.set(basis_of({c: self.ctl[c] for c in ctrls}))
                    self.levmar(ctrls, ra, dims(s), iters=25)
                else:
                    self.levmar(ctrls, ra, dims(s), iters=15)
        return self.result()

    def implied(self):
        """The categories the spec asks for, as posecat names them."""
        sp, out = self.spec, {}
        for s in "LR":
            for k, v in sp.get(f"arm.{s}", {}).items():
                key = {"dir": f"arm.{s}", "elbow": f"elbow.{s}", "hand": f"hand.{s}", "level": f"level.{s}"}.get(k)
                if key: out[key] = v.split("_") if k == "dir" else v       # a joined direction reads as either word
        if sp.get("hands"): out["hands"] = sp["hands"]
        for k, v in sp.get("head", {}).items(): out[f"head.{k}"] = v
        for k, v in sp.get("trunk", {}).items(): out[f"trunk.{k}"] = v
        if sp.get("shoulders"): out["shoulders"] = sp["shoulders"]
        out["posture"] = sp.get("posture", "stand")
        return out

    def balance(self):
        """Weight over the feet (the animator's balance): standing, if the pose carries the centre of mass
        within BALANCE_IN of the edge of the feet, the torso (hips and all above them) slides over them,
        the feet staying put. A lean back pushes the hips forward, as a person does."""
        if self.spec.get("posture", "stand") != "stand": return
        sk = fk.data()["skin"]
        t = self.ctl.setdefault("torso", {"rot": [0, 0, 0]}); t.setdefault("loc", [0.0, 0.0, 0.0])
        for _ in range(5):
            com = npcheck.skinned(self.rig.M, sk["sample"]).mean(axis=0)
            soles = npcheck.skinned(self.rig.M, sk["sole"]["L"] + sk["sole"]["R"]); on = soles[soles[:, 2] < 0.02]
            if len(on) < 3: return
            move = [0.0, 0.0]
            for i in (0, 1):
                lo, hi = on[:, i].min() + BALANCE_IN, on[:, i].max() - BALANCE_IN
                if lo > hi: lo = hi = (on[:, i].min() + on[:, i].max()) / 2
                move[i] = (lo - com[i]) if com[i] < lo else (hi - com[i]) if com[i] > hi else 0.0
            if abs(move[0]) < 0.003 and abs(move[1]) < 0.003: return
            t["loc"] = [t["loc"][0] + move[0] / 0.75, t["loc"][1] + move[1] / 0.75, t["loc"][2]]   # ~3/4 of the mass rides the torso
            self.rig.set(basis_of({"torso": t}))

    def result(self):
        self.balance()
        cats = posecat.categories(self.world)
        want = self.implied()
        met = {k: (cats.get(k) in v) if isinstance(v, list) else cats.get(k) == v for k, v in want.items()}
        m = self.measure()
        for s in "LR":                                  # a hand touches its landmark when it is within reach of it
            h = want.get(f"hand.{s}")
            if h and h not in ("free", "above_head") and not met.get(f"hand.{s}"):
                p, rad = self.landmark(m, h, s)
                met[f"hand.{s}"] = bool(np.linalg.norm(m[f"palm.{s}"] - p) < rad + CONTACT_SLACK)
        if self.lying: met["posture"] = True             # the compiler lays the body down
        cats = dict(cats, posture="lie") if self.lying else cats
        J = npcheck.joints(self.rig.M)
        return {"controls": {c: {k: [round(x, 2) for x in v[k]] for k in v} for c, v in self.ctl.items()},
                "categories": cats, "want": want, "met": met, "all_met": all(met.values()),
                "violations": npcheck.anatomy.violations(J, npcheck.anatomy_slack())}


def solve(text, base=None):
    return Solver(parse(text), base).solve()
