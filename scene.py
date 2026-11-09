"""A scene of two characters, stepped on the server (2026-11-08).

The simulator's page ran the scene in the browser (server/static/index.html: saySequence, runStep,
bodyActs, stepEnds, faceTalk, walkThrough). This is that loop in Python, so a scene can be made,
cast, started, paused and stopped by commands (the `scene` program) and watched from any page.

A step: qwen-say decides each character's move in turn (one hears the moves made before it in the
step), the moves become body actions, the actions become one sentence per character in its own
frame ("walks to a spot 1.2 m ahead"), and qwen-animate writes each one's program, which Blender
compiles, checks and exports as a clip. The scene keeps the record (every statement, s1, s2, ...),
where each character stands and faces, and every step's clips.

The models are a backend: `Backend.say(body)` is /api/say's answer, `Backend.animate(items)` a
finished /api/animate job. HttpBackend calls a running simulator server; the Modal app passes its
own in-process one.
"""
import json
import math
import random
import re
import time
import urllib.request
from datetime import datetime, timedelta, timezone

MAX_CAST = 2                      # a scene has two characters, never a third
SPACING = 1.4                     # metres between cast members, side by side
INTERRUPT = 0.15                  # a character not on the floor speaks this often anyway
TOWARD = ["walks to", "runs to", "turns to", "looks at", "points at", "reaches towards"]
ALONG = ["follows", "leads"]
STOP_SHORT, MIN_WALK, MIN_RUN, FACING_DEG, FEW_STEPS_M, LEAD_M = 1.2, 0.5, 1.5, 15, 1.0, 3.0
FACE_TALK, MAX_ACTS = 30, 6
D2R = math.pi / 180
MARK_ACT = {"open": "opens their hands", "fist": "clenches their fists"}
FEEL_ACT = {"agrees": "gives a small smile", "disagrees": "frowns", "amused": "smiles", "doubtful": "frowns",
            "surprised": "looks surprised", "worried": "looks worried", "annoyed": "clenches their fists",
            "bored": "yawns", "interested": "waits for a second"}
# the looks a cast member can have (MPFB people on the one rig, humans/cast.json), each with the
# name it has until someone renames it; the animation is the rig's, so any look plays any clip
CHARACTERS = {"sam": "Sam", "maya": "Maya", "leo": "Leo", "ana": "Ana", "kenji": "Kenji", "zoe": "Zoe"}
MAX_NAME = 16
# words the record and qwen-say's grammar use around names ("[s3 I] ...", "to Maya and Sam, next Sam")
NOT_NAMES = {"i", "me", "my", "myself", "you", "we", "us", "they", "them", "he", "him", "she", "her", "it",
             "to", "and", "next", "or", "the", "a", "an", "everyone", "everybody", "someone", "nobody", "no one"}
PUSHES = ["flattering", "contemptuous", "poetic", "upbeat", "suspicious", "condescending", "wistful",
          "oracle", "bleak", "dismissive", "ornate", "curt", "hostile"]


LAYERS, NEURONS = 32, 9216                          # Qwen3.5-4B
CUSTOM = re.compile(r"L(\d{1,2})N(\d{1,4})([+-])")


def custom_push(push):
    """(layer, neuron, sign) of a push written L<layer>N<neuron><+|-> (any neuron, pushed up or
    down), or None; the engine takes it as it takes a catalog one (vsteer.custom_push)."""
    m = CUSTOM.fullmatch(push or "")
    if not m or not (0 <= int(m[1]) < LAYERS and 0 <= int(m[2]) < NEURONS): return None
    return int(m[1]), int(m[2]), 1 if m[3] == "+" else -1


def goes(v): return v in ("walks to", "runs to")


def home(i, n): return (i - (n - 1) / 2) * SPACING


def split_acts(acts): return [a.strip() for a in acts.split(", then ")] if acts else []


# ---- places ----------------------------------------------------------------------------------
def rel_to(me, other):
    """other's place in me's frame (me at 0, facing -y)."""
    wx, wy = other["pos"][0] - me["pos"][0], other["pos"][1] - me["pos"][1]
    c, s = math.cos(me["yaw"] * D2R), math.sin(me["yaw"] * D2R)
    return [c * wx + s * wy, -s * wx + c * wy]


def where(rel):
    left, ahead, parts = rel[0], -rel[1], []
    if abs(ahead) >= 0.3: parts.append(f"{abs(ahead):.1f} m {'ahead' if ahead > 0 else 'behind'}")
    if abs(left) >= 0.3: parts.append(f"{abs(left):.1f} m to the {'left' if left > 0 else 'right'}")
    return "a spot " + (" and ".join(parts) if parts else "right here")


def walk_through(i, acts, at, names, ends):
    """Each action as qwen-animate's sentence puts it, and where i stands (its own frame at the start
    of the line) and faces once they are done."""
    here, yaw, moved, out = [0.0, 0.0], 0.0, False, []
    for a in split_acts(acts):
        if a == "turns around": yaw += 180
        elif a == "walks a few steps":
            here = [here[0] + math.sin(yaw * D2R) * FEW_STEPS_M, here[1] - math.cos(yaw * D2R) * FEW_STEPS_M]; moved = True
        verb = next((t for t in TOWARD + ALONG if a.startswith(t + " ")
                     and any(o != i and n == a[len(t) + 1:] for o, n in enumerate(names))), None)
        if not verb: out.append(a); continue
        o = next(k for k, n in enumerate(names) if k != i and n == a[len(verb) + 1:])
        tgt = rel_to(at[i], ends[o] if (verb == "follows" or moved) and ends and ends[o] else at[o])
        if verb == "leads":                            # away from them
            ax, ay = here[0] - tgt[0], here[1] - tgt[1]; n = math.hypot(ax, ay) or 1
            tgt = [here[0] + ax / n * (LEAD_M + STOP_SHORT), here[1] + ay / n * (LEAD_M + STOP_SHORT)]
            verb = "walks to"
        elif verb == "follows":
            out.append("waits for a second"); verb = "walks to"
        dx, dy = tgt[0] - here[0], tgt[1] - here[1]
        d = math.hypot(dx, dy) or 1; ang = math.atan2(dx, -dy) / D2R
        off = ((ang - yaw + 180) % 360 + 360) % 360 - 180
        if verb == "runs to" and d - STOP_SHORT < MIN_RUN: verb = "walks to"
        if verb == "walks to" and d - STOP_SHORT < MIN_WALK: verb = "turns to"
        if verb == "turns to" and abs(off) < FACING_DEG: verb = "looks at"
        spot = tgt
        if goes(verb): k = (d - STOP_SHORT) / d; spot = [here[0] + dx * k, here[1] + dy * k]
        c, s = math.cos(yaw * D2R), math.sin(yaw * D2R); vx, vy = spot[0] - here[0], spot[1] - here[1]
        out.append(f"{verb} {where([c * vx + s * vy, -s * vx + c * vy])}")
        if goes(verb): here, yaw, moved = spot, ang, True
        elif verb == "turns to": yaw = ang
    return out, here, yaw


def action_text(i, acts, at, names, ends):
    out, _, _ = walk_through(i, acts, at, names, ends)
    return f"The character {', then '.join(out) if out else 'stands for a moment'}."


def body_acts(moves, order):
    """The body acting out each move: a speaker's gestures, a listener's feeling; two walking to
    each other in one step: the second to move stays put; one who is led follows its leader."""
    walks = set()
    for m in (moves[i] for i in order):
        if m["kind"] == "listen": m["act"] = FEEL_ACT.get(m.get("feel"), ""); continue
        def to(a):
            return a[a.index(" to ") + 4:] if a.startswith("walks to ") or a.startswith("runs to ") else None
        acts = [a for a in split_acts(m.get("act", "")) if not (to(a) and f"{to(a)}>{m['who']}" in walks)]
        for a in acts:
            if to(a): walks.add(f"{m['who']}>{to(a)}")
        for k in m.get("marks") or []:
            a = f"points at {k['target']}" if k["mark"] == "point" else MARK_ACT.get(k["mark"])
            if a and a not in acts: acts.append(a)
        m["act"] = ", then ".join(acts[:MAX_ACTS - 1])
    for m in moves:
        for a in split_acts(m.get("act", "")):
            if not a.startswith("leads "): continue
            x = next((y for y in moves if y is not m and y["who"] == a[6:]), None)
            if not x: continue
            own = [b for b in split_acts(x.get("act", "")) if not b.startswith(("walks ", "follows ", "leads "))]
            x["act"] = ", then ".join([f"follows {m['who']}"] + own[:MAX_ACTS - 2])


def step_ends(moves, at, names, ends=None, passes=2):
    """Where each character ends this step, in the floor's frame."""
    out = []
    for i, m in enumerate(moves):
        _, here, yaw = walk_through(i, m.get("act", ""), at, names, ends)
        c, s = math.cos(at[i]["yaw"] * D2R), math.sin(at[i]["yaw"] * D2R)
        out.append({"pos": [at[i]["pos"][0] + c * here[0] - s * here[1], at[i]["pos"][1] + s * here[0] + c * here[1]],
                    "yaw": at[i]["yaw"] + yaw})
    return step_ends(moves, at, names, out, passes - 1) if passes > 1 else out


def face_talk(items, at, names, ends):
    """A speaker turns to whom it speaks to, a listener to whom it listens to."""
    for i, x in enumerate(items):
        p = next((names.index(n) for n in x.get("to") or [] if n in names and names.index(n) != i), None)
        if p is None: p = next((j for j, y in enumerate(items) if j != i and names[i] in (y.get("to") or [])), None)
        if p is None: continue

        def off(here, yaw, them=None):
            t = rel_to(at[i], them or at[p]); ang = math.atan2(t[0] - here[0], -(t[1] - here[1])) / D2R
            return abs(((ang - yaw + 180) % 360 + 360) % 360 - 180)
        acts = split_acts(x.get("act", ""))
        turn = f"turns to {names[p]}"
        to_p = lambda a: any(a == f"{t} {names[p]}" for t in TOWARD) and not a.startswith(("looks at", "points at"))
        if off([0, 0], 0) > FACE_TALK and not (acts and to_p(acts[0])): acts = [turn] + acts
        _, here, yaw = walk_through(i, ", then ".join(acts), at, names, ends)
        them = ends[p] if ends and ends[p] and (here[0] or here[1]) else at[p]
        if off(here, yaw, them) > FACE_TALK: acts = acts[:MAX_ACTS - 1] + [turn]
        x["act"] = ", then ".join(acts[:MAX_ACTS])


# ---- the record ------------------------------------------------------------------------------
def entry(e): return f'[s{e["sid"]} {e["who"]}] "{e["text"]}"' + (f' ({e["act"]})' if e.get("act") else "")


def record_for(i, log):
    return [{"sid": e["sid"], "who": "I" if e["slot"] == i else e["who"], "text": e["text"], "act": e.get("act", "")}
            for e in [e for e in log if e["kind"] == "speak"][-8:]]


def animate_input(time_line, rec, own, sentence):
    t = time_line
    if t.lower().startswith("time is "): t = t[8:]
    lines = (["I woke up."] + [entry(e) for e in rec] + [entry(own)])[-5:]
    return "\n".join([f"Time is {t.rstrip('.')}."] + lines + [sentence])


def seed_time(tz_minutes=0, now=None):
    d = (now or datetime.now(timezone.utc)) + timedelta(minutes=tz_minutes)
    hour = d.hour % 12 or 12
    return f"Today {d.strftime('%A')}, {d.day} {d.strftime('%B')}, it's {hour}:{d.minute:02d} {'am' if d.hour < 12 else 'pm'}"


# ---- backends --------------------------------------------------------------------------------
class HttpBackend:
    """A running simulator server (server/app.py), e.g. http://localhost:8765."""

    def __init__(self, base): self.base = base.rstrip("/")

    def _post(self, path, body):
        req = urllib.request.Request(self.base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r: return json.load(r)

    def say(self, body): return self._post("/api/say", body)

    def animate(self, items):
        job = self._post("/api/animate", {"items": items, "session": "scene"})["job"]
        while True:
            with urllib.request.urlopen(f"{self.base}/api/job/{job}", timeout=60) as r: j = json.load(r)
            if j.get("stage") in ("done", "failed", "cancelled") and not any(m.get("checking") for m in j.get("items", [])):
                return j
            time.sleep(0.5)


# ---- the scene -------------------------------------------------------------------------------
def new_scene(scene_id, situation="", tz_minutes=0):
    return {"id": scene_id, "situation": str(situation or "")[:600], "tz_minutes": int(tz_minutes or 0),
            "cast": [], "state": "created", "log": [], "sid": 0, "steps": [], "created": time.time()}


def check_name(scene, name, me=None):
    """A name a cast member can go by: one word of letters (a hyphen or an apostrophe inside it),
    at most MAX_NAME, not a word the record uses, and not the other member's. Returned with its
    first letter capital."""
    n = str(name or "").strip()
    if not n: raise ValueError("a name is needed")
    if len(n) > MAX_NAME: raise ValueError(f"a name is at most {MAX_NAME} letters")
    if not (n[0].isalpha() and n[-1].isalpha() and all(ch.isalpha() or ch in "-'" for ch in n)):
        raise ValueError("a name is one word of letters")
    if n.casefold() in NOT_NAMES: raise ValueError(f"{n!r} cannot be a name here")
    n = n[0].upper() + n[1:]
    if any(m is not me and m["name"].casefold() == n.casefold() for m in scene["cast"]):
        raise ValueError(f"someone in the scene is already called {n}")
    return n


def member(scene, character):
    m = next((m for m in scene["cast"] if m["character"] == character), None)
    if not m: raise ValueError(f"{character!r} is not in the scene")
    return m


def add_character(scene, character, push=None, dose=1.0, name=None):
    """A character joins the scene: one of CHARACTERS (its look), never a third, never twice; it
    goes by `name`, or by the look's own name."""
    if character not in CHARACTERS: raise ValueError(f"unknown character {character!r}: one of {', '.join(CHARACTERS)}")
    if len(scene["cast"]) >= MAX_CAST: raise ValueError(f"a scene has {MAX_CAST} characters at most")
    if any(m["character"] == character for m in scene["cast"]): raise ValueError(f"{CHARACTERS[character]} is already in the scene")
    scene["members_made"] = scene.get("members_made", 0) + 1     # an id that stays through renames and new looks
    m = {"id": f"m{scene['members_made']}", "character": character, "name": check_name(scene, name or CHARACTERS[character]),
         "push": None, "dose": 1.0, "pos": [0.0, 0.0], "yaw": 0.0}
    set_personality(m, push, dose)
    scene["cast"].append(m)
    n = len(scene["cast"])
    for i, c in enumerate(scene["cast"]):                    # side by side until the first step moves them
        if not scene["steps"]: c["pos"] = [home(i, n), 0.0]
    return m


def remove_character(scene, character):
    before = len(scene["cast"])
    scene["cast"] = [m for m in scene["cast"] if m["character"] != character]
    if len(scene["cast"]) == before: raise ValueError(f"{character!r} is not in the scene")


def rename(scene, character, name):
    """A cast member goes by a new name from the next step. The record's moves are kept under it
    (who said what, to whom), so the member and the model still know what it said; the lines
    themselves stay as they were said."""
    m = member(scene, character)
    new, old = check_name(scene, name, m), m["name"]
    if new == old: return m
    for e in scene["log"]:
        if e.get("who") == old: e["who"] = new
        if e.get("next") == old: e["next"] = new
        if e.get("to"): e["to"] = [new if t == old else t for t in e["to"]]
        if e.get("act"): e["act"] = re.sub(rf"(?<!\w){re.escape(old)}(?!\w)", new, e["act"])
    m["name"] = new
    return m


def set_look(scene, character, look):
    """A cast member changes how it looks (another of CHARACTERS) where it stands, keeping its
    personality and place; its name follows the look unless someone named it."""
    m = member(scene, character)
    if look not in CHARACTERS: raise ValueError(f"unknown look {look!r}: one of {', '.join(CHARACTERS)}")
    if look == character: return m
    if any(o["character"] == look for o in scene["cast"]): raise ValueError(f"{CHARACTERS[look]} is already in the scene")
    if m["name"] == CHARACTERS[character]:                     # never renamed: it takes the new look's name
        try: rename(scene, character, CHARACTERS[look])
        except ValueError: pass                                # the other member already goes by it
    m["character"] = look
    return m


def set_personality(m, push, dose=1.0):
    """A character's push (a persona neuron) and its dose, or none: the model as it is."""
    push = (push or "").strip() or None
    if push and push.lower() in PUSHES + ["none", "neutral", "off"]: push = push.lower()
    if push in ("none", "neutral", "off"): push = None
    if push and push not in PUSHES and not custom_push(push):
        raise ValueError(f"unknown personality {push!r}: one of {', '.join(PUSHES)}, a neuron as L<layer>N<neuron><+|-> "
                         f"(layer 0-{LAYERS - 1}, neuron 0-{NEURONS - 1}), or none")
    dose = float(dose if dose is not None else 1.0)
    if push and not 0.5 <= dose <= 1.5: raise ValueError("dose must be 0.5 to 1.5")
    m["push"], m["dose"] = push, (dose if push else 1.0)


MAX_SEED_STEPS = 60


def seed_from(scene, seed):
    """A scene that goes on from one already made (a prerendered sample): its cast where they ended,
    its steps (clips and all) and the record of what was said, rebuilt from the steps, so the next
    step answers what came before. The steps it starts with are not counted as animation made."""
    for m in (seed.get("cast") or [])[:MAX_CAST]:
        c = add_character(scene, m["character"], m.get("push"), m.get("dose", 1.0), m.get("name"))
        if m.get("pos"): c["pos"] = [float(m["pos"][0]), float(m["pos"][1])]
        c["yaw"] = float(m.get("yaw") or 0.0)
    steps = [st for st in (seed.get("steps") or [])[:MAX_SEED_STEPS] if st.get("members")]
    log, sid = [], 0
    for st in steps:
        ms = st["members"]
        names = [m.get("name", "") for m in ms]
        for k in sorted(range(len(ms)), key=lambda k: ms[k].get("order", k)):
            m = ms[k]; sid += 1
            others = [n for n in names if n != m.get("name")]
            e = {"sid": sid, "slot": k, "who": m.get("name", ""), "kind": m.get("kind", "listen"),
                 "to": m.get("to") or others[:1], "text": m.get("says", ""), "act": m.get("act", "")}
            if e["kind"] == "speak": e["next"] = others[0] if others else None
            if m.get("feel"): e["feel"] = m["feel"]
            log.append(e)
    if steps:                                                # each one stands where the last step left it
        for c, m in zip(scene["cast"], steps[-1]["members"]):
            end = m.get("end") or {}
            if end.get("pos"): c["pos"] = [float(end["pos"][0]), float(end["pos"][1])]
            if end.get("yaw") is not None: c["yaw"] = float(end["yaw"])
    scene.update(steps=steps, log=log, sid=sid, anim_s=0.0, seeded=len(steps))
    return scene


def _say_sequence(scene, backend, prompt):
    """qwen-say one character at a time; the first to move is the one the newest line wants to
    answer; the step's first move and the one the newest line wants to answer may speak, anyone else
    only listens, but for an interruption now and then."""
    cast, names = scene["cast"], [m["name"] for m in scene["cast"]]
    fresh, items, sids, recs, order, seconds = [], [None] * len(cast), [0] * len(cast), [None] * len(cast), [], 0.0
    spoke = lambda log: next((e for e in reversed(log) if e["kind"] == "speak"), None)
    first = spoke(scene["log"])
    i = names.index(first["next"]) if first and first.get("next") in names else 0
    while len(order) < len(cast):
        log = scene["log"] + fresh
        sid = scene["sid"] + len(order) + 1
        last = spoke(log)
        speak = not fresh or (last and last.get("next") == names[i]) or random.random() < INTERRUPT
        it = {"character": cast[i]["character"], "push": cast[i]["push"], "dose": cast[i]["dose"], "sid": sid,
              "name": names[i], "others": [n for o, n in enumerate(names) if o != i],
              "record": [{k: v for k, v in e.items() if k not in ("slot",)} for e in log[-8:]],
              "heard": sorted({e["who"] for e in fresh if e["kind"] == "speak"}), "speak": bool(speak)}
        out = backend.say({"time": prompt, "items": [it]})
        x = out["items"][0]
        e = dict(x["move"], sid=sid, slot=i, who=names[i])
        if e["kind"] == "listen": e["act"] = ""
        items[i], sids[i], recs[i] = x, sid, record_for(i, log)
        order.append(i); fresh.append(e); seconds += out.get("seconds", 0)
        nx = names.index(e["next"]) if e["kind"] == "speak" and e.get("next") in names else -1
        i = nx if nx >= 0 and nx not in order else next((o for o in range(len(cast)) if o not in order), 0)
    return {"items": items, "sids": sids, "recs": recs, "order": order, "entries": fresh, "seconds": round(seconds, 2)}


def step(scene, backend, clip_url=lambda step_n, k, m: m.get("glb")):
    """One step of the scene; returns the step as it is kept (and appends it to scene["steps"])."""
    cast = scene["cast"]
    if not cast: raise ValueError("the scene has no characters: add one first")
    t0 = time.time()
    prompt = "\n".join(x for x in (seed_time(scene["tz_minutes"]), scene["situation"]) if x)
    names = [m["name"] for m in cast]
    at = [{"pos": list(m["pos"]), "yaw": m["yaw"]} for m in cast]
    said = _say_sequence(scene, backend, prompt)
    moves = [x["move"] for x in said["items"]]
    for i, m in enumerate(moves): m["who"] = names[i]
    body_acts(moves, said["order"])
    ends = step_ends(moves, at, names)
    face_talk(moves, at, names, ends)
    ends = step_ends(moves, at, names)
    # the record keeps the moves in the order made, with the actions body_acts and face_talk gave them
    by_slot = {e["slot"]: e for e in said["entries"]}
    for i, m in enumerate(moves): by_slot[i]["act"] = m.get("act", "")
    scene["log"].extend(said["entries"]); scene["sid"] += len(cast)
    texts = [animate_input(prompt.split("\n")[0], said["recs"][i], {"sid": said["sids"][i], "who": "I", "text": m["text"], "act": m.get("act", "")},
                           action_text(i, m.get("act", ""), at, names, ends))
             if m["kind"] == "speak" else action_text(i, m.get("act", ""), at, names, ends)
             for i, m in enumerate(moves)]
    t1 = time.time()
    job = backend.animate([{"character": c["character"], "push": c["push"], "dose": c["dose"], "text": t} for c, t in zip(cast, texts)])
    n = len(scene["steps"]) + 1
    members = []
    for k, (c, m, jm) in enumerate(zip(cast, moves, job.get("items") or [{}] * len(cast))):
        ok = jm.get("stage") == "done"
        members.append({"id": c.get("id", c["character"]), "character": c["character"], "name": c["name"], "push": c["push"], "dose": c["dose"],
                        "kind": m["kind"], "says": m.get("text", "") if m["kind"] == "speak" else "",
                        "to": m.get("to") or [], "act": m.get("act", ""), "feel": m.get("feel"),
                        "start": at[k], "end": ends[k] if ok else at[k], "stage": jm.get("stage"),
                        "error": (jm.get("error") or "")[-300:] or None,
                        "clip": clip_url(n, k, jm) if ok else None, "frames": jm.get("frames"), "fps": jm.get("fps"),
                        "end_yaw": jm.get("end_yaw"),
                        # the program's beats (verb, start and end frame), for the page's timeline lanes, and
                        # when this one moved in the step (the speakers' lines play in this order)
                        "timeline": jm.get("timeline") if ok else None, "order": said["order"].index(k),
                        "checks_failed": [x for x, v in (jm.get("checks") or {}).items() if not v.get("pass")]})
        if ok: c["pos"], c["yaw"] = list(ends[k]["pos"]), ends[k]["yaw"]
    s = {"n": n, "prompt": prompt, "members": members, "t_say": said["seconds"],
         "t_animate": round(time.time() - t1, 1), "t_step": round(time.time() - t0, 1),
         "frames": max([m["frames"] or 0 for m in members if m["clip"]] or [0]), "at": time.time()}
    scene["steps"].append(s)
    # seconds of animation delivered, summed over the cast (what a free allowance counts)
    scene["anim_s"] = round(scene.get("anim_s", 0.0) + sum(m["frames"] / (m["fps"] or 24) for m in members if m["clip"]), 1)
    return s


def public(scene, since=0):
    """What a caller sees: the cast, the state, and the steps after `since`."""
    out = {k: scene[k] for k in ("id", "situation", "state", "cast")}
    out.update(anim_s=scene.get("anim_s", 0.0), steps=[s for s in scene["steps"] if s["n"] > since], step_count=len(scene["steps"]))
    return out
