"""The program as qwen-animate writes it (v6): one line per part, the same values as the JSON.

    happy at 0.31: turn; look_at
    24 52 109 0.3
    style speed_ms=0.9 cadence_spm=101 arm_swing=1.09
    turn deg=-90 @0
    look_at target=2.8,0,1.55 +b0 36
    key head rot 12 0,0,18 hold

Lines: the plan; fps, frames, seed and variation; the style dials ("style" alone when there are
none); one line per beat, its verb, its args as name=value, "@<frame>" or "+<earlier beat id>"
and its duration if it has one; then the sparse keys. A beat's id is its place (b0, b1, …).

It takes about half the tokens of the JSON for the same program (pack_probe.py), and decode
gives the JSON program back exactly: encode/decode only change how the values are written. The
style dials are written in GAIT's order, each at most once, which the grammar holds.
"""
import json

from . import poseprog, world
from .program import GAIT, POSE_KEYS, SHAPES

MAX_BEATS, MAX_KEYS = 8, 8
PLAN_MOODS = ["sleepy", "sneaky", "exhausted", "sad", "confident", "nervous", "angry", "happy", "proud"]
# each verb's args in the order they are written, and what kind each value is
ARGS = {"walk": [("distance", "num"), ("speed", "num")], "walk_to": [("target", "vec"), ("speed", "num")],
        "run": [("distance", "num"), ("speed", "num")], "run_to": [("target", "vec"), ("speed", "num")],
        "jump": [("height", "num"), ("distance", "num")],
        "crouch": [], "kneel": [], "sit": [], "lie": [], "pose": [],
        "turn": [("deg", "num")], "idle": [], "look_at": [("target", "vec"), ("amount", "num")],
        "express": [("keys", "face"), ("onset", "int"), ("apex", "int"), ("offset", "int")],
        "hand": [("side", "word"), ("shape", "word"), ("amount", "num")],
        "reach": [("side", "word"), ("target", "vec"), ("hold", "int")]}
REQUIRED = {"walk_to": {"target"}, "run_to": {"target"}, "turn": {"deg"}, "look_at": {"target"}, "express": {"keys"},
            "hand": {"side", "shape"}, "reach": {"side", "target"}}


def _n(v):
    return json.dumps(v)


def _val(kind, v):
    if kind == "vec": return ",".join(_n(x) for x in v)
    if kind == "face": return ",".join(f"{k}:{_n(x)}" for k, x in v.items())
    if kind == "word": return v
    return _n(v)


def encode(prog):
    c = prog["clip"]
    out = [prog["plan"], f'{c["fps"]} {c["frames"]} {c["seed"]} {_n(c["variation"])}',
           " ".join(["style"] + [f"{k}={_n(prog['style'][k])}" for k in GAIT if k in (prog.get("style") or {})])]
    for b in prog["beats"]:
        args = b.get("args") or {}
        parts = [b["verb"]] + [f"{k}={_val(kind, args[k])}" for k, kind in ARGS[b["verb"]] if k in args]
        parts.append(f'@{b["start"]}' if "start" in b else f'+{b["after"]}')
        if "dur" in b: parts.append(str(b["dur"]))
        out.append(" ".join(parts))
        for k in args.get("keys", []) if b["verb"] == "pose" else []:   # a pose's keys, one line each
            if "spec" in k: out.append(poseprog.render_key(k)); continue  # in body words (v13 on)
            out.append(f' at {k["at"]} ' + "; ".join(f"{c} {n} {_val('vec', v)}" for c, chs in k["set"].items() for n, v in chs.items()))
    for k in prog.get("keys") or []:
        v = ",".join(_n(x) for x in k["v"]) if isinstance(k["v"], list) else _n(k["v"])
        out.append(" ".join(["key", k["ctrl"], k["ch"], str(k["f"]), v] + ({True: ["hold"], False: ["free"]}[k["hold"]] if "hold" in k else [])))
    return "\n".join(out)


def decode(text):
    lines = [l for l in text.strip().split("\n")]
    fps, frames, seed, var = lines[1].split()
    style = {}
    for kv in lines[2].split()[1:]:
        k, v = kv.split("=", 1); style[k] = json.loads(v)
    prog = {"plan": lines[0], "clip": {"fps": int(fps), "frames": int(frames), "seed": int(seed), "variation": json.loads(var)},
            "style": style, "beats": []}
    keys = []
    for line in lines[3:]:
        w = line.split()
        if w[0] == "at" and len(w) > 2 and w[2] in poseprog.PARTS:   # a key in body words
            prog["beats"][-1]["args"].setdefault("keys", []).append(poseprog.parse_key(line)); continue
        if w[0] == "at":                                   # a key pose of the pose beat above
            st = {}
            for part in line.split(None, 2)[2].split("; "):
                c, n, v = part.split()
                st.setdefault(c, {})[n] = [json.loads(x) for x in v.split(",")]
            prog["beats"][-1]["args"].setdefault("keys", []).append({"at": int(w[1]), "set": st})
            continue
        if w[0] == "key":
            v = [json.loads(x) for x in w[4].split(",")] if "," in w[4] else json.loads(w[4])
            k = {"ctrl": w[1], "ch": w[2], "f": int(w[3]), "v": v}
            if len(w) > 5: k["hold"] = w[5] == "hold"
            keys.append(k); continue
        verb, kinds = w[0], dict(ARGS[w[0]])
        b = {"id": f'b{len(prog["beats"])}', "verb": verb, "args": {}}
        for tok in w[1:]:
            if "=" in tok:
                k, v = tok.split("=", 1); kind = kinds[k]
                if kind == "vec": v = [json.loads(x) for x in v.split(",")]
                elif kind == "face": v = {f.split(":")[0]: json.loads(f.split(":")[1]) for f in v.split(",")}
                elif kind != "word": v = json.loads(v)
                b["args"][k] = v
            elif tok[0] == "@": b["start"] = int(tok[1:])
            elif tok[0] == "+": b["after"] = tok[1:]
            else: b["dur"] = int(tok)
        prog["beats"].append(b)
    if keys: prog["keys"] = keys
    return prog


def lit(text):
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def ebnf(reg):
    """The format as a grammar for constrained decoding, bounded as grammar.ebnf is."""
    words = lambda names: "(" + " | ".join(lit(n) for n in names) + ")"
    tail = ""
    for i in reversed(range(1, MAX_BEATS)):
        tail = f'("\\n" beat{i} {tail})?' if tail else f'("\\n" beat{i})?'
    val = {"num": "num", "int": "int", "vec": None, "face": 'face ("," face){0,4}'}
    def args(verb):
        out = []
        for k, kind in ARGS[verb]:
            if kind == "word":
                v = words(["L", "R", "both"] if verb == "hand" else ["L", "R"]) if k == "side" else words(sorted(s for s in SHAPES if world.allows(f"hand:{s}")))
            elif kind == "vec":
                v = "vec2" if verb in ("walk_to", "run_to") else "vec3"
            else:
                v = val[kind]
            part = f'{lit(" " + k + "=")} {v}'
            out.append(part if k in REQUIRED.get(verb, ()) else f"({part})?")
        return " ".join(out) if out else '""'
    controls = " | ".join(f'{lit(k + " ")} {words(c["channels"])}' for k, c in reg["controls"].items())
    rules = {
        "root": f'plan "\\n" clip "\\n" style "\\n" beats (keyline){{0,{MAX_KEYS}}}',
        "plan": f'mood ": " planverb ("; " planverb){{0,{MAX_BEATS - 1}}}',
        "mood": '"neutral" | (' + " | ".join(lit(m) for m in PLAN_MOODS) + ') " at " strength',
        "strength": '"0" ("." [0-9]{1,2})? | "1" ("." "0"{1,2})?',
        "planverb": words(sorted(ARGS)),
        "clip": '("24" | "25" | "30") " " int " " int " " num',
        # every dial in GAIT's order, each at most once
        "style": '"style" ' + " ".join(f'({lit(" " + k + "=")} num)?' for k in GAIT),
        "beats": f"beat0 {tail}",
        **{f"beat{i}": f'(verbline timing{i} dur | "pose" timing{i} dur poselines)' for i in range(MAX_BEATS)},
        # a pose's keys in body words (poseprog.py): the model says what each part does, the solver the angles
        **poseprog.ebnf_rules(),
        "timing0": '" @" int',
        **{f"timing{i}": f'(" @" int | " +" {words(f"b{j}" for j in range(i))})' for i in range(1, MAX_BEATS)},
        "dur": '(" " int)?',
        "verbline": " | ".join(f"{lit(v)} {args(v)}" for v in ARGS if v != "pose"),
        "face": f'{words(reg["face"])} ":" num',
        "keyline": f'"\\nkey " keyctrl " " int " " keyv (" hold" | " free")?',
        "keyctrl": (f'{controls} | {words(reg["digits"])} (" curl" | " spread") | {words(reg["face"])} " value"'),
        "keyv": "num | vec3",
        "vec2": 'num "," num',
        "vec3": 'num "," num "," num',
        "int": '"0" | [1-9] [0-9]{0,3}',
        "num": '"-"? ("0" | [1-9] [0-9]{0,3}) ("." [0-9]{1,3})?',
    }
    return "".join(f"{k} ::= {v}\n" for k, v in rules.items())
