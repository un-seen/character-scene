"""The program format as a grammar, for constrained decoding.

A persona push acts on every token the model writes, so on its own it can push the model off
the format: prose instead of JSON, made-up style keys, a program that never closes. Decoding
under this grammar leaves the push only the choices the format allows (which verbs, which
style dials, which numbers), so every pushed program parses.

The grammar follows the training programs exactly: compact separators, the fixed key order,
style dials in any order. Every repetition and every number is bounded, so a program always
ends within the token budget. Beats are numbered b0, b1, … in order and `after` can name
only an earlier one. Ranges are left to the validator, except that numbers a push drives past
their stops are clamped to them (`clamp`).
"""
from . import world
from .program import GAIT, SHAPES, VERBS

MAX_BEATS, MAX_KEYS, MAX_STYLE = 8, 8, 24
# The moods the model was trained to name in its plan (tools/dataset.py KINDS, and neutral).
PLAN_MOODS = ["sleepy", "sneaky", "exhausted", "sad", "confident", "nervous", "angry", "happy", "proud"]


def lit(text):
    """An EBNF literal matching the JSON text exactly."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def one_of(names):
    """A JSON string that is one of names."""
    return "(" + " | ".join(lit(f'"{n}"') for n in names) + ")"


def opt(text, rule):
    return f"({lit(text)} {rule})?"


ARGS = {
    "walk": f'"{{" {opt("\"distance\":", "num")} {opt(",\"speed\":", "num")} "}}"',
    "walk_to": f'{lit("{\"target\":")} vec2 {opt(",\"speed\":", "num")} "}}"',
    "run": f'"{{" {opt("\"distance\":", "num")} {opt(",\"speed\":", "num")} "}}"',
    "run_to": f'{lit("{\"target\":")} vec2 {opt(",\"speed\":", "num")} "}}"',
    "jump": f'"{{" {opt("\"height\":", "num")} {opt(",\"distance\":", "num")} "}}"',
    "turn": f'{lit("{\"deg\":")} num "}}"',
    "idle": lit("{}"), "crouch": lit("{}"), "kneel": lit("{}"), "sit": lit("{}"), "lie": lit("{}"),
    "look_at": f'{lit("{\"target\":")} vec3 {opt(",\"amount\":", "num")} "}}"',
    "express": (f'{lit("{\"keys\":{")} face ("," face){{0,4}} "}}" '
                f'{opt(",\"onset\":", "int")} {opt(",\"apex\":", "int")} {opt(",\"offset\":", "int")} "}}"'),
    "hand": (f'{lit("{\"side\":")} {one_of(["L", "R", "both"])} {lit(",\"shape\":")} {one_of(sorted(s for s in SHAPES if world.allows(f"hand:{s}")))} '
             f'{opt(",\"amount\":", "num")} "}}"'),
    "reach": (f'{lit("{\"side\":")} {one_of(["L", "R"])} {lit(",\"target\":")} vec3 '
              f'{opt(",\"hold\":", "int")} "}}"'),
}


def ebnf(reg):
    controls = " | ".join(f'{lit(chr(34) + k + chr(34) + ",\"ch\":")} {one_of(c["channels"])}'
                          for k, c in reg["controls"].items())
    # Beat i is "b<i>" and may follow only an earlier beat; the list nests so ids stay in order.
    tail = ""
    for i in reversed(range(1, MAX_BEATS)):
        tail = f'("," beat{i} {tail})?' if tail else f'("," beat{i})?'
    rules = {
        "root": (f'{lit("{\"plan\":")} plan {lit(",\"clip\":")} clip {lit(",\"style\":")} style '
                 f'{lit(",\"beats\":[")} beats "]" '
                 f'({lit(",\"keys\":[")} key ("," key){{0,{MAX_KEYS - 1}}} "]")? "}}"'),
        # "<mood> at <strength>: <verb>; <verb>" or "neutral: <verb>; ...", as trained.
        "plan": f'"\\"" mood ": " planverb ("; " planverb){{0,{MAX_BEATS - 1}}} "\\""',
        "mood": '"neutral" | (' + " | ".join(lit(m) for m in PLAN_MOODS) + ') " at " strength',
        "strength": '"0" ("." [0-9]{1,2})? | "1" ("." "0"{1,2})?',
        "planverb": " | ".join(lit(v) for v in sorted(VERBS)),
        "clip": (f'{lit("{\"fps\":")} ("24" | "25" | "30") {lit(",\"frames\":")} int '
                 f'{lit(",\"seed\":")} int {lit(",\"variation\":")} num "}}"'),
        "style": f'"{{}}" | "{{" dial ("," dial){{0,{MAX_STYLE - 1}}} "}}"',
        "dial": f'{one_of(GAIT)} ":" num',
        "beats": f"beat0 {tail}",
        **{f"beat{i}": f'{lit(f"{{\"id\":\"b{i}\",\"verb\":")} verbargs timing{i} "}}"' for i in range(MAX_BEATS)},
        "timing0": f'{lit(",\"start\":")} int dur',
        **{f"timing{i}": f'({lit(",\"start\":")} int | {lit(",\"after\":")} {one_of(f"b{j}" for j in range(i))}) dur'
           for i in range(1, MAX_BEATS)},
        "dur": opt(",\"dur\":", "int"),
        "verbargs": " | ".join(f'{lit(chr(34) + v + chr(34) + ",\"args\":")} args_{v}' for v in ARGS),
        **{f"args_{v}": g for v, g in ARGS.items()},
        "face": f'{one_of(reg["face"])} ":" num',
        "key": (f'{lit("{\"ctrl\":")} keyctrl {lit(",\"f\":")} int {lit(",\"v\":")} keyv '
                f'({lit(",\"hold\":")} ("true" | "false"))? "}}"'),
        # The channel follows from the control's kind; the validator checks the value's shape.
        "keyctrl": (f'{controls} | {one_of(reg["digits"])} {lit(",\"ch\":")} ("\\"curl\\"" | "\\"spread\\"") '
                    f'| {one_of(reg["face"])} {lit(",\"ch\":\"value\"")}'),
        "keyv": "num | vec3",
        "vec2": '"[" num "," num "]"',
        "vec3": '"[" num "," num "," num "]"',
        "int": '"0" | [1-9] [0-9]{0,3}',
        "num": '"-"? ("0" | [1-9] [0-9]{0,3}) ("." [0-9]{1,3})?',
    }
    return "".join(f"{k} ::= {v}\n" for k, v in rules.items())


# Ranges a push can drive a number past, as the validator states them.
ARG_RANGES = {"distance": (0.2, 50), "speed": (0.3, 1.8), "deg": (-180, 180), "amount": (0, 1), "hold": (0, 240)}


def clamp(prog):
    """Clamp numbers past their stops (style dials, clip variation, beat args, face keys);
    return what was moved, with the old values."""
    moved = {}
    def fix(where, d, k, lo, hi):
        v = d[k]
        if isinstance(v, (int, float)) and not lo <= v <= hi:
            d[k] = min(max(v, lo), hi); moved[where] = v
    for k in list(prog.get("style") or {}):
        fix(f"style.{k}", prog["style"], k, *GAIT[k][:2])
    if "variation" in prog.get("clip", {}): fix("clip.variation", prog["clip"], "variation", 0, 1)
    for b in prog.get("beats", []):
        a = b.get("args") or {}
        for k, (lo, hi) in ARG_RANGES.items():
            if k in a: fix(f"{b['id']}.{k}", a, k, lo, hi)
        for k in list(a["keys"] if isinstance(a.get("keys"), dict) else {}):   # an express's face (a pose's keys are a list)
            fix(f"{b['id']}.keys.{k}", a["keys"], k, 0, 1)
    return moved
