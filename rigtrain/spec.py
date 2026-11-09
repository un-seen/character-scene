"""The system prompt: a compact spec of the rigctl program format, built from the code so it
never drifts from what the validator accepts."""
import json, os, sys
sys.path.insert(0, os.path.expanduser("~/character"))
from rigctl import program as P


def system_prompt():
    style = ", ".join(f"{k} {lo}..{hi} (neutral {n})" for k, (lo, hi, n) in P.GAIT.items())
    return f"""You write animation programs for the character as compact JSON, nothing else.
Format: {{"plan": str, "clip": {{"fps": 24, "frames": int, "seed": int, "variation": 0.3}}, "style": {{param: value}}, "beats": [...], "keys": [...]}}
Write "plan" first: the attitude and strength, then the verbs in order, e.g. "sleepy at 0.6: walk; idle".
"style" is the character's attitude for the whole clip; give only parameters that differ from neutral. Parameters: {style}.
Beats: {{"id": "b0", "verb": V, "start": 0 | "after": "b<prev>", "dur": frames?, "args": {{...}}}}. Verbs:
 walk {{distance m}}; walk_to {{target [x, y]}}; run {{distance m, speed 2.5..6}}; run_to {{target [x, y], speed}};
 turn {{deg (+ left)}}; idle {{}} with dur;
 look_at {{target [x, y, z]}} with dur; express {{keys {{smile|frown|brows_up|jaw_open|blink: 0..1}}, onset, offset}} with dur;
 hand {{side L|R|both, shape relaxed|fist|point|open|grip}} with dur; reach {{side L|R, target [x, y, z], hold}} with dur.
World: metres, start at [0, 0] facing -Y, +X is the character's left, the floor is z = 0.
Keys (optional, sparse): {{"ctrl": name, "ch": "rot"|"loc"|"curl"|"spread"|"value", "f": frame, "v": value, "mode": "abs"|"add", "hold": bool}}.
 head rot [nod down +, turn left +, tilt right +] degrees. Face keys and digits (thumb, f_index, f_middle, f_ring, f_pinky + .L/.R) take scalars.
"clip.frames" must cover every beat plus about one second."""


SHORT = ("You write rigctl animation programs for the character as compact JSON, nothing else. "
         "Write plan first, then clip, style (only what differs from neutral), beats and keys.")


CARDS = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cards.json")))


def card(ch):
    """The body a program is written for, in one line: the heights a look or a reach aims at,
    the arm it reaches with, and whether it has a face to express with."""
    c = CARDS[ch]
    face = ", ".join(c["face"]) if c["face"] else "none: no express beats, no face keys or face dials"
    return (f"Body: {c['height_m']:.2f} m tall; eyes {c['eye_m']:.2f} m, shoulders {c['shoulder_m']:.2f} m, "
            f"chest {c['chest_m']:.2f} m, waist {c['waist_m']:.2f} m; arm {c['arm_m']:.2f} m; leg {c['leg_m']:.2f} m. Face keys: {face}.")


def short_prompt(ch=None):
    """What the fine-tuned model is trained and run with: it has learned the format itself.
    With a character, the prompt carries its body card (cards.json); without one it is the
    one-character prompt the v2 adapter was trained on."""
    if ch is None: return SHORT
    return ("You write rigctl animation programs for one character as compact JSON, nothing else. " + card(ch) +
            " Write plan first, then clip, style (only what differs from neutral), beats and keys.")
