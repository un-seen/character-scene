"""qwen-say: one character's next move in a conversation, as a line of the record.

It is the base Qwen3.5-4B with the character's push (no adapter, no chat template, no system
prompt), continuing the record under the grammar below; the page asks the cast one at a time.

Each step every character moves once, one after another, each hearing the moves made before it
(`order`). A move either speaks or listens:

    [s7 Red to Blue and Green, next Blue] "You said you'd be here at noon." (walks to Blue, then frowns) {stress noon; point Blue at you}
    [s8 Green listens to Red] {doubtful 0.6}

Speaking names whom it is said to and the one who should answer (one of them), the line (a bubble:
at most MAX_WORDS words), two to four actions in qwen-animate's phrases, then up to three marks
tied to words of the line: a stressed word, or a gesture on a word (pointing at someone, a beat,
open hands, a fist). Listening names whom it listens to (one who spoke this step) and how it takes
what was said: a feeling and how strongly. The first move of a step always speaks, so no step is
silent; the others may listen, and in a group most of them should.

Each character is shown the record from its own side ("I" for its own moves, "me" for itself
where it is spoken to or listened to). The marks are not in the record: what was stressed is in
the line, and the page shows the gestures.

The page applies the floor (`speak`): the step's first move and the one it was said to may speak;
anyone else may only listen, but for an interruption now and then. Left to choose, a model (the
base one and the 9B teacher of data_say7 alike) has everyone talk every step.
"""
import math
import os
import re
import sys

sys.path.insert(0, os.path.expanduser("~/character"))
from rigctl import world  # noqa: E402

AWARENESS = 8                                          # moves of the record shown
WOKE = "I woke up."
CANDIDATES = 4                                         # moves written per character; `choose` keeps one
MAX_WORDS = 20
TOWARD = ["walks to", "turns to", "looks at", "points at", "reaches towards"]
# the ones qwen-say may choose: the page turns bodies to whom they talk with (faceTalk) and moves
# eyes and heads to them, so a turn or a look of its own only spent an action on what happens
# anyway (the base model chose little else, and bodies stood still)
# and following and leading, which the page turns into walks qwen-animate knows (one who is led
# follows: it walks after the leader, who walks off away from it), and running to someone
# (qwen-animate v7's run_to; the page has it walk when they are too near to run)
SAY_TOWARD = ["walks to", "runs to", "points at", "reaches towards", "follows", "leads"]
# and, with someone there, not turning its back on them mid-line only to be turned back (faceTalk),
# nor a few steps straight ahead, which is into whom it faces ("walks to" stops short of them)
TALKING = {"turns around", "walks a few steps"}
# the actions qwen-animate's training prompts use, word for word (as say.py)
# and the gestures it learned as key poses (v11's pose beat): offered, never asked for
BODY = ["clenches their fists", "opens their hands", "turns around", "walks a few steps", "jumps",
        "crouches", "kneels", "sits down", "lies down",
        "waves", "waves hello", "bows", "gives a small bow", "nods", "shakes their head", "shrugs",
        "puts their hands up", "stretches"]
ALONE = BODY + ["smiles", "gives a small smile", "frowns", "looks surprised", "looks worried", "yawns",
                "waits for a second", "stands for a moment"]
WALKS = ("walks to", "runs to", "walks a few steps", "follows", "leads")   # at most one per line
GESTURES = ["beat", "open", "fist"]                    # and "point NAME at WORD"
FEELINGS = ["agrees", "disagrees", "amused", "doubtful", "surprised", "worried", "annoyed", "interested", "bored"]


def time_line(t):
    t = (t or "").strip().rstrip(".")
    return (t if t.lower().startswith("time is") else f"Time is {t}") + "."


def entry_text(e, me):
    """A move as character `me` reads it in the record. e: {sid, who, kind, to, next, text, act,
    feel, amount} with names (who/to/next) as in the cast."""
    who = "I" if e["who"] == me else e["who"]
    nm = lambda n: "me" if n == me else n
    if e["kind"] == "listen":
        return f"[s{e['sid']} {who} listens to {nm(e['to'][0])}] {{{e['feel']} {e['amount']}}}"
    if not e["to"]: return f'[s{e["sid"]} {who}] "{e["text"]}" ({e["act"]})'          # alone: said to no one
    return f'[s{e["sid"]} {who} to {" and ".join(map(nm, e["to"]))}, next {nm(e["next"])}] "{e["text"]}" ({e["act"]})'


def context(time, entries, sid, me):
    """What qwen-say continues for `me`: the time, the situation, its name, "I woke up" and the
    record's last AWARENESS moves, then its own move, open after its id. time is the page's
    prompt: its first line the time, any lines after it the situation ("We are sitting in a
    quiet café.")."""
    t, _, scene = (time or "").strip().partition("\n")
    head = [time_line(t)] + [x.strip() for x in scene.split("\n") if x.strip()]
    lines = [WOKE] + [entry_text(e, me) for e in entries]
    return "\n".join(head + [f"I am {me}."] + lines[-AWARENESS:] + [f"[s{sid} I"])


SPEECH = "[A-Za-z'] ([A-Za-z'-] | [,;:.!?] \" \" [A-Za-z'] | \" \" [A-Za-z'])* [.!?]?"
WORD = "[A-Za-z'] [A-Za-z'-]*"


def lit(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


AVOID_LINES = 2


def last_actions(record, me):
    """The actions of `me`'s newest AVOID_LINES spoken lines (a walk or a point as its verb), for
    grammar to leave out: the base model copies its own last actions into every line it writes
    after, and with only the newest left out it went back and forth between two sets."""
    acts = [e["act"] for e in reversed(record) if e["who"] == me and e["kind"] == "speak"][:AVOID_LINES]
    out = set()
    for a in (x for act in acts if act for x in act.split(", then ")):
        out.add(next((t for t in SAY_TOWARD if a.startswith(t + " ")), a))
    return out


def grammar(others, heard, speak=True, avoid=()):
    """The rest of the open move. others: the other characters' names; heard: those who spoke
    this step, whom it may listen to (empty for the step's first move, which must speak);
    speak: whether it may speak (the teacher's scenes hold most of a group to listening); avoid:
    actions (a toward verb or a whole action) it may not take this time (last_actions). A
    character alone says its line to no one: `] "words" (actions)`, its actions its own."""
    alt = lambda xs: "(" + " | ".join(lit(x) for x in sorted(xs)) + ")"   # sorted: one grammar per set of names
    avoid = set(avoid) | (TALKING if others else set())
    ok = lambda a: a not in avoid and world.allows(a, bool(others))     # and the world has what it needs
    ALONE_, BODY_ = [a for a in ALONE if ok(a)], [a for a in BODY if ok(a)]
    TOWARD_ = [t for t in SAY_TOWARD if ok(t)]
    if not others:
        return (f'root ::= "] \\"" {SPEECH} "\\" (" act ")" marks?\n'
                f'act ::= {alt(BODY_)} (", then " {alt(ALONE_)}){{1,3}}\n'
                f'marks ::= " {{" mark ("; " mark){{0,2}} "}}"\n'
                f'mark ::= "stress " {WORD} | {alt(GESTURES)} " at " {WORD}\n')
    g = {
        "root": " | ".join((["speak"] if speak or not heard else []) + (["listen"] if heard else [])),
        # the line: spoken words only, one space and at most one mark between them. It is written out
        # here, not as rules of its own: as words and gaps xgrammar takes ~60 ms a token to mask
        # (each space could end a word or start a gap), and as a rule of its own ~2 ms; inline,
        # 0.03 ms. It is unbounded, as bounded repeats are slow to compile; `problems` holds a line
        # to MAX_WORDS.
        "speak": f'" to " who (" and " who){{0,{len(others) - 1}}} ", next " who "] \\"" {SPEECH} "\\" (" act ")" marks?',
        "act": 'first (", then " action){1,3}',
        "first": " | ".join(([f'{alt([t + " " for t in TOWARD_])} who'] if TOWARD_ else []) + [alt(BODY_)]),
        "action": " | ".join(([f'{alt([t + " " for t in TOWARD_])} who'] if TOWARD_ else []) + [alt(ALONE_)]),
        "marks": '" {" mark ("; " mark){0,2} "}"',
        "mark": f'"stress " {WORD} | "point " who " at " {WORD} | {alt(GESTURES)} " at " {WORD}',
        "who": alt(others),
    }
    if heard:
        g["listen"] = f'" listens to " {alt(heard)} "] {{" {alt(FEELINGS)} " 0." [1-9] "}}"'
    return "".join(f"{k} ::= {v}\n" for k, v in g.items())


ALONE_LINE = re.compile(r'^\] "(.*)" \((.*?)\)(?: \{(.*)\})?$', re.S)
SPEAK = re.compile(r'^ to (.+?), next (\w+)\] "(.*)" \((.*?)\)(?: \{(.*)\})?$', re.S)
LISTEN = re.compile(r"^ listens to (\w+)\] \{(\w+) (0\.\d)\}$")


def parse(raw):
    """The move from its continuation (after "[sN I"): {kind, to, next, text, act, marks} or
    {kind, to, feel, amount}. Raises ValueError when it is not one. The served text can come back
    without its first space (vsteer's detokenizing), so it is put back."""
    raw = " " + (raw or "").strip()
    m = ALONE_LINE.match(raw[1:])
    if m: return _speak([], None, m.group(1), m.group(2), m.group(3))
    m = LISTEN.match(raw)
    if m:
        return {"kind": "listen", "to": [m.group(1)], "feel": m.group(2), "amount": float(m.group(3))}
    m = SPEAK.match(raw)
    if not m: raise ValueError(f"not a move: {raw[:120]!r}")
    to = []
    for n in m.group(1).split(" and "):
        if n not in to: to.append(n)
    return _speak(to, m.group(2), m.group(3), m.group(4), m.group(5))


def _speak(to, nxt, text, act, marks_text):
    marks = []
    for x in (marks_text or "").split("; "):
        w = x.split()
        if not w: continue
        if w[0] == "stress": marks.append({"mark": "stress", "word": w[1]})
        elif w[0] == "point": marks.append({"mark": "point", "target": w[1], "word": w[3]})
        else: marks.append({"mark": w[0], "word": w[2]})
    acts = []
    for x in act.split(", then "):
        if x not in acts: acts.append(x)
    return {"kind": "speak", "to": to, "next": nxt, "text": text.strip(), "act": ", then ".join(acts), "marks": marks}


def continuation(e):
    """The move written as it continues "[sN I" (the inverse of parse)."""
    if e["kind"] == "listen":
        return f" listens to {e['to'][0]}] {{{e['feel']} {e['amount']}}}"
    out = (f' to {" and ".join(e["to"])}, next {e["next"]}]' if e["to"] else "]") + f' "{e["text"]}" ({e["act"]})'
    marks = [f"stress {k['word']}" if k["mark"] == "stress" else f"point {k['target']} at {k['word']}" if k["mark"] == "point"
             else f"{k['mark']} at {k['word']}" for k in e["marks"]]
    return out + (" {" + "; ".join(marks) + "}" if marks else "")


def effect(e, cont, toks):
    """What the push did to move e, from its continuation and the push's token scores
    (vsteer.score). Per word of the line, per action and for a listener's feeling: d, the log of
    how many times likelier the push made it (its tokens' log probabilities with the push less
    without), and p and q, its probability with and without. strain is how surprising the line's
    words (a listener's feeling) are to the model without the push (nats a token), own the same
    with it."""
    if not toks: return None
    spans = []
    if e["kind"] == "listen":
        i = cont.index("{") + 1; spans.append(("feel", cont[i:cont.index(" ", i)], i, cont.index(" ", i)))
    else:
        i = cont.index('"') + 1; j = cont.index('"', i)
        spans += [("words", m.group(), i + m.start(), i + m.end()) for m in re.finditer(r"\S+", cont[i:j])]
        a0 = cont.index("(", j) + 1; a1 = cont.index(")", a0); pos = a0
        for act in cont[a0:a1].split(", then "):
            spans.append(("acts", act, pos, pos + len(act))); pos += len(act) + len(", then ")
    start = lambda t: t[0] + len(cont[t[0]:t[1]]) - len(cont[t[0]:t[1]].lstrip())   # its first character that is not a space
    out, said = {"words": [], "acts": [], "feel": None}, []
    for kind, label, s0, s1 in spans:
        mine = [t for t in toks if s0 <= start(t) < s1]
        if not mine: continue
        lp, lq = sum(t[2] for t in mine), sum(t[3] for t in mine)
        x = {"w": label, "d": round(lp - lq, 2), "p": round(math.exp(lp), 4), "q": round(math.exp(lq), 4)}
        if kind == "feel": out["feel"] = x
        else: out[kind].append(x)
        if kind in ("words", "feel"): said += mine
    if said:
        out["strain"] = round(-sum(t[3] for t in said) / len(said), 2)
        out["own"] = round(-sum(t[2] for t in said) / len(said), 2)
    return out


# why grammar leaves an option out, as the card says it
DROPPED = {"turns to": "the page turns bodies itself", "looks at": "the page moves eyes and heads itself"}
WHY = {"avoid": f"it did that in its last {AVOID_LINES} lines", "talking": "not while talking to someone",
       "first": "not as the first action", "floor": "not its turn to speak", "first_move": "the step's first move speaks",
       "unheard": "has not spoken this step", "me": "its own name"}
SHOWN = 8                                              # top tokens kept per choice, of those over 0.5%


def slots(e, cont, me, others, heard, speak=True, avoid=()):
    """Where move e (written as cont) chose among what grammar offers: [(name, offset in cont,
    the option chosen, {option text: None if allowed, else why it is left out})]."""
    out = []
    if e["kind"] == "listen":
        out.append(("kind", 0, "listens", {" to ": None if speak else WHY["floor"], " listens to ": None}))
        i = len(" listens to ")
        out.append(("to", i, e["to"][0], {n: None if n in heard else WHY["unheard"] for n in others} | {me: WHY["me"]}))
        i = cont.index("{") + 1
        out.append(("feeling", i, e["feel"], {f: None for f in FEELINGS}))
        return out
    if others:
        out.append(("kind", 0, "speaks", {" to ": None, " listens to ": None if heard else WHY["first_move"]}))
        out.append(("to", len(" to "), e["to"][0], {n: None for n in others} | {me: WHY["me"]}))
    avoid = set(avoid)
    talking = TALKING if others else set()
    a0 = cont.index("(", cont.index('"', cont.index('"') + 1)) + 1; pos = a0
    for k, act in enumerate(cont[a0:cont.index(")", a0)].split(", then ")):
        opts = {}
        for t in TOWARD + [t for t in SAY_TOWARD if t not in TOWARD]:
            if not others: continue
            opts[t + " "] = DROPPED.get(t) or (WHY["avoid"] if t in avoid else None)
        for a in ALONE:
            opts[a] = (WHY["avoid"] if a in avoid else WHY["talking"] if a in talking
                       else WHY["first"] if k == 0 and a not in BODY else None)
        out.append((f"action {k + 1}", pos, act, opts)); pos += len(act) + len(", then ")
    return out


def choices(e, cont, sc, me, others, heard, speak=True, avoid=()):
    """What the model wanted where grammar gave move e a choice (slots), from the tokens it found
    likeliest there (vsteer.score's top). Per choice: its name, the option chosen, kept (how much
    of the model's probability went to tokens grammar allows), share (the chosen token's part of
    that: its chance under grammar, before temperature) and the likeliest tokens and the chosen
    one, each {t, p, q, why, of, chosen}: its text, its probability as served and without the
    push, None if grammar allows it, why it leaves it out, or "not in the grammar", and the
    options it begins."""
    out = []
    for name, at, chose, opts in slots(e, cont, me, others, heard, speak, avoid):
        i = next((i for i, (s0, s1) in enumerate(sc["offs"]) if s0 <= at < s1 and not cont[s0:at].strip(" (\"{")), None)
        if i is None: continue
        lead = cont[sc["offs"][i][0]:at]
        read = cont[sc["offs"][i][0]:sc["offs"][i][1]]
        top = []
        for t, lp, lq in sc["top"][i]:
            hits = {o: w for o, w in opts.items() if t.strip() and ((lead + o).startswith(t) or t.startswith(lead + o))}
            why = "not in the grammar" if not hits else None if None in hits.values() else next(iter(hits.values()))
            top.append({"t": t, "p": math.exp(lp), "q": None if lq is None else math.exp(lq), "why": why,
                        "of": [o.strip() for o, w in hits.items() if w == why][:4], "chosen": t == read})
        kept = sum(x["p"] for x in top if x["why"] is None)
        mine = next((x["p"] for x in top if x["chosen"]), None)
        shown = [x | {"p": round(x["p"], 4), "q": None if x["q"] is None else round(x["q"], 4)}
                 for k, x in enumerate(top) if x["chosen"] or (k < SHOWN and x["p"] >= 0.005)]
        out.append({"slot": name, "chose": chose, "kept": round(kept, 4),
                    "share": round(mine / kept, 4) if mine and kept else None, "top": shown})
    return out


def neurons(e, cont, sc, names, pushed):
    """The catalog's persona neurons on move e (vsteer.score's trace), from its line (a listener's
    feeling) to its end: {names, toks: each token's text, on: per token each neuron as served,
    off: the same without the push (None for a member with no push)}. 1 is a neuron firing as
    its push does at dose 1."""
    start = cont.index("{") if e["kind"] == "listen" else cont.index('"')
    keep = [i for i, (s0, s1) in enumerate(sc["offs"]) if s1 > start and sc["trace"]["on"][i] is not None]
    return {"names": names, "toks": [cont[sc["offs"][i][0]:sc["offs"][i][1]] for i in keep],
            "on": [sc["trace"]["on"][i] for i in keep], "off": [sc["trace"]["off"][i] for i in keep] if pushed else None}


def clean_marks(e, others):
    """Drops the marks on words not in the line, and pointing at no one present."""
    ws = words(e["text"])
    e["marks"] = [k for k in e["marks"] if k["word"].strip(".,;:!?'\"").lower() in ws
                  and (k["mark"] != "point" or k["target"] in others)]
    return e


def words(text):
    return [w.strip(".,;:!?'\"").lower() for w in text.split()]


def problems(e, me, others):
    """What is wrong with a parsed move of `me`, as the filters for training rows see it."""
    out = []
    if e["kind"] == "listen":
        return out if e["to"][0] in others else ["listens to no one present"]
    if e["next"] not in e["to"]: out.append("next is not an addressee")
    ws = words(e["text"])
    if not ws or len(ws) > MAX_WORDS: out.append("line length")
    for n in others:                                    # a name said in the line is someone spoken to
        if n.lower() in ws and n not in e["to"]: out.append(f"names {n} without addressing them")
    if me.lower() in ws: out.append("says its own name")
    for k in e["marks"]:
        if k["word"].strip(".,;:!?'\"").lower() not in ws: out.append(f"mark on a word not in the line: {k['word']}")
        if k["mark"] == "point" and k["target"] not in others: out.append("points at no one present")
    if sum(a.startswith(WALKS) for a in e["act"].split(", then ")) > 1: out.append("walks twice")
    return out


def first_of_step(entries, cast):
    """Who moves first in a step: the one the newest spoken line wants to answer, else the first
    of the cast."""
    last = next((e for e in reversed(entries) if e["kind"] == "speak"), None)
    return last["next"] if last and last["next"] in cast else cast[0]


def after(e, cast, moved):
    """Who moves next in a step after move e: the one it wants to answer, if they have not moved
    yet this step, else the first of the cast who has not; None when all have moved."""
    if e["kind"] == "speak" and e["next"] in cast and e["next"] not in moved: return e["next"]
    return next((n for n in cast if n not in moved), None)


def repair(e, me, others):
    """A served move made to hold together: the marks on words not in the line dropped, a name said
    in the line made one it is said to, and `next` one of those it is said to."""
    if e["kind"] == "listen": return e
    clean_marks(e, others)
    ws = words(e["text"])
    for n in others:
        if n.lower() in ws and n not in e["to"]: e["to"].append(n)
    if e["to"] and e["next"] not in e["to"]: e["next"] = e["to"][0]
    return e


def choose(moves, record, me):
    """Speaking or listening as most of the moves have it (a tie speaks); then the move least like
    what the record has just said, and not the character's own last actions again. A character
    shown its own earlier lines tends to say them again (and a strong push makes it surer of
    them); sampling alone did not stop the repeats."""
    speaks = sum(e["kind"] == "speak" for e in moves)
    moves = [e for e in moves if (e["kind"] == "speak") == (2 * speaks >= len(moves))]
    recent = [set(words(e["text"])) for e in record[-AWARENESS:] if e["kind"] == "speak"]
    mine = next((e.get("act") for e in reversed(record) if e["who"] == me), None)
    feels = [e["feel"] for e in record[-AWARENESS:] if e["kind"] == "listen"]
    my_feel = next((e["feel"] for e in reversed(record) if e["who"] == me and e["kind"] == "listen"), None)

    def cost(e):
        if e["kind"] == "listen":                      # a feeling the record has less of
            return feels.count(e["feel"]) / max(1, len(feels)) + (0.5 if e["feel"] == my_feel else 0.0)
        w = set(words(e["text"]))
        same = max((len(w & r) / max(1, len(w | r)) for r in recent), default=0.0)
        return same + (0.5 if e["act"] == mine else 0.0)
    return min(moves, key=cost)
