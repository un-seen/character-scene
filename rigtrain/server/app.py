"""The simulator's two model services: qwen-say and qwen-animate.

Runs on the GPU box, one vLLM engine for both (vsteer.py). Each knows nothing of the other or of
the scene; the browser's scene loop (static/index.html) owns the cast, the record, where everyone
stands, and the playing, and calls them each step:

  POST /api/say      {session, time, items: [{character, push, dose, sid, record, others}]}
                     -> {items: [{line, actions, to, candidates, input}]}      (answers directly)
                     record: the statements the character is aware of, {sid, who, to, text, act},
                     who "I" for its own, to the names it was said to ("me" for this character);
                     others: the other characters' names; to: whom the new line is said to
  POST /api/dose     {time, items: [one say item], move} -> {rows: [{dose, move, effect, choices,
                     shown}]}: the dose strip, the same moment said greedily at each of DOSES, and
                     the move shown (as /api/say chose it) read at each
  POST /api/animate  {session, items: [{character, push, dose, text}]} -> {job}
  GET  /api/job/ID   {stage, items: [{stage, program, glb, frames, fps, end_yaw, checks, ...}], ...}
                     text: one sentence of action text ("The character walks to a spot 2.0 m to
                     the left, then smiles."); qwen-animate writes the program for that body and
                     the warm Blender pool compiles, checks and exports it (.glb)

  ~/rigtrain/server/run.sh [PORT]
"""
import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.expanduser("~/character"))
from rigctl import compact, grammar, program as P  # noqa: E402
from spec import short_prompt  # noqa: E402
import say as SAY  # noqa: E402
from catalog import *  # noqa: E402,F401,F403

MAX_CAST = 5
SCENE_CAST = 2      # a scene has two characters, never a third (the page's cast)

jobs, work = {}, queue.Queue()

# Character time: the seconds of animation delivered to each session, summed over the cast (one
# character for 30 s, or three for 10 s each). It is tracked only; billing comes later. People
# will animate characters free for FREE_CHARACTER_S (RIG_FREE_CHARACTER_S, set from what serving costs
# us), which the meter reports but does not enforce. A session is the id the page keeps in the
# browser, so it lasts until that browser's storage is cleared.
FREE_CHARACTER_S = float(os.environ.get("RIG_FREE_CHARACTER_S", "30"))
METER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runs", "meter.json")
meter_lock = threading.Lock()
try: kept = json.load(open(METER))
except (OSError, ValueError): kept = {}


def meter(session):
    used = round(kept.get(session, 0.0), 1)
    return {"character_s": used, "free_s": FREE_CHARACTER_S, "left_s": round(max(0.0, FREE_CHARACTER_S - used), 1)}


def charge(session, seconds):
    """Add seconds of animation delivered to a session and save the meter."""
    with meter_lock:
        kept[session] = kept.get(session, 0.0) + seconds
        tmp = METER + ".tmp"; json.dump(kept, open(tmp, "w")); os.replace(tmp, METER)


# the model every character runs on, for its tag: "Qwen3.5-4B · v17" (the animation adapter)
LLM_NAME = MODEL.split("/")[-1] + "".join(" · " + m for m in re.findall(r"v\d+", os.path.basename(ADAPTER))[:1])


class Models:
    """qwen-say and qwen-animate, one vLLM engine (vsteer.py)."""

    def __init__(self):
        from vsteer import VSteered
        self.v = VSteered(ADAPTER, json.load(open(MEDIAN)), PUSHES,
                          gpu_memory_utilization=float(os.environ.get("RIG_GPU_MEM", "0.7")))
        self.program_grammar = compact.ebnf(REG)

    def say(self, texts, members, grammars, n=1, seed=None):
        return self.v.say(texts, members, grammars, n, seed)

    def score(self, rows):
        return self.v.score(rows)

    def write(self, prompts, members, cancelled=None, done=None):
        # every member is on the one rig: its body card in the system prompt
        systems = [short_prompt("character") for m in members]
        return self.v.write(prompts, members, systems, self.program_grammar, cancelled, done)


class BlenderPool:
    """Blenders that stay running (rigctl/worker.py) and take compile jobs: a frame range to
    pose, or a clip to key, check and export. A cold Blender spends about a second starting and
    loading the character; a warm one opens the .blend in a fraction of that."""

    def __init__(self, n):
        self.idle = queue.Queue()
        for _ in range(n): self.idle.put(self._spawn())

    def _spawn(self):
        # one thread each: many workers run at once, and Blender would otherwise start a thread
        # per core in every one of them
        return subprocess.Popen([BLENDER, "-b", "-t", os.environ.get("RIG_BLENDER_THREADS", "1"),
                                 "--python", os.path.join(CHARACTER_DIR, "rigctl", "worker.py")], cwd=CHARACTER_DIR,
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def run(self, job, playable=None):
        """None when the job is done, else the error. playable() is called when a finish job has
        written its .glb, before it checks the clip."""
        p = self.idle.get()
        try:
            if p.poll() is not None: p = self._spawn()
            p.stdin.write(json.dumps(job) + "\n"); p.stdin.flush()
            for line in p.stdout:
                if line.startswith("@@PLAYABLE") and playable: playable()
                if line.startswith("@@DONE"): return None
                if line.startswith("@@FAIL"): return json.loads(line[7:])
            p = self._spawn(); return "the Blender worker died"
        finally:
            self.idle.put(p)


MAX_TEXT = 4000
POOL_SIZE = int(os.environ.get("RIG_BLENDERS", "16"))
# Posing a frame costs ~50 ms of Blender, nearly all of it evaluating the rig. Where the body moves
# fast, every frame is posed and keyed: keying every 2nd blurs the foot roll of a walk between keys
# (+0.5 cm slide, +1.5 mm sinking), and a gesture's or a reach's stroke. Elsewhere (standing,
# looking, breathing, the face's own beats) every KEY_STEP-th frame is, and the curves
# interpolate between (within 1 mm of posing every frame, at the hands, feet and head); the face
# is keyed on every frame all the same, as a blink is quicker.
KEY_STEP = int(os.environ.get("RIG_KEY_STEP", "3"))
DENSE = {"walk": 4, "walk_to": 4, "run": 4, "run_to": 4, "jump": 4, "crouch": 2, "kneel": 2, "sit": 2, "lie": 2, "pose": 2,
         "turn": 4, "reach": 30, "hand": 14}   # verb: frames after it ends still moving


def posed_frames(norm, frames):
    """The frame indices the clip poses: every one in and around a fast beat and on the program's
    own keys, every KEY_STEP-th elsewhere, and the first and last."""
    out = set(range(0, frames, KEY_STEP)) | {0, frames - 1} | {k["f"] for k in norm["keys"] if 0 <= k["f"] < frames}
    for b in norm["beats"]:
        if b["verb"] in DENSE: out |= set(range(max(0, b["start"] - 2), min(frames, b["end"] + DENSE[b["verb"]] + 1)))
    return sorted(out)
pool = None


def compile_member(jid, idx, m, n_active=1):
    """Validate and compile one member's program on its body; fill in its result. One member's
    failure never stops the others."""
    try:
        _compile(jid, idx, m, n_active)
    except Exception:
        m.update(stage="failed", error=traceback.format_exc()[-800:])


def end_yaw(norm):
    """Where the program leaves the body facing, in degrees from where it started (+ left): the
    page turns the next step's clip to it."""
    loco = [b for b in norm["beats"] if b["verb"] in P.WALKS + ("turn",) and "to" in b]
    return round(math.degrees(max(loco, key=lambda b: b["end"])["to"][2]), 2) if loco else 0.0


def _compile(jid, idx, m, n_active):
    """On the warm pool: the clip's frames are posed in parallel ranges (each frame starts from the
    reset pose), then one worker keys them, checks the clip and exports it."""
    prog = m["program"]
    m["words"] = json.loads(json.dumps(prog))             # the program as the model wrote it (lowering rewrites the pose keys)
    try:
        P.validate(json.loads(json.dumps(prog)), REG)       # the words first
        if any(k.get("spec") for b in prog["beats"] if b["verb"] == "pose" for k in b["args"]["keys"]):
            from rigctl import poseprog                       # keys in body words: solved to angles (poseprog.py)
            t0 = time.time(); prog = poseprog.lower(prog); m["t_solve"] = round(time.time() - t0, 1)
        norm, _ = P.validate(prog, REG)
    except P.ProgramError as e:
        m.update(stage="invalid", error="; ".join(e.args[0])); return
    except Exception as e:
        m.update(stage="invalid", error=f"malformed program: {e}"); return
    m["end_yaw"] = end_yaw(norm)
    m["timeline"] = {"beats": [{"id": b["id"], "verb": b["verb"], "start": b["start"], "end": b["end"]} for b in norm["beats"]],
                     "keys": sorted({k["f"] for k in norm["keys"]})}
    out = os.path.join(RUNS, jid, str(idx)); os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "program.json"); json.dump(prog, open(path, "w"), indent=1)
    m["stage"] = "compiling"; t = time.time()
    blend, frames = RIG_FILE, prog["clip"]["frames"]
    keyed = posed_frames(norm, frames)
    face = sorted(set(range(frames)) - set(keyed))
    n = max(1, min(POOL_SIZE // max(1, n_active), math.ceil(len(keyed) / 15)))
    cuts = [round(i * len(keyed) / n) for i in range(n + 1)]
    chunks = [os.path.join(out, f"frames{i}.json") for i in range(n)]
    with ThreadPoolExecutor(n) as ex:
        errs = list(ex.map(pool.run, [{"op": "frames", "blend": blend, "program": path, "frames": keyed[cuts[i]:cuts[i + 1]],
                                       "face": face[i::n], "out": chunks[i]} for i in range(n)]))
    if any(errs):
        m.update(stage="failed", error=next(e for e in errs if e)[-1500:]); return
    m["t_frames"] = round(time.time() - t, 1)
    ready = threading.Event()

    def playable():                                   # the clip can play while it is being checked
        m["t_blender"] = round(time.time() - t, 1)
        m.update(stage="done", checking=True, glb=f"/runs/{jid}/{idx}/clip.glb", fps=prog["clip"]["fps"], frames=frames)
        ready.set()
    threading.Thread(target=_finish, args=(m, blend, path, chunks, out, t, playable, ready), daemon=True).start()
    ready.wait()


def _finish(m, blend, path, chunks, out, t, playable, ready):
    err = pool.run({"op": "finish", "blend": blend, "program": path, "chunks": chunks, "out_dir": out, "glb": True},
                   playable=playable)
    if err:
        m.update(stage="failed", error=err[-1500:], checking=False); ready.set()
        return
    rep = json.load(open(os.path.join(out, "report.json")))
    chk = rep.get("check", {})
    m["checks"] = {k: {"pass": v["pass"], "value": v["value"], "need": v["need"]} for k, v in chk.get("checks", {}).items()}
    m["reward"], m["ok"], m["checking"] = rep.get("reward"), chk.get("ok"), False
    m["t_checked"] = round(time.time() - t, 1)


def session_of(sid, handler):
    """The page's session id (letters, digits, - and _), or the caller's address without one."""
    sid = str(sid or "")[:64]
    return sid if sid and all(c.isalnum() or c in "-_" for c in sid) else "ip:" + handler.client_address[0]


class Cancelled(Exception):
    pass


engine = threading.Lock()                              # one vLLM engine: one call into it at a time


def say(steer, body):
    """qwen-say for a batch of characters: each one's next move (server/say.py), chosen among
    CANDIDATES. An item: character, push, dose, name, sid, others (the others' names), record (the
    moves so far, names as in the cast), heard (who spoke this step), speak (whether it may)."""
    time_, items = str(body.get("time") or ""), body.get("items") or []
    members, texts, grammars = {}, {}, {}
    for k, it in enumerate(items):
        members[k], texts[k], grammars[k] = say_item(it, time_)
    t = time.time()
    with engine: raws = steer.say(texts, members, grammars, SAY.CANDIDATES)
    out = []
    for k, it in enumerate(items):
        moves = []
        for r in raws.get(k, []):
            try: moves.append(SAY.repair(SAY.parse(r), it["_me"], it["_others"]))
            except ValueError: pass                    # cut off at the token limit
        if not moves: raise ValueError(f"qwen-say wrote no whole move for {it['_me']}: {raws.get(k, [''])[0][:200]!r}")
        out.append({"move": SAY.choose(moves, it["_record"], it["_me"]), "input": texts[k],
                    "candidates": [SAY.continuation(m) for m in moves]})
    # what each member's push did to the move it chose (the move read again with the push and
    # without), and what the model wanted where grammar gave it a choice
    conts = [SAY.continuation(o["move"]) for o in out]
    t_score = time.time()
    with engine: scored = steer.score([(texts[k], conts[k], members[k]) for k in range(len(items))])
    for o, c, sc, it in zip(out, conts, scored, items):
        o["effect"] = SAY.effect(o["move"], c, sc["toks"])
        o["choices"] = SAY.choices(o["move"], c, sc, it["_me"], it["_others"], it["_heard"], it["_speak"], it["_avoid"])
        o["neurons"] = SAY.neurons(o["move"], c, sc, steer.v.trace_names, bool(it.get("push")))
    return {"items": out, "seconds": round(time.time() - t, 2), "score_s": round(time.time() - t_score, 2)}


def motions():
    """Everything a character's body can be given, for the page's catalog: what qwen-say chooses
    from, and what qwen-animate writes (its verbs and their args, plan moods, style dials, the
    controls its keys can name), all read from where they are defined."""
    from rigctl import compact as C, world as W
    return {"world": {"objects": W.OBJECTS, "has": sorted(W.has()), "needs": {k: sorted(v) for k, v in W.NEEDS.items()}},
            "say": {"first": SAY.BODY, "then": SAY.ALONE, "toward": SAY.SAY_TOWARD, "gestures": SAY.GESTURES,
                    "feelings": SAY.FEELINGS, "max_actions": 4},
            "verbs": [{"verb": v, "does": P.VERB_DOC[v][0], "dur": P.DEFAULT_DUR.get(v),
                       "args": [{"name": n, "kind": k, "required": n in C.REQUIRED.get(v, ()), "range": P.VERB_DOC[v][1].get(n, "")}
                                for n, k in C.ARGS[v]]} for v in C.ARGS],
            "plan_moods": grammar.PLAN_MOODS, "beat_moods": {m: g for m, g in P.MOOD_GAIT.items()},
            "shapes": sorted(P.SHAPES), "face": list(REG["face"]),
            "style": [{"name": k, "low": lo, "high": hi, "neutral": n, "standing": k in P.STYLE} for k, (lo, hi, n) in P.GAIT.items()],
            "controls": sorted(REG["controls"]), "digits": sorted(REG["digits"])}


def say_item(it, time_):
    """One /api/say item read: (member, its record as qwen-say continues it, its grammar), and
    what choices needs kept on the item."""
    ch, push = it.get("character"), it.get("push") or None
    if ch not in CHARACTERS: raise ValueError(f"unknown character {ch!r}")
    if push and not push_ok(push): raise ValueError(f"unknown push {push!r}")
    member = {"push": push, "dose": float(it.get("dose", 1.0)) if push else 0.0}
    me, others = str(it.get("name") or "").strip() or ch, [str(n) for n in it.get("others") or []]
    heard = [str(n) for n in it.get("heard") or [] if str(n) in others]
    try:
        record = [{"sid": int(e["sid"]), "who": str(e["who"]), "kind": e["kind"], "to": [str(n) for n in e["to"]]}
                  | ({"feel": str(e["feel"]), "amount": float(e["amount"])} if e["kind"] == "listen" else
                     {"next": e.get("next"), "text": str(e["text"]), "act": str(e["act"])})
                  for e in it.get("record") or []]
    except (KeyError, TypeError) as x:
        raise ValueError(f"bad record entry: {x}")
    speak, avoid = bool(it.get("speak", True)), SAY.last_actions(record, me)
    it["_record"], it["_me"], it["_others"], it["_heard"], it["_speak"], it["_avoid"] = record, me, others, heard, speak, avoid
    return member, SAY.context(time_, record, int(it.get("sid", 1)), me), SAY.grammar(others, heard, speak, avoid)


DOSES = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5]
DOSE_SEED = 7


def dose_strip(steer, body):
    """The same moment of one character at each of DOSES of its push (and its own dose): the move
    qwen-say writes (one seed for all, so only the dose changes it; greedy, it says one sentence
    over and over until cut off), what the push did to it and what grammar kept, and how the model
    at that dose reads the move shown (`move`, the one /api/say chose): nats a token of its line,
    or of a listener's feeling. A move cut off at the token limit is its raw text. Dose 0 is the
    push at no strength, the model as it is."""
    it = (body.get("items") or [None])[0]
    member, text, grammar = say_item(it, str(body.get("time") or ""))
    if not member["push"]: raise ValueError("no push to dose")
    doses = sorted(set(DOSES) | {member["dose"]})
    ms = {k: {"push": member["push"], "dose": d} for k, d in enumerate(doses)}
    t = time.time()
    with engine: raws = steer.say({k: text for k in ms}, ms, {k: grammar for k in ms}, 1, seed=DOSE_SEED)
    shown = SAY.continuation(body["move"]) if body.get("move") else None
    rows, conts = [], []
    for k, d in enumerate(doses):
        try: e = SAY.repair(SAY.parse(raws[k][0]), it["_me"], it["_others"])
        except ValueError: e = None
        rows.append({"dose": d, "move": e} | ({} if e else {"raw": raws[k][0]})); conts.append(SAY.continuation(e) if e else None)
    reads = [(text, c, ms[k]) for k, c in enumerate(conts) if c] + ([(text, shown, ms[k]) for k in ms] if shown else [])
    with engine: scored = steer.score(reads)
    own = iter(scored[:len(reads) - (len(ms) if shown else 0)])
    for r, c in zip(rows, conts):
        if not c: continue
        sc = next(own)
        r["effect"] = SAY.effect(r["move"], c, sc["toks"])
        r["choices"] = SAY.choices(r["move"], c, sc, it["_me"], it["_others"], it["_heard"], it["_speak"], it["_avoid"])
    if shown:
        for r, sc in zip(rows, scored[-len(ms):]):
            ef = SAY.effect(body["move"], shown, sc["toks"])
            r["shown"] = ef and ef.get("own")
    return {"rows": rows, "seconds": round(time.time() - t, 2)}


def worker(steer):
    """qwen-animate, one job at a time: each item's program written for its body (each compiles as
    soon as it is written, while the rest are still being written), then checked."""
    members_pool = ThreadPoolExecutor(max_workers=MAX_CAST)
    while True:
        jid = work.get(); j = jobs[jid]
        try:
            if j.get("cancel"): raise Cancelled()
            t0 = time.time(); j["stage"] = "writing the programs"
            ms = j["items"]
            futs, t1 = [], None

            def ready(k, raw):
                nonlocal t1
                m = ms[k]
                try:
                    m["program"] = compact.decode(raw)             # the compact program, as JSON
                    m["clamped"] = grammar.clamp(m["program"])
                except (ValueError, IndexError, KeyError) as e:
                    m.update(stage="invalid", error=f"unreadable program ({e}): " + raw[:300]); return
                if j.get("cancel"): return
                t1 = t1 or time.time()
                futs.append(members_pool.submit(compile_member, jid, k, m, len(ms)))
            with engine:
                raws, terms = steer.write([m["text"] for m in ms], ms, lambda: j.get("cancel"), ready)
            j["t_model"] = round(time.time() - t0, 1)
            j["write_profile"] = steer.v.profile
            for m, term in zip(ms, terms): m["push_terms"] = term
            if j.get("cancel"): raise Cancelled()
            j["stage"] = "compiling and checking in Blender"
            for f in futs: f.result()
            delivered = sum(m["frames"] / m["fps"] for m in ms if m.get("stage") == "done")
            charge(j["session"], delivered); j["meter"] = meter(j["session"])
            t1 = t1 or time.time()
            j["t_blender"] = round(time.time() - t1, 1)
            j["stage"] = "done"
        except Cancelled:
            j["stage"] = "cancelled"
        except Exception:
            j.update(stage="failed", error=traceback.format_exc()[-1500:])


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)): body = json.dumps(body).encode()
        elif isinstance(body, str): body = body.encode()
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body))); self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(body)

    def file(self, path, ctype):
        if not os.path.isfile(path): return self.send(404, {"error": "not found"})
        self.send(200, open(path, "rb").read(), ctype)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"): return self.file(os.path.join(HERE, "static", "index.html"), "text/html; charset=utf-8")
        if p.startswith("/static/") and ".." not in p:
            ext = p.rsplit(".", 1)[-1]
            return self.file(os.path.join(HERE, p.lstrip("/")), {"jpg": "image/jpeg", "js": "text/javascript", "svg": "image/svg+xml"}.get(ext, "application/octet-stream"))
        if p.startswith("/runs/") and p.endswith("/clip.glb") and ".." not in p:
            return self.file(os.path.join(HERE, p.lstrip("/")), "model/gltf-binary")
        if p == "/api/options":
            return self.send(200, {"characters": [{"id": k, "name": v["name"], "model": v["model"], "eyes": v["eyes"]} for k, v in CHARACTERS.items()],
                                   "pushes": sorted(PUSHES), "max_cast": SCENE_CAST, "llm": LLM_NAME, "motions": motions()})
        if p == "/api/meter":
            q = dict(x.split("=", 1) for x in self.path.partition("?")[2].split("&") if "=" in x)
            return self.send(200, meter(session_of(q.get("session"), self)))
        if p.startswith("/api/job/"):
            j = jobs.get(p.rsplit("/", 1)[-1])
            return self.send(200 if j else 404, j or {"error": "no such job"})
        self.send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.startswith("/api/cancel/"):
            j = jobs.get(self.path.rsplit("/", 1)[-1])
            if j and j["stage"] not in ("done", "failed"): j["cancel"] = True
            return self.send(200 if j else 404, {"ok": bool(j)})
        n = int(self.headers.get("Content-Length", 0))
        try: body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError: body = {}
        session = session_of(body.get("session"), self)
        items = body.get("items")
        if not isinstance(items, list) or not 1 <= len(items) <= MAX_CAST:
            return self.send(400, {"error": f"items must have 1–{MAX_CAST} entries"})
        if self.path in ("/api/say", "/api/dose"):
            try: return self.send(200, (say if self.path == "/api/say" else dose_strip)(steer, body))
            except ValueError as e: return self.send(400, {"error": str(e)})
        if self.path != "/api/animate": return self.send(404, {"error": "not found"})
        ms = []
        for it in items:
            ch, push = it.get("character"), it.get("push") or None
            dose = float(it.get("dose", 1.0)) if push else 0.0
            text = str(it.get("text") or "").strip()
            if ch not in CHARACTERS: return self.send(400, {"error": f"unknown character {ch!r}"})
            if push and not push_ok(push): return self.send(400, {"error": f"unknown push {push!r}"})
            if push and not 0.5 <= dose <= 1.5: return self.send(400, {"error": "dose must be 0.5–1.5"})
            # qwen-animate reads the record's last lines, and a strong push writes long ones: 4000
            # characters is ~1000 tokens, which leaves its 3072 the program's 900 and the system prompt
            if not text or len(text) > MAX_TEXT: return self.send(400, {"error": f"text must be 1–{MAX_TEXT} characters"})
            ms.append({"character": ch, "push": push, "dose": dose, "text": text, "stage": "queued"})
        jid = uuid.uuid4().hex[:10]
        jobs[jid] = {"id": jid, "session": session, "items": ms, "stage": "queued", "queued_ahead": work.qsize()}
        work.put(jid)
        self.send(200, {"job": jid})

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    os.makedirs(RUNS, exist_ok=True)
    print("loading model…", flush=True)
    steer = Models()
    pool = BlenderPool(POOL_SIZE); print(f"{POOL_SIZE} Blender workers warm", flush=True)
    threading.Thread(target=worker, args=(steer,), daemon=True).start()
    print(f"READY http://0.0.0.0:{port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()
