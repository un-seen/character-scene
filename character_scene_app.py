"""Live two-character scenes on Modal: the models (qwen-say, and qwen-animate with persona pushes;
rigtrain/server/vsteer.py) and a warm Blender pool, stepped by scene.py, one scene per GPU container.

  Runner.run(scene_id)   the scene's loop: a step at a time while it is running; paused when its
                         allowance (what the account's credit covers) is used up or nobody has
                         looked at it for IDLE_S; the container leaves PAUSE_KEEP_S after a pause
  api (asgi)             create, cast, personality, situation, start, pause, stop, status, list,
                         behind a bearer token (the secret's CHARACTER_TOKEN); clips are public at
                         unguessable URLs

A scene's state is the dict entry `scenes[id]` (scene.py's scene plus owner, clip key, usage, run).
Control is `scenes[id + ":ctl"]`, written only by the api: what the caller wants (running, paused,
stopped), the allowance in seconds, when someone last looked, and how much the caller has billed. Edits to a scene
whose loop is live go through `edits` (a queue per scene) and are applied between steps.

Usage is the container's time while it serves the scene: its loading, if the scene started it,
plus the time its loop runs; the caller bills it (scene.py knows nothing of money).

Settings, read when you deploy (see README):
  CHARACTER_WEB_URL     the api's own address, https://<workspace>--character-scene-api.modal.run
  CHARACTER_SECRET      the Modal secret holding CHARACTER_TOKEN (default character-scene-token)
  CHARACTER_MODEL_DIR   where the animation model sits on the Volume (default /m/character-animate-4b)
  CHARACTER_BASE_VOLUME the Volume for Qwen3.5-4B's own files (default qwen35-4b)

  modal run character_scene_app.py::fetch     # the published model and Qwen3.5-4B onto the Volumes
  modal deploy character_scene_app.py
  modal run character_scene_app.py::smoke     # a scene end to end
"""
import copy
import hmac
import os
import secrets
import sys
import threading
import time
import uuid

import modal

APP = "character-scene"
WEB = os.environ.get("CHARACTER_WEB_URL", "").rstrip("/")
ADAPTER = os.environ.get("CHARACTER_MODEL_DIR", "/m/character-animate-4b")
PUBLISHED = "hedwigai/character-animate-4b"             # the animation model, merged, with its `unmerge`
BLENDER = "blender-5.1.1-linux-x64"
IDLE_S = 120            # nobody has asked about the scene for this long: it pauses (no one is watching)
PAUSE_KEEP_S = 60       # a paused loop waits this long for a resume before its container leaves
DEAD_S = 240            # a running scene whose loop has not checked in for this long has lost its container
STARTING_S = 600        # a scene still starting after this long never got its container (it failed to load)
MAX_LIVE_PER_ACCOUNT = 2
MAX_SCENES_PER_ACCOUNT = 20

app = modal.App(APP)
runner_image = (
    modal.Image.from_registry("python:3.12-slim-bookworm")   # debian_slim is bullseye, whose security mirror is gone
    # Blender's libraries and a C compiler (torch.compile and Triton build kernels); the index is fetched
    # in the same layer (a cached index goes stale)
    .run_commands("apt-get update && apt-get install -y build-essential curl xz-utils libx11-6 libxi6 libxxf86vm1 libxfixes3 "
                  "libxrender1 libgl1 libxkbcommon0 libsm6 libice6 libegl1 libglu1-mesa libgomp1")
    .run_commands(f"mkdir -p /root/blender && curl -sL https://download.blender.org/release/Blender5.1/{BLENDER}.tar.xz"
                  " | tar -xJ -C /root/blender")
    .pip_install("vllm==0.32.0", "fastapi[standard]", "huggingface_hub[hf_transfer]", "flash-linear-attention", "numpy",
                 "peft")
    .env({"HF_HOME": "/w/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0",
          "VLLM_CACHE_ROOT": "/m/vllm-cache", "TRITON_CACHE_DIR": "/m/triton", "RIG_ADAPTER": ADAPTER,
          "RIG_BLENDERS": "8", "RIG_GPU_MEM": "0.85", "CHARACTER_WEB_URL": WEB})
    .add_local_dir("character", "/root/character", ignore=["**/__pycache__/**"])
    .add_local_dir("rigtrain", "/root/rigtrain", ignore=["**/__pycache__/**"])
    .add_local_file("scene.py", "/root/scene.py")
)
web_image = (modal.Image.debian_slim(python_version="3.12").pip_install("fastapi[standard]", "httpx")
             .env({"CHARACTER_WEB_URL": WEB}).add_local_file("scene.py", "/root/scene.py")
             .add_local_dir("chars", "/root/chars"))
LOOKS = ("sam", "maya", "leo", "ana", "kenji", "zoe")                     # scene.CHARACTERS
MODELS = {c: f"{WEB}/chars/{c}.glb" for c in LOOKS}

hf = modal.Volume.from_name(os.environ.get("CHARACTER_BASE_VOLUME", "qwen35-4b"), create_if_missing=True)  # Qwen3.5-4B (base W_down for the pushes)
model_vol = modal.Volume.from_name("character-scene", create_if_missing=True)  # the animation model (+ unmerge), compile caches
runs = modal.Volume.from_name("character-scene-runs", create_if_missing=True)  # the clips
scenes = modal.Dict.from_name("character-scenes", create_if_missing=True)
edits = modal.Queue.from_name("character-scene-edits", create_if_missing=True)
token = modal.Secret.from_name(os.environ.get("CHARACTER_SECRET", "character-scene-token"))


def _token():
    return os.environ.get("CHARACTER_TOKEN") or os.environ["PERSONA_TOKEN"]


def _ctl(sid):
    return scenes.get(sid + ":ctl") or {"want": "paused", "allowance_s": 0.0, "touch": 0.0}


def _apply(sc, e):
    """One edit to a scene (from the api, or drained from the queue by its loop)."""
    import scene as S
    op = e["op"]
    if op == "add": S.add_character(sc, e["character"], e.get("personality"), e.get("dose", 1.0), e.get("name"))
    elif op == "remove": S.remove_character(sc, e["character"])
    elif op == "personality":
        m = next((m for m in sc["cast"] if m["character"] == e["character"]), None)
        if not m: raise ValueError(f"{e['character']!r} is not in the scene")
        S.set_personality(m, e.get("personality"), e.get("dose", 1.0))
    elif op == "situation": sc["situation"] = str(e.get("situation") or "")[:600]
    elif op == "rename": S.rename(sc, e["character"], e.get("name"))
    elif op == "look": S.set_look(sc, e["character"], str(e.get("look") or "").lower())


@app.cls(gpu="L40S", cpu=8, memory=32768, image=runner_image, volumes={"/w": hf, "/m": model_vol, "/runs": runs},
         timeout=4 * 3600, scaledown_window=30, max_containers=4, secrets=[token])
class Runner:
    @modal.enter()
    def load(self):
        t = time.time()
        sys.path[:0] = ["/root/rigtrain/server", "/root/rigtrain", "/root"]
        os.chdir("/root/rigtrain")
        import app as A                               # the simulator server's own module (server/app.py)
        A.RUNS = "/runs"
        os.makedirs(os.path.join(os.path.dirname(A.METER)), exist_ok=True)
        A.steer = A.Models()
        A.pool = A.BlenderPool(int(os.environ["RIG_BLENDERS"]))
        threading.Thread(target=A.worker, args=(A.steer,), daemon=True).start()
        self.A, self.load_s, self.fresh = A, time.time() - t, True

    def say(self, body):
        return self.A.say(self.A.steer, body)

    def animate(self, items):
        A = self.A
        ms = [{"character": it["character"], "push": it["push"], "dose": float(it["dose"]) if it["push"] else 0.0,
               "text": it["text"][:A.MAX_TEXT], "stage": "queued"} for it in items]
        jid = uuid.uuid4().hex[:10]
        A.jobs[jid] = {"id": jid, "session": "scene", "items": ms, "stage": "queued"}
        A.work.put(jid)
        while True:
            j = A.jobs[jid]
            if j["stage"] in ("done", "failed", "cancelled") and not any(m.get("checking") for m in ms):
                j["id"] = jid
                for k, m in enumerate(ms): m["jid"], m["k"] = jid, k
                return j
            time.sleep(0.3)

    @modal.method()
    def run(self, sid):
        import scene as S
        started, extra = time.time(), (self.load_s if self.fresh else 0.0)
        self.fresh = False
        sc = scenes[sid]
        base = float(sc.get("used_s", 0.0)) + extra
        sc.update(state="running", reason=None, beat=time.time(), loop=uuid.uuid4().hex[:8])
        scenes[sid] = sc
        url = lambda n, k, m: f"{WEB}/clips/{sid}/{sc['clip_key']}/{m['jid']}/{m['k']}.glb"
        paused_at = None
        try:
            while True:
                got = edits.get_many(100, partition=sid, block=False) if edits.len(partition=sid) else []
                for e in got:
                    try: _apply(sc, e)
                    except ValueError as x: sc["last_error"] = str(x)
                if got: scenes[sid + ":pending"] = (scenes.get(sid + ":pending") or [])[len(got):]
                ctl = _ctl(sid)
                now = time.time()
                sc["used_s"] = round(base + now - started, 1)
                sc["beat"] = now
                if ctl["want"] == "stopped":
                    sc.update(state="stopped", reason="stopped"); break
                why = None
                if ctl["want"] == "paused": why = "paused"
                elif sc["used_s"] >= ctl.get("allowance_s", 0.0): why = "credit"
                elif sc.get("anim_s", 0.0) >= ctl.get("anim_limit_s", float("inf")): why = "free allowance used"
                elif now - ctl.get("touch", 0.0) > IDLE_S: why = "idle"
                elif not sc["cast"]: why = "no characters"
                if why:
                    if paused_at is None: paused_at = now; sc.update(state="paused", reason=why)
                    scenes[sid] = sc
                    if now - paused_at > PAUSE_KEEP_S: break
                    time.sleep(2); continue
                paused_at = None
                sc.update(state="running", reason=None); scenes[sid] = sc
                try:
                    S.step(sc, self, url)
                    runs.commit()
                except Exception as x:                # a step that fails is kept as an error; the loop goes on
                    sc["last_error"] = f"{type(x).__name__}: {str(x)[-300:]}"
                    time.sleep(3)
                sc["used_s"] = round(base + time.time() - started, 1)
                scenes[sid] = sc
        finally:
            sc["used_s"] = round(base + time.time() - started, 1)
            if sc.get("state") == "running": sc["state"] = "paused"
            sc["loop"] = None
            scenes[sid] = sc


@app.function(image=runner_image, volumes={"/w": hf, "/m": model_vol}, timeout=3600)
def fetch(repo: str = PUBLISHED, out: str = ADAPTER):
    """The published animation model (with its `unmerge`) and Qwen3.5-4B, onto the Volumes."""
    from huggingface_hub import snapshot_download
    snapshot_download("Qwen/Qwen3.5-4B")
    hf.commit()
    snapshot_download(repo, local_dir=out)
    model_vol.commit()
    print(sorted(os.listdir(out)))


@app.function(image=runner_image, volumes={"/w": hf, "/m": model_vol}, cpu=8, memory=49152, timeout=3600)
def merge(adapter: str = "/m/ckpt_v23", out: str = ADAPTER):
    """The animation adapter merged into Qwen3.5-4B, with its `unmerge` (server/../merge_lora.py)."""
    from huggingface_hub import snapshot_download
    sys.path.insert(0, "/root/rigtrain")
    import merge_lora
    snapshot_download(merge_lora.MODEL)               # the shared Volume's snapshot lacks a few small files
    hf.commit()
    merge_lora.main(adapter, out)
    model_vol.commit()
    print(sorted(os.listdir(out)))


# ---- the api ---------------------------------------------------------------------------------
@app.function(image=web_image, volumes={"/runs": runs}, secrets=[token], scaledown_window=300)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def api():
    from fastapi import Body, FastAPI, Header, HTTPException
    from fastapi.responses import FileResponse, JSONResponse
    sys.path.insert(0, "/root")
    import scene as S
    web = FastAPI()

    def auth(authorization):
        if not hmac.compare_digest(authorization or "", "Bearer " + _token()):
            raise HTTPException(401, "unauthorized")

    def owned(sid, account):
        sc = scenes.get(sid)
        if not sc or sc.get("owner") != account: raise HTTPException(404, "no such scene")
        return sc

    def live(sc):
        return sc.get("loop") is not None and time.time() - sc.get("beat", 0) < DEAD_S

    def view(sc, since=0):
        out = S.public(sc, since)
        c = _ctl(sc["id"])
        out.update(models=MODELS, usage_s=sc.get("used_s", 0.0), billed_s=c.get("billed_s", 0.0), reason=sc.get("reason"),
                   last_error=sc.get("last_error"), want=c["want"])
        return out

    def body_of(b):
        if not isinstance(b, dict) or not b.get("account"): raise HTTPException(400, "account is required")
        b["account"] = str(b["account"]); return b

    def edit(sc, e):
        """Applied now when no loop holds the scene, else queued for the loop."""
        if live(sc):
            # validated against the scene as it will stand: the edits its loop has not taken yet, then this
            key, trial = sc["id"] + ":pending", copy.deepcopy(sc)
            pending = scenes.get(key) or []
            for p in pending:
                try: _apply(trial, p)
                except ValueError: pass
            _apply(trial, e)
            scenes[key] = pending + [e]; edits.put(e, partition=sc["id"]); return view(trial)
        _apply(sc, e); scenes[sc["id"]] = sc; return view(sc)

    def touch(sid, allowance_s=None, want=None, anim_limit_s=None):
        c = _ctl(sid); c["touch"] = time.time()
        if allowance_s is not None: c["allowance_s"] = float(allowance_s)
        if anim_limit_s is not None: c["anim_limit_s"] = float(anim_limit_s)
        if want: c["want"] = want
        scenes[sid + ":ctl"] = c

    def run_edit(fn):
        try: return fn()
        except ValueError as x: raise HTTPException(400, str(x))

    @web.post("/scenes")
    def create(body: dict = Body(default={}), authorization: str = Header(None)):
        auth(authorization); b = body_of(body)
        mine = scenes.get("acct:" + b["account"]) or []
        if len(mine) >= MAX_SCENES_PER_ACCOUNT:
            raise HTTPException(400, f"an account keeps {MAX_SCENES_PER_ACCOUNT} scenes; stop one and delete it first")
        sid = uuid.uuid4().hex[:12]
        sc = S.new_scene(sid, b.get("situation", ""), b.get("tz_minutes", 0))
        sc.update(owner=b["account"], clip_key=secrets.token_urlsafe(12), used_s=0.0, loop=None)
        for c in (b.get("characters") or [])[:S.MAX_CAST]:
            run_edit(lambda: _apply(sc, {"op": "add", **(c if isinstance(c, dict) else {"character": c})}))
        scenes[sid] = sc; scenes["acct:" + b["account"]] = mine + [sid]
        touch(sid, 0.0, "paused")
        return view(sc)

    @web.post("/scenes/list")
    def listing(body: dict = Body(default={}), authorization: str = Header(None)):
        auth(authorization); b = body_of(body)
        out = []
        for sid in scenes.get("acct:" + b["account"]) or []:
            sc = scenes.get(sid)
            if sc: out.append({"id": sid, "state": sc["state"], "situation": sc["situation"], "steps": len(sc["steps"]),
                               "cast": [{"character": m["character"], "name": m["name"], "personality": m["push"], "dose": m["dose"]} for m in sc["cast"]],
                               "usage_s": sc.get("used_s", 0.0), "anim_s": sc.get("anim_s", 0.0),
                               "billed_s": _ctl(sid).get("billed_s", 0.0)})
        return {"scenes": out}

    @web.post("/scenes/{sid}/status")
    def status(sid: str, body: dict = Body(default={}), authorization: str = Header(None)):
        auth(authorization); b = body_of(body); sc = owned(sid, b["account"])
        lost = (sc["state"] == "running" and not live(sc)) or \
            (sc["state"] == "starting" and time.time() - sc.get("beat", 0) > STARTING_S)
        if lost: sc.update(state="paused", reason="lost", loop=None); scenes[sid] = sc
        touch(sid, b.get("allowance_s"), anim_limit_s=b.get("anim_limit_s"))
        return view(sc, int(b.get("since") or 0))

    @web.post("/scenes/{sid}/billed")
    def billed(sid: str, body: dict = Body(default={}), authorization: str = Header(None)):
        auth(authorization); b = body_of(body); sc = owned(sid, b["account"])
        c = _ctl(sid); c["billed_s"] = max(float(c.get("billed_s", 0.0)), float(b.get("through_s") or 0.0))
        scenes[sid + ":ctl"] = c
        return {"billed_s": c["billed_s"]}

    @web.post("/scenes/{sid}/cast")
    def cast(sid: str, body: dict = Body(default={}), authorization: str = Header(None)):
        auth(authorization); b = body_of(body); sc = owned(sid, b["account"])
        op = "remove" if b.get("remove") else "add"
        return run_edit(lambda: edit(sc, {"op": op, "character": str(b.get("character") or "").lower(),
                                          "personality": b.get("personality"), "dose": b.get("dose", 1.0),
                                          "name": b.get("name")}))

    @web.post("/scenes/{sid}/rename")
    def rename(sid: str, body: dict = Body(default={}), authorization: str = Header(None)):
        auth(authorization); b = body_of(body); sc = owned(sid, b["account"])
        return run_edit(lambda: edit(sc, {"op": "rename", "character": str(b.get("character") or "").lower(),
                                          "name": b.get("name")}))

    @web.post("/scenes/{sid}/look")
    def look(sid: str, body: dict = Body(default={}), authorization: str = Header(None)):
        auth(authorization); b = body_of(body); sc = owned(sid, b["account"])
        return run_edit(lambda: edit(sc, {"op": "look", "character": str(b.get("character") or "").lower(),
                                          "look": b.get("look")}))

    @web.post("/scenes/{sid}/personality")
    def personality(sid: str, body: dict = Body(default={}), authorization: str = Header(None)):
        auth(authorization); b = body_of(body); sc = owned(sid, b["account"])
        return run_edit(lambda: edit(sc, {"op": "personality", "character": str(b.get("character") or "").lower(),
                                          "personality": b.get("personality"), "dose": b.get("dose", 1.0)}))

    @web.post("/scenes/{sid}/situation")
    def situation(sid: str, body: dict = Body(default={}), authorization: str = Header(None)):
        auth(authorization); b = body_of(body); sc = owned(sid, b["account"])
        return run_edit(lambda: edit(sc, {"op": "situation", "situation": b.get("situation", "")}))

    @web.post("/scenes/{sid}/start")
    def start(sid: str, body: dict = Body(default={}), authorization: str = Header(None)):
        auth(authorization); b = body_of(body); sc = owned(sid, b["account"])
        if sc["state"] == "stopped": raise HTTPException(400, "the scene is stopped; make a new one")
        if not sc["cast"]: raise HTTPException(400, "the scene has no characters: add one first")
        touch(sid, b.get("allowance_s"), "running", b.get("anim_limit_s"))
        if not live(sc):
            others = [x for x in (scenes.get(s) for s in scenes.get("acct:" + b["account"]) or [] if s != sid) if x and live(x)]
            if len(others) >= MAX_LIVE_PER_ACCOUNT:
                touch(sid, want="paused")
                raise HTTPException(400, f"{MAX_LIVE_PER_ACCOUNT} scenes of this account are already live; pause one first")
            sc.update(state="starting", reason=None, loop="spawning", beat=time.time()); scenes[sid] = sc
            call = Runner().run.spawn(sid)
            print("spawned", sid, call.object_id, flush=True)
        return view(sc)

    @web.post("/scenes/{sid}/pause")
    def pause(sid: str, body: dict = Body(default={}), authorization: str = Header(None)):
        auth(authorization); b = body_of(body); sc = owned(sid, b["account"])
        touch(sid, want="paused")
        if not live(sc) and sc["state"] in ("running", "starting"): sc["state"] = "paused"; scenes[sid] = sc
        return view(sc)

    @web.post("/scenes/{sid}/stop")
    def stop(sid: str, body: dict = Body(default={}), authorization: str = Header(None)):
        auth(authorization); b = body_of(body); sc = owned(sid, b["account"])
        touch(sid, want="stopped")
        if not live(sc): sc["state"] = "stopped"; scenes[sid] = sc
        return view(sc)

    @web.get("/chars/{name}.glb")
    def char(name: str):
        if name not in MODELS: raise HTTPException(404, "no such character")
        return FileResponse(f"/root/chars/{name}.glb", media_type="model/gltf-binary",
                            headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "public, max-age=604800"})

    @web.get("/clips/{sid}/{key}/{jid}/{k}.glb")
    def clip(sid: str, key: str, jid: str, k: int):
        sc = scenes.get(sid)
        if not sc or not hmac.compare_digest(sc.get("clip_key", ""), key) or not jid.isalnum():
            raise HTTPException(404, "no such clip")
        path = f"/runs/{jid}/{k}/clip.glb"
        if not os.path.exists(path): runs.reload()
        if not os.path.exists(path): raise HTTPException(404, "no such clip")
        return FileResponse(path, media_type="model/gltf-binary",
                            headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "public, max-age=86400"})

    return web


@app.function(image=web_image, secrets=[token], timeout=1800)
def smoke(steps: int = 3, account: str = "smoke-test"):
    """The deployed api end to end, as toolkit drives it: a scene made, cast, started and watched
    until `steps` steps have come, a clip fetched, then stopped. modal run character_scene_app.py::smoke"""
    import httpx
    h = {"Authorization": "Bearer " + _token()}
    post = lambda p, **b: httpx.post(WEB + p, json={"account": account, **b}, headers=h, timeout=120).json()
    t0 = time.time()
    sc = post("/scenes", situation="They meet at a bus stop in the rain.", characters=[{"character": "leo", "name": "Omar"}], tz_minutes=330)
    sid = sc["id"]; print("created", sid, [(m["character"], m["name"]) for m in sc["cast"]])
    print("add maya:", [(m["character"], m["push"]) for m in post(f"/scenes/{sid}/cast", character="maya", personality="hostile", dose=0.75)["cast"]])
    print("start:", post(f"/scenes/{sid}/start", allowance_s=1200)["state"])
    seen = 0
    while seen < steps and time.time() - t0 < 1500:
        time.sleep(5)
        v = post(f"/scenes/{sid}/status", since=seen, allowance_s=1200)
        for s in v["steps"]:
            if s["n"] == 1:                                    # between steps: a new name and a new look
                print("rename maya:", [m["name"] for m in post(f"/scenes/{sid}/rename", character="maya", name="Priya")["cast"]])
                print("look maya->zoe:", [(m["character"], m["name"]) for m in post(f"/scenes/{sid}/look", character="maya", look="zoe")["cast"]])
            print(f"[{time.time() - t0:5.0f}s] step {s['n']} ({s['t_step']} s, {s['frames']} frames)")
            for m in s["members"]: print("   ", m["name"], m["kind"], repr(m["says"])[:70], "|", m["act"], "|", m["stage"], m["checks_failed"])
            seen = s["n"]
        if not v["steps"]: print(f"[{time.time() - t0:5.0f}s]", v["state"], v.get("reason"), v.get("last_error"), "used", v["usage_s"])
    clip = next((m["clip"] for s in [post(f"/scenes/{sid}/status")] for st in s["steps"] for m in st["members"] if m["clip"]), None)
    if clip: r = httpx.get(clip, timeout=60); print("clip", r.status_code, len(r.content), "bytes", r.headers.get("access-control-allow-origin"))
    for c in LOOKS: r = httpx.get(MODELS[c], timeout=60); print("model", c, r.status_code, len(r.content))
    v = post(f"/scenes/{sid}/stop")
    print("stop:", v["state"], "used", v["usage_s"], "s, anim", v["anim_s"], "s")
