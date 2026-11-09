# character-scene

Two people stand in an empty world. Each one has a look, a name and a personality. A small language model decides what each says and does. A second model, fine-tuned from the same base, writes how each body moves. Blender turns those moves into animation clips. The scene runs one step at a time, and you can change a name, a look or a personality while it runs.

- **Try it:** [app.hedwigai.com/try/live-scene](https://app.hedwigai.com/try/live-scene)
- **The animation model:** [hedwigai/character-animate-4b](https://huggingface.co/hedwigai/character-animate-4b)

## How a step is made

1. **What each one says.** Qwen3.5-4B continues the scene's record for one character at a time, so the second hears the first. A personality is one persona neuron of the model, pushed at a dose from 0.5 to 1.5 (`rigtrain/server/vsteer.py`). Thirteen are offered: flattering, contemptuous, poetic, upbeat, suspicious, condescending, wistful, oracle, bleak, dismissive, ornate, curt and hostile. A grammar keeps each move in the record's format.
2. **What each one does.** The moves become body actions ("turns to Priya, then points at Priya") and then one sentence per character in its own frame ("walks to a spot 1.2 m ahead").
3. **How each body moves.** The animation model writes a short JSON program for that sentence: a plan, a style, timed beats (walk, turn, look, reach, express, pose) and sparse keys. Decoding is constrained by the program's grammar.
4. **The clip.** `character/rigctl` compiles the program onto the rig in headless Blender, checks it in world space (balance, lean, feet that slide or sink, the body passing through itself, joint limits, pops between frames) and exports a glTF clip.

The page plays each clip on the character's model by bone name, so any look plays any clip.

## What is in this repository

| Path | What it is |
|---|---|
| `character_scene_app.py` | The Modal app: one GPU container per running scene, and a web API to make, cast, rename, restyle, start, pause and stop scenes |
| `scene.py` | The scene loop: the record, whose turn it is, where each one stands, and one step from start to clip |
| `rigtrain/server/` | The model services: `say.py` (qwen-say and its grammar), `vsteer.py` (persona pushes on vLLM), `app.py` (the animate jobs and the Blender pool), `catalog.py` |
| `rigtrain/spec.py` | The animation model's prompt, built from the program format |
| `character/rigctl/` | The program format, its validator, the compiler, the pose solver and the checker |
| `character/humans/` | How the six people were built with MPFB and fitted to the rig |
| `web/ScenePlayer.tsx` | The three.js player the page uses |

## What you supply

The rig we animate is not ours to publish, so these files are not in the repository. The code expects them at these paths:

- `character/out/rig.blend`: an armature named `Rig` and its body mesh named `Body`. The deform bones are named `DEF-*` (for example `DEF-HIPS`, `DEF-HEAD`, `DEF-ARM-01.L`). The body carries five face shape keys: `blink`, `smile`, `brows_up`, `frown` and `jaw_open`.
- `character/rigctl/data/registry.json`: every control a program may use, with its channels and limits. `blender -b character/out/rig.blend --python character/rigctl/run.py -- registry OUT.json` writes it from your rig.
- `character/rigctl/data/fk.json`, `posecat.json` and `face_keys.json`: the rig's rest pose, its skin landmarks and its face keys, measured on the rig.
- `character/rigctl/data/run_cycle.json`: a run cycle on the rig, as local transforms per frame.
- `chars/<look>.glb`: one model per look (`sam`, `maya`, `leo`, `ana`, `kenji`, `zoe`) on the same skeleton. `character/humans/` shows how ours were made.

The animation model learned the control names in `rigtrain/spec.py`. A different rig needs those names mapped onto its own bones (`character/rigctl/registry.py`, `BONE`).

## Running it on Modal

```bash
export CHARACTER_WEB_URL=https://<your-workspace>--character-scene-api.modal.run
modal secret create character-scene-token CHARACTER_TOKEN=<a long random string>
modal run character_scene_app.py::fetch     # the animation model and Qwen3.5-4B onto the Volumes
modal deploy character_scene_app.py
modal run character_scene_app.py::smoke     # makes a scene, runs three steps, renames one and changes a look
```

A runner takes an L40S. The first step after a quiet spell waits two to three minutes while vLLM loads and compiles; later steps take about 15 seconds and make about ten seconds of animation for two people.

### The API

All calls are `POST` with a JSON body that carries `account`, behind `Authorization: Bearer <CHARACTER_TOKEN>`.

| Path | Does |
|---|---|
| `/scenes` | Make a scene: `situation`, `characters` (looks, or `{character, name, personality, dose}`) |
| `/scenes/list` | The account's scenes |
| `/scenes/{id}/status` | The scene and its steps after `since`; asking is what keeps a running scene running |
| `/scenes/{id}/cast` | Add or remove a character |
| `/scenes/{id}/rename` | A character's new name (one word of letters, up to 16) |
| `/scenes/{id}/look` | A character's new look, where it stands |
| `/scenes/{id}/personality` | A character's push and dose, or none |
| `/scenes/{id}/situation` | A new situation |
| `/scenes/{id}/start`, `/pause`, `/stop` | Run it, hold it, end it |

A running scene takes changes at its next step. `GET /chars/{look}.glb` serves the models, and each step's clips are served at unguessable URLs.

## Credits

This builds on open work, and we thank the people behind it.

- [MPFB](https://static.makehumancommunity.org/mpfb.html), MakeHuman for Blender (GPL 3), built the six people.
- [MakeHuman](https://www.makehumancommunity.org/) system assets (CC0): the bodies, skins, eyes, hair and clothes.
- [Blender](https://www.blender.org/) (GPL) compiles, checks and exports every clip.
- [Qwen3.5-4B](https://huggingface.co/Qwen) (Apache 2.0) speaks for each character, and, fine-tuned, writes how they move.
- [vLLM](https://github.com/vllm-project/vllm) (Apache 2.0) serves the models.
- [PyTorch](https://pytorch.org/) (BSD), [Transformers](https://github.com/huggingface/transformers) and [PEFT](https://github.com/huggingface/peft) (Apache 2.0) trained and merged the fine-tune.
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) (MIT) runs the model's linear-attention layers.
- [FastAPI](https://fastapi.tiangolo.com/) (MIT) serves the API.
- [three.js](https://threejs.org/) (MIT) plays the clips in the browser.
- [Modal](https://modal.com/) runs the GPUs.

## License

GPL-3.0-or-later (see `LICENSE`). The compiler and the character scripts run inside Blender and call MPFB, which are GPL. The animation model's weights are Apache 2.0, as their base is.
