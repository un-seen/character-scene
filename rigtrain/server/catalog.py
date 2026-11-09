"""What the server shares with its tools: the model, the characters, the pushes."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "Qwen/Qwen3.5-4B"
# the animation adapter (v8: the compact program, long scenes, runs, jumps, the empty world) merged into the base model (merge_lora.py)
ADAPTER = os.environ.get("RIG_ADAPTER") or os.path.join(os.path.dirname(HERE), "merged_v23e1")
if not os.path.isabs(ADAPTER): ADAPTER = os.path.join(os.path.dirname(HERE), ADAPTER)
CHARACTER_DIR = os.path.expanduser("~/character")
BLENDER = os.path.expanduser("~/blender/blender-5.1.1-linux-x64/blender")
RUNS = os.path.join(HERE, "runs")
MEDIAN = os.path.join(os.path.dirname(HERE), "median_norms.json")
REG = json.load(open(os.path.join(CHARACTER_DIR, "rigctl", "data", "registry.json")))
# Every character is a person on the one rig (out/rig.blend), so one program animates any
# of them: the server compiles every clip on the reference rig body, and the page plays it on the person's
# own model (static/chars/<id>.glb: character/humans, MakeHuman bodies fitted to the rig's deform bones).
RIG_FILE = os.path.join(CHARACTER_DIR, "out", "rig.blend")
# Their eyes, left and right, in the rest pose (Blender world, metres): the page finds the vertex there
# on the body and turns the head so the eyes meet whom it speaks to (character/humans/fit_human.py).
EYES = json.load(open(os.path.join(HERE, "eyes.json")))
CHARACTERS = {"sam": {"name": "Sam", "model": "/static/chars/sam.glb", "eyes": EYES["sam"]},
              "maya": {"name": "Maya", "model": "/static/chars/maya.glb", "eyes": EYES["maya"]}}
# more looks (humans/cast.json, built on out/rig.blend 2026-11-09); the page offers Sam and Maya
LOOKS = {"leo": "Leo", "ana": "Ana", "kenji": "Kenji", "zoe": "Zoe"}
CHARACTERS.update({k: {"name": v, "model": f"/static/chars/{k}.glb", "eyes": EYES[k]} for k, v in LOOKS.items()})
# The perspectives catalog: name -> (layer, neuron, sign), from the post's 12 voices and hostile.
PUSHES = {
    "flattering": (20, 7450, -1), "contemptuous": (17, 8623, +1), "poetic": (17, 7426, +1),
    "upbeat": (5, 8348, -1), "suspicious": (5, 6573, +1), "condescending": (11, 6819, +1),
    "wistful": (5, 8940, -1), "oracle": (5, 4530, -1), "bleak": (11, 2930, +1),
    "dismissive": (6, 6424, -1), "ornate": (15, 9125, +1), "curt": (5, 7656, -1),
    "hostile": (14, 8949, +1),
}
CALIB = ["What's the best way to learn a new language?", "Can you recommend a book to read?",
         "How do I fix a flat bicycle tyre?", "What do you think about the weather today?",
         "Should I get a cat or a dog?", "How does the internet work?"]


def push_ok(push):
    """A push the engine takes: one of PUSHES, or any neuron written L<layer>N<neuron><+|->."""
    from vsteer import custom_push
    return push in PUSHES or custom_push(push) is not None
