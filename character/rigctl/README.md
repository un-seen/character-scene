# rigctl

`rigctl` is a program format, a compiler and a checker for driving the character rig (out/rig.blend) from text. A language model writes a short JSON program. The compiler turns that program into animation on the rig in headless Blender. The checker then scores the result in world space, and its score is what a training loop uses as a reward.

```
blender -b out/rig.blend --python rigctl/run.py -- registry OUT.json
blender -b out/rig.blend --python rigctl/run.py -- compile PROGRAM.json OUT_DIR [--preview]
blender -b ANIMATED.blend      --python rigctl/run.py -- check OUT_DIR
```

`compile` writes two files, `OUT_DIR/scene.blend` and `OUT_DIR/report.json`. The report holds the validation errors or warnings, the compile corrections and the check results. A program that fails validation is not compiled, and its report lists every error. `--preview` also renders `preview.mp4` with a following camera.

## The program

A program has three layers, and only `clip` is required. Program frames are 0-based. Distances are in metres. Angles are in degrees. Yaw 0 faces −Y. The walker's left is +X.

```jsonc
{
  "plan": "free text first: what the clip should show",
  "clip": {"fps": 24, "frames": 192, "seed": 11, "variation": 0.4},
  "beats": [                                           // 1. verbs on a timeline
    {"id": "go", "verb": "walk_to", "start": 0, "args": {"target": [1.2, -2.2], "mood": "confident"}},
    {"id": "reach", "verb": "reach", "after": "go", "dur": 24, "args": {"side": "R", "target": [1.15, -2.55, 1.05], "hold": 12}}
  ],
  "primitives": [                                      // 2. override a beat's gait parameters
    {"beat": "go", "type": "gait", "params": {"cadence_spm": 100, "arm_swing": 1.2}}
  ],
  "keys": [                                            // 3. sparse keys on named controls
    {"ctrl": "head", "ch": "rot", "f": 44, "v": [-10, 0, 18], "pin": true},
    {"ctrl": "chest", "ch": "rot", "f": 30, "v": [-6, 0, 0], "mode": "add", "hold": true},
    {"ctrl": "f_index.L", "ch": "curl", "f": 40, "v": 0.0},
    {"ctrl": "frown", "f": 50, "v": 0.9, "hold": true}
  ],
  "props": [{"prop": "upper_arm_parent.R:IK_FK", "f": 0, "v": 1.0}]
}
```

**Verbs:**

| Verb | What it does | Arguments |
|---|---|---|
| `walk` | Walks forward | `distance` (m), `speed`, `mood` |
| `walk_to` | Turns toward a point, then walks to it | `target` [x, y], `speed`, `mood` |
| `turn` | Turns in place | `deg` |
| `idle` | Stands | — |
| `look_at` | Looks at a point | `target` [x, y, z], `amount` |
| `express` | Sets face keys | `keys` {name: 0..1}, `onset`, `apex`, `offset` |
| `hand` | Takes a hand shape | `side` L/R/both, `shape` relaxed/fist/point/open/grip, `amount` |
| `reach` | Reaches a hand to a point | `side`, `target` [x, y, z], `hold` |

Each beat takes either `start` (a frame) or `after` (a beat id), and an optional `dur`.

**Moods:** neutral, tired, confident, sneaky, happy, angry, sad. Each sets the 15 gait parameters, and `primitives` can override any of them within its bounds. See `program.GAIT`.

**Keys:** each key can set these fields.
- `mode`: `abs` replaces the value, `add` offsets it.
- `ease`: `auto_clamped`, `linear` or `constant`. Interpolation never overshoots between keys.
- `blend`: the number of frames used to ease the key in and out at the ends of its track. The default is 6.
- `hold`: keeps the last value until the clip ends.
- `pin`: makes the key a hard constraint, which the checker then verifies.

**Channels:** units are listed in the registry.
- `rot`: local Euler XYZ, in degrees from rest.
- `loc`: bone-local metres from rest. Use `axes_at_rest` to see which world direction each local axis points.
- `curl`: 0 is straight and 1 is a fist. Used for digits.
- `spread`: in degrees.
- face keys take a `value` from 0 to 1.

## What the compiler guarantees

- **Only real names.** The registry (`registry.py`) is built from the generated rig: 36 controls, 10 digits, 8 limb switches and 5 face keys. Unknown names and out-of-range values are rejected before compiling.
- **Every legal pose is reachable.** Limits are each control's full range, written by hand, not a dataset's percentiles. Values are clamped again as a safety net, and every clamp goes into the report.
- **Feet do not skate.** The footstep planner fixes each plant in the world. A walk starts with a weight shift, its last step lands where the body stops, and the trailing foot closes beside it. Heel strike and push-off roll on the mesh's own sole: the foot turns about whichever vertex is lowest, and the toes stay on the floor during push-off.
- **The rig stays intact.** Rigify's 109 drivers (the IK/FK switches, follow and stretch) are kept. Clearing them once broke the arm switches without any error, so the checker now checks for them.

## What the checker measures

Everything is measured on the evaluated mesh and armature in world space. The floor is z = 0 by definition. A frame that fails to evaluate scores zero.

| Check | Pass condition |
|---|---|
| `rig_intact` | 100 or more Rigify drivers are present |
| `foot_slide` | Sole points drift no more than 1.5 cm while on the floor (95th percentile, stretches of 3 or more frames) |
| `penetration` | The lowest sole vertex is no more than 5 mm below the floor |
| `airborne` | No frame has both feet up (there is no jump verb) |
| `foot_clip` | The feet stay at least 4 cm apart |
| `lean` | The neck is between 8 cm behind and 20 cm ahead of the hips (a proud lean back and a deep hunch are both allowed) |
| `no_pops` | Per frame, feet move at most 0.25 m and accelerate at most 150 m/s². Hands, head and spine move at most 0.12 m and accelerate at most 60 m/s² |
| `moving` | Body motion relative to the root is at least 0.01 m/s, which blocks a frozen character |
| `balance` | On still frames, the centre of mass is over the support polygon at least 90% of the time |
| `frame_range` | The clip has exactly `clip.frames` frames |
| `walk_to_err_cm` | The walk ends within 25 cm of its target |
| `reach_err_cm` | The hand comes within 5 cm of its target |
| `look_err_deg` | The head points within 25° of its target |
| `pins` | Each pinned value is met to within 0.5 (cm or degrees) |

`reward` is 0 if the clip did not evaluate. Otherwise it is the mean of smooth terms of the form exp(−error / scale), anchored to these thresholds. Compile clamps also count against it.

## Examples (all on the GPU box)

| Program | Result |
|---|---|
| `examples/tired_walk.json` | Passes all checks. Reward 0.953, slide (95th percentile) 0.8 cm, balance 100% |
| `examples/walk_to_reach.json` | Passes all checks. Reward 0.906. Walk_to and reach errors are 0 cm, look error 22° |
| `examples/keys_layer.json` | Passes all checks. Reward 0.934. The pin is met exactly |
| `examples/adversarial_skate.json` | Fails `foot_slide` at 21.5 cm, as intended |
| `examples/invalid.json` | Rejected, with all 6 errors listed |

The earlier hand-keyed walk (`walk_scene.py`) fails the slide check at 2.7 cm. The planner with rolling contact gets 0.8–1.2 cm.

## Not built yet

- A learned motion prior. Unpinned keys are applied as written, and the "naturalness dial" from the report is not in place.
- Self-intersection checks, apart from feet clipping each other.
- A VLM judge on rendered frames, and a learned critic.
- Retargeting the reference rig's clips into programs.
