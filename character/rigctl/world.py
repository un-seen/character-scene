"""The world the characters are in, and what each action needs from it.

The world is the floor and the cast, and nothing else: no objects. So there is nothing to hold,
grip or pick up, and a body only changes where it is and how it stands, or acts towards another
character. An action is offered (qwen-say's and qwen-animate's grammars), written (the training
rows, tools/dataset.py) and compiled (program.validate) only if everything it needs is in the
world. Putting a thing in OBJECTS would make the actions that need one possible again; nothing
else needs to change.

Needs:
  floor    somewhere to stand on: to walk, run, turn, sit, lie on
  someone  another character: to walk to, point at, reach towards, follow, lead
  object   a thing: to grip, hold, pick up, reach for
"""
OBJECTS = []                                       # the things in the world: none
WORLD = {"floor"} | ({"object"} if OBJECTS else set())   # "someone" is there when a scene has more than one character

# what each action needs, by the names qwen-animate's programs and qwen-say's lines use
NEEDS = {
    # qwen-animate: verbs, and hand shapes as "hand:<shape>"
    "walk": {"floor"}, "walk_to": {"floor"}, "run": {"floor"}, "run_to": {"floor"}, "jump": {"floor"}, "turn": {"floor"},
    "crouch": {"floor"}, "kneel": {"floor"}, "sit": {"floor"}, "lie": {"floor"}, "pose": {"floor"},
    "idle": set(), "look_at": set(), "express": set(), "reach": {"someone"},
    "hand:relaxed": set(), "hand:fist": set(), "hand:open": set(), "hand:point": set(), "hand:grip": {"object"},
    # qwen-say: actions towards someone, and the body's own
    "walks to": {"floor", "someone"}, "runs to": {"floor", "someone"}, "follows": {"floor", "someone"},
    "leads": {"floor", "someone"}, "points at": {"someone"}, "reaches towards": {"someone"},
    "turns to": {"floor", "someone"}, "looks at": {"someone"},
    "walks a few steps": {"floor"}, "turns around": {"floor"}, "jumps": {"floor"},
    "crouches": {"floor"}, "kneels": {"floor"}, "sits down": {"floor"}, "lies down": {"floor"},
}


def has(others=True):
    """What the world holds for a character: the world's own, and someone if it is not alone."""
    return WORLD | ({"someone"} if others else set())


def allows(action, others=True):
    """Whether everything `action` needs is in the world (an action not listed needs nothing)."""
    return NEEDS.get(action, set()) <= has(others)


def lacks(action, others=True):
    """What `action` needs that the world does not have."""
    return sorted(NEEDS.get(action, set()) - has(others))
