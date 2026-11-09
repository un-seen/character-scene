"""Root path and footsteps for the locomotion beats.

Feet are planned as plants: a foot lands at a fixed place in the world and stays there until
it lifts, so a planted foot cannot slide whatever the body does. Between plants it swings.
A walking plant lands heel first and peels off over the ball; a turning or closing step is
flat to flat. Yaw 0 faces −Y and forward is (sin yaw, −cos yaw); the walker's left is +X.

Everything here is in seconds and metres; the compiler samples it per frame.
"""
import math
import random
from dataclasses import dataclass, field

from .program import RUN_RAMP_STEPS, TURN_STEP_DEG, WALKS, jump_times, turn_frames

STANCE = 0.65          # share of a stride a foot spends planted while walking
# Running, a foot is down while the body passes over it, RUN_SPAN (the reference rig's run lands 0.56 m
# ahead of the hips and leaves 0.64 behind), so the faster
# the run the shorter the stance; never under RUN_STANCE[0] nor over [1] of a stride.
RUN_SPAN, RUN_STANCE = 1.20, (0.18, 0.35)   # slower, the steps shorten, not the swing
RUN_BACK = 0.16        # running plants this much further back: the reference rig's span is centred under the hips


def stance_of(g, v):
    """The share of a stride a foot is planted at speed v (m/s) in gait g."""
    if not g.get("run"): return STANCE
    return min(max(RUN_SPAN / max(v, 1e-3) / (2 * 60.0 / g["cadence_spm"]), RUN_STANCE[0]), RUN_STANCE[1])
LEAD = 0.25            # seconds of weight shift before the first step lifts
TURN_WIDE = 0.02       # turning steps land this much wider each side, so a rotating foot clears the other


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def lerp(a, b, u):
    return a + (b - a) * u


def fwd(yaw):
    return math.sin(yaw), -math.cos(yaw)


@dataclass
class Plant:
    x: float              # the plant frame: where the walker's root would stand for this foot
    y: float
    yaw: float
    land: float           # time the foot arrives (−inf for the starting stance)
    lift: float           # time it leaves (+inf if it never does)
    roll_in: bool = False  # arrived heel first
    peel_out: bool = False  # leaves over the ball
    lift_h: float = 0.05
    wide: float = 0.0     # extra step width each side (turning steps stand wider)
    jump_out: bool = False  # leaves in a jump's push (both feet at once)
    jump_in: bool = False   # arrives from a jump's flight, toes first


@dataclass
class Segment:
    kind: str             # walk | turn
    t0: float
    t1: float
    x0: float; y0: float; yaw0: float
    x1: float; y1: float; yaw1: float
    gait: dict = field(default_factory=dict)
    turn_t: float = 0.0   # walk_to: seconds spent turning before walking
    yaw_from: float = None  # walk_to: the heading it sets off from, when the turn is too small to step
    jump: tuple = None      # jump: (height, crouch, push, flight, land, recover)
    moving: float = 0.0   # seconds the root is moving (walk)
    v: float = 0.0
    step_t: float = 0.0
    stance: float = STANCE  # share of a stride a foot is planted
    ramp: float = 0.0       # seconds it takes to reach speed, and to stop


class Plan:
    def __init__(self, beats, fps, seed, variation):
        self.fps, self.var = fps, variation
        self.rng = random.Random(seed)
        self.segs = []
        self.plants = {"L": [Plant(0, 0, 0, -math.inf, math.inf)], "R": [Plant(0, 0, 0, -math.inf, math.inf)]}
        for b in beats:
            if b["verb"] in WALKS + ("turn", "jump"):
                self._add(b)

    # -- building ---------------------------------------------------------------------------
    def _add(self, b):
        t0 = b["start"] / self.fps
        x0, y0, yaw0 = b["from"]
        x1, y1, yaw1 = b["to"]
        if b["verb"] == "jump":
            return self._jump(t0, b["jump_h"], x0, y0, x1, y1, yaw0)
        if b.get("turn_deg"):
            tt = (b["dur"] if b["verb"] == "turn" else turn_frames(b["turn_deg"])) / self.fps
            self._turn(t0, tt, x0, y0, yaw0, yaw0 + math.radians(b["turn_deg"]))
            t0 += tt
        if b["verb"] == "turn" or b["distance"] < 1e-3:
            return
        self._walk(t0, b["gait"], x0, y0, yaw1, b["distance"], None if b.get("turn_deg") else yaw0)

    def _jump(self, t0, h, x0, y0, x1, y1, yaw):
        """Both feet stay down through the crouch and the push, leave together and land together
        where the flight ends (the root crosses the distance in the air, at constant speed)."""
        c, pu, fl, la, re = jump_times(h)
        s = Segment("jump", t0, t0 + c + pu + fl + la + re, x0, y0, yaw, x1, y1, yaw, jump=(h, c, pu, fl, la, re))
        self.segs.append(s)
        take, land = t0 + c + pu, t0 + c + pu + fl
        for foot in "LR":
            self._step(foot, take, land, x1, y1, yaw, False, True, 0.05)
            self.plants[foot][-2].jump_out = True
            self.plants[foot][-1].jump_in = True

    def jump_at(self, t):
        """The jump under way at t: (segment, seconds since it began), or None."""
        for s in self.segs:
            if s.kind == "jump" and s.t0 <= t <= s.t1: return s, t - s.t0
        return None

    def _turn(self, t0, dur, x, y, yaw0, yaw1):
        s = Segment("turn", t0, t0 + dur, x, y, yaw0, x, y, yaw1)
        self.segs.append(s)
        # Pairs of quick shuffle steps re-square the feet under the turning body, the inside foot
        # first, at most TURN_STEP_DEG a pair: in one pair a big turn swings each foot straight
        # across to the far side, through the other foot.
        first, second = ("L", "R") if yaw1 > yaw0 else ("R", "L")
        n = max(1, math.ceil(abs(math.degrees(yaw1 - yaw0)) / TURN_STEP_DEG - 1e-9))
        d = dur / n
        for k in range(n):
            a, yk = t0 + k * d, lerp(yaw0, yaw1, (k + 1) / n)
            self._step(first, a + 0.12 * d, a + 0.52 * d, x, y, yk, False, False, 0.035, TURN_WIDE)
            self._step(second, a + 0.50 * d, a + 0.90 * d, x, y, yk, False, False, 0.035, TURN_WIDE)

    def _walk(self, t0, g, x0, y0, yaw, dist, yaw_from=None):
        t0 += LEAD                                       # shift weight onto the stance foot first
        st = 60.0 / g["cadence_spm"]                     # seconds per step
        ramp = (RUN_RAMP_STEPS if g.get("run") else 1) * st
        v = g["speed_ms"]
        if dist / v < ramp: v = dist / ramp              # too short to cruise: slow down
        cruise = max(0.0, dist / v - ramp)
        moving = cruise + 2 * ramp
        S = stance_of(g, v)
        s = Segment("walk", t0, t0 + moving + st, x0, y0, yaw, 0, 0, yaw, gait=g,
                    moving=moving, v=v, step_t=st, stance=S, ramp=ramp,
                    yaw_from=None if yaw_from is None or abs(yaw_from - yaw) < 1e-4 else yaw + (yaw_from - yaw + math.pi) % (2 * math.pi) - math.pi)
        fx, fy = fwd(yaw)
        s.x1, s.y1 = x0 + fx * dist, y0 + fy * dist
        self.segs.append(s)
        c = 2 * st
        # Lead foot L lifts at t0; strikes alternate every step while the root moves.
        lead, other = "L", "R"
        strikes = []
        t_strike = t0 + (1 - S) * c                      # the lead foot's first landing
        while t_strike <= t0 + moving + 1e-6:
            strikes.append(t_strike); t_strike += st
        for k, ts in enumerate(strikes):
            foot = lead if k % 2 == 0 else other
            if k == len(strikes) - 1:
                px, py = s.x1, s.y1                      # the last step lands where the body stops
            else:
                rx, ry, _ = self.root_at(ts)
                sv = self.speed_at(ts)
                # a walking foot lands half a step ahead; a running one half its stance's travel
                span = S * c if g.get("run") else st
                reach = 0.5 * max(sv, 0.25 * v) * span * (1 + 0.08 * self.var * self.rng.uniform(-1, 1))
                if g.get("run"): reach -= RUN_BACK
                px, py = rx + fx * reach, ry + fy * reach
            lift = ts - (1 - S) * c                      # this foot left its previous plant then
            if k < 2:
                lift = t0 if foot == lead else t0 + st   # the first two steps leave the standing pose
            self._step(foot, lift, ts, px, py, yaw, True, k >= 2, g["lift_cm"] / 100.0)
        # Close: each foot that is not under the stopped body steps there, one after the other.
        # Running, a foot leaves when its stance is done, as every running step does: waiting for
        # the body to stop would leave it planted far behind, out of the leg's reach.
        ex, ey = s.x1, s.y1
        t_free, t_done = t0 + moving, -math.inf          # t_done: running, the last closing step's landing
        for foot in sorted("LR", key=lambda f: self.plants[f][-1].land):
            p = self.plants[foot][-1]
            if math.hypot(p.x - ex, p.y - ey) > 0.03:
                lift = max(p.land + S * c, t_done) if g.get("run") else max(t_free, p.land + S * c * 0.5)
                land = lift + (1.2 if g.get("run") else 0.7) * st   # a run comes to a stop with more to lose
                self._step(foot, lift, land, ex, ey, yaw, False, True, 0.04)
                t_free = t_done = land
        s.t1 = max(s.t1, t_free)

    def _step(self, foot, lift, land, x, y, yaw, roll_in, peel_out, h, wide=0.0):
        """The foot leaves its current plant at `lift` (over the ball if peel_out) and lands on a
        new one at `land` (heel first if roll_in)."""
        prev = self.plants[foot][-1]
        prev.lift, prev.peel_out = lift, peel_out and prev.land > -math.inf
        self.plants[foot].append(Plant(x, y, yaw, land, math.inf, roll_in=roll_in, lift_h=h, wide=wide))

    # -- sampling ---------------------------------------------------------------------------
    def root_at(self, t):
        x, y, yaw = 0.0, 0.0, 0.0
        for s in self.segs:
            if t < s.t0: break
            if s.kind == "turn":
                u = ease((t - s.t0) / (s.t1 - s.t0))
                x, y, yaw = s.x0, s.y0, lerp(s.yaw0, s.yaw1, u)
            elif s.kind == "jump":                       # across in the air, at the flight's own speed
                _, c, pu, fl, _, _ = s.jump
                u = min(max((t - s.t0 - c - pu) / fl, 0.0), 1.0)
                x, y, yaw = lerp(s.x0, s.x1, u), lerp(s.y0, s.y1, u), s.yaw0
            else:
                d = self._dist(s, t - s.t0)
                fx, fy = fwd(s.yaw0)
                x, y, yaw = s.x0 + fx * d, s.y0 + fy * d, s.yaw0
                if s.yaw_from is not None:               # the small turn, eased over the first steps
                    yaw = lerp(s.yaw_from, s.yaw0, ease((t - s.t0) / s.ramp))
        return x, y, yaw

    def _dist(self, s, t):
        """Distance along a walk segment after t seconds: ease in over its ramp, cruise, ease out."""
        st, v, m = s.ramp, s.v, s.moving
        t = min(max(t, 0.0), m)
        if t < st:                       # accelerate: v * t^2 / (2 st)
            return v * t * t / (2 * st)
        if t < m - st:
            return v * st / 2 + v * (t - st)
        r = m - t                        # decelerate symmetrically
        total = v * st / 2 + v * (m - 2 * st) + v * st / 2
        return total - v * r * r / (2 * st)

    def speed_at(self, t):
        for s in self.segs:
            if s.kind == "walk" and s.t0 <= t <= s.t0 + s.moving:
                u = t - s.t0
                if u < s.ramp: return s.v * u / s.ramp
                if u > s.moving - s.ramp: return s.v * (s.moving - u) / s.ramp
                return s.v
        return 0.0

    def gait_at(self, t):
        """(amplitude 0..1, stride phase 0..1 with 0 = left heel strike, gait params, segment)."""
        for s in self.segs:
            if s.kind == "walk" and s.t0 <= t <= s.t1:
                a = self.speed_at(t) / s.gait["speed_ms"]
                c = 2 * s.step_t
                phase = ((t - s.t0) / c + s.stance) % 1.0
                return min(a, 1.0), phase, s.gait, s
        return 0.0, 0.0, None, None

    def foot_at(self, foot, t):
        """The foot's state: plant frame and how it is posed relative to it.
        Returns (x, y, yaw, kind, data) with kind 'stance' (data: time since landing, time to lift,
        plant) or 'swing' (data: u, from plant, to plant)."""
        ps = self.plants[foot]
        for i, p in enumerate(ps):
            if p.land <= t < p.lift or (p.land <= t and p.lift == math.inf):
                return p.x, p.y, p.yaw, "stance", (t - p.land, p.lift - t, p)
            nxt = ps[i + 1] if i + 1 < len(ps) else None
            if nxt and p.lift <= t < nxt.land:
                u = (t - p.lift) / max(nxt.land - p.lift, 1e-6)
                return None, None, None, "swing", (u, p, nxt)
        p = ps[0]
        return p.x, p.y, p.yaw, "stance", (math.inf, math.inf, p)

    def end_time(self):
        return max([s.t1 for s in self.segs] + [0.0])
