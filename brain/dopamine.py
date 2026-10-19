"""Dopamine-like learning for the decision layer: reward when an obstacle is cleared, punishment at death.

Rajan's idea: let the fly learn in the game itself, with a positive signal when it survives an obstacle and a
negative one when it dies. This is the textbook three-factor rule for dopamine-gated plasticity:

  1. choose   each frame the layer presses with probability  p = sigmoid(weights . features / T)  (a little
              randomness, so it keeps trying variations; T sets how much)
  2. tag      every connection that took part leaves a fading tag (an eligibility trace, ~0.5 s):
                tag = 0.97 * tag + (pressed - p) * features      (per frame; 0.97 per frame = a 0.5 s fade)
              "pressed - p" says whether this frame did more or less pressing than the layer usually would
  3. dopamine  when the cube clears an obstacle:  weights += rate * R_CLEAR * tag
              when it dies:                       weights += rate * R_DEATH * tag   (R_DEATH < 0)
              so recent choices that led to clearing get stronger, those that led to death weaker

The plasticity is in the decision layer only (allowed by the project's rules); the fly's connectome is fixed. In
words for the post: "a dopamine-like reward rule on a small decision layer on top of the fly's looming cells", not
"the fly's mushroom body learned" (Stretch C: the looming signal never reaches the Kenyon cells).

Obstacles (for the reward): every spike, thorn pit and block face on the level, from the level file; "cleared"
when the cube's front passes its far end alive.

    python -m brain.dopamine sim --attempts 300           # test the rule in Fly Dash's Stereo Madness first
    (live: python -m brain.live --press --dopamine ...)   # the same rule, learning in the real game
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from brain.paths import FIGURES, RESULTS
from brain.shadow import SAFE_START

DECAY_TAG = 0.97                 # per frame: tags fade over ~0.5 s (a jump lasts 0.45 s)
R_CLEAR, R_DEATH = 0.3, -1.0
RATE = 0.02
TEMP = 1.0


def clear_points(lv) -> np.ndarray:
    """x positions where an obstacle counts as cleared: the far end of every spike, pit and block face."""
    blocks, thorns, peaks = lv
    xs = [p.x + 1 for p in peaks] + [t.x1 for t in thorns] + [b.x0 + 1 for b in blocks]
    xs = np.unique(np.round(xs, 1))
    keep = [xs[0]]
    for x in xs[1:]:
        if x - keep[-1] >= 0.5:
            keep.append(x)
    return np.array(keep)


class Dopamine:
    """The decision layer with dopamine-like plasticity. Use step() every frame, reward() at events."""

    def __init__(self, layer, rate=RATE, temp=TEMP, seed=0):
        self.layer, self.rate, self.temp = layer, rate, temp
        self.tag = np.zeros_like(layer.w)
        self.rng = np.random.default_rng(seed)

    def step(self, counts, standing: bool) -> bool:
        x = self.layer.features(counts, standing)
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ self.layer.w / self.temp, -30, 30)))
        press = bool(self.rng.random() < p)
        self.tag = DECAY_TAG * self.tag + (float(press) - p) * x / self.temp
        return press

    def reward(self, r: float):
        self.layer.w = self.layer.w + self.rate * r * self.tag

    def new_attempt(self):
        self.tag[:] = 0
        self.layer.mem[:] = 0


class LiveTeacher:
    """The dopamine signal in the real game, from the shadow (brain.shadow): +R_CLEAR when the shadow's cube passes
    an obstacle (half a block past its end, so a death inside it isn't rewarded first), R_DEATH when the real cube
    vanishes for good (a death; the punishment uses the tags from the moment it vanished, not the respawn pause).
    No learning while `learning` is False (e.g. the game isn't the front app: presses don't reach it then).
    Weights and per-attempt results are saved after every death."""

    def __init__(self, dop, points, out, meta):
        from collections import deque
        self.dop, self.points, self.out, self.meta = dop, points, out, meta
        self.recent = deque(maxlen=90)            # (t, tag) for the last ~1.5 s
        self.seen_attempts, self.next, self.punished = 0, 0, True
        self.reached, self.cur, self.last_press = [], 0.0, False
        self.cleared = self.deaths = 0

    def frame(self, t, counts, shadow, visible: bool, learning: bool) -> bool:
        env = shadow.env
        if env is None:
            return False
        if shadow.attempts != self.seen_attempts:              # a new attempt began (respawn)
            self.seen_attempts = shadow.attempts
            self.dop.new_attempt()
            self.next = int(np.searchsorted(self.points, env.px + 1))
            self.punished, self.cur = False, env.px
        if not visible:                                        # cube hidden: a death, or a flicker mid-jump
            if (not self.punished and shadow.missing_since is not None and t - shadow.missing_since > 0.25
                    and env.px > SAFE_START):                    # earlier, it's the respawn fading in
                then = [tag for tt, tag in self.recent if tt <= shadow.missing_since]
                if learning and then:
                    self.dop.layer.w = self.dop.layer.w + self.dop.rate * R_DEATH * then[-1]
                self.punished = True
                self.deaths += 1
                self.reached.append(round(self.cur, 1))
                self.save()
                return self.report()
            return self.last_press
        press = self.dop.step(counts, env.on_ground)
        self.recent.append((t, self.dop.tag.copy()))
        while self.next < len(self.points) and env.px + 1 > self.points[self.next] + 0.5:
            if learning:
                self.dop.reward(R_CLEAR)
            self.cleared += 1
            self.next += 1
        self.cur = max(self.cur, env.px)
        self.last_press = press
        return press

    def report(self) -> bool:
        r = np.array(self.reached)
        last = r[-20:]
        print(f"attempt {len(r):5d}: reached block {r[-1]:6.1f} | best {r.max():6.1f} | last {len(last)}: median "
              f"{np.median(last):5.1f}, past the towers {np.mean(last > 60):.0%}, past 145 {np.mean(last > 145):.0%}",
              flush=True)
        return False

    def save(self):
        self.out.write_text(json.dumps({**self.meta, "mean": self.dop.layer.w.tolist(), "reached": self.reached,
                                        "cleared": self.cleared, "deaths": self.deaths}) + "\n")


def sim(attempts: int, init: str, rate: float, temp: float, seed: int = 0, report: int = 25):
    """The rule in Fly Dash's Stereo Madness, attempts from the start (as normal mode), random 30-55 ms lag."""
    import flydash
    from brain import learn
    learn._init("real")
    w0 = np.array(json.loads(open(init).read())["mean"])
    W = learn._W
    layer = learn.Layer(w0, W["vision"])
    dop = Dopamine(layer, rate, temp, seed)
    points = clear_points(W["level"])
    rng = np.random.default_rng(seed + 1)
    log, reached, t0 = {"init": init, "rate": rate, "temp": temp, "r_clear": R_CLEAR, "r_death": R_DEATH,
                        "attempts": []}, [], time.perf_counter()
    for k in range(attempts):
        bp = W["player"].BrainPlayer(W["pack"], W["idx"], learn.R_REF, learn.STEEP, W["lesion"],
                                     seed=int(rng.integers(1, 2**31)), delay_ms=float(rng.uniform(*learn.DELAYS)),
                                     vision=W["vision"])
        env = flydash.FlyDashTerrain(level=W["level"], length=W["end"])
        obs, done, nxt = env.reset(), False, 0
        dop.new_attempt()
        while not done:
            bp(obs)
            press = dop.step(bp.brain.last_counts, env.on_ground)
            obs, _, done = env.step(press)
            while nxt < len(points) and env.px + 1 > points[nxt] and not env.dead:
                dop.reward(R_CLEAR)
                nxt += 1
        if env.dead:
            dop.reward(R_DEATH)
        reached.append(env.px)
        log["attempts"].append(round(env.px, 1))
        if (k + 1) % report == 0:
            last = np.array(reached[-report:])
            print(f"[+{(time.perf_counter() - t0) / 60:5.1f} min] attempts {k + 1 - report + 1}-{k + 1}: median block "
                  f"{np.median(last):.0f}, past the towers (60) {np.mean(last > 60):.0%}, past 145 "
                  f"{np.mean(last > 145):.0%}, reached the ship {np.mean(last >= W['end'] - 0.5):.0%}", flush=True)
            log["weights"] = layer.w.tolist()
            (RESULTS / "dopamine_sim.json").write_text(json.dumps(log) + "\n")
    return log


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["sim"])
    ap.add_argument("--attempts", type=int, default=300)
    ap.add_argument("--init", default="figures/learn_real_gen25.json")
    ap.add_argument("--rate", type=float, default=RATE)
    ap.add_argument("--temp", type=float, default=TEMP)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    sim(a.attempts, a.init, a.rate, a.temp, a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
