"""Dopamine practice in Fly Dash on the floating slabs (blocks 147-170), mixed with the earlier sections (2026-11-01).

In the real game most deaths past the staircase are at the two floating slabs over spike rows (153-154: slab
155-157; 161-162: slab 163-165): the cube must land on top of a slab 2 blocks up (a jump reaches 2.1). Recorded in
Fly Dash, both deaths are the layer holding the key: coming down off the staircase it still presses in the last
80 ms before landing at ~150.3, so the real game's landing rule bounces it at once (2.4 blocks too early) onto the
spikes; or, on top of slab 1, it bounces off the slab instead of walking off it and lands on the second spike row.

So the slab section starts *mid-air*, where the planner is at block 146.5 on its way down from the staircase (a
standing start at 148.5 was tried first: the layer jumps at once, which never happens in the real game), and ends
at 170, before the ceiling at 172 the looming cells can't see (README "Practice mode"). The other sections start at
brain.learn's checkpoints. Each attempt picks a section, weak ones more often, and runs to death or the section's
end; the rule is brain.dopamine's (reward per cleared obstacle, punishment at death, ~1 s tags).

    python -m brain.slab_practice <init weights> <out> [--attempts 1500] [--seed 0] [--temp 1] [--rate 0.02]
    python -m brain.slab_practice <weights> --score 20          # deterministic layer, n attempts per section
    python -m brain.slab_practice <weights> --score-random 40   # with the random presses (how --dopamine plays)
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

import flydash
from brain import dopamine, learn

SLAB_FROM, SLAB_TO = 146.5, 170.0


def planner_state(x):
    """The planner's full state (position, vertical speed, on the ground or not) when it first reaches block x."""
    lv, end = learn._W["level"], learn._W["end"]
    jumps = flydash.plan(lv, end)
    env = flydash.FlyDashTerrain(level=lv, length=end, buffer=learn._W["buffer"])
    env.reset()
    f = 0
    while env.px < x:
        env.step(f in jumps)
        f += 1
    return dict(px=env.px, py=env.py, vy=env.vy, grounded=env.grounded)


def sections():
    """(start state, stop x) per section: brain.learn's first four checkpoints, then the slab section."""
    st = learn._W["starts"]
    standing = lambda s: dict(px=s[0], py=s[1], vy=0.0, grounded=True)
    return [(standing(st[0]), st[1][0]), (standing(st[1]), st[2][0]), (standing(st[2]), st[3][0]),
            (standing(st[3]), SLAB_FROM), (planner_state(SLAB_FROM), SLAB_TO)]


def _setup(state, seed, delay):
    W = learn._W
    bp = W["player"].BrainPlayer(W["pack"], W["idx"], learn.R_REF, learn.STEEP, W["lesion"], seed=seed,
                                 delay_ms=delay, vision=W["vision"])
    env = flydash.FlyDashTerrain(level=W["level"], length=W["end"], buffer=W["buffer"])
    env.reset()
    for k, v in state.items():
        setattr(env, k, v)
    return bp, env


def score(w, n=20, random_presses=False):
    """Share of attempts that survive each section: the deterministic layer, or (random_presses) the dopamine
    layer's sampled presses with learning off, which is how a --dopamine run plays."""
    W, out = learn._W, []
    rng = np.random.default_rng(7 if not random_presses else 99)
    for state, stop in sections():
        ok = 0
        for i in range(n):
            bp, env = _setup(state, int(rng.integers(1, 2**31)), float(rng.uniform(*learn.DELAYS)))
            layer = learn.Layer(np.array(w, float), W["vision"])
            dop = dopamine.Dopamine(layer, 0.0, dopamine.TEMP, seed=1000 + i) if random_presses else None
            obs, done = env.observe(), False
            while not done and env.px < stop:
                bp(obs)
                c = bp.brain.last_counts
                obs, _, done = env.step(dop.step(c, env.on_ground, env) if dop else layer(c, env.on_ground, env))
            ok += not env.dead
        out.append(f"{state['px']:.0f}-{stop:.0f}: {ok / n:.0%}")
    return " | ".join(out)


def practice(init, out, attempts, seed, temp, rate, tag_decay):
    saved = json.load(open(init))
    W = learn._W
    layer = learn.Layer(np.array(saved.get("mean", saved.get("weights")), float), W["vision"])
    dop = dopamine.Dopamine(layer, rate, temp, seed, tag_decay)
    points = dopamine.clear_points(W["level"])
    secs = sections()
    rate_ema = np.full(len(secs), 0.5)
    rng = np.random.default_rng(seed + 1)
    t0 = time.perf_counter()
    for k in range(attempts):
        chance = 1 - rate_ema + 0.15
        s = int(rng.choice(len(secs), p=chance / chance.sum()))
        state, stop = secs[s]
        bp, env = _setup(state, int(rng.integers(1, 2**31)), float(rng.uniform(*learn.DELAYS)))
        obs, done = env.observe(), False
        nxt = int(np.searchsorted(points, env.px + 1, side="right"))
        dop.new_attempt()
        while not done and env.px < stop:
            bp(obs)
            obs, _, done = env.step(dop.step(bp.brain.last_counts, env.on_ground, env))
            while nxt < len(points) and env.px + 1 > points[nxt] and not env.dead:
                dop.reward(dopamine.R_CLEAR)
                nxt += 1
        if env.dead:
            dop.reward(dopamine.R_DEATH)
        rate_ema[s] = 0.9 * rate_ema[s] + 0.1 * float(not env.dead)
        if (k + 1) % 100 == 0:
            print(f"[+{(time.perf_counter() - t0) / 60:4.1f} min] attempt {k + 1}: cleared lately "
                  + " ".join(f"{sec[0]['px']:.0f}:{r:.0%}" for sec, r in zip(secs, rate_ema)), flush=True)
            json.dump({**{k2: v for k2, v in saved.items() if k2 not in ("mean", "weights", "reached", "from")},
                       "mean": layer.w.tolist(), "from": f"{init} + brain.slab_practice ({k + 1} attempts, seed "
                       f"{seed}, temp {temp}, rate {rate})"}, open(out, "w"))
    print("scored after practice:", score(layer.w), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("init")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--attempts", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temp", type=float, default=dopamine.TEMP)
    ap.add_argument("--rate", type=float, default=dopamine.RATE)
    ap.add_argument("--tag-decay", type=float, default=0.985)
    ap.add_argument("--score", type=int, default=0, help="score the deterministic layer, n attempts per section")
    ap.add_argument("--score-random", type=int, default=0, help="score with the random presses, learning off")
    a = ap.parse_args()
    learn._init("real")
    if a.score or a.score_random:
        w = json.load(open(a.init))["mean"]
        print(a.init, score(w, a.score or a.score_random, random_presses=bool(a.score_random)), flush=True)
        return 0
    practice(a.init, a.out, a.attempts, a.seed, a.temp, a.rate, a.tag_decay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
