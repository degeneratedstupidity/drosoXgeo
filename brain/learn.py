"""A learned decision layer on top of the fly's looming cells, trained by trial and error on Stereo Madness.

Rajan's goal: the fly clears Stereo Madness up to the ship (block 266). The escape reflex alone can't (README
"The block sections"; Stretch C), so here a small layer *learns* when to press. Claim, precisely: the fly's looming
cells (LPLC2/LC4 with FlyWire receptive fields, brain.vision) and its Giant Fiber are the eyes and the reflex; ~20
learned weights on top decide. The connectome is fixed; only the layer learns (allowed by the project's rules).

The layer, every frame (1/60 s):
  features  the 314 looming cells' spikes in this frame, pooled into 8 bands by where the cells look (elevation:
            below the horizon ... above), log(1 + count); a fading memory of each band (x 0.6 per frame); the Giant
            Fiber's spikes and their memory; whether the cube stands on something (a fly feels that through its
            legs; without it the layer can't tell "just landed, wait" from "in the air"); a constant
  press     space is held while  weights . features > 0

Learning: the cross-entropy method (try a population of weight vectors, keep the best quarter, refit, repeat), with
practice-mode checkpoints. Each candidate makes one attempt from each checkpoint (the start, and standing spots on the
planner's winning run near blocks 40, 80, ..., 240); fitness = the share of the way to the ship it covers from each,
averaged. Starting only at 0, every candidate died at the first hard spot (block 79) and never practised the rest.
Every attempt has its own random input spikes and its own delay of 30-55 ms (the real game's lag: capture -> decision ~8 ms, press -> visible ~15-66 ms, median ~37).
Level: the real Stereo Madness layout in Fly Dash physics (brain.stereo). Scored afterwards on 30 fresh attempts,
all from the start (block 0), like a normal-mode run.

    python -m brain.learn train --setup real            # ~1 h, 4 attempts at a time; -> figures/learn_<setup>.json
    python -m brain.learn score --setup real            # 30 fresh attempts with the learned weights
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

from brain.paths import FIGURES, RESULTS

R_REF, STEEP = 25.0, 1.0                   # vision alarm as tuned for the real game's lag (Stretch C stage 1)
BANDS = np.array([-90, -40, -25, -10, 0, 10, 25, 40, 90])   # elevation band edges (deg), 8 bands
DECAY = 0.6
DELAYS = (30.0, 55.0)                      # ms, drawn per attempt
N_FEAT = 2 * (len(BANDS) - 1) + 2 + 1 + 1
CHECKPOINTS = (0, 40, 80, 120, 160, 200, 240)   # blocks; each becomes the planner's last standing spot before it
STD_FLOOR = 0.05

_W = {}                                    # per worker process: the brain, vision, level


class Layer:
    """The decision layer: from one frame's spikes (LPLC2, LC4, DNp01 order) and whether the cube stands, press?"""

    def __init__(self, weights, vision):
        self.w = np.asarray(weights, dtype=float)
        el = np.concatenate([vision.groups["LPLC2"][1], vision.groups["LC4"][1]])
        self.band = np.digitize(el, BANDS[1:-1])
        self.nb = len(BANDS) - 1
        self.mem = np.zeros(self.nb + 1)

    def __call__(self, counts, standing: bool) -> bool:
        nb = self.nb
        now = np.log1p(np.bincount(self.band, weights=counts[:314], minlength=nb))
        gf = np.log1p(counts[314:].sum())
        self.mem[:nb] = DECAY * self.mem[:nb] + now
        self.mem[nb] = DECAY * self.mem[nb] + gf
        x = np.concatenate([now, self.mem[:nb], [gf, self.mem[nb], float(standing), 1.0]])
        return bool(x @ self.w > 0)


def _init(setup: str):
    from brain import neurons, player
    from brain.fastpath import FastPathway
    from brain.stereo import level
    from brain.vision import Vision
    pack, lesion = player.load(setup)
    idx = neurons.indices(pack)
    fp = FastPathway(pack, idx, lesion=len(lesion) > 0)
    player.Brain = lambda *a, seed=0, **k: (fp.reset(seed), fp)[1]      # BrainPlayer reuses this one pathway
    vision = Vision(pack, idx)
    el = np.concatenate([vision.groups["LPLC2"][1], vision.groups["LC4"][1]])
    lv, end = level()
    _W.update(pack=pack, idx=idx, lesion=lesion, vision=vision, band=np.digitize(el, BANDS[1:-1]), level=lv,
              end=end, player=player, starts=checkpoints(lv, end))


def checkpoints(lv, end):
    """Standing spots (x, height) on the planner's winning run, the last one at or before each CHECKPOINTS block
    with nothing ahead within 3 blocks of its front (a fair place to resume, like a practice-mode checkpoint)."""
    import flydash
    jumps = flydash.plan(lv, end)
    env = flydash.FlyDashTerrain(level=lv, length=end)
    obs, done, frame, spots = env.reset(), False, 0, []
    while not done:
        if env.on_ground and obs["distance"] > 3:
            spots.append((env.px, env.py))
        obs, _, done = env.step(frame in jumps)
        frame += 1
    return [max((s for s in spots if s[0] <= c), default=(0.0, 0.0)) for c in CHECKPOINTS]


def attempt(weights, seed: int, delay_ms: float, record: bool = False, start=(0.0, 0.0)):
    """One attempt at Stereo Madness up to the ship, from `start` (x, height; default the level's start).
    Returns progress (share of the way from start to the ship), won, and optionally the frames."""
    import flydash
    w = _W
    bp = w["player"].BrainPlayer(w["pack"], w["idx"], R_REF, STEEP, w["lesion"], seed=seed, delay_ms=delay_ms,
                                 vision=w["vision"])
    env = flydash.FlyDashTerrain(level=w["level"], length=w["end"])
    env.reset()
    env.px, env.py = start
    obs, done = env.observe(), False
    layer, frames = Layer(weights, w["vision"]), []
    while not done:
        bp(obs)
        c = bp.brain.last_counts
        press = layer(c, env.on_ground)
        if record:
            frames.append({"x": round(env.px, 2), "y": round(env.py, 2), "press": press, "gf": int(c[314:].sum())})
        obs, _, done = env.step(press)
    progress = min((env.px - start[0]) / (w["end"] - start[0]), 1.0)
    return (progress, env.won, frames) if record else (progress, env.won)


def _fitness(job):
    weights, seeds, delays = job
    return float(np.mean([attempt(weights, s, d, start=st)[0] for s, d, st in zip(seeds, delays, _W["starts"])]))


def train(setup: str, generations: int, pop: int, workers: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    mean, std = np.zeros(N_FEAT), np.full(N_FEAT, 1.0)
    log = {"setup": setup, "r_ref": R_REF, "steep": STEEP, "bands": BANDS.tolist(), "decay": DECAY,
           "delays_ms": list(DELAYS), "pop": pop, "generations": []}
    best_w, best_f = None, -1.0
    t0 = time.perf_counter()
    with Pool(workers, initializer=_init, initargs=(setup,)) as pool:
        for gen in range(generations):
            cands = mean + std * rng.standard_normal((pop, N_FEAT))
            seeds = rng.integers(1, 2**31, size=len(CHECKPOINTS))
            delays = rng.uniform(*DELAYS, size=len(CHECKPOINTS))
            fit = np.array(pool.map(_fitness, [(c, seeds, delays) for c in cands]))
            elite = cands[np.argsort(-fit)[: max(2, pop // 4)]]
            mean, std = elite.mean(0), elite.std(0) + STD_FLOOR      # a floor keeps it exploring
            if fit.max() > best_f:
                best_f, best_w = float(fit.max()), cands[int(np.argmax(fit))]
            log["generations"].append({"gen": gen, "best": float(fit.max()), "mean": float(fit.mean()),
                                       "elite_mean": float(np.sort(fit)[-len(elite):].mean())})
            print(f"[+{(time.perf_counter() - t0) / 60:5.1f} min] gen {gen:3d}: best {fit.max():.0%}, elite "
                  f"{np.sort(fit)[-len(elite):].mean():.0%}, population {fit.mean():.0%} of the way to the ship "
                  f"(averaged over the checkpoints)", flush=True)
            log["mean"], log["best_weights"], log["best_fitness"] = mean.tolist(), best_w.tolist(), best_f
            (FIGURES / f"learn_{setup}.json").write_text(json.dumps(log, indent=1) + "\n")
    return log


def score(setup: str, n: int, workers: int, which: str = "mean"):
    log = json.loads((FIGURES / f"learn_{setup}.json").read_text())
    weights = np.array(log[which])
    rng = np.random.default_rng(12345)
    seeds = rng.integers(1, 2**31, size=n)
    delays = rng.uniform(*DELAYS, size=n)
    with Pool(workers, initializer=_init, initargs=(setup,)) as pool:
        res = pool.starmap(attempt, [(weights, int(s), float(d)) for s, d in zip(seeds, delays)])
    prog = [p for p, _ in res]
    won = sum(w for _, w in res)
    print(f"{setup}, {which} weights, {n} fresh attempts: reached the ship {won}/{n}, mean progress "
          f"{np.mean(prog):.0%} (median {np.median(prog):.0%})")
    log.setdefault("scores", {})[which] = {"attempts": n, "won": int(won), "progress": prog}
    (FIGURES / f"learn_{setup}.json").write_text(json.dumps(log, indent=1) + "\n")
    return won, prog


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["train", "score"])
    ap.add_argument("--setup", default="real")
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--pop", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--attempts", type=int, default=30)
    ap.add_argument("--weights", default="mean", choices=["mean", "best_weights"])
    a = ap.parse_args()
    if a.cmd == "train":
        train(a.setup, a.generations, a.pop, a.workers)
    else:
        score(a.setup, a.attempts, a.workers, a.weights)
    return 0


if __name__ == "__main__":
    sys.exit(main())
