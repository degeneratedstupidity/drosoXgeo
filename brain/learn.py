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
            optional (--ground, 2026-11-01): where the ground under the standing cube ends (flydash ground_edge():
            a step down or a gap, in 5 distance bins), from the level layout like the obstacles. Not a looming
            signal: a ledge falling away doesn't loom, and without it the second half can't be learned (README
            "Practice mode"). Said plainly: "looming cells plus a ground-edge cue".
  press     space is held while  weights . features > 0

Learning: the cross-entropy method (try a population of weight vectors, keep the best quarter, refit, repeat), with
practice-mode checkpoints. The level is cut into sections at checkpoints (the start, and standing spots on the
planner's winning run near blocks 40, 80, ..., 240). Each candidate makes 2 attempts at each section, starting at its
checkpoint and stopping at the next one; fitness = the share of each section it clears, every section weighted
equally (a real run must clear all of them in a row). Starting only at 0, every candidate died at the first hard
spot (block 79) and never practised the rest; scoring "share of the remaining way to the ship" instead let late,
short sections outweigh losing the early ones.
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
AZ_EDGES = ()                              # inner azimuth band edges (deg); none = one band (as the first runs)
AZ_FINE = (25.0, 50.0)                     # --fine: split by how far to the side cells look (near objects reach them)
DECAY = 0.6
DELAYS = (30.0, 55.0)                      # ms, drawn per attempt
EDGE_BINS = np.array([-1.0, 0.0, 0.5, 1.0, 2.0, 4.0])   # --ground: blocks from the cube's front to the ground's end
EDGE_KINDS = ("step", "gap")


def n_features(az_edges=AZ_EDGES, ground=False):
    return 2 * (len(BANDS) - 1) * (len(az_edges) + 1) + 2 + 1 + 1 + (len(EDGE_KINDS) * (len(EDGE_BINS) - 1)
                                                                      if ground else 0)


def edge_features(env) -> np.ndarray:
    """The ground-edge cue: one-hot (kind, distance bin) of env.ground_edge(), all zero in the air or with no edge."""
    nb = len(EDGE_BINS) - 1
    x = np.zeros(len(EDGE_KINDS) * nb)
    e = env.ground_edge() if env is not None else None
    if e is not None:
        i = int(np.digitize(e[0], EDGE_BINS)) - 1
        if 0 <= i < nb:
            x[EDGE_KINDS.index(e[1]) * nb + i] = 1.0
    return x


def with_ground(weights, az_edges=AZ_EDGES):
    """Weights without the ground-edge cue -> with it (its weights start at 0: the same decisions as before)."""
    w = np.asarray(weights, float)
    return w if w.size == n_features(az_edges, True) else np.concatenate([w, np.zeros(n_features(az_edges, True) -
                                                                                      w.size)])


N_FEAT = n_features()
CHECKPOINTS = (0, 40, 80, 120, 160, 200, 240)   # blocks; each becomes the planner's last standing spot before it
PER_SECTION = 2                                  # attempts per section per candidate
STD_FLOOR = 0.05

_W = {}                                    # per worker process: the brain, vision, level


class Layer:
    """The decision layer: from one frame's spikes (LPLC2, LC4, DNp01 order) and whether the cube stands, press?"""

    def __init__(self, weights, vision, az_edges=AZ_EDGES, ground=False):
        self.w = np.asarray(weights, dtype=float)
        self.ground = ground
        el = np.concatenate([vision.groups["LPLC2"][1], vision.groups["LC4"][1]])
        az = np.concatenate([vision.groups["LPLC2"][0], vision.groups["LC4"][0]])
        n_az = len(az_edges) + 1
        self.band = np.digitize(el, BANDS[1:-1]) * n_az + np.digitize(az, az_edges)
        self.nb = (len(BANDS) - 1) * n_az
        self.mem = np.zeros(self.nb + 1)

    def features(self, counts, standing: bool, env=None) -> np.ndarray:
        """This frame's input to the layer (and the fading memory updated with it). env: the Fly Dash level (or the
        shadow's), for the ground-edge cue."""
        nb = self.nb
        now = np.log1p(np.bincount(self.band, weights=counts[:314], minlength=nb))
        gf = np.log1p(counts[314:].sum())
        self.mem[:nb] = DECAY * self.mem[:nb] + now
        self.mem[nb] = DECAY * self.mem[nb] + gf
        x = np.concatenate([now, self.mem[:nb], [gf, self.mem[nb], float(standing), 1.0]])
        return np.concatenate([x, edge_features(env)]) if self.ground else x

    def __call__(self, counts, standing: bool, env=None) -> bool:
        return bool(self.features(counts, standing, env) @ self.w > 0)


def _init(setup: str, az_edges=AZ_EDGES, buffer=None, ground=False):
    """buffer: the real game's jump-on-landing window (s); default brain.stereo.REAL_BUFFER. ground: the layer gets
    the ground-edge cue."""
    from brain import neurons, player
    from brain.fastpath import FastPathway
    from brain.stereo import REAL_BUFFER, level
    from brain.vision import Vision
    pack, lesion = player.load(setup)
    idx = neurons.indices(pack)
    fp = FastPathway(pack, idx, lesion=len(lesion) > 0)
    player.Brain = lambda *a, seed=0, **k: (fp.reset(seed), fp)[1]      # BrainPlayer reuses this one pathway
    vision = Vision(pack, idx)
    el = np.concatenate([vision.groups["LPLC2"][1], vision.groups["LC4"][1]])
    lv, end = level()
    _W.update(pack=pack, idx=idx, lesion=lesion, vision=vision, band=np.digitize(el, BANDS[1:-1]), level=lv,
              end=end, player=player, starts=checkpoints(lv, end), az_edges=tuple(az_edges),
              buffer=REAL_BUFFER if buffer is None else buffer, ground=ground)


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


def attempt(weights, seed: int, delay_ms: float, record: bool = False, start=(0.0, 0.0), stop=None):
    """One attempt at Stereo Madness up to the ship, from `start` (x, height; default the level's start), ending at
    x = `stop` if given (a section). Returns progress (share of the way from start to stop, or to the ship), won (got
    there), and optionally the frames."""
    import flydash
    w = _W
    bp = w["player"].BrainPlayer(w["pack"], w["idx"], R_REF, STEEP, w["lesion"], seed=seed, delay_ms=delay_ms,
                                 vision=w["vision"])
    env = flydash.FlyDashTerrain(level=w["level"], length=w["end"], buffer=w.get("buffer", 0.0))
    env.reset()
    env.px, env.py = start
    obs, done = env.observe(), False
    layer, frames = Layer(weights, w["vision"], w.get("az_edges", AZ_EDGES), w.get("ground", False)), []
    goal = w["end"] if stop is None else stop
    while not done and env.px < goal:
        bp(obs)
        c = bp.brain.last_counts
        press = layer(c, env.on_ground, env)
        if record:
            frames.append({"x": round(env.px, 2), "y": round(env.py, 2), "press": press, "gf": int(c[314:].sum())})
        obs, _, done = env.step(press)
    progress = min((env.px - start[0]) / (goal - start[0]), 1.0)
    won = env.won or (stop is not None and env.px >= stop)
    return (progress, won, frames) if record else (progress, won)


def _fitness(job):
    weights, seeds, delays = job
    starts = _W["starts"]
    stops = [s[0] for s in starts[1:]] + [_W["end"]]
    sections = [(st, sp) for st, sp in zip(starts, stops) for _ in range(PER_SECTION)]
    return float(np.mean([attempt(weights, s, d, start=st, stop=sp)[0]
                          for s, d, (st, sp) in zip(seeds, delays, sections)]))


def expand(weights, az_edges):
    """Coarse layer weights (one azimuth band) -> a layer with more azimuth bands: each elevation band's weight is
    copied to all its azimuth sub-bands, so the finer layer starts out doing roughly what the coarse one learned (not
    exactly: each band's count goes through log(1 + count), so splitting a band changes its sum a little; on 200
    random spike patterns the two made the same 200 decisions)."""
    w, nb, k = np.asarray(weights, float), len(BANDS) - 1, len(az_edges) + 1
    return np.concatenate([np.repeat(w[:nb], k), np.repeat(w[nb:2 * nb], k), w[2 * nb:]])


def train(setup: str, generations: int, pop: int, workers: int, seed: int = 0, az_edges=AZ_EDGES, tag: str = "",
          init: str | None = None, ground: bool = False):
    rng = np.random.default_rng(seed)
    nf = n_features(az_edges, ground)
    mean, std = np.zeros(nf), np.full(nf, 1.0)
    if init:                                          # warm start from an earlier run's mean (coarse or same size)
        saved = json.loads(open(init).read())
        start = np.array(saved.get("mean", saved.get("weights")))
        if start.size != n_features(az_edges, saved.get("ground_edge", False)):
            start = expand(start, az_edges)
        mean = with_ground(start, az_edges) if ground else start
        std = np.full(nf, 0.3)
    out = FIGURES / f"learn_{setup}{tag}.json"
    log = {"setup": setup, "r_ref": R_REF, "steep": STEEP, "bands": BANDS.tolist(), "az_edges": list(az_edges),
           "ground_edge": ground, "init": init,
           "decay": DECAY,
           "delays_ms": list(DELAYS), "pop": pop, "generations": []}
    best_w, best_f = None, -1.0
    t0 = time.perf_counter()
    with Pool(workers, initializer=_init, initargs=(setup, az_edges, None, ground)) as pool:
        for gen in range(generations):
            cands = mean + std * rng.standard_normal((pop, nf))
            seeds = rng.integers(1, 2**31, size=len(CHECKPOINTS) * PER_SECTION)
            delays = rng.uniform(*DELAYS, size=len(CHECKPOINTS) * PER_SECTION)
            fit = np.array(pool.map(_fitness, [(c, seeds, delays) for c in cands]))
            elite = cands[np.argsort(-fit)[: max(2, pop // 4)]]
            mean, std = elite.mean(0), elite.std(0) + STD_FLOOR      # a floor keeps it exploring
            if fit.max() > best_f:
                best_f, best_w = float(fit.max()), cands[int(np.argmax(fit))]
            log["generations"].append({"gen": gen, "best": float(fit.max()), "mean": float(fit.mean()),
                                       "elite_mean": float(np.sort(fit)[-len(elite):].mean())})
            print(f"[+{(time.perf_counter() - t0) / 60:5.1f} min] gen {gen:3d}: best {fit.max():.0%}, elite "
                  f"{np.sort(fit)[-len(elite):].mean():.0%}, population {fit.mean():.0%} of each section cleared "
                  f"(sections weighted equally)", flush=True)
            log["mean"], log["best_weights"], log["best_fitness"] = mean.tolist(), best_w.tolist(), best_f
            out.write_text(json.dumps(log, indent=1) + "\n")
    return log


def score(setup: str, n: int, workers: int, which: str = "mean", tag: str = ""):
    path = FIGURES / f"learn_{setup}{tag}.json"
    log = json.loads(path.read_text())
    weights = np.array(log[which])
    rng = np.random.default_rng(12345)
    seeds = rng.integers(1, 2**31, size=n)
    delays = rng.uniform(*DELAYS, size=n)
    with Pool(workers, initializer=_init, initargs=(setup, tuple(log.get("az_edges", ())), None,
                                                    log.get("ground_edge", False))) as pool:
        res = pool.starmap(attempt, [(weights, int(s), float(d)) for s, d in zip(seeds, delays)])
    prog = [p for p, _ in res]
    won = sum(w for _, w in res)
    print(f"{setup}, {which} weights, {n} fresh attempts: reached the ship {won}/{n}, mean progress "
          f"{np.mean(prog):.0%} (median {np.median(prog):.0%})")
    log.setdefault("scores", {})[which] = {"attempts": n, "won": int(won), "progress": prog}
    path.write_text(json.dumps(log, indent=1) + "\n")
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
    ap.add_argument("--fine", action="store_true", help="also split the cells by azimuth (3 bands): 52 weights")
    ap.add_argument("--init", default=None, help="warm start from this run's mean weights (a figures/learn_*.json)")
    ap.add_argument("--ground", action="store_true", help="add the ground-edge cue (10 more weights, start at 0)")
    a = ap.parse_args()
    edges, tag = (AZ_FINE, "_fine") if a.fine else (AZ_EDGES, "")
    tag += "_ground" if a.ground else ""
    if a.cmd == "train":
        train(a.setup, a.generations, a.pop, a.workers, az_edges=edges, tag=tag, init=a.init, ground=a.ground)
    else:
        score(a.setup, a.attempts, a.workers, a.weights, tag=tag)
    return 0


if __name__ == "__main__":
    sys.exit(main())
