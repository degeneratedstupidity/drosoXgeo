"""One-neuron stand-in for the Giant Fiber (SUGGESTIONS.md #1).

If one simulated neuron, fed the same random alarm spikes through a single tuned weight, plays as well as
the connectome, then the game only needs "add up the alarm and fire past a threshold", and the claim
should be about that reflex, not about the brain's wiring.

The neuron uses exactly the model's physics (lif.core constants, the same update order as the engine's
state kernel): voltage leaks toward rest, synaptic input decays, a spike past threshold resets it and
holds it refractory for 2.2 ms, and input arriving while refractory is lost.

Its input: every tick, each of the 314 alarm cells (210 LPLC2 + 104 LC4) fires with probability rate * 0.1 ms,
the same alarm rate the brain player uses. In the brain an input spike makes an alarm cell fire on the next
tick and its synapses deliver 1.8 ms later, so here each alarm spike arrives 19 ticks later and adds
`contacts` synapses' worth of input (W_SYN = 0.275 mV each). In FlyWire an alarm cell makes on average about
2.9 (LPLC2) or 3.9 (LC4) contacts onto each Giant Fiber; here `contacts` is one number, tuned.

Same protocol as every other setup: tune (r_ref, steep, contacts) on practice seeds 0-5 (hard, 150 blocks),
then play the 90 unseen levels once.

    python -m brain.standin            # CPU only, a few minutes; -> figures/phase3_play_standin.json
    python -m brain.standin noise      # replay the 90 levels with noise seeds 1-4 -> figures/noise_seeds_standin.json
"""

from __future__ import annotations

import itertools
import json
import sys
import time

import numpy as np

import flydash
from lif import core
from brain.paths import FIGURES
from brain.player import MAX_RATE, NOISE_STRIDE, REF_LOOM, TICKS_PER_FRAME
from brain.tune import PRACTICE, R_REF, STEEP, TEST

N_ALARM = 210 + 104
CONTACTS = [1, 2, 3, 4, 6]
DELAY = 1 + core.DELAY_TICKS          # alarm cell fires one tick after its input, delivers 18 ticks later


class StandIn:
    def __init__(self, r_ref, steep, contacts, seed):
        self.r_ref, self.steep, self.contacts = r_ref, steep, contacts
        self.rng = np.random.default_rng(seed)
        self.v, self.g, self.rfc = core.V_0, 0.0, 0
        self.pipe = np.zeros(DELAY, dtype=np.int64)        # alarm spikes in flight, by arrival tick
        self.tick = self.frame = 0
        self.first_spike_tick = None                       # for the latency check against Experiment 1

    def __call__(self, obs) -> bool:
        rate = min(MAX_RATE, self.r_ref * (obs["theta_dot"] / REF_LOOM) ** self.steep)
        n = round((self.frame + 1) * TICKS_PER_FRAME) - round(self.frame * TICKS_PER_FRAME)
        draws = self.rng.binomial(N_ALARM, rate * core.DT / 1000.0, size=n)
        spiked = False
        w = self.contacts * core.W_SYN
        for k in range(n):
            slot = self.tick % DELAY
            arriving = self.pipe[slot]
            self.pipe[slot] = draws[k]                     # these arrive DELAY ticks from now
            self.rfc = max(self.rfc - 1, 0)
            if self.rfc == 0:                              # same order as the engine's state kernel
                v = core.V0_TERM + self.g * core.COUPLE_G + self.v * core.DECAY_V
                self.g *= core.DECAY_G
                self.v = v
                sp = self.v > core.V_TH
                self.g += arriving * w
                if sp:
                    self.v, self.g, self.rfc = core.V_0, 0.0, core.RFC_TICKS
                    spiked = True
                    if self.first_spike_tick is None:
                        self.first_spike_tick = self.tick
            self.tick += 1
        self.frame += 1
        return spiked


def play(r_ref, steep, contacts, seed, difficulty, length, noise=0):
    env = flydash.FlyDash(seed, difficulty, length)
    neuron = StandIn(r_ref, steep, contacts, seed + noise * NOISE_STRIDE)
    obs, done, jd = env.reset(), False, []
    while not done:
        jump = neuron(obs)
        if jump and env.on_ground:
            jd.append(obs["distance"])
        obs, _, done = env.step(jump)
    return {"seed": seed, "progress": obs["progress"], "won": env.won, "jumps": env.jumps, "jump_distances": jd}


def noise_repeats():
    """Same treatment the brain got (brain.noise_seeds): the 90 unseen levels with 4 more input draws."""
    from brain.noise_seeds import wilson
    best = json.loads((FIGURES / "phase3_play_standin.json").read_text())
    plays = {(0, d, r["seed"]): r["won"] for d, v in best["test"].items() for r in v["per_level"]}
    b = best["chosen"]
    for n in (1, 2, 3, 4):
        for d in ("easy", "normal", "hard"):
            for s in TEST:
                plays[(n, d, s)] = play(b["r_ref"], b["steep"], b["contacts"], s, d, 300.0, noise=n)["won"]
        print(f"noise {n}: won {sum(w for (nn, _, _), w in plays.items() if nn == n)}/90", flush=True)
    k, total = sum(plays.values()), len(plays)
    lo, hi = wilson(k, total)
    print(f"all plays: won {k}/{total} = {k / total:.1%}, 95% interval {lo:.1%}-{hi:.1%}")
    out = FIGURES / "noise_seeds_standin.json"
    out.write_text(json.dumps({"setting": b, "won": k, "plays": total, "win_rate": k / total, "ci95": [lo, hi],
                               "per_noise": {n: sum(w for (nn, _, _), w in plays.items() if nn == n) for n in range(5)}},
                              indent=2) + "\n")
    print(f"wrote {out.relative_to(FIGURES.parent)}")


def main() -> int:
    if sys.argv[1:] == ["noise"]:
        noise_repeats()
        return 0
    t0 = time.perf_counter()
    grid = []
    for r_ref, steep, c in itertools.product(R_REF, STEEP, CONTACTS):
        rows = [play(r_ref, steep, c, s, "hard", 150.0) for s in PRACTICE]
        g = {"r_ref": r_ref, "steep": steep, "contacts": c,
             "progress": float(np.mean([r["progress"] for r in rows])), "won": sum(r["won"] for r in rows)}
        grid.append(g)
        print(f"practice  r_ref {r_ref:>3} steep {steep} contacts {c}:  won {g['won']}/{len(PRACTICE)}  "
              f"progress {g['progress']:4.0%}", flush=True)
    best = max(grid, key=lambda g: (g["progress"], g["won"], -g["steep"]))     # same tie-break as brain.tune
    print(f"\nchosen on practice levels: r_ref {best['r_ref']} Hz, steep {best['steep']}, contacts {best['contacts']}\n")
    test = {}
    for diff in ("easy", "normal", "hard"):
        rows = [play(best["r_ref"], best["steep"], best["contacts"], s, diff, 300.0) for s in TEST]
        jd = [d for r in rows for d in r["jump_distances"]]
        test[diff] = {"progress": float(np.mean([r["progress"] for r in rows])), "won": sum(r["won"] for r in rows),
                      "levels": len(rows),
                      "takeoff_blocks": {"p10": float(np.percentile(jd, 10)) if jd else None,
                                         "median": float(np.median(jd)) if jd else None,
                                         "p90": float(np.percentile(jd, 90)) if jd else None},
                      "per_level": [{k: r[k] for k in ("seed", "progress", "won", "jumps")} for r in rows]}
        print(f"TEST {diff:<6} won {test[diff]['won']}/{len(rows)}, mean progress {test[diff]['progress']:.0%}", flush=True)
    out = FIGURES / "phase3_play_standin.json"
    out.write_text(json.dumps({"setup": "standin", "practice_seeds": list(PRACTICE), "test_seeds": list(TEST),
                               "practice_grid": grid, "chosen": best, "test": test}, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(FIGURES.parent)}  ({time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
