"""The fly brain plays Fly Dash through its own escape circuit.

Every game frame (1/60 s):
  1. eye      the game reports theta_dot, how fast the next spike grows in view (rad/s)
  2. alarm    LPLC2 and LC4 fire at  rate = r_ref * (theta_dot / REF_LOOM) ** steep
              (Hz, capped at MAX_RATE). REF_LOOM is the looming speed with the spike
              0.6 blocks away, so r_ref reads as "alarm rate at 0.6 blocks"; steep > 1
              means the alarm stays quiet far away and climbs fast up close, as LPLC2 does.
     (with --delay-ms D, the alarm cells get the rate the game showed D ms earlier: the
     fly's eye and early visual processing take ~19 ms, which the model otherwise skips)
  3. brain    the connectome model runs 16.7 ms of fly time (lock-step: the game waits)
  4. button   if either Giant Fiber (DNp01) spiked during that frame, press jump

No "jump now" threshold lives in this code. The alarm grows smoothly as a spike
approaches, and the brain's own wiring decides when the Giant Fiber fires.
r_ref and steep are the only tuned numbers: they shape the alarm, not the decision.

Setups (the controls of Phase 4 use the same code):
  real        FlyWire v783 wiring
  shuffledN   degree-preserving rewiring, seed N
  shuffledNgG the same, with every synapse G times stronger (the gain-matched shuffle:
              brain.activity --match finds the G at which it is as active as the real brain)
  lesion      real wiring, both DNp01 unable to spike
  cut         real wiring with only the LPLC2/LC4 -> DNp01 synapses removed: the targeted
              lesion. The alarm can still reach the Giant Fiber through any longer route
  wire        real wiring, but every neuron except LPLC2, LC4 and DNp01 is silenced (its synapses
              deliver nothing): the Giant Fiber hears only its direct inputs from the alarm cells

    python -m brain.player --difficulty hard --seeds 0-4 --r-ref 5 --steep 2
    python -m brain.player --difficulty hard --seeds 3 --r-ref 2 --steep 3 --watch   # see it play
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import sys
import time

import mlx.core as mx
import numpy as np

from lif import core

import flydash
from brain import neurons
from brain.paths import PACK_V783, RESULTS, shuffled
from brain.stepper import Brain

TICKS_PER_FRAME = flydash.DT / 1e-4          # 166.67 ticks of 0.1 ms
MAX_RATE = 300.0                             # Hz; keeps the alarm in the range we tested
REF_LOOM = flydash.SPEED / (0.6 ** 2 + 0.25)  # theta_dot with the spike 0.6 blocks away, ~17 rad/s


def load(setup: str):
    """(pack, lesion indices) for a setup name."""
    real = core.load_pack(PACK_V783)
    if setup == "real":
        return real, ()
    if setup == "lesion":
        return real, neurons.indices(real)["DNp01"]
    if setup.startswith("shuffled"):
        pack = core.load_pack(shuffled(PACK_V783, int(setup[len("shuffled"):].split("g")[0])))
        neurons.check_same_neurons(real, pack)
        return pack, ()
    if setup == "wire":
        return real, ()
    if setup == "cut":
        return cut_alarm_to_gf(real), ()
    raise ValueError(f"unknown setup {setup!r}")


def cut_alarm_to_gf(pack):
    """A copy of the pack whose LPLC2/LC4 -> DNp01 edges carry no synapses; every other edge is untouched."""
    idx = neurons.indices(pack)
    alarm = np.concatenate([idx["LPLC2"], idx["LC4"]])
    row_ptr, dst = np.asarray(pack.row_ptr), np.asarray(pack.destinations)
    src = np.repeat(np.arange(pack.n_neurons), np.diff(row_ptr))
    cut = np.isin(src, alarm) & np.isin(dst, idx["DNp01"])
    counts = np.asarray(pack.signed_counts).copy()
    removed = int(counts[cut].sum())
    counts[cut] = 0
    manifest = dict(pack.manifest, cut={"edges": int(cut.sum()), "synapses": removed,
                                        "what": "LPLC2 and LC4 -> DNp01 signed counts set to 0"})
    return dataclasses.replace(pack, signed_counts=mx.array(counts), manifest=manifest)


def silence_for(setup: str, pack, idx) -> np.ndarray:
    """Neurons whose synapses deliver nothing in this setup (only 'wire' silences any)."""
    if setup != "wire":
        return np.array([], dtype=np.int64)
    keep = np.concatenate([idx["LPLC2"], idx["LC4"], idx["DNp01"]])
    return np.setdiff1d(np.arange(pack.n_neurons), keep)


def gain_for(setup: str) -> float:
    """Synapse gain for this setup: G for shuffledNgG, else 1 (the validated model)."""
    if setup.startswith("shuffled") and "g" in setup:
        return float(setup.split("g", 1)[1])
    return 1.0


class BrainPlayer:
    """One brain for one level. Call it with an observation, get a jump decision."""

    def __init__(self, pack, idx, r_ref: float, steep: float, lesion=(), seed: int = 0, record: bool = False,
                 silence=(), gain: float = 1.0, delay_ms: float = 0.0):
        self.r_ref, self.steep = r_ref, steep
        # Sensory delay: a conveyor belt of per-tick rates. Each frame puts its rate on the back and the
        # alarm cells get what comes off the front, so they see the game delay_ms late (0 = no belt).
        self.belt = collections.deque([0.0] * round(delay_ms / 0.1))
        self.brain = Brain(pack, {"LPLC2": idx["LPLC2"], "LC4": idx["LC4"]},
                           monitor=idx["DNp01"], lesion=lesion, silence=silence, seed=seed, gain=gain)
        self.frame = 0
        self.trace = [] if record else None
        self.last_rate, self.last_gf = 0.0, 0

    def alarm_rate(self, obs) -> float:
        return min(MAX_RATE, self.r_ref * (obs["theta_dot"] / REF_LOOM) ** self.steep)

    def __call__(self, obs) -> bool:
        rate = self.alarm_rate(obs)
        # 166.67 ticks per frame: alternate 167/167/166 so game time and brain time stay in step
        n = round((self.frame + 1) * TICKS_PER_FRAME) - round(self.frame * TICKS_PER_FRAME)
        self.belt.extend([rate] * n)
        felt = [self.belt.popleft() for _ in range(n)]
        parts, start = [], 0                       # one step per run of equal rates (at most 2 per frame)
        for i in range(1, n + 1):
            if i == n or felt[i] != felt[start]:
                parts.append(self.brain.step(i - start, {"LPLC2": felt[start], "LC4": felt[start]}))
                start = i
        spikes = np.concatenate(parts)
        gf = int(spikes.sum())
        self.last_rate, self.last_gf = rate, gf
        if self.trace is not None:
            first = np.flatnonzero(spikes.any(axis=1))
            self.trace.append({"frame": self.frame, "distance": round(obs["distance"], 4),
                               "theta_dot": round(obs["theta_dot"], 4), "alarm_hz": round(rate, 2),
                               "gf_spikes": gf, "gf_first_ms": round(first[0] / 10, 1) if first.size else None,
                               "y": round(obs["y"], 4)})
        self.frame += 1
        return gf > 0


NOISE_STRIDE = 1_000_000                     # noise seed k plays level s with input-spike seed s + k * NOISE_STRIDE


def play_level(pack, idx, r_ref, steep, seed, difficulty, lesion=(), length=300.0, record=False, on_frame=None,
               noise=0, silence=(), gain=1.0, delay_ms=0.0):
    """noise picks a different random draw of the alarm cells' input spikes; 0 is the draw every earlier run used."""
    env = flydash.make_env(seed, difficulty, length)
    player = BrainPlayer(pack, idx, r_ref, steep, lesion, seed=seed + noise * NOISE_STRIDE, record=record,
                         silence=silence, gain=gain, delay_ms=delay_ms)
    obs, done = env.reset(), False
    jump_d = []                                   # distance to the spike at each takeoff
    while not done:
        jump = player(obs)
        if jump and env.on_ground:
            jump_d.append(obs["distance"])
        if on_frame:
            on_frame(env, obs, player, jump)
        obs, _, done = env.step(jump)
    return {"seed": seed, "noise": noise, "progress": obs["progress"], "won": env.won, "jumps": env.jumps,
            "jump_distances": [round(d, 3) for d in jump_d], "trace": player.trace}


# ------------------------------------------------------------------ watch mode: see it play in the terminal

ORANGE, GREEN, RED, DIM, BOLD, RESET = "\033[38;5;208m", "\033[32m", "\033[31m", "\033[2m", "\033[1m", "\033[0m"


class Watcher:
    """Draws every frame in the terminal: the game, the alarm cells, the jump cell."""

    def __init__(self, title: str, slow: float):
        self.title, self.slow = title, slow
        self.history: list[int] = []
        self.t_last = time.perf_counter()
        print("\033[2J", end="")                      # clear the screen once

    def __call__(self, env, obs, player, jump):
        self.history = (self.history + [player.last_gf])[-120:]
        rate, gf = player.last_rate, player.last_gf
        bar = int(round(30 * (rate / MAX_RATE) ** 0.5))
        fired = "".join(f"{GREEN}|{RESET}" if g else f"{DIM}·{RESET}" for g in self.history)
        jump_txt = (f"{GREEN}{BOLD}{'█' * min(gf, 10)} {gf} spike{'s' if gf != 1 else ''} this frame"
                    f"{'  → JUMP' if env.on_ground else '  (already in the air)'}{RESET}" if gf else f"{DIM}quiet{RESET}")
        lines = [
            f"{BOLD}FLY BRAIN PLAYS FLY DASH{RESET}   {self.title}   {DIM}(Ctrl+C to stop){RESET}",
            "",
            *env.render(width=52).splitlines(),
            "",
            f"next spike      {obs['distance']:5.2f} blocks away",
            f"alarm cells     {ORANGE}{'█' * bar}{RESET}{DIM}{'░' * (30 - bar)}{RESET} {rate:6.1f} per second  {DIM}(LPLC2 + LC4, 314 cells){RESET}",
            f"jump cell       {jump_txt}",
            f"last 2 s        {fired}",
            "",
            f"progress {obs['progress']:4.0%}   jumps {env.jumps:>3}   game time {env.t:5.1f} s"
            f"   {DIM}slow-motion x{self.slow:g}{RESET}",
        ]
        print("\033[H" + "\n".join(l + "\033[K" for l in lines), end="", flush=True)
        wait = flydash.DT * self.slow - (time.perf_counter() - self.t_last)
        if wait > 0:
            time.sleep(wait)
        self.t_last = time.perf_counter()

    def finish(self, r):
        msg = f"{GREEN}{BOLD}WON the level{RESET}" if r["won"] else f"{RED}{BOLD}hit a spike{RESET} at {r['progress']:.0%}"
        print(f"\n\n{msg}\033[K", flush=True)
        time.sleep(2.0)


def parse_seeds(text: str) -> list[int]:
    if "-" in text:
        a, b = text.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(s) for s in text.split(",")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup", default="real", help="real, lesion, cut, wire, shuffledN or shuffledNgG")
    ap.add_argument("--difficulty", default="easy", choices=flydash.LEVELS)
    ap.add_argument("--seeds", default="0-4", help="level seeds, e.g. 0-29 or 1,5,7")
    ap.add_argument("--r-ref", type=float, default=5.0, help="alarm rate (Hz) with the spike 0.6 blocks away")
    ap.add_argument("--steep", type=float, default=2.0, help="how steeply the alarm climbs as the spike nears")
    ap.add_argument("--length", type=float, default=300.0, help="level length in blocks")
    ap.add_argument("--delay-ms", type=float, default=0.0, help="sensory delay before the alarm cells (ms)")
    ap.add_argument("--record", action="store_true", help="save a frame-by-frame trace per level")
    ap.add_argument("--watch", action="store_true", help="draw the game live in the terminal")
    ap.add_argument("--slow", type=float, default=4.0, help="with --watch: slow-motion factor (1 = real time)")
    args = ap.parse_args()

    pack, lesion = load(args.setup)
    idx = neurons.indices(pack)
    silence = silence_for(args.setup, pack, idx)
    t0 = time.perf_counter()
    rows = []
    for seed in parse_seeds(args.seeds):
        watcher = (Watcher(f"setup {args.setup} · {args.difficulty} · level {seed} · "
                           f"alarm r_ref {args.r_ref:g} Hz, steep {args.steep:g}", args.slow) if args.watch else None)
        r = play_level(pack, idx, args.r_ref, args.steep, seed, args.difficulty, lesion, args.length,
                       args.record, on_frame=watcher, silence=silence, gain=gain_for(args.setup),
                       delay_ms=args.delay_ms)
        if watcher:
            watcher.finish(r)
        rows.append(r)
        jd = r["jump_distances"]
        print(f"{args.setup:<10} {args.difficulty:<6} level {seed:>2}: "
              f"{'WON ' if r['won'] else 'died'} at {r['progress']:5.0%}, {r['jumps']:>2} jumps"
              + (f", took off {min(jd):.2f}-{max(jd):.2f} blocks from the spike" if jd else ""), flush=True)
    won = sum(r["won"] for r in rows)
    print(f"\n{args.setup} {args.difficulty} r_ref {args.r_ref:g} steep {args.steep:g}: won {won}/{len(rows)}, "
          f"mean progress {np.mean([r['progress'] for r in rows]):.0%}  ({time.perf_counter() - t0:.0f} s)")
    if args.record:
        tag = f"_d{args.delay_ms:g}" if args.delay_ms else ""
        out = RESULTS / "player" / f"{args.setup}_{args.difficulty}_r{args.r_ref:g}_s{args.steep:g}{tag}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"setup": args.setup, "difficulty": args.difficulty, "r_ref": args.r_ref, "steep": args.steep,
                                   "levels": rows}))
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
