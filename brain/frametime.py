"""Review #6: how slow are the slowest frames?

"1.2x faster than real time" is an average. Playing the real game live, every frame of brain simulation has to
finish within one game frame (16.7 ms), and the brain is busiest just before a jump, when the alarm is loudest:
the worst moment to fall behind. So this plays unseen hard levels with the tuned settings and times every frame
(one BrainPlayer call: 166-167 ticks of simulation, synced back to the CPU), then reports percentiles, overall
and for the frames that matter: alarm on, and the 3 frames before each takeoff (the hard-level jump window
is ~2.7 frames).

Run it on an idle Mac: anything else on the GPU makes the numbers worse than they are.

    python -m brain.frametime                 # real brain, 10 hard levels -> figures/frametime_real.json
    python -m brain.frametime --setup wire
"""

import argparse
import json
import platform
import statistics
import subprocess
import time

import numpy as np

import flydash
from brain import neurons
from brain.paths import FIGURES
from brain.player import BrainPlayer, gain_for, load, silence_for
from brain.tune import TEST

BUDGET_MS = flydash.DT * 1000                 # 16.67 ms


def time_level(pack, idx, c, seed, lesion, silence, gain):
    env = flydash.FlyDash(seed, "hard", 300.0)
    player = BrainPlayer(pack, idx, c["r_ref"], c["steep"], lesion, seed=seed, silence=silence, gain=gain)
    obs, done = env.reset(), False
    ms, alarm, takeoff = [], [], []
    while not done:
        t = time.perf_counter()
        jump = player(obs)
        ms.append((time.perf_counter() - t) * 1000)
        alarm.append(player.last_rate > 0.5)
        takeoff.append(bool(jump and env.on_ground))
        obs, _, done = env.step(jump)
    return np.array(ms), np.array(alarm), np.array(takeoff), env.won


def pct(x):
    x = np.asarray(x)
    return {"frames": int(x.size), "p50": float(np.percentile(x, 50)), "p95": float(np.percentile(x, 95)),
            "p99": float(np.percentile(x, 99)), "max": float(x.max()),
            "over_budget": float((x > BUDGET_MS).mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup", default="real")
    ap.add_argument("--levels", type=int, default=10)
    args = ap.parse_args()

    pack, lesion = load(args.setup)
    idx = neurons.indices(pack)
    silence, gain = silence_for(args.setup, pack, idx), gain_for(args.setup)
    c = json.loads((FIGURES / f"phase3_play_{args.setup}.json").read_text())["chosen"]
    time_level(pack, idx, c, 0, lesion, silence, gain)          # warm-up: compile the Metal kernels

    all_ms, on_ms, pre_ms, won = [], [], [], 0
    for seed in list(TEST)[:args.levels]:
        ms, alarm, takeoff, w = time_level(pack, idx, c, seed, lesion, silence, gain)
        before = np.zeros_like(takeoff)
        for i in np.flatnonzero(takeoff):
            before[max(0, i - 2):i + 1] = True                  # the takeoff frame and the 2 before it
        all_ms += ms.tolist()
        on_ms += ms[alarm].tolist()
        pre_ms += ms[before].tolist()
        won += w
        print(f"level {seed}: {'WON ' if w else 'died'}  p50 {np.percentile(ms, 50):5.1f} ms  "
              f"p99 {np.percentile(ms, 99):5.1f} ms  max {ms.max():5.1f} ms", flush=True)

    out = {"setup": args.setup, "setting": c, "levels": args.levels, "won": won, "budget_ms": BUDGET_MS,
           "machine": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                     text=True).stdout.strip() or platform.processor(),
           "all": pct(all_ms), "alarm_on": pct(on_ms), "before_takeoff": pct(pre_ms)}
    print(f"\n{args.setup}, {args.levels} hard levels, frame budget {BUDGET_MS:.1f} ms:")
    for k in ("all", "alarm_on", "before_takeoff"):
        v = out[k]
        print(f"  {k:<15} {v['frames']:>6} frames  p50 {v['p50']:5.1f}  p95 {v['p95']:5.1f}  p99 {v['p99']:5.1f}  "
              f"max {v['max']:6.1f} ms   over budget {v['over_budget']:.1%}")
    path = FIGURES / f"frametime_{args.setup}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
