"""Watch-only: the fly brain watches a recording of the real Geometry Dash, and we see when it would have jumped.

Nothing is pressed. Each recorded frame goes through the eye (brain.realeye: distance -> theta_dot), the alarm cells
(LPLC2/LC4 fire at r_ref * (theta_dot / REF_LOOM) ** steep, as in brain.player) and the connectome model, which runs
for as long as that frame was on screen (lock-step, so the replay doesn't need to be real time). A Giant Fiber
(DNp01) spike = "would jump now".

For each approach to an obstacle (the distance counts down, then resets), it compares the brain's first Giant Fiber
spike with the human's takeoff in the same recording. Controls as always: lesion (DNp01 silenced), shuffled0.

    python -m brain.realwatch results/realgame/rec_YYYYMMDD-HHMMSS.npz --setups real,lesion,shuffled0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from brain import neurons
from brain.player import MAX_RATE, REF_LOOM, load
from brain.realeye import Eye
from brain.stepper import Brain

TICK = 1e-4                      # s


def approaches(dist: list) -> list[tuple[int, int]]:
    """(first, last) frame of each approach: a run where an obstacle is seen and the distance doesn't jump up."""
    runs, start = [], None
    for i, d in enumerate(dist):
        if d is None or d >= 12:
            if start is not None:
                runs.append((start, i - 1))
            start = None
        elif start is None:
            start = i
        elif d > dist[i - 1] + 1.0:          # a new obstacle took over
            runs.append((start, i - 1))
            start = i
    if start is not None:
        runs.append((start, len(dist) - 1))
    return [(a, b) for a, b in runs if b - a >= 3]


def human_takeoffs(tops: list) -> list[int]:
    """Frames where the cube leaves the ground: it rises >= 3 px after two frames at the same height."""
    out = []
    for i in range(2, len(tops)):
        a, b, c = tops[i - 2], tops[i - 1], tops[i]
        if None not in (a, b, c) and abs(a - b) <= 1 and c <= b - 3:
            out.append(i - 1)                # the last frame on the ground
    return out


def watch(setup: str, dist: list, t: np.ndarray, r_ref: float, steep: float, seed: int = 0) -> list[int]:
    """Giant Fiber spikes per frame for this brain watching the recording."""
    pack, lesion = load(setup)
    idx = neurons.indices(pack)
    brain = Brain(pack, {"LPLC2": idx["LPLC2"], "LC4": idx["LC4"]}, monitor=idx["DNp01"], lesion=lesion, seed=seed)
    gf, clock = [], 0.0
    for i, d in enumerate(dist):
        end = t[i + 1] if i + 1 < len(t) else t[i] + (t[i] - t[i - 1])
        n = round((end - t[0]) / TICK) - round(clock / TICK)   # ticks this frame was on screen, no drift
        clock = end - t[0]
        rate = 0.0 if d is None else min(MAX_RATE, r_ref * ((10.4 / (d * d + 0.25)) / REF_LOOM) ** steep)
        gf.append(int(brain.step(n, {"LPLC2": rate, "LC4": rate}).sum()) if n > 0 else 0)
    return gf


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", type=Path)
    ap.add_argument("--setups", default="real,lesion,shuffled0")
    ap.add_argument("--r-ref", type=float, default=5.0)
    ap.add_argument("--steep", type=float, default=2.0)
    args = ap.parse_args()

    rec = np.load(args.recording)
    t = rec["t"]
    eye = Eye()
    seen = [eye(f) for f in rec["frames"]]
    dist = [None if s is None else s["distance"] for s in seen]
    tops = [None if s is None else s["cube"][0] for s in seen]
    runs = approaches(dist)
    took_off = human_takeoffs(tops)
    print(f"{len(t)} frames ({1 / np.median(np.diff(t)):.0f} per s), {len(runs)} approaches, "
          f"{len(took_off)} human takeoffs")

    result = {"recording": args.recording.name, "r_ref": args.r_ref, "steep": args.steep, "setups": {}}
    for setup in args.setups.split(","):
        t0 = time.perf_counter()
        gf = watch(setup, dist, t, args.r_ref, args.steep)
        rows = []
        for a, b in runs:
            human = next((i for i in took_off if a <= i <= b), None)
            fly = next((i for i in range(a, b + 1) if gf[i]), None)
            rows.append({"frames": [a, b], "closest": round(min(dist[a: b + 1]), 2),
                         "human_d": None if human is None else round(dist[human], 2),
                         "fly_d": None if fly is None else round(dist[fly], 2)})
        result["setups"][setup] = {"approaches": rows, "gf_spikes": gf}
        fired = [r for r in rows if r["fly_d"] is not None]
        print(f"\n{setup}: Giant Fiber fired in {len(fired)}/{len(rows)} approaches, {sum(gf)} spikes "
              f"({time.perf_counter() - t0:.0f} s)")
        print("  approach  t(s)    closest  human took off  fly would jump")
        for k, r in enumerate(rows):
            h = "-" if r["human_d"] is None else f"{r['human_d']:.2f} blocks"
            f = "-" if r["fly_d"] is None else f"{r['fly_d']:.2f} blocks"
            print(f"  {k:>8}  {t[r['frames'][0]]:5.1f}  {r['closest']:7.2f}  {h:>14}  {f:>14}")
    out = args.recording.with_suffix(".watch.json")
    out.write_text(json.dumps(result))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
