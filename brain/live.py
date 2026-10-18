"""Live: the fly brain watches the real Geometry Dash as it runs and, with --press, plays it.

The brain is the fly's escape pathway (brain.fastpath: LPLC2 + LC4 + the Giant Fibers, the "wire" setup of Phase 4b,
identical spike for spike to the GPU model, ~4x faster than real time on the CPU). --brain full uses the whole
138,639-neuron GPU model instead, which can't keep up while the alarm is on (1.3-1.5x slower than real time).

Every new frame of the game window (brain.stream, ScreenCaptureKit, game window only):
  1. eye      brain.realeye: distance to the next obstacle in the cube's lane -> theta_dot
  2. alarm    LPLC2 and LC4 fire at r_ref * (theta_dot / REF_LOOM) ** steep Hz, as in brain.player
  3. brain    the connectome model runs for exactly the game time since the previous frame
  4. button   a Giant Fiber (DNp01) spike in that time = jump: with --press, space goes down (and comes back up
              on the next frame without a spike). Without --press nothing is pressed (watch-only).

The game doesn't wait. If the brain is slow, frames pile up: we always take the newest frame and simulate all the
game time we missed, so the brain stays in step with the game but decides late. Every frame is logged with its lag
(time from capture to decision) to results/realgame/live_*.json; that lag is the number Phase 4d says matters
(fine to ~33 ms with the alarm retuned, broken by 50 ms).

Key presses go through FlyJump.app (brain/flyapps.py): space only, sent only to the game's process, never to whatever
window has focus. FlyJump holds the Accessibility permission itself; Python and the IDE need none.

    python -m brain.live --seconds 60                 # watch-only: you play, the terminal shows "WOULD JUMP"
    python -m brain.live --seconds 60 --press         # the fly plays
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time

import numpy as np

from brain import neurons
from brain.fastpath import FastPathway
from brain.player import MAX_RATE, REF_LOOM, load
from brain.paths import RESULTS
from brain.realeye import Eye
from brain.stepper import Brain
from brain.flyapps import Jump
from brain.stream import LiveWindow

TICK = 1e-4                                   # s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--press", action="store_true", help="press jump (the fly plays); default: watch only")
    ap.add_argument("--brain", default="fast", choices=["fast", "full"],
                    help="fast: escape pathway on the CPU (default); full: the whole brain on the GPU")
    ap.add_argument("--setup", default="real", help="real, lesion or shuffledN (controls, as in brain.player)")
    ap.add_argument("--r-ref", type=float, default=5.0, help="tuned in Phase 3/4b: 5 for both brains")
    ap.add_argument("--steep", type=float, default=None, help="default: 1 for fast (Phase 4b), 2 for full")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.steep is None:
        args.steep = 1.0 if args.brain == "fast" else 2.0
    pack, lesion = load(args.setup)
    idx = neurons.indices(pack)
    if args.brain == "fast":
        brain = FastPathway(pack, idx, lesion=len(lesion) > 0, seed=args.seed)
    else:
        brain = Brain(pack, {"LPLC2": idx["LPLC2"], "LC4": idx["LC4"]}, monitor=idx["DNp01"], lesion=lesion,
                      seed=args.seed)
    brain.step(167, {"LPLC2": 0.0, "LC4": 0.0})            # warm up (the GPU model compiles its kernels here)
    eye = Eye()

    stop = []
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    log, last_n, last_t = [], -1, None
    mode = "PLAYING (pressing space)" if args.press else "WATCH-ONLY (nothing pressed)"
    with LiveWindow() as win:
        button = Jump(win.game_pid) if args.press else None
        print(f"{win.message}\n{mode} · {args.brain} brain · setup {args.setup} · r_ref {args.r_ref:g} steep {args.steep:g} · "
              f"{args.seconds:g} s, Ctrl+C to stop\n", flush=True)
        t_start = time.monotonic()
        while not stop and time.monotonic() - t_start < args.seconds:
            got = win.latest(last_n)
            if got is None:
                continue
            n, t_cap, t_recv, rgb = got
            skipped, last_n = (n - last_n - 1 if last_n >= 0 else 0), n
            seen = eye(rgb)
            d = None if seen is None else seen["distance"]
            rate = 0.0 if seen is None else min(MAX_RATE, args.r_ref * (seen["theta_dot"] / REF_LOOM) ** args.steep)
            ticks = 167 if last_t is None else max(1, round((t_cap - last_t) / TICK))   # game time since last frame
            last_t = t_cap
            t0 = time.monotonic()
            gf = int(brain.step(ticks, {"LPLC2": rate, "LC4": rate}).sum())
            t_dec = time.monotonic()
            if button:
                button.set(gf > 0)
            log.append({"n": n, "skipped": skipped, "t": round(t_cap - t_start, 4), "distance": None if d is None
                        else round(d, 3), "alarm_hz": round(rate, 2), "ticks": ticks, "gf": gf,
                        "brain_ms": round((t_dec - t0) * 1000, 2), "lag_ms": round((t_dec - t_cap) * 1000, 2)})
            if gf:
                print(f"{t_cap - t_start:7.2f} s  {'JUMP' if button else 'WOULD JUMP'}  {gf} spike{'s' * (gf > 1)}"
                      f"  next obstacle {d:.2f} blocks  lag {log[-1]['lag_ms']:.0f} ms", flush=True)
        if button:
            button.close()

    lag = np.array([r["lag_ms"] for r in log])
    on = np.array([r["lag_ms"] for r in log if r["alarm_hz"] > 1])
    skipped = sum(r["skipped"] for r in log)
    span = log[-1]["t"] - log[0]["t"] if len(log) > 1 else 0
    print(f"\n{len(log)} frames in {span:.1f} s, {skipped} skipped (the brain was still busy); lag capture -> decision: "
          f"median {np.median(lag):.1f} ms, p95 {np.percentile(lag, 95):.1f} ms"
          + (f"; alarm on: median {np.median(on):.1f}, p95 {np.percentile(on, 95):.1f} ms" if len(on) else ""))
    out = (RESULTS / "realgame" /
           f"live_{time.strftime('%Y%m%d-%H%M%S')}_{args.brain}_{args.setup}{'_press' if button else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"brain": args.brain, "setup": args.setup, "press": bool(button), "r_ref": args.r_ref, "steep": args.steep,
                               "seed": args.seed, "frames": log}))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
