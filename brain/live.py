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
from collections import Counter
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from brain import neurons
from brain.fastpath import FastPathway
from brain.player import MAX_RATE, REF_LOOM, load
from brain.paths import RESULTS
from brain.realeye import Eye, find_cube
from brain.stepper import Brain
from brain.flyapps import Jump
from brain.stream import LiveWindow

TICK = 1e-4                                   # s
MAX_GAP = 500                                 # ticks (50 ms): longer gaps between frames are pauses (death, menus)
WIDTH = 440                                   # frame width in px, whatever the window size: the cube is then
                                              # ~23 px, plenty for the eye, and frames arrive ~4 ms old (~11 ms at
                                              # 2x the size)


class FrontApp:
    """Which app is in front, checked 4 times a second (lsappinfo: built in, no permission). Wine only passes key
    presses to the game while its window is the active one, so a run is only valid while this says "wine"."""

    def __init__(self, t0: float):
        self.t0, self.name, self.changes, self.stop = t0, None, [], False
        threading.Thread(target=self._poll, daemon=True).start()

    def _poll(self):
        while not self.stop:
            try:
                asn = subprocess.run(["lsappinfo", "front"], capture_output=True, text=True).stdout.strip()
                out = subprocess.run(["lsappinfo", "info", "-only", "name", asn], capture_output=True, text=True).stdout
                m = re.search(r'"([^"]*)"', out)
                name = m.group(1) if m else out.strip()
            except OSError:
                name = "?"
            if name != self.name:
                self.name = name
                self.changes.append({"t": round(time.monotonic() - self.t0, 3), "front": name})
            time.sleep(0.25)


RESPAWN_S = 1.07                              # s from death to respawn (Stereo Madness, normal mode: measured)


def attempts(frames: list[dict]) -> list[float]:
    """Blocks reached in each complete attempt (normal mode, so every attempt starts at 0%).

    An attempt starts when the cube respawns, which the game shows as a burst of blinks: the cube flickers on and
    off every ~50 ms at ground height with nothing ahead. Counting deaths instead is unreliable: explosion pieces
    can look like the cube, and the cube flickers mid-jump when rotated. Blocks reached = (time to the next
    respawn - the death-to-respawn pause) * 10.4 blocks/s."""
    t, top = [r["t"] for r in frames], [r["cube_top"] for r in frames]
    seen = [x for x in top if x is not None]
    if not seen:
        return []
    ground = Counter(seen).most_common(1)[0][0]
    short, i = [], 0                                   # start times of short gaps that end on the ground
    while i < len(frames):
        if top[i] is None:
            j = i
            while j < len(frames) and top[j] is None:
                j += 1
            if (j < len(frames) and t[j] - t[i] < 0.1 and abs(top[j] - ground) <= 2
                    and frames[j]["distance"] is not None and frames[j]["distance"] >= 12):
                short.append(t[i])
            i = j
        else:
            i += 1
    starts = []
    for s in short:
        if sum(0 <= x - s < 0.4 for x in short) >= 3 and (not starts or s - starts[-1] > 0.6):
            starts.append(s)
    return [round((b - a - RESPAWN_S) * 10.4, 1) for a, b in zip(starts, starts[1:])]


def tap_test(win, button, seconds: float = 4.0) -> tuple[bool, list]:
    """Tap space for 0.1 s every 0.8 s and watch the cube: (it left the ground, the cube's top per frame)."""
    t0, tops, last = time.monotonic(), [], -1
    while time.monotonic() - t0 < seconds:
        button.set((time.monotonic() - t0) % 0.8 < 0.1)
        got = win.latest(last, timeout=0.2)
        if got:
            last = got[0]
            cube = find_cube(got[3])
            if cube:
                tops.append(cube[0])
    button.set(False)
    return bool(tops) and min(tops) < max(tops) - 10, tops


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
    ap.add_argument("--countdown", type=float, default=0, help="seconds to wait first (to click into the game)")
    ap.add_argument("--tap-test", action="store_true", help="with --press: first check that taps make the cube jump")
    ap.add_argument("--layer", default=None,
                    help="a learned decision layer (figures/learn_<setup>.json, brain.learn): the fly's looming cells "
                         "see the obstacles placed by a shadow of the level (brain.shadow), the layer presses")
    ap.add_argument("--record", action="store_true",
                    help="also save the frames the eye saw (game window only, ~20 MB/s in memory: keep runs short)")
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
    layer = shadow = vision = None
    if args.layer:
        import json as _json
        from brain.learn import Layer
        from brain.shadow import Shadow
        from brain.vision import Vision
        if args.brain != "fast":
            raise SystemExit("--layer needs the fast brain (it was trained on it)")
        trained = _json.loads(Path(args.layer).read_text())
        vision = Vision(pack, idx)
        layer, shadow = Layer(trained["mean"], vision), Shadow()
        args.r_ref, args.steep = trained["r_ref"], trained["steep"]

    stop = []
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    log, last_n, last_t, frames = [], -1, None, []
    mode = "PLAYING (pressing space)" if args.press else "WATCH-ONLY (nothing pressed)"
    button = None
    with LiveWindow(scale=0.25, width=WIDTH) as win:
        try:
            button = Jump(win.game_pid) if args.press else None
            front, warned = FrontApp(time.monotonic()), False
            for k in range(int(args.countdown), 0, -1):
                print(f"starting in {k} s: click into the game window and leave it there", flush=True)
                time.sleep(1)
            if button and args.tap_test:
                ok, tops = tap_test(win, button)
                seen = f"cube seen in {len(tops)} frames" + (f", top {min(tops)}-{max(tops)} px" if tops else "")
                if not ok:
                    print(f"tap test: the cube didn't jump ({seen}; front app {front.name!r}, "
                          f"changes {front.changes})")
                    return 1
                print(f"tap test: the cube jumps, the game takes our keys ({seen}; front app {front.name!r})", flush=True)
                eye = Eye()                    # forget the lane the taps made
            print(f"{win.message}\n{mode} · {args.brain} brain · setup {args.setup} · r_ref {args.r_ref:g} "
                  f"steep {args.steep:g} · {args.seconds:g} s, Ctrl+C to stop\n", flush=True)
            t_start = time.monotonic()
            front.t0 = t_start
            while not stop and time.monotonic() - t_start < args.seconds:
                got = win.latest(last_n)
                if got is None:
                    continue
                n, t_cap, t_recv, rgb = got
                t_got = time.monotonic()
                skipped, last_n = (n - last_n - 1 if last_n >= 0 else 0), n
                seen = eye(rgb)
                if args.record:
                    frames.append(rgb.copy())
                d = None if seen is None else seen["distance"]
                rate = (0.0 if seen is None
                        else min(MAX_RATE, args.r_ref * (seen["theta_dot"] / REF_LOOM) ** args.steep))
                # game time since the last frame; after a pause (no new frames while the cube respawns, or a menu)
                # simulate at most 50 ms of it, or the brain spends the next frame catching up on nothing
                gap = 167 if last_t is None else max(1, round((t_cap - last_t) / TICK))
                ticks = min(gap, MAX_GAP)
                last_t = t_cap
                t0 = time.monotonic()
                if layer is None:
                    gf = int(brain.step(ticks, {"LPLC2": rate, "LC4": rate}).sum())
                    press = gf > 0
                else:
                    env = shadow.frame(t_cap, None if seen is None else seen["cube"], d,
                                       None if seen is None else seen["block_px"])
                    rates = vision.rates(env.objects() if env is not None else [], args.r_ref, args.steep)
                    gf = int(brain.step(ticks, rates).sum())
                    press = layer(brain.last_counts, env is not None and env.on_ground)
                    rate = float(np.mean(np.concatenate([rates["LPLC2"], rates["LC4"]])))
                t_dec = time.monotonic()
                if button:
                    button.set(press)
                ms = lambda a, b: round((b - a) * 1000, 2)
                log.append({"n": n, "skipped": skipped, "t": round(t_cap - t_start, 4),
                            "distance": None if d is None else round(d, 3), "alarm_hz": round(rate, 2),
                            "cube_top": None if seen is None else seen["cube"][0], "pressed": bool(button and press),
                            "shadow_x": None if layer is None or shadow.env is None else round(shadow.env.px, 2),
                            "ticks": ticks, "gap_ticks": gap, "gf": gf,
                            "deliver_ms": ms(t_cap, t_recv),   # FlyEye -> pipe -> Python
                            "wait_ms": ms(t_recv, t_got),      # frame sat here while we were busy with the last one
                            "eye_ms": ms(t_got, t0), "brain_ms": ms(t0, t_dec), "lag_ms": ms(t_cap, t_dec)})
                if button and front.name not in (None, "wine") and not warned:
                    print(f"{t_cap - t_start:7.2f} s  WARNING: the game isn't the front app ({front.name!r}): "
                          f"key presses won't reach it", flush=True)
                    warned = True
                warned = warned and front.name != "wine"
                if gf and layer is None:
                    where = "nothing in view" if d is None else f"next obstacle {d:.2f} blocks"
                    print(f"{t_cap - t_start:7.2f} s  {'JUMP' if button else 'WOULD JUMP'}  {gf} spike{'s' * (gf > 1)}"
                          f"  {where}  lag {log[-1]['lag_ms']:.0f} ms", flush=True)
        finally:
            if button:
                button.close()                 # space up, FlyJump exits
            if "front" in locals():
                front.stop = True

    if not log:
        print("no frames from the game")
        return 1
    lag = np.array([r["lag_ms"] for r in log])
    on = np.array([r["lag_ms"] for r in log if r["alarm_hz"] > 1])
    skipped = sum(r["skipped"] for r in log)
    span = log[-1]["t"] - log[0]["t"] if len(log) > 1 else 0
    print(f"\n{len(log)} frames in {span:.1f} s, {skipped} skipped (the brain was still busy); lag capture -> decision: "
          f"median {np.median(lag):.1f} ms, p95 {np.percentile(lag, 95):.1f} ms"
          + (f"; alarm on: median {np.median(on):.1f}, p95 {np.percentile(on, 95):.1f} ms" if len(on) else ""))
    parts = ", ".join(f"{k[:-3]} {np.median([r[k] for r in log]):.1f}" for k in ("deliver_ms", "wait_ms", "eye_ms", "brain_ms"))
    print(f"where the lag goes (median ms): {parts}")
    reached = attempts(log)
    if reached:
        print(f"complete attempts: {len(reached)}; blocks reached: {reached}; past the first spike (> 20 blocks): "
              f"{sum(r > 20 for r in reached)}/{len(reached)}")
    if args.press:
        print("front app during the run: " + ", ".join(f"{c['front']} from {c['t']:.1f} s" for c in front.changes))
    out = (RESULTS / "realgame" /
           f"live_{time.strftime('%Y%m%d-%H%M%S')}_{args.brain}_{args.setup}{'_press' if args.press else ''}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"brain": args.brain, "setup": args.setup, "press": args.press, "r_ref": args.r_ref, "steep": args.steep,
                               "seed": args.seed, "front_app": front.changes if "front" in locals() else [],
                               "frames": log}))
    print(f"wrote {out}")
    if args.record and frames:
        rec = out.with_suffix(".npz")
        np.savez_compressed(rec, frames=np.stack(frames), t=np.array([r["t"] for r in log[: len(frames)]]))
        print(f"wrote {rec} ({len(frames)} frames; python -m brain.realeye {rec} draws what the eye saw)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
