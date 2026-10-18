"""A live feed of the Geometry Dash window, through ScreenCaptureKit (FlyEye.app, brain/flyeye.swift).

Asking macOS for a fresh snapshot each time costs ~25 ms. Instead FlyEye subscribes to the game window once and
macOS hands it every new frame as the game draws it (~4 ms old when Python gets it); a background thread here keeps
the newest one. Only the game window is captured (FlyEye's filter is that one window). FlyEye is its own app with
its own Screen Recording permission (see brain/flyapps.py); Python needs none.

    python -m brain.stream info                      # frames per second and how old a frame is when we get it
    python -m brain.stream record --seconds 20       # 60 fps recording -> results/realgame/rec_*.npz
"""

from __future__ import annotations

import argparse
import os
import re
import signal
import struct
import sys
import threading
import time

import numpy as np

from brain import flyapps
from brain.paths import RESULTS

OUT = RESULTS / "realgame"
HEADER = struct.Struct("<4sIddII")                  # magic, frame number, capture time, send time, width, height


def _read_exact(f, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


class LiveWindow:
    """Starts FlyEye; .latest() returns (frame number, capture time, received time, RGB array) or None.
    .game_pid is the game's process (for FlyJump)."""

    def __init__(self, scale: float = 0.5, fps: float = 60.0, title: str = "Geometry Dash"):
        pipe = flyapps._fifo("flyeye.out")
        box = []
        opener = threading.Thread(target=lambda: box.append(open(pipe, "rb", buffering=0)), daemon=True)
        opener.start()                         # opening a pipe waits for the other side, so start before launching
        log = flyapps.launch("FlyEye", ["--title", title, "--scale", str(scale), "--fps", str(fps)], stdout=pipe)
        self.message = flyapps.wait_for_status(log, "streaming")
        self.game_pid = int(re.search(r"\bpid (\d+)", self.message).group(1))
        self.app_pid = flyapps.own_pid(self.message)
        opener.join(10.0)
        if not box:
            raise SystemExit("FlyEye started but never opened its output pipe")
        self.f = box[0]
        self._latest = None
        self._cond = threading.Condition()
        self.received = 0
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        try:
            self._read_frames(self.f)
        except (ValueError, OSError):          # close() closed the pipe under us: the normal way to stop
            pass

    def _read_frames(self, f):
        while (head := _read_exact(f, HEADER.size)) is not None:
            magic, n, t_cap, _t_send, w, h = HEADER.unpack(head)
            if magic != b"GDFR":
                raise RuntimeError("FlyEye stream out of sync")
            px = _read_exact(f, w * h * 4)
            if px is None:
                break
            t_recv = time.monotonic()
            with self._cond:
                self._latest = (n, t_cap, t_recv, px, w, h)
                self.received += 1
                self._cond.notify_all()

    def latest(self, newer_than: int = -1, timeout: float = 1.0):
        """The newest frame, waiting (up to timeout) for one numbered above newer_than."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._latest and self._latest[0] > newer_than, timeout):
                return None
            n, t_cap, t_recv, px, w, h = self._latest
        rgb = np.frombuffer(px, np.uint8).reshape(h, w, 4)[:, :, 2::-1]   # BGRA -> RGB (a view; copy to keep)
        return n, t_cap, t_recv, rgb

    def close(self):
        self.f.close()                         # FlyEye's next write fails and it exits
        if self.app_pid:
            try:
                os.kill(self.app_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["info", "record"])
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--scale", type=float, default=0.5, help="pixels per screen point (0.5: half size)")
    ap.add_argument("--fps", type=float, default=60.0)
    args = ap.parse_args()

    with LiveWindow(args.scale, args.fps) as win:
        print(win.message)
        seconds = 5.0 if args.cmd == "info" else args.seconds
        frames, t_cap, ages, last = [], [], [], -1
        t0 = time.monotonic()
        while time.monotonic() - t0 < seconds:
            got = win.latest(last)
            if got is None:
                print("no new frame for 1 s (game paused, hidden or minimized?)")
                continue
            n, tc, tr, rgb = got
            ages.append((time.monotonic() - tc) * 1000)
            t_cap.append(tc)
            if args.cmd == "record":
                frames.append(rgb.copy())
            skipped = n - last - 1 if last >= 0 else 0
            last = n
        span = t_cap[-1] - t_cap[0]
        print(f"{len(t_cap)} frames in {span:.1f} s = {(len(t_cap) - 1) / span:.1f} per s "
              f"({win.received} delivered by the helper); frame age when we read it: "
              f"median {np.median(ages):.1f} ms, p95 {np.percentile(ages, 95):.1f} ms")
        if args.cmd == "record":
            OUT.mkdir(parents=True, exist_ok=True)
            out = OUT / f"rec_{time.strftime('%Y%m%d-%H%M%S')}.npz"
            np.savez_compressed(out, frames=np.stack(frames), t=np.array(t_cap) - t_cap[0], scale=args.scale)
            print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
