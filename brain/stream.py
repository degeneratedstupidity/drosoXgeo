"""A live feed of the Geometry Dash window, through ScreenCaptureKit (brain/gdcapture.swift).

brain.screen asks macOS for a fresh snapshot each time (~25 ms). Here a small Swift helper subscribes to the game
window once and macOS hands it every new frame as the game draws it; a background thread here keeps the newest one.
Same rule as brain.screen: only the game window is captured (the helper's filter is that one window).

The helper is compiled on first use with the Swift compiler that comes with Xcode's command line tools
(swiftc, nothing downloaded) into results/bin/gdcapture.

    python -m brain.stream info                      # frames per second and how old a frame is when we get it
    python -m brain.stream record --seconds 20       # 60 fps recording -> results/realgame/rec_*.npz
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from brain.paths import RESULTS

SOURCE = Path(__file__).with_name("gdcapture.swift")
BINARY = RESULTS / "bin" / "gdcapture"
OUT = RESULTS / "realgame"
HEADER = struct.Struct("<4sIddII")                  # magic, frame number, capture time, send time, width, height


def build() -> Path:
    """Compile the helper if it's missing or older than its source."""
    if not BINARY.exists() or BINARY.stat().st_mtime < SOURCE.stat().st_mtime:
        BINARY.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["swiftc", "-O", "-swift-version", "5", str(SOURCE), "-o", str(BINARY)], check=True)
    return BINARY


def _read_exact(f, n: int) -> bytes | None:
    buf = bytearray()
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return bytes(buf)


class LiveWindow:
    """Starts the helper; .latest() returns (frame number, capture time, received time, RGB array) or None."""

    def __init__(self, scale: float = 0.5, fps: float = 60.0, title: str = "Geometry Dash"):
        self.proc = subprocess.Popen([str(build()), "--title", title, "--scale", str(scale), "--fps", str(fps)],
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self.message = self.proc.stderr.readline().decode().strip()   # "streaming window ..." or the error
        if self.proc.poll() is not None or not self.message.startswith("streaming"):
            self.proc.kill()
            raise SystemExit(f"gdcapture: {self.message or 'exited'}")
        self._latest = None
        self._cond = threading.Condition()
        self.received = 0
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        f = self.proc.stdout
        while (head := _read_exact(f, HEADER.size)) is not None:
            magic, n, t_cap, _t_send, w, h = HEADER.unpack(head)
            if magic != b"GDFR":
                raise RuntimeError("gdcapture stream out of sync")
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
        self.proc.terminate()

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
