"""Eyes on the real Geometry Dash: grab the game's window, and nothing else, as a numpy image.

Stretch B, step 1 (watch-only). Uses macOS's own CoreGraphics through ctypes, so nothing to install. The game runs
in Wine (Sikarugir), so macOS sees a window owned by "wine" titled "Geometry Dash". Every grab asks for that one
window by its ID: other windows are never captured, even where they overlap it, and if the game's window can't be
found we stop rather than fall back to the screen.

Needs the Screen Recording permission for the app that runs this (the terminal/IDE).
A grab costs ~22 ms whatever its size (macOS composites the whole window each time), so ~45 grabs/s at most:
fine to watch, too slow for live play next to the brain's ~11 ms per frame.

    python -m brain.screen info                     # find the window, time a few grabs
    python -m brain.screen snapshot                 # one PNG of the game window -> results/realgame/
    python -m brain.screen record --seconds 20      # frames + timestamps -> results/realgame/rec_*.npz
"""

from __future__ import annotations

import argparse
import ctypes
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np

from brain.paths import RESULTS

OUT = RESULTS / "realgame"

_cg = ctypes.CDLL("/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics")
_cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
_V = ctypes.c_void_p
_UTF8 = 0x08000100
_ALL, _INCLUDING_WINDOW = 0, 8                         # kCGWindowListOption...
_IGNORE_FRAMING, _NOMINAL_RES = 1, 16                  # kCGWindowImage...: no title bar shadow, 1 pixel per point


class _Rect(ctypes.Structure):
    _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double), ("w", ctypes.c_double), ("h", ctypes.c_double)]


def _sig(lib, name, restype, *argtypes):
    f = getattr(lib, name)
    f.restype, f.argtypes = restype, list(argtypes)


_sig(_cg, "CGPreflightScreenCaptureAccess", ctypes.c_bool)
_sig(_cg, "CGWindowListCopyWindowInfo", _V, ctypes.c_uint32, ctypes.c_uint32)
_sig(_cg, "CGRectMakeWithDictionaryRepresentation", ctypes.c_bool, _V, ctypes.POINTER(_Rect))
_sig(_cg, "CGWindowListCreateImage", _V, _Rect, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32)
for _f in ("CGImageGetWidth", "CGImageGetHeight", "CGImageGetBytesPerRow"):
    _sig(_cg, _f, ctypes.c_size_t, _V)
_sig(_cg, "CGImageGetDataProvider", _V, _V)
_sig(_cg, "CGDataProviderCopyData", _V, _V)
_sig(_cf, "CFArrayGetCount", ctypes.c_long, _V)
_sig(_cf, "CFArrayGetValueAtIndex", _V, _V, ctypes.c_long)
_sig(_cf, "CFDictionaryGetValue", _V, _V, _V)
_sig(_cf, "CFStringCreateWithCString", _V, _V, ctypes.c_char_p, ctypes.c_uint32)
_sig(_cf, "CFStringGetCString", ctypes.c_bool, _V, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32)
_sig(_cf, "CFNumberGetValue", ctypes.c_bool, _V, ctypes.c_int, _V)
_sig(_cf, "CFDataGetBytePtr", _V, _V)
_sig(_cf, "CFDataGetLength", ctypes.c_long, _V)
_sig(_cf, "CFRelease", None, _V)

_KEYS = {k: _cf.CFStringCreateWithCString(None, k.encode(), _UTF8)
         for k in ("kCGWindowNumber", "kCGWindowOwnerName", "kCGWindowName", "kCGWindowBounds")}


def _str(ref) -> str:
    if not ref:
        return ""
    buf = ctypes.create_string_buffer(1024)
    return buf.value.decode() if _cf.CFStringGetCString(ref, buf, 1024, _UTF8) else ""


def _int(ref) -> int:
    x = ctypes.c_int64()
    _cf.CFNumberGetValue(ref, 4, ctypes.byref(x))      # kCFNumberSInt64Type
    return x.value


def find_game_window(owner: str = "wine", title: str = "Geometry Dash") -> tuple[int, _Rect]:
    """(window ID, bounds in screen points) of the game's window, on any desktop. Raises if it isn't there."""
    if not _cg.CGPreflightScreenCaptureAccess():
        raise SystemExit("no Screen Recording permission for this app (System Settings > Privacy & Security)")
    arr = _cg.CGWindowListCopyWindowInfo(_ALL, 0)  # window metadata only, every desktop (Space): no pixels
    try:
        for i in range(_cf.CFArrayGetCount(arr)):
            d = _cf.CFArrayGetValueAtIndex(arr, i)
            if (_str(_cf.CFDictionaryGetValue(d, _KEYS["kCGWindowOwnerName"])) == owner
                    and _str(_cf.CFDictionaryGetValue(d, _KEYS["kCGWindowName"])) == title):
                r = _Rect()
                _cg.CGRectMakeWithDictionaryRepresentation(_cf.CFDictionaryGetValue(d, _KEYS["kCGWindowBounds"]),
                                                           ctypes.byref(r))
                return _int(_cf.CFDictionaryGetValue(d, _KEYS["kCGWindowNumber"])), r
    finally:
        _cf.CFRelease(arr)
    raise SystemExit(f"no window {title!r} owned by {owner!r}: is the game running? (open geometrydash.app)")


class GameWindow:
    """The game's window. grab() returns its pixels as an (H, W, 3) uint8 RGB array, 1 pixel per screen point."""

    def __init__(self):
        self.wid, self.bounds = find_game_window()

    def grab(self) -> np.ndarray:
        everything = _Rect(float("inf"), float("inf"), 0.0, 0.0)  # CGRectNull: the window's own bounds
        img = _cg.CGWindowListCreateImage(everything, _INCLUDING_WINDOW, self.wid, _IGNORE_FRAMING | _NOMINAL_RES)
        if not img:   # the window closed, or Wine replaced it (e.g. after a video-mode change)
            raise RuntimeError(f"window {self.wid} can't be grabbed any more; restart to find the game again")
        try:
            w, h, bpr = _cg.CGImageGetWidth(img), _cg.CGImageGetHeight(img), _cg.CGImageGetBytesPerRow(img)
            data = _cg.CGDataProviderCopyData(_cg.CGImageGetDataProvider(img))
            try:
                buf = (ctypes.c_uint8 * _cf.CFDataGetLength(data)).from_address(_cf.CFDataGetBytePtr(data))
                bgra = np.ctypeslib.as_array(buf).reshape(h, bpr // 4, 4)
                return bgra[:, :w, 2::-1].copy()        # BGRA -> RGB, drop row padding
            finally:
                _cf.CFRelease(data)
        finally:
            _cf.CFRelease(img)


def write_png(path: Path, rgb: np.ndarray) -> None:
    """Minimal PNG writer, so we don't need Pillow."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["info", "snapshot", "record"])
    ap.add_argument("--seconds", type=float, default=20.0, help="record: how long")
    ap.add_argument("--fps", type=float, default=30.0, help="record: target grabs per second (max ~45)")
    ap.add_argument("--shrink", type=int, default=2, help="record: keep every Nth pixel to save space")
    args = ap.parse_args()

    win = GameWindow()
    b = win.bounds
    print(f"game window {win.wid}: {b.w:.0f}x{b.h:.0f} points at ({b.x:.0f}, {b.y:.0f})")
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    if args.cmd == "info":
        ts = []
        for _ in range(20):
            t = time.perf_counter()
            img = win.grab()
            ts.append((time.perf_counter() - t) * 1000)
        print(f"grab {img.shape[1]}x{img.shape[0]} px: median {np.median(ts):.1f} ms, max {max(ts):.1f} ms")
    elif args.cmd == "snapshot":
        out = OUT / f"snap_{stamp}.png"
        write_png(out, win.grab())
        print(f"wrote {out}")
    else:
        frames, times = [], []
        t0 = time.perf_counter()
        while (now := time.perf_counter() - t0) < args.seconds:
            frames.append(win.grab()[::args.shrink, ::args.shrink])
            times.append(now)
            wait = len(frames) / args.fps - (time.perf_counter() - t0)
            if wait > 0:
                time.sleep(wait)
        out = OUT / f"rec_{stamp}.npz"
        np.savez_compressed(out, frames=np.stack(frames), t=np.array(times), shrink=args.shrink,
                            window=np.array([b.x, b.y, b.w, b.h]))
        print(f"wrote {out}: {len(frames)} frames in {times[-1]:.1f} s ({len(frames) / times[-1]:.1f} per s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
