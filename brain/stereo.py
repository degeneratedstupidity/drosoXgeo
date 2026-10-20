"""Stereo Madness in Fly Dash: the real level's layout, from the start to the first ship portal (block 266).

The game ships its official levels as data: Resources/levels/1.txt is Stereo Madness (base64 + gzip, then
"key,value,..." per object; key 1 = object type, 2/3 = x/y in game units, 30 per block). We only read it; the game's
files are never changed, and the decoded level stays in results/ (gitignored; it's RobTop's content).

Object types used before the ship (checked against our recordings: the spike at block 17 every early death hit, the
double spikes + step + towers at 48-60):
  1-7    blocks (edges, corners, fill)       -> solid 1x1 cells, merged into boxes
  40     half-height slab (floating steps)   -> solid box, 0.5 high
  8      spike                               -> Peak (Fly Dash's spike hitbox)
  39     small spike                         -> Peak with a lower hitbox
  9      ground thorns (pit floors)          -> Thorns
  13     ship portal                         -> the end of the cube part: reaching it counts as a win here
  others decoration, colour triggers, coins  -> ignored

    python -m brain.stereo            # decode, build, check it's beatable in Fly Dash physics, draw it
"""

from __future__ import annotations

import base64
import sys
import zlib
from collections import defaultdict

import flydash
from brain.paths import RESULTS, ROOT

GAME_LEVEL = (ROOT / "geometrydash.app/Contents/SharedSupport/prefix/drive_c/Program Files/Geometry Dash"
              / "Resources/levels/1.txt")
DECODED = RESULTS / "stereo_madness_level.txt"
SOLID, SLAB, SPIKE, SMALL_SPIKE, THORNS, SHIP = {1, 2, 3, 4, 5, 6, 7}, 40, 8, 39, 9, 13
SMALL_SPIKE_H = 0.4
REAL_BUFFER = 0.08                # s: the real game still jumps on landing after a press this recent (flydash buffer)


def objects() -> list[dict]:
    """The level's objects as dicts (decoded once into results/, read-only from the game)."""
    if not DECODED.exists():
        raw = GAME_LEVEL.read_bytes().strip()
        data = base64.urlsafe_b64decode(raw + b"=" * (-len(raw) % 4))
        DECODED.parent.mkdir(parents=True, exist_ok=True)
        DECODED.write_text(zlib.decompress(data, 15 + 32).decode())
    out = []
    for part in DECODED.read_text().split(";")[1:]:
        if part:
            kv = part.split(",")
            out.append(dict(zip(kv[0::2], kv[1::2])))
    return out


def ship_x(objs=None) -> float:
    """Where the cube part ends: the first ship portal's x, in blocks."""
    return min(float(o["2"]) / 30 for o in (objs or objects()) if int(o["1"]) == SHIP)


def level(end: float | None = None):
    """(blocks, thorns, peaks) for Fly Dash, up to `end` blocks (default: the first ship portal), and that end."""
    objs = objects()
    end = ship_x(objs) if end is None else end
    cells = defaultdict(list)                        # column -> [(bottom, top)] of solid cells
    blocks, thorns, peaks = [], [], []
    for o in objs:
        kind, cx, cy = int(o["1"]), float(o["2"]) / 30, float(o["3"]) / 30
        if cx - 0.5 >= end:
            continue
        turned = ("6" in o and float(o["6"]) % 360) or o.get("4") == "1" or o.get("5") == "1"
        if turned and kind in (SPIKE, SMALL_SPIKE, SLAB, THORNS):    # a turned block is still the same square
            raise ValueError(f"rotated/flipped object before the ship, not modelled: {o}")
        if kind in SOLID:
            cells[round(cx - 0.5)].append((cy - 0.5, cy + 0.5))
        elif kind == SLAB:
            blocks.append(flydash.Block(cx - 0.5, cx + 0.5, cy + 0.25, cy - 0.25))
        elif kind == SPIKE:
            peaks.append(flydash.Peak(cx - 0.5, round(cy - 0.5, 3)))
        elif kind == SMALL_SPIKE:
            peaks.append(flydash.Peak(cx - 0.5, max(round(cy - 0.5, 1), 0.0), SMALL_SPIKE_H))
        elif kind == THORNS:
            thorns.append((cx - 0.5, cx + 0.5))
    for col, spans in cells.items():                 # stack a column's cells into boxes
        spans.sort()
        b0, t0 = spans[0]
        for b, t in spans[1:]:
            if b <= t0 + 1e-6:
                t0 = max(t0, t)
            else:
                blocks.append(flydash.Block(col, col + 1, round(t0, 3), round(b0, 3)))
                b0, t0 = b, t
        blocks.append(flydash.Block(col, col + 1, round(t0, 3), round(b0, 3)))
    blocks.sort(key=lambda b: (b.bottom, b.top, b.x0))   # merge side-by-side boxes of the same height
    merged = []
    for b in blocks:
        m = merged[-1] if merged else None
        if m and m.bottom == b.bottom and m.top == b.top and abs(m.x1 - b.x0) < 1e-6:
            m.x1 = b.x1
        else:
            merged.append(flydash.Block(b.x0, b.x1, b.top, b.bottom))
    thorns.sort()
    runs = []
    for a, b in thorns:
        if runs and a <= runs[-1][1] + 1e-6:
            runs[-1][1] = max(runs[-1][1], b)
        else:
            runs.append([a, b])
    merged.sort(key=lambda b: b.x0)
    peaks.sort(key=lambda p: p.x)
    return (merged, [flydash.Thorns(a, b) for a, b in runs], peaks), end


def make_env(buffer=REAL_BUFFER):
    lv, end = level()
    return flydash.FlyDashTerrain(level=lv, length=end, buffer=buffer)


def main() -> int:
    lv, end = level()
    blocks, thorns, peaks = lv
    print(f"Stereo Madness up to the ship portal at block {end:.1f}: {len(blocks)} solid boxes "
          f"({sum(b.bottom > 0 for b in blocks)} floating), {len(thorns)} thorn pits, {len(peaks)} spikes")
    jumps = flydash.plan(lv, end)
    print("beatable in Fly Dash physics:", "yes, " + f"{len(jumps)} jumps" if jumps else "NO")
    env = flydash.FlyDashTerrain(level=lv, length=end)
    for x in (10, 44, 120, 170, 245):
        env.px = x
        print(f"--- at block {x}\n" + env.render(48))
    return 0


if __name__ == "__main__":
    sys.exit(main())
