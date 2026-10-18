"""The eye for the real Geometry Dash: from a picture of the game window to "how far is the next obstacle".

It produces the same numbers Fly Dash gives the brain (flydash.FlyDash.observe): distance, the gap from the cube's
front edge to the next obstacle in blocks (capped at 12), and theta_dot, how fast that obstacle grows in view.

How it reads a frame:
  1. cube    the player's icon is found by its lime-green colour (Rajan's icon: lime square, cyan centre). The
             camera follows the cube, so it sits at the same x on screen while playing.
  2. lane    the rows the cube occupies the last time it stood still on something (same height two frames in a row).
             Mid-jump we keep looking along that lane, as Fly Dash does: its spikes are always on the floor.
  3. ahead   in the lane, right of the cube, the first column with obstacle pixels: near-black (spikes, thorns) or
             bright white (the outlines of spikes and blocks). The background is saturated blue/purple, so neither
             shows up there.
  4. blocks  pixels -> blocks with the cube's size. The lime square is the icon without its black outline, so one
             block is 1.135 lime widths: that factor makes the measured approach speed 10.4 blocks/s, the game's
             normal speed (fitted on 14 approaches in the first recording).

Known gaps (Stereo Madness first; check before trusting it elsewhere): pits are invisible (the thorns sit below the
lane), and an obstacle the cube will land on looks the same as one it must jump over.

    python -m brain.realeye results/realgame/rec_YYYYMMDD-HHMMSS.npz     # trace + annotated contact sheet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import flydash
from brain.screen import write_png

BLOCK_PER_LIME = 1.135                  # one block in lime-square widths (see 4. above)
VIEW = flydash.VIEW                     # blocks; farther than this counts as "nothing ahead"
TITLE_BAR = 0.04                        # top fraction of the window to ignore (Wine's title bar)


def find_cube(rgb: np.ndarray):
    """(top, bottom, left, right) of the lime icon, or None (menus, the death explosion).

    A real cube's box is square-ish (up to ~1.4 wide when rotated 45 degrees) and about half lime (the rest is
    its purple centre); menus and the explosion's scattered pieces give big, mostly empty boxes (5-15% lime)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lime = (g > 200) & (r > 60) & (r < 200) & (b < 80)
    lime[: int(TITLE_BAR * len(rgb))] = False
    ys, xs = np.nonzero(lime)
    if len(ys) < 100:
        return None
    top, bottom, left, right = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    h, w = bottom - top + 1, right - left + 1
    if max(h, w) > 1.5 * min(h, w) or len(ys) < 0.3 * h * w:
        return None
    return top, bottom, left, right


class Eye:
    """Stateful (it remembers the lane). Call it once per frame, in order."""

    def __init__(self):
        self.lane = None                # (top, bottom, lime size in px)
        self.prev_bottom = None
        self.unit = None                # the upright cube's lime size: the smallest box it ever has (rotated = bigger)

    def __call__(self, rgb: np.ndarray) -> dict | None:
        cube = find_cube(rgb)
        if cube is None:
            self.prev_bottom = None
            return None
        top, bottom, left, right = cube
        size = max(bottom - top, right - left) + 1
        self.unit = size if self.unit is None else min(self.unit, size)
        if size > 1.5 * self.unit:                                     # not the cube we've been following
            self.prev_bottom = None
            return None
        upright = size <= 1.08 * self.unit                             # a rotated cube's box is bigger
        if upright and self.prev_bottom is not None and abs(bottom - self.prev_bottom) <= 1:
            self.lane = (top, bottom, size)
        self.prev_bottom = bottom
        if self.lane is None:
            return None
        lt, lb, lime = self.lane
        trim, gap = max(1, round(0.05 * lime)), max(2, round(0.13 * lime))   # skip the lane's edges, the cube's outline
        band = rgb[lt + trim: lb - trim + 1, right + 1 + gap:].astype(np.int16)
        hit = (band.max(-1) < 40) | (band.min(-1) > 235)
        cols = np.flatnonzero(hit.sum(0) >= 2)
        block_px = BLOCK_PER_LIME * lime
        d = (cols[0] + gap) / block_px if len(cols) else VIEW
        d = min(max(d, 0.05), VIEW)
        return {"distance": d, "theta_dot": flydash.SPEED / (d * d + 0.25), "cube": cube, "lane": (lt, lb),
                "block_px": block_px, "obstacle_x": None if not len(cols) else int(right + 1 + gap + cols[0])}


def _mark(img: np.ndarray, seen: dict | None) -> np.ndarray:
    """Draw the lane (yellow) and the detected obstacle edge (red) on a copy."""
    img = img.copy()
    if seen:
        lt, lb = seen["lane"]
        x0 = seen["cube"][3]
        img[lt: lt + 2, x0:], img[lb - 1: lb + 1, x0:] = (255, 230, 0), (255, 230, 0)
        if seen["obstacle_x"] is not None:
            img[max(0, lt - 15): lb + 15, seen["obstacle_x"]: seen["obstacle_x"] + 4] = (255, 0, 0)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", type=Path, help="an .npz from python -m brain.screen record")
    ap.add_argument("--sheet", type=int, default=12, help="frames on the annotated contact sheet")
    args = ap.parse_args()

    rec = np.load(args.recording)
    frames, t = rec["frames"], rec["t"]
    eye = Eye()
    seen = [eye(f) for f in frames]
    trace = [{"t": round(float(t[i]), 3), "distance": None if s is None else round(s["distance"], 3),
              "cube_top": None if s is None else s["cube"][0]} for i, s in enumerate(seen)]
    out = args.recording.with_suffix(".eye.json")
    out.write_text(json.dumps({"recording": args.recording.name, "trace": trace}))
    n = sum(s is not None for s in seen)
    print(f"{len(frames)} frames, cube + lane found in {n}; wrote {out}")

    # contact sheet: frames where something is close, to check the red line sits on the obstacle
    close = [i for i, s in enumerate(seen) if s and s["distance"] < 3]
    pick = [close[k] for k in np.linspace(0, len(close) - 1, min(args.sheet, len(close))).astype(int)]
    tiles = [_mark(frames[i], seen[i])[::2, ::2] for i in pick]
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    sheet = np.concatenate([np.concatenate(tiles[r: r + 3], axis=1) for r in range(0, len(tiles), 3)], axis=0)
    png = args.recording.with_suffix(".eye.png")
    write_png(png, sheet)
    print(f"wrote {png}  (frames {pick})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
