"""The eye for the real Geometry Dash: from a picture of the game window to "how far is the next obstacle".

It produces the same numbers Fly Dash gives the brain (flydash.FlyDash.observe): distance, the gap from the cube's
front edge to the next obstacle in blocks (capped at 12), and theta_dot, how fast that obstacle grows in view.

How it reads a frame:
  1. cube    the player's icon is found by its lime-green colour (Rajan's icon: lime square, cyan centre). The
             camera follows the cube, so it sits at the same x on screen while playing.
  2. band    the rows to look along. Standing (same height two frames in a row): the cube's rows, its "lane". In the
             air: the rows the cube is in now, so whatever it is flying over stops counting (looking along the old
             lane kept a block it was passing over at distance 0, the alarm stayed on, and the held key made the cube
             jump again the moment it landed).
  3. ahead   in the band, from the cube's front (its centre + half a block, the same whether or not it is rotating),
             the first column with obstacle pixels: near-black (spikes, thorns) or bright white (the outlines of
             spikes and blocks). The background is saturated blue/purple, so neither shows up there.
  4. pits    standing on something raised (above the usual floor), the strip just under the lane is its top. Where
             that support ends and black thorns lie below, the edge is an obstacle too: the jump has to happen there,
             not at the next tower's wall.
  5. blocks  pixels -> blocks with the cube's size. The lime square is the icon without its black outline, so one
             block is 1.135 lime widths: that factor makes the measured approach speed 10.4 blocks/s, the game's
             normal speed (fitted on 14 approaches in the first recording).
  The distance is the nearer of 3 and 4, capped at 12 blocks.

Known gaps (Stereo Madness first; check before trusting it elsewhere): a block to land on looks the same as one to
jump over (both mean "jump", which is right for the cube); drops without thorns are ignored; no ship/ball modes.

    python -m brain.realeye results/realgame/rec_YYYYMMDD-HHMMSS.npz     # trace + annotated contact sheet
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
import sys
from pathlib import Path

import numpy as np

import flydash
from brain.png import write_png

BLOCK_PER_LIME = 1.135                  # one block in lime-square widths (see 4. above)
VIEW = flydash.VIEW                     # blocks; farther than this counts as "nothing ahead"
TITLE_BAR = 0.04                        # top fraction of the window to ignore (Wine's title bar)


def _patches(mask: np.ndarray, cell: int = 8) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group the mask's pixels into separate patches (pixels of the same patch lie within ~cell px of each other).
    A flood fill over cell x cell squares, so it stays fast without scipy. Returns (ys, xs) per patch."""
    ys, xs = np.nonzero(mask)
    if not len(ys):
        return []
    cy, cx = ys // cell, xs // cell
    occupied = set(zip(cy.tolist(), cx.tolist()))
    label, n = {}, 0
    for start in occupied:
        if start in label:
            continue
        stack, label[start] = [start], n
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    nb = (y + dy, x + dx)
                    if nb in occupied and nb not in label:
                        label[nb] = n
                        stack.append(nb)
        n += 1
    lab = np.array([label[c] for c in zip(cy.tolist(), cx.tolist())])
    return [(ys[lab == k], xs[lab == k]) for k in range(n)]


def find_cube(rgb: np.ndarray):
    """(top, bottom, left, right) of the lime icon, or None (menus, the death explosion)."""
    found = _find_cube(rgb)
    return None if found is None else found[0]


def _find_cube(rgb: np.ndarray):
    """((top, bottom, left, right), lime pixel count) of the cube, or None.

    Lime pixels are grouped into patches, and the biggest patch that looks like a cube wins: its box is square-ish
    (up to ~1.4 wide when rotated 45 degrees) and about half lime (the rest is its purple centre). That skips the
    practice-mode checkpoint buttons (small green diamonds), menus and the explosion's scattered pieces."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lime = (g > 200) & (r > 60) & (r < 200) & (b < 80)
    lime[: int(TITLE_BAR * len(rgb))] = False
    best = None
    for ys, xs in _patches(lime):
        if len(ys) < 100:
            continue
        top, bottom, left, right = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
        h, w = bottom - top + 1, right - left + 1
        if max(h, w) > 1.5 * min(h, w) or len(ys) < 0.3 * h * w:
            continue
        if best is None or len(ys) > best[1]:
            best = ((top, bottom, left, right), len(ys))
    return best


class Eye:
    """Stateful (it remembers the lane). Call it once per frame, in order."""

    def __init__(self):
        self.lane = None                # (top, bottom, lime size in px)
        self.prev_bottom = None
        self.sizes = Counter()          # box sizes of dense (upright-looking) detections
        self.unit = None                # the upright cube's lime size: the most common of those (rotated = bigger)
        self.floors = Counter()         # lane bottoms seen while standing; the most common is the ground

    def __call__(self, rgb: np.ndarray) -> dict | None:
        found = _find_cube(rgb)
        cube = None if found is None else found[0]
        if cube is None:
            self.prev_bottom = None
            return None
        top, bottom, left, right = cube
        size = max(bottom - top, right - left) + 1
        if found[1] >= 0.5 * size * size:                              # upright: ~54-64% lime (less when small and
                                                                       # blurred), rotated ~35%
            self.sizes[size] += 1
            self.unit = self.sizes.most_common(1)[0][0]
        if self.unit is None or not 0.85 * self.unit <= size <= 1.5 * self.unit:   # not the cube we've been following
            self.prev_bottom = None
            return None
        upright = size <= 1.08 * self.unit                             # a rotated cube's box is bigger
        standing = upright and self.prev_bottom is not None and abs(bottom - self.prev_bottom) <= 1
        if standing:
            self.lane = (top, bottom, size)
            self.floors[bottom] += 1
        self.prev_bottom = bottom
        if self.lane is None:
            return None
        lime = self.unit
        block_px = BLOCK_PER_LIME * lime
        trim = max(1, round(0.05 * lime))                              # skip the band's edges
        if standing:
            bt, bb = top, bottom
        else:                                                          # in the air: where the cube is now
            mid = (top + bottom) // 2
            bt, bb = mid - lime // 2, mid + lime // 2
        front = (left + right) // 2 + round(block_px / 2)              # the cube's front edge, rotating or not
        clear = max(front, right + max(2, round(0.13 * lime))) + 1     # look past the cube's own outline/corners
        band = rgb[max(0, bt + trim): bb - trim + 1, clear:].astype(np.int16)
        hit = (band.max(-1) < 40) | (band.min(-1) > 235)
        cols = np.flatnonzero(hit.sum(0) >= 2)
        wall = clear + int(cols[0]) if len(cols) else None
        edge = self._pit_edge(rgb, bb, clear, block_px) if standing else None
        ahead = [x for x in (wall, edge) if x is not None]
        d = (min(ahead) - front) / block_px if ahead else VIEW
        d = min(max(d, 0.05), VIEW)
        nearest = None if not ahead else min(ahead)
        return {"distance": d, "theta_dot": flydash.SPEED / (d * d + 0.25), "cube": cube, "lane": (bt, bb),
                "block_px": block_px, "obstacle_x": nearest, "pit": edge is not None and edge == min(ahead)}

    def _pit_edge(self, rgb: np.ndarray, lane_bottom: int, front: int, block_px: float) -> int | None:
        """Standing on something raised: the x where its top ends with black thorns below, or None."""
        ground = self.floors.most_common(1)[0][0]
        if lane_bottom > ground - 0.5 * block_px:                      # on the usual floor: nothing to fall off
            return None
        depth = max(2, round(0.2 * block_px))
        strip = rgb[lane_bottom + 1: lane_bottom + 1 + depth, front + 1:].astype(np.int16)
        support = ((strip.max(-1) < 110) | (strip.min(-1) > 200)).any(0)   # block top: white outline or dark fill
        run = max(2, round(0.25 * block_px))                           # a gap at least a quarter block wide
        missing = np.convolve(~support, np.ones(run, dtype=int), "valid") == run
        starts = np.flatnonzero(missing)
        if not len(starts):
            return None
        x = front + 1 + int(starts[0])
        below = rgb[lane_bottom + 1:, x: x + round(1.5 * block_px)].astype(np.int16)
        thorns = (below.max(-1) < 40).sum()
        return x if thorns >= 3 else None


def _mark(img: np.ndarray, seen: dict | None) -> np.ndarray:
    """Draw the band looked along (yellow) and the nearest obstacle (red: a wall or spike, cyan: a pit edge)."""
    img = img.copy()
    if seen:
        lt, lb = seen["lane"]
        x0 = seen["cube"][3]
        img[max(0, lt): lt + 2, x0:], img[lb - 1: lb + 1, x0:] = (255, 230, 0), (255, 230, 0)
        if seen["obstacle_x"] is not None:
            colour = (0, 255, 255) if seen.get("pit") else (255, 0, 0)
            img[max(0, lt - 15): lb + 15, seen["obstacle_x"]: seen["obstacle_x"] + 4] = colour
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", type=Path, help="an .npz from python -m brain.stream record")
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
