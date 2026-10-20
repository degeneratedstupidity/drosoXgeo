"""A shadow of the real game: Fly Dash's copy of Stereo Madness, kept in step with the cube on screen.

The learned decision layer (brain.learn) was trained on Fly Dash's Stereo Madness. To give it the same input in the
real game, a copy of that level runs alongside it:
  start  when the game respawns the cube (it blinks on and off at the start; brain.live.attempts uses the same sign)
  move   at the game's speed, by the real elapsed time; while the first spike approaches, the eye's measured
         distance to it (brain.realeye, in the cube's band) pins down where the cube is
  jump   when the screen shows the real cube leaving the ground, the shadow rewinds to that moment and jumps too
  land   by Fly Dash physics on the level's real geometry
The obstacles the fly's looming cells see then come from the shadow (Fly Dash's objects(): distance and height of
every obstacle ahead), i.e. from the level file, placed where the real cube is. Said plainly for the post: the
cube's position and jumps come from the screen; the obstacle layout comes from the game's level file.

    python -m brain.shadow results/realgame/live_*_press.npz     # check it against a recorded real run
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

import flydash
from brain.stereo import level

GD_SPEED = 10.41                  # blocks/s, fitted on two recorded real runs (least drift vs the eye; 10.386 drifted 0.4 blocks by block 145)
FIRST_SPIKE = 17.0                # x of the first spike's left edge (block units)
RISE_PX = 2                       # the cube's top moving up this many pixels = it left the ground
SAFE_START = 12.0                 # blocks: nothing can kill the cube before the first spike (17)
SYNC_FRAMES = 3                   # frames of the eye's first-spike distance that agree before the shadow moves


class Shadow:
    def __init__(self):
        self.lv, self.end = level()
        self.env = None
        self.t0 = None                # capture time of the attempt's start
        self.history = []             # (game time, env copy) per frame, for rewinding to a takeoff
        self.ground_top = None        # the cube's top in pixels when standing on the floor
        self.stand_top = None         # the cube's top in pixels the last time the shadow stood on something
        self.missing_since = None
        self.sync = []                # first-spike position estimates
        self.attempts = 0             # how many attempts have started (each respawn)

    def _game_time(self, t):
        return (t - self.t0) * GD_SPEED / flydash.SPEED          # Fly Dash time that covers the same distance

    def start(self, t):
        self.env = flydash.FlyDashTerrain(level=self.lv, length=self.end)
        self.env.reset()
        self.t0, self.history, self.sync, self.stand_top = t, [], [], None
        self.attempts += 1

    def _advance(self, t_game, jump_now=False):
        """Step the shadow to game time t_game; with jump_now, the first step is a jump (at least one step).
        The shadow can't die: it only places the obstacles for the fly's eye, so a near miss in the shadow must not
        freeze it while the real cube carries on."""
        e = self.env
        while (jump_now or e.t + flydash.DT <= t_game + 1e-9) and not e.won:
            self.history.append((e.t, copy.copy(e)))
            e.step(jump_now)
            e.dead = False
            jump_now = False

    def frame(self, t, cube, eye_distance=None, unit=None):
        """One captured frame: t = capture time (s), cube = (top, bottom, left, right) px or None, eye_distance =
        brain.realeye's distance (blocks) to the nearest obstacle in the cube's band, if any; unit = one block in
        pixels (realeye's block_px). Returns the shadow env (None between attempts)."""
        if unit is None:
            unit = 1.135 * (cube[1] - cube[0] + 1) if cube is not None else 1.0
        if cube is None:
            if self.missing_since is None:
                self.missing_since = t
            return self.env
        top = cube[0]
        if (self.missing_since is not None and t - self.missing_since > 0.25
                and (self.env is None or self.env.px > SAFE_START)):
            self.start(t)                                           # back after a death: a new attempt
            # (a gap before SAFE_START is the respawned cube fading in, not a death: nothing can kill it there)
        elif self.env is None:
            self.start(t)
        self.missing_since = None
        if self.ground_top is None or (self.env.t < 1.0 and top > self.ground_top):
            self.ground_top = top                                   # standing on the floor at the start
        tg = self._game_time(t)
        if self.env.grounded and self.stand_top is not None and top <= self.stand_top - RISE_PX:
            # the real cube has left the ground (since the last frame, or during frames the eye lost it). How long
            # ago follows from how high it is now, on the jump's arc: rewind to that moment and jump there
            rise = (self.stand_top - top) / unit
            v, g = flydash.JUMP_V, flydash.GRAVITY
            tau = (v - np.sqrt(max(v * v - 2 * g * min(rise, flydash.APEX), 0.0))) / g
            takeoff = tg - tau
            while self.history and self.history[-1][0] > takeoff:
                self.history.pop()
            if self.history:
                self.env = copy.copy(self.history[-1][1])
            self._advance(tg, jump_now=True)
        else:
            self._advance(tg)
        if self.env.grounded:
            self.stand_top = top
        # first spike in view, cube on the floor: the eye's distance says where the cube is
        if eye_distance is not None and self.env.px < FIRST_SPIKE - 2 and self.env.grounded and 1.0 < eye_distance < 8:
            measured = FIRST_SPIKE - 1 - eye_distance
            self.sync.append(measured - self.env.px)
            if len(self.sync) >= SYNC_FRAMES:
                # the attempt started earlier/later than we thought: move the cube AND the clock together (moving
                # only the start time would make the shadow step to catch up on top of the move: twice the shift)
                shift = float(np.median(self.sync))
                self.t0 -= shift / GD_SPEED
                self.env.px += shift
                self.env.t += shift / flydash.SPEED
                self.history, self.sync = [], []
        return self.env


def check(npz: Path):
    """Run the shadow on a recorded real run (brain.live --record) and compare its cube height with the screen."""
    from brain.realeye import Eye
    log = json.loads(npz.with_suffix(".json").read_text())["frames"]
    frames = np.load(npz)["frames"]
    eye, sh = Eye(), Shadow()
    rows = []
    for f, rec in zip(frames, log):
        seen = eye(f)
        cube = None if seen is None else seen["cube"]
        env = sh.frame(rec["t"], cube, None if seen is None else seen["distance"],
                       None if seen is None else seen["block_px"])
        if env is not None and seen is not None and sh.ground_top is not None:
            unit = seen["block_px"]
            real_y = (sh.ground_top - cube[0]) / unit
            rows.append((rec["t"], env.px, env.py, real_y, env.dead))
    r = np.array(rows, dtype=float)
    alive = np.ones(len(r), dtype=bool)
    err = np.abs(r[alive, 2] - r[alive, 3])
    print(f"{len(r)} frames with the cube; shadow vs screen cube height: median error {np.median(err):.2f} blocks, "
          f"90% {np.percentile(err, 90):.2f}, 99% {np.percentile(err, 99):.2f}")
    for k in range(0, len(r), max(1, len(r) // 25)):
        t, px, py, ry, dead = r[k]
        print(f"  t {t:6.2f}  shadow x {px:6.1f} height {py:5.2f}   screen height {ry:5.2f}  {'(shadow dead)' if dead else ''}")
    return r


def main() -> int:
    check(Path(sys.argv[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
