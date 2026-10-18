"""Vision input: each LPLC2/LC4 cell is driven by what looms in its own patch of the visual field (Stretch C).

Until now all 314 alarm cells got one rate, from the nearest obstacle (brain.player). Here every obstacle ahead
(flydash ... observe()["objects"]: distance, height relative to the eye) drives the cells whose receptive fields
(brain.receptive, from FlyWire's column map) point at it:

  rate of cell i = sum over objects o of  r_ref * (theta_dot_o / REF_LOOM) ** steep * exp(-delta_io^2 / (2 s_io^2))
                   capped at MAX_RATE
  theta_dot_o     how fast o grows in view (SPEED / (d^2 + 0.25)), as before
  delta_io        angle between cell i's centre and o's direction: objects are straight ahead (azimuth 0, the front
                  edge of both eyes) at elevation atan(height / distance)
  s_io            cell i's receptive-field radius, widened by half the object's angular size: a near object covers
                  more of the eye

So a tower above, a spike at eye level and a pit below drive different cells. r_ref and steep keep their meaning.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from brain.player import MAX_RATE, REF_LOOM
from brain.receptive import OUT as RF_CSV

import flydash


class Vision:
    """Per-cell rates for the LPLC2 and LC4 cells of a pack, in the pack's index order."""

    def __init__(self, pack, idx):
        rf = pd.read_csv(RF_CSV, dtype={"root_id": str}).set_index("root_id")
        self.groups = {}
        for t in ("LPLC2", "LC4"):
            ids = [str(int(r)) for r in np.asarray(pack.neuron_ids)[idx[t]]]
            g = rf.loc[ids]
            self.groups[t] = (g.azimuth.to_numpy(), g.elevation.to_numpy(), g.radius.to_numpy())

    def rates(self, objects, r_ref: float, steep: float) -> dict[str, np.ndarray]:
        out = {t: np.zeros(len(az)) for t, (az, _, _) in self.groups.items()}
        for d, height in objects:
            loom = r_ref * ((flydash.SPEED / (d * d + 0.25)) / REF_LOOM) ** steep
            if loom < 1e-3:
                continue
            el = math.degrees(math.atan2(height, d))
            half = math.degrees(math.atan(0.5 / d))              # half the object's angular size
            for t, (az, cel, rad) in self.groups.items():
                s2 = rad ** 2 + half ** 2
                out[t] += loom * np.exp(-(az ** 2 + (cel - el) ** 2) / (2 * s2))
        return {t: np.minimum(r, MAX_RATE) for t, r in out.items()}
