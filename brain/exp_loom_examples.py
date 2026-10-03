"""Record one example trial per test of exp_loom, spike by spike, for the visual test bench.

Same protocol as brain.exp_loom (50 ms quiet, 300 ms drive, 50 ms quiet), seed 0.
Monitors 16 LPLC2 and 16 LC4 neurons (evenly spaced through each list) and both
DNp01. Also counts how many neurons anywhere in the brain fired during the trial.

    python -m brain.exp_loom_examples      -> figures/phase3_loom_examples.json
"""

import json
import sys

import numpy as np

from lif import core

from brain import neurons
from brain.exp_loom import DRIVE, POST, PRE, RATES
from brain.paths import FIGURES, PACK_V783, shuffled
from brain.stepper import Brain

SAMPLE = 16


def main() -> int:
    real = core.load_pack(PACK_V783)
    idx = neurons.indices(real)
    pick = lambda a: a[np.linspace(0, a.size - 1, SAMPLE).round().astype(int)]
    rows = {"LPLC2": pick(idx["LPLC2"]), "LC4": pick(idx["LC4"]), "DNp01": idx["DNp01"]}
    monitor = np.concatenate(list(rows.values()))

    tests = [("real_both", real, ("LPLC2", "LC4"), ()),
             ("real_lplc2", real, ("LPLC2",), ()),
             ("real_lc4", real, ("LC4",), ()),
             ("lesioned", real, ("LPLC2", "LC4"), idx["DNp01"])]
    tests += [(f"shuffled{s}", core.load_pack(shuffled(PACK_V783, s)), ("LPLC2", "LC4"), ())
              for s in range(5)]

    out = {"ticks_per_ms": 10, "quiet_ms": PRE / 10, "drive_ms": DRIVE / 10, "after_ms": POST / 10,
           "rows": {k: int(v.size) for k, v in rows.items()}, "trials": {}}
    for name, pack, groups, lesion in tests:
        neurons.check_same_neurons(real, pack)
        for rate in RATES:
            inputs = {g: idx[g] for g in ("LPLC2", "LC4")}
            brain = Brain(pack, inputs, monitor=monitor, lesion=lesion, seed=0)
            on = {g: (rate if g in groups else 0.0) for g in inputs}
            off = {g: 0.0 for g in inputs}
            sp = np.concatenate([brain.step(PRE, off), brain.step(DRIVE, on), brain.step(POST, off)])
            # spike times per monitored neuron, in tenths of a millisecond
            trains = [np.flatnonzero(sp[:, j]).tolist() for j in range(sp.shape[1])]
            active = int((brain.spike_counts() > 0).sum())
            out["trials"][f"{name}@{rate}"] = {"trains": trains, "active": active}
            gf = sp[:, -2:].any(axis=1)
            first = np.flatnonzero(gf[PRE:])
            print(f"{name:<11} {rate:>3} Hz  DNp01 spikes {int(sp[:, -2:].sum()):>3}  "
                  f"first {first[0] / 10 if first.size else '-':>5} ms  brain active {active:>6}", flush=True)
    path = FIGURES / "phase3_loom_examples.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
