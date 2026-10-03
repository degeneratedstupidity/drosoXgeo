"""Phase 3, experiment 1: does driving the looming detectors make the Giant Fiber fire?

LPLC2 and LC4 are the visual "something is coming at me" neurons. They are the two
largest input types of DNp01, the Giant Fiber, which triggers the escape jump.
Here they get Poisson input at a fixed rate, with no game yet, and we record
DNp01's response:
  * how often it fires while the drive is on (Hz)
  * how long after the drive starts it fires the first time (latency, ms)

Each trial: 50 ms quiet, 300 ms drive, 50 ms quiet. Conditions:
  real          v783 wiring, drive LPLC2+LC4, LPLC2 only, or LC4 only
  shuffled 0-4  degree-preserving rewiring, LPLC2+LC4
  lesioned      real wiring, both DNp01 unable to spike (checks the readout control)

    python -m brain.exp_loom
"""

import argparse
import json
import sys
import time

import numpy as np

from lif import core

from brain import neurons
from brain.paths import FIGURES, PACK_V783, shuffled
from brain.stepper import Brain

PRE, DRIVE, POST = 500, 3000, 500            # ticks: 50 ms, 300 ms, 50 ms
RATES = [0, 10, 25, 50, 100, 150, 200]       # Hz on every driven neuron


def trial(pack, idx, groups, rate, seed, lesion=()):
    inputs = {g: idx[g] for g in ("LPLC2", "LC4")}
    brain = Brain(pack, inputs, monitor=idx["DNp01"], lesion=lesion, seed=seed)
    on = {g: (rate if g in groups else 0.0) for g in inputs}
    off = {g: 0.0 for g in inputs}
    spikes = np.concatenate([brain.step(PRE, off), brain.step(DRIVE, on), brain.step(POST, off)])
    gf = spikes.any(axis=1)                                  # either Giant Fiber
    during = spikes[PRE:PRE + DRIVE]
    first = np.flatnonzero(gf[PRE:PRE + DRIVE])
    return {
        "rate_hz": float(during.sum() / during.shape[1] / (DRIVE * 1e-4)),   # per DNp01 neuron
        "latency_ms": float(first[0] * 0.1) if first.size else None,
        "spontaneous": int(spikes[:PRE].sum()),
    }


def summarise(rows):
    rates = [r["rate_hz"] for r in rows]
    lats = [r["latency_ms"] for r in rows if r["latency_ms"] is not None]
    return {
        "rate_hz_mean": float(np.mean(rates)), "rate_hz_sd": float(np.std(rates)),
        "fired_trials": len(lats), "trials": len(rows),
        "latency_ms_median": float(np.median(lats)) if lats else None,
        "latency_ms_min": float(np.min(lats)) if lats else None,
        "latency_ms_max": float(np.max(lats)) if lats else None,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--shuffles", type=int, default=5)
    args = ap.parse_args()

    real = core.load_pack(PACK_V783)
    idx = neurons.indices(real)
    print(f"LPLC2 {idx['LPLC2'].size}, LC4 {idx['LC4'].size}, DNp01 {idx['DNp01'].size} neurons\n")

    conditions = [("real", "LPLC2+LC4", real, ("LPLC2", "LC4"), ()),
                  ("real", "LPLC2 only", real, ("LPLC2",), ()),
                  ("real", "LC4 only", real, ("LC4",), ()),
                  ("lesioned", "LPLC2+LC4", real, ("LPLC2", "LC4"), idx["DNp01"])]
    for s in range(args.shuffles):
        conditions.append((f"shuffled{s}", "LPLC2+LC4", core.load_pack(shuffled(PACK_V783, s)),
                           ("LPLC2", "LC4"), ()))

    results = []
    t0 = time.perf_counter()
    for wiring, drive, pack, groups, lesion in conditions:
        if wiring != "real":
            neurons.check_same_neurons(real, pack)
        for rate in RATES:
            rows = [trial(pack, idx, groups, rate, seed, lesion) for seed in range(args.trials)]
            s = summarise(rows)
            results.append({"wiring": wiring, "drive": drive, "input_hz": rate, **s})
            lat = f"{s['latency_ms_median']:5.1f} ms" if s["latency_ms_median"] is not None else "   never"
            print(f"{wiring:<10} {drive:<10} input {rate:>3} Hz -> DNp01 {s['rate_hz_mean']:6.1f} Hz "
                  f"(fired in {s['fired_trials']:>2}/{s['trials']} trials, first spike {lat})", flush=True)
    print(f"\n{time.perf_counter() - t0:.0f} s")

    FIGURES.mkdir(exist_ok=True)
    (FIGURES / "phase3_loom_dnp01.json").write_text(json.dumps({
        "protocol": {"quiet_ms": PRE / 10, "drive_ms": DRIVE / 10, "after_ms": POST / 10,
                     "trials": args.trials, "pack": real.manifest["dataset"]},
        "neurons": {k: int(v.size) for k, v in idx.items()},
        "results": results}, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
