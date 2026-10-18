"""Review #2: are the scrambled brains "differently wired" or just "dead"?

Phase 2 found the shuffled brains much quieter than the real one (~2,330 vs 9,091
spikes on the sugar drive). If scrambling makes any input die out, the shuffle
control only shows "a broken brain can't play". So here the drive is nothing the
fly evolved for: 300 random ordinary neurons (not hubs, not the escape circuit),
all at the same rate, with the same input spikes in every brain. We count what
the rest of the brain does: spikes and neurons active, the driven 300 excluded.

Each trial: 1 s of drive. 5 random draws of the 300 neurons, real + 5 shuffles.

If the shuffles are much quieter, --match finds the gain-matched shuffle: the one
number that multiplies every synapse's weight so that a shuffled brain is as
active as the real one on this drive (halving the search interval each step).

    python -m brain.activity                     # real vs shuffled, gain 1
    python -m brain.activity --match shuffled0   # the gain that matches the real brain
"""

import argparse
import json
import time

import numpy as np

from lif import core

from brain import neurons
from brain.paths import FIGURES, PACK_V783, shuffled
from brain.stepper import Brain

N_DRIVEN = 300
RATE_HZ = 100.0
TICKS = 10_000                    # 1 s
HUB_PCT = 90                      # drive only neurons below this percentile of in- and out-degree


def ordinary_neurons(pack: core.Pack, exclude: np.ndarray) -> np.ndarray:
    """Neurons with some output whose in- and out-degree are both below HUB_PCT.
    Degrees are the same in every shuffle, so this list is too."""
    out_deg = pack.out_degree
    in_deg = np.bincount(np.asarray(pack.destinations), minlength=pack.n_neurons)
    ok = ((out_deg > 0) & (in_deg > 0)
          & (out_deg < np.percentile(out_deg, HUB_PCT)) & (in_deg < np.percentile(in_deg, HUB_PCT)))
    ok[exclude] = False
    return np.flatnonzero(ok)


def trial(pack: core.Pack, driven: np.ndarray, seed: int, gain: float = 1.0) -> dict:
    brain = Brain(pack, {"random": driven}, monitor=driven[:1], seed=seed, gain=gain)
    t0 = time.perf_counter()
    brain.step(TICKS, {"random": RATE_HZ})
    counts = brain.spike_counts()
    rest = np.ones(pack.n_neurons, dtype=bool)
    rest[driven] = False
    return {
        "driven_hz": float(counts[driven].mean() / (TICKS * 1e-4)),
        "spikes_rest": int(counts[rest].sum()),
        "active_rest": int((counts[rest] > 0).sum()),
        "wall_s": round(time.perf_counter() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--out", default=None, help="default: figures/activity_check.json, or activity_match_<brain>.json")
    ap.add_argument("--match", metavar="SHUFFLE", help="find the gain that matches this shuffle (e.g. shuffled0) to the real brain")
    ap.add_argument("--steps", type=int, default=12, help="halvings of the gain interval in --match")
    args = ap.parse_args()

    real = core.load_pack(PACK_V783)
    idx = neurons.indices(real)
    pool = ordinary_neurons(real, np.concatenate(list(idx.values())))
    sets = [np.sort(np.random.default_rng(1000 + d).choice(pool, N_DRIVEN, replace=False)).astype(np.int32)
            for d in range(args.draws)]
    print(f"{pool.size} ordinary neurons to draw from; {args.draws} draws of {N_DRIVEN} at {RATE_HZ:.0f} Hz, 1 s",
          flush=True)

    if args.match:
        return match(real, sets, args)

    brains = [("real", real)]
    for s in range(5):
        p = core.load_pack(shuffled(PACK_V783, s))
        neurons.check_same_neurons(real, p)
        brains.append((f"shuffled{s}", p))

    rows = []
    for d, driven in enumerate(sets):
        for name, pack in brains:
            r = {"brain": name, "draw": d, **trial(pack, driven, seed=d)}
            rows.append(r)
            print(f"draw {d} {name:<10} driven {r['driven_hz']:5.1f} Hz  rest: {r['spikes_rest']:>8,} spikes, "
                  f"{r['active_rest']:>6,} neurons active  ({r['wall_s']} s)", flush=True)

    summary = {}
    for name, _ in brains:
        mine = [r for r in rows if r["brain"] == name]
        summary[name] = {k: float(np.mean([r[k] for r in mine])) for k in ("spikes_rest", "active_rest")}
    real_s = summary["real"]["spikes_rest"]
    print("\nmean over draws (driven neurons excluded):")
    for name, s in summary.items():
        print(f"  {name:<10} {s['spikes_rest']:>10,.0f} spikes  {s['active_rest']:>8,.0f} active  "
              f"({s['spikes_rest'] / real_s:.2f}x real)")

    with open(args.out or FIGURES / "activity_check.json", "w") as f:
        json.dump({"setting": {"n_driven": N_DRIVEN, "rate_hz": RATE_HZ, "ticks": TICKS,
                               "hub_percentile": HUB_PCT, "pool": int(pool.size),
                               "set_seeds": [1000 + d for d in range(args.draws)]},
                   "summary": summary, "trials": rows}, f, indent=2)
    print(f"wrote {args.out or FIGURES / 'activity_check.json'}")
    return 0


def mean_spikes(pack, sets, gain, log, label):
    rows = [trial(pack, driven, seed=d, gain=gain) for d, driven in enumerate(sets)]
    m = float(np.mean([r["spikes_rest"] for r in rows]))
    a = float(np.mean([r["active_rest"] for r in rows]))
    log.append({"brain": label, "gain": gain, "spikes_rest": m, "active_rest": a, "trials": rows})
    print(f"{label:<10} gain {gain:7.4f}: {m:>10,.0f} spikes  {a:>8,.0f} active", flush=True)
    return m


def match(real, sets, args) -> int:
    """Bisect log(gain) until the shuffle's mean spikes (driven excluded) match the real brain's."""
    pack = core.load_pack(shuffled(PACK_V783, int(args.match[len("shuffled"):])))
    neurons.check_same_neurons(real, pack)
    log = []
    target = mean_spikes(real, sets, 1.0, log, "real")
    lo, hi = 1.0, 2.0
    while mean_spikes(pack, sets, hi, log, args.match) < target:
        lo, hi = hi, hi * 2
        if hi > 64:
            raise SystemExit("no gain up to 64 matches the real brain")
    for _ in range(args.steps):
        mid = float(np.sqrt(lo * hi))
        if mean_spikes(pack, sets, mid, log, args.match) < target:
            lo = mid
        else:
            hi = mid
    best = min((r for r in log if r["brain"] == args.match), key=lambda r: abs(np.log(r["spikes_rest"] / target)))
    print(f"\n{args.match}: gain {best['gain']:.4f} gives {best['spikes_rest']:,.0f} spikes vs real {target:,.0f} "
          f"({best['active_rest']:,.0f} vs {log[0]['active_rest']:,.0f} neurons active)")
    out = args.out or FIGURES / f"activity_match_{args.match}.json"
    with open(out, "w") as f:
        json.dump({"brain": args.match, "target_spikes": target, "gain": best["gain"],
                   "setting": {"n_driven": N_DRIVEN, "rate_hz": RATE_HZ, "ticks": TICKS, "draws": len(sets)},
                   "search": log}, f, indent=2)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
