"""Tune the two alarm knobs on practice levels, then score once on unseen test levels.

Practice and test levels are different seeds, so a good test score can't come from
memorising the practice levels. Every setup (real, shuffled, lesion) gets the same
search, so a control is never handicapped by settings tuned for the real brain.

    python -m brain.tune --setup real
"""

import argparse
import itertools
import json
import sys
import time

import numpy as np

from brain import neurons
from brain.paths import FIGURES
from brain.player import gain_for, load, play_level, silence_for

R_REF = [2, 3, 5, 8, 12]
STEEP = [1, 2, 3]
PRACTICE = range(0, 6)
TEST = range(100, 130)


def score(pack, idx, lesion, r_ref, steep, seeds, difficulty, length, silence=(), gain=1.0):
    rows = []
    for s in seeds:
        r = play_level(pack, idx, r_ref, steep, s, difficulty, lesion, length, silence=silence, gain=gain)
        rows.append(r)
        print(f"    {difficulty:<6} level {s:>3}  r_ref {r_ref:>3} steep {steep}:  "
              f"{'WON ' if r['won'] else 'died'} at {r['progress']:4.0%}, {r['jumps']:>3} jumps", flush=True)
    return float(np.mean([r["progress"] for r in rows])), int(sum(r["won"] for r in rows)), rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup", default="real")
    ap.add_argument("--practice-difficulty", default="hard")
    args = ap.parse_args()

    pack, lesion = load(args.setup)
    idx = neurons.indices(pack)
    silence = silence_for(args.setup, pack, idx)
    gain = gain_for(args.setup)
    t0 = time.perf_counter()
    grid = []
    for r_ref, steep in itertools.product(R_REF, STEEP):
        prog, won, _ = score(pack, idx, lesion, r_ref, steep, PRACTICE, args.practice_difficulty, 150.0, silence, gain)
        grid.append({"r_ref": r_ref, "steep": steep, "progress": prog, "won": won})
        print(f"practice  r_ref {r_ref:>4} Hz  steep {steep}:  won {won}/{len(PRACTICE)}  progress {prog:4.0%}", flush=True)
    best = max(grid, key=lambda g: (g["progress"], g["won"], -g["steep"]))
    print(f"\nchosen on practice levels: r_ref {best['r_ref']} Hz, steep {best['steep']}\n")

    test = {}
    for diff in ("easy", "normal", "hard"):
        prog, won, rows = score(pack, idx, lesion, best["r_ref"], best["steep"], TEST, diff, 300.0, silence, gain)
        jd = [d for r in rows for d in r["jump_distances"]]
        test[diff] = {"progress": prog, "won": won, "levels": len(rows),
                      "takeoff_blocks": {"p10": float(np.percentile(jd, 10)) if jd else None,
                                         "median": float(np.median(jd)) if jd else None,
                                         "p90": float(np.percentile(jd, 90)) if jd else None},
                      "per_level": [{k: r[k] for k in ("seed", "progress", "won", "jumps")} for r in rows]}
        print(f"TEST {diff:<6} won {won}/{len(rows)}, mean progress {prog:.0%}", flush=True)
    out = FIGURES / f"phase3_play_{args.setup}.json"
    out.write_text(json.dumps({"setup": args.setup, "practice_seeds": list(PRACTICE), "test_seeds": list(TEST),
                               "practice_grid": grid, "chosen": best, "test": test}, indent=2) + "\n")
    print(f"\nwrote {out}  ({time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
