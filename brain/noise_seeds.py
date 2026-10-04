"""How much of 89/90 is luck? (SUGGESTIONS.md #3)

The alarm cells get random (Poisson) input spikes: we set the rate, not the exact spike times. Phase 3 played
each unseen level once, with one draw of those spikes (noise seed 0). This replays the same 90 levels with
noise seeds 1-4, same brain, same settings (r_ref 5 Hz, steep 2, chosen on practice levels in Phase 3),
and reports the win rate over all 5 x 90 plays with a 95% interval.

    python -m brain.noise_seeds run          # ~1-1.5 h on the M1 Pro, 4 jobs at a time, resumable
    python -m brain.noise_seeds summary      # -> figures/noise_seeds_real.json
    python -m brain.noise_seeds --setup wire run    # any setup tuned by brain.tune (settings from phase3_play_<setup>.json)
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from brain.paths import FIGURES, RESULTS
from brain.tune import TEST

NOISES = (1, 2, 3, 4)
DIFFS = ("easy", "normal", "hard")


def out_dir(setup):
    return RESULTS / "noise_seeds" if setup == "real" else RESULTS / "noise_seeds" / setup


def chosen(setup):
    return json.loads((FIGURES / f"phase3_play_{setup}.json").read_text())["chosen"]


def play(setup: str, noise: int, difficulty: str, seeds: list[int]):
    from brain import neurons
    from brain.player import load, play_level, silence_for
    pack, lesion = load(setup)
    idx = neurons.indices(pack)
    silence = silence_for(setup, pack, idx)
    c = chosen(setup)
    rows = []
    for s in seeds:
        r = play_level(pack, idx, c["r_ref"], c["steep"], s, difficulty, lesion, noise=noise, silence=silence)
        r.pop("trace")
        rows.append(r)
        print(f"{setup} noise {noise} {difficulty:<6} level {s}: {'WON ' if r['won'] else 'died'} at {r['progress']:4.0%}, "
              f"{r['jumps']} jumps", flush=True)
    path = out_dir(setup) / f"{noise}_{difficulty}_{seeds[0]}-{seeds[-1]}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))


def run(setup: str, workers: int):
    OUT = out_dir(setup)
    t0 = time.perf_counter()
    chunks = [list(TEST)[i:i + 10] for i in range(0, len(TEST), 10)]
    jobs = [(n, d, c) for n in NOISES for d in DIFFS for c in chunks
            if not (OUT / f"{n}_{d}_{c[0]}-{c[-1]}.json").exists()]          # resume: skip finished batches
    print(f"[{time.strftime('%H:%M:%S')}] {len(jobs)} batches of 10 levels, {workers} at a time", flush=True)

    def one(job):
        n, d, c = job
        log = OUT / "logs" / f"{n}_{d}_{c[0]}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as f:
            subprocess.run([sys.executable, "-m", "brain.noise_seeds", "--setup", setup, "play", str(n), d, f"{c[0]}-{c[-1]}"],
                           stdout=f, stderr=subprocess.STDOUT, check=True)
        rows = json.loads((OUT / f"{n}_{d}_{c[0]}-{c[-1]}.json").read_text())
        print(f"[{time.strftime('%H:%M:%S')} +{(time.perf_counter() - t0) / 60:4.1f} min] noise {n} {d:<6} "
              f"levels {c[0]}-{c[-1]}: won {sum(r['won'] for r in rows)}/{len(rows)}", flush=True)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, jobs))
    print(f"[{time.strftime('%H:%M:%S')}] ALL DONE", flush=True)


def wilson(k: int, n: int, z: float = 1.96):
    """95% interval for a win rate of k out of n (Wilson score interval; behaves well near 100%)."""
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def summary(setup: str):
    OUT = out_dir(setup)
    phase3 = json.loads((FIGURES / f"phase3_play_{setup}.json").read_text())["test"]
    plays = {(0, d, r["seed"]): r["won"] for d in DIFFS for r in phase3[d]["per_level"]}
    for f in OUT.glob("*_*_*.json"):
        n, d, _ = f.stem.split("_")
        for r in json.loads(f.read_text()):
            plays[(int(n), d, r["seed"])] = r["won"]
    noises = sorted({k[0] for k in plays})
    print(f"{'noise seed':<12}" + "".join(f"{d:>10}" for d in DIFFS) + f"{'total':>10}")
    per_noise = {}
    for n in noises:
        by = {d: [w for (nn, dd, _), w in plays.items() if nn == n and dd == d] for d in DIFFS}
        per_noise[n] = {d: f"{sum(v)}/{len(v)}" for d, v in by.items()} | {"total": f"{sum(map(sum, by.values()))}/{sum(map(len, by.values()))}"}
        print(f"{n:<12}" + "".join(f"{per_noise[n][d]:>10}" for d in DIFFS) + f"{per_noise[n]['total']:>10}")
    k, total = sum(plays.values()), len(plays)
    lo, hi = wilson(k, total)
    lost = sorted({(d, s) for (n, d, s), w in plays.items() if not w})
    print(f"\nall plays: won {k}/{total} = {k / total:.1%}, 95% interval {lo:.1%}-{hi:.1%}")
    print(f"levels lost at least once: {len(lost)} of 90: " + ", ".join(f"{d} {s}" for d, s in lost))
    lo1, hi1 = wilson(90, 90)
    print(f"for comparison, 90/90 (one-number rule, deterministic) has a 95% interval of {lo1:.1%}-{hi1:.1%}")
    out = FIGURES / f"noise_seeds_{setup}.json"
    out.write_text(json.dumps({"setup": setup, "setting": chosen(setup), "test_seeds": list(TEST), "noise_seeds": noises,
                               "per_noise": per_noise, "won": k, "plays": total, "win_rate": k / total,
                               "ci95": [lo, hi], "levels_lost_at_least_once": [f"{d} {s}" for d, s in lost]}, indent=2) + "\n")
    print(f"wrote {out.relative_to(FIGURES.parent)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup", default="real")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("play"); p.add_argument("noise", type=int); p.add_argument("difficulty"); p.add_argument("seeds")
    p = sub.add_parser("run"); p.add_argument("--workers", type=int, default=4)
    sub.add_parser("summary")
    a = ap.parse_args()
    if a.cmd == "play":
        lo, hi = map(int, a.seeds.split("-"))
        play(a.setup, a.noise, a.difficulty, list(range(lo, hi + 1)))
    elif a.cmd == "run":
        run(a.setup, a.workers)
    else:
        summary(a.setup)
    return 0


if __name__ == "__main__":
    sys.exit(main())
