"""Phase 4: the real brain next to its controls, on the same 90 unseen levels.

Every row went through the same protocol as the real brain in brain/tune.py:
pick settings on the practice levels, then play the unseen test levels once.
  real            the FlyWire v783 wiring                       (figures/phase3_play_real.json)
  lesion          real wiring, both Giant Fibers unable to spike (phase3_play_lesion.json)
  shuffled0-4     degree-preserving scrambled wiring            (phase3_play_shuffledN.json)
  random          press jump on 5% of frames at random          (computed here)
  one number      jump when the spike's angular size > threshold (computed here, threshold
                  picked on the same practice levels)

    python -m brain.phase4           # reads the tune results; the baselines take ~1 min
"""

import json
import random
import statistics
import sys

import flydash
from brain.paths import FIGURES
from brain.tune import PRACTICE, TEST

DIFFS = ("easy", "normal", "hard")
SETUPS = ["real", "lesion"] + [f"shuffled{i}" for i in range(5)]


def play(policy, seed, difficulty, length):
    env = flydash.FlyDash(seed, difficulty, length)
    obs, done, takeoffs = env.reset(), False, []
    while not done:
        jump = policy(obs)
        if jump and env.on_ground:
            takeoffs.append(obs["distance"])
        obs, _, done = env.step(jump)
    return {"seed": seed, "progress": obs["progress"], "won": env.won, "jumps": env.jumps, "takeoffs": takeoffs}


def summarise(rows):
    jd = sorted(d for r in rows for d in r["takeoffs"])
    pct = lambda q: jd[min(len(jd) - 1, int(q * len(jd)))] if jd else None
    return {"won": sum(r["won"] for r in rows), "levels": len(rows),
            "progress": statistics.mean(r["progress"] for r in rows),
            "takeoff_blocks": {"p10": pct(0.1), "median": statistics.median(jd) if jd else None, "p90": pct(0.9)}}


def baselines():
    out = {}
    rng = random.Random(1)
    out["random"] = {"chosen": {"p": 0.05},
                     "test": {d: summarise([play(lambda o: rng.random() < 0.05, s, d, 300.0) for s in TEST]) for d in DIFFS}}
    grid = []
    for th in [x / 100 for x in range(10, 280, 4)]:
        rows = [play(lambda o: o["theta"] > th, s, "hard", 150.0) for s in PRACTICE]
        grid.append((statistics.mean(r["progress"] for r in rows), th))
    th = max(grid, key=lambda g: g[0])[1]          # first (lowest) threshold with the best practice score
    out["one_number"] = {"chosen": {"theta_threshold": th},
                         "test": {d: summarise([play(lambda o: o["theta"] > th, s, d, 300.0) for s in TEST]) for d in DIFFS}}
    return out


def main() -> int:
    rows = {}
    for setup in SETUPS:
        path = FIGURES / f"phase3_play_{setup}.json"
        if not path.exists():
            print(f"missing {path.name}: run  python -m brain.tune --setup {setup}")
            continue
        d = json.loads(path.read_text())
        rows[setup] = {"chosen": d["chosen"], "test": {k: {kk: v[kk] for kk in ("won", "levels", "progress", "takeoff_blocks")}
                                                       for k, v in d["test"].items()}}
    rows.update(baselines())

    print(f"{'setup':<14}{'chosen on practice':<26}" + "".join(f"{d:>16}" for d in DIFFS) + "   median takeoff (hard)")
    for name, r in rows.items():
        ch = ", ".join(f"{k} {v:g}" for k, v in r["chosen"].items() if k in ("r_ref", "steep", "p", "theta_threshold"))
        cells = "".join(f"{r['test'][d]['won']:>4}/{r['test'][d]['levels']} {r['test'][d]['progress']:>5.0%}  " for d in DIFFS)
        med = r["test"]["hard"]["takeoff_blocks"]["median"]
        print(f"{name:<14}{ch:<26}{cells}   {'-' if med is None else f'{med:.2f} blocks'}")
    out = FIGURES / "phase4_controls.json"
    out.write_text(json.dumps({"practice_seeds": list(PRACTICE), "test_seeds": list(TEST), "rows": rows}, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(FIGURES.parent)}   (cells: levels won, mean progress)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
