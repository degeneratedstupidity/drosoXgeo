"""Review #5: add the fly's sensory delay, then sweep it.

In Fly Dash the alarm cells see the game instantly. A real fly needs ~19 ms from light to the Giant Fiber
(Ache et al. 2019), and the model covers only the last ~2 ms of it (alarm cells -> Giant Fiber). Here the
alarm cells get the game as it was D ms earlier (brain.player --delay-ms), for D = 0, 17, 33, 50 ms
(17 ms ~ the missing biology, ~1 frame; 33 and 50 add screen-capture lag for the real game).

For each delay, the same protocol as every other row: tune the two alarm knobs on practice seeds 0-5
(hard, 150 blocks), then play the 30 unseen hard levels (seeds 100-129) once. The grid adds r_ref 20 and 30
to brain.tune's, at every delay, since a late eye needs an earlier alarm. Also played on the test levels:
the delay-0 setting from Phase 3 (r_ref 5, steep 2) without retuning, and the one-number rule seeing the
game the same number of frames late (its threshold retuned on the same practice levels).

    python -m brain.delay run               # resumable, 4 jobs at a time
    python -m brain.delay summary           # -> figures/delay_sweep_<setup>.json and .svg
    python -m brain.delay --setup wire run  # any brain.player setup
    python -m brain.delay --setup fastwire --delays 0,37,45,50 run   # the escape pathway on the CPU (brain.fastpath,
                                            # identical to "wire", ~3.5x faster), at the real game's measured lag
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import flydash
from brain.paths import FIGURES, RESULTS
from brain.tune import PRACTICE, STEEP, TEST

DELAYS = (0, 17, 33, 50)
R_REF = [2, 3, 5, 8, 12, 20, 30]
FIXED = (5, 2)                                # Phase 3's choice for the real brain, delay 0
FIXED_FOR = {"wire": (5, 1), "fastwire": (5, 1)}   # Phase 4b's choice for the escape pathway, delay 0
CHUNKS = [list(TEST)[i:i + 10] for i in range(0, len(TEST), 10)]


def out_dir(setup):
    return RESULTS / "delay" / setup


def job_path(setup, stage, delay, r_ref, steep, seeds):
    return out_dir(setup) / f"{stage}_d{delay}_r{r_ref:g}_s{steep:g}_{seeds[0]}-{seeds[-1]}.json"


def play(setup, stage, delay, r_ref, steep, seeds):
    from brain import neurons
    from brain.player import gain_for, load, play_level, silence_for
    fast = setup == "fastwire"
    if fast:
        setup = "wire"
    pack, lesion = load(setup)
    idx = neurons.indices(pack)
    silence = silence_for(setup, pack, idx)
    if fast:
        from brain.fastpath import use_in_player
        use_in_player(idx)
    length = 150.0 if stage == "practice" else 300.0
    rows = []
    for s in seeds:
        r = play_level(pack, idx, r_ref, steep, s, "hard", lesion, length, silence=silence,
                       gain=gain_for(setup), delay_ms=delay)
        r.pop("trace")
        rows.append(r)
        print(f"{setup} delay {delay} ms {stage} level {s} r_ref {r_ref:g} steep {steep:g}: "
              f"{'WON ' if r['won'] else 'died'} at {r['progress']:4.0%}, {r['jumps']} jumps", flush=True)
    path = job_path("fastwire" if fast else setup, stage, delay, r_ref, steep, seeds)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))


def load_rows(setup, stage, delay, r_ref, steep, chunks):
    rows = []
    for c in chunks:
        p = job_path(setup, stage, delay, r_ref, steep, c)
        if not p.exists():
            return None
        rows += json.loads(p.read_text())
    return rows


def best_setting(setup, delay):
    """As brain.tune: best mean progress, then most wins, then the gentlest steep. None until all are done."""
    grid = []
    for r_ref, steep in itertools.product(R_REF, STEEP):
        rows = load_rows(setup, "practice", delay, r_ref, steep, [list(PRACTICE)])
        if rows is None:
            return None, None
        grid.append({"r_ref": r_ref, "steep": steep, "progress": statistics.mean(r["progress"] for r in rows),
                     "won": sum(r["won"] for r in rows)})
    return max(grid, key=lambda g: (g["progress"], g["won"], -g["steep"])), grid


def run(setup: str, workers: int):
    t0 = time.perf_counter()

    def one(job):
        stage, d, r_ref, steep, seeds = job
        if job_path(setup, stage, d, r_ref, steep, seeds).exists():
            return
        log = out_dir(setup) / "logs" / f"{stage}_d{d}_r{r_ref:g}_s{steep:g}_{seeds[0]}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as f:
            subprocess.run([sys.executable, "-m", "brain.delay", "--setup", setup, "play", stage, str(d),
                            str(r_ref), str(steep), f"{seeds[0]}-{seeds[-1]}"],
                           stdout=f, stderr=subprocess.STDOUT, check=True)
        rows = json.loads(job_path(setup, stage, d, r_ref, steep, seeds).read_text())
        print(f"[{time.strftime('%H:%M:%S')} +{(time.perf_counter() - t0) / 60:5.1f} min] delay {d:>2} ms {stage:<8} "
              f"r_ref {r_ref:>4g} steep {steep:g} levels {seeds[0]}-{seeds[-1]}: "
              f"won {sum(r['won'] for r in rows)}/{len(rows)}", flush=True)

    # Practice first, fixed-setting test levels alongside (they need no tuning), then each delay's tuned test.
    jobs = [("practice", d, r, s, list(PRACTICE)) for d in DELAYS for r, s in itertools.product(R_REF, STEEP)]
    jobs += [("test", d, *FIXED, c) for d in DELAYS for c in CHUNKS]
    print(f"[{time.strftime('%H:%M:%S')}] {setup}: {len(jobs)} jobs before tuned tests, {workers} at a time", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, jobs))
    tuned = []
    for d in DELAYS:
        best, _ = best_setting(setup, d)
        print(f"delay {d} ms: chosen on practice r_ref {best['r_ref']:g}, steep {best['steep']:g} "
              f"(won {best['won']}/6, progress {best['progress']:.0%})", flush=True)
        tuned += [("test", d, best["r_ref"], best["steep"], c) for c in CHUNKS]
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, tuned))
    print(f"[{time.strftime('%H:%M:%S')}] ALL DONE", flush=True)


# ------------------------------------------------------------------ the one-number rule, seeing the game late

def rule_play(th, frames_late, seed, length):
    env = flydash.FlyDash(seed, "hard", length)
    obs, done = env.reset(), False
    seen = collections.deque([0.0] * frames_late)
    takeoffs = []
    while not done:
        seen.append(obs["theta"])
        jump = seen.popleft() > th
        if jump and env.on_ground:
            takeoffs.append(obs["distance"])
        obs, _, done = env.step(jump)
    return {"seed": seed, "progress": obs["progress"], "won": env.won, "jump_distances": takeoffs}


def rule(delay):
    frames_late = round(delay / (flydash.DT * 1000))
    grid = []
    for th in [x / 100 for x in range(10, 280, 4)]:
        rows = [rule_play(th, frames_late, s, 150.0) for s in PRACTICE]
        grid.append((statistics.mean(r["progress"] for r in rows), th))
    th = max(grid, key=lambda g: g[0])[1]
    return {"frames_late": frames_late, "theta_threshold": th},  [rule_play(th, frames_late, s, 300.0) for s in TEST]


# ------------------------------------------------------------------ summary and chart

def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def score(rows):
    jd = [d for r in rows for d in r["jump_distances"]]
    won = sum(r["won"] for r in rows)
    return {"won": won, "levels": len(rows), "interval": wilson(won, len(rows)),
            "progress": statistics.mean(r["progress"] for r in rows),
            "median_takeoff": statistics.median(jd) if jd else None}


def summary(setup: str):
    out = {"setup": setup, "delays_ms": list(DELAYS), "r_ref_grid": R_REF, "steep_grid": STEEP,
           "practice_seeds": list(PRACTICE), "test_seeds": list(TEST), "difficulty": "hard", "rows": []}
    for d in DELAYS:
        best, grid = best_setting(setup, d)
        tuned = load_rows(setup, "test", d, best["r_ref"], best["steep"], CHUNKS) if best else None
        fixed = load_rows(setup, "test", d, *FIXED, CHUNKS)
        if tuned is None or fixed is None:
            print(f"delay {d} ms not finished: run  python -m brain.delay --setup {setup} run")
            return 1
        chosen, rrows = rule(d)
        row = {"delay_ms": d, "chosen": best, "practice_grid": grid, "brain_tuned": score(tuned),
               "brain_fixed": {"r_ref": FIXED[0], "steep": FIXED[1], **score(fixed)},
               "one_number": {**chosen, **score(rrows)}}
        out["rows"].append(row)
        t, f, o = row["brain_tuned"], row["brain_fixed"], row["one_number"]
        mt = "-" if t["median_takeoff"] is None else f"{t['median_takeoff']:.2f}"
        print(f"delay {d:>2} ms: brain retuned (r_ref {best['r_ref']:g}, steep {best['steep']:g}) {t['won']:>2}/30, "
              f"takeoff {mt} blocks | brain at r_ref {FIXED[0]:g} steep {FIXED[1]:g} {f['won']:>2}/30 | "
              f"one-number rule (theta > {o['theta_threshold']:.2f}) {o['won']:>2}/30")
    path = FIGURES / f"delay_sweep_{setup}.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    (FIGURES / f"delay_sweep_{setup}.svg").write_text(chart(out))
    print(f"wrote {path} and {path.with_suffix('.svg').name}")
    return 0


def chart(out):
    """Win rate on the 30 unseen hard levels against the added delay, with 95% intervals."""
    W, H, L, R, T, B = 640, 380, 64, 180, 48, 52
    x = lambda d: L + (W - L - R) * d / max(DELAYS)
    y = lambda p: T + (H - T - B) * (1 - p)
    series = [("brain_tuned", "brain, alarm retuned per delay", "#d9480f"),
              ("brain_fixed", "brain, alarm not retuned", "#868e96"),
              ("one_number", "one-number rule, same delay", "#1971c2")]
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
         f'font-family="Helvetica, Arial, sans-serif" font-size="12" fill="#222222">',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="{L}" y="24" font-size="15" font-weight="bold">Win rate vs sensory delay ({out["setup"]}, 30 unseen hard levels)</text>']
    for p in (0, .25, .5, .75, 1):
        s.append(f'<line x1="{L}" x2="{W - R}" y1="{y(p):.1f}" y2="{y(p):.1f}" stroke="#e9ecef"/>')
        s.append(f'<text x="{L - 8}" y="{y(p) + 4:.1f}" text-anchor="end">{p:.0%}</text>')
    for d in DELAYS:
        s.append(f'<text x="{x(d):.1f}" y="{H - B + 18}" text-anchor="middle">{d}</text>')
    s.append(f'<text x="{(L + W - R) / 2:.1f}" y="{H - 12}" text-anchor="middle">added delay before the alarm cells (ms)</text>')
    s.append(f'<line x1="{x(17):.1f}" x2="{x(17):.1f}" y1="{T}" y2="{H - B}" stroke="#adb5bd" stroke-dasharray="3 3"/>')
    s.append(f'<text x="{x(17) + 4:.1f}" y="{T + 12}" fill="#868e96">real fly</text>')
    for i, (key, label, col) in enumerate(series):
        pts = [(x(r["delay_ms"]), r[key]) for r in out["rows"]]
        s.append('<polyline fill="none" stroke="{}" stroke-width="2" points="{}"/>'.format(
            col, " ".join(f"{px:.1f},{y(v['won'] / v['levels']):.1f}" for px, v in pts)))
        for px, v in pts:
            lo, hi = v["interval"]
            s.append(f'<line x1="{px:.1f}" x2="{px:.1f}" y1="{y(lo):.1f}" y2="{y(hi):.1f}" stroke="{col}" stroke-opacity="0.5"/>')
            s.append(f'<circle cx="{px:.1f}" cy="{y(v["won"] / v["levels"]):.1f}" r="4" fill="{col}"/>')
        ly = T + 30 + 40 * i
        s.append(f'<rect x="{W - R + 14}" y="{ly - 9}" width="12" height="12" fill="{col}"/>')
        for j, part in enumerate(label.split(", ", 1)):
            s.append(f'<text x="{W - R + 32}" y="{ly + 1 + 15 * j}">{part}{"," if j == 0 and ", " in label else ""}</text>')
    s.append("</svg>")
    return "\n".join(s) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setup", default="real")
    ap.add_argument("--delays", default=None, help="comma-separated, ms (default 0,17,33,50)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("play")
    p.add_argument("stage", choices=["practice", "test"])
    p.add_argument("delay", type=int)
    p.add_argument("r_ref", type=float)
    p.add_argument("steep", type=float)
    p.add_argument("seeds")
    r = sub.add_parser("run")
    r.add_argument("--workers", type=int, default=4)
    sub.add_parser("summary")
    a = ap.parse_args()
    global DELAYS, FIXED
    if a.delays:
        DELAYS = tuple(int(d) for d in a.delays.split(","))
    FIXED = FIXED_FOR.get(a.setup, FIXED)
    if a.cmd == "play":
        lo, hi = map(int, a.seeds.split("-"))
        play(a.setup, a.stage, a.delay, a.r_ref, a.steep, list(range(lo, hi + 1)))
    elif a.cmd == "run":
        run(a.setup, a.workers)
    else:
        return summary(a.setup)
    return 0


if __name__ == "__main__":
    sys.exit(main())
