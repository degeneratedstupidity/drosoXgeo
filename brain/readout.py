"""Phase 5A: a trained readout on top of the fixed brain.

The brain's wiring never changes. A small linear "reader" learns to estimate how far away
the next spike is from what the brain's neurons are doing, and the fly jumps when that
estimate reaches the middle of the safe window. The jump rule comes from the game's
geometry; the only thing learned is the reader's weights.

  1. record  The brain watches practice levels (seeds 0-9, all difficulties, 150 blocks)
             while a teacher plays: jump when the spike is ~0.4 blocks away, so every
             approach passes through the whole take-off window on the ground. We store
             which neurons fired in every frame.
  2. train   Ridge regression from neuron activity (this frame and the last) to the spike's
             distance. Regularisation strength picked by cross-validation across practice
             levels. Never sees a test level.
  3. test    The reader plays seeds 100-129 on its own: jump when estimated distance < 0.6.

Two readers, and the difference between them is the experiment:
  A  "all"         every neuron that fires, including the LPLC2/LC4 alarm cells we drive
  B  "downstream"  every neuron except the alarm cells: only what the wiring carries onward

Every setup (real, lesion, shuffledN) records, trains and tests its own readers the same way,
with the same eye (alarm settings r_ref 5 Hz, steep 2 for everyone).

    python -m brain.readout run --setups real,lesion,shuffled0,shuffled1,shuffled2,shuffled3,shuffled4
    python -m brain.readout summary
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import flydash
from brain.paths import FIGURES, RESULTS
from brain.tune import TEST

OUT = RESULTS / "readout"
TRAIN_SEEDS = range(0, 10)
TRAIN_LEN = 150.0
TEST_LEN = 300.0
DIFFS = ("easy", "normal", "hard")
R_REF, STEEP = 5.0, 2.0          # the eye: same alarm encoding for every setup
TEACHER_THETA = 1.75             # teacher jumps when the spike is ~0.42 blocks away
JUMP_AT = 0.6                    # blocks: middle of the hard-level safe window (0.34-0.8)
MAX_D = 3.0                      # train on frames with the spike at most this close
MIN_SPIKES = 20                  # a neuron must fire this often in practice to be read
ALPHAS = [0.1, 1, 10, 100, 1000, 10000]
READERS = {"A": "all neurons that fire", "B": "downstream only (no LPLC2/LC4)"}


# ------------------------------------------------------------------ brain access (needs the simulator)

def _player(setup, seed):
    from brain import neurons
    from brain.player import BrainPlayer, load
    pack, lesion = load(setup)
    idx = neurons.indices(pack)
    return BrainPlayer(pack, idx, R_REF, STEEP, lesion, seed=seed), idx


def record(setup: str, difficulty: str) -> Path:
    """Brain watches the teacher play every practice level of one difficulty."""
    ids, cnts, offs, dist, ground, level = [], [], [0], [], [], []
    for seed in TRAIN_SEEDS:
        player, idx = _player(setup, seed)
        env = flydash.FlyDash(seed, difficulty, TRAIN_LEN)
        obs, done, prev = env.reset(), False, player.brain.spike_counts()
        while not done:
            player(obs)                                    # brain sees this frame
            now = player.brain.spike_counts()
            fired = np.flatnonzero(now != prev)
            ids.append(fired.astype(np.int32)); cnts.append((now - prev)[fired].astype(np.int16))
            offs.append(offs[-1] + fired.size)
            dist.append(obs["distance"]); ground.append(env.on_ground); level.append(seed)
            prev = now
            obs, _, done = env.step(obs["theta"] > TEACHER_THETA)
        print(f"recorded {setup} {difficulty} level {seed}: {'WON' if env.won else 'died'}, {len(dist)} frames so far", flush=True)
    path = OUT / "train" / f"{setup}_{difficulty}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.concatenate(ids), counts=np.concatenate(cnts), offsets=np.array(offs),
                        distance=np.array(dist), ground=np.array(ground), level=np.array(level),
                        alarm=np.concatenate([idx["LPLC2"], idx["LC4"]]))
    return path


# ------------------------------------------------------------------ training (numpy only)

def _features(frames_ids, frames_counts, col):
    """[this frame, previous frame] spike counts for the selected neurons, one row per frame.
    col maps a neuron index to its column, -1 if the neuron isn't read."""
    cur = np.zeros((len(frames_ids), int((col >= 0).sum())), dtype=np.float32)
    for f, (i, c) in enumerate(zip(frames_ids, frames_counts)):
        j = col[i]
        keep = j >= 0
        cur[f, j[keep]] = c[keep]
    prev = np.vstack([np.zeros((1, cur.shape[1]), np.float32), cur[:-1]])
    return np.hstack([cur, prev])


def train(setup: str) -> dict:
    parts = [np.load(OUT / "train" / f"{setup}_{d}.npz") for d in DIFFS]
    alarm = parts[0]["alarm"]
    frames_ids, frames_counts, dist, ground, group, newlevel = [], [], [], [], [], []
    for p, d in zip(parts, DIFFS):
        o, ids, cnt = p["offsets"], p["ids"], p["counts"]       # read each array from disk once
        frames_ids += np.split(ids, o[1:-1]); frames_counts += np.split(cnt, o[1:-1])
        dist.append(p["distance"]); ground.append(p["ground"])
        group.append(np.array([f"{d}{s}" for s in p["level"]]))
        newlevel.append(np.r_[True, p["level"][1:] != p["level"][:-1]])
    dist, ground, group, newlevel = map(np.concatenate, (dist, ground, group, newlevel))
    n_neurons = int(max(i.max() for i in frames_ids if i.size)) + 1
    total = np.zeros(n_neurons)
    for i, c in zip(frames_ids, frames_counts):
        total[i] += c
    model = {"setup": setup}
    for name in READERS:
        cand = np.flatnonzero(total >= MIN_SPIKES)
        if name == "B":
            cand = np.setdiff1d(cand, alarm)
        col = np.full(n_neurons, -1, dtype=np.int64)
        col[cand] = np.arange(cand.size)
        X = _features(frames_ids, frames_counts, col) if cand.size else np.zeros((len(dist), 0), np.float32)
        X[newlevel, cand.size:] = 0                         # no "previous frame" across level boundaries
        use = ground & (dist <= MAX_D)
        Xu, yu, gu = X[use], dist[use], group[use]
        mu, sd = Xu.mean(0), Xu.std(0)
        sd[sd == 0] = 1
        Z = (Xu - mu) / sd
        levels = np.unique(gu)
        folds = [levels[k::5] for k in range(5)]
        cv = {}
        for a in ALPHAS:                                     # leave-levels-out cross-validation
            err = []
            for fold in folds:
                te = np.isin(gu, fold)
                w, b = _ridge(Z[~te], yu[~te], a)
                err.append(np.mean((Z[te] @ w + b - yu[te]) ** 2))
            cv[a] = float(np.mean(err))
        alpha = min(cv, key=cv.get)
        w, b = _ridge(Z, yu, alpha)
        pred = Z @ w + b
        near = yu <= 1.0
        model[name] = {"neurons": cand.tolist(), "mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist(), "b": float(b),
                       "alpha": alpha, "cv_mse": cv, "n_frames": int(use.sum()),
                       "train_mae_near": float(np.mean(np.abs(pred[near] - yu[near]))) if near.any() else None}
        print(f"trained {setup} reader {name}: {cand.size} neurons, alpha {alpha}, "
              f"cv error {cv[alpha] ** 0.5:.2f} blocks (rms), {use.sum()} frames", flush=True)
    path = OUT / "models" / f"{setup}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model))
    return model


def _ridge(Z, y, alpha):
    if Z.shape[1] == 0:
        return np.zeros(0), float(y.mean())
    ym = y.mean()
    A = Z.T @ Z + alpha * np.eye(Z.shape[1])
    w = np.linalg.solve(A, Z.T @ (y - ym))
    return w, ym - float(Z.mean(0) @ w)


# ------------------------------------------------------------------ test: the reader plays

def test(setup: str, reader: str, difficulty: str, seeds) -> Path:
    import mlx.core as mx
    m = json.loads((OUT / "models" / f"{setup}.json").read_text())[reader]
    cols = np.array(m["neurons"], dtype=np.int32)
    mu, sd, w = (np.array(m[k]) for k in ("mu", "sd", "w"))
    b = m["b"]
    rows = []
    for seed in seeds:
        player, _ = _player(setup, seed)
        sel = mx.array(cols)
        env = flydash.FlyDash(seed, difficulty, TEST_LEN)
        obs, done = env.reset(), False
        last = np.asarray(player.brain.counts[sel]).astype(np.float64) if cols.size else np.zeros(0)
        prev_frame = np.zeros(cols.size)
        takeoffs, est_at_takeoff = [], []
        while not done:
            player(obs)
            now = np.asarray(player.brain.counts[sel]).astype(np.float64) if cols.size else np.zeros(0)
            cur = now - last
            est = float(((np.r_[cur, prev_frame] - mu) / sd) @ w + b)
            last, prev_frame = now, cur
            jump = est < JUMP_AT
            if jump and env.on_ground:
                takeoffs.append(obs["distance"]); est_at_takeoff.append(est)
            obs, _, done = env.step(jump)
        rows.append({"seed": seed, "progress": obs["progress"], "won": env.won, "jumps": env.jumps,
                     "takeoffs": [round(t, 3) for t in takeoffs]})
        print(f"{setup:<10} reader {reader} {difficulty:<6} level {seed:>3}: {'WON ' if env.won else 'died'} at "
              f"{obs['progress']:4.0%}, {env.jumps:>3} jumps", flush=True)
    path = OUT / "test" / f"{setup}_{reader}_{difficulty}_{seeds[0]}-{seeds[-1]}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))
    return path


# ------------------------------------------------------------------ orchestration: 4 at a time

def _job(args, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w") as f:
        r = subprocess.run([sys.executable, "-m", "brain.readout", *args], stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError(f"{' '.join(args)} failed, see {log}")


def run(setups, workers):
    t0 = time.perf_counter()
    stamp = lambda: f"[{time.strftime('%H:%M:%S')} +{(time.perf_counter() - t0) / 60:4.1f} min]"
    logs = OUT / "logs"
    with ThreadPoolExecutor(workers) as pool:
        print(f"{stamp()} recording practice levels for {', '.join(setups)}", flush=True)
        todo = [(s, d) for s in setups for d in DIFFS if not (OUT / "train" / f"{s}_{d}.npz").exists()]
        list(pool.map(lambda j: _job(["record", *j], logs / f"record_{j[0]}_{j[1]}.log"), todo))   # keeps earlier recordings
        for setup in setups:
            _job(["train", setup], logs / f"train_{setup}.log")
            print(f"{stamp()} " + "; ".join(open(logs / f"train_{setup}.log").read().strip().splitlines()), flush=True)
        chunks = [list(TEST)[i:i + 10] for i in range(0, len(TEST), 10)]
        jobs = [(s, r, d, c) for s in setups for r in READERS for d in DIFFS for c in chunks
                if not (OUT / "test" / f"{s}_{r}_{d}_{c[0]}-{c[-1]}.json").exists()]      # resume: skip finished batches
        print(f"{stamp()} testing: {len(jobs)} jobs of 10 levels, {workers} at a time", flush=True)

        def one(j):
            s, r, d, c = j
            _job(["test", s, r, d, f"{c[0]}-{c[-1]}"], logs / f"test_{s}_{r}_{d}_{c[0]}.log")
            rows = json.loads((OUT / "test" / f"{s}_{r}_{d}_{c[0]}-{c[-1]}.json").read_text())
            print(f"{stamp()} {s:<10} reader {r} {d:<6} levels {c[0]}-{c[-1]}: won {sum(x['won'] for x in rows)}/{len(rows)}", flush=True)
        list(pool.map(one, jobs))
    print(f"{stamp()} ALL DONE", flush=True)


def summary():
    rows = {}
    for f in sorted((OUT / "test").glob("*.json")):
        setup, reader, diff, _ = f.stem.split("_")
        rows.setdefault((setup, reader), {}).setdefault(diff, []).extend(json.loads(f.read_text()))
    out = {}
    print(f"{'setup':<11}{'reader':<34}" + "".join(f"{d:>12}" for d in DIFFS) + "   median takeoff (hard)")
    for (setup, reader), by in sorted(rows.items(), key=lambda kv: (kv[0][0] != "real", kv[0])):
        cell = {}
        for d in DIFFS:
            r = by.get(d, [])
            jd = sorted(t for x in r for t in x["takeoffs"])
            cell[d] = {"won": sum(x["won"] for x in r), "levels": len(r),
                       "progress": float(np.mean([x["progress"] for x in r])) if r else None,
                       "takeoff_median": float(np.median(jd)) if jd else None}
        out[f"{setup}/{reader}"] = cell
        med = cell["hard"]["takeoff_median"]
        print(f"{setup:<11}{reader + ' ' + READERS[reader]:<34}"
              + "".join(f"{cell[d]['won']:>5}/{cell[d]['levels']:<3}{(cell[d]['progress'] or 0):>4.0%}" for d in DIFFS)
              + f"   {'-' if med is None else f'{med:.2f} blocks'}")
    models = {p.stem: {r: {k: v for k, v in json.loads(p.read_text())[r].items() if k in ("alpha", "n_frames", "train_mae_near")}
                       | {"n_neurons": len(json.loads(p.read_text())[r]["neurons"])} for r in READERS}
              for p in sorted((OUT / "models").glob("*.json"))}
    path = FIGURES / "phase5a_readout.json"
    path.write_text(json.dumps({"train_seeds": list(TRAIN_SEEDS), "test_seeds": list(TEST), "jump_at_blocks": JUMP_AT,
                                "readers": READERS, "models": models, "test": out}, indent=2) + "\n")
    print(f"\nwrote {path.relative_to(FIGURES.parent)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("record"); p.add_argument("setup"); p.add_argument("difficulty")
    p = sub.add_parser("train"); p.add_argument("setup")
    p = sub.add_parser("test"); p.add_argument("setup"); p.add_argument("reader"); p.add_argument("difficulty"); p.add_argument("seeds")
    p = sub.add_parser("run"); p.add_argument("--setups", default="real"); p.add_argument("--workers", type=int, default=4)
    sub.add_parser("summary")
    a = ap.parse_args()
    if a.cmd == "record":
        record(a.setup, a.difficulty)
    elif a.cmd == "train":
        train(a.setup)
    elif a.cmd == "test":
        lo, hi = map(int, a.seeds.split("-"))
        test(a.setup, a.reader, a.difficulty, list(range(lo, hi + 1)))
    elif a.cmd == "run":
        run(a.setups.split(","), a.workers)
    else:
        summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
