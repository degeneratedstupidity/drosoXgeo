"""Stretch C stage 2: does "what's coming" survive downstream of the looming cells?

With vision input (brain.vision) a spike ahead, a block to jump onto and a thorn pit drive different LPLC2/LC4 cells.
Learning to time jumps (stage 3) only makes sense if the rest of the brain carries that difference onward, and
the claim is only about the wiring if the real wiring carries it better than a scrambled one.

  1. record  every brain watches the same terrain levels along the same winning path (flydash.plan: a planner
             plays, the brain only watches), with the same input spikes. Stored per frame: which neurons fired, the
             314 input rates, and the label: the nearest thing ahead within 4 blocks (clear / spike / block / pit).
  2. read    a linear reader (ridge regression to the four classes) learns the label from the downstream neurons
             only (everything except the 314 driven cells, as Reader B in Phase 5A), this frame and the last. Its
             strength is picked by leave-levels-out cross-validation on practice levels (seeds 0-9); it is scored
             once on unseen levels (seeds 100-109). Score: balanced accuracy (mean hit rate over the four classes;
             chance 25%), on frames where the cube stands (where a jump could be decided).
  3. compare real wiring vs the same scramble plain (shuffled0) and turned up to the real brain's activity
             (shuffled0g2.83, Phase 4c), and the input itself (the 314 rates) as the ceiling.

Gate: the real wiring must carry clearly more than the gain-matched scramble. If not, the wiring adds nothing to
what is coming, and stage 3 stops.

    python -m brain.context run                 # record (resumable, 3 jobs at a time) and read all setups
    python -m brain.context summary             # -> figures/stretchc_context.json
    python -m brain.context equal               # same number of neurons per brain -> figures/stretchc_context_equal.json

The gain-matched scramble (x2.83, matched in Phase 4c for the single-rate input) is far more active than the real
brain under vision input (9,771 downstream neurons fire often enough to read vs 318), so "equal" gives every brain's
reader the same number of neurons: its most active downstream ones (69, the plain scramble's count, and 318).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import flydash
from brain.paths import FIGURES, RESULTS

OUT = RESULTS / "context"
SETUPS = ("real", "shuffled0", "shuffled0g2.83")
PRACTICE, TEST = range(0, 10), range(100, 110)
LENGTH = 150.0
R_REF, STEEP = 25.0, 1.0          # vision alarm as tuned for the real game's lag (37 ms; stage 1)
HORIZON = 4.0                     # blocks: "what's coming" = the nearest object this close
KINDS = ("clear", "spike", "block", "pit")
MIN_SPIKES = 20                   # a neuron must fire this often over the practice levels to be read
ALPHAS = [1, 10, 100, 1000, 10000, 100000]


def label(objects) -> int:
    near = [(d, kind) for d, _, kind in objects if d <= HORIZON]
    return KINDS.index(min(near)[1]) if near else 0


# ------------------------------------------------------------------ recording (needs the simulator)

def record(setup: str, seed: int):
    from brain import neurons
    from brain.player import BrainPlayer, gain_for, load
    from brain.vision import Vision
    pack, lesion = load(setup)
    idx = neurons.indices(pack)
    vision = Vision(pack, idx)
    player = BrainPlayer(pack, idx, R_REF, STEEP, lesion, seed=seed, gain=gain_for(setup), vision=vision)
    level = flydash.make_terrain(seed, LENGTH)
    jumps = flydash.plan(level, LENGTH)
    env = flydash.FlyDashTerrain(level=level, length=LENGTH)
    obs, done, frame = env.reset(), False, 0
    ids, cnts, offs, rates, labels, ground = [], [], [0], [], [], []
    prev = player.brain.spike_counts()
    while not done:
        player(obs)
        now = player.brain.spike_counts()
        fired = np.flatnonzero(now != prev)
        ids.append(fired.astype(np.int32))
        cnts.append((now - prev)[fired].astype(np.int16))
        offs.append(offs[-1] + fired.size)
        r = vision.rates(obs["objects"], R_REF, STEEP)
        rates.append(np.concatenate([r["LPLC2"], r["LC4"]]).astype(np.float32))
        labels.append(label(obs["objects"]))
        ground.append(env.on_ground)
        prev = now
        obs, _, done = env.step(frame in jumps)
        frame += 1
    path = OUT / setup / f"{seed}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, ids=np.concatenate(ids), counts=np.concatenate(cnts), offsets=np.array(offs),
                        rates=np.stack(rates), label=np.array(labels), ground=np.array(ground),
                        alarm=np.concatenate([idx["LPLC2"], idx["LC4"]]), won=env.won)
    print(f"{setup} level {seed}: {frame} frames, {'WON' if env.won else 'died'} "
          f"(labels {np.bincount(labels, minlength=4).tolist()})", flush=True)


# ------------------------------------------------------------------ reading (numpy only)

def _load(setup: str, seeds):
    parts = []
    for s in seeds:
        p = np.load(OUT / setup / f"{s}.npz")
        o = p["offsets"]
        parts.append({"ids": np.split(p["ids"], o[1:-1]), "counts": np.split(p["counts"], o[1:-1]),
                      "rates": p["rates"], "label": p["label"], "ground": p["ground"], "alarm": p["alarm"],
                      "level": np.full(len(p["label"]), s)})
    return parts


def _neural(parts, cols, n_cols):
    """[this frame, previous frame] spike counts of the read neurons; no "previous" across levels."""
    blocks = []
    for p in parts:
        cur = np.zeros((len(p["label"]), n_cols), np.float32)
        for f, (i, c) in enumerate(zip(p["ids"], p["counts"])):
            j = cols[i]
            keep = j >= 0
            cur[f, j[keep]] = c[keep]
        prev = np.vstack([np.zeros((1, n_cols), np.float32), cur[:-1]])
        blocks.append(np.hstack([cur, prev]))
    return np.vstack(blocks)


def _input(parts):
    blocks = []
    for p in parts:
        cur = p["rates"]
        blocks.append(np.hstack([cur, np.vstack([np.zeros((1, cur.shape[1]), np.float32), cur[:-1]])]))
    return np.vstack(blocks)


def _balanced(y, pred):
    return float(np.mean([np.mean(pred[y == k] == k) for k in range(len(KINDS)) if np.any(y == k)]))


class Ridge:
    """Ridge regression to one-hot classes in the dual form (many more neurons than frames)."""

    def fit(self, X, y, alpha):
        self.mu, sd = X.mean(0), X.std(0)
        self.sd = np.where(sd == 0, 1, sd)
        Z = (X - self.mu) / self.sd
        Y = np.eye(len(KINDS))[y]
        self.ym = Y.mean(0)
        K = Z @ Z.T
        A = np.linalg.solve(K + alpha * np.eye(len(K)), Y - self.ym)
        self.W = Z.T @ A
        return self

    def predict(self, X):
        return np.argmax(((X - self.mu) / self.sd) @ self.W + self.ym, axis=1)


def read(setup: str) -> dict:
    """Train on practice (alpha by leave-levels-out CV), score once on the unseen levels."""
    train, test = _load(setup, PRACTICE), _load(setup, TEST)
    alarm = train[0]["alarm"]
    n = int(max(int(i.max()) for p in train for i in p["ids"] if i.size)) + 1
    total = np.zeros(n)
    for p in train:
        for i, c in zip(p["ids"], p["counts"]):
            total[i] += c
    keep = np.setdiff1d(np.flatnonzero(total >= MIN_SPIKES), alarm)          # downstream only
    cols = np.full(max(n, int(max(int(i.max()) for p in test for i in p["ids"] if i.size)) + 1), -1, np.int64)
    cols[keep] = np.arange(keep.size)
    result = {"setup": setup, "neurons_read": int(keep.size),
              "downstream_spikes_per_frame": float(sum(c.sum() for p in train for i, c in zip(p["ids"], p["counts"])
                                                       if True) / sum(len(p["label"]) for p in train))}
    cat = lambda parts, k: np.concatenate([p[k] for p in parts])
    y_tr, y_te = cat(train, "label"), cat(test, "label")
    g_tr, g_te = cat(train, "ground"), cat(test, "ground")
    lv_tr = cat(train, "level")
    for name, X_tr, X_te in (("downstream", _neural(train, cols, keep.size), _neural(test, cols, keep.size)),
                             ("input", _input(train), _input(test))):
        if X_tr.shape[1] == 0:
            result[name] = {"cv": None, "test": 0.25, "note": "no downstream neuron fired"}
            continue
        Xt, yt, lt = X_tr[g_tr], y_tr[g_tr], lv_tr[g_tr]
        folds = [list(PRACTICE)[k::5] for k in range(5)]
        cv = {}
        for a in ALPHAS:
            scores = []
            for f in folds:
                held = np.isin(lt, f)
                m = Ridge().fit(Xt[~held], yt[~held], a)
                scores.append(_balanced(yt[held], m.predict(Xt[held])))
            cv[a] = float(np.mean(scores))
        best = max(cv, key=cv.get)
        m = Ridge().fit(Xt, yt, best)
        pred = m.predict(X_te[g_te])
        conf = np.zeros((len(KINDS), len(KINDS)), int)
        for a_, b_ in zip(y_te[g_te], pred):
            conf[a_, b_] += 1
        result[name] = {"features": int(X_tr.shape[1]), "alpha": best, "cv": cv[best],
                        "test": _balanced(y_te[g_te], pred), "confusion": conf.tolist()}
        print(f"{setup:>15} {name:>10}: {X_tr.shape[1]:6d} features, alpha {best:g}, practice CV {cv[best]:.0%}, "
              f"unseen levels {result[name]['test']:.0%} (balanced accuracy, chance 25%)", flush=True)
    return result


def equal(sizes=(69, 318)) -> dict:
    """Each brain read through its N most active downstream neurons; per-level scores and bootstrap intervals."""
    out, per = {"sizes": list(sizes), "rows": []}, {}
    rng = np.random.default_rng(0)
    for setup in SETUPS:
        train, test = _load(setup, PRACTICE), _load(setup, TEST)
        n = 1 + max(int(i.max()) for parts in (train, test) for p in parts for i in p["ids"] if i.size)
        total = np.zeros(n)
        for p in train:
            for i, c in zip(p["ids"], p["counts"]):
                total[i] += c
        total[train[0]["alarm"]] = 0
        order = np.argsort(-total)
        cat = lambda parts, k: np.concatenate([p[k] for p in parts])
        y_tr, g_tr, lv_tr = cat(train, "label"), cat(train, "ground"), cat(train, "level")
        y_te, g_te, lv_te = cat(test, "label"), cat(test, "ground"), cat(test, "level")
        folds = [list(PRACTICE)[k::5] for k in range(5)]
        for size in sizes:
            keep = order[:size]
            keep = keep[total[keep] > 0]
            cols = np.full(n, -1, np.int64)
            cols[keep] = np.arange(keep.size)
            X_tr = _neural(train, cols, keep.size)[g_tr]
            yt, lt = y_tr[g_tr], lv_tr[g_tr]
            cv = {a: float(np.mean([_balanced(yt[np.isin(lt, f)], Ridge().fit(
                X_tr[~np.isin(lt, f)], yt[~np.isin(lt, f)], a).predict(X_tr[np.isin(lt, f)])) for f in folds]))
                  for a in ALPHAS}
            best = max(cv, key=cv.get)
            pred = Ridge().fit(X_tr, yt, best).predict(_neural(test, cols, keep.size)[g_te])
            ye, le = y_te[g_te], lv_te[g_te]
            levels = [_balanced(ye[le == s], pred[le == s]) for s in TEST]
            boots = [np.mean(rng.choice(levels, len(levels))) for _ in range(2000)]
            per[(setup, size)] = levels
            row = {"setup": setup, "size": size, "active": int(keep.size), "alpha": best, "test": _balanced(ye, pred),
                   "per_level": levels, "interval": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}
            out["rows"].append(row)
            print(f"{setup:>15} top {size:3d} downstream neurons: unseen {row['test']:.0%} (per-level mean "
                  f"{np.mean(levels):.0%}, 95% {row['interval'][0]:.0%}-{row['interval'][1]:.0%})", flush=True)
    for size in sizes:
        d = np.array(per[("real", size)]) - np.array(per[("shuffled0g2.83", size)])
        out[f"real_minus_gain_matched_{size}"] = {"mean": float(d.mean()), "levels_real_better": int((d > 0).sum())}
        print(f"top {size}: real minus gain-matched scramble {d.mean():+.1%} per level, real better on {int((d > 0).sum())}/10")
    (FIGURES / "stretchc_context_equal.json").write_text(json.dumps(out, indent=2) + "\n")
    return out


def run(workers: int):
    jobs = [(s, seed) for s in SETUPS for seed in (*PRACTICE, *TEST) if not (OUT / s / f"{seed}.npz").exists()]
    t0 = time.perf_counter()

    def one(job):
        s, seed = job
        log = OUT / "logs" / f"{s}_{seed}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as f:
            subprocess.run([sys.executable, "-m", "brain.context", "record", s, str(seed)],
                           stdout=f, stderr=subprocess.STDOUT, check=True)
        print(f"[+{(time.perf_counter() - t0) / 60:5.1f} min] recorded {s} level {seed}", flush=True)

    print(f"{len(jobs)} recordings to do, {workers} at a time", flush=True)
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, jobs))
    summary()


def summary():
    rows = [read(s) for s in SETUPS]
    out = {"practice_seeds": list(PRACTICE), "test_seeds": list(TEST), "length": LENGTH, "r_ref": R_REF,
           "steep": STEEP, "horizon": HORIZON, "kinds": list(KINDS), "rows": rows}
    path = FIGURES / "stretchc_context.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("setup")
    r.add_argument("seed", type=int)
    w = sub.add_parser("run")
    w.add_argument("--workers", type=int, default=3)
    sub.add_parser("summary")
    sub.add_parser("equal")
    a = ap.parse_args()
    if a.cmd == "record":
        record(a.setup, a.seed)
    elif a.cmd == "run":
        run(a.workers)
    elif a.cmd == "equal":
        equal()
    else:
        summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
