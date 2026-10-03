"""Record one level as the brain plays it: the game, and every neuron that fires, frame by frame.

Feeds the brain replay page (docs/brainreplay.template.html). For each 1/60 s frame it
stores the player's position, the alarm rate, the jump cell's spike count, and
which neurons anywhere in the brain fired, with how many spikes. Positions come from
the FlyWire annotations: the soma (cell body) where known, else a point on the neuron,
seen from the front of the head.

Records the same level and the same alarm on two setups (by default the real wiring and
a scrambled one), so the page can switch between them.

    python -m brain.record_level --difficulty hard --seed 1 --length 120
    -> results/replay/<difficulty>_<seed>.json
"""

import argparse
import base64
import csv
import json
import sys

import numpy as np

import flydash
from brain import neurons
from brain.paths import DATA, RESULTS
from brain.player import BrainPlayer, load

SUPER = ["optic", "central", "visual_projection", "sensory", "descending", "ascending",
         "sensory_ascending", "visual_centrifugal", "motor", "endocrine", ""]


def b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def annotations(pack):
    """Front-view x, y (micrometres), super class code, cell type and side for every pack neuron."""
    info = {}
    with open(DATA / "neuron_annotations.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            sx, sy = r["soma_x"], r["soma_y"]
            x, y = (float(sx), float(sy)) if sx and sy else (float(r["pos_x"]), float(r["pos_y"]))
            info[int(r["root_id"])] = (x * 4 / 1000, y * 4 / 1000, r["super_class"], r["cell_type"], r["side"])
    n = pack.n_neurons
    xs, ys = np.full(n, np.nan), np.full(n, np.nan)
    sup = np.full(n, len(SUPER) - 1, dtype=np.uint8)
    types, sides = [""] * n, [""] * n
    for i, rid in enumerate(pack.neuron_ids.tolist()):
        if rid in info:
            x, y, s, t, side = info[rid]
            xs[i], ys[i], types[i], sides[i] = x, y, t, side
            sup[i] = SUPER.index(s) if s in SUPER else len(SUPER) - 1
    return xs, ys, sup, types, sides


def record(setup, difficulty, seed, length, r_ref, steep):
    pack, lesion = load(setup)
    idx = neurons.indices(pack)
    env = flydash.FlyDash(seed, difficulty, length)
    player = BrainPlayer(pack, idx, r_ref, steep, lesion, seed=seed)
    obs, done = env.reset(), False
    prev = player.brain.spike_counts()
    frames = []
    while not done:
        jump = player(obs)
        now = player.brain.spike_counts()
        fired = np.flatnonzero(now != prev)
        frames.append((env.px, env.py, player.last_rate, fired, (now - prev)[fired], player.last_gf,
                       bool(jump and env.on_ground), obs["distance"]))
        prev = now
        obs, _, done = env.step(jump)
    print(f"{setup:<10} {difficulty} level {seed}: {'WON' if env.won else 'died'} at {obs['progress']:.0%}, "
          f"{env.jumps} jumps, {len(frames)} frames, "
          f"{len(np.unique(np.concatenate([f[3] for f in frames])))} different neurons fired", flush=True)
    return pack, idx, env, frames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--difficulty", default="hard", choices=list(flydash.DIFFICULTY))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--length", type=float, default=120.0)
    ap.add_argument("--r-ref", type=float, default=5.0)
    ap.add_argument("--steep", type=float, default=2.0)
    ap.add_argument("--setups", default="real,shuffled0")
    args = ap.parse_args()

    runs = {}
    for setup in args.setups.split(","):
        pack, idx, env, frames = record(setup, args.difficulty, args.seed, args.length, args.r_ref, args.steep)
        runs[setup] = (env, frames)

    xs, ys, sup, types, sides = annotations(pack)          # same neuron list in every setup
    ok = ~np.isnan(xs)
    x0, x1, y0, y1 = xs[ok].min(), xs[ok].max(), ys[ok].min(), ys[ok].max()
    qx = np.where(ok, (xs - x0) / (x1 - x0) * 65535, 0).astype(np.uint16)
    qy = np.where(ok, (ys - y0) / (y1 - y0) * 65535, 0).astype(np.uint16)

    # neurons that fire in any run get a short local id and their names
    active = np.unique(np.concatenate([f[3] for _, fr in runs.values() for f in fr]))
    local = {int(g): i for i, g in enumerate(active)}
    group = np.zeros(active.size, dtype=np.uint8)               # 0 other, 1 LPLC2, 2 LC4, 3 DNp01
    for code, name in ((1, "LPLC2"), (2, "LC4"), (3, "DNp01")):
        group[np.isin(active, idx[name])] = code

    out = {
        "level": {"difficulty": args.difficulty, "seed": args.seed, "length": args.length,
                  "spikes": [s.x for s in flydash.make_level(args.seed, args.difficulty, args.length)],
                  "speed": flydash.SPEED, "dt": flydash.DT},
        "alarm": {"r_ref": args.r_ref, "steep": args.steep},
        "brain": {"n": int(pack.n_neurons), "x": b64(qx), "y": b64(qy), "super": b64(sup), "super_names": SUPER,
                  "width_um": float(x1 - x0), "height_um": float(y1 - y0), "has_pos": int(ok.sum())},
        "active": {"index": b64(active.astype(np.uint32)), "group": b64(group),
                   "type": [types[i] for i in active], "side": [sides[i] for i in active],
                   "super": [SUPER[sup[i]] for i in active]},
        "runs": {},
    }
    for setup, (env, frames) in runs.items():
        ids = np.concatenate([np.array([local[int(g)] for g in f[3]], dtype=np.uint16) for f in frames])
        cnt = np.concatenate([np.minimum(f[4], 255).astype(np.uint8) for f in frames])
        offs = np.cumsum([0] + [len(f[3]) for f in frames]).astype(np.uint32)
        out["runs"][setup] = {
            "won": env.won, "progress": env.px / args.length, "jumps": env.jumps,
            "px": [round(f[0], 3) for f in frames], "py": [round(f[1], 3) for f in frames],
            "alarm": [round(f[2], 2) for f in frames], "gf": [f[5] for f in frames],
            "takeoff": [i for i, f in enumerate(frames) if f[6]],
            "distance": [round(f[7], 3) for f in frames],
            "fired_ids": b64(ids), "fired_counts": b64(cnt), "frame_offsets": b64(offs),
        }
    path = RESULTS / "replay" / f"{args.difficulty}_{args.seed}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB), {active.size} neurons fired in some run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
