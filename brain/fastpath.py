"""The escape pathway on the CPU, fast enough for the real game: LPLC2 + LC4 + the two Giant Fibers (316 neurons).

This is the "wire" setup of brain.player (Phase 4b: every other neuron silenced, 448/450 unseen levels won), but
without the other 138,323 neurons. In "wire" those neurons can't affect the 316: silenced means their synapses
deliver nothing. So the 316 and the synapses among them are the whole story, and 316 neurons fit in a few
small numpy arrays.

Same physics as the GPU model (lif.engine_fused's state kernel, transcribed line by line in step_draws), same
constants (lif.core), float32 like the GPU, same input draws for the same seed (columns LPLC2 then LC4, as
brain.stepper). python -m brain.fastpath checks it against the GPU spike for spike.

    python -m brain.fastpath                         # check vs the GPU model, then speed
    python -m brain.fastpath levels 1 hard           # replay saved Phase 4b plays (wire, noise 1-4), must match
"""

from __future__ import annotations

import sys
import time

import numpy as np

from lif import core

from brain import neurons
from brain.stepper import NEVER, TICK_S


class FastPathway:
    """Drop-in for brain.stepper.Brain with inputs LPLC2/LC4 and monitor DNp01, on the escape pathway only."""

    def __init__(self, pack: core.Pack, idx: dict, lesion: bool = False, seed: int = 0):
        keep = np.concatenate([idx["LPLC2"], idx["LC4"], idx["DNp01"]]).astype(np.int64)
        local = np.full(pack.n_neurons, -1, dtype=np.int64)
        local[keep] = np.arange(keep.size)
        n = keep.size
        # synapse counts among the 316, as a dense source x destination matrix
        row_ptr, dst = np.asarray(pack.row_ptr), np.asarray(pack.destinations)
        counts = np.asarray(pack.signed_counts)
        w = np.zeros((n, n), dtype=np.int64)
        for i, src in enumerate(keep):
            a, b = row_ptr[src], row_ptr[src + 1]
            d = local[dst[a:b]]
            ok = d >= 0
            np.add.at(w[i], d[ok], counts[a:b][ok])
        self.w = w
        self.n, self.n_targets = n, idx["LPLC2"].size + idx["LC4"].size
        self.groups = {"LPLC2": idx["LPLC2"].size, "LC4": idx["LC4"].size}
        self.gf = np.arange(self.n_targets, n)              # DNp01, in the order given
        self.c = core.constants_f32()
        # as brain.stepper: the driven cells are never refractory; a lesioned Giant Fiber always is
        self.rfc_reload = np.full(n, core.RFC_TICKS, dtype=np.int32)
        self.rfc_reload[: self.n_targets] = 0
        if lesion:
            self.rfc_reload[self.gf] = NEVER
        self.lesion = lesion
        self.reset(seed)

    def reset(self, seed: int = 0):
        """Back to rest (as a fresh FastPathway), with a new input-spike seed. Cheap: no rebuilding of the wiring."""
        n = self.n
        self.rfc = np.zeros(n, dtype=np.int32)
        if self.lesion:
            self.rfc[self.gf] = NEVER
        self.v = np.full(n, self.c["v_0"], dtype=np.float32)
        self.g = np.zeros(n, dtype=np.float32)
        self.ring = np.zeros((core.DELAY_TICKS, n), dtype=bool)
        self.tick = 0
        self.rng = np.random.default_rng(seed)
        self.last_counts = np.zeros(n, dtype=np.int32)    # spikes per neuron during the last step() call

    def step(self, n_ticks: int, rates: dict[str, float | np.ndarray]) -> np.ndarray:
        """As Brain.step: n_ticks at these rates (Hz, one number or one per cell); bool[n_ticks, 2], which Giant
        Fiber spiked when. self.last_counts: every neuron's spikes during this call (LPLC2, LC4, DNp01 order)."""
        p = np.zeros(self.n_targets)
        p[: self.groups["LPLC2"]] = np.asarray(rates.get("LPLC2", 0.0)) * TICK_S
        p[self.groups["LPLC2"]:] = np.asarray(rates.get("LC4", 0.0)) * TICK_S
        return self.step_draws((self.rng.random((n_ticks, self.n_targets)) < p).astype(np.uint8))

    def step_draws(self, draws: np.ndarray) -> np.ndarray:
        c, v, g, rfc = self.c, self.v, self.g, self.rfc
        stim = np.zeros(self.n, dtype=bool)
        out = np.zeros((draws.shape[0], self.gf.size), dtype=bool)
        counts = np.zeros(self.n, dtype=np.int32)
        for j in range(draws.shape[0]):
            s = self.tick % core.DELAY_TICKS
            arrived = self.ring[s]
            contrib = self.w[arrived].sum(0) if arrived.any() else 0     # signed counts from spikes 1.8 ms ago
            rfc = np.maximum(rfc - 1, 0)
            free = rfc == 0
            v = np.where(free, c["v0_term"] + (g * c["couple_g"] + v * c["decay_v"]), v)
            g = np.where(free, g * c["decay_g"], g)
            sp = free & (v > c["v_th"])
            stim[: self.n_targets] = draws[j] != 0
            g = np.where(free, g + np.float32(contrib) * c["w_syn"] if np.isscalar(contrib)
                         else g + contrib.astype(np.float32) * c["w_syn"], g)
            v = np.where(free & stim, v + c["w_ext"], v)
            v = np.where(sp, c["v_0"], v).astype(np.float32)
            g = np.where(sp, np.float32(0.0), g).astype(np.float32)
            rfc = np.where(sp, self.rfc_reload, rfc)
            self.ring[s] = sp
            out[j] = sp[self.gf]
            counts += sp
            self.tick += 1
        self.v, self.g, self.rfc = v, g, rfc
        self.last_counts = counts
        return out


def use_in_player(idx: dict):
    """Make brain.player build FastPathway instead of the GPU brain (for the "wire" setup: identical results)."""
    from brain import player

    def fast_brain(pack, inputs, monitor, lesion=(), silence=(), seed=0, gain=1.0):
        assert gain == 1.0 and len(lesion) == 0
        return FastPathway(pack, idx, seed=seed)

    player.Brain = fast_brain                               # BrainPlayer builds its brain through this name


def replay_levels(noise: int, difficulty: str) -> int:
    """Play the 30 unseen levels of one noise draw and difficulty with FastPathway in place of the GPU brain, and
    compare with the saved GPU plays of the wire setup (results/noise_seeds/wire): wins and every takeoff distance."""
    import json
    from brain import player
    from brain.noise_seeds import chosen, out_dir

    pack, _ = player.load("real")
    idx = neurons.indices(pack)

    use_in_player(idx)
    s = chosen("wire")
    same = 0
    for block in ("100-109", "110-119", "120-129"):
        saved = json.loads((out_dir("wire") / f"{noise}_{difficulty}_{block}.json").read_text())
        for old in saved:
            new = player.play_level(pack, idx, s["r_ref"], s["steep"], old["seed"], difficulty, noise=noise)
            ok = new["won"] == old["won"] and new["jump_distances"] == old["jump_distances"]
            same += ok
            print(f"noise {noise} {difficulty} level {old['seed']}: {'WON ' if new['won'] else 'died'} "
                  f"{'identical' if ok else 'DIFFERENT from the GPU play'}", flush=True)
    print(f"noise {noise} {difficulty}: {same}/30 plays identical to the GPU model")
    return 0 if same == 30 else 1


def main() -> int:
    if sys.argv[1:2] == ["levels"]:
        return replay_levels(int(sys.argv[2]), sys.argv[3])
    from brain.player import load
    from brain.stepper import Brain

    pack, _ = load("real")
    idx = neurons.indices(pack)
    keep = np.concatenate([idx["LPLC2"], idx["LC4"], idx["DNp01"]])
    silence = np.setdiff1d(np.arange(pack.n_neurons), keep)
    # alarm rates that sweep quiet -> loud -> quiet, like spikes passing; 3 s at 4 seeds
    ramp = np.concatenate([np.zeros(2000), np.geomspace(1, 300, 8000), np.zeros(5000), np.geomspace(1, 300, 8000),
                           np.zeros(7000)])
    total_gpu = total_cpu = same = 0
    for seed in range(4):
        gpu = Brain(pack, {"LPLC2": idx["LPLC2"], "LC4": idx["LC4"]}, monitor=idx["DNp01"], silence=silence, seed=seed)
        cpu = FastPathway(pack, idx, seed=seed)
        rng = np.random.default_rng(1000 + seed)
        a, b = [], []
        for k in range(0, ramp.size, 167):                  # one 60 fps frame at a time, as in play
            r = float(ramp[k])
            draws = (rng.random((min(167, ramp.size - k), cpu.n_targets)) < r * TICK_S).astype(np.uint8)
            a.append(gpu.step_draws(draws))
            b.append(cpu.step_draws(draws))
        a, b = np.concatenate(a), np.concatenate(b)
        ta, tb = set(zip(*np.nonzero(a))), set(zip(*np.nonzero(b)))
        total_gpu, total_cpu, same = total_gpu + len(ta), total_cpu + len(tb), same + len(ta & tb)
        print(f"seed {seed}: Giant Fiber spikes GPU {len(ta)}, CPU {len(tb)}, same tick and cell {len(ta & tb)}")
    print(f"all: GPU {total_gpu}, CPU {total_cpu}, identical {same} "
          f"({same / max(total_gpu, 1):.1%} of GPU spikes)")

    cpu = FastPathway(pack, idx)
    for rate in (0, 20, 100, 300):
        t = time.perf_counter()
        cpu.step(1670, {"LPLC2": rate, "LC4": rate})
        us = (time.perf_counter() - t) / 1670 * 1e6
        print(f"speed at {rate:3d} Hz: {us:.0f} us per 0.1 ms tick ({us * 167 / 1000:.1f} ms per 60 fps frame)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
