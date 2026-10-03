"""Run the connectome model a little at a time, with input that changes as you go.

lif.engine_fused.run needs the whole input planned before it starts. A game can't
work that way: what the brain sees next depends on what it did last frame. This
keeps the brain's state between calls and reuses the engine's two Metal kernels
(propagate spikes along the wiring, then update every neuron), so the physics is
exactly the validated model's.

Plain-language map of the state, per neuron:
  v    membrane voltage (the water level in the bucket)
  g    synaptic input still arriving (water on its way into the bucket)
  rfc  ticks left resting after a spike (refractory period)
  ring spikes in flight, delivered 1.8 ms later (axonal delay)

One tick is 0.1 ms. One game frame (1/60 s) is 166.7 ticks.

Two ways to switch a neuron off, which mean different things:
  silence  its synapses deliver nothing (Shiu et al.'s method). It still spikes.
  lesion   it can never spike at all: held permanently refractory, and silenced.
           This is the control for the readout neuron, since a silenced DNp01 would
           still "press the button".
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from lif import core
from lif.engine_fused import _state_kernel
from lif.engine_metal import propagate, silenced_row_end

TICK_S = core.DT / 1000.0          # 0.1 ms
NEVER = 2**30                      # refractory ticks for a lesioned neuron (~30 hours)


class Brain:
    def __init__(self, pack: core.Pack, inputs: dict[str, np.ndarray], monitor: np.ndarray,
                 silence: np.ndarray = (), lesion: np.ndarray = (), seed: int = 0,
                 edge_split: int = 1):
        """inputs: group name -> neuron indices that receive Poisson input.
        monitor: neuron indices whose every spike is reported, tick by tick."""
        self.pack, self.edge_split = pack, edge_split
        N = pack.n_neurons
        self.groups = {k: np.asarray(v, dtype=np.int32) for k, v in inputs.items()}
        targets = np.concatenate(list(self.groups.values()))
        if np.unique(targets).size != targets.size:
            raise ValueError("a neuron is in two input groups")
        self.n_targets = targets.size
        slot = np.full(N, -1, dtype=np.int32)
        slot[targets] = np.arange(targets.size, dtype=np.int32)
        self.target_slot = mx.array(slot)

        mask = np.zeros(N, dtype=bool)
        mask[np.asarray(silence, dtype=np.int64)] = True
        mask[np.asarray(lesion, dtype=np.int64)] = True
        self.row_end = silenced_row_end(pack, core.silenced_mask(pack, mask))

        # As upstream: Poisson targets are never refractory. Lesioned neurons always are.
        reload = np.full(N, core.RFC_TICKS, dtype=np.int32)
        reload[targets] = 0
        lesion = np.asarray(lesion, dtype=np.int64)
        reload[lesion] = NEVER
        rfc0 = np.zeros(N, dtype=np.int32)
        rfc0[lesion] = NEVER
        self.rfc_reload = mx.array(reload)

        cf = core.constants_f32()
        self.k = [mx.array([float(cf[n])], dtype=mx.float32)
                  for n in ("v0_term", "couple_g", "decay_v", "decay_g", "v_th", "w_syn", "w_ext", "v_0")]
        self.n_arr = mx.array([N], dtype=mx.uint32)
        self.monitor = np.asarray(monitor, dtype=np.int32)
        self.monitor_mx = mx.array(self.monitor)
        self.rng = np.random.default_rng(seed)

        self.v = mx.full((N,), core.V_0, dtype=mx.float32)
        self.g = mx.zeros((N,), dtype=mx.float32)
        self.rfc = mx.array(rfc0)
        self.counts = mx.zeros((N,), dtype=mx.int32)
        self.ring = [mx.zeros((N,), dtype=mx.uint8) for _ in range(core.DELAY_TICKS)]
        self.tick = 0
        mx.eval(self.v, self.g, self.rfc, self.counts, *self.ring)

    def _probabilities(self, rates: dict[str, float | np.ndarray]) -> np.ndarray:
        p = np.zeros(self.n_targets)
        i = 0
        for name, idx in self.groups.items():
            r = rates.get(name, 0.0)
            p[i:i + idx.size] = np.broadcast_to(np.asarray(r, dtype=np.float64), idx.size) * TICK_S
            i += idx.size
        if (p < 0).any() or (p > 1).any():
            raise ValueError("input rates must be between 0 and 10,000 Hz")
        return p

    def step(self, n_ticks: int, rates: dict[str, float | np.ndarray]) -> np.ndarray:
        """Advance n_ticks with these input rates (Hz, per group: one number or one per neuron).
        Returns bool[n_ticks, len(monitor)]: which monitored neurons spiked on which tick."""
        p = self._probabilities(rates)
        return self.step_draws((self.rng.random((n_ticks, self.n_targets)) < p).astype(np.uint8))

    def step_draws(self, draws: np.ndarray) -> np.ndarray:
        """Advance one tick per row of draws (uint8[T, n_targets], 1 = an input spike
        arrives at that target on that tick, columns in the order of `inputs`)."""
        n_ticks = draws.shape[0]
        draws = mx.array(np.asarray(draws, dtype=np.uint8))
        v, g, rfc, counts = self.v, self.g, self.rfc, self.counts
        seen = []
        N = self.pack.n_neurons
        for j in range(n_ticks):
            s = self.tick % core.DELAY_TICKS
            contrib = propagate(self.ring[s], self.pack, self.n_arr, self.edge_split, row_end=self.row_end)
            v, g, rfc, counts, spike = _state_kernel(
                inputs=[v, g, rfc, counts, contrib, draws[j], self.target_slot, self.rfc_reload,
                        self.n_arr, *self.k],
                output_shapes=[(N,)] * 5,
                output_dtypes=[mx.float32, mx.float32, mx.int32, mx.int32, mx.uint8],
                grid=(N, 1, 1), threadgroup=(256, 1, 1))
            self.ring[s] = spike
            seen.append(spike[self.monitor_mx])
            self.tick += 1
            if (j + 1) % 32 == 0:              # keep each GPU batch a sensible size
                mx.eval(v, g, rfc, counts, *self.ring, *seen)
        out = mx.stack(seen) if seen else mx.zeros((0, self.monitor.size), dtype=mx.uint8)
        mx.eval(v, g, rfc, counts, *self.ring, out)
        self.v, self.g, self.rfc, self.counts = v, g, rfc, counts
        return np.asarray(out).astype(bool)

    def spike_counts(self) -> np.ndarray:
        """Total spikes per neuron since the start."""
        return np.asarray(self.counts)
