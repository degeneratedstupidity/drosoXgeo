"""The frame-by-frame stepper must match the validated engine spike for spike."""

import numpy as np
import pytest

from lif import core, engine_fused
from brain.paths import PACK_V783
from brain.stepper import Brain

SUGAR = [720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
         720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367]
MN9 = 720575940660219265


@pytest.fixture(scope="module")
def pack():
    if not (PACK_V783 / "manifest.json").is_file():
        pytest.skip("build the pack first: python -m brain.build_pack")
    return core.load_pack(PACK_V783)


@pytest.mark.parametrize("silence", [False, True])
def test_matches_engine(pack, silence):
    index = {int(n): i for i, n in enumerate(pack.neuron_ids)}
    targets = np.array([index[i] for i in SUGAR], dtype=np.int32)
    stim = core.make_stimulus_for(pack, targets, 200.0, n_ticks=1500, seed=5)
    mask = np.zeros(pack.n_neurons, dtype=bool)
    if silence:
        mask[index[SUGAR[0]]] = True
    ref = engine_fused.run(pack, stim, silenced=mask if silence else None, chunk=32, edge_split=1, warmup=0)

    brain = Brain(pack, {"sugar": targets}, monitor=[index[MN9]],
                  silence=np.flatnonzero(mask))
    draws = np.asarray(stim.draws).astype(np.uint8)
    seen = np.concatenate([brain.step_draws(draws[:700]), brain.step_draws(draws[700:])])
    assert np.array_equal(brain.spike_counts(), ref.spike_counts)
    assert seen.sum() == ref.spike_counts[index[MN9]]
    assert ref.spike_counts.sum() > 500           # the drive really did something


def test_lesion_never_spikes(pack):
    index = {int(n): i for i, n in enumerate(pack.neuron_ids)}
    targets = np.array([index[i] for i in SUGAR], dtype=np.int32)
    brain = Brain(pack, {"sugar": targets}, monitor=[index[MN9]], lesion=[index[MN9]])
    seen = brain.step(3000, {"sugar": 200.0})
    assert seen.sum() == 0
