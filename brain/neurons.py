"""The neurons of the escape circuit, looked up by cell type in our v783 annotations."""

import csv

import numpy as np

from lif import core

from brain.paths import DATA

# The Giant Fiber, one per side of the brain (v783 root IDs).
DNP01 = (720575940632499757, 720575940622838154)


def ids_of_type(cell_type: str) -> list[int]:
    with open(DATA / "neuron_annotations.tsv") as f:
        return [int(r["root_id"]) for r in csv.DictReader(f, delimiter="\t") if r["cell_type"] == cell_type]


def indices(pack: core.Pack) -> dict[str, np.ndarray]:
    """Model indices of LPLC2, LC4 and DNp01. Refuses anything but a v783 pack."""
    if not pack.manifest["dataset"].startswith("flywire-v783"):
        raise ValueError(f"these are v783 root IDs; the pack is {pack.manifest['dataset']}")
    index = {int(n): i for i, n in enumerate(pack.neuron_ids)}
    out = {}
    for name, ids in (("LPLC2", ids_of_type("LPLC2")), ("LC4", ids_of_type("LC4")), ("DNp01", DNP01)):
        missing = [i for i in ids if i not in index]
        if missing:
            raise ValueError(f"{len(missing)} {name} neurons are not in the pack: {missing[:5]}")
        out[name] = np.array(sorted(index[i] for i in ids), dtype=np.int32)
    return out


def check_same_neurons(a: core.Pack, b: core.Pack) -> None:
    """A shuffled pack must keep the neuron list, so the same indices mean the same cells."""
    if not np.array_equal(a.neuron_ids, b.neuron_ids):
        raise ValueError("packs list different neurons")
