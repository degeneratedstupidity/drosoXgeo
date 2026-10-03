"""Compile FlyWire v783 (our data/) into an MLX pack, labelled as v783.

lif.compile_pack only knows the v630 defaults and stamps every pack "flywire-v630".
Our files have the same schema (they come from the same upstream repo), so this
reuses its loader, its ~40 integrity checks and its writer, and only changes the
inputs and the dataset label. A v783 pack must never be mistaken for v630: the
root IDs differ between versions.

    python -m brain.build_pack
"""

import sys

from lif import compile_pack as cp

from brain.paths import DATA, PACK_V783

COMPLETENESS = DATA / "Completeness_783.csv"
CONNECTIVITY = DATA / "Connectivity_783.parquet"


def main() -> int:
    log = cp.CheckLog()
    src_hashes = {
        "completeness_csv": {"path": COMPLETENESS.name, "sha256": cp.sha256_file(COMPLETENESS),
                             "bytes": COMPLETENESS.stat().st_size},
        "connectivity_parquet": {"path": CONNECTIVITY.name, "sha256": cp.sha256_file(CONNECTIVITY),
                                 "bytes": CONNECTIVITY.stat().st_size},
    }
    ids = cp.load_neurons(COMPLETENESS, log)
    pre_raw, post_raw, w_raw, ids_pre, ids_post = cp.load_edges(CONNECTIVITY, log)
    pre, post, counts = cp.check_edges(pre_raw, post_raw, w_raw, ids.size, log)
    cp.check_id_mapping(ids, pre, post, ids_pre, ids_post, log)
    row_ptr, dst, cnt, stats = cp.build_csr(pre, post, counts, ids.size, log)
    print(log.render())
    if log.failed:
        print(f"\n>>> {log.failed} CHECK(S) FAILED - nothing written.")
        return 1

    semantics = dict(cp.FLYWIRE_SEMANTICS)
    semantics["index_mapping"] = "model index i == row i of Completeness_783.csv, in file order"
    PACK_V783.parent.mkdir(parents=True, exist_ok=True)
    manifest = cp.write_pack(
        PACK_V783,
        {"neuron_ids": ids, "row_ptr": row_ptr, "destinations": dst, "signed_counts": cnt},
        stats, src_hashes, log, dataset="flywire-v783", semantics=semantics,
    )
    print(f"\n>>> pack written: {manifest['neurons']} neurons, {manifest['edges']} edges -> {PACK_V783}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
