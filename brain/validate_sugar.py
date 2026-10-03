"""Phase 2 validation: sugar GRNs -> MN9 on our FlyWire v783 pack, with the shuffle control.

Shiu et al. 2024 drive 21 sugar-sensing gustatory receptor neurons (GRNs) of one
labellum with Poisson input and find that MN9, the motor neuron behind proboscis
extension (the feeding reflex), fires. That is the published result we reproduce
before trusting the model with anything else.

Version bridge: the 21 GRNs are v630 root IDs. A root ID names one exact
segmentation, so an ID present in v783 is the same, unedited neuron. 20 of the 21
survive (720575940620900446 was proofread after v630); MN9 keeps its ID too. No
cell-type guessing is involved. To separate "different version" from "one GRN
fewer", v630 is also run with only the 20 survivors.

Every run uses the same seed, so the same Poisson input spikes. Runs:
  v630 / 21 GRNs     the published setup (reference)
  v630 / 20 GRNs     same, minus the GRN missing from v783
  v783 / 20 GRNs     our connectome
  v783 shuffled x5   degree-preserving rewiring: same in/out degrees, random partners

    python -m brain.validate_sugar
"""

import argparse
import json
import sys

import numpy as np
import pyarrow.parquet as pq

from lif import core, experiment
from lif.control_demo import Panel, figure, rate_over_time
from lif.stimulus_flybrain import RIGHT_SUGAR_GRN_IDS

from brain.paths import FIGURES, PACK_V630, PACK_V783, RESULTS, shuffled

MN9 = 720575940660219265
SUGAR_V630 = [int(i) for i in RIGHT_SUGAR_GRN_IDS]


def run(label, pack, sugar, rate, seed, bin_s):
    params = dict(experiment.default_params, r_poi=rate)
    path = experiment.run_exp(f"sugarR_{rate:g}Hz", sugar, RESULTS / "validate_sugar" / label,
                              params=params, force_overwrite=True, pack=pack, seed=seed)
    spikes = pq.read_table(path, columns=["t", "flywire_id"])
    ids, t = spikes["flywire_id"].to_numpy(), spikes["t"].to_numpy()
    n_run, t_run = params["n_run"], params["t_run"]
    edges, mn9 = rate_over_time(t[ids == MN9], n_run, t_run, bin_s)
    _, grn = rate_over_time(t[np.isin(ids, sugar)], n_run, t_run, bin_s, n_neurons=len(sugar))
    row = {
        "label": label,
        "dataset": pack.manifest["dataset"],
        "n_sugar": len(sugar),
        "mn9_hz": float(mn9.mean()),
        "sugar_hz": float(grn.mean()),
        "spikes_per_trial": ids.size / n_run,
        "neurons_spiked": int(np.unique(ids).size),
    }
    print(f"{label:<18} {row['dataset']:<36} GRNs {len(sugar):>2}  MN9 {row['mn9_hz']:6.2f} Hz  "
          f"sugar {row['sugar_hz']:6.2f} Hz  {row['spikes_per_trial']:>8.0f} spikes/trial  "
          f"{row['neurons_spiked']:>6} neurons")
    return row, edges, mn9, grn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rate", type=float, default=100.0, help="Poisson rate on each sugar GRN, Hz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shuffles", type=int, default=5)
    ap.add_argument("--bin-ms", type=float, default=20.0)
    args = ap.parse_args()
    bin_s = args.bin_ms / 1000

    v630, v783 = core.load_pack(PACK_V630), core.load_pack(PACK_V783)
    if v783.manifest["dataset"] != "flywire-v783":
        raise SystemExit(f"{PACK_V783} is {v783.manifest['dataset']}, expected flywire-v783")
    in_v783 = set(v783.neuron_ids.tolist())
    sugar_v783 = [i for i in SUGAR_V630 if i in in_v783]
    if MN9 not in in_v783:
        raise SystemExit("MN9 root ID not in v783")
    print(f"{len(sugar_v783)} of {len(SUGAR_V630)} sugar GRNs keep their root ID in v783; "
          f"missing: {sorted(set(SUGAR_V630) - in_v783)}\n")

    runs = [("v630_21grn", v630, SUGAR_V630), ("v630_20grn", v630, sugar_v783),
            ("v783", v783, sugar_v783)]
    for s in range(args.shuffles):
        pack = core.load_pack(shuffled(PACK_V783, s))
        if pack.manifest.get("derived_from", {}).get("arrays") != {
                k: v["sha256"] for k, v in v783.manifest["arrays"].items()}:
            raise SystemExit(f"{shuffled(PACK_V783, s)} is not a shuffle of {PACK_V783}")
        runs.append((f"v783_shuffled{s}", pack, sugar_v783))

    rows, panels = [], {}
    for label, pack, sugar in runs:
        row, edges, mn9, grn = run(label, pack, sugar, args.rate, args.seed, bin_s)
        rows.append(row)
        panels[label] = (row, edges, mn9, grn)

    FIGURES.mkdir(exist_ok=True)
    (FIGURES / "phase2_sugar_mn9.json").write_text(json.dumps(
        {"rate_hz": args.rate, "seed": args.seed, "n_run": experiment.default_params["n_run"],
         "t_run_s": experiment.default_params["t_run"], "mn9": MN9, "sugar_grns": sugar_v783,
         "runs": rows}, indent=2) + "\n")

    shown = [("v783", "Real wiring: FlyWire v783"),
             ("v783_shuffled0", "Shuffled wiring, same degrees (seed 0)")]
    mn9_row, grn_row = [], []
    for label, title in shown:
        row, edges, mn9, grn = panels[label]
        mn9_row.append(Panel(title, f"MN9 {row['mn9_hz']:.1f} Hz; brain: "
                                    f"{row['spikes_per_trial']:,.0f} spikes/trial", edges, mn9,
                             note="" if row["mn9_hz"] else "no MN9 spike in 30 trials"))
        grn_row.append(Panel("", f"the {row['n_sugar']} sugar GRNs: {row['sugar_hz']:.1f} Hz each",
                             edges, grn))
    shuf = [r["mn9_hz"] for r in rows if "shuffled" in r["label"]]
    caption = (f"30 trials of 1 s per pack, identical input spikes (seed {args.seed}). "
               f"v630 reference MN9: {rows[0]['mn9_hz']:.1f} Hz.\n"
               f"All {len(shuf)} shuffles: MN9 {', '.join(f'{x:.1f}' for x in shuf)} Hz. "
               "Bars: 20 ms bins.\nRegenerate with python -m brain.validate_sugar.")
    out = FIGURES / "phase2_sugar_mn9.svg"
    out.write_text(figure([mn9_row, grn_row], ["MN9 rate (Hz)", "sugar GRN rate (Hz)"],
                          f"MN9 while {len(sugar_v783)} sugar GRNs receive {args.rate:g} Hz input",
                          caption))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
