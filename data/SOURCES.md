# Where these files came from (downloaded 2026-10-02)

| File | What it is | Source |
|---|---|---|
| `Connectivity_783.parquet` (100 MB) | Who-connects-to-whom: presynaptic ID, postsynaptic ID, synapse count, excitatory/inhibitory sign. FlyWire FAFB v783, the adult female brain. | https://github.com/philshiu/Drosophila_brain_model |
| `Completeness_783.csv` | The 138,639 neurons the simulation uses | same repo |
| `neuron_annotations.tsv` (32 MB) | Cell type of every neuron (e.g. LPLC2, LC4, DNp01), neurotransmitter, side | https://github.com/flyconnectome/flywire_annotations |
| `../shiu_model/` | The leaky integrate-and-fire simulation code (`model.py`, `utils.py`, example notebook) | https://github.com/philshiu/Drosophila_brain_model |

Escape-circuit neurons in this data: 210 LPLC2, 104 LC4, 2 DNp01 / Giant Fiber
(720575940632499757, 720575940622838154). All of them are in the simulation.

## FlyWire Codex downloads (2026-10-19, `codex/`, gitignored, 16 GB)
Downloaded by Rajan from https://codex.flywire.ai/api/download (FAFB v783, Google sign-in). Used so far:
- `column_assignment.csv.gz`: 45,528 columnar optic-lobe neurons placed on each eye's hexagonal column grid
  (`brain/receptive.py` derives LPLC2/LC4 receptive fields from it). From the FlyWire visual system paper:
  Matsliah et al. 2024, *Nature*, "Neuronal parts list and wiring diagram for a visual system" (cite when used).
- `synapse_coordinates.csv.gz`: synapse positions (nm); a neuron's ID is only on its first row. Used to measure
  which way the column grid's axes point in the brain.
- Also there, unused so far: connection tables (princeton/buhmann, thresholded or not), the full synapse table
  (2.5 GB), cell types, classifications, labels, neuropil synapse counts, and skeletons (`sk_lod1_783_healed.zip`,
  13 GB). Same license as the rest of FlyWire: CC BY-NC 4.0.

## The game's own level file (read-only)
`brain/stereo.py` reads Stereo Madness from the installed game (`Resources/levels/1.txt` in the Wine prefix under
`geometrydash.app`, gitignored): base64 + gzip, objects as "key,value" pairs (1 = type, 2/3 = x/y, 30 units per
block). Only read, never changed; the decoded copy is `results/stereo_madness_level.txt` (gitignored). It is RobTop
Games' content: don't commit or publish it. Fly Dash's version of the level is built from it on the fly.

## Cite these when you post
- Dorkenwald et al. 2024, *Nature*: "Neuronal wiring diagram of an adult brain" (FlyWire connectome)
- Schlegel et al. 2024, *Nature*: "Whole-brain annotation and multi-connectome cell typing of Drosophila" (cell types)
- Shiu et al. 2024, *Nature*: "A Drosophila computational brain model reveals sensorimotor processing" (the simulation)
- License: FlyWire data is CC BY-NC 4.0. Credit is required and commercial use is not allowed.

## Other official downloads (not needed yet)
- FlyWire Codex download portal (Google sign-in): https://codex.flywire.ai/api/download
  - FAFB v783 (this brain)
  - BANC v888 (brain + nerve cord)
- Google/HHMI male whole-CNS connectome (Sept 2026, ~166k neurons): check the Google Research blog post "A Connectomics Milestone" for the official download link (not verified here)
