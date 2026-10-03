# AGENTS.md — drosoXgeo handoff

You're picking up a project already in progress. Read this file, then `README.md` (the roadmap) and
`data/SOURCES.md` (where the data came from) before changing anything.

## Goal
A simulated fruit-fly brain, built from the real FlyWire connectome, plays Geometry Dash.
It's for a LinkedIn post and for learning, and may also feed a term paper on simulating the fly
from its connectome. The owner, Rajan, is an experienced software developer (Python, TypeScript,
mobile, backend) but **new to computational neuroscience**. Explain neuroscience concepts briefly
as you use them.

**The angle:** Geometry Dash is a one-button jump game. Flies have a dedicated jump circuit:
looming detectors **LPLC2** and **LC4** drive the **Giant Fiber (DNp01)**, which triggers the escape takeoff.
We feed approaching spikes into that circuit and read DNp01 out as the jump button.
Other "fly brain plays a game" demos (Doom, Mario 64, Game Boy, all from Sept 2026) mostly map pixels
into arbitrary neurons. What sets this project apart is the specific circuit plus honest controls.

## Machine
- MacBook Pro, **M1 Pro**, Apple Silicon. Prefer Metal/MLX tools over CUDA. Colab isn't needed.
- Rajan will run the real game himself at the very end. Don't work on running or wrapping the game binary.

## Current state
| Path | What it is |
|---|---|
| `flydash.py` | ✅ Done. Dependency-free, headless Geometry Dash clone, lock-step (waits for the brain). `python3 flydash.py` prints baselines. |
| `README.md` | Phased roadmap (Phases 0–6 plus Stretch A and B). Keep the Status checklist up to date. |
| `data/Connectivity_783.parquet` | FlyWire FAFB **v783** wiring table: pre ID, post ID, synapse count, sign. 100 MB. |
| `data/Completeness_783.csv` | The 138,639 neurons the simulation uses. |
| `data/neuron_annotations.tsv` | Cell types (column `cell_type`), keyed by v783 `root_id`. |
| `data/SOURCES.md` | Sources, citations, license (CC BY-NC 4.0). |
| `shiu_model/` | Reference Brian2 code from Shiu et al. 2024 (`model.py`, `utils.py`, `example.ipynb`). |
| `brain/` | ✅ Phase 2. `build_pack.py` compiles our v783 data into an MLX pack labelled `flywire-v783` (`data/pack/v783`, gitignored). `validate_sugar.py` runs sugar→MN9 on v630, v783 and 5 shuffles. `paths.py` holds the locations. |
| `figures/` | Committed results: `phase2_sugar_mn9.{svg,json}`, `bench_m1pro_*.json`. |
| `../drosophila-brain-mlx/` | The simulator, a sibling checkout. **Its `.venv/bin/python` runs our `brain` modules** (`-m brain.xxx` from this repo root). Packs: v630 in its `data/pack/`. |

**Fly Dash facts**
- Speed 10.4 blocks/s, 60 fps (`DT = 1/60`), jump apex 2.1 blocks, airtime 0.45 s.
- `env.step(jump: bool)` returns `(obs, reward, done)`.
- `obs` has `retina` (a 4×24 grid), `distance`, `theta` (angular size of the next spike), `theta_dot`, `tau`, `y`, `airborne`, `progress`.
- Difficulties: easy, normal, hard. Hard includes triple spikes.

**Baselines (30 levels each)**
- Random button mashing: ~6% progress.
- Jump when `theta > ~1.1–1.9 rad`: wins 100% of levels on every difficulty.
- On hard, the jump must fire with the spike 0.34–0.8 blocks away, a window of about **45 ms**.

**Neurons of interest (v783 IDs, all present in Completeness_783)**
- LPLC2: 210 neurons
- LC4: 104 neurons
- DNp01 (Giant Fiber): `720575940632499757`, `720575940622838154`

## Phase 2: done (2026-10-03). Results are in `README.md`.
- The MLX engine matches the published Brian2 results on v630 (`lif.validate_notebook`: 15/15 comparisons, |z| < 2).
- **v783:** sugar→MN9 at 63.0 Hz (v630 on the same inputs: 61.9 Hz). All 5 degree-shuffled packs: MN9 at 0 Hz.
- **Speed on the M1 Pro:**
  - 0.45 s per biological s with a sparse drive (2.2× faster than real time)
  - 1.20 s with a dense hub drive (too slow for real time)
- **Version gotcha, solved:** our pack is genuine v783, built by `brain.build_pack`. The vendored code
  hardcodes v630 sugar IDs in `lif.stimulus_flybrain` and `lif.control_demo`, and the benchmark's `sugar`
  stimulus also assumes them; one of those IDs is missing on v783, so use our scripts or `--stimulus hubs` there.
  A v630 ID exists in v783 only if that neuron is unedited, so it's the same neuron. Never mix versions otherwise.

## Next task: Phase 3
- Write `brain_player.py`.
- Encode `theta`/`theta_dot` as Poisson rates on LPLC2 and LC4.
- Step the brain 1/60 s per Fly Dash tick.
- Jump when DNp01's spike count in a window crosses a threshold.
- First experiment: does DNp01 fire when LPLC2 is driven in the model, and with what latency?

## Rules for this project (don't break these)
- **Validate before building on the simulation.** Reproduce a published result first.
- **Always include the controls:**
  - a degree-preserving **shuffled connectome** (the MLX repo has `lif.shuffle_pack`)
  - **DNp01 silenced**
  - the **one-number baseline** above

  If the shuffled brain plays as well as the real one, report that honestly.
- **Don't run PPO or backprop on all connectome weights.** Spikes aren't differentiable, and retraining
  every weight erases the claim that this is the fly's brain. Learning is allowed in exactly two places:
  - a trained **readout** (the brain stays fixed)
  - reward-modulated plasticity on a **small** set of mushroom-body synapses (KC→MBON, dopamine via PPL1)
- **Keep claims precise** in any write-up: say "connectome-constrained LIF model", not "uploaded fly brain".
- **Cite** Dorkenwald 2024, Schlegel 2024 and Shiu 2024 (full citations in `data/SOURCES.md`).
  The data is CC BY-NC 4.0, so no commercial use.
- Keep `flydash.py` dependency-free. Put heavy dependencies in the brain modules.
- Large outputs (spike recordings can be several GB) go in `results/` and stay out of git.
