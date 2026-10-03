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
- Rajan will run the real game himself at the very end. Don't work on running or wrapping the game binary
  until Rajan says so (see "Open decisions").
- Google Chrome is installed. Headless Chrome screenshots work for checking pages locally:
  `"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --screenshot=out.png --window-size=1200,1400 "file://$PWD/docs/<page>.html"`

## Current state (2026-10-04: Phases 1–4 done, paused before Phase 5)
| Path | What it is |
|---|---|
| `flydash.py` | ✅ Dependency-free, headless Geometry Dash clone, lock-step (waits for the brain). `python3 flydash.py` prints baselines. |
| `README.md` | Phased roadmap (Phases 0–6 plus Stretch A and B) with every result so far. Keep the Status checklist up to date. |
| `data/` | FlyWire FAFB **v783**: `Connectivity_783.parquet` (wiring), `Completeness_783.csv` (138,639 neurons), `neuron_annotations.tsv` (cell types, soma positions), `SOURCES.md` (citations, CC BY-NC 4.0). |
| `shiu_model/` | Reference Brian2 code from Shiu et al. 2024. |
| `brain/build_pack.py`, `validate_sugar.py` | ✅ Phase 2: v783 MLX pack (`data/pack/v783`, gitignored; shuffles `v783-shuffled-seed0..4`) and the sugar→MN9 validation. |
| `brain/stepper.py` | Steps the brain one game frame at a time with changing input; matches the validated engine spike for spike (`tests/test_stepper.py`). |
| `brain/neurons.py` | LPLC2 / LC4 / DNp01 indices in a pack. |
| `brain/exp_loom.py`, `exp_loom_examples.py`, `checks.py` | ✅ Phase 3 Experiment 1 (alarm drive → Giant Fiber latency) and the 10-check checklist (`python -m brain.checks`, all pass). |
| `brain/player.py` | ✅ The brain plays Fly Dash. Setups `real`, `lesion`, `shuffledN`. `--watch` draws it live in the terminal. |
| `brain/tune.py` | Picks the two alarm knobs (r_ref, steep) on practice seeds 0–5, then scores once on unseen seeds 100–129 × 3 difficulties. Prints every level. |
| `brain/phase4.py` | ✅ Phase 4 table: real vs lesion vs 5 shuffles vs random vs one-number rule → `figures/phase4_controls.json`. |
| `brain/record_level.py` | Records one level (real + scrambled), every neuron's firing per frame → `results/replay/*.json`. |
| `docs/` | Pages: `explainer.html` (plain intro), `testbench.html` (Phase 3 checks; `build_testbench.py`), `brainreplay.html` (whole brain lighting up as it plays; `build_brainreplay.py`, link option `#frame=80&setup=shuffled0`). |
| `figures/` | Committed results JSON/SVG for Phases 2–4. |
| `results/` | Gitignored logs and big outputs. `run_phase4.sh` reruns all control brains. |
| `../drosophila-brain-mlx/` | The simulator, a sibling checkout. **Its `.venv/bin/python` runs our `brain` modules** (`-m brain.xxx` from this repo root). |

**Fly Dash facts**
- Speed 10.4 blocks/s, 60 fps (`DT = 1/60`), jump apex 2.1 blocks, airtime 0.45 s.
- `env.step(jump: bool)` returns `(obs, reward, done)`.
- `obs` has `retina` (a 4×24 grid), `distance`, `theta` (angular size of the next spike), `theta_dot`, `tau`, `y`, `airborne`, `progress`.
- Difficulties: easy, normal, hard. Hard includes triple spikes; the jump must fire 0.34–0.8 blocks out (~45 ms).

**Neurons of interest (v783 IDs, all present in Completeness_783)**
- LPLC2: 210 neurons · LC4: 104 neurons
- DNp01 (Giant Fiber): `720575940632499757`, `720575940622838154`

## Results so far (details and tables in `README.md`)
- **Phase 2:** the MLX engine matches the published Brian2 results (15/15 comparisons). v783 sugar→MN9 63.0 Hz;
  all 5 shuffles 0 Hz. Version gotcha: vendored code hardcodes v630 sugar IDs; use our scripts on v783.
- **Phase 3:** driving LPLC2/LC4 fires DNp01 4–23 ms later; lesion and shuffles don't. The player (alarm rate =
  r_ref·(θ̇/θ̇₀)^steep, jump on any DNp01 spike, no jump threshold in code) chose r_ref 5 Hz, steep 2 and won
  **89/90 unseen levels** (hard 29/30). Median takeoff 0.41 blocks. The one loss: an early noisy DNp01 spike on a triple spike.
- **Phase 4:** same protocol for every control. Lesion and all 5 shuffles: **0/90 each, zero jumps in 1,080 levels**.
  Random mashing 0/90. One-number rule 90/90 (takeoff 0.69 blocks). Headline: the real wiring matches the
  idealised reflex with nothing hand-coded, and scrambling it destroys it. It does **not** beat the one-number rule.
- **Speed:** game play ran ~1.2× faster than real time on the M1 Pro (54 min compute for ~65 min of play).

## Next task: Phase 5A (agreed plan, not started)
- **Trained readout:** the brain stays fixed; learn a small mapping (e.g. logistic regression) from the activity of the
  neurons the alarm wakes up to "jump now", trained on practice seeds only, scored on seeds 100–129.
- Goal: better jump timing (closer to the middle of the safe window) than "any DNp01 spike".
- **Run the shuffled control with the same readout.** Expected risk: shuffled brains' alarm cells still fire normally,
  so a readout may make them play well too. If so, report it honestly. Train only on non-alarm neurons, or report both,
  so the result says what the wiring adds.
- Estimate given to Rajan: ~1–2 h of work + 1–4 h of background compute on the M1 Pro.

## Open decisions (ask Rajan)
- After 5A: do **5B** (dopamine-gated KC→MBON plasticity; several sessions, uncertain, since the mushroom body
  doesn't wire directly to DNp01) or go to **Phase 6** (video + post) and the real game.
- **Real game (Stretch B):** Rajan wants it at the very end. Told Rajan it's feasible: key presses (Quartz, Accessibility
  permission) are certain, and spike detection from screen capture is very likely. Real time is borderline (~1.2×),
  but a game slow-down / frame-step mod is the guaranteed fallback, disclosed honestly. Fair demo: a spike-only level
  (editor-built) or the opening of an official level, since the brain only has the looming→jump skill.
  Still to decide: whether Claude builds the connector or Rajan does. A "real-time mode" for the clone (no waiting,
  added screen latency) was suggested as the feasibility test first.

## Working with Rajan
- Explain in plain language, analogies first (Rajan is new to neuroscience).
- Commit each tested piece and push to `origin` (private GitHub `degeneratedstupidity/drosoXgeo`).
  **No `Co-Authored-By` or "Generated with" lines**: Rajan must be the only author. This repo's git email is Rajan's
  GitHub noreply address, so commits link to Rajan's account.
- For every long run: write output to a log in `results/`, give Rajan a `tail -f` command to follow it, and say whether
  a command would disturb the run (reading logs: safe; a second simulation: slows both, same results).
  Wrap long runs in `caffeinate -i` so the Mac doesn't sleep.
- Don't publish artifacts or send anything to outside services without asking (privacy).

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
