# drosoXgeo — a fruit-fly brain plays Geometry Dash

**Core idea:** Geometry Dash is a one-button *jump* game. Fruit flies already have a jump circuit:
looming-detector neurons (**LPLC2**, **LC4**) see something growing fast on the retina, drive the
**Giant Fiber** neuron (**DNp01**), and the fly jumps away. So the plan is to wire an approaching
spike to the fly's own escape circuit, run it on the real connectome, and see whether that circuit
can time jumps well enough to clear levels.

## Status
- [x] **Phase 1 – Fly Dash game** (`flydash.py`): headless GD-style runner, lock-step, looming signals
- [ ] Phase 0 – Orientation reading
- [ ] Phase 2 – Run the whole-brain model, reproduce a published result
- [ ] Phase 3 – Wire the escape reflex (no training)
- [ ] Phase 4 – Controls (the part that makes it science, not a meme)
- [ ] Phase 5 – Learning
- [ ] Phase 6 – Demo video + LinkedIn post

Baselines from `python3 flydash.py` (30 levels each):

| policy | easy | normal | hard |
|---|---|---|---|
| random button mashing | 7% progress | 5% | 6% |
| one-number looming threshold | 100% won | 100% won | 100% won |

The fly brain has to beat that bar or explain why it doesn't. On hard (triple spikes) the jump must
fire when the spike is 0.34–0.8 blocks away, a window of about **45 ms**. That is a real timing
test for a spiking network.

---

## Phase 0 – Orientation (a few evenings)
- Shiu et al., *Nature* 2024, "A Drosophila computational brain model reveals sensorimotor processing".
  It is the whole-brain leaky integrate-and-fire (LIF) model everyone builds on.
- Browse FlyWire Codex (codex.flywire.ai). Search the cell types `LPLC2`, `LC4` and `DNp01` and see how they connect.
- Learn the LIF neuron: membrane voltage leaks, inputs add, crossing threshold gives a spike, then a reset.
- Read about Giant Fiber escape (von Reyn et al., 2014 and 2017; Ache et al., 2019 on LPLC2).

## Phase 1 – Game ✅
Why a clone and not the real game: the brain simulation doesn't run in real time, so the game has to
**wait for the brain** each tick. You also get ground truth for plots and thousands of fast runs.

## Phase 2 – Run the fly brain
- **Simulator for the M1 Pro:** `Kisame76/drosophila-brain-mlx` (Apple Silicon / Metal). No Colab needed.
  The original `philshiu/Drosophila_brain_model` (Brian2, CPU) is the reference to check results against.
- The data is already downloaded in `data/`; see `data/SOURCES.md`.
- **Validation:** reproduce one result from the paper (sugar-sensing neurons → feeding motor neuron MN9).
  If you can't reproduce it, nothing later can be trusted.

## Phase 3 – Wire the reflex (no training)
1. **Encode.** Map the looming signal (`theta`, `theta_dot`) to Poisson firing rates on LPLC2/LC4 neurons.
2. **Simulate.** Step the brain 1/60 s per game tick.
3. **Decode.** Count DNp01 (Giant Fiber) spikes in a short window; above a threshold, press jump.
4. **First experiment:** does DNp01 fire at all when LPLC2 is driven in the model? How long is the delay?
   The answer is the first plot of the project.

## Phase 4 – Controls
- **Shuffled connectome:** same neurons and same connection counts, but rewired at random.
  If it plays just as well, the fly's wiring wasn't doing the work.
- **Silence DNp01:** the brain should lose the ability to jump.
- **Compare against the one-number baseline** above.

## Phase 5 – "Training" (be precise about what learns)
The connectome's synapse weights are fixed. Learning can live in only two places:
- **A. Trained readout.** Treat the brain as a fixed "reservoir" and learn only the output mapping
  (e.g. logistic regression from neuron rates to jump). Run the shuffled control again.
- **B. Dopamine-gated plasticity.** Let a small set of mushroom-body synapses (Kenyon cell → MBON) change,
  with reward or punishment delivered through dopamine neurons (PPL1). Flies actually learn this way.

**Not recommended:** PPO or backprop on all 50M connectome weights.
- Spikes aren't differentiable.
- PPO is built for differentiable policy networks.
- Once every weight has been retrained, it is no longer the fly's wiring, just a spiking network that started from a fly-shaped layout.

## Phase 6 – Demo
- Side-by-side video: game | spike raster of LPLC2 → DNp01 | jump presses.
- Post the controls; most fly-brain game demos don't have them.
- Credit FlyWire / Shiu et al. Connectome data is CC BY-NC 4.0: fine for a personal post, not for commercial use.

## Stretch A – Full vision (the fly actually sees the screen)
- Instead of handing the brain a ready-made looming number, map downscaled game frames onto the fly's
  eye map: photoreceptor and lamina columns in the optic lobe, each with a position in the visual field.
- The fly's own motion detectors (T4/T5) and looming detectors (LPLC2) then have to work out
  "something is coming" by themselves. It's harder, and much more impressive if it works.
- First check: does a looming square on screen drive LPLC2 in the model at all?

## Stretch B – The real Geometry Dash on macOS
- The Steam version runs natively on macOS, so no wrapper is needed.
- Read the game state:
  - screen capture (`mss`) plus OpenCV, or
  - a Geode mod that exports the player's and obstacles' positions (more precise)
- Press jump with `pynput` or Quartz key events.
- macOS needs Screen Recording and Accessibility permissions for both.
- The real game won't wait for the brain. The brain has to keep up at 60 fps and hit a 45 ms jump window,
  so measure the simulation speed on the M1 Pro first.
