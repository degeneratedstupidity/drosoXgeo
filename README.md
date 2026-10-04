# drosoXgeo — a fruit-fly brain plays Geometry Dash

**Core idea:** Geometry Dash is a one-button *jump* game. Fruit flies already have a jump circuit:
looming-detector neurons (**LPLC2**, **LC4**) see something growing fast on the retina, drive the
**Giant Fiber** neuron (**DNp01**), and the fly jumps away. So the plan is to wire an approaching
spike to the fly's own escape circuit, run it on the real connectome, and see whether that circuit
can time jumps well enough to clear levels.

## Status
- [x] **Phase 1 – Fly Dash game** (`flydash.py`): headless GD-style runner, lock-step, looming signals
- [ ] Phase 0 – Orientation reading
- [x] **Phase 2 – Run the whole-brain model, reproduce a published result** (`brain/`, see results below)
- [x] **Phase 3 – Wire the escape reflex (no training)** (`brain/player.py`, results below)
- [x] **Phase 4 – Controls (the part that makes it science, not a meme)** (`brain/phase4.py`, results below)
- [ ] Phase 5 – Learning (**5A done**, `brain/readout.py`; 5B not started)
- [ ] Phase 6 – Demo video + LinkedIn post

**Next:** decide between 5B (mushroom-body learning) and Phase 6 (demo); see `AGENTS.md`.

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

## Phase 2 – Run the fly brain ✅
- **Simulator for the M1 Pro:** `Kisame76/drosophila-brain-mlx` (Apple Silicon / Metal). No Colab needed.
  The original `philshiu/Drosophila_brain_model` (Brian2, CPU) is the reference to check results against.
- The data is already downloaded in `data/`; see `data/SOURCES.md`.
- **Validation:** reproduce one result from the paper (sugar-sensing neurons → feeding motor neuron MN9).
  If you can't reproduce it, nothing later can be trusted.

**Setup** (the simulator is a sibling checkout, `../drosophila-brain-mlx`, and its venv runs our code):
```
cd ../drosophila-brain-mlx
uv venv --python 3.13 && uv pip install -e '.[dev,reference]'
./tools/fetch_upstream.sh && python -m lif.compile_pack        # v630 pack, for the reference check
cd ../drosoXgeo
PY=../drosophila-brain-mlx/.venv/bin/python
$PY -m brain.build_pack                                          # data/pack/v783, labelled flywire-v783
for s in 0 1 2 3 4; do $PY -m lif.shuffle_pack --pack data/pack/v783 --seed $s; done
$PY -m brain.validate_sugar                                      # figures/phase2_sugar_mn9.{svg,json}
```

**Result 1: the MLX engine matches the published Brian2 model (v630).** `python -m lif.validate_notebook`
reran Shiu's example notebook and compared it with the published result files: 15 comparisons, all within |z| < 2.
MN9 at 100 Hz sugar input: 67.0 Hz published vs 67.3 / 67.7 Hz here (two seeds).

**Result 2: it holds on our v783 connectome, and the wiring is what produces it.** 30 trials × 1 s, same input spikes everywhere:

| run | sugar GRNs | MN9 | spikes/trial | neurons active |
|---|---|---|---|---|
| v630 (published setup) | 21 | 67.3 Hz | 9,699 | 408 |
| v630, same 20 GRNs as v783 | 20 | 61.9 Hz | 9,062 | 403 |
| **v783** | 20 | **63.0 Hz** | 9,091 | 402 |
| v783 degree-shuffled, 5 seeds | 20 | **0.0 Hz** (all 5) | ~2,330 | ~100 |

- **Version bridge:** a root ID names one exact segmentation, so an ID present in both v630 and v783 is the same, unedited neuron.
  20 of the 21 sugar GRNs and MN9 keep their IDs. `720575940620900446` was proofread after v630 and is dropped. No mapping by cell type was needed.
- **v630 → v783 on the same 20 GRNs:** 61.9 → 63.0 Hz, within trial noise.
- **Shuffled control:** the sugar GRNs still fire at 99.5 Hz, but activity dies out after ~100 neurons and MN9 never spikes.

![MN9 on real vs shuffled v783](figures/phase2_sugar_mn9.svg)

**Speed on the M1 Pro** (fused lane, the fastest of 3 runs; `figures/bench_m1pro_*.json`):

| pack, drive | s per biological s | ms per tick | vs real time |
|---|---|---|---|
| v630, 21 sugar GRNs @ 150 Hz (sparse; M4 Pro gets 0.29) | **0.45** | 0.045 | 2.2× faster |
| v783, 100 top hubs @ 150 Hz (dense, 2,241 neurons active) | **1.20** | 0.120 | 0.83×, too slow |

The real game's verdict depends on how much of the brain the looming drive wakes up:
- A sparse drive keeps up with 60 fps.
- A hub-like drive doesn't.

Phase 3 measures this on the actual LPLC2/LC4 drive.

## Phase 3 – Wire the reflex (no training)
1. **Encode.** Map the looming signal (`theta`, `theta_dot`) to Poisson firing rates on LPLC2/LC4 neurons.
2. **Simulate.** Step the brain 1/60 s per game tick.
3. **Decode.** Count DNp01 (Giant Fiber) spikes in a short window; above a threshold, press jump.
4. **First experiment:** does DNp01 fire at all when LPLC2 is driven in the model? How long is the delay?
   The answer is the first plot of the project.

**Progress (2026-10-04):** all four steps run. The player lives in `brain/player.py` (`--watch` shows it live in the terminal);
`brain/tune.py` picks the two alarm-shape knobs on practice levels.
- `brain/stepper.py` advances the brain one frame at a time with input that changes as it goes. It matches the
  validated engine spike for spike (`tests/test_stepper.py`).
- **Experiment 1** (`brain/exp_loom.py`): drive LPLC2/LC4 at 0–200 Hz for 300 ms, 10 trials each.
  - DNp01 fired in every trial once the drive was on.
  - Its first spike comes 4.3 ms after drive onset at 200 Hz and 22.7 ms at 10 Hz.
  - LPLC2 alone drives it almost as well as both together; LC4 alone needs a stronger drive.
  - DNp01 lesioned: 0 spikes.
  - 5 degree-shuffled brains: 0 spikes, except shuffle 4 at 200 Hz (0.8 Hz, first spike after ~216 ms; real wiring: 201 Hz).
- **Checklist:** `python -m brain.checks` reruns all 10 checks (published, control, scrambled, consistency) in ~90 s
  and prints expected vs got with PASS/FAIL. All 10 pass.
- **Pages:** `docs/explainer.html` (plain-language intro) and `docs/testbench.html` (checklist, test grid, spike replays).
  Rebuild the test bench page with `python3 docs/build_testbench.py`.
- **Brain replay** (`docs/brainreplay.html`): watch one hard level frame by frame with all 138,639 neurons drawn
  from the front, lighting up as they fire. `python -m brain.record_level --difficulty hard --seed 1` records it
  (real wiring and a scrambled one), `python3 docs/build_brainreplay.py` builds the page.
  On hard level 1 the real brain wins with 11 jumps and 652 different neurons fire at some point. The scrambled brain's
  alarm cells fire just the same, but the signal goes nowhere: 0 jumps, crash at 12%.

**Result: the escape circuit plays Fly Dash (2026-10-04).** `python -m brain.tune --setup real` (54 min on the M1 Pro)
picks the two alarm settings on 6 practice levels, then plays 30 unseen levels per difficulty with them, once.
- **Chosen on practice:** r_ref 5 Hz, steep 2.
- **Unseen levels:** easy 30/30 won, normal 30/30, hard 29/30 (mean progress 99%). Results: `figures/phase3_play_real.json`.
- **Timing:** the brain takes off a median 0.41 blocks from the spike (10th–90th percentile 0.27–0.55), on every
  difficulty. On hard that's inside the ~45 ms window.
- **The one loss** (hard level 117, at 74%): on a triple spike the Giant Fiber fired a single early spike while the alarm
  was still weak (1.4 Hz), 0.94 blocks out, just outside the window, and the fly landed on the third spike. The input
  spikes are random (Poisson), so the jump cell sometimes fires early. That's noise in the model, and it isn't hidden.
- **Does the wiring matter?** Yes; see Phase 4.

## Phase 4 – Controls
- **Shuffled connectome:** same neurons and same connection counts, but rewired at random.
  If it plays just as well, the fly's wiring wasn't doing the work.
- **Silence DNp01:** the brain should lose the ability to jump.
- **Compare against the one-number baseline** above.

**Result (2026-10-04): the wiring does the work.** Every row used the same protocol: settings picked on the
6 practice levels, then one pass over the same 30 unseen levels per difficulty (seeds 100–129, 300 blocks).
The six control brains took 17 min in total (`results/run_phase4.sh`). `python -m brain.phase4` builds the table
(`figures/phase4_controls.json`).

| setup | easy | normal | hard | median takeoff, hard |
|---|---|---|---|---|
| **real FlyWire wiring** | **30/30** | **30/30** | **29/30** | 0.41 blocks |
| Giant Fiber silenced | 0/30 | 0/30 | 0/30 | never jumps |
| scrambled wiring, 5 seeds | 0/30 each | 0/30 each | 0/30 each | never jumps |
| random button mashing (5% of frames) | 0/30 | 0/30 | 0/30 | – |
| one-number rule (θ > 1.14 rad) | 30/30 | 30/30 | 30/30 | 0.69 blocks |

- **Scrambled and silenced brains never jumped once,** in any of 1,080 levels (90 practice runs over 15 alarm
  settings plus 90 test levels, for each of the 6 control brains). Their alarm cells fire as usual; the signal doesn't
  reach the Giant Fiber. They die at the first spike (5% progress).
- **The real brain matches the one-number rule but doesn't beat it** (89/90 vs 90/90). That rule is the idealised
  reflex the circuit implements, so it's the ceiling. The brain jumps later (0.41 vs 0.69 blocks), nearer the edge of
  the safe window, which is how its one noisy early spike cost a level.
- **Caveat:** the alarm search tops out near 160 Hz (r_ref 12, steep 3). In Experiment 1, scrambled brain 4 gave the
  Giant Fiber a weak 0.8 Hz at a constant 200 Hz drive, but only after ~216 ms, far too slow to clear a spike.
  So "never jumps" holds for the alarm range the real brain was tuned in.

## Phase 5 – "Training" (be precise about what learns)
The connectome's synapse weights are fixed. Learning can live in only two places:
- **A. Trained readout.** Treat the brain as a fixed "reservoir" and learn only the output mapping
  (e.g. logistic regression from neuron rates to jump). Run the shuffled control again.
- **B. Dopamine-gated plasticity.** Let a small set of mushroom-body synapses (Kenyon cell → MBON) change,
  with reward or punishment delivered through dopamine neurons (PPL1). Flies actually learn this way.

**Result 5A (2026-10-04): a trained readout doesn't beat the fly's own jump neuron, and a readout that can see
the input fools the controls.** `python -m brain.readout run --setups real,lesion,shuffled0,...,shuffled4` then
`python -m brain.readout summary` (`figures/phase5a_readout.json`). About 3.5 h on the M1 Pro, 4 jobs at a time.
- **Method.** The brain stays fixed. It watches 30 practice levels (seeds 0–9, all difficulties) while a teacher plays.
  A ridge regression learns the spike's distance from the neurons' spike counts (this frame and the last), with the
  strength picked by cross-validation across practice levels. On the 90 unseen levels it jumps when its estimate
  drops below 0.6 blocks (the middle of the safe window). Every brain trains its own readers the same way, with the same eye.
- **Reader A** reads every neuron that fires, including the LPLC2/LC4 alarm cells we drive. **Reader B** reads only
  neurons downstream of them.

| brain | untrained Giant Fiber (Phase 3–4) | Reader A, all neurons | Reader B, downstream only |
|---|---|---|---|
| real FlyWire wiring | **89/90** | 80/90 | 34/90 |
| Giant Fiber silenced | 0/90 | 80/90 | 30/90 |
| scrambled wiring, 5 seeds | 0/90 each | **90/90 each** | **0/90 each** |

- **The fly's Giant Fiber beats every reader we trained** (89 vs 80 on the real brain). The readers aim well (median
  takeoff 0.57 blocks, right where we pointed them, vs 0.41 for the Giant Fiber), but about 7% of their jumps come far
  too late (< 0.3 blocks): better aim, worse reliability. They estimate distance from one noisy 1/60 s frame; the
  Giant Fiber effectively adds up the alarm over time.
- **Reader A plays perfectly on every scrambled brain.** It decodes the alarm cells, whose firing *is* our input
  signal, so it measures the input, not the wiring. A likely reason the scrambled brains do even better (not
  tested): without the real wiring, nothing feeds extra activity back into the alarm cells. **Lesson: a readout that
  can see the driven neurons is not a test of the connectome.**
- **Reader B is the fair test, and it supports the wiring claim.** Downstream of the alarm cells, the real brain
  carries enough to win 34/90. The scrambled brains carry nothing: their Reader B training error (0.78 blocks) equals
  always guessing the average (0.79), and they win 0/90.
- **With a reader, the Giant Fiber isn't needed** (silenced brain: 80/90 and 30/90). The looming information reaches
  other downstream neurons as well.
- **Takeaway for the demo:** the honest showcase stays the untrained escape circuit. Learning a readout added nothing
  to performance, and its controls show why readouts need care.

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
