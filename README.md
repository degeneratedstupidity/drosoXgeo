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

**Next:** Phase 6 (demo + post). The review's checks are done: scrambled-brain activity and targeted lesion (Phase 4c), sensory delay and slow frames (Phase 4d). 5B is future work (see Phase 4b).

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

## Phase 4b – How much of the brain does the work? (review follow-ups, 2026-10-05)
An outside review asked whether the whole brain plays, or just one wire. Every row below used
the same protocol: settings tuned on practice seeds 0–5, then the 90 unseen levels played with **5 draws of the
random input spikes** (450 plays), with a 95% interval (Wilson).

| setup | what's left of the brain | tuned settings | won | 95% interval |
|---|---|---|---|---|
| **full brain** | all 138,639 neurons | r_ref 5, steep 2 | 434/450 = **96.4%** | 94.3–97.8% |
| **direct wire** (`--setup wire`) | only LPLC2/LC4 → Giant Fiber; every other neuron silenced | r_ref 5, steep 1 | 448/450 = **99.6%** | 98.4–99.9% |
| **one-neuron stand-in** (`brain/standin.py`) | one LIF neuron, same physics, one tuned weight | r_ref 3, steep 1, 6 contacts | 449/450 = **99.8%** | 98.8–100% |
| one-number rule | – | θ > 1.14 rad | 90/90 (deterministic) | 95.9–100% |
| scrambled / Giant Fiber silenced | (Phase 4) | | 0/90 | |

- **The escape pathway alone plays better than the whole brain.** The direct wire and the one-neuron stand-in
  both beat the full brain, and the intervals don't overlap. The rest of the brain doesn't help at this task; it adds
  misfires. The full brain's 16 losses are spread over 14 levels (early or late jumps), the direct wire's 2 over 2.
- **Why the rest of the brain might add noise:** the Giant Fibers get 9,474 input synapses from 862 neurons. LPLC2
  supplies 11% and LC4 9%; the other 80% comes from elsewhere, and half of all the input is inhibitory.
- **Not just a lucky setting:** at the same setting (r_ref 5, steep 1), the full brain won 3/6 practice levels and
  the direct wire 6/6. Overall, 9 of 15 practice settings were perfect for the direct wire vs 7 of 15 for the full brain.
- **The stand-in reproduces the Giant Fiber's timing:** with 3 contacts per alarm spike (FlyWire averages 2.9 for
  LPLC2, 3.9 for LC4), its first spike comes 22.8 ms after onset at 10 Hz (real brain 22.7) and 4.5 ms at 200 Hz (4.3).
- **The original 89/90 was a slightly lucky draw:** the full brain's five draws gave 89, 87, 88, 84, 86. The
  brain-vs-one-number-rule gap is not significant; the stand-in had 3 tuning knobs, the brain and the rule 2 and 1.
- **What this means for the claim:** not "the whole fly brain plays Geometry Dash", but **"the fly's escape pathway,
  wired as in the FlyWire connectome, plays it untrained; scrambling the wiring breaks it."** At this task, the
  pathway is a good doorbell, and the rest of the brain adds a little static.
- **Learning center (5B check):** no MBON connects directly to the Giant Fiber (0 of 9,474 synapses). 23 MBONs reach
  it in 2 steps, but the first step is thin (best: MBON20 → DNp70, 6–9 synapses ≈ 2.5 mV per spike vs ~7 mV to
  threshold; DNp70 → Giant Fiber is strong, 248–339). Learning could at most nudge the jump, so 5B is future work.
- **Real fly's delay (for the delay sweep, #5):** stimulus to Giant Fiber response is about **19 ms** (Ache et al. 2019,
  seen in search excerpts; check in the paper). LC4 → Giant Fiber is ~1.3 ms (von Reyn et al. 2017). The model
  responds ~2 ms after the alarm cells, so it is missing ~17 ms of early visual processing, one game frame.

## Phase 4c – Is the scrambled brain just dead? And the targeted lesion (review #2, 2026-10-18)
**The shuffle keeps what it should** (checked on all 5 packs, `lif.shuffle_pack`): every neuron keeps its sign
(all outputs excitatory or all inhibitory, 0 mixed rows), its in- and out-degree, and its own outgoing synapse
counts. What changes is where strong input lands: in the real brain it piles onto a few cells (99.9th percentile of
excitatory input 7,694 synapses vs ~3,950 shuffled), and the Giant Fibers get ~2,400 excitatory synapses each vs ~1,000.

**Activity check** (`python -m brain.activity`, `figures/activity_check.json`): 300 random ordinary neurons (below the
90th percentile of in- and out-degree, not in the escape circuit) driven at 100 Hz for 1 s, the same input spikes in
every brain, 5 draws. Counted: the rest of the brain.

| brain | spikes | neurons active |
|---|---|---|
| real | 273,725 | 5,550 |
| scrambled, 5 seeds | 1,743–1,924 (**~150× fewer**) | 238–253 |
| scrambled 0, every synapse ×2.83 (`--match shuffled0`) | 275,102 (matched) | **29,873** |

- **Yes, a scrambled brain is mostly quiet,** on any input, not just the looming drive. The review's worry was right.
- **No single volume knob fixes it.** At ×2.83 the spike count matches, but it is spread over 5× as many neurons.
  Matching the number of active neurons instead (×2.38) leaves 7× fewer spikes. ×2 is still quiet, ×4 lights up
  78,000 neurons. The real wiring concentrates activity on particular cells; scrambled wiring can't, at any gain.

**Game rerun** (same protocol as Phase 4: tuned on practice seeds 0–5, then the 90 unseen levels, one draw):

| setup | what changed | tuned settings | easy | normal | hard | won (95% interval) | median takeoff, hard |
|---|---|---|---|---|---|---|---|
| real | – | r_ref 5, steep 2 | 30/30 | 30/30 | 29/30 | 89/90 | 0.41 blocks |
| **gain-matched scramble** (`--setup shuffled0g2.83`) | scramble 0, synapses ×2.83 | r_ref 2, steep 1 | 0/30 | 0/30 | 0/30 | **0/90** (0–4.1%) | never jumps |
| **targeted lesion** (`--setup cut`) | only the 293 LPLC2/LC4 → Giant Fiber edges (1,885 synapses) removed | r_ref 5, steep 3 | 22/30 | 23/30 | 18/30 | **63/90 = 70%** (60–79%) | 0.29 blocks |

- **The gain-matched scramble never jumped once** in 180 levels (6 practice levels × 15 settings, plus the 90 test
  levels). As busy as the real brain, and its Giant Fiber stays silent. So the scramble control is about wiring,
  not about a dead brain.
- **The direct synapses aren't the only way in.** With them cut, the alarm still reaches the Giant Fiber through
  2-step routes, and the brain plays 70% of levels. Practice even looked perfect (6/6 hard at steep 3).
  It jumps later (0.29 vs 0.41 blocks) and loses at random points along a level, not at the first spike.
- **The routes:** 211 neurons get input from the alarm cells and connect to the Giant Fiber. The main excitatory
  relays are PVLP151 (3,228 synapses in from LPLC2/LC4, 190 out to the Giant Fiber) and PVLP122b (1,332 in, 281 out);
  smaller ones are DNp103 and AMMC-A1. Others inhibit the Giant Fiber (LHAD1g1, PVLP024, PVLP010). These are wiring
  counts only; we haven't tested which relay the model actually uses.
- **Together with Phase 4b:** the direct synapses alone play best (99.6%), the full brain without them still plays
  (70%), and the Giant Fiber itself is required (0%). The circuit has a fast main road and slower side roads.

## Phase 4d – Timing: sensory delay and slow frames (review #5 and #6, 2026-10-18)
**Sensory delay** (`python -m brain.delay run`, then `summary`; 67 min, 4 jobs at a time;
`figures/delay_sweep_real.json`). In Fly Dash the alarm cells see the game instantly; a real fly takes ~19 ms from
light to the Giant Fiber, and the model covers only the last ~2 ms. `brain.player --delay-ms D` feeds the alarm cells
what the game showed D ms earlier. Each delay got the full protocol (alarm tuned on practice seeds 0–5, then the 30
unseen hard levels once), with the grid extended to r_ref 20 and 30 at every delay. At 0 ms it reproduces Phase 3
exactly (same practice scores, 29/30).

| added delay | brain, alarm retuned | 95% interval | median takeoff | brain, alarm not retuned (r_ref 5, steep 2) | one-number rule, same delay |
|---|---|---|---|---|---|
| 0 ms | r_ref 5, steep 2: **29/30** | 83–99% | 0.41 blocks | 29/30 | 30/30 (θ > 1.14) |
| **17 ms (real fly)** | r_ref 8, steep 2: **27/30** | 74–97% | 0.36 | 6/30 | 30/30 (θ > 0.98) |
| 33 ms | r_ref 20, steep 3: **30/30** | 89–100% | 0.31 | 0/30 | 30/30 (θ > 0.82) |
| 50 ms | r_ref 30, steep 2: **5/30** | 7–34% | 0.44 | 0/30 | 30/30 (θ > 0.74) |

![Win rate vs sensory delay](figures/delay_sweep_real.svg)

- **With the fly's real delay, the brain still plays,** once the alarm is tuned for it (27/30, interval overlaps 0 ms).
  A late eye needs an earlier alarm: r_ref climbs from 5 to 8 to 20. Without retuning, 17 ms alone drops it to 6/30.
- **It holds to 33 ms and breaks by 50 ms.** At 50 ms the best setting sits at the top of the grid (r_ref 30) and
  won 5/6 practice levels but only 5/30 unseen ones. 17 vs 33 ms (27 vs 30) is within noise (one draw each).
- **The one-number rule shrugs off every delay** by lowering its threshold. It sees the exact angle; the brain sees
  it through noisy spiking cells, so the brain's timing margin is what runs out.
- **For the real game:** screen capture and key events add lag on top of the fly's 17 ms. Up to ~33 ms total is
  fine if the alarm is retuned for it; ~50 ms is not. The lag has to be measured and the alarm tuned to it.

**Slow frames** (`python -m brain.frametime`, idle M1 Pro, 10 unseen hard levels, tuned settings, every frame timed:
one game frame = 166–167 ticks of simulation, synced back to the CPU; `figures/frametime_*.json`). Budget for live
play: 16.7 ms per frame.

| brain | frames | p50 | p95 | p99 | max | frames over 16.7 ms |
|---|---|---|---|---|---|---|
| **full brain**, all frames | 17,310 | 11.0 ms | 18.7 | 25.4 | 45.7 | 31% |
| full brain, alarm on | 7,819 | **17.1** | 19.9 | 29.6 | 45.7 | **64%** |
| full brain, takeoff frame + 2 before | 948 | 11.5 | 18.5 | 24.1 | 37.0 | 8% |
| direct wire, all frames | 17,310 | 10.9 | 18.2 | 21.3 | 24.0 | 7% |

- **The average hides it, as the review warned.** The median frame fits (11 ms), but while the alarm is on, the full
  brain needs more than a frame's time on 64% of frames, up to 46 ms. The busiest frames come just *after* takeoff,
  when the spike passes closest and the alarm peaks; the frames that decide the jump are mostly in budget (8% over).
- **~11 ms is the engine's floor, not the brain's work:** the direct wire, with all but 316 neurons silenced, also
  takes 10.9 ms. Most likely the cost of 167 ticks × 2 GPU kernel launches per frame (not profiled yet).
- **For Stretch B:** live play on the M1 Pro would fall behind exactly while obstacles are close. Options: the game
  slow-down mod (the honest fallback), batching several ticks per kernel launch, or a faster Mac. A frame running
  late adds lag, and per the delay sweep, the brain tolerates ~33 ms of retuned lag, not 50.

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

## Stretch B – The real Geometry Dash on macOS (2026-10-19: the escape pathway plays it live)
**Result: in the real game (Stereo Madness, normal mode), the fly's escape pathway clears the opening spikes on 7 of
9 attempts; with the Giant Fiber silenced or the wiring scrambled it never jumps and dies at the first spike
(0/22, 0/21).** One 60 s run each, same eye, same alarm settings, game in front the whole time:

| brain | Giant Fiber spikes | presses | attempts past the first spike | where attempts end |
|---|---|---|---|---|
| **real FlyWire wiring** (escape pathway) | 1,380 | 53 | **7/9** | 52–57 blocks (first block section) |
| Giant Fiber silenced | 0 | 0 | 0/22 | 16.4–16.8 blocks (first spike) |
| scrambled wiring (shuffled0) | 0 | 0 | 0/21 | 15.7–17.8 blocks (first spike) |

Stereo Madness is roughly 900–1,000 blocks, so ~55 blocks is ~6% of the level (Rajan's first try: 9%).

**How it's wired** (`python -m brain.live --press --r-ref 20 --steep 2 --countdown 5 --tap-test`):
- **Eye** (`brain/realeye.py`): finds the lime cube, keeps its "lane" (the rows it last stood in), and measures the
  gap to the first near-black/white column ahead in blocks (1 block = 1.135 lime widths, fitted so approaches run at
  10.4 blocks/s). Same number Fly Dash gives the brain. Checked on two recordings with annotated contact sheets.
- **Capture** (`FlyEye.app`, `brain/flyeye.swift`): ScreenCaptureKit stream of the game window only, 440 px wide,
  ~4 ms from capture to Python. **Keys** (`FlyJump.app`, `brain/flyjump.swift`): space down/up, sent only to the
  game's process. Each is its own app with its own permission (Screen Recording, Accessibility), launched through
  `open` so the IDE needs neither (`brain/flyapps.py`). Wine only takes keys while the game window is the active one.
- **Brain** (`brain/fastpath.py`): the escape pathway (LPLC2 + LC4 + DNp01, the Phase 4b "wire" setup) on the CPU,
  identical to the GPU model spike for spike (1,043/1,043 Giant Fiber spikes) and play for play (360/360 saved
  Phase 4b plays). 16–26 µs per 0.1 ms tick, so ~4× faster than real time. **The full brain can't play live:** with
  the game running it needs 134–150 µs per tick while the alarm is on, falls behind and decides 100+ ms late.
- **Lag:** capture → decision 8 ms (median; p95 11 ms). Press → first frame showing the cube rising: 34 ms
  (16 taps: 15–66 ms). So the fly acts on a picture ~40 ms old.
- **Alarm retuned for that lag** (`python -m brain.delay --setup fastwire --delays 0,37,45,50 run`, then chosen on
  practice only as the gentlest setting perfect at 37, 45 and 50 ms): r_ref 20, steep 2 wins 30/30 unseen hard
  levels in Fly Dash at each of those delays (takeoff 0.41, 0.31, 0.27 blocks). The delay-0 setting (r_ref 5,
  steep 1) wins 0/30 at any of them, and in the real game it died at the first spike in 22 of 23 attempts.

**What went wrong on the way (and is logged now):**
- A run where the presses took ~100 ms to show (vs 34 ms) died at the first spike 20 of 21 times. Most likely the
  game lost focus; runs now log the front app 4×/s and start with a tap test (does the cube jump?).
- Counting deaths from "cube gone" glued attempts together (explosion pieces look like the cube; the respawned cube
  blinks). Attempts are now counted from the respawn blink.
- Watch-only replays of two human recordings (`brain/realwatch.py`): real wiring fires before 14/14 and 13/15
  obstacles (clean spikes at 0.28–0.47 blocks; the human jumps at 0.6–1.5), silenced and scrambled 0.

### The block sections: where a reflex stops (2026-10-19)
**The escape pathway clears spikes in the real game but not Stereo Madness's first block section, and neither does
the best possible one-number looming rule. Jumping there depends on where you will land, which an escape reflex
doesn't consider.** The section: two spikes, a 1-block step right after them, then towers rising out of a thorn pit.

- **Recorded the fly's own deaths** (`brain.live --record`): with the spike setting it cleared the double spikes but
  flew over the step and fell into the pit (3 of 4 deaths; the 4th hit the double spikes late).
- **Taught the eye** (`brain/realeye.py`): in the air it looks along the rows the cube is in now (before, a block it
  was flying over stayed at distance 0); standing on something raised, the end of its top above black thorns is an
  obstacle (checked on both recordings: the edges of platforms and towers over pits). Distance from the cube's true
  front (centre + half a block), so a rotating cube doesn't read closer.
- **Taught the clone** (`flydash.FlyDashTerrain`, `--difficulty terrain`): landable blocks, thorn pits, death on
  block sides, levels built only from Stereo Madness's patterns and each checked beatable by a search over every jump
  timing. Its observation copies the new eye. The spike-only levels are unchanged (30/30 saved plays identical; the
  `python3 flydash.py` baselines unchanged). The real jump matches the clone (measured on 7 jumps: 0.443 s in the air,
  2.2 blocks high; clone 0.45 s, 2.1).
- **No single takeoff distance fits** (from the jump arc, 33 ms late): a single spike can be cleared from ~0.2 to ~2.4
  blocks away, three spikes only from 0.17 to 0.45, one spike before a step needs > 1.0 (or you fly over the step),
  a pit between platforms needs a jump right at the edge. A looming reflex has one threshold.
- **Tuned on terrain practice levels** (`python -m brain.delay --setup fastwire --difficulty terrain --delays 37,45,50
  --r-refs 2,3,5,8,12,20,30,50,80 run`; best mean progress over the three delays: r_ref 30, steep 1). 30 unseen
  terrain levels at 45 ms:

| player | won | mean progress |
|---|---|---|
| escape pathway, terrain setting (r_ref 30, steep 1) | 0/30 | **18%** |
| best one-number looming rule (θ > 0.42) | 0/30 | 20% |
| escape pathway, spike setting (r_ref 5, steep 1) | 0/30 | 5% |
| Giant Fiber silenced / scrambled wiring (shuffled0) | 0/30 | 5% (no jumps) |

- **Real game with the terrain setting** (30 s, normal mode): 4/4 attempts past the opening spikes, reaching 52–57
  blocks as before; but now it lands on the step every time and reaches the first tower in 2 of 4. It dies jumping
  off too early: the alarm is already up when it lands, so the held key bounces it straight off and it comes down
  short of the next tower, where a player would wait for the edge.
- **What it means for the claim:** the fly's escape pathway plays the part of Geometry Dash that is an escape
  problem (something looms, get away), with the controls failing as they should. The block sections are a planning
  problem. Learning in the mushroom body (5B) or other visual inputs could in principle add context; future work.

**Other limits:**
- One 60 s (or 30 s) run per brain, one noise seed, one level. More runs before any percentage goes in a post.
- The real delay varies (press → rise 15–66 ms); the clone tunes for a fixed one.
- The game is the Windows build in Wine (Sikarugir); never modified.
