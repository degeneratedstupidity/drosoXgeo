# AGENTS.md — drosoXgeo handoff

Read this, then `README.md` (roadmap + every result) and `data/SOURCES.md` (data, citations) before changing anything.

## Goal
A connectome-constrained LIF model of the fruit-fly brain (FlyWire v783, Shiu et al. 2024 model) plays a
Geometry Dash clone through its own escape circuit: looming detectors **LPLC2**/**LC4** drive the **Giant Fiber
(DNp01)**, and any DNp01 spike presses jump. Write for developers **new to computational neuroscience**: plain
language, analogies first.

## Where things stand (2026-11-01)
Phases 1–4 and 5A are done (results in `README.md`). Headline: real wiring wins 89/90 unseen levels untrained;
Giant Fiber silenced and 5 scrambled brains win 0/90; trained readouts don't beat the Giant Fiber.
An external review's checks are done: README "Phase 4b" (escape pathway alone beats the full brain, 99.6% vs 96.4%;
5B → future work), "Phase 4c" (scrambled brains ~150× quieter; gain-matched scramble still 0/90; cutting only
LPLC2/LC4→GF leaves 63/90 via 2-step relays), "Phase 4d" (with +17 ms delay and a retuned alarm 27/30 hard, fine to
33 ms, breaks at 50 ms; full brain is over the 16.7 ms frame budget on 64% of alarm-on frames, ~11 ms engine floor).
Headline wording: "the fly's escape pathway", not "the whole brain".
**Stretch B works** (README "Stretch B"): the escape pathway (`brain/fastpath.py`, CPU, identical to the GPU "wire"
setup) plays the real game live: 7/9 attempts past Stereo Madness's opening spikes, silenced and scrambled 0/22 and
0/21. The block section after that is a planning problem a looming reflex can't solve (README "The block sections":
eye and clone now know blocks/pits; best one-number rule and the fly both ~20% of terrain levels). Stretch C (learning
to time jumps) stopped at its stage-2 gate: receptive-field vision input works, but downstream of the looming cells
the real wiring carries no more "what's coming" than a gain-matched scramble (README "Stretch C").
**Stretch D (stopped 2026-11-01):** the aim was to clear Stereo Madness up to the ship (block 266.5)
with a *learned decision layer* on the looming cells, labelled honestly (README "Stretch D"). Real level from the
game's file (read-only), floating blocks in Fly Dash, cross-entropy learning with checkpoints (best from the start:
7/12 sim attempts reach ~145, `figures/learn_real_gen25.json`), a shadow of the level kept in step with the real
cube (`brain/shadow.py`), live play with the layer (real game: one attempt past the towers, others died at 33–48),
and a dopamine-like reward rule (`brain/dopamine.py`; sim: past the towers 8% → ~50%). Done since: overnight
real-game dopamine learning (`brain.live --dopamine`), the real game's jump-on-landing rule in Fly Dash, ~1 s tags,
the shadow sync fix (real game now matches Fly Dash; best 161). **2026-11-01:** practice mode (`--practice`) taught
the second half nothing (0% in all 4 late sections, 0/80 runs to the ship): there the eye would need "where does
the ground end", which looming cells don't signal (README "Practice mode"). Then two layout cues
(`--ground`, `--headroom`; named in each weight file as `"cues"`): set by hand they clear both ceilings (157–191
70%, 240→ship 100%), but no setting or search clears the two staircases (114–157, 191–240) and the towers at once:
0/20 complete runs (README "Headroom cue and the last round", "Where Stretch D ends"). Afternoon: two more real-game
runs (looming only, dopamine on) and Fly Dash slab practice (`brain/slab_practice.py`): **real-game best 173.9**;
practice made the first half more reliable but slab 1 (153) didn't transfer (README "Back in the real game").
Wording: "looming pathway + learned decision layer: block 173.9 of 266 in the real game". No more training for now.
**Phase 6 (2026-11-01 evening): post assets ready.** Illustration clip (Blender: cartoon fruit fly at
an RGB rig; the screen is a real recorded run, presses synced; `docs/build_clip.py`, caption `docs/build_caption.py`)
→ `results/clip/fly_gaming_captioned.mp4` (1080×1080, 15.9 s) and a 16:9 `fly_gaming_wide_captioned.mp4`; walls chart
`docs/build_walls_chart.py` → `results/clip/walls.png`; story video (illustration, the same run from inside, the
scrambled control) `docs/build_story.py` → `results/clip/fly_story.mp4`. Game footage and the chart's level
silhouette are RobTop's content: they stay in results/ (gitignored), never in the repository.

## How to run
- Simulator: sibling checkout `../drosophila-brain-mlx/`; its `.venv/bin/python` runs our modules from this repo
  root (`-m brain.xxx`). Packs in `data/pack/` (gitignored): `v783`, `v783-shuffled-seed0..4`.
- `brain/player.py` plays (setups `real`, `lesion`, `cut`, `wire`, `shuffledN`, `shuffledNgG` = synapses ×G; `--watch` draws it live). `brain/tune.py` tunes on
  practice seeds 0–5, scores once on seeds 100–129 × easy/normal/hard. `brain/phase4.py` builds the controls table.
  `brain/readout.py` is Phase 5A (resumable, 4 jobs at a time). Review checks: `brain/activity.py` (4c),
  `brain/delay.py` (4d, resumable), `brain/frametime.py` (4d; run on an idle Mac only). `brain/checks.py` reruns the 10 sanity checks.
- Pages in `docs/` (rebuild with `docs/build_*.py`). Check them with headless Chrome:
  `"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --screenshot=out.png "file://$PWD/docs/<page>.html"`
- `flydash.py`: dependency-free clone, lock-step. Speed 10.4 blocks/s, 60 fps, hard-level jump window ~45 ms.
- IDs (v783): DNp01 `720575940632499757`, `720575940622838154`; LPLC2 210 cells, LC4 104 (`brain/neurons.py`).
- M1 Pro: 3–4 processes at once ≈ 2× throughput. Background tasks are cut off at 2 h: keep runs shorter or resumable,
  or start them detached (`nohup caffeinate -i ... &; disown`). Stop runs by PID and its children (`pgrep -P`), never
  by a broad name match: that once killed a learning run's workers and hung it.
- Stretch C/D modules: `brain/receptive.py` (receptive fields, needs `codex/`, gitignored, 16 GB of Codex downloads),
  `brain/vision.py` (per-cell input), `brain/context.py` (stage 2), `brain/stereo.py` (the real level),
  `brain/learn.py` (decision layer, cross-entropy, checkpoints), `brain/shadow.py`, `brain/dopamine.py`;
  `brain.live --layer figures/learn_real_gen25.json` plays the real game with the layer. Commands in README "Stretch D".

## Open decisions (ask the owner)
- 5B (mushroom-body learning): future work; the MBON→DNp01 path check showed only weak 2-step routes (README Phase 4b),
  and the looming signal never reaches the Kenyon cells (Stretch C).
- Stretch D is stopped. Ideas left (none started): other visual projection neurons instead of layout
  cues (FlyEM optic lobe; flyvis, to be checked), the jump via the nerve cord's jump motor neuron (FlyEM male CNS;
  means rebuilding the model), a non-linear layer, learning from a perfect player first (changes the wording).
- Real game (Stretch B): built 2026-10-18/19. Run `python -m brain.live --press --r-ref 20 --steep 2 --countdown 5
  --tap-test` (normal mode; click into the game during the countdown, then hands off). Capture, keys and recording
  live in ~/Applications/FlyEye.app (Screen Recording), FlyJump.app (Accessibility) and FlyRec.app (Screen
  Recording), built from brain/flyeye.swift, flyjump.swift, flyrec.swift; rebuilding changes their ad-hoc signature,
  so macOS asks for the permission again. Capture only the game window, never the screen. Long runs: the Mac must
  stay untouched (another app in front pauses it; every attempt then dies at the first spike), and add
  `caffeinate -d` or the screen locks. Keys only reach the game while it's the front app. The real-game parts need
  your own copy of the game, launched as `geometrydash.app` from the repo root (gitignored). Never modify the game's
  own files; no mods. Real time is borderline (README Phase 4d).

## Conventions
- Commit each tested piece.
- Long runs: log to `results/`, wrap in `caffeinate -i`, give a `tail -f` command and say whether anything would
  disturb the run (reading logs: safe; a second simulation: slower, same results).

## Rules (don't break these)
- **Validate before building on the simulation.** Reproduce a published result first.
- **Always include controls:** degree-preserving shuffled connectome (`lif.shuffle_pack`), DNp01 silenced, and the
  one-number baseline. If a control plays as well as the real brain, report it.
- **No PPO/backprop on connectome weights.** Learning only in a readout (brain fixed) or in a small set of
  mushroom-body synapses (KC→MBON, dopamine via PPL1).
- **Precise claims:** "connectome-constrained LIF model", not "uploaded fly brain". Readouts must exclude the
  neurons we drive (Phase 5A lesson).
- **Cite** Dorkenwald 2024, Schlegel 2024, Shiu 2024 (`data/SOURCES.md`). Data is CC BY-NC 4.0: no commercial use.
- Keep `flydash.py` dependency-free. Large outputs go in `results/` (gitignored).
