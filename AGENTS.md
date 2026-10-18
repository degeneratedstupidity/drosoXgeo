# AGENTS.md — drosoXgeo handoff

Read this, then `README.md` (roadmap + every result) and `data/SOURCES.md` (data, citations) before changing anything.

## Goal
A connectome-constrained LIF model of the fruit-fly brain (FlyWire v783, Shiu et al. 2024 model) plays a
Geometry Dash clone through its own escape circuit: looming detectors **LPLC2**/**LC4** drive the **Giant Fiber
(DNp01)**, and any DNp01 spike presses jump. For a LinkedIn post, learning, and maybe a term paper. Owner: Rajan,
an experienced developer **new to computational neuroscience**: explain in plain language, analogies first.

## Where things stand (2026-10-18)
Phases 1–4 and 5A are done (results in `README.md`). Headline: real wiring wins 89/90 unseen levels untrained;
Giant Fiber silenced and 5 scrambled brains win 0/90; trained readouts don't beat the Giant Fiber.
**Now:** working through `SUGGESTIONS.md` (an external review; local only, Rajan keeps it out of git) before Phase 6. Done: #11, #3, #1, #7 (README
"Phase 4b": escape pathway alone beats the full brain, 99.6% vs 96.4%; 5B → future work), #2 (README "Phase 4c":
scrambled brains ~150× quieter; gain-matched scramble still 0/90; cutting only LPLC2/LC4→GF leaves 63/90 via 2-step
relays), #5 + #6 (README "Phase 4d": with +17 ms delay and a retuned alarm 27/30 hard, fine to 33 ms, breaks at
50 ms; full brain is over the 16.7 ms frame budget on 64% of alarm-on frames, ~11 ms engine floor). Next: Phase 6 post.
Headline wording: "the fly's escape pathway", not "the whole brain".

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
- M1 Pro: 3–4 processes at once ≈ 2× throughput. Background tasks are cut off at 2 h: keep runs shorter or resumable.

## Open decisions (ask Rajan)
- 5B (mushroom-body learning): future work; the MBON→DNp01 path check showed only weak 2-step routes (README Phase 4b).
- Real game (Stretch B) comes last, run by Rajan. Feasible; real time is borderline, and a game slow-down mod is the
  honest fallback. Undecided: whether Claude builds the screen/keyboard connector. Don't touch the game binary until then.

## Working with Rajan
- Commit each tested piece and push to `origin` (private `degeneratedstupidity/drosoXgeo`). **No `Co-Authored-By`
  or "Generated with" lines**; Rajan is the only author (repo git email = Rajan's GitHub noreply).
- Long runs: log to `results/`, wrap in `caffeinate -i`, give Rajan a `tail -f` command and say whether anything
  would disturb the run (reading logs: safe; a second simulation: slower, same results).
- Ask before publishing artifacts or sending anything to outside services (privacy), including web searches.

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
