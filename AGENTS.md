# AGENTS.md — drosoXgeo handoff

Read this, then `README.md` (roadmap + every result) and `data/SOURCES.md` (data, citations) before changing anything.

## Goal
A connectome-constrained LIF model of the fruit-fly brain (FlyWire v783, Shiu et al. 2024 model) plays a
Geometry Dash clone through its own escape circuit: looming detectors **LPLC2**/**LC4** drive the **Giant Fiber
(DNp01)**, and any DNp01 spike presses jump. For a LinkedIn post, learning, and maybe a term paper. Owner: Rajan,
an experienced developer **new to computational neuroscience**: explain in plain language, analogies first.

## Where things stand (2026-10-19, evening)
Phases 1–4 and 5A are done (results in `README.md`). Headline: real wiring wins 89/90 unseen levels untrained;
Giant Fiber silenced and 5 scrambled brains win 0/90; trained readouts don't beat the Giant Fiber.
**Now:** working through `SUGGESTIONS.md` (an external review; local only, Rajan keeps it out of git) before Phase 6. Done: #11, #3, #1, #7 (README
"Phase 4b": escape pathway alone beats the full brain, 99.6% vs 96.4%; 5B → future work), #2 (README "Phase 4c":
scrambled brains ~150× quieter; gain-matched scramble still 0/90; cutting only LPLC2/LC4→GF leaves 63/90 via 2-step
relays), #5 + #6 (README "Phase 4d": with +17 ms delay and a retuned alarm 27/30 hard, fine to 33 ms, breaks at
50 ms; full brain is over the 16.7 ms frame budget on 64% of alarm-on frames, ~11 ms engine floor). Next: Phase 6 post.
Headline wording: "the fly's escape pathway", not "the whole brain".
**Stretch B works** (README "Stretch B"): the escape pathway (`brain/fastpath.py`, CPU, identical to the GPU "wire"
setup) plays the real game live: 7/9 attempts past Stereo Madness's opening spikes, silenced and scrambled 0/22 and
0/21. The block section after that is a planning problem a looming reflex can't solve (README "The block sections":
eye and clone now know blocks/pits; best one-number rule and the fly both ~20% of terrain levels). Stretch C (learning
to time jumps) stopped at its stage-2 gate: receptive-field vision input works, but downstream of the looming cells
the real wiring carries no more "what's coming" than a gain-matched scramble (README "Stretch C").
**Stretch D (in progress):** Rajan wants it to clear Stereo Madness up to the ship (block 266.5)
with a *learned decision layer* on the looming cells, labelled honestly (README "Stretch D"). Real level from the
game's file (read-only), floating blocks in Fly Dash, cross-entropy learning with checkpoints (best from the start:
7/12 sim attempts reach ~145, `figures/learn_real_gen25.json`), a shadow of the level kept in step with the real
cube (`brain/shadow.py`), live play with the layer (real game: one attempt past the towers, others died at 33–48),
and a dopamine-like reward rule (`brain/dopamine.py`; sim: past the towers 8% → ~50%). Done since: overnight
real-game dopamine learning (`brain.live --dopamine`), the real game's jump-on-landing rule in Fly Dash, ~1 s tags,
the shadow sync fix (real game now matches Fly Dash; best 161). **2026-11-01:** practice mode (`--practice`) taught
the second half nothing (0% in all 4 late sections, 0/80 runs to the ship): there the eye would need "where does
the ground end", which looming cells don't signal (README "Practice mode"). **Next: Rajan's call** between adding a
ground-edge cue to the layer (changes the wording) and stopping Stretch D at ~60–65% of the way. Phase 6 after.

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

## Open decisions (ask Rajan)
- 5B (mushroom-body learning): future work; the MBON→DNp01 path check showed only weak 2-step routes (README Phase 4b),
  and the looming signal never reaches the Kenyon cells (Stretch C).
- Stretch D: overnight real-game dopamine learning (Rajan says when). Learning from a perfect player (the planner)
  first was offered and not chosen; it would change the post's wording ("watched a perfect player, then practised").
- Real game (Stretch B): built 2026-10-18/19. Run `python -m brain.live --press --r-ref 20 --steep 2 --countdown 5
  --tap-test` (normal mode; Rajan clicks into the game during the countdown, hands off). Capture and keys live in
  ~/Applications/FlyEye.app (Screen Recording) and FlyJump.app (Accessibility), built from brain/flyeye.swift and
  brain/flyjump.swift; rebuilding changes their ad-hoc signature, so macOS asks Rajan for the permission again.
  Capture only the game window, never the screen (Rajan's rule). Keys only reach the game while it's the front app.
  Rajan's local copy is the Windows build, run through the Sikarugir (Wine) wrapper `geometrydash.app` in the repo root
  (gitignored; game copied into its C:\Program Files; `open geometrydash.app`). It must start from its own folder,
  or it can't find Resources and crashes in CCNode::addChild. Never modify the game's own files;
  ask Rajan before any mod (e.g. Geode for the slow-down). Real time is borderline (README Phase 4d).

## Working with Rajan
- Commit each tested piece and push to `origin` (private `degeneratedstupidity/drosoXgeo`). **No `Co-Authored-By`
  or "Generated with" lines**; Rajan is the only author (repo git email = Rajan's GitHub noreply).
- Long runs: log to `results/`, wrap in `caffeinate -i`, give Rajan a `tail -f` command and say whether anything
  would disturb the run (reading logs: safe; a second simulation: slower, same results).
- Ask before publishing artifacts or sending anything to outside services (privacy), including web searches.
- Pushing: a second GitHub account is now logged in on this Mac, and git's stored credential answers "Repository not
  found". Push with the GitHub CLI's active account for that one command (no config change):
  `git -c credential.helper= -c 'credential.helper=!gh auth git-credential' push origin main`.
  `gh auth setup-git` would fix it permanently but changes global git config: Rajan's call.

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
