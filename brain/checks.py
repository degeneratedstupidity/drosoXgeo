"""The project's checklist: is the simulated brain doing what it must?

Every check is written the same way:
  question  what we want to know, in plain words
  expect    what the answer must be, decided BEFORE looking, and where that comes from
  got       what this run measured
  verdict   PASS or FAIL

Expectations come from four places:
  published   a number other scientists measured (Shiu et al. 2024)
  control     a setup whose answer is known by construction (no input, cell switched off)
  scrambled   random rewiring must break the result, or the wiring isn't doing the work
  consistent  two ways of computing the same thing must agree

Runs everything fresh, in about 3 minutes. Writes figures/checks.json.

    python -m brain.checks
"""

import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import pyarrow.parquet as pq

from lif import core, engine_fused, experiment
from lif.stimulus_flybrain import RIGHT_SUGAR_GRN_IDS
from lif.validate_notebook import compare

from brain import neurons
from brain.exp_loom import DRIVE, PRE
from brain.paths import FIGURES, MLX_REPO, PACK_V630, PACK_V783, RESULTS, shuffled
from brain.stepper import Brain

MN9 = 720575940660219265
PUBLISHED = MLX_REPO / "data/ref/results/example/sugarR_100Hz.parquet"
TRIALS = 5                       # per loom condition; the full experiment uses 10


@dataclass
class Check:
    phase: str
    question: str
    kind: str
    expect: str
    got: str
    passed: bool
    why: str


checks: list[Check] = []


def report(c: Check) -> None:
    c.passed = bool(c.passed)
    checks.append(c)
    mark = "\033[32mPASS\033[0m" if c.passed else "\033[31mFAIL\033[0m"
    print(f" [{mark}] {len(checks):>2}. {c.question}  ({c.kind})")
    print(f"         expect  {c.expect}")
    print(f"         got     {c.got}")
    print(f"         why     {c.why}\n", flush=True)


# ------------------------------------------------------------------ phase 2: is the simulator right?

def sugar_run(name, pack, sugar):
    params = dict(experiment.default_params, r_poi=100.0)
    return experiment.run_exp(name, sugar, RESULTS / "checks", params=params,
                              force_overwrite=True, pack=pack, seed=0)


def phase2():
    print("PHASE 2 · is the simulator right?\n")
    v630, v783 = core.load_pack(PACK_V630), core.load_pack(PACK_V783)
    sugar21 = [int(i) for i in RIGHT_SUGAR_GRN_IDS]
    sugar20 = [i for i in sugar21 if i in set(v783.neuron_ids.tolist())]
    n_run, t_run = experiment.default_params["n_run"], experiment.default_params["t_run"]

    ours = sugar_run("v630_21", v630, sugar21)
    a, b, z = compare(PUBLISHED, ours, n_run, t_run).of(MN9)
    report(Check("2", "Does our simulator give the published answer?", "published",
                 f"eating cell (MN9) at {a:.1f} Hz like Shiu et al.'s own run, within noise (|z| < 3)",
                 f"{b:.1f} Hz (z = {z:+.2f})", abs(z) < 3,
                 "if our engine disagrees with the published one, nothing built on it can be trusted"))

    old = sugar_run("v630_20", v630, sugar20)
    new = sugar_run("v783_20", v783, sugar20)
    a, b, z = compare(old, new, n_run, t_run).of(MN9)
    report(Check("2", "Does the newer brain map give the same answer?", "consistent",
                 "same 20 sugar cells on the old (v630) and new (v783) map: MN9 rates within noise (|z| < 3)",
                 f"old {a:.1f} Hz, new {b:.1f} Hz (z = {z:+.2f})", abs(z) < 3,
                 "we build on the new map, so it must not change a result the old one got right"))
    real_mn9 = b

    scr = sugar_run("v783_shuffled0", core.load_pack(shuffled(PACK_V783, 0)), sugar20)
    rate = dict(zip(*[experiment.rates(scr)[c].to_pylist() for c in ("flywire_id", "rate_hz")])).get(MN9, 0.0)
    report(Check("2", "Does scrambled wiring break the sugar → eating result?", "scrambled",
                 f"MN9 below 5% of the real wiring's {real_mn9:.1f} Hz",
                 f"{rate:.1f} Hz", rate < 0.05 * real_mn9,
                 "if a randomly wired brain also 'eats', the fly's wiring isn't what produces it"))

    # the frame-by-frame stepper must equal the validated engine
    index = {int(n): i for i, n in enumerate(v783.neuron_ids)}
    targets = np.array([index[i] for i in sugar20], dtype=np.int32)
    stim = core.make_stimulus_for(v783, targets, 200.0, n_ticks=2000, seed=11)
    ref = engine_fused.run(v783, stim, chunk=32, edge_split=1, warmup=0)
    brain = Brain(v783, {"sugar": targets}, monitor=[index[MN9]])
    brain.step_draws(np.asarray(stim.draws).astype(np.uint8))
    diff = int((brain.spike_counts() != ref.spike_counts).sum())
    report(Check("2", "Does our frame-by-frame stepper match the validated engine?", "consistent",
                 "identical spike counts for every one of the 138,639 neurons, same input",
                 f"{diff} neurons differ ({int(ref.spike_counts.sum())} spikes compared)", diff == 0,
                 "the game needs the stepper, so it has to be the same physics, not an approximation"))


# ------------------------------------------------------------------ phase 3: the jump circuit

def loom(pack, idx, rate, seed, groups=("LPLC2", "LC4"), lesion=()):
    inputs = {g: idx[g] for g in ("LPLC2", "LC4")}
    monitor = np.concatenate([idx["LPLC2"], idx["DNp01"]])
    brain = Brain(pack, inputs, monitor=monitor, lesion=lesion, seed=seed)
    off = {g: 0.0 for g in inputs}
    sp = np.concatenate([brain.step(PRE, off), brain.step(DRIVE, {g: (rate if g in groups else 0.0) for g in inputs})])
    gf = sp[PRE:, -2:]
    first = np.flatnonzero(gf.any(axis=1))
    return {"lplc2_hz": sp[PRE:, :-2].sum() / (idx["LPLC2"].size * DRIVE * 1e-4),
            "gf_hz": gf.sum() / (2 * DRIVE * 1e-4),
            "latency": first[0] / 10 if first.size else None,
            "quiet_spikes": int(sp[:PRE].sum())}


def phase3():
    print("PHASE 3 · the jump circuit\n")
    real = core.load_pack(PACK_V783)
    idx = neurons.indices(real)
    run = lambda rate, **kw: [loom(real, idx, rate, s, **kw) for s in range(TRIALS)]

    r0 = run(0)
    gf0 = sum(r["gf_hz"] for r in r0)
    report(Check("3", "With no alarm, does the jump cell stay quiet?", "control",
                 "0 jump-cell spikes when nothing is shown",
                 f"{gf0:.0f} spikes in {TRIALS} tries", gf0 == 0,
                 "a jump cell that fires on its own would press jump for no reason"))

    r100 = run(100)
    got = np.mean([r["lplc2_hz"] for r in r100])
    report(Check("3", "Do the alarm cells fire at the rate we asked for?", "consistent",
                 "LPLC2 cells at 100 per second, within 5%",
                 f"{got:.1f} per second", abs(got - 100) < 5,
                 "if the input is wrong, everything measured downstream of it is wrong too"))

    fired = sum(r["latency"] is not None for r in r100)
    report(Check("3", "Does the alarm make the jump cell fire?", "the main test",
                 f"the jump cell fires in all {TRIALS} tries at alarm strength 100",
                 f"fired in {fired}/{TRIALS} tries, {np.mean([r['gf_hz'] for r in r100]):.0f} spikes per second",
                 fired == TRIALS,
                 "this is the escape reflex: no response here means no game"))

    r25, r200 = run(25), run(200)
    lat = lambda rows: float(np.median([r["latency"] for r in rows if r["latency"] is not None] or [np.inf]))
    l25, l100, l200 = lat(r25), lat(r100), lat(r200)
    report(Check("3", "Does a stronger alarm give a faster reaction?", "consistent",
                 "time to first jump-cell spike shrinks: strength 25 > 100 > 200",
                 f"{l25:.1f} ms > {l100:.1f} ms > {l200:.1f} ms", l25 > l100 > l200,
                 "the game needs timing that follows how close the spike is"))

    les = run(200, lesion=idx["DNp01"])
    gfl = sum(r["gf_hz"] for r in les)
    report(Check("3", "With the jump cell switched off, does it stay silent?", "control",
                 "0 jump-cell spikes even at the strongest alarm",
                 f"{gfl:.0f} spikes", gfl == 0,
                 "this is the 'no jump cell' control for the game; the switch has to work"))

    real200 = np.mean([r["gf_hz"] for r in r200])
    worst = 0.0
    for s in range(5):
        pack = core.load_pack(shuffled(PACK_V783, s))
        neurons.check_same_neurons(real, pack)
        worst = max(worst, np.mean([loom(pack, idx, 200, t)["gf_hz"] for t in range(TRIALS)]))
    report(Check("3", "Does scrambled wiring break the alarm → jump link?", "scrambled",
                 f"every one of 5 scrambled brains below 5% of the real wiring's {real200:.0f} spikes/s at strength 200",
                 f"worst scrambled brain: {worst:.1f} spikes/s ({100 * worst / real200:.1f}% of real)",
                 worst < 0.05 * real200,
                 "if a randomly wired brain also jumps, the fly's wiring isn't what makes it jump"))


def main() -> int:
    t0 = time.perf_counter()
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    print(f"drosoXgeo checklist · {time.strftime('%Y-%m-%d %H:%M')} · {chip or platform.machine()}\n")
    phase2()
    phase3()
    n_pass = sum(c.passed for c in checks)
    print(f"{n_pass}/{len(checks)} checks passed in {time.perf_counter() - t0:.0f} s")
    FIGURES.mkdir(exist_ok=True)
    (FIGURES / "checks.json").write_text(json.dumps(
        {"when": time.strftime("%Y-%m-%d %H:%M"), "machine": chip, "trials_per_loom_condition": TRIALS,
         "checks": [asdict(c) for c in checks]}, indent=2) + "\n")
    return 0 if n_pass == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
