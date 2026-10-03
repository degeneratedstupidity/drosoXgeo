"""
flydash.py - a tiny, dependency-free Geometry-Dash-style runner built for fly-brain experiments.

Why a clone instead of the real game?
  * A simulated fly brain does not run at real-time speed. A clone can wait for the
    brain on every tick ("lock-step"); the real Geometry Dash can't.
  * You get perfect ground truth (distances, looming angles) for analysis and plots.
  * Thousands of headless runs per minute for tuning, training and controls.
  (Hooking the real game up is a stretch goal - see README.)

Units: 1 block = one Geometry Dash grid square. Time is in seconds.

The observation includes a LOOMING signal on purpose: an approaching spike grows on the
retina just like an approaching predator. In real flies, looming-sensitive neurons
(LPLC2 / LC4) drive the Giant Fiber neuron, which triggers the fly's escape JUMP.
Geometry Dash is a one-button jump game, so that circuit is the natural thing to wire up.

Run:  python3 flydash.py
"""
import math
import random
from dataclasses import dataclass

# ---------------- physics (roughly matches GD's cube at normal speed) ----------------
SPEED = 10.4                      # blocks / second
APEX = 2.1                        # jump height, blocks
AIRTIME = 0.45                    # seconds from takeoff to landing
GRAVITY = 8 * APEX / AIRTIME ** 2
JUMP_V = GRAVITY * AIRTIME / 2
DT = 1 / 60                       # game tick, seconds (60 fps like the real game)

SPIKE_HIT = (0.3, 0.7, 0.75)      # spike hitbox: x0, x1 (from spike's left edge), height
VIEW = 12.0                       # blocks visible ahead of the player
RETINA_W, RETINA_H, RES = 24, 4, 0.5   # retina: 24 x 4 cells of 0.5 blocks

DIFFICULTY = {                    # spike-group sizes and gaps between groups (blocks)
    "easy":   dict(groups=[1, 1, 1, 2],       gap=(6, 12)),
    "normal": dict(groups=[1, 1, 2, 2, 3],    gap=(5, 11)),
    "hard":   dict(groups=[1, 2, 3, 3],       gap=(5, 8)),
}


@dataclass
class Spike:
    x: float  # left edge, world coordinates


def make_level(seed=0, difficulty="easy", length=300.0):
    rng = random.Random(seed)
    cfg = DIFFICULTY[difficulty]
    spikes, x = [], 15.0
    while x < length - 10:
        n = rng.choice(cfg["groups"])
        spikes += [Spike(x + i) for i in range(n)]
        x += n + rng.uniform(*cfg["gap"])
    return spikes


class FlyDash:
    """Headless environment.  obs = env.reset();  obs, reward, done = env.step(jump)"""

    def __init__(self, seed=0, difficulty="easy", length=300.0):
        self.length = length
        self.spikes = make_level(seed, difficulty, length)
        self.reset()

    def reset(self):
        self.t, self.px, self.py, self.vy = 0.0, 0.0, 0.0, 0.0
        self.dead = self.won = False
        self.jumps = 0
        self._i = 0                             # index of first spike not yet passed
        return self.observe()

    @property
    def on_ground(self):
        return self.py == 0.0 and self.vy == 0.0

    def step(self, jump: bool):
        if self.dead or self.won:
            return self.observe(), 0.0, True
        if jump and self.on_ground:            # holding the button re-jumps on landing, like GD
            self.vy = JUMP_V
            self.jumps += 1
        if not self.on_ground:
            self.py += self.vy * DT
            self.vy -= GRAVITY * DT
            if self.py <= 0.0:
                self.py, self.vy = 0.0, 0.0
        self.px += SPEED * DT
        self.t += DT
        x0, x1, h = SPIKE_HIT
        while self._i < len(self.spikes) and self.spikes[self._i].x + x1 <= self.px:
            self._i += 1                       # drop spikes we've fully passed
        for s in self.spikes[self._i:self._i + 4]:   # player box is [px, px+1] x [py, py+1]
            if self.px + 1 > s.x + x0 and self.px < s.x + x1 and self.py < h:
                self.dead = True
                break
        if self.px >= self.length:
            self.won = True
        return self.observe(), SPEED * DT, self.dead or self.won

    # ---------------- what the "eye" sees ----------------
    def next_spike(self):
        for s in self.spikes[self._i:]:
            if s.x + SPIKE_HIT[1] > self.px:
                return s
        return None

    def observe(self):
        s = self.next_spike()
        d = max((s.x - (self.px + 1)) if s else VIEW, 0.05)   # gap from player's front edge
        d = min(d, VIEW)
        theta = 2 * math.atan(0.5 / d)                          # angular size of a 1-block object
        theta_dot = SPEED / (d * d + 0.25)                      # how fast it grows (looming)
        retina = [[0] * RETINA_W for _ in range(RETINA_H)]      # row 0 = ground level
        for sp in self.spikes[self._i:self._i + 20]:
            rel = sp.x - self.px
            if -1 < rel < VIEW:
                for c in range(RETINA_W):
                    cx = c * RES
                    if cx < rel + 1 and cx + RES > rel:
                        retina[0][c] = retina[1][c] = 1
        return dict(retina=retina, distance=d, theta=theta, theta_dot=theta_dot,
                    tau=theta / theta_dot, y=self.py, airborne=not self.on_ground,
                    progress=min(self.px / self.length, 1.0))

    def render(self, width=40):
        """One ASCII frame:  ■ = player,  ▲ = spike."""
        rows = [[" "] * width for _ in range(4)]
        for s in self.spikes:
            c = int(round(s.x - self.px)) + 2
            if 0 <= c < width:
                rows[0][c] = "▲"
        r = min(int(self.py + 0.5), 3)
        rows[r][2] = "■"
        return "\n".join("".join(row) for row in reversed(rows)) + "\n" + "▔" * width


# ---------------- baseline policies (the bar the fly brain has to beat) ----------------
def random_policy(obs, p=0.05, rng=random.Random(1)):
    return rng.random() < p


def looming_reflex(obs, theta_threshold=0.30):
    """Jump when the approaching spike's angular size crosses a threshold.
    This is a cartoon of the fly's Giant Fiber escape: no learning, one number."""
    return obs["theta"] > theta_threshold


def play(policy, seed=0, difficulty="easy", **kw):
    env = FlyDash(seed, difficulty)
    obs, done = env.reset(), False
    while not done:
        obs, _, done = env.step(policy(obs, **kw))
    return obs["progress"], env.won


def evaluate(policy, difficulty, seeds=range(30), **kw):
    results = [play(policy, s, difficulty, **kw) for s in seeds]
    return (sum(p for p, _ in results) / len(results),
            sum(w for _, w in results) / len(results))


if __name__ == "__main__":
    env = FlyDash(seed=0)
    for _ in range(40):
        env.step(False)
    print("A frame of Fly Dash:\n" + env.render() + "\n")

    print(f"{'policy':<28}{'difficulty':<10}{'avg progress':>14}{'levels won':>12}")
    for diff in DIFFICULTY:
        prog, win = evaluate(random_policy, diff)
        print(f"{'random button mashing':<28}{diff:<10}{prog:>13.0%}{win:>12.0%}")
        best = max(((evaluate(looming_reflex, diff, theta_threshold=th), th)
                    for th in [x / 100 for x in range(10, 280, 4)]), key=lambda r: r[0])
        (prog, win), th = best
        label = f"looming reflex (θ>{th:.2f} rad)"
        print(f"{label:<28}{diff:<10}{prog:>13.0%}{win:>12.0%}")
