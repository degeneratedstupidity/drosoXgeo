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


# ---------------- terrain: blocks, towers and thorn pits, like Stereo Madness ----------------
# The real game isn't only spikes on flat ground. Its first block section (measured on recordings) is: two spikes,
# a 1-block step right after them, then towers rising out of a thorn pit. You must land on the step and jump from
# block top to block top. The jump itself matches Fly Dash (measured: 0.443 s in the air, 2.2 blocks high).
# FlyDash above is unchanged; this is a separate level type, "terrain".

THORN_H = 0.35                    # thorn hitbox height on the pit floor (they're drawn ~0.5 high)
THORN_VIS = 0.5                   # how high the eye sees them
SNAP = 0.25                       # landing tolerance: up to this far below a block's top still lands on it


@dataclass
class Block:
    x0: float                     # a solid column from the ground up to `top` (towers, steps, platforms)
    x1: float
    top: float


@dataclass
class Thorns:
    x0: float                     # a pit floor covered in thorns
    x1: float


@dataclass
class Peak:
    x: float                      # a spike standing on something: left edge and the height of its base
    base: float = 0.0


def _segment(rng, x):
    """One Stereo Madness-style obstacle starting at x. Returns (blocks, thorns, peaks, end x)."""
    kind = rng.choice(["spikes", "spikes", "step", "platforms"])
    if kind == "spikes":
        n = rng.choice([1, 1, 2, 2, 3])
        return [], [], [Peak(x + i) for i in range(n)], x + n
    if kind == "step":                                # spikes, a 1-block step, towers out of a thorn pit
        n = rng.choice([0, 1, 2, 2])
        peaks = [Peak(x + i) for i in range(n)]
        x0 = x + n
        blocks = [Block(x0, x0 + 1, 1.0)]
        h, end = 1.0, x0 + 1
        for _ in range(rng.choice([1, 2, 3])):
            h = min(h + 1, 3.0)
            start = end + rng.uniform(2.0, 3.0)
            blocks.append(Block(start, start + 1, h))
            end = start + 1
        land = end + rng.uniform(1.5, 2.5)
        return blocks, [Thorns(x0 + 1, land)], peaks, land
    blocks, thorns, peaks, cur = [], [], [], x       # platforms: 1 high, thorn pits between them
    for k in range(rng.choice([2, 3])):
        length = rng.uniform(3, 7)
        blocks.append(Block(cur, cur + length, 1.0))
        if length >= 5 and rng.random() < 0.5:
            peaks.append(Peak(cur + length / 2, 1.0))
        cur += length
        if k < 2:
            gap = rng.uniform(2.0, 3.0)
            thorns.append(Thorns(cur, cur + gap))
            cur += gap
    return blocks, thorns, peaks, cur


def make_terrain(seed=0, length=300.0):
    """A level of terrain segments with 5-9 blocks of flat ground between them, checked to be beatable
    (solvable(): a search over every jump timing). An unbeatable draw is replaced by the next one."""
    for attempt in range(100):
        rng = random.Random(seed * 1000 + attempt)
        blocks, thorns, peaks, x = [], [], [], 15.0
        while x < length - 25:
            b, t, p, x = _segment(rng, x)
            blocks += b
            thorns += t
            peaks += p
            x += rng.uniform(5, 9)
        level = (blocks, thorns, peaks)
        if solvable(level, length):
            return level
    raise RuntimeError(f"no beatable terrain level for seed {seed}")


class FlyDashTerrain(FlyDash):
    """Fly Dash with blocks, towers and thorn pits. Same interface and observation as FlyDash."""

    def __init__(self, seed=0, difficulty="terrain", length=300.0, level=None):
        self.length = length
        self.blocks, self.thorns, self.peaks = level if level is not None else make_terrain(seed, length)
        self.spikes = []                              # FlyDash's spike list stays empty; peaks replace it
        self.reset()

    def reset(self):
        self.t, self.px, self.py, self.vy = 0.0, 0.0, 0.0, 0.0
        self.grounded = True
        self.dead = self.won = False
        self.jumps = 0
        return self.observe()

    @property
    def on_ground(self):
        return self.grounded

    def _near(self, items, lo, hi, x0="x0", x1="x1"):
        return [o for o in items if getattr(o, x1) > lo and getattr(o, x0) < hi]

    def _surface(self, was=None):
        """The highest top under the player that it could stand on (ground = 0): within SNAP below its feet, or
        anything it was above before this frame's move (a fast fall can cross a top in one frame)."""
        reach = self.py + SNAP if was is None else max(self.py + SNAP, was)
        tops = [b.top for b in self._near(self.blocks, self.px, self.px + 1) if b.top <= reach]
        return max(tops, default=0.0)

    def step(self, jump: bool):
        if self.dead or self.won:
            return self.observe(), 0.0, True
        if jump and self.grounded:                    # holding the button re-jumps on landing, like GD
            self.vy, self.grounded = JUMP_V, False
            self.jumps += 1
        was = self.py
        if not self.grounded:
            self.py += self.vy * DT
            self.vy -= GRAVITY * DT
        self.px += SPEED * DT
        self.t += DT
        surface = self._surface(was)
        if self.grounded and surface < self.py:       # walked off an edge: start falling
            self.grounded, self.vy = False, 0.0
        elif not self.grounded and self.vy <= 0 and self.py <= surface:
            self.py, self.vy, self.grounded = surface, 0.0, True
        for b in self._near(self.blocks, self.px, self.px + 1):
            if self.py < b.top - SNAP:                # ran or jumped into its side
                self.dead = True
        x0, x1, h = SPIKE_HIT
        for p in self.peaks:
            if self.px + 1 > p.x + x0 and self.px < p.x + x1 and p.base - 1 < self.py < p.base + h:
                self.dead = True
        for th in self._near(self.thorns, self.px + 0.15, self.px + 0.85):
            if self.py < THORN_H:
                self.dead = True
        if self.px >= self.length:
            self.won = True
        return self.observe(), SPEED * DT, self.dead or self.won

    def observe(self):
        """What brain.realeye would report: the nearest obstacle in the band of rows the cube is in now, or, standing
        on something raised, the edge of its top where thorns lie below."""
        front, lo, hi = self.px + 1, self.py + 0.05, self.py + 0.95
        seen = []
        for b in self.blocks:
            if b.x1 > front and lo < b.top:
                seen.append(b.x0)
        for p in self.peaks:
            if p.x + 1 > front and lo < p.base + 1 and hi > p.base:
                seen.append(p.x + 0.5 * max(lo - p.base, 0.0))     # a triangle narrows towards its tip
        for th in self.thorns:
            if th.x1 > front and lo < THORN_VIS:
                seen.append(th.x0)
        if self.grounded and self.py >= 0.5:
            edge = self._support_end()
            if any(th.x0 < edge + 1.5 and th.x1 > edge for th in self.thorns):
                seen.append(edge)
        d = min((max(x, front) - front for x in seen), default=VIEW)
        d = min(max(d, 0.05), VIEW)
        theta = 2 * math.atan(0.5 / d)
        theta_dot = SPEED / (d * d + 0.25)
        return dict(retina=None, distance=d, theta=theta, theta_dot=theta_dot, tau=theta / theta_dot, y=self.py,
                    airborne=not self.grounded, progress=min(self.px / self.length, 1.0))

    def _support_end(self):
        """Where the surface the player stands on ends (adjacent blocks of the same height count as one)."""
        end = max(b.x1 for b in self._near(self.blocks, self.px, self.px + 1) if abs(b.top - self.py) < 1e-6)
        while True:
            nxt = [b.x1 for b in self.blocks if abs(b.top - self.py) < 1e-6 and b.x0 <= end + 1e-6 < b.x1]
            if not nxt:
                return end
            end = max(nxt)

    def render(self, width=40):
        """One ASCII frame:  ■ = player,  ▲ = spike,  █ = block,  ^ = thorns."""
        rows = [[" "] * width for _ in range(5)]
        for c in range(width):
            x = self.px - 2 + c + 0.5
            for b in self.blocks:
                if b.x0 <= x < b.x1:
                    for r in range(int(round(b.top))):
                        rows[r][c] = "█"
            if any(th.x0 <= x < th.x1 for th in self.thorns):
                rows[0][c] = "^"
            for p in self.peaks:
                if p.x <= x < p.x + 1:
                    rows[int(round(p.base))][c] = "▲"
        rows[min(int(self.py + 0.5), 4)][2] = "■"
        return "\n".join("".join(row) for row in reversed(rows)) + "\n" + "▔" * width


def solvable(level, length=300.0):
    """Can some sequence of jumps beat this terrain level? Searches every frame at which a standing player could
    jump (the only choice the game offers). States are (frame, height) while standing, so the search is small."""
    import copy
    env = FlyDashTerrain(level=level, length=length)
    todo, seen = [env], set()
    while todo:
        e = todo.pop()
        key = (round(e.px / (SPEED * DT)), round(e.py, 3))
        if key in seen:
            continue
        seen.add(key)
        for jump in (True, False):
            f = copy.copy(e)
            f.step(jump)
            while not (f.dead or f.won or f.grounded):
                f.step(False)
            if f.won:
                return True
            if not f.dead:
                todo.append(f)
    return False


LEVELS = list(DIFFICULTY) + ["terrain"]


def make_env(seed=0, difficulty="easy", length=300.0):
    """FlyDash for the spike-only difficulties, FlyDashTerrain for "terrain"."""
    if difficulty == "terrain":
        return FlyDashTerrain(seed, difficulty, length)
    return FlyDash(seed, difficulty, length)


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
