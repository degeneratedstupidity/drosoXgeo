"""Where the fly dies in the real game: a bar per block, with the level's silhouette underneath (for the post).

Data: every real-game attempt from the three runs of 2026-11-01 (two --dopamine runs and the --record clip run),
counted where it ended (the furthest block the shadow reached), keeping only attempts made entirely while the game
was the front app (otherwise the keys don't reach it and the cube dies at the first spike), and dropping each run's
unfinished last attempt. The silhouette comes from the level file, so the page and the image stay in results/
(gitignored, like the decoded level: it's RobTop's content).

    ../drosophila-brain-mlx/.venv/bin/python docs/build_walls_chart.py
    -> results/clip/walls.html, results/clip/walls.png (headless Chrome, 1200x675 at 2x)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from brain.stereo import level  # noqa: E402

RUNS = ["live_20261101-160021_fast_real_press.json", "live_20261101-174209_fast_real_press.json",
        "live_20261101-175245_fast_real_press.json"]
END = 266.5
WALLS = [(16.5, "first spike"), (52, "towers"), (102, "after the pit"), (153.5, "slab 1"), (161.2, "slab 2"),
         (172.5, "ceiling")]                      # the staircase (143) is a wall in Fly Dash only, not here


def attempts(path):
    d = json.load(open(path))
    front = d.get("front_app", [])
    spans, cur = [], None                                  # (start, end) of time with the game in front
    for ch in front + [{"t": 1e9, "front": None}]:
        if cur is not None:
            spans.append((cur, ch["t"]))
            cur = None
        if ch["front"] == "wine":
            cur = ch["t"]
    frames = [f for f in d["frames"] if f.get("shadow_x") is not None]
    out, start, best, prev = [], None, 0.0, None
    for f in frames:
        x, t = f["shadow_x"], f["t"]
        if prev is not None and x < prev - 5:              # back near the start: the attempt ended
            out.append((start, t, best))
            start, best = t, x
        if start is None:
            start = t
        best = max(best, x)
        prev = x
    kept = [b for s, e, b in out if any(a <= s and e <= z for a, z in spans)]
    return kept, len(out)


def main():
    reached, total = [], 0
    for r in RUNS:
        k, n = attempts(os.path.join(ROOT, "results", "realgame", r))
        reached += k
        total += n
    counts = Counter(min(int(x), int(END)) for x in reached)
    (blocks, thorns, peaks), _ = level()
    best = max(reached)

    W, H = 1200, 675
    L, R = 70, 40                                          # plot margins
    top, base = 150, 470                                   # bar area (y px): top of the tallest bar, baseline
    sil_top, sil_bot = 490, 590                            # level silhouette band
    sx = lambda b: L + (W - L - R) * b / END
    peak = max(counts.values())
    sy = lambda c: base - (base - top) * c / peak
    hmax = 12.0
    hy = lambda h: sil_bot - (sil_bot - sil_top) * min(h, hmax) / hmax

    svg = []
    for gy in range(0, peak + 1, 10):                      # recessive grid
        svg.append(f'<line x1="{L}" x2="{W - R}" y1="{sy(gy):.1f}" y2="{sy(gy):.1f}" class="grid"/>'
                   f'<text x="{L - 10}" y="{sy(gy) + 4:.1f}" class="tick" text-anchor="end">{gy}</text>')
    bw = (W - L - R) / END
    for b, c in sorted(counts.items()):                    # one bar per block, 1px surface gap, rounded top
        x, y = sx(b) + 0.5, sy(c)
        svg.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(bw - 1, 1):.1f}" height="{base - y:.1f}" rx="1.5" '
                   f'class="bar"><title>block {b}: {c} attempt{"s" if c != 1 else ""} ended here</title></rect>')
    svg.append(f'<line x1="{L}" x2="{W - R}" y1="{base}" y2="{base}" class="axis"/>')
    # wall labels: the tallest bar near each wall, a short leader line, the name
    for i, (wx, name) in enumerate(WALLS):
        near = [counts.get(b, 0) for b in range(int(wx) - 2, int(wx) + 3)]
        c = max(near) if near else 0
        y = sy(c) - 10 - (16 if name in ("slab 2",) else 0)
        svg.append(f'<text x="{sx(wx):.1f}" y="{y:.1f}" class="wall" text-anchor="middle">{name}</text>')
    # best run and the goal
    svg.append(f'<line x1="{sx(best):.1f}" x2="{sx(best):.1f}" y1="{top - 20}" y2="{sil_bot}" class="best"/>'
               f'<text x="{sx(best) + 6:.1f}" y="{top - 8}" class="note">best: block {best:.1f}</text>')
    svg.append(f'<line x1="{sx(END):.1f}" x2="{sx(END):.1f}" y1="{top - 20}" y2="{sil_bot}" class="goal"/>'
               f'<text x="{sx(END) - 6:.1f}" y="{top - 8}" class="note" text-anchor="end">first ship (goal)</text>')
    # level silhouette: blocks and slabs, spikes, thorn pits
    svg.append(f'<rect x="{L}" y="{sil_top - 6}" width="{W - L - R}" height="{sil_bot - sil_top + 12}" class="band"/>')
    for b in blocks:
        if b.x0 >= END:
            continue
        x0, x1 = sx(b.x0), sx(min(b.x1, END))
        svg.append(f'<rect x="{x0:.1f}" y="{hy(b.top):.1f}" width="{x1 - x0:.1f}" '
                   f'height="{hy(b.bottom) - hy(b.top):.1f}" class="block"/>')
    for p in peaks:
        if p.x >= END:
            continue
        x0, x1, yb, yt = sx(p.x), sx(p.x + 1), hy(p.base), hy(p.base + 0.9)
        svg.append(f'<polygon points="{x0:.1f},{yb:.1f} {x1:.1f},{yb:.1f} {(x0 + x1) / 2:.1f},{yt:.1f}" '
                   f'class="spike"/>')
    for t in thorns:
        x0, x1 = sx(t.x0), sx(min(t.x1, END))
        svg.append(f'<rect x="{x0:.1f}" y="{hy(0.5):.1f}" width="{x1 - x0:.1f}" height="{hy(0) - hy(0.5):.1f}" '
                   f'class="thorns"/>')
    for b in range(0, 261, 20):
        svg.append(f'<text x="{sx(b):.1f}" y="{sil_bot + 22}" class="tick" text-anchor="middle">{b}</text>')
    svg.append(f'<text x="{(L + W - R) / 2:.1f}" y="{sil_bot + 44}" class="tick" text-anchor="middle">'
               f'block (Stereo Madness, start to the first ship)</text>')
    svg.append(f'<text x="22" y="{(top + base) / 2:.1f}" class="tick" text-anchor="middle" '
               f'transform="rotate(-90 22 {(top + base) / 2:.1f})">attempts ended here</text>')

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Where the fly dies</title>
<style>
:root {{ --surface-1:#141419; --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#8d8c86;
  --series-1:#9085e9; --grid:#2a2a33; --level:#4b4b57; --hazard:#d95926; --accent:#3987e5; }}
html,body {{ margin:0; background:var(--surface-1); }}
body {{ font-family: -apple-system, "SF Pro Text", Helvetica, sans-serif; color:var(--text-primary); }}
.viz-root {{ width:{W}px; height:{H}px; position:relative; }}
h1 {{ position:absolute; left:{L}px; top:22px; margin:0; font-size:28px; font-weight:650; letter-spacing:-0.01em; }}
p.sub {{ position:absolute; left:{L}px; top:62px; margin:0; font-size:15px; color:var(--text-secondary); max-width:1050px;
  line-height:1.4; }}
svg {{ position:absolute; left:0; top:0; }}
.grid {{ stroke:var(--grid); stroke-width:1; }} .axis {{ stroke:var(--text-muted); stroke-width:1; }}
.tick {{ fill:var(--text-muted); font-size:12px; }}
.bar {{ fill:var(--series-1); }} .bar:hover {{ fill:#b3abf2; }}
.wall {{ fill:var(--text-primary); font-size:13px; font-weight:600; }}
.note {{ fill:var(--text-secondary); font-size:12px; }}
.best {{ stroke:var(--accent); stroke-width:1.5; stroke-dasharray:4 3; }}
.goal {{ stroke:var(--text-muted); stroke-width:1.5; stroke-dasharray:2 3; }}
.band {{ fill:#1b1b22; }} .block {{ fill:var(--level); }} .spike {{ fill:var(--hazard); }}
.thorns {{ fill:var(--hazard); opacity:0.55; }}
.foot {{ position:absolute; left:{L}px; bottom:14px; font-size:11px; color:var(--text-muted); }}
</style></head><body><div class="viz-root">
<h1>Where a fruit-fly brain model dies in Geometry Dash</h1>
<p class="sub">{len(reached)} real-game attempts (1 Nov 2026): the fly's escape pathway plus a small learned layer. It clears
the spikes; it stops where the level asks for a precisely timed release (the floating slabs) or for information
its looming cells don't have (the low ceiling, reached only by the best run).</p>
<svg width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="Histogram of where attempts ended">
{chr(10).join(svg)}
</svg>
<div class="foot">Connectome-constrained LIF model (Shiu et al. 2024) on FlyWire (Dorkenwald et al. 2024; Schlegel et al.
2024). Orange: spikes and thorn pits; grey: blocks. Attempts made while another app was in front are left out
({total - len(reached)} of {total}).</div>
</div></body></html>"""
    out_html = os.path.join(ROOT, "results", "clip", "walls.html")
    open(out_html, "w").write(html)
    out_png = os.path.join(ROOT, "results", "clip", "walls.png")
    subprocess.run(["/Applications/Google Chrome.app/Contents/MacOS/Google Chrome", "--headless=new",
                    f"--screenshot={out_png}", f"--window-size={W},{H}", "--force-device-scale-factor=2",
                    "--hide-scrollbars", f"file://{out_html}"], check=True, capture_output=True)
    print(f"{len(reached)} attempts kept of {total}; best {best:.1f}; most at one block: {peak}; wrote {out_png}")
    print("top blocks:", sorted(counts.items(), key=lambda kv: -kv[1])[:8])


if __name__ == "__main__":
    main()
