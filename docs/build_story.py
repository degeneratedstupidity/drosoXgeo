"""The post's video: the illustration, then the same run from inside, then the control (2026-11-01, Rajan's idea).

  1. the Blender illustration (results/clip/fly_gaming_captioned.mp4, docs/build_clip.py)
  2. "What the model actually saw": the same attempt (clip run, attempt 2) as the 440-px picture the eye program got,
     with the neurons' activity from the run's log scrolling underneath: looming input to LPLC2/LC4, Giant Fiber
     (DNp01) spikes, and the space bar (pressed by the learned decision layer)
  3. the control: the reflex alone (Giant Fiber -> jump), wiring scrambled (shuffled0), same eye and alarm: the
     looming input arrives, the Giant Fiber stays silent, no jump
  with short title cards between (LinkedIn autoplays muted). Panels are drawn with numpy; text comes from small HTML
  pages screenshotted by headless Chrome (transparent PNGs). Everything goes to results/clip/ (game footage).

    ../drosophila-brain-mlx/.venv/bin/python docs/build_story.py --control results/realgame/live_<control>.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP = os.path.join(ROOT, "results", "clip")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
FPS, S = 30, 1080
CLIP_RUN = os.path.join(ROOT, "results", "realgame", "live_20261101-175245_fast_real_press")
T0, T1 = 3.90, 19.75                         # attempt 2 of the clip run: the one on the illustration's monitor
GAME = (34, 150, 1012, 568)                  # x, y, w, h of the game picture
LANES = {"alarm": (770, 100), "gf": (905, 44), "press": (985, 24)}   # y, height
WINDOW = 4.0                                 # seconds of history shown in the traces
BG = np.array([20, 20, 25], np.uint8)
LANE_BG = np.array([27, 27, 34], np.uint8)
VIOLET, VIOLET_HI = np.array([144, 133, 233], np.uint8), np.array([179, 171, 242], np.uint8)
WHITE, CYAN = np.array([245, 245, 250], np.uint8), np.array([51, 217, 255], np.uint8)


def html_png(name, body, transparent=True, size=(S, S)):
    """A PNG of a small HTML page (text and labels), via headless Chrome."""
    page = os.path.join(CLIP, f"{name}.html")
    open(page, "w").write(f"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{{margin:0;width:{size[0]}px;height:{size[1]}px;background:{'transparent' if transparent else '#141419'};
font-family:-apple-system,"SF Pro Text",Helvetica,sans-serif;color:#fff;overflow:hidden}}
.h{{position:absolute;left:34px;top:34px;font-size:40px;font-weight:700;letter-spacing:-0.01em}}
.s{{position:absolute;left:34px;top:88px;font-size:20px;color:#c3c2b7;width:1010px;line-height:1.35}}
.l{{position:absolute;left:34px;font-size:17px;font-weight:600;color:#e8e7e2}}
.n{{position:absolute;font-size:13px;color:#8d8c86}}
.card{{position:absolute;left:80px;right:80px;top:0;bottom:0;display:flex;flex-direction:column;justify-content:center}}
.card .t{{font-size:58px;font-weight:750;line-height:1.1;letter-spacing:-0.015em}}
.card .u{{margin-top:22px;font-size:26px;color:#c3c2b7;line-height:1.35}}
.card .f{{margin-top:40px;font-size:16px;color:#8d8c86;line-height:1.4}}
</style></head><body>{body}</body></html>""")
    out = os.path.join(CLIP, f"{name}.png")
    subprocess.run([CHROME, "--headless=new", f"--screenshot={out}", f"--window-size={size[0]},{size[1]}",
                    "--hide-scrollbars", "--default-background-color=00000000" if transparent else
                    "--default-background-color=141419ff", f"file://{page}"], check=True, capture_output=True)
    return out


def labels(title, sub, press_label):
    ya, yg, yp = LANES["alarm"][0], LANES["gf"][0], LANES["press"][0]
    return (f'<div class="h">{title}</div><div class="s">{sub}</div>'
            f'<div class="l" style="top:{ya - 28}px">Looming input to the looming detectors (LPLC2, LC4)</div>'
            f'<div class="l" style="top:{yg - 28}px">Giant Fiber (DNp01) spikes</div>'
            f'<div class="l" style="top:{yp - 26}px">{press_label}</div>'
            f'<div class="n" style="left:34px;top:{yp + 30}px">{WINDOW:.0f} s ago</div>'
            f'<div class="n" style="right:34px;top:{yp + 30}px">now</div>')


def panel_frames(frames, log, t_frames):
    """Yield 1080x1080 RGB frames: the game picture plus the scrolling traces (numpy only)."""
    lt = np.array([f["t"] for f in log])
    alarm = np.array([f["alarm_hz"] for f in log], float)
    gf = np.array([f["gf"] for f in log], float)
    press = np.array([bool(f["pressed"]) for f in log])
    gx, gy, gw, gh = GAME
    amax = max(60.0, alarm.max())
    for img, now in zip(frames, t_frames):
        c = np.empty((S, S, 3), np.uint8)
        c[:] = BG
        # the game picture, enlarged with nearest-neighbour pixels (it really is 440 px wide)
        h, w = img.shape[:2]
        ys = (np.arange(gh) * h // gh).clip(0, h - 1)
        xs = (np.arange(gw) * w // gw).clip(0, w - 1)
        c[gy:gy + gh, gx:gx + gw] = img[ys][:, xs]
        # traces: x = time over the last WINDOW seconds, newest at the right
        col_t = now - WINDOW + (np.arange(gw) + 0.5) * WINDOW / gw
        valid = (col_t >= lt[0]) & (col_t <= lt[-1])
        y0, hh = LANES["alarm"]
        c[y0:y0 + hh, gx:gx + gw] = LANE_BG
        a = np.interp(col_t, lt, alarm) * valid
        top = (y0 + hh - (a / amax * hh)).astype(int)
        rows = np.arange(y0, y0 + hh)[:, None]
        fill = rows >= top[None, :]
        lane = c[y0:y0 + hh, gx:gx + gw]
        lane[fill] = VIOLET
        edge = (rows >= top[None, :] - 1) & (rows <= top[None, :] + 1) & (a[None, :] > 0)
        lane[edge] = VIOLET_HI
        y0, hh = LANES["gf"]
        c[y0:y0 + hh, gx:gx + gw] = LANE_BG
        sel = (lt > now - WINDOW) & (lt <= now) & (gf > 0)
        for t_, n_ in zip(lt[sel], gf[sel]):
            x = int((t_ - (now - WINDOW)) / WINDOW * gw)
            th = int(min(1.0, 0.45 + 0.15 * n_) * hh)
            c[y0 + hh - th:y0 + hh, gx + max(x - 1, 0):gx + min(x + 1, gw)] = WHITE
        y0, hh = LANES["press"]
        c[y0:y0 + hh, gx:gx + gw] = LANE_BG
        idx = np.searchsorted(lt, col_t).clip(0, len(lt) - 1)
        on = press[idx] & valid
        c[y0:y0 + hh, gx:gx + gw][:, on] = CYAN
        # a thin playhead at "now"
        c[LANES["alarm"][0] - 4:LANES["press"][0] + LANES["press"][1] + 4, gx + gw - 2:gx + gw] = WHITE
        yield c


def encode(name, frames, overlay):
    out = os.path.join(CLIP, f"{name}.mp4")
    p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{S}x{S}",
                          "-r", str(FPS), "-i", "-", "-i", overlay, "-filter_complex", "[0][1]overlay,format=yuv420p",
                          "-c:v", "libx264", "-crf", "17", "-preset", "slow", out], stdin=subprocess.PIPE)
    n = 0
    for f in frames:
        p.stdin.write(f.tobytes())
        n += 1
    p.stdin.close()
    p.wait()
    return out, n


def card(name, png, seconds):
    out = os.path.join(CLIP, f"{name}.mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-loop", "1", "-framerate", str(FPS), "-t", str(seconds),
                    "-i", png, "-vf", f"fade=in:st=0:d=0.25,fade=out:st={seconds - 0.25}:d=0.25,format=yuv420p",
                    "-c:v", "libx264", "-crf", "17", out], check=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--control", required=True, help="the control run's json (its npz next to it)")
    ap.add_argument("--control-seconds", type=float, nargs=2, default=(0.0, 8.0), help="window of the control run")
    a = ap.parse_args()

    # part 2: the same attempt as the illustration, from inside
    log = json.load(open(CLIP_RUN + ".json"))["frames"]
    z = np.load(CLIP_RUN + ".npz")
    zt, zf = z["t"], z["frames"]
    t_out = np.arange(T0, T1, 1 / FPS)
    pick = [int(np.argmin(abs(zt - x))) for x in t_out]
    crop = lambda im: im[15:, :]                                   # the window's title bar
    ov2 = html_png("story_overlay2", labels(
        "What the model actually saw",
        "The same run, from inside: the picture its eye program receives, and what the fly's neurons did with it.",
        "Space bar (pressed by the small learned layer)"))
    part2, n2 = encode("story_part2", panel_frames((crop(zf[i]) for i in pick), log, t_out), ov2)

    # part 3: the control
    cpath = a.control[:-5] if a.control.endswith(".json") else a.control
    clog = json.load(open(cpath + ".json"))["frames"]
    cz = np.load(cpath + ".npz")
    ct, cf = cz["t"], cz["frames"]
    c0, c1 = a.control_seconds
    tc = np.arange(c0, c1, 1 / FPS)
    cpick = [int(np.argmin(abs(ct - x))) for x in tc]
    ov3 = html_png("story_overlay3", labels(
        "Control: same neurons, wiring scrambled",
        "The reflex alone (Giant Fiber → jump), same eye and settings, connections shuffled at random. The looming "
        "input still arrives; the Giant Fiber never fires.",
        "Space bar (pressed whenever the Giant Fiber fires)"))
    part3, n3 = encode("story_part3", panel_frames((crop(cf[i]) for i in cpick), clog, tc), ov3)

    # title cards and the end card
    c_a = card("story_card_a", html_png("story_card_a", '<div class="card"><div class="t">What the model actually '
                                        'saw</div><div class="u">The same run, from inside the simulated fly brain'
                                        '</div></div>', transparent=False), 2.0)
    c_b = card("story_card_b", html_png("story_card_b", '<div class="card"><div class="t">The control</div>'
                                        '<div class="u">Same neurons, connections scrambled at random</div></div>',
                                        transparent=False), 2.0)
    c_end = card("story_card_end", html_png("story_card_end", '<div class="card"><div class="t">The wiring plays.'
                                            '</div><div class="u">Real FlyWire wiring: past the opening spikes, and '
                                            'with a small learned layer to block 174 of 266.<br>Scrambled wiring: '
                                            'never past the first spike.</div><div class="f">Connectome-constrained '
                                            'LIF model (Shiu et al. 2024) on FlyWire (Dorkenwald et al. 2024; '
                                            'Schlegel et al. 2024). Data CC BY-NC 4.0.</div></div>',
                                            transparent=False), 3.5)

    parts = [os.path.join(CLIP, "fly_gaming_captioned.mp4"), c_a, part2, c_b, part3, c_end]
    lst = os.path.join(CLIP, "story_list.txt")
    open(lst, "w").write("".join(f"file '{p}'\n" for p in parts))
    out = os.path.join(CLIP, "fly_story.mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-vf", "fps=30,format=yuv420p", "-c:v", "libx264", "-crf", "18", "-preset", "slow",
                    "-movflags", "+faststart", out], check=True)
    print(f"part 2: {n2} frames, part 3: {n3} frames -> {out}")


if __name__ == "__main__":
    main()
