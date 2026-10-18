"""Build and launch the two helper apps that touch the real game, each with only the permission it needs.

  FlyEye.app   (brain/flyeye.swift)   Screen Recording   streams the Geometry Dash window, nothing else
  FlyJump.app  (brain/flyjump.swift)  Accessibility      presses/releases space in the game's process, nothing else

Why apps, and why `open`: macOS gives a permission to whoever is "responsible" for a process. A program started
directly from Python counts as part of the terminal or IDE running Python, so it would use (and need) that app's
permissions, which then cover everything started from it. Launched through LaunchServices (`open`), each helper is
its own app with its own permission, and the IDE needs neither. They talk to Python through named pipes (FIFOs):
`open --stdout` / `--stdin` connect the app's output or input to a pipe in results/bin/.

The apps are installed in ~/Applications, a standard place where System Settings' "+" picker finds them (the project
path has a folder name ending in a space, which file pickers handle badly).

Built on first use with swiftc (comes with Xcode's command line tools; nothing downloaded) and signed ad hoc
(codesign -s -). Rebuilding changes the signature, so macOS may ask for the permission again afterwards: the apps
are only rebuilt when their source changes.
"""

from __future__ import annotations

import os
import plistlib
import re
import subprocess
import threading
import time
from pathlib import Path

from brain.paths import RESULTS

BIN = RESULTS / "bin"                       # pipes and logs
APPS = Path.home() / "Applications"         # the apps themselves
HERE = Path(__file__).parent


def build(name: str) -> Path:
    """~/Applications/<name>.app from brain/<name lowercase>.swift, rebuilt only if the source is newer."""
    source = HERE / f"{name.lower()}.swift"
    app = APPS / f"{name}.app"
    exe = app / "Contents" / "MacOS" / name
    if exe.exists() and exe.stat().st_mtime >= source.stat().st_mtime:
        return app
    exe.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["swiftc", "-O", "-swift-version", "5", str(source), "-o", str(exe)], check=True)
    (app / "Contents" / "Info.plist").write_bytes(plistlib.dumps({
        "CFBundleIdentifier": f"local.drosoxgeo.{name.lower()}", "CFBundleName": name, "CFBundleExecutable": name,
        "CFBundlePackageType": "APPL", "CFBundleVersion": "1", "CFBundleShortVersionString": "1",
        "LSUIElement": True,                                     # no Dock icon, no menu bar
    }))
    subprocess.run(["codesign", "-s", "-", "--force", str(app)], check=True, capture_output=True)
    return app


def _fifo(name: str) -> Path:
    path = BIN / name
    if path.exists():
        path.unlink()
    os.mkfifo(path)
    return path


def launch(name: str, args: list[str], *, stdin: Path | None = None, stdout: Path | None = None) -> Path:
    """Start a fresh instance of the app through LaunchServices. Returns its log (its stderr)."""
    app = build(name)
    BIN.mkdir(parents=True, exist_ok=True)
    log = BIN / f"{name}.log"
    log.write_text("")
    cmd = ["open", "-n", "-g", str(app), "--stderr", str(log)]
    if stdin:
        cmd += ["--stdin", str(stdin)]
    if stdout:
        cmd += ["--stdout", str(stdout)]
    subprocess.run(cmd + ["--args", *args], check=True)
    return log


def wait_for_status(log: Path, prefix: str, timeout: float = 15.0) -> str:
    """The app's first log line; raises SystemExit with it if it doesn't start with prefix."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        line = log.read_text().strip()
        if line:
            first = line.splitlines()[0]
            if not first.startswith(prefix):
                raise SystemExit(f"{log.stem}: {first}")
            return first
        time.sleep(0.05)
    raise SystemExit(f"{log.stem}: no answer in {timeout:g} s (blocked by a permission dialog? see System Settings)")


def own_pid(status: str) -> int | None:
    m = re.search(r"\bself (\d+)", status)
    return int(m.group(1)) if m else None


def open_in_thread(path: Path, mode: str, timeout: float = 15.0):
    """Open a FIFO (which blocks until the other side opens it) with a timeout."""
    box = []
    t = threading.Thread(target=lambda: box.append(open(path, mode, buffering=0)), daemon=True)
    t.start()
    t.join(timeout)
    if not box:
        raise SystemExit(f"the helper app never opened {path.name} (did it start? see results/bin/*.log)")
    return box[0]


class Jump:
    """The jump key, through FlyJump.app: set(True) = space down, set(False) = space up."""

    def __init__(self, game_pid: int):
        pipe = _fifo("flyjump.in")
        # Hold the pipe open (read+write, which doesn't wait for the other side) before FlyJump starts: otherwise
        # FlyJump can find no writer yet, read "end of input" and quit. Closing it later is FlyJump's signal to stop.
        self.pipe = os.fdopen(os.open(pipe, os.O_RDWR), "wb", buffering=0)
        log = launch("FlyJump", ["--pid", str(game_pid)], stdin=pipe)
        self.status = wait_for_status(log, "ready", timeout=30.0)
        self.down = False

    def set(self, down: bool):
        if down != self.down:
            self.pipe.write(b"d" if down else b"u")
            self.down = down

    def close(self):
        try:
            self.set(False)
            self.pipe.close()                  # end of input: FlyJump releases space and exits
        except OSError:
            pass
