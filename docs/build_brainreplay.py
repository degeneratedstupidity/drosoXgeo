"""Fill docs/brainreplay.template.html with a recorded level -> docs/brainreplay.html.

    python -m brain.record_level --difficulty hard --seed 1      # -> results/replay/hard_1.json
    python3 docs/build_brainreplay.py [hard_1]
"""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
name = sys.argv[1] if len(sys.argv) > 1 else "hard_1"
page = (root / "docs/brainreplay.template.html").read_text()
data = (root / "results/replay" / f"{name}.json").read_text().strip().replace("</", "<\\/")
page = page.replace("/*REPLAY*/null", data, 1)
out = root / "docs/brainreplay.html"
out.write_text(page)
print(f"wrote {out.relative_to(root)} ({out.stat().st_size / 1e6:.1f} MB) from results/replay/{name}.json")
