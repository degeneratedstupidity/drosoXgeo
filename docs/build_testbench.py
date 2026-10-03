"""Fill docs/testbench.template.html with the latest results -> docs/testbench.html.

    python3 docs/build_testbench.py
"""
from pathlib import Path

root = Path(__file__).resolve().parents[1]
page = (root / "docs/testbench.template.html").read_text()
for slot, name in (("SUMMARY", "phase3_loom_dnp01.json"), ("EXAMPLES", "phase3_loom_examples.json"), ("CHECKS", "checks.json")):
    data = (root / "figures" / name).read_text().strip().replace("</", "<\\/")
    page = page.replace(f"/*{slot}*/null", data, 1)
(root / "docs/testbench.html").write_text(page)
print("wrote docs/testbench.html")
