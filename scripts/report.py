"""Builds the ONE combined report: results/REPORT.md = ANALYSIS.md + Parts A, B, C. No GPU, seconds.

  uv run python scripts/report.py

Regenerates both part reports first (a5_report.py, b5_report.py), then stitches them together.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

for script, needs in [("a5_report.py", RESULTS / "partA" / "predictions"), ("b5_report.py", RESULTS / "partB" / "predictions"),
                      ("c4_report.py", RESULTS / "partC" / "predictions"), ("d1_controls_report.py", RESULTS / "partD" / "predictions")]:
    if needs.exists():
        subprocess.run([sys.executable, str(HERE / script)], stdout=subprocess.DEVNULL, check=False)

parts = [(HERE.parent / "ANALYSIS.md").read_text(encoding="utf-8").strip(), "", "---", "",
         "*Everything below is generated from `results/` by `report.py` (Part A: `a5_report.py`, Part B: `b5_report.py`, "
         "Part C: `c4_report.py`, Part D: `d1_controls_report.py`).*", ""]
for sub in ["partA", "partB", "partC", "partD"]:
    rep = RESULTS / sub / "REPORT.md"
    if rep.exists():
        # image links in part reports are relative to results/<part>/; rebase them to results/
        parts += [rep.read_text(encoding="utf-8").replace("](figures/", f"]({sub}/figures/"), ""]
(RESULTS / "REPORT.md").write_text("\n".join(parts), encoding="utf-8")
print(f"wrote {RESULTS / 'REPORT.md'}")
