"""Make `ftlib` and the scripts/ modules importable without installing the project (CI runs CPU-only,
without torch), the same way `python scripts/<name>.py` sees them."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
