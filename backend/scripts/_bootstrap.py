"""Make `app` importable when scripts are run as `python scripts/<name>.py` from backend/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
