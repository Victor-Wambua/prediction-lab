import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (avoids a dependency). Existing env vars win."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv(BASE_DIR / ".env")

PREDICTION_LAB_URL = os.environ.get("PREDICTION_LAB_URL", "https://example.com/game")
HISTORY_ITEM_SELECTOR = os.environ.get("PREDICTION_LAB_HISTORY_ITEM_SELECTOR", "").strip() or None
DB_PATH = Path(os.environ.get("PREDICTION_LAB_DB_PATH", BASE_DIR / "data" / "prediction_lab.db"))
if not DB_PATH.is_absolute():
    DB_PATH = BASE_DIR / DB_PATH
POLL_SECONDS = float(os.environ.get("PREDICTION_LAB_POLL_SECONDS", "1.0"))
REPORTS_DIR = BASE_DIR / "reports"

# Collector tuning
MIN_OVERLAP = 8            # rounds that must match the stored tail to accept an alignment
STABLE_READS = 2           # identical consecutive snapshots required before processing
RELOAD_AFTER_SECONDS = 180  # reload the page if no valid snapshot for this long
MAX_MULTIPLIER = 1_000_000.0
