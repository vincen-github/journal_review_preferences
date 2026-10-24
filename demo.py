"""Double-click helper for the authored offline demonstration."""

from datetime import datetime, timezone
from pathlib import Path

from journal_prefs.__main__ import main


if __name__ == "__main__":
    folder = Path(__file__).resolve().parent / ("demo_" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    print("Offline synthetic plumbing demonstration. No API, GPU, training, or academic quality claim.")
    raise SystemExit(main(["demo", "--out", str(folder)]))
