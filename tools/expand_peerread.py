"""Explicit, bounded public-source expansion; never launches models or upstream code."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from journal_prefs.expanded_pilot import build_expansion

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--old-pilot", required=True)
    parser.add_argument("--allow-public-network", action="store_true")
    args = parser.parse_args()
    result = build_expansion(args.out, args.old_pilot, args.allow_public_network)
    print(json.dumps({k: result[k] for k in ("paired_count", "eligible_count", "frame_band_counts", "group_band_counts", "unique_recommendations", "retained_unanimous_high_rejected", "partition_sha256")}))
