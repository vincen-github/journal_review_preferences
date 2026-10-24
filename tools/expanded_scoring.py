"""Prepare/verify cached scores and freeze calibration selection; no model calls."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from journal_prefs.expanded_scoring import prepare_scoring, combine_group, fit_learning_curve, lock_scoring_inputs
from journal_prefs.expanded_evaluation import release_test, evaluate_expanded
from run_codex_baseline import SETTINGS

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "lock", "combine", "fit", "release-test", "evaluate"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--old-pilot")
    parser.add_argument("--group", choices=["train", "validation", "test"])
    parser.add_argument("--prompt", default=str(ROOT / "prompts/iclr2017_official_form.v1.json"))
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare_scoring(args.root, args.old_pilot, args.prompt, SETTINGS)
    elif args.action == "lock":
        result = lock_scoring_inputs(args.root, args.old_pilot, args.prompt)
    elif args.action == "combine":
        result = combine_group(args.root, args.group, args.prompt)
    elif args.action == "fit":
        result = fit_learning_curve(args.root)
    elif args.action == "release-test":
        result = release_test(args.root)
    else:
        result = evaluate_expanded(args.root)
    print(json.dumps(result))
