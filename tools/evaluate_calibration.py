"""Join test labels only after the learning/validation calibration is frozen."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from journal_prefs.data import read_json, write_json
from journal_prefs.score_calibration import evaluate_test, load_state
from fit_public_calibration import load_predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True)
    parser.add_argument("--test-run", required=True)
    parser.add_argument("--test-labels", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise ValueError("Evaluation exists; do not overwrite")
    state = load_state(args.state)
    predictions, signature = load_predictions(args.test_run)
    if signature != state["provenance"]["scoring_signature"]:
        raise ValueError("Test scoring settings differ from frozen calibration")
    report = evaluate_test(state, predictions, read_json(args.test_labels))
    report["notice"] = ("Exploratory score calibration, not a blind evaluation or evidence of better content judgment. "
                        "Six score-balanced test papers, prior analyst label exposure, unknown manuscript/review stages "
                        "and possible model pretraining contamination. All fixed papers retained; no test-based method choice.")
    write_json(args.out, report)
    import json
    print(json.dumps(report, indent=2))
