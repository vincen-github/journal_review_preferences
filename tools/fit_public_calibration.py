"""Fit/choose absolute-score calibration using learning and validation only."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from journal_prefs.data import fingerprint, read_json
from journal_prefs.score_calibration import fit_state, save_state


def load_predictions(run):
    root = Path(run)
    manifest = read_json(root / "manifest.json")
    predictions = read_json(root / "predictions.json")
    if manifest["status"] != "completed_predictions_frozen_before_label_evaluation" or fingerprint(predictions) != manifest["predictions_sha256"]:
        raise ValueError("Raw predictions are incomplete or changed")
    signature = fingerprint({"settings": manifest["settings"], "instructions": manifest["instructions_sha256"], "schema": manifest["schema_sha256"]})
    return predictions, signature


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-run", required=True)
    parser.add_argument("--train-labels", required=True)
    parser.add_argument("--validation-run", required=True)
    parser.add_argument("--validation-labels", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise ValueError("Choose a new calibration state file")
    train, train_signature = load_predictions(args.train_run)
    validation, validation_signature = load_predictions(args.validation_run)
    if train_signature != validation_signature:
        raise ValueError("Model/prompt/settings differ between learning and validation")
    state = fit_state(train, read_json(args.train_labels), validation, read_json(args.validation_labels),
                      provenance={"scoring_signature": train_signature,
                                  "label_target": "same-year mean raw recommendations on1..10",
                                  "prior_analyst_label_exposure": True, "manuscript_stage": "unverified", "blind_claim": False})
    save_state(state, args.out)
    import json
    print(json.dumps({"calibration": state["calibration"], "state_sha256": state["state_sha256"]}))
