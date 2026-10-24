"""Separate post-prediction evaluation; never calls a model or changes rules."""
import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from journal_prefs.data import fingerprint, number, read_json, write_json


def evaluate(run, label_path):
    root = Path(run)
    manifest = read_json(root / "manifest.json")
    if manifest["status"] != "completed_predictions_frozen_before_label_evaluation":
        raise ValueError("All fixed predictions must be frozen before reading labels")
    predictions = read_json(root / "predictions.json")
    if fingerprint(predictions) != manifest["predictions_sha256"]:
        raise ValueError("Frozen predictions changed")
    # The evaluator is the first process in this run that opens the label file.
    labels = read_json(label_path)
    label_map = {r["paper_id"]: r for r in labels}
    ids = [r["paper_id"] for r in predictions]
    if len(ids) != len(set(ids)) or set(ids) != set(label_map) or ids != manifest["paper_ids"]:
        raise ValueError("Frozen prediction cohort does not match labels")
    rows, errors = [], []
    for prediction in predictions:
        label = label_map[prediction["paper_id"]]
        scores = [number(x, "human score") for x in label["individual_scores"]]
        if not scores:
            raise ValueError("Missing human scores")
        mean = sum(scores) / len(scores)
        error = number(prediction["score"], "prediction") - mean
        errors.append(error)
        rows.append({"paper_id": prediction["paper_id"], "model_score": prediction["score"],
                     "human_individual_scores": scores, "human_mean": mean, "absolute_error": abs(error)})
    report = {"schema_version": 1, "arm": "generic", "n": len(rows), "per_paper": rows,
              "score_mae": sum(abs(x) for x in errors) / len(errors),
              "score_rmse": math.sqrt(sum(x*x for x in errors) / len(errors)),
              "mean_signed_error": sum(errors) / len(errors),
              "predictions_sha256": manifest["predictions_sha256"], "labels_sha256": fingerprint(labels),
              "score_calibration": "none", "comparison_improvement_claim": False,
              "notice": "Exploratory public-data baseline, not a blind trial or acceptance probability. "
                        "Small score-balanced cohort; manuscript/rating stages unverified; prior analyst label exposure; "
                        "possible public-data pretraining contamination. All six fixed papers retained."}
    if (root / "evaluation.json").exists():
        raise ValueError("Evaluation already exists; no overwrite")
    write_json(root / "evaluation.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--labels", required=True)
    args = parser.parse_args()
    report = evaluate(args.run, args.labels)
    import json
    print(json.dumps(report, indent=2))
