"""Release the fixed test only after selection, then evaluate that state once."""

import math
from datetime import datetime, timezone
from pathlib import Path

from .data import fingerprint, read_json, write_json
from .score_calibration import evaluate_test, load_state


def frozen_selection(root):
    root = Path(root)
    selection = read_json(root / "selection.json")
    plan, partition = read_json(root / "scoring_plan.json"), read_json(root / "partition.json")
    from .expanded_scoring import validate_partition, verify_integrity
    validate_partition(partition, plan)
    verify_integrity(root)
    state = load_state(root / ("calibration_n" + str(selection["selected_n"]) + ".json"))
    if (selection.get("frozen") is not True or selection["partition_sha256"] != partition["partition_sha256"]
            or plan["partition_sha256"] != partition["partition_sha256"]
            or selection["state_sha256"] != state["state_sha256"]
            or set(state["train_ids"]) != set(partition["nested_train_ids"][str(selection["selected_n"])])
            or set(state["validation_ids"]) != set(partition["ids"]["validation"])):
        raise ValueError("Frozen selection/state/partition mismatch")
    return selection, state, plan, partition


def release_test(root):
    root = Path(root)
    selection, state, plan, partition = frozen_selection(root)
    path = root / "test_release.json"
    if path.exists() or any((root / j["run_directory"]).exists() for j in plan["jobs"] if j["group"] == "test"):
        raise ValueError("Test release or test scoring already exists")
    receipt = {"selection_sha256": fingerprint(selection), "state_sha256": state["state_sha256"],
               "plan_sha256": fingerprint(plan), "partition_sha256": partition["partition_sha256"],
               "released_at_utc": datetime.now(timezone.utc).isoformat(),
               "test_labels_opened": False}
    write_json(path, receipt)
    return receipt


def verify_test_release(root):
    root = Path(root)
    selection, state, plan, partition = frozen_selection(root)
    receipt = read_json(root / "test_release.json")
    if (receipt["selection_sha256"] != fingerprint(selection) or receipt["state_sha256"] != state["state_sha256"]
            or receipt["plan_sha256"] != fingerprint(plan) or receipt["partition_sha256"] != partition["partition_sha256"]):
        raise ValueError("Selection/plan changed after test release")
    return selection, state, plan, partition


def evaluate_expanded(root):
    root = Path(root)
    if (root / "final_test_evaluation.json").exists():
        raise ValueError("Final evaluation already exists; do not overwrite")
    selection, state, plan, partition = verify_test_release(root)
    manifest = read_json(root / "combined_test/manifest.json")
    predictions = read_json(root / "combined_test/predictions.json")
    signature = fingerprint({"settings": manifest["settings"], "instructions": manifest["instructions_sha256"],
                             "schema": manifest["schema_sha256"]})
    if (manifest["status"] != "completed_predictions_frozen_before_label_evaluation"
            or fingerprint(predictions) != manifest["predictions_sha256"]
            or signature != state["provenance"]["scoring_signature"]
            or {p["paper_id"] for p in predictions} != set(partition["ids"]["test"])):
        raise ValueError("Incomplete, changed or incompatible test predictions")
    # Test labels enter only here, after all model predictions and the choice are frozen.
    report = evaluate_test(state, predictions, read_json(root / "cohort/test.labels.json"))
    train = {p["paper_id"]: p for p in read_json(root / "cohort/train.labels.json")}
    means = [sum(train[i]["individual_scores"]) / len(train[i]["individual_scores"]) for i in state["train_ids"]]
    constant = sum(means) / len(means)
    errors = [constant - p["human_mean"] for p in report["per_paper"]]
    report["training_mean_baseline"] = {"n_train": len(means), "prediction": constant, "n": len(errors),
                                        "mae": sum(abs(e) for e in errors) / len(errors),
                                        "rmse": math.sqrt(sum(e * e for e in errors) / len(errors)),
                                        "signed_bias": sum(errors) / len(errors)}
    report["selected_n"] = selection["selected_n"]
    report["selection_sha256"] = fingerprint(selection)
    report["notice"] = ("Same-year score-stratified held-out score agreement; all fixed test papers retained. "
                        "No test-based model/prompt/calibration choice. Source stages and model pretraining contamination "
                        "remain unverified; no formal blinding certificate, acceptance-probability or content-quality claim.")
    write_json(root / "final_test_evaluation.json", report)
    return report
