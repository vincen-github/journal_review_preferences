"""Bounded offline comparison. select never opens TEST; evaluate requires its frozen roster."""
import argparse
import random
import sys
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from journal_prefs.data import fingerprint, read_json, write_json
from journal_prefs.nonlinear_scores import fit_score_model, predict_score_model, score_metrics

GRID = ([{"kind": kind} for kind in ("raw", "mean", "affine")]
        + [{"kind": "quadratic_ridge", "alpha": a} for a in (10.0, 1.0, 0.1, 0.01)]
        + [{"kind": "tree", "max_depth": d, "min_leaf": leaf} for d in (1, 2, 3) for leaf in (10, 5)])


def cohort(source, group):
    predictions = read_json(source / ("combined_" + group) / "predictions.json")
    manifest = read_json(source / ("combined_" + group) / "manifest.json")
    labels = read_json(source / "cohort" / (group + ".labels.json"))
    ids = read_json(source / "partition.json")["ids"][group]
    partition = read_json(source / "partition.json")
    flat = sum(partition["ids"].values(), [])
    if fingerprint(partition["ids"]) != partition["partition_sha256"] or len(flat) != len(set(flat)):
        raise ValueError("Source partition changed or paper groups overlap")
    if manifest["status"] != "completed_predictions_frozen_before_label_evaluation" or fingerprint(predictions) != manifest["predictions_sha256"] or {r["paper_id"] for r in predictions} != set(ids) or len(predictions) != len(ids):
        raise ValueError("Changed/incomplete cached predictions")
    label_map = {r["paper_id"]: r for r in labels}
    if len(label_map) != len(labels) or set(label_map) != set(ids):
        raise ValueError("Cached labels differ from paper groups")
    by_id = {r["paper_id"]: r for r in predictions}
    # Only this whitelisted scalar is a predictor. IDs, human reviews/decisions and targets never are.
    return ([by_id[i]["score"] for i in ids],
            [mean(label_map[i]["individual_scores"]) for i in ids], ids,
            fingerprint({"predictions": predictions, "labels": labels}))


def select(source, output):
    if output.exists():
        raise ValueError("Use a new experiment output; never overwrite a selection")
    output.mkdir(parents=True)
    protocol = {"schema_version": 1, "seed": 20261023, "cv_folds": 5, "grid": GRID,
                "features": ["cached_model_total_score"], "content_dimension_scores_available": False,
                "scope": "Univariate nonlinear mappings, not multidimensional manuscript prediction or base-model training",
                "selection": "TRAIN-only 5-fold OOF MAE chooses one hyperparameter setting per family; fixed VALIDATION MAE chooses primary among five families. Numerical ties within 1e-12 follow fixed grid/family order (stronger ridge/shallow tree/larger leaves first). No refit on validation.",
                "test_status": "Already viewed exploratory test; not new strict blind testing",
                "test_comparison_roster": "Five family finalists, frozen before one test pass; legacy N20 affine is an ineligible historical reference",
                "no_new_llm_calls": True, "source_partition_sha256": read_json(source / "partition.json")["partition_sha256"]}
    write_json(output / "protocol.json", protocol)
    x, y, ids, train_hash = cohort(source, "train")
    vx, vy, vids, validation_hash = cohort(source, "validation")
    if set(ids) & set(vids) or (len(ids), len(vids)) != (60, 20):
        raise ValueError("Expected disjoint fixed 60/20 development cohorts")
    order = list(range(len(ids)))
    random.Random(protocol["seed"]).shuffle(order)
    folds = [order[i::5] for i in range(5)]
    write_json(output / "cv_folds.json", [[ids[i] for i in fold] for fold in folds])
    trials = []
    for config in GRID:
        try:
            oof, fold_maes = [None] * len(ids), []
            for held in folds:
                train = [i for i in range(len(ids)) if i not in held]
                fitted = fit_score_model(config["kind"], [x[i] for i in train], [y[i] for i in train], **{k: v for k, v in config.items() if k != "kind"})
                values = predict_score_model(fitted, [x[i] for i in held])
                for i, prediction in zip(held, values):
                    oof[i] = prediction
                fold_maes.append(score_metrics(values, [y[i] for i in held])["mae"])
            model = fit_score_model(config["kind"], x, y, **{k: v for k, v in config.items() if k != "kind"})
            trials.append({"config": config, "status": "completed", "model": model,
                           "cv": score_metrics(oof, y), "cv_fold_maes": fold_maes, "cv_fold_mae_sd": pstdev(fold_maes),
                           "train": score_metrics(predict_score_model(model, x), y)})
        except (ValueError, ZeroDivisionError) as error:
            trials.append({"config": config, "status": "failed", "error": str(error)})
    write_json(output / "cv_trials.json", trials)
    finalists = []
    for family in ("raw", "mean", "affine", "quadratic_ridge", "tree"):
        options = [r for r in trials if r["config"]["kind"] == family and r["status"] == "completed"]
        if not options:
            continue
        best = min(r["cv"]["mae"] for r in options)
        chosen = next(r for r in options if r["cv"]["mae"] <= best + 1e-12)
        finalists.append(dict(chosen, validation=score_metrics(predict_score_model(chosen["model"], vx), vy)))
    if not finalists:
        raise ValueError("No valid candidates")
    best_validation = min(r["validation"]["mae"] for r in finalists)
    chosen = next(r for r in finalists if r["validation"]["mae"] <= best_validation + 1e-12)
    old_selection = read_json(source / "selection.json")
    legacy = read_json(source / ("calibration_n" + str(old_selection["selected_n"]) + ".json"))
    legacy_model = dict(legacy["calibration"], parameter_count=2)
    if legacy_model["kind"] == "identity":
        legacy_model = {"kind": "raw", "parameter_count": 0}
    result = {"schema_version": 1, "frozen": True, "protocol_sha256": fingerprint(protocol), "cv_trials_sha256": fingerprint(trials),
              "train_sha256": train_hash, "validation_sha256": validation_hash, "selected_family": chosen["config"]["kind"],
              "finalists": finalists, "legacy_reference": {"model": legacy_model, "training_n": old_selection["selected_n"],
                  "validation": score_metrics(predict_score_model(legacy_model, vx), vy), "eligible_for_selection": False},
              "test_status": protocol["test_status"], "test_read_by_select": False}
    result["state_sha256"] = fingerprint(result)
    write_json(output / "selection.json", result)
    return result


def evaluate(source, output):
    target = output / "test_evaluation.json"
    if target.exists():
        raise ValueError("Test already evaluated; no overwrite or automatic repeated inspection")
    state = read_json(output / "selection.json")
    if state["state_sha256"] != fingerprint({k: v for k, v in state.items() if k != "state_sha256"}) or not state["frozen"]:
        raise ValueError("Selection changed or not frozen")
    if fingerprint(read_json(output / "protocol.json")) != state["protocol_sha256"] or fingerprint(read_json(output / "cv_trials.json")) != state["cv_trials_sha256"]:
        raise ValueError("Protocol/CV trials changed after selection")
    for group in ("train", "validation"):
        if cohort(source, group)[3] != state[group + "_sha256"]:
            raise ValueError("Development inputs changed after selection")
    x, y, ids, test_hash = cohort(source, "test")
    results = []
    for candidate in state["finalists"]:
        predictions = predict_score_model(candidate["model"], x)
        results.append({"family": candidate["config"]["kind"], "config": candidate["config"],
                        "selected_primary": candidate["config"]["kind"] == state["selected_family"],
                        "test": score_metrics(predictions, y), "predictions": predictions})
    legacy = predict_score_model(state["legacy_reference"]["model"], x)
    report = {"state_sha256": state["state_sha256"], "test_sha256": test_hash,
              "test_status": state["test_status"], "results": results,
              "legacy_reference": {"test": score_metrics(legacy, y), "predictions": legacy},
              "paper_ids": ids, "human_means": y, "no_new_llm_calls": True}
    write_json(target, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["select", "evaluate"])
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = select(Path(args.source), Path(args.out)) if args.action == "select" else evaluate(Path(args.source), Path(args.out))
    print({"selected_family": result.get("selected_family"), "state_sha256": result.get("state_sha256"),
           "test_status": result["test_status"], "no_new_llm_calls": True})
