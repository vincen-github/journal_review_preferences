"""Frozen historical retrieval, train-only calibration, and chronological backtest."""

import math
import re
from pathlib import Path

from .data import (available_at, fingerprint, one_version_per_paper, paper_input,
                   number, target_score, timestamp, validate_paper, write_json)
from .models import validate_response


def tokens(text):
    return set(re.findall(r"\w{2,}", text.lower()))


def retrieve(paper, training, top_k=3):
    query = tokens(paper["title"] + " " + paper["text"])
    ranked = []
    for row in training:
        if row["manuscript_id"] == paper["manuscript_id"] or row["venue_id"] != paper["venue_id"]:
            continue
        if available_at(row) >= timestamp(paper["submitted_at"]):
            continue
        words = tokens(row["title"] + " " + row["text"])
        overlap = len(query & words) / max(1, len(query | words))
        if overlap > 0:
            ranked.append((overlap, row))
    ranked.sort(key=lambda item: (-item[0], item[1]["manuscript_id"]))
    return [{"manuscript_id": row["manuscript_id"], "version_id": row["version_id"],
             "title": row["title"], "text_excerpt": row["text"][:2000],
             "reviews": [{"text": r["text"], "score": r["score"]} for r in row["reviews"]],
             "decision": row["decision"]["label"], "similarity": round(similarity, 6),
             "available_at": available_at(row).isoformat()}
            for similarity, row in ranked[:top_k]]


def predict(paper, training, rubric, model, top_k=3):
    validate_paper(paper, rubric)
    request = {"paper": paper_input(paper), "rubric": rubric,
               "historical_examples": retrieve(paper, training, top_k)}
    response = validate_response(model.review(request), request)
    return {"manuscript_id": paper["manuscript_id"], "version_id": paper["version_id"],
            "raw_score": response["score"], "review": response["review"], "evidence": response["evidence"],
            "synthetic_output": model.synthetic, "human_review_required": True,
            "retrieved_ids": [e["manuscript_id"] for e in request["historical_examples"]],
            "inference_input_sha256": fingerprint(request)}


def accepted_percentile(values, quantile):
    """Lower order statistic, adapted from LSAR-public _pctl (MIT).

    Copyright (c) 2026 Chenguang Pan. See THIRD_PARTY_LICENSES.md.
    """
    if not values or not 0 <= quantile <= 1:
        raise ValueError("Percentile needs nonempty values and quantile in [0, 1]")
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(quantile * (len(ordered) - 1))))
    return float(ordered[index])


def fit_affine(predictions, rows):
    if len(rows) < 3:
        return None
    x = [p["raw_score"] for p in predictions]
    y = [target_score(row) for row in rows]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    variance = sum((v - mx) ** 2 for v in x)
    if variance < 1e-12:
        return None
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / variance
    if slope <= 0 or not math.isfinite(slope):
        return None
    return {"kind": "affine", "slope": slope, "intercept": my - slope * mx}


def calibrated_score(raw, calibration, rubric):
    low, high = rubric["score_range"]
    score = raw * calibration["slope"] + calibration["intercept"]
    return round(min(high, max(low, score)), 6)


def apply_scores(predictions, calibration, rubric, threshold):
    return [dict(p, calibrated_score=calibrated_score(p["raw_score"], calibration, rubric),
                 meets_score_threshold=calibrated_score(p["raw_score"], calibration, rubric) >= threshold)
            for p in predictions]


def metrics(predictions, rows, rubric):
    if not rows:
        raise ValueError("Metrics require a nonempty split")
    errors = [p["calibrated_score"] - target_score(row) for p, row in zip(predictions, rows)]
    expected = [r["decision"]["label"] == rubric["positive_decision"] for r in rows]
    agreement = sum(p["meets_score_threshold"] == truth for p, truth in zip(predictions, expected))
    true_positive = sum(p["meets_score_threshold"] and truth for p, truth in zip(predictions, expected))
    false_negative = sum(not p["meets_score_threshold"] and truth for p, truth in zip(predictions, expected))
    false_positive = sum(p["meets_score_threshold"] and not truth for p, truth in zip(predictions, expected))
    return {"n": len(rows), "score_mae": round(sum(abs(e) for e in errors) / len(rows), 6),
            "score_rmse": round(math.sqrt(sum(e * e for e in errors) / len(rows)), 6),
            "decision_agreement": round(agreement / len(rows), 6),
            "positive_decision_rate": sum(expected) / len(rows),
            "false_negatives": false_negative, "false_positives": false_positive,
            "positive_recall": round(true_positive / sum(expected), 6) if any(expected) else None}


def split_history(rows, train_before, validation_before):
    one_version_per_paper(rows)
    first, second = timestamp(train_before), timestamp(validation_before)
    if first >= second:
        raise ValueError("train-before must precede validation-before")
    train = [r for r in rows if timestamp(r["submitted_at"]) < first]
    validation = [r for r in rows if first <= timestamp(r["submitted_at"]) < second]
    test = [r for r in rows if timestamp(r["submitted_at"]) >= second]
    if not train or not validation or not test:
        raise ValueError("Chronological train, validation, and test splits must all be nonempty")
    if any(available_at(r) >= first for r in train):
        raise ValueError("A TRAIN review/decision was unavailable before train cutoff; revise the cohort/cutoff")
    if any(available_at(r) >= second for r in validation):
        raise ValueError("A VALIDATION label was unavailable before test cutoff; revise the cohort/cutoff")
    return train, validation, test


def run_backtest(rows, rubric, model, config, train_before, validation_before,
                 calibration="auto", top_k=3):
    if calibration not in ("none", "affine", "auto"):
        raise ValueError("calibration must be none, affine, or auto")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 0 <= top_k <= 10:
        raise ValueError("top_k must be an integer within [0, 10]")
    train, validation, test = split_history(rows, train_before, validation_before)
    train_predictions = [predict(r, train, rubric, model, top_k) for r in train]
    identity = {"kind": "none", "slope": 1.0, "intercept": 0.0}
    affine = fit_affine(train_predictions, train) if calibration != "none" else None
    if calibration == "affine" and affine is None:
        raise ValueError("Affine fitting needs >=3 train papers, varying predictions, and a positive slope")
    candidates = [identity] if calibration == "none" else ([affine] if calibration == "affine" else [identity] + ([affine] if affine else []))
    threshold = rubric["accept_threshold"]
    threshold_source = "institution_configured"
    if rubric.get("threshold_mode") == "accepted_quantile":
        accepted = [target_score(r) for r in train if r["decision"]["label"] == rubric["positive_decision"]]
        if not accepted:
            raise ValueError("accepted_quantile requires at least one accepted TRAIN paper")
        threshold = accepted_percentile(accepted, rubric.get("accepted_quantile", 0.25))
        threshold_source = "train_accepted_human_scores_lower_quantile"
    validation_raw = [predict(r, train, rubric, model, top_k) for r in validation]
    candidate_results = [{"parameters": c,
                          "validation_metrics": metrics(apply_scores(validation_raw, c, rubric, threshold), validation, rubric)}
                         for c in candidates]
    # Stable tie handling favors the identity candidate in auto mode.
    selected = min(candidate_results, key=lambda c: c["validation_metrics"]["score_mae"])["parameters"]
    state = {"schema_version": 1, "calibration": selected, "threshold": threshold,
             "threshold_source": threshold_source, "train_before": train_before,
             "validation_before": validation_before, "top_k": top_k,
             "rubric_sha256": fingerprint(rubric), "model_config_sha256": fingerprint(config),
             "training_sha256": fingerprint(train), "training_ids": [r["manuscript_id"] for r in train],
             "validation_ids": [r["manuscript_id"] for r in validation],
             "test_ids": [r["manuscript_id"] for r in test],
             "synthetic_inputs": any(r["synthetic"] for r in rows),
             "synthetic_input_count": sum(r["synthetic"] for r in rows), "synthetic_model": model.synthetic}
    if "prompt_template" in config:
        from .prompts import snapshot
        state["prompt_snapshot"] = snapshot(config["prompt_template"])
    frozen_state_hash = fingerprint(state)
    # Test labels never enter retrieval, fit, candidate selection, or adapter input.
    test_raw = [predict(r, train, rubric, model, top_k) for r in test]
    train_final = apply_scores(train_predictions, selected, rubric, threshold)
    validation_final = apply_scores(validation_raw, selected, rubric, threshold)
    test_final = apply_scores(test_raw, selected, rubric, threshold)
    report = {"notice": "SYNTHETIC DEMO METRICS: no empirical quality claim." if state["synthetic_inputs"] or model.synthetic
              else "Retrospective held-out metrics; no academic quality or future performance guarantee.",
              "split_ids": {"train": state["training_ids"], "validation": state["validation_ids"],
                            "test": [r["manuscript_id"] for r in test]},
              "protocol": {"fit": "Affine parameters use TRAIN human scores only; no LLM weights updated.",
                           "selection": "Validation MAE selects calibration; validation is not untouched test.",
                           "test": "Parameters frozen before TEST inference; labels read only for metrics.",
                           "retrieval": "Frozen TRAIN corpus only, same manuscript excluded, labels strictly earlier than query.",
                           "threshold": threshold_source, "rubric": "Institution-supplied and fixed before all submissions."},
              "candidate_results": candidate_results, "selected_calibration": selected,
              "threshold": threshold, "frozen_state_sha256_before_test": frozen_state_hash,
              "frozen_state_sha256_after_test": fingerprint(state),
              "metrics": {"train_fit_diagnostic": metrics(train_final, train, rubric),
                          "validation_selection": metrics(validation_final, validation, rubric),
                          "untouched_test": metrics(test_final, test, rubric)},
              "dataset_sha256": fingerprint(rows), "rubric_sha256": fingerprint(rubric),
              "model_config_sha256": fingerprint(config),
              "privacy_mode": config["mode"], "synthetic_model": model.synthetic,
              "limitations": ["Small retrospective cohorts do not establish validity or fairness.",
                              "Historical decisions encode institutional biases; no causal or novelty claims.",
                              "Base model pretraining contamination cannot be ruled out for real LLMs.",
                              "Repeated inspection/tuning on TEST destroys its untouched status."]}
    return state, report, {"train": train_final, "validation": validation_final, "test": test_final}


def review_with_state(paper, history, rubric, model, config, state):
    one_version_per_paper(history)
    if not isinstance(state, dict):
        raise ValueError("Saved state must be a JSON object")
    if state.get("schema_version") != 1:
        raise ValueError("Unsupported state schema")
    if timestamp(state["train_before"]) >= timestamp(state["validation_before"]):
        raise ValueError("Saved state cutoffs are out of order")
    calibration = state["calibration"]
    if not isinstance(calibration, dict):
        raise ValueError("Saved calibration must be an object")
    if calibration.get("kind") not in ("none", "affine"):
        raise ValueError("Saved calibration kind is unsupported")
    slope = number(calibration["slope"], "saved slope")
    intercept = number(calibration["intercept"], "saved intercept")
    if slope <= 0 or (calibration["kind"] == "none" and (slope != 1 or intercept != 0)):
        raise ValueError("Saved calibration parameters are invalid")
    low, high = rubric["score_range"]
    if not low <= number(state["threshold"], "saved threshold") <= high:
        raise ValueError("Saved threshold is outside score_range")
    top_k = state["top_k"]
    if not isinstance(top_k, int) or isinstance(top_k, bool) or not 0 <= top_k <= 10:
        raise ValueError("Saved top_k must be an integer within [0, 10]")
    cohorts = [state[key] for key in ("training_ids", "validation_ids", "test_ids")]
    if any(not isinstance(ids, list) or not ids or any(not isinstance(i, str) for i in ids) for ids in cohorts):
        raise ValueError("Saved cohort IDs must be nonempty string lists")
    flat_ids = sum(cohorts, [])
    if len(flat_ids) != len(set(flat_ids)):
        raise ValueError("Saved cohort IDs overlap or contain duplicates")
    if timestamp(paper["submitted_at"]) < timestamp(state["validation_before"]):
        raise ValueError("Review target must postdate frozen validation-selection cutoff")
    # A fresh prediction cannot quietly become another evaluation of fitted/selected papers.
    if paper["manuscript_id"] in flat_ids:
        raise ValueError("Review target was already used for fitting, validation, or test evaluation")
    train = [r for r in history if timestamp(r["submitted_at"]) < timestamp(state["train_before"])]
    if [r["manuscript_id"] for r in train] != state["training_ids"]:
        raise ValueError("Saved training IDs do not match the training corpus")
    if any(available_at(r) >= timestamp(state["train_before"]) for r in train):
        raise ValueError("Saved training cohort contains future labels")
    for label, actual, expected in (
        ("rubric", fingerprint(rubric), state["rubric_sha256"]),
        ("model config", fingerprint(config), state["model_config_sha256"]),
        ("training corpus", fingerprint(train), state["training_sha256"]),
    ):
        if actual != expected:
            raise ValueError(f"{label} changed; recalibrate with a new untouched test cohort")
    prediction = predict(paper, train, rubric, model, state["top_k"])
    return apply_scores([prediction], state["calibration"], rubric, state["threshold"])[0]


def save_run(output, state, report, predictions):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "model_state.json", state)
    write_json(output / "report.json", report)
    write_json(output / "predictions.json", predictions)
    if "prompt_snapshot" in state:
        write_json(output / "prompt_snapshot.json", state["prompt_snapshot"])
    lines = ["# Review-preference backtest", "", report["notice"], "",
             f"Privacy mode: {report['privacy_mode']}. Human editors retain authority.", "",
             "| Split | n | Score MAE | Score RMSE | Decision agreement |", "|---|---:|---:|---:|---:|"]
    for name, result in report["metrics"].items():
        lines.append(f"| {name} | {result['n']} | {result['score_mae']:.4f} | {result['score_rmse']:.4f} | {result['decision_agreement']:.4f} |")
    lines += ["", f"Selected external calibration: {state['calibration']['kind']}.",
              f"Threshold: {state['threshold']:.4f} ({state['threshold_source']}).", "",
              "Train is used for fitting; validation for selecting calibration. Only the final test cohort is untouched.",
              "Historical examples come only from the frozen training cohort. LLM weights are never trained.", "",
              "See report.json for split IDs, candidate metrics, provenance hashes and limitations."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
