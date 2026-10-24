"""Offline absolute-score calibration, selected on validation and then frozen.

Targets are means of human individual recommendations. No text, model calls,
pair construction, test-set fitting, or acceptance-probability interpretation.
Checksums detect changes; they are not tamperproof signatures.
"""

import math
import re
from copy import deepcopy

from .data import fingerprint, nonempty, number as _base_number, read_json, write_json


_IDENTITY = {"kind": "identity", "slope": 1.0, "intercept": 0.0}
_SELECTION = "Lowest validation MAE after clipping to [1, 10]; numerical ties favor identity"
_HASH = re.compile(r"[a-f0-9]{64}")


def _number(value, name):
    try:
        return _base_number(value, name)
    except OverflowError:
        raise ValueError(f"{name} must be a finite number") from None


def _score(value, name):
    result = _number(value, name)
    if not 1 <= result <= 10:
        raise ValueError(f"{name} must be within [1, 10]")
    return result


def _cohort(predictions, labels):
    if not isinstance(predictions, list) or not predictions or not isinstance(labels, list) or not labels:
        raise ValueError("Predictions and labels must be nonempty lists")
    prediction_map, label_map = {}, {}
    for item in predictions:
        if not isinstance(item, dict):
            raise ValueError("Prediction must be an object")
        paper_id = nonempty(item.get("paper_id"), "prediction paper_id")
        if paper_id in prediction_map:
            raise ValueError("Prediction paper IDs must be unique")
        prediction_map[paper_id] = {"paper_id": paper_id, "score": _score(item.get("score"), "prediction score")}
    for item in labels:
        if not isinstance(item, dict):
            raise ValueError("Label must be an object")
        paper_id = nonempty(item.get("paper_id"), "label paper_id")
        if paper_id in label_map:
            raise ValueError("Label paper IDs must be unique")
        values = item.get("individual_scores")
        if not isinstance(values, list) or not values:
            raise ValueError("Labels require a nonempty individual_scores list")
        row = {"paper_id": paper_id, "individual_scores": [_score(v, "individual recommendation") for v in values]}
        if "accepted" in item:
            if not isinstance(item["accepted"], bool):
                raise ValueError("Optional accepted label must be boolean")
            row["accepted"] = item["accepted"]
        label_map[paper_id] = row
    if set(prediction_map) != set(label_map):
        raise ValueError("Prediction and label paper IDs must match exactly")
    ids = sorted(prediction_map)
    return [prediction_map[i] for i in ids], [label_map[i] for i in ids]


def _mean(label):
    return sum(label["individual_scores"]) / len(label["individual_scores"])


def _apply(score, calibration):
    return min(10.0, max(1.0, score * calibration["slope"] + calibration["intercept"]))


def _metrics(predicted, targets):
    errors = [prediction - target for prediction, target in zip(predicted, targets)]
    return {"n": len(errors), "mae": sum(abs(e) for e in errors) / len(errors),
            "rmse": math.sqrt(sum(e * e for e in errors) / len(errors)),
            "signed_bias": sum(errors) / len(errors)}


def _fit_affine(predictions, labels):
    # Three papers minimum avoids fitting a two-point interpolation as a pilot.
    if len(predictions) < 3:
        return None
    x, y = [p["score"] for p in predictions], [_mean(label) for label in labels]
    mx, my = sum(x) / len(x), sum(y) / len(y)
    variance = sum((v - mx) ** 2 for v in x)
    if variance < 1e-12:
        return None
    slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / variance
    intercept = my - slope * mx
    if slope <= 0 or not math.isfinite(slope) or not math.isfinite(intercept):
        return None
    return {"kind": "affine", "slope": slope, "intercept": intercept}


def _choose(candidates):
    best = candidates[0]
    for candidate in candidates[1:]:
        previous, current = best["validation"]["mae"], candidate["validation"]["mae"]
        if current < previous and not math.isclose(current, previous, rel_tol=1e-12, abs_tol=1e-12):
            best = candidate
    return deepcopy(best["calibration"])


def _json_metadata(value):
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, (int, float)):
        _number(value, "provenance number")
        return
    if isinstance(value, list):
        for item in value:
            _json_metadata(item)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _json_metadata(item)
        return
    raise ValueError("Provenance must contain only JSON values and string object keys")


def _checksum(state):
    return fingerprint({key: value for key, value in state.items() if key != "state_sha256"})


def fit_state(train_predictions, train_labels, validation_predictions, validation_labels, provenance=None):
    """Fit on TRAIN only; select identity/positive affine by VALIDATION MAE.

    Inputs align by unique paper IDs, independently of list order. There is no
    test argument. Affine OLS needs >=3 train papers, varying scores, and a
    positive slope. All candidates are clipped to the recommendation scale.
    """
    train_predictions, train_labels = _cohort(train_predictions, train_labels)
    validation_predictions, validation_labels = _cohort(validation_predictions, validation_labels)
    train_ids = [p["paper_id"] for p in train_predictions]
    validation_ids = [p["paper_id"] for p in validation_predictions]
    if set(train_ids) & set(validation_ids):
        raise ValueError("Training and validation paper IDs must be disjoint")
    if provenance is not None and not isinstance(provenance, dict):
        raise ValueError("Provenance must be a JSON object or None")
    _json_metadata(provenance)
    candidates = [deepcopy(_IDENTITY)]
    affine = _fit_affine(train_predictions, train_labels)
    if affine is not None:
        candidates.append(affine)
    targets = [_mean(label) for label in validation_labels]
    results = [{"calibration": candidate,
                "validation": _metrics([_apply(p["score"], candidate) for p in validation_predictions], targets)}
               for candidate in candidates]
    state = {"schema_version": 1, "algorithm": "absolute_score_calibration", "algorithm_version": "1",
             "frozen": True, "score_range": [1, 10], "calibration": _choose(results),
             "train_ids": train_ids, "validation_ids": validation_ids,
             "train_sha256": fingerprint({"predictions": train_predictions, "labels": train_labels}),
             "validation_sha256": fingerprint({"predictions": validation_predictions, "labels": validation_labels}),
             "train_predictions_sha256": fingerprint(train_predictions), "train_labels_sha256": fingerprint(train_labels),
             "validation_predictions_sha256": fingerprint(validation_predictions),
             "validation_labels_sha256": fingerprint(validation_labels),
             "candidate_results": results, "selection_rule": _SELECTION, "provenance": deepcopy(provenance)}
    state["state_sha256"] = _checksum(state)
    return state


def _calibration(candidate):
    if not isinstance(candidate, dict) or set(candidate) != {"kind", "slope", "intercept"}:
        raise ValueError("Invalid calibration parameters")
    slope = _number(candidate["slope"], "calibration slope")
    _number(candidate["intercept"], "calibration intercept")
    if candidate["kind"] == "identity":
        if candidate != _IDENTITY:
            raise ValueError("Identity calibration must use slope=1 and intercept=0")
    elif candidate["kind"] != "affine" or slope <= 0:
        raise ValueError("Calibration must be identity or positive affine")


def validate_state(state):
    """Validate types, partition separation, selection, and the content checksum."""
    fields = {"schema_version", "algorithm", "algorithm_version", "frozen", "score_range", "calibration",
              "train_ids", "validation_ids", "train_sha256", "validation_sha256", "train_predictions_sha256",
              "train_labels_sha256", "validation_predictions_sha256", "validation_labels_sha256",
              "candidate_results", "selection_rule", "provenance", "state_sha256"}
    if not isinstance(state, dict) or set(state) != fields:
        raise ValueError("Invalid absolute-score calibration state schema")
    if type(state["schema_version"]) is not int or state["schema_version"] != 1:
        raise ValueError("Unsupported score calibration schema_version")
    if state["algorithm"] != "absolute_score_calibration" or state["algorithm_version"] != "1":
        raise ValueError("Unsupported score calibration algorithm/version")
    scale = state["score_range"]
    if (state["frozen"] is not True or not isinstance(scale, list) or len(scale) != 2
            or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in scale)
            or scale != [1, 10] or state["selection_rule"] != _SELECTION):
        raise ValueError("Score calibration state must be frozen with the supported scale and selection rule")
    for cohort in ("train", "validation"):
        ids = state[cohort + "_ids"]
        if not isinstance(ids, list) or not ids or any(not isinstance(i, str) or not i.strip() for i in ids):
            raise ValueError("State paper IDs must be nonempty string lists")
        if len(set(ids)) != len(ids) or ids != sorted(ids):
            raise ValueError("State paper IDs must be unique and sorted")
    if set(state["train_ids"]) & set(state["validation_ids"]):
        raise ValueError("Training and validation paper IDs must be disjoint")
    for key in fields:
        if key.endswith("sha256") and (not isinstance(state[key], str) or not _HASH.fullmatch(state[key])):
            raise ValueError("State hashes must be SHA-256 hex strings")
    if state["provenance"] is not None and not isinstance(state["provenance"], dict):
        raise ValueError("Provenance must be a JSON object or None")
    _json_metadata(state["provenance"])
    _calibration(state["calibration"])
    results = state["candidate_results"]
    if not isinstance(results, list) or not 1 <= len(results) <= 2:
        raise ValueError("State must contain identity and at most one affine candidate")
    for index, item in enumerate(results):
        if not isinstance(item, dict) or set(item) != {"calibration", "validation"}:
            raise ValueError("Invalid calibration candidate result")
        _calibration(item["calibration"])
        if item["calibration"]["kind"] != ("identity" if index == 0 else "affine"):
            raise ValueError("Candidates must list identity before affine")
        metrics = item["validation"]
        if not isinstance(metrics, dict) or set(metrics) != {"n", "mae", "rmse", "signed_bias"}:
            raise ValueError("Invalid validation metrics")
        if type(metrics["n"]) is not int or metrics["n"] != len(state["validation_ids"]):
            raise ValueError("Validation metric count must match selection paper IDs")
        for name in ("mae", "rmse"):
            if not 0 <= _number(metrics[name], "validation " + name) <= 9:
                raise ValueError("Validation errors must be within the recommendation scale")
        if not -9 <= _number(metrics["signed_bias"], "validation signed_bias") <= 9:
            raise ValueError("Validation signed bias must be within [-9, 9]")
    if state["calibration"] != _choose(results):
        raise ValueError("Selected calibration does not match validation selection")
    if state["state_sha256"] != _checksum(state):
        raise ValueError("Score calibration state checksum mismatch")
    return deepcopy(state)


def evaluate_test(state, test_predictions, test_labels):
    """Evaluate disjoint held-out labels without changing fitted state or scores."""
    checked = validate_state(state)
    predictions, labels = _cohort(test_predictions, test_labels)
    if {p["paper_id"] for p in predictions} & set(checked["train_ids"] + checked["validation_ids"]):
        raise ValueError("Test paper IDs must be disjoint from training and validation selection")
    rows = []
    for prediction, label in zip(predictions, labels):
        scores, mean = label["individual_scores"], _mean(label)
        rows.append({"paper_id": prediction["paper_id"], "raw_score": prediction["score"],
                     "calibrated_score": _apply(prediction["score"], checked["calibration"]),
                     "human_mean": mean, "individual_scores": deepcopy(scores),
                     "score_std_population": math.sqrt(sum((s - mean) ** 2 for s in scores) / len(scores)),
                     "score_min": min(scores), "score_max": max(scores)})
    targets = [row["human_mean"] for row in rows]
    for row in rows:
        row["raw_error"] = row["raw_score"] - row["human_mean"]
        row["calibrated_error"] = row["calibrated_score"] - row["human_mean"]
        row["raw_absolute_error"] = abs(row["raw_error"])
        row["calibrated_absolute_error"] = abs(row["calibrated_error"])
    raw = _metrics([row["raw_score"] for row in rows], targets)
    calibrated = _metrics([row["calibrated_score"] for row in rows], targets)
    return {"state_sha256": checked["state_sha256"], "raw": raw, "calibrated": calibrated,
            "improvement": {"mae": raw["mae"] - calibrated["mae"], "rmse": raw["rmse"] - calibrated["rmse"]},
            "per_paper": rows,
            "notice": "Held-out recommendation-score agreement only; no acceptance probability or content-quality improvement claim."}


def save_state(state, path):
    write_json(path, validate_state(state))


def load_state(path):
    return validate_state(read_json(path))
