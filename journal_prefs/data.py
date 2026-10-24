"""Strict, dependency-free input validation and explicit inference projections."""

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError("Timestamps must be ISO 8601 strings with a timezone")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Invalid timestamp: {value}") from exc
    if result.tzinfo is None:
        raise ValueError("Timestamps must include Z or a timezone offset")
    return result.astimezone(timezone.utc)


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def nonempty(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_rubric(rubric):
    if not isinstance(rubric, dict):
        raise ValueError("Rubric must be a JSON object")
    if rubric.get("schema_version") != 1:
        raise ValueError("Rubric schema_version must be 1")
    for field in ("institution_id", "venue_id", "name", "source", "positive_decision", "negative_decision"):
        nonempty(rubric.get(field), field)
    if rubric["positive_decision"] == rubric["negative_decision"]:
        raise ValueError("Positive and negative decision labels must differ")
    timestamp(rubric["available_at"])
    low, high = rubric["score_range"]
    low, high = number(low, "score minimum"), number(high, "score maximum")
    if low >= high:
        raise ValueError("score_range must be increasing")
    threshold = number(rubric["accept_threshold"], "accept_threshold")
    if not low <= threshold <= high:
        raise ValueError("accept_threshold is outside score_range")
    if rubric.get("threshold_mode", "configured") not in ("configured", "accepted_quantile"):
        raise ValueError("threshold_mode must be configured or accepted_quantile")
    quantile = number(rubric.get("accepted_quantile", 0.25), "accepted_quantile")
    if not 0 <= quantile <= 1:
        raise ValueError("accepted_quantile must be within [0, 1]")
    criteria = rubric.get("criteria")
    if not isinstance(criteria, list) or not criteria:
        raise ValueError("At least one institution-configured criterion is required")
    seen = set()
    for criterion in criteria:
        if not isinstance(criterion, dict):
            raise ValueError("Each criterion must be an object")
        for field in ("id", "description"):
            nonempty(criterion.get(field), f"criterion {field}")
        if criterion["id"] in seen:
            raise ValueError("Duplicate criterion id")
        seen.add(criterion["id"])
        if number(criterion["weight"], "criterion weight") <= 0:
            raise ValueError("Criterion weights must be positive")
        keywords = criterion.get("keywords", [])
        if not isinstance(keywords, list) or any(not isinstance(k, str) or not k.strip() for k in keywords):
            raise ValueError("keywords must be a list of nonempty strings")
    return rubric


def validate_paper(row, rubric):
    if not isinstance(row, dict):
        raise ValueError("Manuscript must be a JSON object")
    if row.get("schema_version") != 1:
        raise ValueError("Manuscript schema_version must be 1")
    for field in ("manuscript_id", "version_id", "title", "text", "institution_id", "venue_id"):
        nonempty(row.get(field), field)
    if row["institution_id"] != rubric["institution_id"] or row["venue_id"] != rubric["venue_id"]:
        raise ValueError("Use one institution and venue per run, matching the rubric")
    if row.get("rights_confirmed") is not True:
        raise ValueError("Institution must confirm rights_confirmed=true for every record")
    if not isinstance(row.get("synthetic"), bool):
        raise ValueError("Every record must explicitly label synthetic as true or false")
    if timestamp(row["version_time"]) > timestamp(row["submitted_at"]):
        raise ValueError("version_time cannot be after the simulated submission")
    if timestamp(row["text_available_at"]) >= timestamp(row["submitted_at"]):
        raise ValueError("Text must have been available strictly before simulated submission time")
    if timestamp(rubric["available_at"]) >= timestamp(row["submitted_at"]):
        raise ValueError("Rubric must be fixed and available strictly before all simulated submissions")
    return row


def validate_history(row, rubric):
    validate_paper(row, rubric)
    low, high = rubric["score_range"]
    reviews = row.get("reviews")
    if not isinstance(reviews, list) or not reviews:
        raise ValueError("Historical records require at least one human review")
    for review in reviews:
        if not isinstance(review, dict):
            raise ValueError("Each human review must be an object")
        nonempty(review.get("text"), "review text")
        if not low <= number(review.get("score"), "review score") <= high:
            raise ValueError("Review score is outside rubric score_range")
        if timestamp(review["available_at"]) < timestamp(row["submitted_at"]):
            raise ValueError("A review cannot predate its submission")
    decision = row.get("decision", {})
    if not isinstance(decision, dict):
        raise ValueError("Decision must be an object")
    if decision.get("label") not in (rubric["positive_decision"], rubric["negative_decision"]):
        raise ValueError("Decision label must match the institution's two configured labels")
    if timestamp(decision["available_at"]) < timestamp(row["submitted_at"]):
        raise ValueError("Decision cannot predate its submission")
    return row


def load_history(path, rubric):
    rows = []
    with Path(path).open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(validate_history(json.loads(line), rubric))
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f"{Path(path).name}, line {line_number}: {exc}") from exc
    if not rows:
        raise ValueError("History is empty")
    keys = [(r["manuscript_id"], r["version_id"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate manuscript_id/version_id")
    return sorted(rows, key=lambda r: (timestamp(r["submitted_at"]), r["manuscript_id"]))


def one_version_per_paper(rows):
    ids = [r["manuscript_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Choose ONE pre-review version per manuscript for this run; versions must not cross splits")


def available_at(row):
    return max([timestamp(row["text_available_at"]), timestamp(row["decision"]["available_at"])]
               + [timestamp(r["available_at"]) for r in row["reviews"]])


def target_score(row):
    return sum(r["score"] for r in row["reviews"]) / len(row["reviews"])


def paper_input(row):
    # Whitelist projection: no human reviews, scores, decisions, or arbitrary fields.
    return {key: row[key] for key in ("manuscript_id", "version_id", "title", "text", "submitted_at", "version_time")}
