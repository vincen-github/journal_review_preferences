"""Provider-independent evidence rules, with explicit validation selection.

EXPERIMENTAL SCAFFOLD, PAUSED: not the default and not an ICAI reproduction.
ICAI remains an optional reference in icai_support.py. The separate
autocalibrate.py interface implements the current criteria-learning trial.

Generation is injected. This module performs no networking, model loading,
training, or affine score calibration. Artifact hashes detect content changes;
they are checksums, not signatures or proof of scientific validity.
"""

from copy import deepcopy
from typing import Protocol, runtime_checkable

from .data import fingerprint, nonempty, number, read_json, write_json


@runtime_checkable
class PreferenceAlgorithm(Protocol):
    """Replaceable preference induction; scoring and providers live elsewhere."""

    def fit(self, history: list) -> dict: ...

    def refine(self, artifact: dict, validation: list, assessments=None) -> dict: ...

    def freeze(self, artifact: dict) -> dict: ...

    def condition(self, artifact: dict, paper: dict) -> list: ...


def _integer(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise ValueError(f"{name} must be an integer within {low}..{high}")
    return value


def _history(history, name):
    if not isinstance(history, list) or not history:
        raise ValueError(f"{name} must be a nonempty list of labelled papers")
    rows, seen = [], set()
    for record in history:
        if not isinstance(record, dict):
            raise ValueError(f"Every {name} paper must be an object")
        paper_id = nonempty(record.get("paper_id"), "paper_id")
        if paper_id in seen:
            raise ValueError("Each paper_id must be unique within a cohort")
        seen.add(paper_id)
        reviews = record.get("reviews")
        if not isinstance(reviews, list) or not reviews:
            raise ValueError("Every source paper needs at least one human review")
        projected_reviews = []
        for review in reviews:
            if not isinstance(review, dict):
                raise ValueError("Human review must be an object")
            projected_reviews.append({"text": nonempty(review.get("text"), "human review text"),
                                      "score": number(review.get("score"), "human review score")})
        decision = record.get("decision")
        if not isinstance(decision, bool):
            decision = nonempty(decision, "human decision")
        rows.append({"paper_id": paper_id, "title": nonempty(record.get("title"), "paper title"),
                     "text": nonempty(record.get("text"), "manuscript text"),
                     "reviews": projected_reviews, "decision": decision})
    return sorted(rows, key=lambda row: row["paper_id"])


def _rules(result, sources, max_rules):
    if not isinstance(result, dict) or not isinstance(result.get("rules"), list):
        raise ValueError("Rule generator must return an object with a rules list")
    if len(result["rules"]) > max_rules:
        raise ValueError("Rule generator exceeded max_rules")
    by_id = {row["paper_id"]: row for row in sources}
    checked, principles = [], set()
    for index, rule in enumerate(result["rules"], 1):
        if not isinstance(rule, dict):
            raise ValueError("Every rule must be an object")
        principle = nonempty(rule.get("principle"), "rule principle")
        if principle.strip().casefold() in principles:
            raise ValueError("Rule principles must be distinct")
        principles.add(principle.strip().casefold())
        evidence = rule.get("evidence")
        if not isinstance(evidence, list) or not 1 <= len(evidence) <= 64:
            raise ValueError("Each rule must cite 1..64 source evidence items")
        quotes, seen = [], set()
        for item in evidence:
            if not isinstance(item, dict):
                raise ValueError("Rule evidence must be an object")
            paper_id = nonempty(item.get("paper_id"), "evidence paper_id")
            if paper_id not in by_id:
                raise ValueError("Rule evidence cites an unknown source paper")
            review_quote = nonempty(item.get("review_quote"), "review_quote")
            manuscript_quote = nonempty(item.get("manuscript_quote"), "manuscript_quote")
            source = by_id[paper_id]
            if not any(review_quote in review["text"] for review in source["reviews"]):
                raise ValueError("Rule review_quote is absent from the cited human reviews")
            if manuscript_quote not in source["text"]:
                raise ValueError("Rule manuscript_quote is absent from the cited manuscript")
            key = (paper_id, review_quote, manuscript_quote)
            if key in seen:
                raise ValueError("Duplicate evidence within a rule")
            seen.add(key)
            quotes.append({"paper_id": paper_id, "review_quote": review_quote,
                           "manuscript_quote": manuscript_quote})
        checked.append({"rule_id": f"R{index:03d}", "principle": principle, "evidence": quotes})
    return checked


def _checksum(artifact):
    return fingerprint({key: value for key, value in artifact.items() if key != "artifact_sha256"})


def _seal(artifact):
    artifact["artifact_sha256"] = _checksum(artifact)
    return artifact


def validate_artifact(artifact):
    """Return a checked copy; verify schema, sources, evidence, and checksum."""
    fields = {"schema_version", "algorithm", "algorithm_version", "parameters", "train_ids",
              "train_sha256", "selection_ids", "selection_sha256", "source_records", "rules",
              "frozen", "refinement_count", "artifact_sha256"}
    if not isinstance(artifact, dict) or set(artifact) != fields:
        raise ValueError("Invalid evidence-rules artifact schema")
    if type(artifact["schema_version"]) is not int or artifact["schema_version"] != 1:
        raise ValueError("Unsupported preference artifact schema_version")
    if artifact["algorithm"] != "evidence_rules" or artifact["algorithm_version"] != "1":
        raise ValueError("Unsupported preference algorithm/version")
    if not isinstance(artifact["frozen"], bool):
        raise ValueError("Artifact frozen must be boolean")
    parameters = artifact["parameters"]
    if not isinstance(parameters, dict) or set(parameters) != {"max_rules", "max_refinements"}:
        raise ValueError("Invalid preference parameters")
    max_rules = _integer(parameters["max_rules"], "max_rules", 1, 64)
    maximum = _integer(parameters["max_refinements"], "max_refinements", 0, 1)
    count = _integer(artifact["refinement_count"], "refinement_count", 0, maximum)
    sources = artifact["source_records"]
    if not isinstance(sources, dict) or set(sources) != {"train", "selection"}:
        raise ValueError("Invalid preference source records")
    train = _history(sources["train"], "training")
    if count:
        selection = _history(sources["selection"], "validation selection")
    elif sources["selection"] == []:
        selection = []
    else:
        raise ValueError("Unrefined artifact must not contain validation sources")
    if train != sources["train"] or selection != sources["selection"]:
        raise ValueError("Artifact sources must use only normalized schema fields")
    for cohort, rows in (("train", train), ("selection", selection)):
        ids = [row["paper_id"] for row in rows]
        if artifact[cohort + "_ids"] != ids:
            raise ValueError("Artifact source IDs do not match source records")
        if artifact[cohort + "_sha256"] != fingerprint(sources[cohort]):
            raise ValueError("Artifact source checksum mismatch")
    if set(artifact["train_ids"]) & set(artifact["selection_ids"]):
        raise ValueError("Training and validation selection paper IDs must be disjoint")
    checked_rules = _rules({"rules": artifact["rules"]}, train + selection, max_rules)
    if checked_rules != artifact["rules"]:
        raise ValueError("Artifact rules must use canonical unique IDs and schema fields")
    if artifact["artifact_sha256"] != _checksum(artifact):
        raise ValueError("Preference artifact checksum mismatch")
    return deepcopy(artifact)


def export_artifact(artifact, path):
    """Save a validated artifact locally, including its sensitive evidence sources."""
    write_json(path, validate_artifact(artifact))


def load_artifact(path):
    """Load and check an artifact. A checksum is not a tamperproof signature."""
    return validate_artifact(read_json(path))


def _assessments(assessments, validation):
    if assessments is None:
        return None
    if not isinstance(assessments, list):
        raise ValueError("Validation assessments must be a list")
    by_id = {row["paper_id"]: row for row in validation}
    projected, seen = [], set()
    for item in assessments:
        if not isinstance(item, dict):
            raise ValueError("Validation assessment must be an object")
        paper_id = nonempty(item.get("paper_id"), "assessment paper_id")
        if paper_id not in by_id or paper_id in seen:
            raise ValueError("Assessments require unique validation-only paper IDs")
        seen.add(paper_id)
        row = {"paper_id": paper_id}
        if "score" in item:
            row["score"] = number(item["score"], "assessment score")
        if "review" in item:
            row["review"] = nonempty(item["review"], "assessment review")
        if "decision" in item:
            row["decision"] = item["decision"] if isinstance(item["decision"], bool) else nonempty(item["decision"], "assessment decision")
        projected.append(row)
    return sorted(projected, key=lambda row: row["paper_id"])


class EvidenceRules:
    """Induce evidence-backed rules through a caller-supplied JSON generator.

    Artifacts retain normalized training/validation sources for revalidation.
    Treat exported artifacts as sensitive local data. Conditioning exposes only
    rule IDs and principles, never labels, quotes, or historical source text.
    """

    def __init__(self, generator, max_rules=8, max_refinements=1):
        if not callable(generator):
            raise ValueError("generator must be callable(purpose, payload)")
        self.generator = generator
        self.max_rules = _integer(max_rules, "max_rules", 1, 64)
        self.max_refinements = _integer(max_refinements, "max_refinements", 0, 1)

    def fit(self, history):
        train = _history(history, "training")
        candidates = []
        for paper in train:
            result = self.generator("per_paper", {"paper": deepcopy(paper), "max_rules": self.max_rules})
            candidates.append({"paper_id": paper["paper_id"],
                               "rules": _rules(result, [paper], self.max_rules)})
        result = self.generator("synthesis", {"papers": deepcopy(train), "candidates": deepcopy(candidates),
                                               "max_rules": self.max_rules})
        rules = _rules(result, train, self.max_rules)
        return _seal({"schema_version": 1, "algorithm": "evidence_rules", "algorithm_version": "1",
                      "parameters": {"max_rules": self.max_rules, "max_refinements": self.max_refinements},
                      "train_ids": [row["paper_id"] for row in train], "train_sha256": fingerprint(train),
                      "selection_ids": [], "selection_sha256": fingerprint([]),
                      "source_records": {"train": train, "selection": []}, "rules": rules,
                      "frozen": False, "refinement_count": 0})

    def refine(self, artifact, validation, assessments=None):
        result = validate_artifact(artifact)
        if result["frozen"]:
            raise ValueError("Frozen preference artifacts cannot be refined")
        if result["refinement_count"] >= result["parameters"]["max_refinements"]:
            raise ValueError("Preference refinement budget exhausted")
        selection = _history(validation, "validation selection")
        if set(result["train_ids"]) & {row["paper_id"] for row in selection}:
            raise ValueError("Training and validation selection paper IDs must be disjoint")
        train = result["source_records"]["train"]
        payload = {"training": deepcopy(train), "validation": deepcopy(selection),
                   "rules": deepcopy(result["rules"]), "assessments": _assessments(assessments, selection),
                   "max_rules": result["parameters"]["max_rules"]}
        generated = self.generator("refinement", payload)
        result.update({"rules": _rules(generated, train + selection, result["parameters"]["max_rules"]),
                       "selection_ids": [row["paper_id"] for row in selection],
                       "selection_sha256": fingerprint(selection), "refinement_count": result["refinement_count"] + 1})
        result["source_records"]["selection"] = selection
        return _seal(result)

    def freeze(self, artifact):
        result = validate_artifact(artifact)
        result["frozen"] = True
        return _seal(result)

    def condition(self, artifact, paper):
        artifact = validate_artifact(artifact)
        if not artifact["frozen"]:
            raise ValueError("Freeze preference artifacts before conditioning a target")
        if not isinstance(paper, dict):
            raise ValueError("Target paper must be an object with paper_id")
        paper_id = nonempty(paper.get("paper_id"), "target paper_id")
        if paper_id in artifact["train_ids"] + artifact["selection_ids"]:
            raise ValueError("Target paper was used for training or validation selection")
        return [{"rule_id": rule["rule_id"], "principle": rule["principle"]}
                for rule in artifact["rules"]]
