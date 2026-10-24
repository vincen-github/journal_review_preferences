"""ICAI input/export bridge and reviewed filtering component, not a full reproduction.

filter_principles is adapted from rdnfn/icai algorithm/filter.py (Apache-2.0),
revision cb88e6588e9a01e8c1f216ea3cfe1b7461ddee7d. Changes: compact expression,
renamed entrypoint, defaults and count validation. See THIRD_PARTY_NOTICES.md
and licenses/ICAI-Apache-2.0.txt. Other bridge code is authored for this project.
"""

import csv
from copy import deepcopy

from .data import fingerprint, nonempty, read_json, write_json

UPSTREAM_REVISION = "cb88e6588e9a01e8c1f216ea3cfe1b7461ddee7d"


def validate_pairs(rows):
    """Application-specific same-paper pairs, with explicit pairwise judgments.

    Upstream requires text_a/text_b/preferred_text and makes input optional.
    We additionally retain paper grouping, common input and annotation provenance.
    Accepted/rejected manuscripts are never converted into response preferences.
    """
    if not isinstance(rows, list) or not rows:
        raise ValueError("Supply explicitly annotated pairwise comparisons")
    result, seen = [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Pair must be an object")
        item = {key: nonempty(row.get(key), key) for key in
                ("comparison_id", "paper_id", "input", "text_a", "text_b", "preference_source")}
        if row.get("preferred_text") not in ("text_a", "text_b"):
            raise ValueError("ICAI needs an explicit text_a/text_b preference; acceptance labels are insufficient")
        item["preferred_text"] = row["preferred_text"]
        if item["text_a"] == item["text_b"] or item["comparison_id"] in seen:
            raise ValueError("Comparisons need distinct responses and unique IDs")
        for key in ("response_a_paper_id", "response_b_paper_id"):
            if row.get(key, item["paper_id"]) != item["paper_id"]:
                raise ValueError("Do not pair assessments of different manuscripts")
        seen.add(item["comparison_id"])
        result.append(item)
    return result


def export_upstream_csv(rows, path):
    """Export the author's documented input schema; does not run upstream code."""
    pairs = validate_pairs(rows)
    with open(path, "x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["input", "text_a", "text_b", "preferred_text"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in pairs)
    return {"upstream_revision": UPSTREAM_REVISION, "comparisons": len(pairs),
            "paper_ids": sorted({row["paper_id"] for row in pairs}), "input_sha256": fingerprint(pairs)}


def filter_principles(votes, require_majority_true=True, require_majority_relevant=False,
                      require_majority_valid=True, require_minimum_relevance=0.1,
                      order_by="for_minus_against", max_principles=None):
    """Preserve the inspected author's vote-filter predicates and stable ranking."""
    keys = ("for", "against", "abstain", "invalid", "both", "neither")
    keep = []
    for principle, counts in votes.items():
        if any(isinstance(counts.get(k), bool) or not isinstance(counts.get(k), int) or counts[k] < 0 for k in keys):
            raise ValueError("Vote counts must be nonnegative integers")
        positive, negative, abstain, invalid = (counts[k] for k in keys[:4])
        if require_majority_true and positive <= negative:
            continue
        if require_majority_relevant and positive <= abstain:
            continue
        if require_minimum_relevance and positive < sum(counts[k] for k in keys) * require_minimum_relevance:
            continue
        if require_majority_valid and invalid >= positive + negative + abstain:
            continue
        keep.append(principle)
    if order_by not in ("for", "for_minus_against"):
        raise ValueError("Invalid principle ordering")
    keep.sort(key=lambda k: votes[k]["for"] - (votes[k]["against"] if order_by == "for_minus_against" else 0), reverse=True)
    return keep[:max_principles] if max_principles else keep


def _seal(artifact):
    artifact = deepcopy(artifact)
    artifact.pop("artifact_sha256", None)
    artifact["artifact_sha256"] = fingerprint(artifact)
    return artifact


def _check(artifact):
    result = deepcopy(artifact)
    checksum = result.pop("artifact_sha256", None)
    if checksum != fingerprint(result) or result.get("schema_version") != 1 or result.get("algorithm") != "icai_upstream_bridge":
        raise ValueError("Invalid ICAI bridge artifact/checksum")
    if not isinstance(result.get("frozen"), bool) or not isinstance(result.get("principles"), list):
        raise ValueError("Invalid frozen state/principles")
    for principle in result["principles"]:
        nonempty(principle, "principle")
    return _seal(result)


class ICAIUpstreamAdapter:
    """Delegate fitting to an explicitly supplied, approved original runner.

    No package installation, credential lookup, fallback algorithm or backend
    execution occurs unless a caller provides runner(pairs)->author_result.
    This adapter itself does not implement proposal, embeddings or LLM voting.
    """

    def __init__(self, runner=None, backend_description="unconfigured"):
        self.runner = runner
        self.backend_description = backend_description

    def fit(self, history):
        pairs = validate_pairs(history)
        if self.runner is None:
            raise ValueError("Original ICAI runner not configured; install/run approval and real pairwise data are separate prerequisites")
        result = self.runner(deepcopy(pairs))
        principles = result.get("final_principles")
        if not isinstance(principles, list) or any(not isinstance(p, str) or not p.strip() for p in principles):
            raise ValueError("Original runner must return final_principles as strings")
        return _seal({"schema_version": 1, "algorithm": "icai_upstream_bridge", "upstream_revision": UPSTREAM_REVISION,
                      "backend_description": nonempty(self.backend_description, "backend description"),
                      "train_ids": sorted({p["paper_id"] for p in pairs}), "train_sha256": fingerprint(pairs),
                      "principles": principles, "frozen": False})

    def refine(self, artifact, validation, assessments=None):
        raise ValueError("No self-designed ICAI refinement: choose original-method configurations on validation, then freeze")

    def freeze(self, artifact):
        result = _check(artifact)
        result["frozen"] = True
        return _seal(result)

    def condition(self, artifact, paper):
        result = _check(artifact)
        if not result["frozen"] or paper["paper_id"] in result["train_ids"]:
            raise ValueError("Use frozen principles on a different paper")
        return [{"rule_id": f"ICAI{i+1:03d}", "principle": p} for i, p in enumerate(result["principles"])]

    def save(self, artifact, path):
        write_json(path, _check(artifact))

    def load(self, path):
        return _check(read_json(path))
