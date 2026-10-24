"""Independently authored Algorithm-1 orchestration; no official code/prompts copied.

Provider calls are injected. The reference fit uses a labelled TRAIN golden set;
the public-paper pilot additionally uses separate validation for final selection.
"""
from copy import deepcopy
import random
from statistics import mean
from typing import Protocol

from .data import fingerprint, nonempty, number, read_json, write_json


class CriteriaProvider(Protocol):
    def draft(self, examples: list) -> dict: ...
    def revise(self, criteria: dict, errors: list) -> dict: ...
    def score(self, criteria: dict, manuscripts: list) -> list: ...


def history_rows(history):
    if not isinstance(history, list) or not history:
        raise ValueError("Nonempty TRAIN history required")
    rows, seen = [], set()
    for row in history:
        paper_id = nonempty(row.get("paper_id"), "paper_id")
        if paper_id in seen:
            raise ValueError("Duplicate training paper")
        seen.add(paper_id)
        scores = row.get("individual_scores")
        if not isinstance(scores, list) or not scores:
            raise ValueError("Individual absolute scores required")
        values = [number(s, "human score") for s in scores]
        if any(not 1 <= s <= 10 for s in values):
            raise ValueError("Human scores outside original rating scale")
        rows.append({"paper_id": paper_id, "title": nonempty(row.get("title"), "title"),
                     "text": nonempty(row.get("text"), "text"), "reviews": deepcopy(row.get("reviews", [])),
                     "individual_scores": values, "human_mean": mean(values)})
    return rows


def criteria_artifact(response, training_ids, candidate_id, provenance=None):
    if not isinstance(response, dict) or not isinstance(response.get("rules"), list) or not 1 <= len(response["rules"]) <= 8:
        raise ValueError("Criteria output needs 1..8 rules")
    rules, seen = [], set()
    for rule in response["rules"]:
        principle = nonempty(rule.get("principle"), "principle").strip()
        sources = rule.get("source_paper_ids")
        if len(principle) > 500 or principle.casefold() in seen:
            raise ValueError("Criteria must be concise and distinct")
        seen.add(principle.casefold())
        if not isinstance(sources, list) or not sources or not set(sources) <= set(training_ids):
            raise ValueError("Rule provenance must cite supplied TRAIN papers only")
        rules.append({"principle": principle, "source_paper_ids": sorted(set(sources))})
    artifact = {"schema_version": 1, "algorithm": "autocalibrate", "implementation": "independent task adaptation",
                "candidate_id": candidate_id, "version": "1", "rules": rules,
                "training_ids": sorted(training_ids), "provenance": deepcopy(provenance or {})}
    artifact["artifact_sha256"] = fingerprint(artifact)
    return artifact


def validate_artifact(artifact):
    if (artifact.get("schema_version") != 1 or artifact.get("algorithm") != "autocalibrate"
            or artifact["artifact_sha256"] != fingerprint({k: v for k, v in artifact.items() if k != "artifact_sha256"})):
        raise ValueError("Criteria artifact changed or unsupported")
    checked = criteria_artifact({"rules": artifact["rules"]}, artifact["training_ids"], artifact["candidate_id"], artifact["provenance"])
    if checked != artifact:
        raise ValueError("Invalid criteria artifact")
    return deepcopy(artifact)


def condition(artifact):
    checked = validate_artifact(artifact)
    return "\n".join(str(i) + ". " + rule["principle"] for i, rule in enumerate(checked["rules"], 1))


def negative_mae(predictions, history):
    if len(predictions) != len(history) or not history:
        raise ValueError("Predictions must align with TRAIN golden papers")
    scores = [number(v, "predicted score") for v in predictions]
    if any(not 1 <= v <= 10 for v in scores):
        raise ValueError("Prediction outside rating scale")
    return -mean(abs(v - row["human_mean"]) for v, row in zip(scores, history))


def disagreements(predictions, history, limit=6):
    negative_mae(predictions, history)
    return sorted([dict(row, predicted_score=score, absolute_error=abs(score - row["human_mean"]))
                   for row, score in zip(history, predictions) if abs(score - row["human_mean"]) > 1e-12],
                  key=lambda r: (-r["absolute_error"], r["paper_id"]))[:limit]


class AutoCalibrateAlgorithm:
    """Algorithm 1 with replaceable drafting/scoring/refinement and metric.

    metric must return a larger-is-better value (a correlation, or -MAE for
    the manuscript task). No held-out cohort is accepted by fit/refine.
    """
    def __init__(self, provider, exemplar_sizes=(4, 6), trials=1, top_k=1, seed=20261023, metric=negative_mae):
        self.provider, self.sizes, self.trials, self.top_k = provider, exemplar_sizes, trials, top_k
        self.seed, self.metric = seed, metric

    def fit(self, history):
        rows = history_rows(history)
        rng, drafts = random.Random(self.seed), []
        for size in self.sizes:
            if not 1 <= size <= len(rows):
                raise ValueError("Few-shot size exceeds TRAIN golden set")
            for _ in range(self.trials):
                examples = rng.sample(rows, size)
                response = self.provider.draft(examples)
                artifact = criteria_artifact(response, [r["paper_id"] for r in rows], "draft_" + str(len(drafts) + 1))
                scores = self.provider.score(artifact, [{k: r[k] for k in ("paper_id", "title", "text")} for r in rows])
                drafts.append({"artifact": artifact, "scores": scores, "metric": self.metric(scores, rows)})
        retained = sorted(drafts, key=lambda d: -d["metric"])[:self.top_k]
        pool = list(retained)
        for old in retained:
            errors = disagreements(old["scores"], rows)
            if not errors:
                continue
            for index in range(self.trials):
                response = self.provider.revise(old["artifact"], errors)
                artifact = criteria_artifact(response, [r["paper_id"] for r in rows], old["artifact"]["candidate_id"] + "_refined_" + str(index + 1),
                                             {"feedback_source": "candidate_own_training_predictions"})
                scores = self.provider.score(artifact, [{k: r[k] for k in ("paper_id", "title", "text")} for r in rows])
                pool.append({"artifact": artifact, "scores": scores, "metric": self.metric(scores, rows)})
        return {"selected": max(pool, key=lambda d: d["metric"])["artifact"], "drafts": drafts, "pool": pool,
                "method": "Algorithm-1 flow; task metric supplied by caller; provider sampling settings must be documented"}

    def refine(self, artifact, training_feedback):
        checked = validate_artifact(artifact)
        rows = history_rows(training_feedback)
        if not {r["paper_id"] for r in rows} <= set(checked["training_ids"]):
            raise ValueError("Refinement feedback must belong to TRAIN")
        return criteria_artifact(self.provider.revise(checked, training_feedback), checked["training_ids"], checked["candidate_id"] + "_refined",
                                 {"feedback_source": "candidate_own_training_predictions"})

    @staticmethod
    def export(artifact, path):
        write_json(path, validate_artifact(artifact))

    @staticmethod
    def load(path):
        return validate_artifact(read_json(path))

    @staticmethod
    def condition(artifact, paper=None):
        return condition(artifact)
